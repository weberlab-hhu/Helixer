"""converts Helixer's output predictions into AUGUSTUS compatible hints"""
import argparse
import h5py
import numpy as np


HINTS = ['irpart', 'UTRpart', 'CDSpart', 'intronpart']


def get_contiguous_ranges(h5):
    """gets h5 coordinates for same species, sequence and strand AKA end to end across a chromosome/scaffold"""
    start_ends = h5['data/start_ends'][:]
    marks_unique = np.stack((h5['data/seqids'], h5['data/species'], start_ends[:, 1] > start_ends[:, 0]))
    items, indexes, lengths = np.unique(marks_unique, axis=1, return_index=True, return_counts=True)
    reindex = np.argsort(indexes)
    for i in reindex:
        seqid, species, is_plus_strand = items[:, i]
        yield {"species": species,
               "seqid": seqid,
               "is_plus_strand": is_plus_strand,
               "start_i": indexes[i],
               "end_i": indexes[i] + lengths[i]}


def read_in_chunks(preds, data, start_i, end_i, step=100):
    for i in range(start_i, end_i, step):
        ei = min(i + step, end_i)
        pred_chunk = preds['predictions'][i:ei]
        pred_chunk = pred_chunk.reshape((-1, 4))
        start = data['data/start_ends'][i, 0]
        end = data['data/start_ends'][ei - 1, 1]
        # in case of padding
        if abs(end - start) % data['data/X'].shape[1]:
            # padded areas have no sequence, so will use that to create a mask for simplicity
            mask = data['data/X'][i:ei].reshape((-1, 4))
            mask = np.any(mask, axis=1)
            pred_chunk = pred_chunk[mask]
            assert pred_chunk.shape[0] == abs(start - end)
        yield pred_chunk, start, end


def find_confident_single_class_regions(pred_chunk, pad=5):
    """find start, end (relative to pred_chunk) bits where in-between is confidently one class"""
    # in the event of a seamless class swap [1,0,0,0] -> [0,1,0,0]
    # and of a thin-seemed swap [1,0,0,0] -> [0.5,0.5,0,0] -> [0,1,0,0]
    # (and any easier swap, as well)
    # the 2-bp smoothing below will allow us to detect a class swap by drop in confidence (max)
    shift_n_averaged = (pred_chunk[:-1] + pred_chunk[1:]) / 2
    # find anywhere network was not confident / or class switch
    lowconf_idx = np.where(np.max(shift_n_averaged, axis=1) < 0.75)[0]
    if lowconf_idx.shape[0] == 0:
        yield 0, pred_chunk.shape[0]
        return
    # handle (& yield) start of chunk edge case if pred_chunk starts with a confident region
    if lowconf_idx[0] > pad * 2:
        yield 0, lowconf_idx[0]
    # invert above to find longer single-class confident regions (double our padding)
    dist2next_lowconf = lowconf_idx[1:] - lowconf_idx[:-1]
    conf_sub_idx = np.where(dist2next_lowconf > pad * 2)[0]

    # yield confident regions
    for sub_idx in conf_sub_idx:
        start = lowconf_idx[sub_idx] + 1  # confidence starts _after_ lowconf ends
        end = start + dist2next_lowconf[sub_idx] - 1  # (exclusive) confidence ends where lowconf starts
        yield start, end

    # handle (& yield) end of chunk edge case if pred_chunk ends with a confident region
    if pred_chunk.shape[0] - lowconf_idx[-1] > pad * 2:
        yield lowconf_idx[-1] + 1, pred_chunk.shape[0]


def divvy_by_confidence(one_class_chunk, step_key, pad=5, stability_threshold=0.1):
    """breaks down contiguous 1-class region to pre-hints with semi-consistent confidence"""
    main_class = np.argmax(one_class_chunk[0])
    min_step, max_size = step_key[main_class]
    diffs = np.abs(one_class_chunk[:-1, main_class] - one_class_chunk[1:, main_class])
    cumulative_diffs = np.cumsum(diffs)
    # we start after padding (and end before), not necessarily at 0 nor sequence end
    cdiff_at_last_yield = cumulative_diffs[pad]
    end_of_last_yield = pad
    padded_seq_end = one_class_chunk.shape[0] - pad
    for i in range(pad, padded_seq_end, min_step):
        end = min(i + min_step, padded_seq_end)
        # if the total observed diffs since yield have passed the threshold, yield again
        # thus where confidence is volatile one gets small chunks, and for continuous predictions
        # one gets large (probably max_size) chunks (saves gff size & there's little gain in breaking down)
        if cumulative_diffs[end] - cdiff_at_last_yield > stability_threshold or \
                end - end_of_last_yield > max_size or \
                end == padded_seq_end:
            start = end_of_last_yield
            yield {'category': main_class,
                   'start': start,
                   'end': end,
                   'confidence': np.mean(one_class_chunk[start:end, main_class])}
            # reset trackers
            end_of_last_yield = end
            cdiff_at_last_yield = cumulative_diffs[end]


def start_end_strand(contiguous_bit, pred_chunk_start, one_category_start, pre_hint):
    if contiguous_bit['is_plus_strand']:
        # start as a combined coordinate
        # hint rel. to one_class_chunk  + one_class_chunk rel. to pred_chunk + pred_chunk rel to seq.
        absolute_start = pred_chunk_start + one_category_start + pre_hint['start']
        # similar for end
        absolute_end = pred_chunk_start + one_category_start + pre_hint['end']
        # both inclusive, both counting from 1
        gff_start = absolute_start + 1
        gff_end = absolute_end  # +1 to count from 1, canceled by -1 for inclusive from exclusive
        strand = '+'
    else:
        # the orientation of pred_chunk and one_class_chunk are flipped relative to genome, thus '-'
        absolute_start = pred_chunk_start - one_category_start - pre_hint['start']
        absolute_end = pred_chunk_start - one_category_start - pre_hint['end']
        # both inclusive, both counting from 1, start and end flipped
        gff_start = absolute_start + 1
        gff_end = absolute_end
        gff_start, gff_end = gff_end, gff_start
        strand = '-'
    return gff_start, gff_end, strand


def main(arguments):
    # read in big chunk of h5s
    data = h5py.File(arguments.h5_data, mode='r')
    preds = h5py.File(arguments.predictions, mode='r')
    # open output file
    hints_handle = open(arguments.hints_out, 'w')

    # setup parameterized step/size for easy parsing based on prediction argmax
    ir_step_and_max = (arguments.step_irpart, arguments.step_irpart)
    genic_step_and_max = (arguments.step_genicpart, arguments.step_genicpart)
    hint_step_key = (ir_step_and_max, genic_step_and_max, genic_step_and_max, genic_step_and_max)

    # step through
    for contiguous_bit in get_contiguous_ranges(h5=data):
        for pred_chunk, start, end in read_in_chunks(preds, data, contiguous_bit['start_i'], contiguous_bit['end_i']):
            # break into pieces anywhere where the confidence drops or the category switches
            for start_conf, end_conf in find_confident_single_class_regions(pred_chunk, arguments.pad):
                one_class_chunk = pred_chunk[start_conf:end_conf]
                # pad and break further, down to min size confidence is volatile or up to max if stable
                # use average prediction confidence as score
                for pre_hint in divvy_by_confidence(one_class_chunk, hint_step_key, pad=arguments.pad):
                    # convert to gff entry & write
                    # gff fields
                    sequence = contiguous_bit['seqid'].decode()
                    source = 'Helixer'
                    feature = HINTS[pre_hint['category']]
                    # resolve all relative coordinates and convert to gff
                    gff_start, gff_end, strand = start_end_strand(contiguous_bit=contiguous_bit,
                                                                  pred_chunk_start=start,
                                                                  one_category_start=start_conf,
                                                                  pre_hint=pre_hint)
                    score = pre_hint['confidence']
                    phase = '.'   # Helixer has no phase info
                    attribute = 'source=H'   # just to say from Helixer, must be specified in extrinsic.cfg to match
                    gff_fields = [sequence, source, feature, gff_start, gff_end, score, strand, phase, attribute]
                    gff_fields = [str(x) for x in gff_fields]
                    gff_entry = '\t'.join(gff_fields)
                    hints_handle.write(gff_entry + '\n')

    hints_handle.close()
    data.close()
    preds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predictions', help='predictions.h5 file produced by Helixer', required=True)
    parser.add_argument('-d', '--h5-data', help='h5 file that was used as input to make predictions', required=True)
    parser.add_argument('-o', '--hints-out', help='output gff file of hints', required=True)
    parser.add_argument('--step-genicpart', default=10, type=int)
    parser.add_argument('--max-genicpart-size', default=500, type=int)
    parser.add_argument('--step-irpart', default=100, type=int)
    parser.add_argument('--max-irpart-size', default=10_000, type=int)
    parser.add_argument('--pad', default=5, type=int)
    parser.add_argument('--stability-threshold', default=0.1, type=float,
                        help='sets hint size by changes in prediction confidence, set high for few hints (push towards '
                             'max-size) and low for many hints (push towards step size)')
    args = parser.parse_args()
    main(args)
