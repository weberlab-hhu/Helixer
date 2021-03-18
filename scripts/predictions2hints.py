"""converts Helixer's output predictions into AUGUSTUS compatible hints"""
import argparse
import dustdas
import h5py
import numpy as np


HINTS = ['irpart', 'UTRpart', 'CDSpart', 'intronpart']


def get_contiguous_ranges(h5):
    start_ends = h5['data/start_ends'][:]
    marks_unique = np.stack((h5['data/seqids'], h5['data/species'], start_ends[:, 1] > start_ends[:, 0]))
    items, indexes, lengths = np.unique(marks_unique, axis=1, return_index=True, return_counts=True)
    reindex = np.argsort(indexes)
    print(items.shape)
    print(reindex.shape, 'r shape')
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
        start = data['data/start_ends'][i, 0]
        end = data['data/start_ends'][ei - 1, 1]
        yield pred_chunk, start, end


def find_confident_single_class_regions(pred_chunk, pad=5):
    # in the event of a seamless class swap [1,0,0,0] -> [0,1,0,0]
    # and of a thin-seemed swap [1,0,0,0] -> [0.5,0.5,0,0] -> [0,1,0,0]
    # (and any easier swap, as well)
    # the 2-bp smoothing below will allow us to detect a class swap by drop in confidence (max)
    shift_n_averaged = (pred_chunk[:-1] + pred_chunk[1:]) / 2
    # find anywhere network was not confident / or class switch
    lowconf_idx = np.where(np.max(shift_n_averaged, axis=1) < 0.75)[0]
    if lowconf_idx.shape[0] == 0:
        print(np.sum(shift_n_averaged, axis=0))
        print(np.min(shift_n_averaged, axis=0))
        yield 0, pred_chunk.shape[0]
        return
    # handle (& yield) start of chunk edge case if pred_chunk starts with a confident region
    if lowconf_idx[0] > pad * 2:
        yield 0, lowconf_idx[0]
    # invert above to find longer single-class confident regions (double our padding)
    dist2next_lowconf = lowconf_idx[1:] - lowconf_idx[:-1]
    conf_sub_idx = np.where(dist2next_lowconf > pad * 2)

    # yield confident regions
    for sub_idx in conf_sub_idx:
        start = lowconf_idx[sub_idx] + 1  # confidence starts _after_ lowconf ends
        end = start + dist2next_lowconf[sub_idx] - 1  # (exclusive) confidence ends where lowconf starts
        yield start, end

    # handle (& yield) end of chunk edge case if pred_chunk ends with a confident region
    if pred_chunk.shape[0] - lowconf_idx[-1] > pad * 2:
        yield lowconf_idx[-1], pred_chunk.shape[0]


def main(arguments):
    # read in big chunk of h5s
    data = h5py.File(arguments.h5_data, mode='r')
    preds = h5py.File(arguments.predictions, mode='r')
    # step through
    for contiguous_bit in get_contiguous_ranges(h5=data):
        for pred_chunk, start, end in read_in_chunks(preds, data, contiguous_bit['start_i'], contiguous_bit['end_i']):
            pred_chunk = pred_chunk.reshape((-1, 4))
            # break into pieces anywhere where the confidence drops or the category switches
            for start_conf, end_conf in find_confident_single_class_regions(pred_chunk, arguments.pad):
                print(f'bit from {start_conf}-{end_conf} in region {start}-{end} on '
                      f'is_plus {contiguous_bit["is_plus_strand"]} of {contiguous_bit["seqid"]} is one class conf')
    # pad and break further, down to min size confidence is volatile or up to max if stable
    # use average prediction confidence as score
    # convert to gff entry & write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predictions', help='predictions.h5 file produced by Helixer', required=True)
    parser.add_argument('-d', '--h5-data', help='h5 file that was used as input to make predictions', required=True)
    parser.add_argument('-o', '--hints-out', help='output gff file of hints', required=True)
    parser.add_argument('--step-genicpart', default=10, type=int)
    parser.add_argument('--max-genicpart-size', default=100, type=int)
    parser.add_argument('--step-irpart', default=100, type=int)
    parser.add_argument('--max-irpart-size', default=10_000, type=int)
    parser.add_argument('--pad', default=5, type=int)
    args = parser.parse_args()
    main(args)
