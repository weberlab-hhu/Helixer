import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings
import intervaltree as ivt
import statistics
import sys
import argparse
import logging
from add_ngs_coverage import add_empty_cov_meta



# identifies start of 5'UTRs according to reference
def get_utr_positions(file, start=0, end="max", stepsize=1000):
    whole_utrs = []
    if end == "max":
        end = file['data/y'].shape[0]
    else:
        end = end
    # loop to iterate over the h5 file , reading in stepsize chunks at once
    for i in range(start, end, stepsize):
        chunk = file['data/y'][i:(i+stepsize)].T
        seqid = file['data/seqids'][i:(i+stepsize)].T
        start_ends = file['data/start_ends'][i:(i+stepsize)].T
        # loop to go over each individual loop of the stepsize chunks loaded at once
        for x in range(0,chunk.shape[2]):
            individual_chunk = chunk[:,:,x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:,x]
            start = individual_start_ends[0]
            end = individual_start_ends[1]
            strand = end - start
            # the strand is determined
            if  strand < 0:
                strand = -1
            else:
                strand = 1
            # dimensions of the array are reduced, to only catch shifts from IG to UTR and only capture 5' UTRs
            individual_chunk = individual_chunk[0:1,:]
            # whole array is shifted by 1 to the right
            shifted_array = individual_chunk[0:1,:-1]
            individual_chunk = individual_chunk[0:1,1:]
            # difference between the array will have a -1 at each transition from IG to UTR
            utr_positions = np.subtract(individual_chunk, shifted_array)
            utr_positions = np.where(utr_positions == -1)
            # the start position is added to preserve the coordinates over multiple chunks
            utr_positions = utr_positions[1] + start
            if len(utr_positions) == 0:
                continue
            utr_positions = utr_positions[0]
            whole_utrs.append((utr_positions, individual_seqid, strand, x+i))
    return whole_utrs

# identifies positions of cage peaks on the chunk
def get_cage_peak(file,threshold, start=0, end="max",stepsize=1000):
    centered = []
    if end == "max":
        end = file['data/y'].shape[0]
    else:
        end = end
    # loop to iterate over the h5 file , reading in stepsize chunks at once
    for i in range(start, end, stepsize):
        chunk = file['evaluation/cage_coverage'][i:(i+stepsize)].T
        seqid = file['data/seqids'][i:(i+stepsize)].T
        start_ends = file['data/start_ends'][i:(i+stepsize)].T
        # loop to go over each individual loop of the stepsize chunks loaded at once
        for x in range(0,chunk.shape[2]):
            individual_chunk = chunk[:,:,x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:,x]
            start = individual_start_ends[0]
            end = individual_start_ends[1]
            strand = end - start
            # the strand is determined
            if  strand < 0:
                strand = -1
            else:
                strand = 1
            chunk_shape = np.shape(individual_chunk) # shape of the input array
            threshold_array = np.full((chunk_shape) , threshold) # creates an array filled with the threshold value of the same shape as input
            #find ways to determine that threshold
            peak_positions = np.subtract(individual_chunk, threshold_array)
            peak_positions = np.where(peak_positions > 0) # only values greater than the threshold will result in positiv value
            peak_positions = peak_positions[1] + start # reducing the dimension of the array
            # print('peak_positions: {}'.format(peak_positions))
            peak_indices = np.where(np.diff(peak_positions) != 1) # selecting only indices of values apart more than 1
            if len(peak_positions) == 0:
                continue
            peak_indices = np.append(peak_indices, len(peak_positions) -1) # does not catch the last index therefore its added here
            ends = np.take(peak_positions, peak_indices) # end positions of the peaks
            start_indices = peak_indices + 1 # same procedure for the start values
            start_indices = np.insert(start_indices, 0, 0) # here the first one needs to be added
            start_indices = np.delete(start_indices, -1)
            starts = np.take(peak_positions, start_indices)
            #print(start_coordinates)
            # print('starts: {}, ends: {}'.format(starts, ends))
            # unfortunately no solution without a loop, in order to combine start and end values in tuples
            try:
                for y in range(len(starts)):
                     centered.append(((starts[y] , ends[y]), individual_seqid, strand, get_max_height(starts[y],ends[y],start,individual_chunk)))
            except UnboundLocalError:
                    print("No peaks")
                    return centered
    return centered


def get_max_height(starts,ends,start,chunk):
    # whole max height is flawed since it will always return max value for that area of the genome no matter what
    # particular peak had this value of coverage, so if there are 3 overlapping peaks, the highest coverage value will
    # be assigned to every peak in that area
    # maybe some sort of average or something will do
    #print('starts:{}, ends:{}, start:{}'.format(starts, ends, start))
    raw_start = starts - start
    raw_end = ends - start
    if raw_start ==  raw_end:
        raw_end += 1
    #print('raw_start:raw_end', raw_start,':', raw_end)
    max_height = chunk[ :, raw_start:raw_end]
    #print(max_height)
    max_height = max_height[max_height > 0]
    max_height = np.median(max_height)
    return max_height

def to_trees(centered_peaks):
    trees = {}
    not_null = 0
    # for loop through all peaks
    for x in centered_peaks:
        if x[0][0] == x[0][1]:
            not_null = 1
        # if a combination of seqid and strand has not been encountered yet its added to the dict accompanied
        # by an intervalltree representing the center of a cage peak
        if "{},{}".format(x[1], x[2]) not in trees.keys():
                peaks = ivt.IntervalTree()
                trees.setdefault(("{},{}".format(x[1], x[2])), peaks)
                peaks[x[0][0]:x[0][1] + not_null ] = (x[0], x[-1])
        # if the combination already exists the corresponing peak is added to the intervalltree at that position
        else:
            peaks[x[0][0]:x[0][1] + not_null] = (x[0], x[-1])
            trees["{},{}".format(x[1], x[2])] = peaks
        not_null = 0
        # peaks = trees[(seqid, strand)]
        # adding an intervall at the position of the peak to the intervaltree
    # returns the dictionary full of intervaltrees
    return trees


def get_distance(trees, utrs, lim):
    overlap = []
    distances = []
    if len(utrs) == 0 or len(trees) == 0:
        print("No UTRs or peaks")
        return overlap, i
    for x in (utrs):
        i = 0
        key = "{},{}".format(x[1], x[2])  # key based on the UTRs seqid and strand is created
        try:
            peaks = trees[key]
        # if there is a KeyError the iteration of the loop is skipped, there must be no peaks for that seqid
        # and/or strand
        except KeyError:
            # print('KeyError')
            continue
        # print(peaks)
        while len(overlap) == 0 and i <= lim:
            # takes peaks for this seqid strand pair and checks if any of the peaks are within an interval around
            # the TSS, if none are found the interval is increased by 1 up to a maximum of 500
            # this increasing value is the distance to the UTR, however, this will only reflect the distance to
            # the edge of the peak, the center might be a couple bp further away from the UTR depending on how
            # wide the peak might be.
            overlap = peaks.overlap(x[0] - i, x[0] + i + 1)
            # print('overlap: ', overlap, i)
            # if there is overlap and the distance is 0 that means there must be a peak around the TSS
            if i == 0 and len(overlap) != 0:
                distances.append((overlap, i, x))
                overlap = []
                break

            # if there is overlap and the distance is > 0 that means there must be a peak > 0 bp away from the TSS
            elif i != 0 and len(overlap) != 0:
                # bp between peak and UTR need to be scored based on the distance and the peak height
                # How to identify the specific peak? How to score its height?
                distances.append((overlap, i, x))
                overlap = []
                break
                # do something with the distance
            i += 1
    return distances

    # if x[0] in range(peaks.begin,  peaks.end):
    # print('score 1')
    # else:
    # print('get distance')

def add_empty_score_datasets(h5):
    length = h5['data/X'].shape[0]
    chunk_size = h5['data/X'].shape[1]
    h5.create_group(SCORE_STR)
    shapes = [(length, chunk_size), (length, 4), (length, 4), (length, ), (length, ), (length, chunk_size, 2)]
    max_shapes = [(None, chunk_size), (None, 4), (None, 4), (None, ), (None, ), (None, chunk_size, 2)]
    for i, key in enumerate(['by_bp', 'four', 'four_centered', 'one', 'one_centered', 'norm_cov_by_bp']):
        h5.create_dataset(f'{SCORE_STR}/{key}',
                          shape=shapes[i],
                          maxshape=max_shapes[i],
                          dtype="float",
                          compression="lzf",
                          fillvalue=-1)

def sigmoid(x, shift, width, span=1):
    f = 1 / (span + np.exp((x - shift) * width))
    return f


# is this necessary?
def find_stretch(cut_off=500):
    # method to find shift and slope combination that ensures 0 > 0.99 cut_off < 0.00
    for shift in range(0, cut_off):
        for slope in np.arange(1, 0.001, -0.01):

            zero = sigmoid(0, shift, slope)
            co = sigmoid(cut_off, shift ,slope)
            if zero > 0.99 and co < 0.01:
                #print(f'shift:{shift}, slope{slope} \n y:{zero}, co:{co} \n')
                return shift, slope


# function that takes in IntervallTree for a peak/UTR pair that needs to be scored, and extracts the peak position
def get_peak_from_tree(s):
    for x in s:
        start = x.begin
        end = x.end
        return (start, end)


# function to score bp between peak and UTR only distance is really used rn, the positions that need to be
# weighted and the score are returned
def get_score(datay, distance, cov, sc):
    # Naive attempt at scoring the coverage based on the distance it has to an annotated cage peak
    # height of the peak is currently not a factor, thinking of simply multiplying the score by a factor
    # based on the height

    # currently the distance is taken and used in a sigmoid function that is adjusted such that the cutoff point
    # is smaller than 0.01 and 0 is larger than 0.99, idk if this is useful this way but it ensures that
    # the score close to the peak is highest and close to the cutoff point lowest
    scores = []
    positions = []
    for x in distance:
        utr_pos = x[2][0]
        peak_pos = get_peak_from_tree(x[0])
        shift, slope = find_stretch(500)
        if peak_pos[0] > utr_pos:
            pos = (utr_pos, peak_pos[1])
            score = sigmoid(x[1], shift, slope)
            scores.append(score)
            positions.append(pos)
        # peak before utr
        elif peak_pos[1] < utr_pos:
            pos = (peak_pos[0], utr_pos)
            score = sigmoid(x[1], shift, slope)
            scores.append(score)
            positions.append(pos)
        # peak around peak
        elif peak_pos[0] <= utr_pos <= peak_pos[1]:
            pos = (peak_pos[0], peak_pos[1])
            score = 1
            scores.append(score)
            positions.append(pos)
        # print(f'score: {score}')

    return scores, positions


class Scorer:
    def __init__(self, median_cov, column, sigmoid=True):
        self.scale_to = 2.6
        self.column = column
        self.median_cov = median_cov
        self.final_scale = self.scale_pos_half
        self.run_sigmoid = sigmoid

    @staticmethod
    def scale_pos_half(score):
        # starts (0.5, 1), target (0, 1)
        return (score - 0.5) * 2

    @staticmethod
    def scale_neg_half(score):
        # starts (0, 0.5), target (0, 1)
        return score * 2

    @staticmethod
    def scale_nada(score):
        return score

    def score(self, datay, coverage, spliced_coverage):
        if self.column is not None:
            mask = datay[:, self.column].astype(bool)
        else:
            # this will effectively mask padding only
            mask = np.sum(datay, axis=1).astype(bool)
        cov = coverage[mask]
        sc = spliced_coverage[mask]
        if cov.shape[0]:
            pre_score = self._pre_score(cov, sc)
            if self.run_sigmoid:
                score = self.sigmoid(pre_score)
            else:
                score = pre_score
            score = self.final_scale(score)
        else:
            score = np.array([])

        return score, mask  # todo, maybe full score can be used _with_ and mask not returned??

    def _pre_score(self, cov, sc):
        # calculates a proxy score for how well the reference is supported by RNAseq data
        # exact equation depends on annotation (subclass)
        raise NotImplementedError

    @staticmethod
    def sigmoid(x):
        # avoid overflow
        x[x > 30] = 30
        x[x < -30] = -30
        # sigmoid
        return 1 / (1 + np.exp(-x))

class ScorerIntergenic(Scorer):
    def __init__(self, median_cov, column):
        super().__init__(median_cov, column=column)
        self.final_scale = self.scale_neg_half

    def _pre_score(self, cov, sc):
        x = np.log(cov + sc + 1)
        # normalize (different .bams have different coverage)
        x = x * self.scale_to / np.log(self.median_cov + 1) * -1
        return x


class NormScorer(Scorer):
    def __init__(self, median_cov, column):
        super().__init__(median_cov, column, sigmoid=False)
        self.final_scale = self.scale_nada


class NormScoreCoverage(NormScorer):
    def _pre_score(self, cov, sc):
        x = np.log10(cov + 1)
        return x / np.log10(self.median_cov + 1)


class NormScoreSplicedCoverage(NormScorer):
    def _pre_score(self, cov, sc):
        x = np.log10(sc + 1)
        return x / np.log10(self.median_cov + 1)


def get_median_expected_coverage(h5, max_expected=1000):
    # this will fail unless max_expected is higher than the median
    # but since we're expecting a median around say 5-20... should be OK
    start, end = 0, h5['data/X'].shape[0]
    bins = list(range(max_expected)) + [np.inf]
    by = 1000
    histo = np.zeros(shape=[len(bins) - 1])
    for i in range(start, end, by):
        counts, _ = histo_expected_coverage(h5, start_i=i, end_i=min(i + by, end), bins=bins)
        histo += counts

    half_total_bp = np.sum(histo) / 2
    cumulative = 0
    for bi in range(len(histo)):
        cumulative += histo[bi]
        if cumulative == half_total_bp:
            # if we're exactly half way, _and_ between bins, then the median is the average of the middle two
            median = bi + 0.5
            break
        elif cumulative > half_total_bp:
            # median was in this bin
            median = bi
            break
    if max_expected == median + 1:
        logging.warning('median is in last bin [{}, {}] and therefore undefined'.format(bins[-2], bins[-1]))
    if median == 0:
        logging.warning('median is 0, ignoring reality and setting to 1 to avoid dividing by 0 later')
        median = 1
    return median


def histo_expected_coverage(h5, start_i, end_i, bins):
    masked_cov = h5[f'evaluation/{COV_STR}'][start_i:end_i][
        np.logical_or(h5['data/y'][start_i:end_i, :, 1], h5['data/y'][start_i:end_i, :, 2])
    ]
    histo = np.histogram(masked_cov, bins=bins)
    return histo


# most of the functions and ways things are done are copied from Helixer/helixer/evaluation/score_rnaseq.py script
def sum_last_and_flatten(x):
    return np.sum(x, axis=-1).ravel()


def main(species, h5_data):

    # open h5
    h5 = h5py.File(h5_data, 'r+')
    # create evaluation, score, & metadata placeholders if they don't exist
    # (evaluation/coverage, "/spliced_coverage, scores/*, meta/*)
    try:
        h5[f'evaluation/{COV_STR}']
    except KeyError as e:
        print('ATTENTION: coverage must now be added via the "add_ngs_coverage.py" script, before calling this one.'
              '--dataset-prefix must match for both scripts (scripts have different defaults!)',
              file=sys.stderr)
        raise e
    try:
        h5[f'{SCORE_STR}/by_bp']
    except KeyError:
        add_empty_score_datasets(h5)

        # add remaining grp not currently part of rnaseq as it's unneeded there
    try:
        h5[f'{META_STR}/median_expected_coverage']
    except KeyError:
        h5[META_STR].create_group('median_expected_coverage')

    try:
        h5[f'{META_STR}/max_normalized_cov_sc']
    except KeyError:
        h5[META_STR].create_group('max_normalized_cov_sc')

    median_coverage = get_median_expected_coverage(h5)
    h5[f'{META_STR}/median_expected_coverage'].attrs.create(name=species, data=median_coverage)
    mec = int(h5[f'{META_STR}/median_expected_coverage'].attrs[species])
    ig_scorer = ScorerIntergenic(column=0, median_cov=mec)
    scorers = [ig_scorer]
    norm_cov_scorer = NormScoreCoverage(column=None, median_cov=mec)
    norm_sc_scorer = NormScoreSplicedCoverage(column=None, median_cov=mec)
    norm_scorers = [norm_cov_scorer, norm_sc_scorer]  # calculate normalized coverage more than "scoring" persay
    max_norm_cov = 0
    max_norm_sc = 0

    i = 0
    by = 500
    start = 0
    end = h5['data/X'].shape[0]
    counts = np.zeros(shape=(end - start, 4))
    # loop to iterate over by chunks at once and score them
    for i in range(start, end, by):
        # print(f'i: {i}')
        if i + by > end:
            by_out = end - i
        else:
            by_out = by
        i_rel = i - start
        y = h5['data/y'][i:(i + by_out)]
        utr = get_utr_positions(h5, start=i, end=i + by, stepsize=by_out)
        peaks = get_cage_peak(h5, 2, start=i, end=i + by, stepsize=by_out)
        peaks_trees = to_trees(peaks)
        dist = get_distance(peaks_trees, utr, 500) # todo: make 500 a parameter
        datay = y.reshape([-1, 4])
        _, chunk_size, n_cats = y.shape
        start_rel = (i * chunk_size)  # relative start of each by chunk_of_chunks in order to correctly add the score
        # different coverages not used as much as in the RNAseq script
        coverage = h5[f'evaluation/{COV_STR}'][i:(i + by_out)]
        spliced_coverage = h5[f'evaluation/{SC_STR}'][i:(i + by_out)]
        # coverage and spliced_coverage will initially have dimensions [n_chunks, chunk_size, n_bams]
        # but scoring assumes shape [n_basepairs], so they must be summed and flattened
        coverage, spliced_coverage = sum_last_and_flatten(coverage), sum_last_and_flatten(spliced_coverage)

        by_bp = np.full(fill_value=-1., shape=[by_out * chunk_size])
        # print(by_bp)
        norm_cov_by_bp = np.full(fill_value=-1., shape=[by_out * chunk_size, 2])  # 2 for [cov, sc]
        score, pos = get_score(datay, dist, coverage, spliced_coverage)  # actual scoring happens for by chunk of chunks
        # loop to add positions and scoring to the by_bp class in the h5file
        for scorer in scorers:
            raw_score, mask = scorer.score(datay=datay, coverage=coverage, spliced_coverage=spliced_coverage)
            if raw_score.size > 0:
                by_bp[mask] = raw_score
        for index_asif, norm_scorer in enumerate(norm_scorers):
            raw_score, mask = norm_scorer.score(datay=datay, coverage=coverage,
                                                spliced_coverage=spliced_coverage)
            if raw_score.size > 0:
                norm_cov_by_bp[mask, index_asif] = raw_score

        if len(pos) > 0:
            for e, x in enumerate(pos):
                pos_start = x[0]
                pos_end = x[1]
                by_bp[pos_start - start_rel:pos_end - start_rel] = score[e]  # adding to flat array size of by chunks
        del coverage, spliced_coverage
        by_bp = by_bp.reshape([by_out, chunk_size])  # reshaped so it fits the chunk structure again
        norm_cov_by_bp = norm_cov_by_bp.reshape([by_out, chunk_size, 2])  # don't know what happens with norm_cov
        h5[f'{SCORE_STR}/by_bp'][i:(i + by_out)] = by_bp
        h5[f'{SCORE_STR}/norm_cov_by_bp'][i:(i + by_out)] = norm_cov_by_bp
        max_norm_cov = max(max_norm_cov, np.max(norm_cov_by_bp[:, 0]))
        max_norm_sc = max(max_norm_sc, np.max(norm_cov_by_bp[:, 1]))

        # additional scoring again idk what exactly these scores do, mostly copied from github
        current_counts = np.sum(h5['data/y'][i:(i + by_out)], axis=1)
        scores_four = np.full(fill_value=-1., shape=[by_out, 4])
        scores_one = np.full(fill_value=-1., shape=[by_out])
        for j in range(by_out):  # todo, can I replace this with some sort of apply?
            for col in range(4):
                mask = y[j, :, col].astype(bool)
                if np.any(mask):
                    scores_four[j, col] = np.mean(by_bp[j][y[j, :, col].astype(bool)])
                else:
                    scores_four[j, col] = 0.
            scores_one[j] = np.sum(current_counts[j] * scores_four[j] / np.sum(current_counts[j]))
        counts[i_rel:(i_rel + by_out)] = current_counts
        h5[f'{SCORE_STR}/four'][i:(i + by_out)] = scores_four
        h5[f'{SCORE_STR}/one'][i:(i + by_out)] = scores_one
        if not i % 2000:
            # is it a problem if this is reached?
            print('reached i={}'.format(i))
        # or this?
        print('fin, {}'.format(i + by_out), file=sys.stderr)
        # save maximums as attributes
        h5[f'{META_STR}/max_normalized_cov_sc'].attrs.create(name=species, data=(max_norm_cov, max_norm_sc))
        # normalize coverage score by species and category
        raw_four = h5[f'{SCORE_STR}/four'][start:end]
        centers = np.sum(raw_four * counts, axis=0) / np.sum(counts, axis=0)
        centered_four = raw_four - centers
        # cat_counts = np.sum(counts, axis=0)
        # weighted_four = centered_four / np.sum(cat_counts) * cat_counts ### make sure works along last axis
        centered_one = np.sum(centered_four * counts, axis=1) / np.sum(counts, axis=1)
        h5[f'{SCORE_STR}/four_centered'][start:end] = centered_four
        h5[f'{SCORE_STR}/one_centered'][start:end] = centered_one
    h5.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--h5-data', help='h5 data file with /data/{X, y, species, seqids, etc...} '
                                                'AND with /evaluation/{{prefix}_coverage, {prefix}_spliced_coverage} '
                                                'to which {prefix}_scores will be added',
                        required=True)
    parser.add_argument('-s', '--species', help='species name matching that used in creation of geenuff/h5',
                        required=True)
    parser.add_argument('--dataset-prefix', help='prefix of h5 datasets to be used for scoring (default "rnaseq")',
                        default='cage')
    args = parser.parse_args()
    COV_STR = f'{args.dataset_prefix}_coverage'
    SC_STR = f'{args.dataset_prefix}_spliced_coverage'
    META_STR = f'{args.dataset_prefix}_meta'
    SCORE_STR = f'{args.dataset_prefix}_scores'
    main(args.species, args.h5_data)
