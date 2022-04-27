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
<<<<<<< HEAD
"""
########################################################################################################################
# Python script to automatically score each bp within a h5 file based on available CAGE data and the distance of the   #
# CAGE peaks to the annotated UTRs (if within a certain distance) or based on the coverage (if outside that distance). #
########################################################################################################################
"""


def get_utr_positions(file: h5py._hl.files.File, start: int = 0, end: int = None, stepsize: int = 1000):
    """Function to return list of tuples (utr position, seqid, strand, chunk) for each annotated 5' UTR."""
    whole_utrs = []
    if end is None:
=======

########################################################################################################################
# Python script to automatically score each bp within an h5 file based on available CAGE data and the distance of the  #
# CAGE peaks to the annotated UTRs (if within a certain distance) or based on the coverage (if outside that distance). #
########################################################################################################################


# identifies start of 5'UTRs according to reference
def get_utr_positions(file, start=0, end="max", stepsize=1000) :
    # function to return list of tuples (utr position, seqid, strand, chunk) for each annotated 5' UTR
    whole_utrs = []
    if end == "max":
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
        end = file['data/y'].shape[0]
    else:
        end = end
    # loop to iterate over the h5 file , reading in stepsize chunks at once
    for i in range(start, end, stepsize):
        chunk = file['data/y'][i:(i+stepsize)].T
        seqid = file['data/seqids'][i:(i+stepsize)].T
        start_ends = file['data/start_ends'][i:(i+stepsize)].T
        # loop to go over each individual chunk of the stepsize chunks loaded at once
<<<<<<< HEAD
        for x in range(0, chunk.shape[2]):
            # todo: define individual function performing the steps on individual chunk
            individual_chunk = chunk[:, :, x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:, x]
=======
        for x in range(0,chunk.shape[2]):
            individual_chunk = chunk[:,:,x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:,x]
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
            start = individual_start_ends[0]
            end = individual_start_ends[1]
            strand = end - start
            # the strand is determined
<<<<<<< HEAD
            if strand < 0:
=======
            if  strand < 0:
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
                strand = -1
            else:
                strand = 1
            # dimensions of the array are reduced, to only catch shifts from IG to UTR and only capture 5' UTRs
<<<<<<< HEAD
            individual_chunk = individual_chunk[0:1, :]
            # whole array is shifted by 1 to the right
            shifted_array = individual_chunk[0:1, :-1]
            individual_chunk = individual_chunk[0:1, 1:]
=======
            individual_chunk = individual_chunk[0:1,:]
            # whole array is shifted by 1 to the right
            shifted_array = individual_chunk[0:1,:-1]
            individual_chunk = individual_chunk[0:1,1:]
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
            # difference between the array will have a -1 at each transition from IG to UTR
            utr_positions = np.subtract(individual_chunk, shifted_array)
            utr_positions = np.where(utr_positions == -1)
            # the start position is added to preserve the coordinates over multiple chunks
            utr_positions = utr_positions[1] + start
            # checking to see if there are any UTRs within the chunk, if not this iteration of the loop is skipped
            if len(utr_positions) == 0:
                continue
            utr_positions = utr_positions[0]
            whole_utrs.append((utr_positions, individual_seqid, strand, x+i))
    return whole_utrs


<<<<<<< HEAD
def get_cage_peak(file: h5py._hl.files.File, threshold: int = 3, start: int = 0, end: int = None, stepsize: int = 1000):
    """Function to return list of tuples containing ((start of peak, end of peak), seqid, strand, height of peak) of
    peaks larger than THRESHOLD."""
    peaks = []
    if end is None:
=======
# identifies positions of cage peaks on the chunk
def get_cage_peak(file, threshold=3, start=0, end="max",stepsize=1000):
    # function to return list of tuples containing ((start of peak, end of peak), seqid, strand, height of peak) of
    # peaks larger than THRESHOLD
    peaks = []
    if end == "max":
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
        end = file['data/y'].shape[0]
    else:
        end = end
    # loop to iterate over the h5 file , reading in stepsize chunks at once
    for i in range(start, end, stepsize):
        chunk = file['evaluation/cage_coverage'][i:(i+stepsize)].T
        seqid = file['data/seqids'][i:(i+stepsize)].T
        start_ends = file['data/start_ends'][i:(i+stepsize)].T
        # loop to go over each individual chunk of the stepsize chunks loaded at once
<<<<<<< HEAD
        for x in range(0, chunk.shape[2]):
            # todo: define individual function performing the steps on individual chunk
            individual_chunk = chunk[:, :, x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:, x]
=======
        for x in range(0,chunk.shape[2]):
            individual_chunk = chunk[:,:,x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:,x]
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
            start = individual_start_ends[0]
            end = individual_start_ends[1]
            strand = end - start
            # the strand is determined
<<<<<<< HEAD
            if strand < 0:
                strand = -1
            else:
                strand = 1
            chunk_shape = np.shape(individual_chunk)  # shape of the input array
            threshold_array = np.full(chunk_shape, threshold)  # creates an array filled with the threshold value of
            # the same shape as input
            # todo: find ways to determine that threshold
            peak_positions = np.subtract(individual_chunk, threshold_array)
            peak_positions = np.where(peak_positions > 0)  # only values greater than the threshold will result in
            # positive value
            peak_positions = peak_positions[1] + start  # reducing the dimension of the array
            peak_indices = np.where(np.diff(peak_positions) != 1)  # selecting only indices of values apart more than 1
            if len(peak_positions) == 0:
                continue
            peak_indices = np.append(peak_indices, len(peak_positions) - 1)  # does not catch the last index therefore
            # its added here
            ends = np.take(peak_positions, peak_indices)  # end positions of the peaks
            start_indices = peak_indices + 1  # same procedure for the start values
            start_indices = np.insert(start_indices, 0, 0)  # here the first one needs to be added
=======
            if  strand < 0:
                strand = -1
            else:
                strand = 1
            chunk_shape = np.shape(individual_chunk) # shape of the input array
            threshold_array = np.full((chunk_shape) , threshold) # creates an array filled with the threshold value of
            # the same shape as input
            # todo: find ways to determine that threshold
            peak_positions = np.subtract(individual_chunk, threshold_array)
            peak_positions = np.where(peak_positions > 0) # only values greater than the threshold will result in positiv value
            peak_positions = peak_positions[1] + start # reducing the dimension of the array
            peak_indices = np.where(np.diff(peak_positions) != 1) # selecting only indices of values apart more than 1
            if len(peak_positions) == 0:
                continue
            peak_indices = np.append(peak_indices, len(peak_positions) -1) # does not catch the last index therefore
                                                                           # its added here
            ends = np.take(peak_positions, peak_indices) # end positions of the peaks
            start_indices = peak_indices + 1 # same procedure for the start values
            start_indices = np.insert(start_indices, 0, 0) # here the first one needs to be added
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
            start_indices = np.delete(start_indices, -1)
            starts = np.take(peak_positions, start_indices)
            # unfortunately no solution without a loop, in order to combine start and end values in tuples
            try:
                for y in range(len(starts)):
<<<<<<< HEAD
                    peaks.append(((starts[y], ends[y]), individual_seqid, strand, get_median_height(starts[y], ends[y],
                                                                                                    start,
                                                                                                    individual_chunk)))
            except UnboundLocalError:
                print("No peaks")
                return peaks
    return peaks


def get_median_height(starts: list, ends: list, start: int, chunk: int):
    """Function to return median height of all peaks in a specified area."""
    raw_start = starts - start  # determines start of peak on the individual chunk
    raw_end = ends - start  # determines end of peak on the individual chunk
    if raw_start == raw_end:
        raw_end += 1
    median_height = chunk[:, raw_start:raw_end]  # selects coverage of bp between raw_start and raw_end
    median_height = median_height[median_height > 0]  # positive coverage is selected
    median_height = np.median(median_height)
    return median_height


def to_trees(elements: list, utr=False):
    """Function to store all peaks/utrs determined by get_cage_peak/get_utr_positions within Intervaltrees stored within
    a dictionary separated by keys based on the peaks/utrs seqid/strand combination."""
    trees = {}
    not_null = 0
    utrs = None
    peaks = None
    if utr:
        # iterating over all utrs within list
        for x in elements:
            # if a seqid/strand combination is not in the dicts keys, the key is created and an intervaltree containing
            # the utrs position is added to that key
            if "{},{}".format(x[1], x[2]) not in trees.keys():
                utrs = ivt.IntervalTree()
                trees.setdefault(("{},{}".format(x[1], x[2])), utrs)
                utrs[x[0]:x[0] + 1] = (x[0])
            # if a seqid/strand combination does exist the utr position is simply added to it
            else:
                utrs[x[0]:x[0] + 1] = (x[0])
                trees["{},{}".format(x[1], x[2])] = utrs
        return trees

    else:
        # iterating over all elements of the list containing peak positions
        for x in elements:
            if x[0][0] == x[0][1]:  # intervaltree intervalls are noninclusive so if start=end the peak will have a
                # width of 0 bp, so if start=end not_null = 1 so it can be added to end later
                not_null = 1
            # if a combination of seqid and strand has not been encountered yet, it's added to the dict accompanied
            # by an intervaltree representing a cage peak
            if "{},{}".format(x[1], x[2]) not in trees.keys():
                peaks = ivt.IntervalTree()
                trees.setdefault(("{},{}".format(x[1], x[2])), peaks)
                peaks[x[0][0]:x[0][1] + not_null] = (x[0], x[-1])
            # if the combination already exists the corresponding peak is added to the intervalltree at that position
            else:
                peaks[x[0][0]:x[0][1] + not_null] = (x[0], x[-1])
                trees["{},{}".format(x[1], x[2])] = peaks
            not_null = 0
        # returns the dictionary full of intervaltrees
        return trees


def get_distance(peaks, utrs, average_positive_coverage):
    """Function to determine distance between CAGE peak and annotated UTR, if there is no UTR close to a peak the peak
    is  scored based on its height (coverage) rather than its distance to a UTR. First the largest peak within LIM bp
    is determined and scored, all remaining peaks outside of LIM bp of the UTR will be scored based on height."""
    scores = []
    poss = []
    _ = ''
    # check to see if UTRs and/or peaks exist within the dicts submitted to the function
    if len(utrs) == 0 or len(peaks) == 0:
        print("No UTRs or peaks")
        return None
    # loop iterating over list of UTRs
    for x in peaks.keys():
        # key based on the UTRs seqid and strand is used to find all peaks and utrs with these seqid/strands
        try:
            utr = utrs[x]
            peak = peaks[x]
        # if there is a KeyError the iteration of the loop is skipped, there must be no peaks/utrs for that seqid
        # and/or strand
        except KeyError:
            continue
        # iterating over all utr with a given seqid/strand
        for y in utr:
            utr_pos = y.begin
            # overlap of peaks within max_distance bp of an utr
            overlap = peak.overlap(utr_pos - max_distance, utr_pos + max_distance)
            # there are multiple peaks within max_distance bp of the UTR
            if len(overlap) > 1:
                # find the largest peak, score up to that peak
                largest_peak = find_largest_peak(overlap)
                distance = utr_pos - largest_peak.begin
                score, pos = get_score(largest_peak, int(distance), y, average_positive_coverage)
                scores.append(score)
                poss.append(pos)
            # there is only 1 peak within max_distance bp of the UTR
            elif len(overlap) == 1:
                # score up to single peak
                peak_pos, _ = get_peak_from_tree(overlap)
                distance = utr_pos - peak_pos
                score, pos = get_score(overlap, int(distance), y, average_positive_coverage, pain=True)
                scores.append(score)
                poss.append(pos)
        # iterating over all peaks for that seqid/strand
        for z in peak:
            start = z.begin
            end = z.end
            overlap = utr.overlap(start - max_distance, end + max_distance)  # overlap between the peak and an UTR
            # if there are not overlapping UTRs this means the peak must be outside the max_distance bp and thus is
            # scored individually
            if len(overlap) == 0:
                score, pos = get_score(z, max_distance, _, average_positive_coverage, score_utr=False)
                scores.append(score)
                poss.append(pos)
    return scores, poss


def get_score(overlap, distance, utr, average_positive_coverage, pain=False, score_utr=True):
    """Function to return are score associated with a position on chunk based on the distance of CAGE-peaks to annotated
    UTRs or their height"""
    # shape based on the average_positive_coverage
    pos = ()
    utr_pos, peak_height = None, None
    # scoring of individual peaks not associated with an UTR
    if not score_utr:
        peak_start = overlap.begin
        peak_end = overlap.end
        h = overlap[-1]  # has to be done this way because of intervalltree shenanigans
        peak_height = h[-1]
        score = 1  # peaks have no distance to a UTR so only the peak height is used for scoring
        score = score * (sigmoid(peak_height, shift=average_positive_coverage, slope=1, neg=False))
        pos = (peak_start, peak_end)
        return score, pos

    # depending on how the intervalltree is passed to the function it becomes more or less painful to extract the peak
    # height from the intervalltree, if largest peak has to be determined before it becomes less painful, if not, more
    if pain:
        peak_start, peak_end = get_peak_from_tree(overlap)
        for x in overlap:
            h = x[-1]
            peak_height = h[-1]
            utr_pos = utr.begin
    else:
        peak_start = overlap.begin
        h = overlap[-1]
        peak_height = h[-1]
        utr_pos = utr.begin

    if distance > 0:
        pos = (peak_start, utr_pos)
    # distance = 0 peak must be around the UTR
    elif distance == 0:
        if peak_start == utr_pos:
            utr_pos += 1
        pos = (peak_start, utr_pos)
    # negative distance, UTR upstream of peak
    elif distance < 0:
        pos = (utr_pos, peak_start)
        distance = distance * -1
    # todo: if no overlap (no utrs around the peak) score peak like intergenic scorer did
    # within max_distance bp of UTR
    if distance <= max_distance and distance != 0:
        score = linear_scoring(distance)
        score = score * (sigmoid(peak_height, shift=average_positive_coverage, slope=1, neg=False))
        return score, pos
    # peak and UTR at same position, upweighting based on peak height
    elif distance == 0:
        score = linear_scoring(distance)
        score = score * (sigmoid(peak_height, shift=-average_positive_coverage, slope=1, neg=True) + 1)
        return score, pos
    # handles the case where end of peak overlaps with utr interval but start of peak is outside the interval
    # it seems a lot easier to simply score the additional bp as if they were within the interval instead of handling
    # them as individual peaks, therefore that is what is implemented
    elif distance > max_distance:
        print(f'[WARNING] distance of {distance} larger than {max_distance} bp, {distance - max_distance} additional bp will be scored...')
        score = linear_scoring(distance)
        score = score * (sigmoid(peak_height, shift=average_positive_coverage/2, slope=1, neg=False))
        return score, pos


def add_empty_score_datasets(h5: h5py._hl.files.File):
    """Function to add necessary dataset to the input h5 data."""
=======
                     peaks.append(((starts[y] , ends[y]), individual_seqid, strand, get_median_height(starts[y],ends[y],
                                                                                            start,individual_chunk)))
            except UnboundLocalError:
                    print("No peaks")
                    return peaks
    return peaks


def get_median_height(starts,ends,start,chunk):
    # function to return median height of all peaks in a specified area
    raw_start = starts - start # determines start of peak on the individual chunk
    raw_end = ends - start # determines end of peak on the individual chunk
    if raw_start ==  raw_end:
        raw_end += 1
    median_height = chunk[ :, raw_start:raw_end] # selects coverage of bp between raw_start and raw_end
    median_height = median_height[median_height > 0] # positive coverage is selected
    median_height = np.median(median_height)
    return median_height

def to_trees(centered_peaks):
    # function to store all peaks determined by get_cage_peak within Intervaltrees stored within a dictionary separated
    # based on the peaks strand and seqid
    trees = {}
    not_null = 0
    # for loop through all peaks
    for x in centered_peaks:
        if x[0][0] == x[0][1]:
            not_null = 1
        # if a combination of seqid and strand has not been encountered yet its added to the dict accompanied
        # by an intervalltree representing a cage peak
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
    # function to determine the distance between annotated 5' UTRs and CAGE peaks within LIM basepairs, returns a list
    # of tuples containing (intervall tree object (? unsure of type of object) of overlaping cage-peaks, their distance,
    # tuple returned by get_utrs for the particular UTR these peaks are associated with)
    overlap = []
    distances = []
    # check to see if UTRs and/or peaks exist within the lists submitted to the function
    if len(utrs) == 0 or len(trees) == 0:
        print("No UTRs or peaks")
        return overlap, i
    # loop iterating over list of UTRs
    for x in (utrs):
        i = 0
        key = "{},{}".format(x[1], x[2])  # key based on the UTRs seqid and strand is created
        try:
            peaks = trees[key]
        # if there is a KeyError the iteration of the loop is skipped, there must be no peaks for that seqid
        # and/or strand
        except KeyError:
            continue
        while len(overlap) == 0 and i <= lim:
            # takes peaks for this seqid strand pair and checks if any of the peaks are within an interval around
            # the TSS, if none are found the interval is increased by 1 up to a maximum of LIM
            # this increasing value is the distance to the UTR, however, this will only reflect the distance to
            # the edge of the peak, the center might be a couple bp further away from the UTR depending on how
            # wide the peak might be.
            overlap = peaks.overlap(x[0] - i, x[0] + i + 1)
            # if there is overlap and the distance is 0 that means there must be a peak around the TSS
            if i == 0 and len(overlap) != 0:
                distances.append((overlap, i, x))
                overlap = []
                break

            # if there is overlap and the distance is > 0 that means there must be a peak > 0 bp away from the TSS
            elif i != 0 and len(overlap) != 0:
                # bp between peak and UTR need to be scored based on the distance and the peak height
                distances.append((overlap, i, x))
                overlap = []
                break
            i += 1
    return distances

def add_empty_score_datasets(h5):
    # function to add necessary dataset to the input h5 data
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
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


<<<<<<< HEAD
def sigmoid(x: float, shift: int = 0, slope: int = 1, neg: bool = False):
    """Function to return the sigmoid of input value with additional parameters in order to fully control the behavior
    of the sigmoid function, neg flag determines if the sigmoid function is inverted or not. """
    # todo: avoid overflow
    if not neg:
        f = 1 / (1 + np.exp((x / slope) - shift))
    else:
        f = 1 / (1 + np.exp((-x / slope) - shift))
    return f


def linear_scoring(distance):
    m = (0.1 - 1) / (max_distance - 0)
    score = distance * m + 1
    #print(f'score at a distance of {distance}: {score}')
    if score > 1:
        print('score greater than 1')
        score = 1
    elif score < 0:
        print('score lessthan 0')
        score = 0.01
    return score


def get_peak_from_tree(s: ivt.interval.Interval):
    """Function to return start position of a peak, its end position and its height from the data stored in an
    interval."""
=======
def sigmoid(x, width=1, shift=0, span=1, neg=False):
    # function to return the sigmoid of input value with additional parameters in order to fully control the behavior of
    # the sigmoid function, neg flag determines if the sigmoid function is inverted or not
    if neg:
        f = 1 / (span + np.exp(-((x / width) - shift)))
    else:
        f = 1 / (span + np.exp(((x / width) - shift)))
    return f

def find_stretch(cut_off=500):
    # method to find shift and slope combination that ensures sigmoid(0) > 0.99 and sigmoid(cut_off) < 0.01 as well as
    # 0.45 < sigmoid(x/2) < 0.55 in order to ensure the score scales properly with distance

    # for loop to try different values for shift
    for shift in range(0, cut_off):
        zero = sigmoid(0,1,shift)
        if zero > 0.99:
            break
    # for loop to try different values for slope
    for slope in range(1, cut_off):
        zero = sigmoid(0,slope,shift)
        co = sigmoid(cut_off,slope,shift)
        mid = sigmoid(cut_off/2,slope,shift)
        if zero > 0.99 and co < 0.01 and 0.45 <= mid <= 0.55:
            return shift, slope
    return 0, 1 # if no shift, slope combination can be determined normal sigmoid is used

def get_peak_from_tree(s):
    # function to return start position of a peak, its end position and its height from the data stored in an interval
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
    # tree
    for x in s:
        start = x.begin
        end = x.end
<<<<<<< HEAD
        return start, end


def get_average_cov(h5: h5py._hl.files.File):
    """Function to return the average of all positive coverage per bp by calculating
    (sum of all positive coverage)/(number of bp with positive coverage)."""
    start = 0
    end = 5000 # h5['data/X'].shape[0]
    by = 1000
    sum_chunks = 0
    length_chunks = 0
=======
        height = x[2][1]
        return (start, end), height


def get_average_cov(h5):
    # function to return the average of all positive coverage per bp by calculating
    # (sum of all positive coverage)/(number of bp with positive coverage)
    start = 0
    end = h5['data/X'].shape[0]
    by = 500
    sum_chunks = 0
    len_chunks = 0
    l = 0
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
    # loop to iterate over BY chunks at once
    for i in range(start, end, by):
        chunks = h5['evaluation/cage_coverage'][i:(i + by)]
        cov_chunks = np.ravel(chunks)  # flattens the array, combining all different bam files into 1 dimension
        pos = np.ravel(np.where(cov_chunks > 0))  # select only positions from array with positive coverage
        cov_chunks = cov_chunks[pos]
        if 0 in cov_chunks:
            return None

        sum_chunks += np.sum(cov_chunks)  # sum of all positive coverage over BY chunks
<<<<<<< HEAD
        len_chunks = np.shape(cov_chunks)  # len of BY flattened chunks
        length_chunks += len_chunks[0]  # sum of length of all chunks
    return int(round(sum_chunks / length_chunks))  # needs to be int otherwise type error later on


def find_largest_peak(overlap):
    """Function to determine peak with the highest coverage out of multiple peaks within max_distance bp of an UTR"""
    max_height = 0
    max_peak = None
    # iterating over the different peaks
    for x in overlap:
        peak = x
        h = x[-1]
        height = h[-1]
        if height > max_height:
            max_height = height
            max_peak = peak
    return max_peak
=======
        len_chunks = np.shape(cov_chunks)  # len of BY flattend chunks
        l += len_chunks[0]  # sum of length of all chunks
    average_positive_cov = sum_chunks / l
    return int(round(sum_chunks / l))  # needs to be int otherwise type error later on


def get_score(distance, max_distance, average_positive_coverage):
    # function to return are score associated with a position on a chunk based on the distance of CAGE-peaks to annotated
    # UTRs
    scores = []
    positions = []
    # for loop to iterate over list of tuples contaiing different cage peaks overlapping with 5' UTRs within a certain
    # inverval
    for x in distance:
        utr_pos = x[2][0]
        peak_pos, peak_height = get_peak_from_tree(x[0])
        shift, slope = find_stretch(max_distance)
        shift_height, slope_height = find_stretch(average_positive_coverage)
        # differentiating various cases of positioning of the CAGE-peak in relation to the UTR
        # todo: integrate peak height into this
        # peak starts after UTR
        if peak_pos[0] > utr_pos:
            pos = (utr_pos, peak_pos[1])
            score = sigmoid(x[1], shift, slope)
            score = score * (sigmoid(peak_height, shift=shift_height, width=slope_height, neg=False))
            # using sigmoid function between 0 and 1 with decreasing values as peak height increases to downweigh even
            # more if it is a tall peak and does not align with the UTR
            scores.append(score)
            positions.append(pos)
        # peak ends before utr
        elif peak_pos[1] < utr_pos:
            pos = (peak_pos[0], utr_pos)
            score = sigmoid(x[1], shift, slope)
            scores.append(score)
            score = score * (sigmoid(peak_height, shift=shift_height, width=slope_height, neg=False))
            # using sigmoid function between 0 and 1 with decreasing values as peak height increases to downweigh even
            # more if it is a tall peak and does not align with the UTR
            positions.append(pos)
        # peak around peak
        elif peak_pos[0] <= utr_pos <= peak_pos[1]:
            pos = (peak_pos[0], peak_pos[1])
            score = 1
            score = score * (sigmoid(peak_height, shift=shift_height, width=slope_height, neg=True) + 1)
            # using sigmoid function between 1 and 2 with increasing values as peak height increases to upweigh
            # UTRs aligning with tall CAGE peaks
            scores.append(score)
            positions.append(pos)

    return scores, positions
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140

# different scorers copied form score_rnaseq.py in order to score CAGE-peaks within IG for example, however, results
# in Peaks being scored based on their coverage and not based on their distance, for peaks in IG and Introns for example
# todo: find a distance based way of scoring for those peaks too?
<<<<<<< HEAD


class Scorer:
    """Main class for all scorers (except UTR scoring) defining a scoring method for all scorers."""
    def __init__(self, median_cov, column, sigmoid=True):
        self.scale_to = 2.6  # why 2.6?
        self.column = column  # is this the axis?
        self.median_cov = median_cov  # usually 1
=======
class Scorer:
    def __init__(self, median_cov, column, sigmoid=True):
        self.scale_to = 2.6
        self.column = column # is this the axis?
        self.median_cov = median_cov # usually 1 
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
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

<<<<<<< HEAD
    def score(self, datay: np.array, coverage: np.array, spliced_coverage: np.array,
              average_positive_coverage: int):
        """Function to calculate a per bp score based on coverage of that bp, also implements threshold check."""
=======
    def score(self, datay, coverage, spliced_coverage):
        # sets all coverage below the threshold to 0 in order to not score peaks below the threshold
        cov_shape = np.shape(coverage)
        sc_shape = np.shape(spliced_coverage)
        threshold_array = np.full((cov_shape), threshold)
        coverage = coverage - threshold_array
        threshold_array = np.full((sc_shape), threshold)
        spliced_coverage = spliced_coverage - threshold_array
        coverage[coverage < 0] = 0
        spliced_coverage[spliced_coverage < 0] = 0
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
        if self.column is not None:
            mask = datay[:, self.column].astype(bool)
        else:
            # this will effectively mask padding only
            mask = np.sum(datay, axis=1).astype(bool)
<<<<<<< HEAD
        # selects only those bp where there is coverage for that particular class (column) ?

        # sets all coverage below the threshold to 0 in order to not score peaks below the threshold
        # but also reduces all coverage by threshold so might skew some coverage
        # does not seem to work for coverage in exons and intergenic but unclear to me why
        cov = coverage[mask]
        threshold_array_cov = np.full(cov.shape, threshold)
        cov = cov / average_positive_coverage
        cov = cov - threshold_array_cov
        cov[cov < 0] = 0
        sc = spliced_coverage[mask]
        threshold_array_sc = np.full(sc.shape, threshold)
        sc = sc / average_positive_coverage
        sc = sc - threshold_array_sc
        sc[sc < 0] = 0
        if cov.shape[0]:
            # todo: find way to not score peaks close to UTRs twice with other scorer, if bp in that class!
            pre_score = self._pre_score(cov, sc)  # what purpose does the prescore serve? Same shape as cov so all
# bp within a class with coverage also have a prescore associated with them linked via mask?
            # print(f'where pre_score < 0:{np.where(pre_score<0)}')
            # print(f'test_pre_score:{pre_score[17531:17559]}')
            # idea is to have
            # a second array with all the peaks heights run it through some sort of function and multiply it with the
            # prescore array in order to account for peak height, although since the prescore is derived from the
            # coverage data maybe this is one sigmoid of peak height to many now
            if self.run_sigmoid:
                # shift, slope = find_stretch(max_distance)
                # score = sigmoid(pre_score, width=slope, shift=shift)
                score = self.sigmoid(pre_score)  # just regular sigmoid, no stretching and still scores sometimes higher

                # print(f'score at run sigmoid:{score}')
            else:
                score = pre_score
            score = self.final_scale(score)  # it makes scores go above 1 if stretched sigmoid is used
            # print(f'score at final_scale:{score}')
        else:
            score = np.array([])
=======
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

>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
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

<<<<<<< HEAD
# todo: find out if this is needed or not
=======
class ScorerIntergenic(Scorer):
    def __init__(self, median_cov, column):
        super().__init__(median_cov, column=column)
        self.final_scale = self.scale_neg_half

    def _pre_score(self, cov, sc):
        x = np.log(cov + sc + 1)
        # normalize (different .bams have different coverage)
        x = x * self.scale_to / np.log(self.median_cov + 1) * -1
        return x


class ScorerIntron(Scorer):
    def __init__(self, median_cov, column):
        super().__init__(median_cov, column=column)
        self.final_scale = self.scale_neg_half

    def _pre_score(self, cov, sc):
        x = np.log(cov + sc + 1)
        x = x * self.scale_to / np.log(self.median_cov + 1) * -1
        return x
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140


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


<<<<<<< HEAD
def get_median_expected_coverage(h5, max_expected: int = 1000):
    """Function to calculate median expected coverage over the whole h5 file"""
    # this will fail unless max_expected is higher than the median
    # but since we're expecting a median around say 5-20... should be OK
    start, end = 0, 5000 # h5['data/X'].shape[0]
    bins = list(range(max_expected)) + [np.inf]
    by = 1000
    median = None
=======
def get_median_expected_coverage(h5, max_expected=1000):
    # this will fail unless max_expected is higher than the median
    # but since we're expecting a median around say 5-20... should be OK
    start, end = 0, h5['data/X'].shape[0]
    bins = list(range(max_expected)) + [np.inf]
    by = 1000
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
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
<<<<<<< HEAD
    """Function to calculate coverage on individual chunk level"""
=======
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
    masked_cov = h5[f'evaluation/{COV_STR}'][start_i:end_i][
        np.logical_or(h5['data/y'][start_i:end_i, :, 1], h5['data/y'][start_i:end_i, :, 2])
    ]
    histo = np.histogram(masked_cov, bins=bins)
    return histo


# most of the functions and ways things are done are copied from Helixer/helixer/evaluation/score_rnaseq.py script
def sum_last_and_flatten(x):
<<<<<<< HEAD
    """Function to sum array along last axis and then flatten the summed array"""
    return np.sum(x, axis=-1).ravel()


def main(species: str, h5_data: h5py._hl.files.File):
=======
    return np.sum(x, axis=-1).ravel()


def main(species, h5_data):
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
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
        print("adding empty score dataset ...")
        add_empty_score_datasets(h5)

        # add remaining grp not currently part of rnaseq as it's unneeded there
    try:
        h5[f'{META_STR}/median_expected_coverage']
    except KeyError:
        print("creating median expected coverage group ...")
        h5[META_STR].create_group('median_expected_coverage')

    try:
        h5[f'{META_STR}/max_normalized_cov_sc']
    except KeyError:
        print("creating normalized coverage/spliced coverage group ...")
        h5[META_STR].create_group('max_normalized_cov_sc')
    print(" Getting median coverage ...")
    median_coverage = get_median_expected_coverage(h5)
    average_positive_coverage = get_average_cov(h5)
    h5[f'{META_STR}/median_expected_coverage'].attrs.create(name=species, data=median_coverage)
    mec = int(h5[f'{META_STR}/median_expected_coverage'].attrs[species])
<<<<<<< HEAD

    # can I remove normscorer etc. ?
    norm_cov_scorer = NormScoreCoverage(column=None, median_cov=mec)
    norm_sc_scorer = NormScoreSplicedCoverage(column=None, median_cov=mec)
    norm_scorers = [norm_cov_scorer, norm_sc_scorer]  # calculate normalized coverage more than "scoring" perse
=======
    ig_scorer = ScorerIntergenic(column=0, median_cov=mec)
    intron_scorer = ScorerIntron(column=3, median_cov=mec)
    # todo: find way to implement threshold for scoreres
    scorers = [ig_scorer, intron_scorer]
    norm_cov_scorer = NormScoreCoverage(column=None, median_cov=mec)
    norm_sc_scorer = NormScoreSplicedCoverage(column=None, median_cov=mec)
    norm_scorers = [norm_cov_scorer, norm_sc_scorer]  # calculate normalized coverage more than "scoring" persay
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
    max_norm_cov = 0
    max_norm_sc = 0

    i = 0
    by = stepsize
<<<<<<< HEAD
    by_out = 0
    start = 0
    end = 5000# h5['data/X'].shape[0]
    counts = np.zeros(shape=(end - start, 4))
    # loop to iterate over BY chunks at once and score them
    print(f'starting to score {start}:{end} in {species}; threshold: {threshold}; max_distance: {max_distance}')
=======
    start = 0
    end = h5['data/X'].shape[0]
    counts = np.zeros(shape=(end - start, 4))
    # loop to iterate over BY chunks at once and score them
    print(f'starting to score {start}:{end} in {species}')
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
    # for loop to iterate over input h5 file BY chunks at a time
    for i in range(start, end, by):
        if i + by > end:
            by_out = end - i
        else:
            by_out = by
        i_rel = i - start
        y = h5['data/y'][i:(i + by_out)]
        utr = get_utr_positions(h5, start=i, end=i + by, stepsize=by_out)
<<<<<<< HEAD
        utr_trees = to_trees(utr, utr=True)
        peaks = get_cage_peak(h5, threshold=threshold, start=i, end=i + by, stepsize=by_out)
        peaks_trees = to_trees(peaks)
        # utr_trees, peak_trees = get_positions(h5, start=i, end=i + by, stepsize=by_out, threshold=threshold)
=======
        peaks = get_cage_peak(h5, threshold=threshold, start=i, end=i + by, stepsize=by_out)
        peaks_trees = to_trees(peaks)
        dist = get_distance(peaks_trees, utr, max_distance)
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
        datay = y.reshape([-1, 4])
        _, chunk_size, n_cats = y.shape
        start_rel = (i * chunk_size)  # relative start of each by chunk_of_chunks in order to correctly add the score
        # different coverages not used as much as in the RNAseq script
        coverage = h5[f'evaluation/{COV_STR}'][i:(i + by_out)]
        spliced_coverage = h5[f'evaluation/{SC_STR}'][i:(i + by_out)]
<<<<<<< HEAD
        number_of_files = coverage.shape[-1]
        # coverage and spliced_coverage will initially have dimensions [n_chunks, chunk_size, n_bams]
        # but scoring assumes shape [n_basepairs], so they must be summed and flattened
        coverage, spliced_coverage = sum_last_and_flatten(coverage), sum_last_and_flatten(spliced_coverage)
        by_bp = np.full(fill_value=1., shape=[by_out * chunk_size])
        norm_cov_by_bp = np.full(fill_value=1., shape=[by_out * chunk_size, 2])  # 2 for [cov, sc]
        scores, poss = get_distance(peaks_trees, utr_trees, average_positive_coverage)
        # actual scoring happens for BY chunk of chunks
        # loop to add positions and scoring to the by_bp class in the h5file
        for index_asif, norm_scorer in enumerate(norm_scorers):
            raw_score, mask = norm_scorer.score(datay=datay, coverage=coverage,
                                                spliced_coverage=spliced_coverage,
                                                average_positive_coverage=average_positive_coverage)
            if raw_score.size > 0:
                norm_cov_by_bp[mask, index_asif] = raw_score
        # adding of scores calculated by get_score() at positions pos
        if len(poss) > 0:
            for e, x in enumerate(poss):
                poss_start = x[0]
                poss_end = x[1]
                by_bp[poss_start - start_rel:poss_end - start_rel] = scores[e]  # adding to flat array size of by chunks
        del coverage, spliced_coverage
        by_bp = by_bp.reshape([by_out, chunk_size])  # reshaped so it fits the chunk structure again

        # normcov and other groups, not as important for scoring
=======
        # coverage and spliced_coverage will initially have dimensions [n_chunks, chunk_size, n_bams]
        # but scoring assumes shape [n_basepairs], so they must be summed and flattened
        coverage, spliced_coverage = sum_last_and_flatten(coverage), sum_last_and_flatten(spliced_coverage)

        by_bp = np.full(fill_value=1., shape=[by_out * chunk_size])
        norm_cov_by_bp = np.full(fill_value=1., shape=[by_out * chunk_size, 2])  # 2 for [cov, sc]
        score, pos = get_score(dist, max_distance, average_positive_coverage )  # actual scoring happens for BY chunk of chunks
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
        # adding of scores calculated by get_score() at positions pos
        if len(pos) > 0:
            for e, x in enumerate(pos):
                pos_start = x[0]
                pos_end = x[1]
                by_bp[pos_start - start_rel:pos_end - start_rel] = score[e]  # adding to flat array size of by chunks
        del coverage, spliced_coverage
        by_bp = by_bp.reshape([by_out, chunk_size])  # reshaped so it fits the chunk structure again
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
        norm_cov_by_bp = norm_cov_by_bp.reshape([by_out, chunk_size, 2])  # don't know what happens with norm_cov
        h5[f'{SCORE_STR}/by_bp'][i:(i + by_out)] = by_bp
        h5[f'{SCORE_STR}/norm_cov_by_bp'][i:(i + by_out)] = norm_cov_by_bp
        max_norm_cov = max(max_norm_cov, np.max(norm_cov_by_bp[:, 0]))
        max_norm_sc = max(max_norm_sc, np.max(norm_cov_by_bp[:, 1]))

<<<<<<< HEAD
        # additional scoring again I don't know what exactly these scores do, copied from score_rnaseq.py
=======
        # additional scoring again idk what exactly these scores do, copied from score_rnaseq.py
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
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

<<<<<<< HEAD

=======
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--h5-data', help='h5 data file with /data/{X, y, species, seqids, etc...} '
                                                'AND with /evaluation/{{prefix}_coverage, {prefix}_spliced_coverage} '
                                                'to which {prefix}_scores will be added',
                        required=True)
<<<<<<< HEAD
    parser.add_argument('-s', '--species',
                        help='species name matching that used in creation of geenuff/h5',
                        required=True)
    parser.add_argument('--dataset-prefix',
                        help='prefix of h5 datasets to be used for scoring (default "cage")',
                        default='cage')
    parser.add_argument('--threshold', dest='threshold',
                        help='minimal coverage for cage peaks to get detected (default = 3)',
                        default=3)
    parser.add_argument('--max-distance', dest='distance',
                        help='maximal distance between CAGE-peak and annotated UTR (default = 1000)',
                        default=500)
    parser.add_argument('--stepsize', dest='stepsize',
                        help='number of chunks that is read in at once (default = 500)',
                        default=500)

=======
    parser.add_argument('-s', '--species', help='species name matching that used in creation of geenuff/h5',
                        required=True)
    parser.add_argument('--dataset-prefix', help='prefix of h5 datasets to be used for scoring (default "cage")',
                        default='cage')
    parser.add_argument('--threshold', dest='threshold', help='minimal coverage for cage peaks to get detected (default = 3)',
                        default=3)
    parser.add_argument('--max-distance', dest='distance', help='maximal distance between CAGE-peak and annotated UTR (default = 500)',
                        default=500)
    parser.add_argument('--stepsize', dest='stepsize', help='number of chunks that is read in at once (default = 500)', 
                        default=500)
>>>>>>> 1130c9a18138a99c21499b7b08d1670bbd87b140
    args = parser.parse_args()
    COV_STR = f'{args.dataset_prefix}_coverage'
    SC_STR = f'{args.dataset_prefix}_spliced_coverage'
    META_STR = f'{args.dataset_prefix}_meta'
    SCORE_STR = f'{args.dataset_prefix}_scores'
    threshold = int(args.threshold)
    max_distance = int(args.distance)
    stepsize = int(args.stepsize)
    main(args.species, args.h5_data)
