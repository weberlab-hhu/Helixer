import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings
import intervaltree as ivt
import statistics
import argparse


# identifies start of 5'UTRs according to reference
def get_utr_positions(file, start=0, end="max", stepsize=1000):
    whole_utrs = []
    if end == "max":
        end = file['data/y'].shape[0]
    else:
        end = end
    # loop to iterate over the h5 file , reading in stepsize chunks at once
    for i in range(start, end, stepsize):
        chunk = file['data/y'][i:(i + stepsize)].T
        seqid = file['data/seqids'][i:(i + stepsize)].T
        start_ends = file['data/start_ends'][i:(i + stepsize)].T
        # loop to go over each individual loop of the stepsize chunks loaded at once
        for x in range(0, chunk.shape[2]):
            individual_chunk = chunk[:, :, x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:, x]
            start = individual_start_ends[0]
            end = individual_start_ends[1]
            strand = end - start
            # the strand is determined
            if strand < 0:
                strand = -1
            else:
                strand = 1
            # dimensions of the array are reduced, to only catch shifts from IG to UTR and only capture 5' UTRs
            individual_chunk = individual_chunk[0:1, :]
            # whole array is shifted by 1 to the right
            shifted_array = individual_chunk[0:1, :-1]
            individual_chunk = individual_chunk[0:1, 1:]
            # difference between the array will have a -1 at each transition from IG to UTR
            utr_positions = np.subtract(individual_chunk, shifted_array)
            utr_positions = np.where(utr_positions == -1)
            # the start position is added to preserve the coordinates over multiple chunks
            utr_positions = utr_positions[1] + start
            if len(utr_positions) == 0:
                continue
            utr_positions = utr_positions[0]
            whole_utrs.append((utr_positions, individual_seqid, strand, x + i))
    return whole_utrs


# method to calculate the distance between CAGE peaks and UTRs, will select the shortest distance
def get_distance(whole_utrs, trees, max_distance=100000):
    minimal_distances = []
    # if a chunk had no UTRs or no peaks, an empty list is returned
    if len(whole_utrs) == 0 or len(trees) == 0:
        print("No UTRs or peaks")
        return minimal_distances
    # loop to iterate over all UTRs
    for x in whole_utrs:
        distances = []
        key = "{},{}".format(x[1], x[2])  # key based on the UTRs seqid and strand is created
        # if the key exists the intervaltree containing peaks matching the seqid and strand is selected
        try:
            peaks = trees[key]
        # if there is a KeyError the iteration of the loop is skipped, there must be no peaks for that seqid
        # and/or strand
        except KeyError:
            continue
        # checking for all peaks that overlap with an interval of +/-  max_distance (100kbp) from the UTRs beginning
        overlap = peaks.overlap(x[0] - max_distance, x[0] + max_distance)  # todo: find way to determine this range
        if len(overlap) == 0:
            continue
        # nested for loop iterating over all peaks overlaping with the UTR +/- 100kpb interval
        for y in overlap:
            # distance between UTR and center of cage peak is calculated, absolute value is kept
            distances.append(abs(x[0] - y.begin))
        # minimal distance for that UTR is determined and added to a list
        minimal_distances.append(min(distances))
    # return of list of minimal distance for each UTR from whole UTRs, if there was a peak for that UTR
    return minimal_distances


# identifies positions of cage peaks on the chunk
def get_cage_peak(file, threshold=2, start=0, end="max", stepsize=1000):

    # can this be done more efficient?
    centered = []
    if end == "max":
        end = file['data/y'].shape[0]
    else:
        end = end
    # loop to iterate over the h5 file , reading in stepsize chunks at once
    for i in range(start, end, stepsize):
        chunk = file['evaluation/cage_coverage'][i:(i + stepsize)].T
        seqid = file['data/seqids'][i:(i + stepsize)].T
        start_ends = file['data/start_ends'][i:(i + stepsize)].T
        # loop to go over each individual loop of the stepsize chunks loaded at once
        for x in range(0, chunk.shape[2]):
            individual_chunk = chunk[:, :, x]
            individual_seqid = seqid[x]
            individual_start_ends = start_ends[:, x]
            start = individual_start_ends[0]
            end = individual_start_ends[1]
            strand = end - start
            # the strand is determined
            if strand < 0:
                strand = -1
            else:
                strand = 1
            chunk_shape = np.shape(individual_chunk)  # shape of the input array
            threshold_array = np.full((chunk_shape),
                                      threshold)  # creates an array filled with the threshold value of the same shape as input
            # todo: find ways to determine that threshold
            peak_positions = np.subtract(individual_chunk, threshold_array)
            peak_positions = np.where(
                peak_positions > 0)  # only values greater than the threshold will result in positive value
            peak_positions = peak_positions[1] + start  # reducing the dimension of the array
            peak_indices = np.where(np.diff(peak_positions) != 1)  # selecting only indices of values apart more than 1
            if len(peak_positions) == 0:
                continue
            peak_indices = np.append(peak_indices,
                                     len(peak_positions) - 1)  # does not catch the last index therefore its added here
            ends = np.take(peak_positions, peak_indices)  # end positions of the peaks
            start_indices = peak_indices + 1  # same procedure for the start values
            start_indices = np.insert(start_indices, 0, 0)  # here the first one needs to be added
            start_indices = np.delete(start_indices, -1)
            starts = np.take(peak_positions, start_indices)
            # print(start_coordinates)
            # unfortunately no solution without a loop, in order to combine start and end values in tuples
            try:
                for y in range(len(starts)):
                    centered.append((round((starts[y] + ends[y]) / 2), individual_seqid, strand))
            except UnboundLocalError:
                print("No peaks")
                return centered
    return centered


# method to create a dictionary filled with peaks separated by strand and  seqid
def to_trees(centered_peaks):
    trees = {}
    # for loop through all peaks
    for x in centered_peaks:
        # if a combination of seqid and strand has not been encountered yet its added to the dict accompanied
        # by an intervalltree representing the center of a cage peak
        if "{},{}".format(x[1], x[2]) not in trees.keys():
            peaks = ivt.IntervalTree()
            trees.setdefault(("{},{}".format(x[1], x[2])), peaks)
            peaks[x[0]:x[0] + 1] = x[0]
        # if the combination already exists the corresponding peak is added to the intervalltree at that position
        else:
            peaks[x[0]:x[0] + 1] = x[0]
            trees["{},{}".format(x[1], x[2])] = peaks

        # peaks = trees[(seqid, strand)]
        # adding an intervall at the position of the peak to the intervaltree
    # returns the dictionary full of intervaltrees
    return trees

def main(species, h5):
    # open h5
    h5 = h5py.File(h5, mode='r')
    print("Starting to calculate annotated UTR distance to CAGE peaks for " + species)
    print("Getting UTR-positions ...")
    positions = get_utr_positions(h5)
    print("Getting CAGE peaks ...")
    peaks = get_cage_peak(h5, threshold)
    peaks = to_trees(peaks)
    print("Calculating distance ...")
    distance = get_distance(positions,peaks)
    print(species)
    print(statistics.median(distance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--h5-data', help='h5 file for which median distance between annotated UTRs and CAGE-peaks'
                                                'is calculated',
                        required=True)
    parser.add_argument('-s', '--species', help='species name matching that used in creation of geenuff/h5',
                        required=True)
    parser.add_argument('--threshold', dest='threshold',
                        help='minimal coverage for cage peaks to get detected (default = 2)',
                        default=2)
    parser.add_argument('--max-distance', dest='distance',
                        help='maximal distance between CAGE-peak and annotated UTR (default = 100000)',
                        default=100000)
    args = parser.parse_args()
    threshold = int(args.threshold)
    max_distance = int(args.distance)
    main(args.species, args.h5_data)