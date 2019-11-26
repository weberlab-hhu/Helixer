#! /usr/bin/env python3
import os
import h5py
import sqlite3
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from intervaltree import IntervalTree
from terminaltables import AsciiTable
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True,
                    help='This HAS to be from only one genome')
parser.add_argument('-p', '--predictions', type=str, required=True,
                    help='This HAS to be from only one genome')
parser.add_argument('-db', '--db-path', type=str, required=True)
parser.add_argument('-g', '--genome', type=str, required=True)
parser.add_argument('-o', '--output-file', type=str, default='alternative_splicing_results')
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

y_true = h5_data['/data/y']
y_pred = h5_pred['/predictions']
sw = h5_data['/data/sample_weights']
seqids = np.array(h5_data['/data/seqids'])
start_ends = np.array(h5_data['/data/start_ends'])

# first fill a dict of intervaltrees with the start_end values of the data
intervals = defaultdict(lambda: {'plus': IntervalTree(), 'minus': IntervalTree()})

for i in range(len(seqids)):
    seqid = seqids[i].decode('utf8')
    interval = start_ends[i]
    if interval[0] <= interval[1]:
        # plus strand
        intervals[seqid]['plus'][interval[0]:interval[1]] = i
    else:
        # minus strand
        intervals[seqid]['minus'][interval[1] + 1:interval[0] + 1] = i

gene_borders = dict()
with sqlite3.connect(args.db_path) as con:
    cur = con.cursor()
    # query all gene intervals with their relevant information
    # each strand is queried seperately
    query_base = '''FROM genome
        CROSS JOIN coordinate ON coordinate.genome_id = genome.id
        CROSS JOIN feature ON feature.coordinate_id = coordinate.id
        CROSS JOIN association_transcript_piece_to_feature
            ON association_transcript_piece_to_feature.feature_id = feature.id
        CROSS JOIN transcript_piece
            ON association_transcript_piece_to_feature.transcript_piece_id = transcript_piece.id
        CROSS JOIN transcript ON transcript_piece.transcript_id = transcript.id
        CROSS JOIN super_locus ON transcript.super_locus_id = super_locus.id
        WHERE genome.species IN ("''' + args.genome + '''") AND super_locus.type = 'gene'
            AND transcript.type = 'mRNA' AND feature.type = 'geenuff_transcript'
            AND feature.is_plus_strand = 1
        GROUP BY super_locus.id;'''

    query_plus = ('SELECT coordinate.seqid, min(feature.start), max(feature.end), '
                  'count(distinct(transcript.id)) ' + query_base)
    query_minus = ('SELECT coordinate.seqid, min(feature.end) + 1, max(feature.start) + 1, '
                   'count(distinct(transcript.id)) ' + query_base)

    cur.execute(query_plus)
    gene_borders['plus'] = cur.fetchall()
    cur.execute(query_minus)
    gene_borders['minus'] = cur.fetchall()

with open(f'{args.output_file}.csv', 'w') as f:
    for strand in ['plus', 'minus']:
        for (seqid, start, end, n_transcripts) in gene_borders[strand]:
            # get the indexes of the sequences in the dataset that overlap with this gene
            seq_idxs = [i.data for i in list(intervals[seqid][strand][start:end])]
            seq_idxs.sort() # indexes have to be sorted for h5 query
            if seq_idxs:
                # concat all the relevant data with it and cut to limits of the gene
                y_true_section = np.concatenate(y_true[seq_idxs])[start:end]
                y_pred_section = np.concatenate(y_pred[seq_idxs])[start:end]
                sw_section = np.concatenate(sw[seq_idxs])[start:end]
                if np.any(sw_section):
                    # run through cm to get the genic f1
                    cm = ConfusionMatrix(None)
                    cm._add_to_cm(y_true_section, y_pred_section, sw_section)
                    genic_f1 = cm._print_results()
                    # writout results
                    print(','.join([seqid, str(n_transcripts), str(genic_f1)]), file=f)
