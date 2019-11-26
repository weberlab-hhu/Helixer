#! /usr/bin/env python3
import os
import h5py
import sqlite3
import numpy as np
import argparse
import intervaltree
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
parser.add_argument('-db', '--db-path', type=str, required=True)
parser.add_argument('-g', '--genome', type=str, required=True)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

with sqlite3.connect(args.db_path) as con:
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

    print(query_plus, query_minus)



