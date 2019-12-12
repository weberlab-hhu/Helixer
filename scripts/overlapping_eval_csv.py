#! /usr/bin/env python3
"""Requires a folder filed with eval_{genome} files. Prints a csv file with the same
columns as make_csv_from_single_genome_eval.py would output."""

import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, required=True)
parser.add_argument('-i', '--ignore', action='append')
args = parser.parse_args()

header = ['genome', 'acc_overall', 'f1_ig', 'f1_utr', 'f1_exon', 'f1_intron', 'legacy_f1_cds',
          'f1_genic']
print(','.join(header))

for eval_file in glob.glob(os.path.join(args.folder, 'eval_*')):
    # get genome name
    genome = os.path.basename(eval_file).split('_')[1]

    if args.ignore and genome in args.ignore:
        continue

    # parse metric table
    log_file = open(eval_file)
    f1_scores = []
    for line in log_file:
        if 'Precision' in line:  # table start
            next(log_file)  # skip line
            for i in range(6):
                line = next(log_file)
                f1_scores.append(line.strip().split('|')[4].strip())
                if i == 3:
                    next(log_file)  # skip line
            break  # stop at the last line of the metric table

    # parse total accuracy
    next(log_file)
    line = next(log_file)
    acc_overall = line.strip().split(' ')[-1]

    # merge everything into one string
    str_rows = [genome, acc_overall] + f1_scores
    print(','.join(str_rows))
