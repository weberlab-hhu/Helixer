#! /usr/bin/env python3
"""Calculates the aggregate of many length wise evaluations by averaging the genic f1 of each
chunk of length (that can vary in length but not in overall count of chunks; 100 chunks are assumed).
Requires a folder filled with {species}/length_wise_eval.log files."""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('main-folder', type=str)
args = parser.parse_args()

for species in os.listdir(args.main_folder):
    if not os.path.isdir(os.path.join(species, args.main_folder)):
        continue
    log_file_path = os.path.join(args.main_folder, species, 'length_wise_eval.log')
    if not os.path.exists(log_file_path) or not os.path.getsize(log_file_path) > 0:
        print(f'Log file {log_file_path} is empty or not existing. Exiting.')
        exit()





