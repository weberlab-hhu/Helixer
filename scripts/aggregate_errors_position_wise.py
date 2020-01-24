#! /usr/bin/env python3
"""Calculates the aggregate of many length wise evaluations by averaging the genic f1 of each
chunk of length (that can vary in length but not in overall count of chunks; 100 chunks are assumed).
Requires a folder filled with {species}/length_wise_eval.log files."""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('main_folder', type=str)
args = parser.parse_args()

genic_f1s, accuracies = [], []

for species in os.listdir(args.main_folder):
    if not os.path.isdir(os.path.join(args.main_folder, species)):
        continue
    log_file_path = os.path.join(args.main_folder, species, 'length_wise_eval.log')
    if not os.path.exists(log_file_path) or not os.path.getsize(log_file_path) > 0:
        print(f'Log file {log_file_path} is empty or not existing. Exiting.')
        # exit()
        continue

    # parse metric table
    species_genic_f1s, species_accuracies = [], []
    f = open(log_file_path)
    while True:
        line = next(f)
        if line.startswith('| genic'):
            species_genic_f1s.append(float(line.strip().split('|')[4].strip()))
            # parse total accuracy
            next(f)
            species_accuracies.append(float(next(f).strip().split(' ')[-1]))
            if len(species_genic_f1s) == 100:
                genic_f1s.append(species_genic_f1s)
                accuracies.append(species_accuracies)
                break

genic_f1s = np.array(genic_f1s)
accuracies = np.array(accuracies)
f1s_avg = np.mean(genic_f1s, axis=0)
accs_avg = np.mean(accuracies, axis=0)

# output overall plot
plt.cla()
plt.title('Aggregate length wise error')
plt.plot(range(100), accs_avg, label='overall acc')
plt.plot(range(100), f1s_avg, label='genic f1')
plt.ylim((0.0, 1.0))
plt.xlabel('chunk of length')
plt.legend()
picture_name = 'aggregate_length_wise.png'
plt.savefig(picture_name)
print(picture_name, 'saved')
