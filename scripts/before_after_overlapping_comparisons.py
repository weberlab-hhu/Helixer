#! /usr/bin/env python3
"""Calculates the length wise error comparison for two models with/without overlapping.
Also generates an aggregate plot.
Requires folders filled with {species}/length_wise_eval.log files from
helixer_scratch/cluster_eval_predictions/evaluations/length_wise/length-wise-eval-from-cluster-jobs.sh"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_comparison(f1_before, f1_after, acc_before, acc_after, title, picture_name, folder,
                    error_bars=None):
    plt.cla()
    plt.title(title)
    if error_bars:
        plt.errorbar(range(100), f1_before, yerr=errors['f1_before'], color='chocolate', fmt='o',
                 label='regular genic f1')
        plt.errorbar(range(100), f1_after, yerr=errors['f1_after'], color='chocolate', fmt='^',
                     label='genic f1 with overlapping')
        plt.errorbar(range(100), acc_before, yerr=errors['acc_before'], color='royalblue', fmt='o',
                     label='regular accuracy')
        plt.errorbar(range(100), acc_after, yerr=errors['acc_after'], color='royalblue', fmt='^',
                     label='accuracy with overlapping')
    else:
        plt.plot(range(100), f1_before, color='chocolate', linestyle='dashed',
                 label='regular genic f1')
        plt.plot(range(100), f1_after, color='chocolate', label='genic f1 with overlapping')
        plt.plot(range(100), acc_before, color='royalblue', linestyle='dashed',
                 label='regular accuracy')
        plt.plot(range(100), acc_after, color='royalblue', label='accuracy with overlapping')
    plt.ylim((0.0, 1.0))
    plt.xlabel('chunk of length')
    plt.legend()
    file_path = os.path.join(folder, picture_name)
    plt.savefig(file_path)
    print(file_path, 'saved')


parser = argparse.ArgumentParser()
parser.add_argument('-before', '--before-main-folder', type=str, required=True)
parser.add_argument('-after', '--after-main-folder', type=str, required=True)
parser.add_argument('-o', '--output-folder', type=str, required=True)
parser.add_argument('-eb', '--error-bars', action='store_true')
args = parser.parse_args()

genic_f1s = {'before': [], 'after': []}
accuracies = {'before': [], 'after': []}

for species in os.listdir(args.before_main_folder):
    if not os.path.isdir(os.path.join(args.before_main_folder, species)):
        continue
    log_files = {
        'before': os.path.join(args.before_main_folder, species, 'length_wise_eval.log'),
        'after': os.path.join(args.after_main_folder, species, 'length_wise_eval.log'),
    }
    for log_file_path in log_files.values():
        if not os.path.exists(log_file_path) or not os.path.getsize(log_file_path) > 0:
            print(f'Log file {log_file_path} is empty or not existing. Exiting.')
            # exit()
            continue

    for type_, log_file_path in log_files.items():
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
                    genic_f1s[type_].append(species_genic_f1s)
                    accuracies[type_].append(species_accuracies)
                    break
    plot_comparison(genic_f1s['before'][-1],
                    genic_f1s['after'][-1],
                    accuracies['before'][-1],
                    accuracies['after'][-1],
                    f'Performance by sequence position of {species}',
                    f'{species}_comparison',
                    args.output_folder)

# make aggregate plot
f1s_avg, accs_avg = {}, {}
error_bars = {}
for type_ in genic_f1s.values():
    f1s_avg[type_] = np.mean(np.array(genic_f1s[type_]), axis=0)
    accs_avg[type_] = np.mean(np.array(accuracies[type_]), axis=0)
    if args.error_bars:
        error_bars['f1_' + type_] = np.std(np.array(genic_f1s[type_]), axis=0)
        error_bars['accs' + type_] = np.std(np.array(accuracies[type_]), axis=0)
    else:
        error_bars = None

plot_comparison(f1s_avg['before'],
                f1s_avg['after'],
                accs_avg['before'],
                accs_avg['after'],
                f'Performance by sequence position of {species}',
                f'{species}_comparison',
                args.output_folder,
                error_bars=error_bars)
