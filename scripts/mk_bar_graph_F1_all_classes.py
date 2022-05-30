import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
import pandas as pd
import csv
import seaborn as sns
import os
import argparse


def get_F1(csv_file):
    # open csv and convert to pandas dataframe
    normalized_cm = pd.read_csv(csv_file)
    # convert to numpy array
    m_array = normalized_cm.to_numpy()
    # remove labels from array
    m_array = m_array[:, 1:]
    # convert to float
    m_array = m_array.astype('float')
    F1_UTR = m_array[:4, -1]
    return F1_UTR


def plot_(data, title):
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x="species", y="score", hue="class", data=data, ci=None)
    ax.set_title(title)
    for x in range(len(ax.containers)):
        ax.bar_label(ax.containers[x])


def mk_csv(rows, new_file):
    header = ['species', 'class', 'score']
    if os.path.isfile("./F1_scores_by_class_comparison.csv") and new_file:
        os.remove("./F1_scores_by_class_comparison.csv")
    with open("F1_scores_by_class_comparison.csv", "a+", newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerows(rows)


def main(data_dir, experiment, species, title, output):
    species = species
    new_file = True
    if title is None:
        title = f'Comparison of F1 scores by class for {experiment}'
    for sp in species:
        i = 0
        rows = []
        file = f'{data_dir}/{sp}/genic_CM_F1_summary.csv'
        with open(file, 'r') as f:
            for line in f:
                if i == 0 or i >= 5:
                    pass
                else:
                    split = line.split(',')
                    row = [sp, split[0], float(split[-1].rstrip())]
                    rows.append(row)
                i += 1
        mk_csv(rows, new_file)
        new_file = False
    data = pd.read_csv('./F1_scores_by_class_comparison.csv')
    plot_(data, title)
    plt.savefig(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir',
                            help='path to csv file of corresponding experiment',
                            required=True)
    parser.add_argument('-e', '--experiment',
                            help='name of the experiment, will be relevant for the naming scheme of the resulting figs',
                            required=True)
    parser.add_argument('-o', '--output',
                            help='path where the figures will be saved',
                            default= './F1_scores_by_class.png')
    parser.add_argument('-s', '--species',
                            help='list of species to create confusion matrices for',
                            nargs='+',
                            default=['gallus_gallus', 'homo_sapiens', 'mus_musculus', 'macaca_mulatta', 'canis_lupus_familiaris','rattus_norvegicus'])
    parser.add_argument('-t', '--title',
                            help='title for the figure',
                            default=None)
    args = parser.parse_args()
    main(args.data_dir, args.experiment, args.species, args.title, args.output)
