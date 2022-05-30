import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
import pandas as pd
import seaborn as sns
import csv
import os
import argparse

def plot_(data, title):
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x="species", y="score", hue="method", data=data, ci=None)
    ax.set_title(title)
    for x in range(len(ax.containers)):
        ax.bar_label(ax.containers[x])


def get_UTR_F1(file):
    # open csv and convert to pandas dataframe
    normalized_cm = pd.read_csv(file)
    # convert to numpy array
    m_array = normalized_cm.to_numpy()
    # remove labels from array
    m_array = m_array[:, 1:]
    # convert to float
    m_array = m_array.astype('float')
    F1_UTR = m_array[1, -1]
    if F1_UTR == None:
        return 0
    return F1_UTR


def mk_csv(rows, new_file):
    header = ['species', 'score', 'method']
    if os.path.isfile("./F1_scores_by_method_comparison.csv") and new_file:
        os.remove("./F1_scores_by_method_comparison.csv")
    with open("F1_scores_by_method_comparison.csv", "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def main(methods, species, title, output):
    rows = []
    new_file = True
    # output_path determines where this figure will be saved
    if title is None:
        title = 'Comparison of UTR F1 score by training method'
    for sp in species:
        for method in methods:
            file = f'/mnt/data/kevin/FANTOM_Data/test_data_FANTOM5/visual/cms/cvs/{method}/{sp}/genic_CM_F1_summary.csv'
            F1_UTR = get_UTR_F1(file)
            row = [sp, F1_UTR, method]
            rows.append(row)

    mk_csv(rows, new_file)
    data = pd.read_csv('F1_scores_by_method_comparison.csv')
    plot_(data, title)
    plt.savefig(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--methods',
                            help='methods to be compared (directories with the same names as the methods have to exist)',
                            nargs='+',
                            required=True)
    parser.add_argument('-o', '--output',
                            help='path where the figures will be saved',
                            default= f'/mnt/data/kevin/FANTOM_Data/test_data_FANTOM5/visual/cms/fig/F1_scores_UTR_by_method.png')
    parser.add_argument('-s', '--species',
                            help='list of species to create confusion matrices for',
                            nargs='+',
                            default=['gallus_gallus', 'homo_sapiens', 'mus_musculus', 'macaca_mulatta','canis_lupus_familiaris','rattus_norvegicus'])
    parser.add_argument('-t', '--title',
                            help='title for the figure',
                            default=None)
    args = parser.parse_args()
    main(args.methods, args.species, args.title, args.output)