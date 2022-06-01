import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
import pandas as pd
import math
from datetime import date
import argparse


def convert_file_to_array(file):
    '''
    Function to convert csv file to a numpy array wthout labels
    '''

    # open csv and convert to pandas dataframe
    normalized_cm = pd.read_csv(file)
    # convert to numpy array
    m_array = normalized_cm.to_numpy()
    # remove labels from array
    m_array = m_array[:,1:]
    # convert to float
    m_array = m_array.astype('float')
    return m_array


def mk_delta_cm(file1, file2, sp, experiments, out_dir, with_nums=False, title=None):
    m1_array = convert_file_to_array(file1)
    m2_array = convert_file_to_array(file2)
    creation_date = date.today()
    if title == None:
        title = f'{sp} {experiments[0]} vs {experiments[1]}'
    else:
        title = title
    if not with_nums:
        x = sp + "_no_nums.png"

    if with_nums:
        x = sp + "_with_nums.png"
        # idk
    creation_date = creation_date.strftime("%d_%m_%Y")
    save_path = f'{out_dir}/{sp}/{sp}_{creation_date}_delta_cm' # f'./{experiment}/{sp}/{sp}_{title}_{creation_date}' # f'/mnt/data/kevin/FANTOM_Data/test_data_FANTOM5/visual/cms/fig/{experiment}/{sp}/{sp}_{title}_{creation_date}'
        # list of labels but so far useless
    delta_array = m1_array - m2_array
    labels = ['IG', 'UTR', 'EXON', 'INTRON']
    fig, (ax) = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(delta_array, cmap='coolwarm')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('pred')
    ax.set_ylabel('ref')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not with_nums:
                pass
            else:
                text = ax.text(i, j, round(delta_array[j,i],4), ha='center', va='center', color='w')
    ax.set_title(title)
    fig.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    fig.savefig(save_path, bbox_inches='tight')


def main(data_dir1, data_dir2, experiments, out_dir, species):
    species = species
    for sp in species:
        path1 = f'{data_dir1}/{sp}/genic_CM_normalized_confusion_matrix.csv'
        path2 = f'{data_dir2}/{sp}/genic_CM_normalized_confusion_matrix.csv'
        mk_delta_cm(path1, path2, sp, experiments, out_dir, with_nums=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', '--data-dir1',
                            help='path to csv file containing F1 score information for the first data',
                            required=True)
    parser.add_argument('-d2', '--data-dir2',
                        help='path to csv file containing F1 score information for the second data',
                        required=True)
    parser.add_argument('-e', '--experiments',
                            help='name of the experiments, that are being compared',
                            nargs='+',
                            required=True)
    parser.add_argument('-o', '--out-dir',
                            help='path where the figures will be saved',
                            default='.')
    parser.add_argument('-s', '--species',
                            help='list of species to create confusion matrices for',
                            nargs='+',
                            default=['gallus_gallus', 'homo_sapiens', 'mus_musculus', 'macaca_mulatta', 'canis_lupus_familiaris','rattus_norvegicus'])
    args = parser.parse_args()
    main(args.data_dir1, args.data_dir2, args.experiments, args.out_dir, args.species)