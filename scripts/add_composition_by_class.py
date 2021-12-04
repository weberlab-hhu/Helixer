"""adds base composition and relative abundance of each class in the 6 class ecoding to each chunk in h5 file"""

import argparse
import h5py
import numpy as np

def main(h5_data):
    h5 = h5py.File(h5_data, mode='r+')
    by = 1000
    n_chunks, chunk_size = h5['data/y'].shape[0:2]
    composition = np.full(fill_value=0, dtype=np.float16, shape=(n_chunks, 6 * 5))
    # 6 classes (IG, UTR, Ntrn, cds0, cds1, cds2) x (abudance, %C, %A, %T, %G)
    for i in range(0, n_chunks, by):
        pre_x = h5['data/X'][i:i+by].astype(float)
        pre_y = h5['data/y'][i:i+by]
        pre_phase = h5['data/phases'][i:i+by]
        for j in range(pre_x.shape[0]):
            x = pre_x[j]
            y = pre_y[j]
            phase = pre_phase[j]
            six_class = np.concatenate((y[:, (0,1,3)], phase[:, 1:]), axis=1)
            for c in range(6):
                subx = x[six_class[:, c].astype(bool)]
                class_start = c * 5
                class_count = subx.shape[0]
                composition[i + j][class_start] = class_count / x.shape[0]
                if class_count:
                    bp_in_class = np.sum(subx, axis=0) / class_count
                else:
                    bp_in_class = [0.25] * 4
                composition[i + j][class_start + 1: class_start + 5] = bp_in_class
    if 'composition' in h5['data'].keys():
        del h5['data/composition']
    h5.create_dataset('data/composition', data=composition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_data', help='h5 file that will be updated to include base composition')
    args = parser.parse_args()
    main(args.h5_data)