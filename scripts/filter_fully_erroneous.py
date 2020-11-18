"""makes a copy of .h5 file with all fully erroneous and unannotated sequences removed"""
# this is for compatibility with previous runs and also just might
# increase total speed, and data reading can bottleneck training

import argparse
import h5py
import numpy as np


def main(data, out, write_by):
    old = h5py.File(data, mode='r')
    new = h5py.File(out, mode='a')
    end = old['data/X'].shape[0]

    # setup all datasets that will need to be masked / filtered
    filter_keys = set(old.keys()).intersection({'data', 'evaluation', 'scores'})
    filter_datasets = []
    if 'predictions' in old.keys():
        filter_datasets.append('predictions')
    for key in filter_keys:
        filter_datasets += ['{}/{}'.format(key, x) for x in  old[key].keys()]
    for ds_key in filter_datasets:  # actually mk datasets
        new.create_dataset_like(ds_key, other=old[ds_key])

    # simply copy everything we don't know how to filter
    for key in old.keys():
        if key not in ['data', 'evaluation', 'predictions', 'scores']:
            bkey = key.encode('utf-8')
            h5py.h5o.copy(old.id, bkey, new.id, bkey)

    new_start = 0
    for old_start in range(0, end, write_by):
        old_end = min(old_start + write_by, end)
        mask = np.logical_and(old['data/err_samples'][old_start:old_end],
                              old['data/is_annotated'][old_start:old_end])
        length = np.sum(mask)
        new_end = new_start + length
        # filter and copy over
        for ds_key in filter_datasets:
            new[ds_key][new_start:new_end] = old[ds_key][old_start:old_end][mask]
        new_start = new_end

    # truncate to length of new data
    for ds_key in filter_datasets:
        shape = list(new[ds_key].shape)
        new[ds_key].resize(tuple([new_start] + shape[1:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-data', '-d', type=str, required=True)
    parser.add_argument('--h5-out', '-o', type=str, required=True)
    parser.add_argument('--write-by', '-b', type=int, default=1000)
    args = parser.parse_args()
    main(args.h5_data, args.h5_out, args.write_by)
