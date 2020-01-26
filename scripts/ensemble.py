#! /usr/bin/env python3
import h5py
import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prediction-files', action='append')
parser.add_argument('-po', '--prediction-output-path', type=str, default='ensembled_predictions.h5')
args = parser.parse_args()

# make sure we have at least 2 predictions
assert args.prediction_files and len(args.prediction_files) > 1

h5_pred_files, y_preds = [], []
for pf in args.prediction_files:
    f = h5py.File(pf, 'r')
    y_preds.append(f['/predictions'])
    h5_pred_files.append(f)

# warn if predictions are not all for the same test data
test_data_files = [f.attrs['test_data_path'] for f in h5_pred_files]
if len(set(test_data_files)) > 1:
    print(f'WARNING: Not all test data file paths of the predictions files are equal: {test_data_files}')

shapes = [dset.shape for dset in y_preds]
assert len(set(shapes)) == 1, f'Prediction shapes are not equal: {shapes}'

# create output dataset
n_seqs = shapes[0][0]
h5_ensembled = h5py.File(args.prediction_output_path, 'w')
h5_ensembled.create_dataset('/predictions',
                            shape= (n_seqs,) + y_preds[0].shape[1:],
                            chunks=(1,) + y_preds[0].shape[1:],
                            dtype='float32',
                            compression='lzf',
                            shuffle=True)

# ensemble and add predictions
for i in range(n_seqs):
    print(i, '/', n_seqs - 1, end='\r')
    seq_predictions = np.stack([dset[i] for dset in y_preds], axis=0)
    seq_predictions = np.mean(seq_predictions, axis=0)
    # save data one sequence at a time to save memory at the expense of speed
    h5_ensembled['/predictions'][i] = seq_predictions

# also save some model attrs
h5_ensembled.attrs['timestamp'] = str(datetime.datetime.now())
h5_ensembled.attrs['model_md5sums'] = ','.join([dset.attrs['model_md5sum'] for dset in h5_pred_files])
h5_ensembled.attrs['model_paths'] = ','.join([dset.attrs['model_path'] for dset in h5_pred_files])
h5_ensembled.attrs['test_data_paths'] = ','.join([dset.attrs['test_data_path'] for dset in h5_pred_files])

h5_ensembled.close()
