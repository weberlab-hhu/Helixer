import h5py
import argparse
import numpy as np
from HelixerPrep.HelixerPrep.helixerprep.prediction.ConfusionMatrix import ConfusionMatrix as ConfusionMatrix
from keras.layers import Reshape
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
parser.add_argument('-ps', '--pool_size', type=int, default=10)

args = parser.parse_args()

h5_pred = h5py.File(args.predictions, 'r')
h5_data = h5py.File(args.data, 'r')
print ("\n______________________________________________________________\n\n"*2)

transitions = h5_data['/data/transitions']
print (transitions)

y_true = h5_data['/data/y']
print (y_true)

y_pred = h5_pred['/predictions']
print (y_pred)

sw = h5_data['/data/sample_weights']
print (sw,"\n\n")

n_seqs = y_true.shape[0]
cm = ConfusionMatrix(None)
 
for i in range(n_seqs):     
    trans = transitions[i].reshape((
    transitions[i].shape[0] // args.pool_size,
    args.pool_size,
    transitions[i].shape[-1],
    ))
    
    y_true_per_seq = y_true[i].reshape((
        y_true[i].shape[0] //args.pool_size,
        args.pool_size,
        y_true[i].shape[-1],
    ))


    y_pred_per_seq = y_pred[i].reshape((
        y_pred[i].shape[0] //args.pool_size,
        args.pool_size,
        y_pred[i].shape[-1],
    ))


    sw_t = [np.any((trans[:, :, col] == 1),axis=1) for col in range(6)]
    sw_t = np.stack(sw_t, axis=1).astype(np.int8)
    sw_t = np.sum(sw_t, axis=1)
    where_no_trans = np.where(sw_t== 0)
    
    s_w = sw[i].reshape((-1, args.pool_size))
    s_w = np.logical_not(np.any(s_w == 0, axis=1)).astype(np.int8)
    s_w[where_no_trans] = 0
    
    assert y_true.shape == y_pred.shape
    assert y_pred.shape[:-1] == sw.shape
    if not np.all(s_w == 0):
        print(i, '/', n_seqs, end='\r')
        cm._add_to_cm(y_true_per_seq, y_pred_per_seq, s_w)
cm.print_cm()



