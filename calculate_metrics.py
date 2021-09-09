import sys
import os
import glob
import h5py
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import pandas as pd
import tensorflow as tf
import functools
import tensorflow_addons as tfa


###################### REFERENCE FILE #######################
fpath = '/mnt/data/niklas/with_coverage/Mesculenta/test_data.h5'

till = 10000

#class for h5 file containing all methods for analysis
class H6FILE:
    def __init__(self, path):
        self.path = path
        self.file = h5py.File(path, mode='r')
        self.is_helixer = self.get_mode()
        self.pred_CDS, self.pred_phase = self.get_pred_data()

        self.ref = h5py.File(fpath, mode='r')
        self.ref_CDS = self.ref["data/y"][0:till]
        self.ref_phase = self.ref["data/phases"][0:till]

    def get_mode(self):
        if len(self.file.keys()) == 2:
            return True
        if len(self.file.keys()) == 1:
            return False

    def get_pred_data(self):
        if self.is_helixer:
            return self.file["predictions"][0:till], self.file["predictions_phase"][0:till]
        else:

    @staticmethod
    def pred_to_argmax(pred):
        idx = np.argmax(pred, axis = 1)
        pred_arg = np.eye(4)[idx]
        pred_arg[-1,:] = 0
        pred_arg[-2,:] = 0
        return pred_arg

    def cce_per_nt(self, ref_CDS, pred_CDS, ref_phase, pred_phase, argmax_=False):
        #removal of sequences that are incomplete
        idx = np.sum(ref_CDS, axis=1) != 0
        ref_CDS = ref_CDS[idx]
        pred_CDS = pred_CDS[idx]
        ref_phase = ref_phase[idx]
        pred_phase = pred_phase[idx]
    
        #argmax function
        if argmax_:
            pred_CDS = self.pred_to_argmax(pred_CDS)
            pred_phase = self.pred_to_argmax(pred_phase)
    
        #calculation of categorical crossentropy
        ent_phase = categorical_crossentropy(ref_phase, pred_phase.astype(np.float16))
        ent_cds = categorical_crossentropy(ref_CDS, pred_CDS.astype(np.float16))
        #sum of crossentropy across different sets of predictions (genic/phase)
        ent_cds = np.sum(tf.cast(ent_cds, tf.int64))
        ent_phase = np.sum(tf.cast(ent_phase, tf.int64))
        ent = ent_cds + ent_phase
        return np.array([ent_cds, ent_phase, ent])

    def entropy(self):
        h_ent = np.sum(np.array(list(map(self.cce_per_nt, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase))), axis=0)
        if self.is_helixer:
            cce_argmax = functools.partial(self.cce_per_nt, argmax_=True)
            arg_ent = np.sum(np.array(list(map(cce_argmax, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase))), axis=0)
        else:
            arg_ent= np.array([0, 0, 0])
        return np.array([h_ent, arg_ent]).reshape((-1, 3))


    def f1_per_chunk(self, ref_CDS, pred_CDS, ref_phase, pred_phase, argmax_=False):
        idx = np.sum(ref_CDS, axis=1) != 0
        ref_CDS = ref_CDS[idx]
        pred_CDS = pred_CDS[idx].astype(np.float16)
    
        ref_phase = ref_phase[idx]
        pred_phase = pred_phase[idx].astype(np.float16)
    
        if argmax_:
            pred_CDS = self.pred_to_argmax(pred_CDS)
            pred_phase = self.pred_to_argmax(pred_phase)
    
        metric = tfa.metrics.F1Score(num_classes=4)
        metric.update_state(ref_CDS, pred_CDS)
        result_cds = metric.result()
        result_cds.numpy().astype(np.float16)
    
        metric = tfa.metrics.F1Score(num_classes=4)
        metric.update_state(ref_phase, pred_phase)
        result_phase = metric.result()
        result_phase.numpy().astype(np.float16)
    
        return np.array([result_cds, result_phase]).reshape(-1)

    def f1_by_class(self):
        h_f1 = np.array(list(map(self.f1_per_chunk, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase)))
        cols = range(h_f1.shape[1])
        means = np.array([np.mean(h_f1[h_f1[:,e] != 0, e]) for e in cols])
        return np.around(means, decimals=3)

    def error_classes(ref_CDS, pred_CDS, argmax_=False):
        idx = np.sum(ref_CDS, axis=1) != 0
        ref_CDS = ref_CDS[idx]
        pred_CDS = pred_CDS[idx]

        #argmax function
        if argmax_:
            pred_CDS = comp.pred_to_argmax(pred_CDS)

        #transformation of data
        ref_CDS = np.argmax(ref_CDS, axis=1)
        pred_CDS = np.argmax(pred_CDS, axis=1)

        #generation of an error matrix
        error_matrix = np.full(ref_CDS.shape[0], 0)
        #### error matrix with 0 means unclassified

        ###############################################
        # correct ig
        idx = np.logical_and(ref_CDS == 0, pred_CDS == 0)
        error_matrix[idx] = 1

        # wrong ig: UTR
        idx = np.logical_and(ref_CDS == 0, pred_CDS == 1)
        error_matrix[idx] = 2

        # wrong ig: CDS
        idx = np.logical_and(ref_CDS == 0, pred_CDS == 2)
        error_matrix[idx] = 3

        # wrong ig: intron
        idx = np.logical_and(ref_CDS == 0, pred_CDS == 3)
        error_matrix[idx] = 4

        ################################################
        # incorrect UTR: ig
        idx = np.logical_and(ref_CDS == 1, pred_CDS == 0)
        error_matrix[idx] = 5

        # correct UTR
        idx = np.logical_and(ref_CDS == 1, pred_CDS == 1)
        error_matrix[idx] = 6

        # incorrect UTR: CDS
        idx = np.logical_and(ref_CDS == 1, pred_CDS == 2)
        error_matrix[idx] = 7

        # incorrect UTR: intron
        idx = np.logical_and(ref_CDS == 1, pred_CDS == 3)
        error_matrix[idx] = 8

        ################################################
        # incorrect CDS: IG
        idx = np.logical_and(ref_CDS == 2, pred_CDS == 0)
        error_matrix[idx] = 9

        # incorrect CDS: UTR
        idx = np.logical_and(ref_CDS == 2, pred_CDS == 1)
        error_matrix[idx] = 10

        # correct CDS
        idx = np.logical_and(ref_CDS == 2, pred_CDS == 2)
        error_matrix[idx] = 11

        # incorrect CDS: intron
        idx = np.logical_and(ref_CDS == 2, pred_CDS == 3)
        error_matrix[idx] = 12

        ################################################
        # incorrect intron: IG
        idx = np.logical_and(ref_CDS == 3, pred_CDS == 0)
        error_matrix[idx] = 13

        # incorrect intron: UTR
        idx = np.logical_and(ref_CDS == 3, pred_CDS == 1)
        error_matrix[idx] = 14

        # incorrect intron: CDS
        idx = np.logical_and(ref_CDS == 3, pred_CDS == 2)
        error_matrix[idx] = 15

        # correct intron
        idx = np.logical_and(ref_CDS == 3, pred_CDS == 3)
        error_matrix[idx] = 16

        return error_matrix.astype(np.int8)



    def to_dataframe(self):
        ent = self.entropy()
        f1_score = self.f1_by_class()
        
        d = {"entropy": ent, 'f1': f1_score}
        df = pd.DataFrame(data=list(d.items()))
        return df 

def run():
    paths = sys.argv
    paths = paths[1:]
    print("___________________")
    for path in paths:
        print("Current directory: \n" + path)
        filenames = path + "/*.h5"
        predictions = glob.glob(filenames)
        for file in predictions:
            print(file)
            data_class = H6FILE(file)
            df = data_class.to_dataframe()
            print(df.to_string())
            data_class.file.close()
            data_class.ref.close()
        print("________________________")    


if __name__ == "__main__":
    run()
