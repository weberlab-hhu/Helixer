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
from collections import Counter, OrderedDict
import itertools

###################### REFERENCE FILE #######################
fpath = '/mnt/data/niklas/with_coverage/Mesculenta/test_data.h5'

till = 1000


class Matrix_class:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.normalized = self.data_nrom(data)
        self.absolute = self.abs_data(data)
        
    def data_nrom(self, data):
        data = np.array(list(data.values())).reshape((4,4))
        #normalize the data
        sums_ = np.sum(data, axis=1).reshape(-1, 1)
        normalized = data/sums_
        return normalized
    
    def abs_data(self, data):
        data = np.array(list(data.values()))
        return data.reshape((4, 4))

#class for h5 file containing all methods for analysis
class H6FILE:
    def __init__(self, path):
        self.path = path
        self.file = h5py.File(path, mode='r')
        self.out_path = self.path[:-3] + "_METRICS.h5"
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
            return self.file["data/y"][0:till], self.file["data/phases"][0:till]

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

    def entropy(self, argmax=False):
        if argmax:
            cce_argmax = functools.partial(self.cce_per_nt, argmax_=True)
            h_ent = np.sum(np.array(list(map(cce_argmax, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase))), axis=0)
        else:
            h_ent = np.sum(np.array(list(map(self.cce_per_nt, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase))), axis=0)
        return h_ent


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

    def f1_by_class(self, argmax=False):
        if argmax:
            f1_argmax = functools.partial(self.f1_per_chunk, argmax_=True)
            h_f1 = np.array(list(map(f1_argmax, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase)))
        else:
            h_f1 = np.array(list(map(self.f1_per_chunk, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase)))
        cols = range(h_f1.shape[1])
        means = np.array([np.mean(h_f1[h_f1[:,e] != 0, e]) for e in cols])
        return np.around(means, decimals=3)

    def error_classes(self, ref_CDS, pred_CDS, argmax_=False):
        idx = np.sum(ref_CDS, axis=1) != 0
        ref_CDS = ref_CDS[idx]
        pred_CDS = pred_CDS[idx]

        #argmax function
        if argmax_:
            pred_CDS = self.pred_to_argmax(pred_CDS)

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

    def error_quantification(self, ref_CDS, pred_CDS, argmax=False):
        if argmax:
            error_argmax = functools.partial(self.error_classes, argmax_=True)
            h_err = list(map(error_argmax, ref_CDS, pred_CDS))
        else: 
            h_err = list(map(self.error_classes, ref_CDS, pred_CDS))
        helixer = Counter(itertools.chain(*h_err))
        helixer = OrderedDict(sorted(helixer.items()))
        helixer = Matrix_class(helixer, "helixer") ####change namHERE !!!!
        return helixer

    def to_dataframe(self, argmax=False):
        ent = self.entropy(argmax=argmax)
        f1_score = self.f1_by_class(argmax=argmax)
        genic = self.error_quantification(self.ref_CDS, self.pred_CDS, argmax=argmax)
        phase = self.error_quantification(self.ref_phase, self.pred_phase, argmax=argmax) 
        df = np.array([ent, f1_score, genic.normalized, genic.absolute, phase.normalized, phase.absolute], dtype=object)
        file_out = h5py.File(self.out_path, 'w')
        file_out.create_dataset("entropy", data=ent)
        file_out.create_dataset("f1_score", data=f1_score)
        file_out.create_dataset("genic_normalized", data=genic.normalized)
        file_out.create_dataset('genic_absolute', data=genic.absolute)
        file_out.create_dataset('phase_normalized', data=phase.normalized)
        file_out.create_dataset('phase_absolute', data=phase.absolute)
        file_out.close()
        return df 


    def calc_metrics(self):
        if self.is_helixer:
            df_helixer = self.to_dataframe(argmax=False)
            df_argmax = self.to_dataframe(argmax=True)
            return np.array([df_helixer, df_argmax]).reshape((2, -1))
        if not self.is_helixer:
            df_post = self.to_dataframe(argmax=False)
            return df_post

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
            df = data_class.calc_metrics()
            print(df)
            print(data_class.out_path)
            data_class.file.close()
            data_class.ref.close()
        print("________________________")    


if __name__ == "__main__":
    run()
