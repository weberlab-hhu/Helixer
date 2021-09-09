import sys
import os
import glob
import h5py
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import pandas as pd
import tensorflow as tf
import functools



###################### REFERENCE FILE #######################
fpath = '/mnt/data/niklas/with_coverage/Mesculenta/test_data.h5'


#class for h5 file containing all methods for analysis
class H6FILE:
    def __init__(self, path):
        self.path = path
        self.file = h5py.File(path, mode='r')
        self.is_helixer = self.get_mode()
        self.pred_CDS, self.pred_phase = self.get_pred_data()

        self.ref = h5py.File(fpath, mode='r')
        self.ref_CDS = self.ref["data/y"][0:100]
        self.ref_phase = self.ref["data/phases"][0:100]

    def get_mode(self):
        if len(self.file.keys()) == 2:
            return True
        if len(self.file.keys()) == 1:
            return False

    def get_pred_data(self):
        if self.is_helixer:
            return self.file["predictions"][0:100], self.file["predictions_phase"][0:100]
        else:
            return self.file["data/y"][0:100], self.file["data/phases"][0:100]        

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

    def to_dataframe(self):
        ent = self.entropy()
        d = {"entropy": ent}
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
            print(df)
        print("________________________")    


if __name__ == "__main__":
    run()
