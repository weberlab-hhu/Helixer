import sys
import glob
import h5py
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import tensorflow as tf
import functools
from collections import Counter, OrderedDict
import itertools


###################### REFERENCE FILE #######################
fpath = '/mnt/data/niklas/with_coverage/Mesculenta/test_data.h5'

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
        self.ref_CDS = self.ref["data/y"]
        self.ref_phase = self.ref["data/phases"]
        self.ref_size = np.arange(0, self.ref_CDS.shape[0])

    def get_mode(self):
        if len(self.file.keys()) == 2:
            return True
        if len(self.file.keys()) == 1:
            return False

    def get_pred_data(self):
        if self.is_helixer:
            return self.file["predictions"], self.file["predictions_phase"]
        else:
            return self.file["data/y"], self.file["data/phases"]

    @staticmethod
    def pred_to_argmax(pred):
        idx = np.argmax(pred, axis = 1)
        pred_arg = np.eye(4)[idx]
        pred_arg[-1,:] = 0
        pred_arg[-2,:] = 0
        return pred_arg

    def cce_per_nt(self, ref_CDS, pred_CDS, ref_phase, pred_phase, index, argmax_=False):
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
        #print(index, "/", len(self.ref_size), end = "\r")
        return np.array([ent_cds, ent_phase, ent])

    def entropy(self, argmax=False):
        if argmax:
            cce_argmax = functools.partial(self.cce_per_nt, argmax_=True)
            h_ent = np.sum(np.array(list(map(cce_argmax, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase, self.ref_size))), axis=0)
        else:
            h_ent = np.sum(np.array(list(map(self.cce_per_nt, self.ref_CDS, self.pred_CDS, self.ref_phase, self.pred_phase, self.ref_size))), axis=0)
        return h_ent

    @staticmethod
    def scores(chunk, ref, mode):
        idx = np.argmax(chunk, axis=1)
        chunk = np.eye(4)[idx].astype(np.int8)
        tp = np.sum(np.logical_and(ref[:, mode] == 1, chunk[:, mode] == 1))
        fp = np.sum(np.logical_and(ref[:, mode] == 0, chunk[:, mode] == 1))
        fn = np.sum(np.logical_and(ref[:, mode] == 1, chunk[:, mode] == 0))
        return np.array([tp, fp, fn])

    @staticmethod
    def f1_calc(metrics):
        tp = metrics[0]
        fp = metrics[1]
        fn = metrics[2]
        f1 = tp / (tp + 0.5 * (fp + fn))
        return f1

    def f1_score(self, chunk, ref):
        # count tp, fp ,fn per class
        ig = np.sum(np.array([self.scores(chunk[e], ref[e], 0) for e in range(len(self.ref_size))]), axis=0)
        utr = np.sum(np.array([self.scores(chunk[e], ref[e], 1) for e in range(len(self.ref_size))]), axis=0)
        cds = np.sum(np.array([self.scores(chunk[e], ref[e], 2) for e in range(len(self.ref_size))]), axis=0)
        intron = np.sum(np.array([self.scores(chunk[e], ref[e], 3) for e in range(len(self.ref_size))]), axis=0)
        # calculate f1 scores
        ig_f1 = self.f1_calc(ig)
        utr_f1 = self.f1_calc(utr)
        cds_f1 = self.f1_calc(cds)
        intron_c1 = self.f1_calc(intron)
        return np.array([ig_f1, utr_f1, cds_f1, intron_c1])

    def f1_total(self):
        f1_genic = self.f1_score(self.pred_CDS, self.ref_CDS)
        f1_phase = self.f1_score(self.pred_phase, self.ref_phase)
        return np.array([f1_genic, f1_phase])

    def error_classes(self, ref_CDS, pred_CDS, index, argmax_=False):
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

        #print(index, "/", len(self.ref_size), end = "\r")
        return error_matrix.astype(np.int8)

    def error_quantification(self, ref_CDS, pred_CDS, argmax=False):
        if argmax:
            error_argmax = functools.partial(self.error_classes, argmax_=True)
            h_err = list(map(error_argmax, ref_CDS, pred_CDS, self.ref_size))
        else: 
            h_err = list(map(self.error_classes, ref_CDS, pred_CDS, self.ref_size))
        helixer = Counter(itertools.chain(*h_err))
        helixer = OrderedDict(sorted(helixer.items()))
        helixer = Matrix_class(helixer, "helixer") ####change namHERE !!!!
        return helixer

    def to_dataframe(self):
        print("\n========== CALCULATING CROSSENTROPY  ==========")
        ent = self.entropy()
        print("CROSS-ENTROPY: ")
        print(ent)
        if self.is_helixer:
            print("\nCalculating argmaxed cross-entropy")
            ent_argmax = self.entropy(argmax=True)
            print("CROSS-ENTROPY: ")
            print(ent_argmax)

        print("\n========== CALCULATING CLASS-WISE F1 SCORE  ==========")
        f1_score = self.f1_total()
        print("F1-SCORE:")
        print(np.around(f1_score, decimals=4))

        print("\n========== CALCULATING GENIC CONFUSION MATRIX  ==========")
        genic = self.error_quantification(self.ref_CDS, self.pred_CDS)
        print("Genic normalized confusion matrix: ")
        print(np.around(genic.normalized, decimals=4))
        print("\n========== CALCULATING PHASE CONFUSION MATRIX  ==========")
        phase = self.error_quantification(self.ref_phase, self.pred_phase)
        print(" Phase normalized confusion matric: ")
        print(np.around(phase.normalized, decimals=4))

        file_out = h5py.File(self.out_path, 'w')
        file_out.create_dataset("entropy", data=ent)
        if self.is_helixer:
            file_out.create_dataset("entropy_argmax", data=ent_argmax)
        file_out.create_dataset("f1_score", data=f1_score)
        file_out.create_dataset("genic_normalized", data=genic.normalized)
        file_out.create_dataset('genic_absolute', data=genic.absolute)
        file_out.create_dataset('phase_normalized', data=phase.normalized)
        file_out.create_dataset('phase_absolute', data=phase.absolute)
        file_out.create_dataset('name', data=str(self.path))
        file_out.close()

    def calc_metrics(self):
        df_helixer = self.to_dataframe()
        return df_helixer

def run():
    paths = sys.argv
    paths = paths[1:]
    #reference_path = str(input("Provide path to reference file: "))
    print("___________________")
    for path in paths:
        print("Current directory: \n" + path)
        filenames = path + "/*.h5"
        predictions = glob.glob(filenames)
        for file in predictions:
            print("\n========== STARING OPERATION WITH: " + file + " ==========")
            data_class = H6FILE(file)
            data_class.calc_metrics()
            print("========== PATH OF H5FILE: " + data_class.out_path)
            data_class.file.close()
            data_class.ref.close()
        print("________________________")    


if __name__ == "__main__":
    run()
