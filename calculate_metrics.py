import sys
import os
import glob
import h5py

###################### REFERENCE FILE #######################
fpath = '/mnt/data/niklas/with_coverage/Mesculenta/test_data.h5'


#class for h5 file containing all methods for analysis
class H6FILE:
    def __init__(self, path):
        self.path = path
        self.file = h5py.File(path, mode='r')

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
        print("________________________")    


if __name__ == "__main__":
    run()
