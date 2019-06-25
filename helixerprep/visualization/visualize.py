#! /usr/bin/env python3
import h5py
import numpy as np
import tkinter as tk
import seaborn
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

class Visualization():
    PIXEL_SIZE = 20
    MARGIN_X = 50
    MARGIN_BOTTOM = 100
    HEATMAP_SIZE_X = 1800 - (2 * MARGIN_X)
    HEATMAP_SIZE_Y = PIXEL_SIZE * 3
    BASE_COUNT_X = HEATMAP_SIZE_X // PIXEL_SIZE
    DPI = 96  # monitor specific

    def __init__(self, root, data_path, predictions_path):
        self.root = root

        # set up GUI
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        # only for developement
        TRUNCATE = 8

        # load data
        h5_data = h5py.File(data_path, 'r')
        h5_predictions = h5py.File(predictions_path, 'r')

        self.labels = np.array(h5_data['/data/y'][:TRUNCATE])
        shape = self.labels.shape
        self.labels = self.labels.reshape((shape[0] * shape[1], shape[2]))
        # np.swapaxes(self.labels, 0, 1)

        self.predictions = np.array(h5_predictions['/predictions'][:TRUNCATE])
        shape = self.predictions.shape
        self.predictions = self.predictions.reshape((shape[0] * shape[1], shape[2]))
        # np.swapaxes(self.predictions, 0, 1)

        self.label_masks = np.array(h5_data['/data/sample_weights'][:TRUNCATE]).ravel()

        fig = Figure(figsize=(self.HEATMAP_SIZE_X/self.DPI, self.HEATMAP_SIZE_Y/self.DPI), dpi=self.DPI)
        ax = fig.add_subplot(111)
        ax = seaborn.heatmap(self.predictions[:self.BASE_COUNT_X].T,
                             square=True,
                             cbar=False,
                             xticklabels=False,
                             yticklabels=False,
                             ax=ax)

        canvas = FigureCanvasTkAgg(fig, self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # GUI elements
        # self.show_pred_button = tk.Button(self.frame, text='Show Pred')
        # self.show_pred_button.bind('<ButtonPress-1>', self.show_pred)
        # self.show_pred_button.pack(side='left')

        # self.text = tk.Label(self.frame, text=0)
        # self.text.pack(side='bottom')


if __name__ == '__main__':
    root = tk.Tk()
    root.title('root')

    vis = Visualization(root,
                        data_path='/home/felix/Desktop/h5_data_1K/validation_data.h5',
                        predictions_path='/home/felix/git/HelixerPrep/helixerprep/prediction/predictions.h5')

    root.mainloop()

