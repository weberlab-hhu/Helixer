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
        self.offset = 0  # of the data

        # set up GUI
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.next_button = tk.Button(self.frame, text='next')
        self.next_button.bind('<ButtonPress-1>', self.next)
        self.next_button.pack(side='left')

        self.previous_button = tk.Button(self.frame, text='previous')
        self.previous_button.bind('<ButtonPress-1>', self.previous)
        self.previous_button.pack(side='left')

        self.offset_label = tk.Label(self.frame, text=self.offset)
        self.offset_label.pack(side='bottom')

        # only for developement
        TRUNCATE = 8

        # load and transform data
        h5_data = h5py.File(data_path, 'r')
        h5_predictions = h5py.File(predictions_path, 'r')

        self.labels = np.array(h5_data['/data/y'][:TRUNCATE])
        shape = self.labels.shape
        self.labels = self.labels.reshape((shape[0] * shape[1], shape[2]))

        self.labels_str = self.labels.astype(str)
        self.labels_str[self.labels_str == '0'] = ''
        self.labels_str[self.labels_str == '1'] = '-'

        self.predictions = np.array(h5_predictions['/predictions'][:TRUNCATE])
        shape = self.predictions.shape
        self.predictions = self.predictions.reshape((shape[0] * shape[1], shape[2]))

        self.errors = np.abs(self.labels - self.predictions)

        self.label_masks = np.array(h5_data['/data/sample_weights'][:TRUNCATE]).ravel()
        self.label_masks = np.repeat(self.label_masks[:, np.newaxis], 3, axis=1)
        self.label_masks = ([1] - self.label_masks).astype(bool)

        fig = Figure(figsize=(self.HEATMAP_SIZE_X/self.DPI, self.HEATMAP_SIZE_Y/self.DPI), dpi=self.DPI)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.draw_current_heatmap()

    def draw_current_heatmap(self):
        self.ax.clear()
        seaborn.heatmap(self.errors[self.offset:self.offset+self.BASE_COUNT_X].T,
                        cmap='RdYlGn_r',
                        center=0.5,
                        square=True,
                        cbar=False,
                        mask=self.label_masks[self.offset:self.offset+self.BASE_COUNT_X].T,
                        annot=self.labels_str[self.offset:self.offset+self.BASE_COUNT_X].T,
                        fmt='',
                        annot_kws={'fontweight': 'bold'},
                        xticklabels=False,
                        yticklabels=False,
                        ax=self.ax)
        self.canvas.draw()

    def next(self, event):
        self.offset += self.BASE_COUNT_X
        self.redraw()

    def previous(self, event):
        self.offset -= self.BASE_COUNT_X
        self.redraw()

    def redraw(self):
        self.draw_current_heatmap()
        self.offset_label.config(text=str(self.offset))


if __name__ == '__main__':
    root = tk.Tk()
    root.title('root')

    vis = Visualization(root,
                        data_path='/home/felix/Desktop/h5_data_1K/validation_data.h5',
                        predictions_path='/home/felix/git/HelixerPrep/helixerprep/prediction/predictions.h5')

    root.mainloop()

