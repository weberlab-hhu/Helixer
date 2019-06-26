#! /usr/bin/env python3
import h5py
import numpy as np
import tkinter as tk
import seaborn
import matplotlib
import argparse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class Visualization():
    """Visualizes predictions for a set of sequences. Internally these sequences are
    concatenated. For the user it does not appear so, which requires some basic
    recalculations.
    """
    BASE_COUNT_X = 100
    MARGIN_BOTTOM = 100
    HEATMAP_SIZE_X = 1920
    PIXEL_SIZE = HEATMAP_SIZE_X // BASE_COUNT_X
    HEATMAP_SIZE_Y = PIXEL_SIZE * 3
    DPI = 96  # monitor specific

    def __init__(self, root, args):
        self.root = root
        self.offset = 0  # of the data
        self.seq_index = 0

        # set up GUI
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.previous_button = tk.Button(self.frame, text='previous')
        self.previous_button.bind('<ButtonPress-1>', self.previous)
        self.previous_button.grid(row=0, column=0)

        self.next_button = tk.Button(self.frame, text='next')
        self.next_button.bind('<ButtonPress-1>', self.next)
        self.next_button.grid(row=0, column=1)

        self.seq_index_label = tk.Label(self.frame)
        self.seq_index_input = tk.Entry(self.frame, width=6)
        self.seq_index_button = tk.Button(self.frame, text='go')
        self.seq_index_button.bind('<ButtonPress-1>', self.go_seq_index)
        self.seq_index_label.grid(row=1, column=0)
        self.seq_index_input.grid(row=1, column=1)
        self.seq_index_button.grid(row=1, column=2)

        self.seq_offset_label = tk.Label(self.frame)
        self.seq_offset_input = tk.Entry(self.frame, width=6)
        self.seq_offset_button = tk.Button(self.frame, text='go')
        self.seq_offset_button.bind('<ButtonPress-1>', self.go_seq_offset)
        self.seq_offset_label.grid(row=2, column=0)
        self.seq_offset_input.grid(row=2, column=1)
        self.seq_offset_button.grid(row=2, column=2)

        # only for developement
        TRUNCATE = 8

        # load and transform data
        h5_data = h5py.File(args.test_data, 'r')
        h5_predictions = h5py.File(args.predictions, 'r')

        self.labels = np.array(h5_data['/data/y'][:TRUNCATE])
        shape = self.labels.shape
        self.labels = self.labels.reshape((shape[0] * shape[1], shape[2]))

        self.labels_str = self.labels.astype(str)
        self.labels_str[self.labels_str == '0'] = ''
        self.labels_str[self.labels_str == '1'] = '-'

        self.predictions = np.array(h5_predictions['/predictions'][:TRUNCATE])
        shape = self.predictions.shape
        # save n_seq and chunk_len from predictions as there are likely a tiny bit fewer
        # than labels, due to the data generator in keras
        self.n_seq = shape[0]
        self.chunk_len = shape[1]
        self.predictions = self.predictions.reshape((shape[0] * shape[1], shape[2]))

        self.errors = np.abs(self.labels - self.predictions)

        self.label_masks = np.array(h5_data['/data/sample_weights'][:TRUNCATE]).ravel()
        self.label_masks = np.repeat(self.label_masks[:, np.newaxis], 3, axis=1)
        self.label_masks = ([1] - self.label_masks).astype(bool)

        fig = Figure(figsize=(self.HEATMAP_SIZE_X/self.DPI, (self.HEATMAP_SIZE_Y + 100)/self.DPI), dpi=self.DPI)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.redraw()

    def draw_current_heatmap(self):
        self.ax.clear()
        seaborn.heatmap(self.errors[self.offset:self.offset+self.BASE_COUNT_X].T,
                        vmin=0 + args.colorbar_offset,
                        vmax=1 - args.colorbar_offset,
                        cmap='bwr',
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
        if self.offset + self.BASE_COUNT_X < self.n_seq * self.chunk_len:
            self.offset += self.BASE_COUNT_X
        else:
            self.offset = self.n_seq * self.chunk_len - self.BASE_COUNT_X
        self.seq_index = self.offset // self.chunk_len
        self.redraw()

    def previous(self, event):
        if self.offset - self.BASE_COUNT_X >= 0:
            self.offset -= self.BASE_COUNT_X
        else:
            self.offset = 0
        self.seq_index = self.offset // self.chunk_len
        self.redraw()

    def go_seq_index(self, event):
        new_seq_index = int(self.seq_index_input.get())
        if new_seq_index <= self.n_seq:
            self.seq_index = new_seq_index
            self.offset = self.seq_index * self.chunk_len
            self.redraw()

    def go_seq_offset(self, event):
        """offset here is within a sequence, as it appears to the user"""
        new_seq_offset = int(self.seq_offset_input.get())
        if new_seq_offset <= self.chunk_len:
            self.offset = self.seq_index * self.chunk_len + new_seq_offset
            self.redraw()

    def redraw(self):
        self.draw_current_heatmap()
        seq_offset = self.offset % self.chunk_len
        self.seq_offset_label.config(text=str('base: {}/{}'.format(seq_offset, self.chunk_len - 1)))
        self.seq_index_label.config(text=str('seq: {}/{}'.format(self.seq_index, self.n_seq - 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test-data', type=str, default='', required=True)
    parser.add_argument('-p', '--predictions', type=str, default='', required=True)
    # how to narrow down the vmin/vmax args of the heatmap as predictions are very close to 0
    parser.add_argument('-cbo', '--colorbar-offset', type=float, default=0.2)
    args = parser.parse_args()

    root = tk.Tk()
    root.title('Helixer Visualization')
    vis = Visualization(root, args)
    root.mainloop()
