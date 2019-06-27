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
    def __init__(self, root, args):
        self.root = root
        self.args = args

        self.BASE_COUNT_X = 100
        self.BASE_COUNT_SCREEN = self.BASE_COUNT_X * args.n_rows
        self.MARGIN_BOTTOM = 100
        self.HEATMAP_SIZE_X = 1920
        self.PIXEL_SIZE = self.HEATMAP_SIZE_X // self.BASE_COUNT_X
        self.HEATMAP_SIZE_Y = self.PIXEL_SIZE * 3 * (args.n_rows * 2 - 1)
        self.DPI = 96  # monitor specific

        self.offset = 0  # of a sequence
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

        # load and transform data
        self.h5_data = h5py.File(args.test_data, 'r')
        self.h5_predictions = h5py.File(args.predictions, 'r')
        # save n_seq and chunk_len from predictions as there are likely a tiny bit fewer
        # than labels, due to the data generator in keras
        self.n_seq = self.h5_predictions['/predictions'].shape[0]
        self.chunk_len = self.h5_predictions['/predictions'].shape[1]

        assert self.chunk_len % self.BASE_COUNT_SCREEN == 0

        fig = Figure(figsize=(self.HEATMAP_SIZE_X/self.DPI, self.HEATMAP_SIZE_Y/self.DPI),
                     dpi=self.DPI)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.redraw()

    def load_sequence(self):
        """loads data for the heatmap and inputs dummy data that serves as margin"""
        def add_dummy_data(dset, new_dset):
            # insert valid data at every second row
            new_dset[0::2,:] = dset.reshape((self.args.n_rows, self.BASE_COUNT_X, 3))
            # reshape back to properly display in heatmap
            new_dset = np.swapaxes(new_dset, 1, 2)
            new_dset = new_dset.reshape((self.args.n_rows * 2 - 1) * 3, self.BASE_COUNT_X)
            return new_dset

        off = self.offset
        off_lim = self.offset + self.BASE_COUNT_SCREEN
        self.labels = np.array(self.h5_data['/data/y'][self.seq_index][off:off_lim])

        self.predictions = np.array(self.h5_predictions['/predictions'][self.seq_index][off:off_lim])
        self.errors = np.abs(self.labels - self.predictions)

        dset = self.h5_data['/data/sample_weights']
        self.label_masks = np.array(dset[self.seq_index][off:off_lim])
        self.label_masks = np.repeat(self.label_masks[:, np.newaxis], 3, axis=1)
        self.label_masks = ([1] - self.label_masks).astype(bool)

        # reshape and add dummy data
        new_labels = np.zeros((self.args.n_rows * 2 - 1, self.BASE_COUNT_X, 3),
                              dtype=self.labels.dtype)
        self.labels = add_dummy_data(self.labels, new_labels)
        new_errors = np.zeros((self.args.n_rows * 2 - 1, self.BASE_COUNT_X, 3),
                              dtype=self.errors.dtype)
        self.errors = add_dummy_data(self.errors, new_errors)
        new_label_masks = np.ones((self.args.n_rows * 2 - 1, self.BASE_COUNT_X, 3)).astype(bool)
        self.label_masks = add_dummy_data(self.label_masks, new_label_masks)

        # make string labels
        self.labels_str = self.labels.astype(str)
        self.labels_str[self.labels_str == '0'] = ''
        self.labels_str[self.labels_str == '1'] = '-'

    def draw_heatmap(self):
        self.ax.clear()
        # massively speeds up painting
        if np.all(self.labels_str == ''):
            self.labels_str = False
        seaborn.heatmap(self.errors,
                        vmin=0 + args.colorbar_offset,
                        vmax=1 - args.colorbar_offset,
                        cmap='bwr',
                        center=0.5,
                        square=True,
                        cbar=False,
                        mask=self.label_masks,
                        annot=self.labels_str,
                        fmt='',
                        annot_kws={'fontweight': 'bold'},
                        xticklabels=False,
                        yticklabels=False,
                        ax=self.ax)
        self.canvas.draw()

    def next(self, event):
        self.offset = (self.offset + self.BASE_COUNT_SCREEN) % self.chunk_len
        if self.offset < self.BASE_COUNT_SCREEN:
            self.seq_index += 1
        self.redraw()

    def previous(self, event):
        if self.offset < self.BASE_COUNT_SCREEN:
            self.offset = self.chunk_len + self.offset - self.BASE_COUNT_SCREEN
            self.seq_index -= 1
        else:
            self.offset -= self.BASE_COUNT_SCREEN
        self.redraw()

    def go_seq_index(self, event):
        new_seq_index = int(self.seq_index_input.get())
        if new_seq_index <= self.n_seq:
            self.seq_index = new_seq_index
            self.offset = 0
            self.redraw()

    def go_seq_offset(self, event):
        """offset here is within a sequence, as it appears to the user"""
        new_seq_offset = int(self.seq_offset_input.get())
        if new_seq_offset <= self.chunk_len:
            self.offset = new_seq_offset
            self.redraw()

    def redraw(self):
        self.seq_offset_label.config(text=str('base: {}/{}'.format(self.offset, self.chunk_len - 1)))
        self.seq_index_label.config(text=str('seq: {}/{}'.format(self.seq_index, self.n_seq - 1)))
        self.load_sequence()
        self.draw_heatmap()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test-data', type=str, default='', required=True)
    parser.add_argument('-p', '--predictions', type=str, default='', required=True)
    parser.add_argument('-r', '--n-rows', type=int, default=2)
    # how to narrow down the vmin/vmax args of the heatmap as predictions are very close to 0
    parser.add_argument('-cbo', '--colorbar-offset', type=float, default=0.2)
    args = parser.parse_args()

    root = tk.Tk()
    root.title('Helixer Visualization')
    vis = Visualization(root, args)
    root.mainloop()
