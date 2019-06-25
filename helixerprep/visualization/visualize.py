#! /usr/bin/env python3
import h5py
import numpy as np
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

class Visualization():
    MARGIN_X = 100
    MARGIN_BOTTOM = 100
    HEATMAP_SIZE_X = 1800 - (2 * MARGIN_X)
    HEATMAP_SIZE_Y = HEATMAP_SIZE_X // (16 / 9) - MARGIN_BOTTOM
    PIXEL_SIZE = 20
    BASE_COUNT_X = HEATMAP_SIZE_X // PIXEL_SIZE

    def __init__(self, root, data_path, predictions_path):
        self.root = root
        self.h5_data = h5py.File(data_path, 'r')
        self.h5_predictions = h5py.File(predictions_path, 'r')

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        canvas = FigureCanvasTkAgg(f, self.root)
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

