#! /usr/bin/env python3
import h5py
import numpy as np
import tkinter as tk

class Visualization(tk.Frame):
    MARGIN_X = 100
    MARGIN_BOTTOM = 100
    CANVAS_SIZE_X = 1800 - (2 * MARGIN_X)
    CANVAS_SIZE_Y = CANVAS_SIZE_X // (16 / 9) - MARGIN_BOTTOM
    PIXEL_SIZE = 20
    BASE_COUNT_X = CANVAS_SIZE_X // PIXEL_SIZE

    def __init__(self, data_path, predictions_path):
        self.data = h5py.File(data_path, 'r')
        self.predictions = h5py.File(predictions_path, 'r')

        self.master = tk.Tk()
        self.canvas = self.init_canvas("#ffffff")
        self.rects = self.init_rects()

        # GUI elements
        self.button_frame = tk.Frame(self.master)

        self.show_pred_button = tk.Button(self.button_frame, text='Show Pred')
        # self.show_pred_button.bind('<ButtonPress-1>', self.show_pred)
        self.show_pred_button.pack(side='left')
        self.button_frame.pack(side='bottom')

        self.text = tk.Label(self.master, text=0)
        self.text.pack(side='bottom')

        self.master.mainloop()

    def init_canvas(self, bg_color):
        canvas = tk.Canvas(self.master, height=self.CANVAS_SIZE_Y,
                width=self.CANVAS_SIZE_X)
        full_rect = canvas.create_rectangle(0, 0, self.CANVAS_SIZE_X,
                self.CANVAS_SIZE_Y, outline='', fill=bg_color)
        canvas.pack()
        return canvas

    def init_rects(self):
        rects = []
        for y in range(3):
            rects.append([])  # one row e.g. representing intron predictions
            for x in range(self.BASE_COUNT_X):
                rect = self.canvas.create_rectangle(x * self.PIXEL_SIZE,
                                                    y * self.PIXEL_SIZE,
                                                    (x + 1) * self.PIXEL_SIZE,
                                                    (y + 1) * self.PIXEL_SIZE,
                                                    outline='')
                rects[y].append(rect)
        return rects


if __name__ == '__main__':
    vis = Visualization(data_path='/home/felix/Desktop/h5_data_1K/validation_data.h5',
                        predictions_path='/home/felix/git/HelixerPrep/helixerprep/prediction/predictions.h5')
