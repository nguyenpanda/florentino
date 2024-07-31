import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

import florentino.nn as ann


class DrawApp:
    # @formatter:off
    LINE_COLOR = 'white'
    BACKGROUND = 'black'
    IMG_SIZE   = 560
    RATIO      = IMG_SIZE // 28
    # @formatter:on

    def __init__(self, root: tk.Tk, model: ann.Network):
        self.model = model

        self.root = root
        root.title("Handwritten Digit Recognizer")
        self._screen_init_size(self.IMG_SIZE + 5, self.IMG_SIZE + 30, root.winfo_screenwidth(),
                               root.winfo_screenheight())

        self.frame = tk.Frame(root, bg="#0000ff")
        self.frame.grid(row=0, column=0, sticky='nsew')

        self.btn_save = tk.Button(self.frame, text='Save', command=self.save_image)
        self.btn_predict = tk.Button(self.frame, text='Predict', command=self.display_prediction)
        self.btn_reset = tk.Button(self.frame, text='Reset', command=self.reset_canvas)
        self.btn_save.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.btn_predict.grid(row=0, column=2, columnspan=2, sticky='ew')
        self.btn_reset.grid(row=0, column=4, columnspan=2, sticky='ew')

        self.prediction_label = tk.Label(self.frame, text="Label: ", font=("Arial", 14))
        self.prediction_label.grid(row=1, column=0, columnspan=2, sticky='ew')

        self.prediction_probability = tk.Label(self.frame, text="Probability: ", font=("Arial", 14))
        self.prediction_probability.grid(row=1, column=2, columnspan=4, sticky='ew')

        self.canvas = tk.Canvas(self.frame, width=self.IMG_SIZE, height=self.IMG_SIZE, bg=self.BACKGROUND)
        self.canvas.grid(row=2, column=0, columnspan=6, sticky='nsew')

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drawing)

        self.image = Image.new('L', (28, 28), self.BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)

        self.old_x = self.old_y = None

    def paint(self, event):
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, x, y, width=self.RATIO,
                                    fill=self.LINE_COLOR, capstyle=tk.ROUND,
                                    smooth=tk.TRUE, splinesteps=36)
            self.draw.line([(self.old_x // self.RATIO, self.old_y // self.RATIO), (x // self.RATIO, y // self.RATIO)],
                           fill=self.LINE_COLOR, width=2)
        self.old_x = x
        self.old_y = y

    def reset_drawing(self, event):
        self.old_x = None
        self.old_y = None

    def save_image(self):
        self.image.save("drawn_image.png")
        print("Image saved as 'drawn_image.png'")

    def predict_digit(self):
        x: np.ndarray = np.array(self.image).reshape(784, 1) / 255.0
        y: np.ndarray = self.model.predict(x).flatten()
        label = np.argmax(y)
        print(f'Prediction: {label}')
        print(f'Probability: {y}')
        print('-' * 10)
        return label, y

    def display_prediction(self):
        label, y = self.predict_digit()
        self.prediction_label.config(text=f'Prediction: {label}')
        self.prediction_probability.config(text=f'Probability: {y}')

    def reset_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (28, 28), self.BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)

    def _screen_init_size(self, ww, wh, sw, sh):
        x_position = int((sw / 2) - (ww / 2))
        y_position = int((sh / 2) - (wh / 2))
        self.root.geometry(f'{ww}x{wh}+{x_position}+{y_position}')


if __name__ == '__main__':
    np.set_printoptions(suppress=True, formatter=dict(float='{:0.2f}'.format))

    try:
        AI = ann.Network.load('parameters/d50_d50_s10_CE', ann.CrossEntropy())
        window = tk.Tk()
        app = DrawApp(window, AI)
        icon = tk.PhotoImage(file="app_icon.png")
        window.iconphoto(True, icon)
        window.mainloop()
    except FileNotFoundError as e:
        print('\033[1;91mYou must train model by running ANN_mnist and '
              'then put the folder `parameters/d50_d50_s10_CE` into `GUI` folder\033[0m')
