import tkinter
from PIL import Image, ImageDraw
import numpy as np
import sys

import _pre
import src.core
from src.config import Config


class Board:
    def __init__(self):
        self.image_size = Config.image_size

        self.window = tkinter.Tk()
        self.window.title("かな入力")

        self.frame = tkinter.Frame(self.window, width = self.image_size + 2, height = self.image_size + 40)
        self.frame.pack()

        self.canvas = tkinter.Canvas(self.frame, bg = "black", width = self.image_size, height = self.image_size)
        self.canvas.place(x = 0, y = 0)

        self.canvas.bind("<ButtonPress-1>", self.click_left)
        self.canvas.bind("<B1-Motion>", self.drag_left)
        self.canvas.bind("<ButtonPress-3>", self.click_right)
        self.canvas.bind("<B3-Motion>", self.drag_right)

        self.button_detect = tkinter.Button(self.frame, bg = "blue", fg = "white", text = "認識", width = 100, height = 40, command = self.press_detect)
        self.button_detect.place(x = 0, y = self.image_size)

        self.button_delete = tkinter.Button(self.frame, bg = "green", fg = "white", text = "削除", width = 100, height = 40, command = self.press_delete)
        self.button_delete.place(x = self.image_size // 2, y = self.image_size)

        self.image = Image.new("L", (self.image_size, self.image_size))
        self.draw = ImageDraw.Draw(self.image)

        self.recognizer = src.core.Recognizer()

    def press_detect(self):
        output = self.recognizer.detect(np.array(self.image).reshape(1, 1, self.image_size, self.image_size))
        sys.stdout.write(output)
        sys.stdout.flush()

    def press_delete(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.image_size, self.image_size), fill = 0)

    def click_left(self, event):
        ex = event.x
        ey = event.y

        self.canvas.create_oval(
            ex, ey, ex, ey,
            outline = "white",
            width = 8
        )

        self.draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill = 255)

        self.x = ex
        self.y = ey

    def drag_left(self, event):
        ex = event.x
        ey = event.y

        self.canvas.create_line(
            self.x, self.y, ex, ey,
            fill = "white",
            width = 8
        )

        self.draw.line((self.x, self.y, ex, ey), fill = 255, width = 8)

        self.x = ex
        self.y = ey

    def click_right(self, event):
        ex = event.x
        ey = event.y

        self.canvas.create_oval(
            ex, ey, ex, ey,
            outline = "black",
            width = 8
        )

        self.draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill = 0)

        self.x = event.x
        self.y = event.y

    def drag_right(self, event):
        ex = event.x
        ey = event.y

        self.canvas.create_line(
            self.x, self.y, ex, ey,
            fill = "black",
            width = 8
        )

        self.draw.line((self.x, self.y, ex, ey), fill = 0, width = 8)

        self.x = event.x
        self.y = event.y

    def start(self):
        self.window.mainloop()


if __name__ == '__main__':
    Board().start()
    sys.stdout.write("\n")
