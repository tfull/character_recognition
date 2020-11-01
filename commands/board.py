import tkinter
from PIL import Image, ImageDraw

import _pre


class Board:
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("かな入力")

        self.frame = tkinter.Frame(self.window, width = 1300, height = 300)
        self.frame.pack()

        self.canvas = tkinter.Canvas(self.frame, bg = "black", width = 1280, height = 256)
        self.canvas.place(x = 0, y = 0)

        self.canvas.bind("<ButtonPress-1>", self.click_left)
        self.canvas.bind("<B1-Motion>", self.drag_left)
        self.canvas.bind("<ButtonPress-3>", self.click_right)
        self.canvas.bind("<B3-Motion>", self.drag_right)

        self.button = tkinter.Button(self.frame, bg = "blue", fg = "yellow", text = "認識", width = 100, height = 40, command = self.press_button)
        self.button.place(x = 0, y = 256)

        self.label = tkinter.Label(self.frame, bg = "green", fg = "yellow", text = "hello", width = 400, height = 40)
        self.label.place(x = 100, y = 256)

        self.image = Image.new("L", (1280, 256))
        self.draw = ImageDraw.Draw(self.image)

    def press_button(self):
        self.image.save("tmp.png")

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
