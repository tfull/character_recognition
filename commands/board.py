import tkinter

import _pre


class Board:
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("かな入力")

        self.canvas = tkinter.Canvas(self.window, bg = "black", width = 512, height = 256)
        self.canvas.pack()

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    Board().run()
