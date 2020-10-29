import tkinter

import _pre


class Board:
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("かな入力")

        self.canvas = tkinter.Canvas(self.window, bg = "black", width = 512, height = 256)
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.click_left)
        self.canvas.bind("<B1-Motion>", self.drag_left)
        self.canvas.bind("<ButtonPress-3>", self.click_right)
        self.canvas.bind("<B3-Motion>", self.drag_right)

    def click_left(self, event):
        self.canvas.create_oval(
            event.x, event.y, event.x, event.y,
            outline = "white",
            width = 8
        )

        self.x = event.x
        self.y = event.y

    def drag_left(self, event):
        self.canvas.create_line(
            self.x, self.y, event.x, event.y,
            fill = "white",
            width = 8
        )

        self.x = event.x
        self.y = event.y

    def click_right(self, event):
        self.canvas.create_oval(
            event.x, event.y, event.x, event.y,
            outline = "black",
            width = 8
        )

        self.x = event.x
        self.y = event.y

    def drag_right(self, event):
        self.canvas.create_line(
            self.x, self.y, event.x, event.y,
            fill = "black",
            width = 8
        )

        self.x = event.x
        self.y = event.y

    def start(self):
        self.window.mainloop()


if __name__ == '__main__':
    Board().start()
