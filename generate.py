import os
import subprocess


data_dir = "data"
image_size = 256
font_list = [
    "Noto-Sans-CJK-JP-Thin"
]


def main():
    os.makedirs(data_dir, exist_ok = True)
    make_template()
    make(range(0x3041, 0x3093 + 1))
    make(range(0x30A1, 0x30F6 + 1))


def make(indices):
    for font in font_list:
        for i_character in indices:
            character = chr(i_character)
            directory = "{0}/{1}_{2}".format(data_dir, i_character, character)
            os.makedirs(directory, exist_ok = True)
            count = 1
            for pointsize in range(120, 248 + 1, 8):
                ds = (image_size - pointsize) // 2
                dr_list = [0]
                if ds > 12:
                    dr_list.extend([-ds // 3, ds // 3])
                for dx in dr_list:
                    for dy in dr_list:
                        for r in [-8, -4, 0, 4, 8]:
                            path = "{}/{}_{}.png".format(directory, character, count)
                            generate(path, font, pointsize, character, r, dx, dy)
                            count += 1


def make_template():
    res = subprocess.call([
        "convert",
        "-size", "{s}x{s}".format(s = image_size),
        "xc:black",
        "{}/tmp.png".format(data_dir)
    ])


def generate(path, font, pointsize, character, rotation, dx, dy):
    res = subprocess.call([
        "convert",
        "-gravity", "Center",
        "-font", font,
        "-pointsize", str(pointsize),
        "-fill", "White",
        "-annotate", format_t(rotation, dx, dy), character,
        "{}/tmp.png".format(data_dir), path
    ])


def format_t(rotation, x, y):
    xstr = "+" + str(x) if x >= 0 else str(x)
    ystr = "+" + str(y) if y >= 0 else str(y)
    return "{r}x{r}{x}{y}".format(r = rotation, x = xstr, y = ystr)


if __name__ == '__main__':
    main()
