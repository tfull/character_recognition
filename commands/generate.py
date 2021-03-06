import os
import subprocess
import yaml
import sys

import _pre
from src.config import Config


font_list = [
    "Noto-Sans-CJK-JP-Thin",
    "Noto-Sans-CJK-JP-Medium",
    "Noto-Serif-CJK-JP"
]


def main():
    os.makedirs(Config.data_directory, exist_ok = True)
    make_template()
    index_data = { "number": 0, "characters": [] }
    make(range(0x3041, 0x3093 + 1), index_data)
    make(range(0x30A1, 0x30F6 + 1), index_data)
    # make(range(0x4E00, 0x9FA0 + 1), index_data)

    with open(Config.data_directory + "/index.yml", "w") as f:
        f.write(yaml.dump(index_data))


def make(indices, index_data):
    for i_character in indices:
        character = chr(i_character)
        directory = "{0}/images/{1}_{2}".format(Config.data_directory, i_character, character)
        os.makedirs(directory, exist_ok = True)
        index_data["characters"].append({ "code": i_character, "character": character })
        count = 0

        for font in font_list:
            for pointsize in range(Config.image_size // 2 - 8, Config.image_size - 8 + 1, 8):
                ds = (Config.image_size - pointsize) // 2
                dr_list = [0]
                if ds >= 16:
                    dr_list.extend([-ds // 3, ds // 3, -ds // 2, ds // 2])
                for dx in dr_list:
                    for dy in dr_list:
                        for rotation in [0 -4, 4, -8, 8]:
                            count += 1
                            path = "{}/{}_{}.png".format(directory, character, count)
                            sys.stderr.write("\033[2K\033[G{}_{}: {} {} {} {} {}".format(character, count, font, pointsize, rotation, dx, dy))
                            sys.stderr.flush()
                            generate(path, font, pointsize, character, rotation, dx, dy)

        index_data["number"] = count

    sys.stderr.write("\033[2K\033[Gcomplete\n")


def make_template():
    res = subprocess.call([
        "convert",
        "-size", "{s}x{s}".format(s = Config.image_size),
        "xc:black",
        "{}/tmp.png".format(Config.data_directory)
    ])


def generate(path, font, pointsize, character, rotation, dx, dy):
    res = subprocess.call([
        "convert",
        "-gravity", "Center",
        "-font", font,
        "-pointsize", str(pointsize),
        "-fill", "White",
        "-annotate", format_t(rotation, dx, dy), character,
        "{}/tmp.png".format(Config.data_directory), path
    ])


def format_t(rotation, x, y):
    xstr = "+" + str(x) if x >= 0 else str(x)
    ystr = "+" + str(y) if y >= 0 else str(y)
    return "{r}x{r}{x}{y}".format(r = rotation, x = xstr, y = ystr)


if __name__ == '__main__':
    main()
