import os

import csv
import numpy as np
import PIL
from PIL import Image


def getcsvlines(path, delimiter=" "):
    text = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            text.append(row)
    return text


output = "build/"
outputresolution = 0.3
size = 1

imagesname = os.listdir("/data/DOTA/images")
imagesname = [name[0:-4] for name in imagesname]
imagesname = sorted(imagesname)

RAHHH = 0
for name in imagesname:
    vt = getcsvlines("/data/DOTA/labelTxt-v1.0/labelTxt/" + name + ".txt")

    if ("gsd" not in vt[1][0]) or ("null" in vt[1][0]):
        print("no gsd in", name)
        continue

    resolution = float(vt[1][0][4 : len(vt[1][0])])
    if resolution > 0.3 or resolution < 0.01:
        continue

    centers = []
    for i in range(2, len(vt)):
        if vt[i][-2] == "small-vehicle":
            vertices = []
            for j in range(0, len(vt[i]) - 2, 2):
                if vt[i][j] != "null" and vt[i][j + 1] != "null":
                    vertices.append((float(vt[i][j]), float(vt[i][j + 1])))

            if vertices != []:
                vertices = np.asarray(vertices)
                center = np.mean(vertices, axis=0)
                centers.append((center[0], center[1]))

    if centers == []:
        continue

    print(name, RAHHH)  # to known the correspondance to be able to avoid bad ones
    x = PIL.Image.open("/data/DOTA/images/" + name + ".png").convert("RGB").copy()
    x = x.resize(
        (
            int(x.size[0] * resolution / outputresolution),
            int(x.size[1] * resolution / outputresolution),
        ),
        PIL.Image.BILINEAR,
    )
    x = np.asarray(x)

    y = np.zeros((x.shape[0], x.shape[1]))
    for c, r in centers:
        r, c = int(r * resolution / outputresolution), int(
            c * resolution / outputresolution
        )
        if (
            r <= size + 1
            or r + size + 1 >= x.shape[0]
            or c <= size + 1
            or c + size + 1 >= x.shape[1]
        ):
            continue

        y[r - size : r + size + 1, c - size : c + size + 1] = 255

    mask = PIL.Image.fromarray(np.uint8(y))
    mask.save(output + str(RAHHH) + "_y.png")

    image = PIL.Image.fromarray(np.uint8(x))
    image.save(output + "/" + str(RAHHH) + "_x.png")

    RAHHH += 1
