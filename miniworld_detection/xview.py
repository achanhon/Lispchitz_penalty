import os

import json
import numpy as np

import PIL
from PIL import Image

imagesname = os.listdir("/data/XView1/train_images")
imagesname = dict.fromkeys(imagesname, [])

output = "build/"

with open("/data/XView1/xView_train.geojson", "r") as infile:
    text = json.load(infile)

    text = text["features"]
    for token in text:
        tokenid = token["properties"]["image_id"]
        if tokenid in imagesname:
            rect = token["geometry"]["coordinates"]
            rect = np.asarray(rect)
            rect = rect[0]
            assert rect.shape == (5, 2)
            center = np.mean(rect, axis=0)

            imagesname[tokenid].append(center)

for i, name in enumerate(imagesname):
    with rasterio.open("/data/XView1/train_images/" + name) as src:
        affine = src.transform
        r = np.int16(src.read(1))
        g = np.int16(src.read(2))
        b = np.int16(src.read(3))

    mask = np.zeros(r.shape[1], r.shape[0])
    centers = imagesname[name]

    for center in centers:
        r, c = rasterio.transform.rowcol(affine, center[0], center[1])
        r, c = int(r), int(c)
        if (
            r <= size + 1
            or r + size + 1 >= label.shape[0]
            or c <= size + 1
            or c + size + 1 >= label.shape[1]
        ):
            continue

        mask[r - size : r + size + 1, c - size : c + size + 1] = 255

    mask.save(output + str(i) + "_y.png")

    rgb = numpy.stack([r, g, b], axis=-1)
    image = PIL.Image.fromarray(np.uint8(rgb))
    image.save(output + "/" + str(i) + "_x.png")
