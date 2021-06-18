import os

import json
import numpy as np

import PIL
from PIL import Image

imagesname = os.listdir("/data/XView1/train_images")
imagesname = dict.fromkeys(imagesname, [])

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
