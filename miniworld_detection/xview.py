import json
import numpy as np

import PIL
from PIL import Image

with open("/data/xview/xView_train.geojson", "r") as infile:
    text = json.load(infile)

    text = text["features"]
    for token in text:
        
        tokenid = token["properties"]["image_id"]
        rect = token["geometry"]
        for item in token:
            print(item)
        quit()
