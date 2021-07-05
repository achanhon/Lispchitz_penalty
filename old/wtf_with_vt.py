import os
import numpy as np
import PIL
from PIL import Image

import json

def inden(n):
    out="  "
    for i in range(n):
        out+="  "
    return out

def getjsonstructure(text,level=0):
    if isinstance(text,str):
        print(inden(level),text)
        return
        
    if isinstance(text,dict):
        for token in text:
            print(inden(level),token)
            getjsonstructure(text[token],level+1)
        return
        
    if isinstance(text,list):
        for token in text:
            print(inden(level),token)
            return
        
            



imagesname = os.listdir("/data/XVIEW1/train_images")
imagesname = sorted(imagesname)
imagesname = [name for name in imagesname if name[0] != "."]
imagesname = dict.fromkeys(imagesname, [])

output = "build/"

with open("/data/XVIEW1/xView_train.geojson", "r") as infile:
    text = json.load(infile)
    
    getjsonstructure(text)
    quit()
    
    #crs
    #    property
    #    type
    #type
    #features

    for token in text["crs"]:
        print(token)
    

    
    for section in text:
        print(section)
        
        for token in text[section]:
            if "1346.tif" in token:
                print(token)
            
            continue
            tokenid = token["properties"]["image_id"]
            tokenclass = token["properties"]["type_id"]
            if tokenid in imagesname and int(tokenclass) == 18:
                rect = token["geometry"]["coordinates"]
                rect = np.asarray(rect)
                rect = rect[0]
                center = np.mean(rect, axis=0)

                imagesname[tokenid].append(center)

import rasterio

size = 1

for i, name in enumerate(imagesname):
    with rasterio.open("/data/XVIEW1/train_images/" + name) as src:
        affine = src.transform
        red = np.int16(src.read(1))
        g = np.int16(src.read(2))
        b = np.int16(src.read(3))

    mask = np.zeros((red.shape[0], red.shape[1]))
    centers = imagesname[name]

    for center in centers:
        r, c = rasterio.transform.rowcol(affine, center[0], center[1])
        r, c = int(r), int(c)
        if (
            r <= size + 1
            or r + size + 1 >= mask.shape[0]
            or c <= size + 1
            or c + size + 1 >= mask.shape[1]
        ):
            continue

        mask[r - size : r + size + 1, c - size : c + size + 1] = 255

    mask = PIL.Image.fromarray(np.uint8(mask))
    mask.save(output + str(i) + "_y.png")

    rgb = np.stack([red, g, b], axis=-1)
    image = PIL.Image.fromarray(np.uint8(rgb))
    image.save(output + "/" + str(i) + "_x.png")
