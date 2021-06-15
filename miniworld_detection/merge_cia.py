import os
import numpy as np
import json
import PIL
from PIL import Image
from skimage import measure


def getcentroide(label):
    return label
    centerlabel = np.zeros(label.shape)

    blobs_image = measure.label(label, background=0)
    blobs = measure.regionprops(blobs_image)

    for blob in blobs:
        r, c = blob.centroid
        r, c = int(r), int(c)
        centerlabel[r][c] = 255

    return centerlabel


def resizefile(root, XY, output, nativeresolution, outputresolution=25.0):
    i = 0
    for name in XY:
        x, y = XY[name]
        image = PIL.Image.open(root + "/" + x).convert("RGB").copy()
        label = PIL.Image.open(root + "/" + y).convert("L").copy()

        if nativeresolution != outputresolution:
            image = image.resize(
                (
                    int(image.size[0] * nativeresolution / outputresolution),
                    int(image.size[1] * nativeresolution / outputresolution),
                ),
                PIL.Image.BILINEAR,
            )
            label = label.resize((image.size[0], image.size[1]), PIL.Image.NEAREST)

        tmp = np.asarray(label)
        tmp = getcentroide(tmp)
        label = PIL.Image.fromarray(np.uint8(tmp))

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")
        i += 1


def resizeram(XY, output, nativeresolution, outputresolution=25.0):
    i = 0
    for name in XY:
        x, y = XY[name]
        image = PIL.Image.fromarray(np.uint8(x))
        y = getcentroide(y)
        label = PIL.Image.fromarray(np.uint8(y))

        if nativeresolution != outputresolution:
            image = image.resize(
                (
                    int(image.size[0] * nativeresolution / outputresolution),
                    int(image.size[1] * nativeresolution / outputresolution),
                ),
                PIL.Image.BILINEAR,
            )
            label = label.resize((image.size[0], image.size[1]), PIL.Image.NEAREST)

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")
        i += 1


whereIam = os.uname()[1]
if whereIam == "super":
    availabledata = ["isprs", "dfc"]
    root = "/data/"

if whereIam == "ldtis706z":
    availabledata = ["isprs", "dfc"]
    root = "/media/achanhon/bigdata/data/"

rootminiworld = root + "/CIA/"
# if whereIam == "wdtim719z":
#    availabledata = ["semcity", "isprs", "airs", "dfc"]
#    root = "/data/"
#    rootminiworld = root + "/miniworld/"

# if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
#    availabledata = [
#        "semcity",
#        "dfc",
#        "spacenet1",
#        "spacenet2",
#        "isprs",
#        "airs",
#        "inria",
#        "bradbery",
#    ]
#    root = "/scratch_ai4geo/DATASETS/"
#    rootminiworld = "/scratch_ai4geo/miniworld/"


def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")


if "dfc" in availabledata:
    print("export dfc 2015 bruges")
    makepath("bruges")

    names = {}
    names["train"] = ["315130_56865", "315130_56870", "315135_56870", "315140_56865"]
    names["test"] = ["315135_56865", "315145_56865"]

    hack = ""
    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        hack = "../"

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = (
                PIL.Image.open(
                    root + hack + "DFC2015/" + "BE_ORTHO_27032011_" + name + ".tif"
                )
                .convert("RGB")
                .copy()
            )
            y = (
                PIL.Image.open(root + hack + "DFC2015/" + "label_" + name + ".tif")
                .convert("RGB")
                .copy()
            )
            y = np.uint8(np.asarray(y))
            y = (
                np.uint8(y[:, :, 0] == 255)
                * np.uint8(y[:, :, 1] == 255)
                * np.uint8(y[:, :, 2] == 0)
                * 255
            )

            XY[name] = (x, y)

        resizeram(XY, rootminiworld + "bruges/" + flag, 5)

if "isprs" in availabledata:
    print("export isprs potsdam")
    makepath("potsdam")

    names = {}
    names["train"] = [
        "top_potsdam_2_10_",
        "top_potsdam_2_11_",
        "top_potsdam_2_12_",
        "top_potsdam_3_10_",
        "top_potsdam_3_11_",
        "top_potsdam_3_12_",
        "top_potsdam_4_10_",
        "top_potsdam_4_11_",
        "top_potsdam_4_12_",
        "top_potsdam_5_10_",
        "top_potsdam_5_11_",
        "top_potsdam_5_12_",
        "top_potsdam_6_7_",
        "top_potsdam_6_8_",
    ]
    names["test"] = [
        "top_potsdam_6_9_",
        "top_potsdam_6_10_",
        "top_potsdam_6_11_",
        "top_potsdam_6_12_",
        "top_potsdam_7_7_",
        "top_potsdam_7_8_",
        "top_potsdam_7_9_",
        "top_potsdam_7_10_",
        "top_potsdam_7_11_",
        "top_potsdam_7_12_",
    ]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = (
                PIL.Image.open(
                    root + "ISPRS_POTSDAM/" + "2_Ortho_RGB/" + name + "RGB.tif"
                )
                .convert("RGB")
                .copy()
            )
            y = (
                PIL.Image.open(
                    root
                    + "ISPRS_POTSDAM/"
                    + "5_Labels_for_participants/"
                    + name
                    + "label.tif"
                )
                .convert("RGB")
                .copy()
            )
            y = np.uint8(np.asarray(y))
            y = (
                np.uint8(y[:, :, 0] == 255)
                * np.uint8(y[:, :, 1] == 255)
                * np.uint8(y[:, :, 2] == 0)
                * 255
            )

            XY[name] = (x, y)

        resizeram(XY, rootminiworld + "potsdam/" + flag, 5)

print("todo", "saclay ?", "vedai", "xview", "dota")
