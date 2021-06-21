import os
import numpy as np
import json
import PIL
from PIL import Image
from skimage import measure


def getcentroide(label, size=1):
    centerlabel = np.zeros(label.shape)

    blobs_image = measure.label(label, background=0)
    blobs = measure.regionprops(blobs_image)

    for blob in blobs:
        r, c = blob.centroid
        r, c = int(r), int(c)
        if (
            r <= size + 1
            or r + size + 1 >= label.shape[0]
            or c <= size + 1
            or c + size + 1 >= label.shape[1]
        ):
            continue

        centerlabel[r - size : r + size + 1, c - size : c + size + 1] = 255

    return centerlabel


def resizefile(root, XY, output, nativeresolution, outputresolution=30.0):
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


def resizeram(XY, output, nativeresolution, outputresolution=30.0):
    i = 0
    for name in XY:
        x, y = XY[name]
        image = PIL.Image.fromarray(np.uint8(x))
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

        tmp = np.asarray(label)
        tmp = getcentroide(tmp)
        label = PIL.Image.fromarray(np.uint8(tmp))

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")
        i += 1


whereIam = os.uname()[1]
if whereIam in ["super", "wdtim719z"]:
    availabledata = ["isprs", "dfc"]
    root = "/data/"

if whereIam == "ldtis706z":
    availabledata = ["isprs", "dfc", "vedai", "xview"]
    root = "/media/achanhon/bigdata/data/"

rootminiworld = root + "/CIA/"


def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")


if "isprs" in availabledata:
    print("export isprs")
    makepath("isprs")

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

        resizeram(XY, rootminiworld + "isprs/" + flag, 5)

if "dfc" in availabledata:
    print("export dfc 2015")
    makepath("dfc")

    names = {}
    names["train"] = ["315130_56865", "315130_56870", "315135_56870", "315140_56865"]
    names["test"] = ["315135_56865", "315145_56865"]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = (
                PIL.Image.open(root + "DFC2015/" + "BE_ORTHO_27032011_" + name + ".tif")
                .convert("RGB")
                .copy()
            )
            y = (
                PIL.Image.open(root + "DFC2015/" + "label_" + name + ".tif")
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

        resizeram(XY, rootminiworld + "dfc/" + flag, 5)


def intfixedlength(i):
    j = str(int(i))
    while len(j) < 8:
        j = "0" + j
    return j


if "vedai" in availabledata:
    print("export vedai")
    nativeresolution = 12.5
    outputresolution = 30.0
    size = 1
    makepath("vedai")

    trainingRadix = np.loadtxt(root + "VEDAI/INDEX/fold01.txt")
    testingRadix = np.loadtxt(root + "VEDAI/INDEX/fold01test.txt")

    todo = [("train", trainingRadix), ("test", testingRadix)]
    for flag, alldata in todo:
        for i in range(alldata.shape[0]):
            x = (
                PIL.Image.open(
                    root
                    + "VEDAI/Vehicules1024/"
                    + intfixedlength(alldata[i])
                    + "_co.png"
                )
                .convert("RGB")
                .copy()
            )
            if nativeresolution != outputresolution:
                x = x.resize(
                    (
                        int(x.size[0] * nativeresolution / outputresolution),
                        int(x.size[1] * nativeresolution / outputresolution),
                    ),
                    PIL.Image.BILINEAR,
                )

            x = np.asarray(x)
            y = np.zeros((x.shape[0], x.shape[1]))

            annotation = np.loadtxt(
                root + "VEDAI/Annotations1024/" + intfixedlength(alldata[i]) + ".txt",
                delimiter=" ",
                ndmin=2,
            )
            if annotation.shape[0] > 0 and annotation.shape[1] > 3:
                for j in range(annotation.shape[0]):
                    if int(annotation[j][3]) == 1:
                        r, c = annotation[j, 0:2]

                        r = int(r * nativeresolution / outputresolution)
                        c = int(c * nativeresolution / outputresolution)
                        if (
                            r <= size + 1
                            or r + size + 1 >= label.shape[0]
                            or c <= size + 1
                            or c + size + 1 >= label.shape[1]
                        ):
                            continue

                        y[r - size : r + size + 1, c - size : c + size + 1] = 255

            image = PIL.Image.fromarray(np.uint8(x))
            label = PIL.Image.fromarray(np.uint8(y))
            image.save(rootminiworld + "vedai/" + flag + "/" + str(i) + "_x.png")
            label.save(rootminiworld + "vedai/" + flag + "/" + str(i) + "_y.png")

quit()


import rasterio

if "xview" in availabledata:
    print("export xview")
    makepath("xview")

    imagesname = os.listdir("/data/XVIEW1/train_images")
    imagesname = [name for name in imagesname if name[0] != "."]
    imagesname = sorted(imagesname)

    testimage = set(imagesname[len(imagesname) * 2 // 3 : len(imagesname)])

    imagesname = dict.fromkeys(imagesname, [])

    output = rootminiworld + "xview/"

    with open("/data/XVIEW1/xView_train.geojson", "r") as infile:
        text = json.load(infile)

        text = text["features"]
        for token in text:
            tokenid = token["properties"]["image_id"]
            tokenclass = token["properties"]["type_id"]
            if tokenid in imagesname and int(tokenclass) == 18:
                rect = token["geometry"]["coordinates"]
                rect = np.asarray(rect)
                rect = rect[0]
                center = np.mean(rect, axis=0)

                imagesname[tokenid].append(center)

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
        rgb = np.stack([red, g, b], axis=-1)
        image = PIL.Image.fromarray(np.uint8(rgb))

        if name not in testimage:
            mask.save(output + "train/" + str(i) + "_y.png")
            image.save(output + "train/" + str(i) + "_x.png")
        else:
            mask.save(output + "test/" + str(i - len(imagesname) * 2 // 3) + "_y.png")
            image.save(output + "test/" + str(i - len(imagesname) * 2 // 3) + "_x.png")


print("todo", "saclay ?", "vedai", "dota")
