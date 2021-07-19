import numpy as np
import os


def safeuint8(x):
    x0 = np.zeros(x.shape, dtype=float)
    x255 = np.ones(x.shape, dtype=float) * 255
    x = np.maximum(x0, np.minimum(x.copy(), x255))
    return np.uint8(x)


def symetrie(x, y, i, j, k):
    if i == 1:
        x, y = np.transpose(x, axes=(1, 0, 2)), np.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    if k == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    return x.copy(), y.copy()


def getindexeddata():
    whereIam = os.uname()[1]

    root = None
    availabledata = ["dfc", "vedai", "saclay", "little_xview", "dota", "isprs"]

    if whereIam in ["super", "wdtim719z"]:
        root = "/data/CIA/"

    if whereIam in ["ldtis706z"]:
        root = "/media/achanhon/bigdata/data/CIA/"

    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        root = "/scratchf/CIA/"

    return root, availabledata


from skimage import measure


def getsomecenters(label):
    centerlabel = np.zeros(label.shape)

    blobs_image = measure.label(label, background=0)
    blobs = measure.regionprops(blobs_image)

    output = []
    for blob in blobs:
        r, c = blob.centroid
        r, c = int(r), int(c)
        if r <= 8 or r + 8 >= label.shape[0] or c <= 8 or c + 8 >= label.shape[1]:
            continue

        output.append((r, c))

    return output


import PIL
from PIL import Image

import torch
import random


class SegSemDataset:
    def __init__(self, pathTOdata):
        self.pathTOdata = pathTOdata

        self.nbImages = 0
        while os.path.exists(
            self.pathTOdata + str(self.nbImages) + "_x.png"
        ) and os.path.exists(self.pathTOdata + str(self.nbImages) + "_y.png"):
            self.nbImages += 1
        if self.nbImages == 0:
            print("wrong path", self.pathTOdata)
            quit()

        self.nbbat, self.nbnonbat = 0, 0
        for i in range(self.nbImages):
            label = (
                PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
            )
            label = np.uint8(np.asarray(label))  # warning wh swapping
            label = np.uint8(label != 0)
            self.nbbat += np.sum((label == 1).astype(int))
            self.nbnonbat += np.sum((label == 0).astype(int))

        self.balance = self.nbnonbat / self.nbbat

    ###
    ### get the hole image
    ### for test usage -- or in internal call for extracting crops
    def getImageAndLabel(self, i):
        assert i < self.nbImages

        image = (
            PIL.Image.open(self.pathTOdata + str(i) + "_x.png").convert("RGB").copy()
        )
        image = np.uint8(np.asarray(image))  # warning wh swapping

        label = PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
        label = np.uint8(np.asarray(label))  # warning wh swapping
        label = np.uint8(label != 0)
        return image, label

    ###
    ### get randomcrops + symetrie
    ### get train usage
    def getrawrandomtiles(
        self, nbtilespositifperimage, nbtilesnegatifperimage, tilesize
    ):
        XY = []

        # crop
        for name in range(self.nbImages):
            image, label = self.getImageAndLabel(name)

            # positive crop
            l = getsomecenters(label)
            random.shuffle(l)
            l = l[0 : min(len(l), nbtilespositifperimage)]
            random01 = np.random.rand(len(l), 2)
            for i, (r, c) in enumerate(l):
                # computing possible value for left corner of the crop containing r,c
                maxR = r - 1
                maxC = c - 1
                # always possible because r,c can not be the left corner of the image
                minR = max(r - tilesize + 1, 0)
                minC = max(c - tilesize + 1, 0)
                # should not be outside the image so max(0,...)

                R = int(random01[i][0] * (maxR - minR)) + minR
                C = int(random01[i][0] * (maxC - minC)) + minC

                im = image[R : R + tilesize, C : C + tilesize, :]
                mask = label[R : R + tilesize, C : C + tilesize]
                XY.append((im.copy(), mask.copy()))

            # random crop
            row = np.random.randint(
                0, image.shape[0] - tilesize - 2, size=nbtilesnegatifperimage
            )
            col = np.random.randint(
                0, image.shape[1] - tilesize - 2, size=nbtilesnegatifperimage
            )

            for i in range(nbtilesnegatifperimage):
                im = image[
                    row[i] : row[i] + tilesize, col[i] : col[i] + tilesize, :
                ].copy()
                mask = label[
                    row[i] : row[i] + tilesize, col[i] : col[i] + tilesize
                ].copy()
                XY.append((im, mask))

        # symetrie
        symetrieflag = np.random.randint(0, 2, size=(len(XY), 3))
        XY = [
            (symetrie(x, y, symetrieflag[i][0], symetrieflag[i][1], symetrieflag[i][2]))
            for i, (x, y) in enumerate(XY)
        ]
        return XY


class CIA:
    def __init__(self, flag, custom=None):
        assert flag in ["train", "test", "custom"]

        self.root, self.towns = getindexeddata()
        if flag == "custom":
            self.towns = custom
        else:
            if flag == "train":
                self.towns = [town + "/train" for town in self.towns if town != "isprs"]
            else:
                self.towns = [town + "/test" for town in self.towns] + ["isprs/train"]

        self.data = {}
        self.nbImages = 0
        self.nbbat, self.nbnonbat = 0, 0
        for town in self.towns:
            self.data[town] = SegSemDataset(self.root + town + "/")
            self.nbImages += self.data[town].nbImages
            self.nbbat += self.data[town].nbbat
            self.nbnonbat += self.data[town].nbnonbat

        self.balance = self.nbnonbat / self.nbbat
        print(
            "indexing cia (mode",
            flag,
            "):",
            len(self.towns),
            "towns found (",
            self.towns,
            ") with a total of",
            self.nbImages,
            "images and a balance of",
            self.balance,
        )

    def getrandomtiles(self, tilesize, batchsize):
        nbtiles = {}
        nbtiles["vedai/train"] = (1, 5)  # very small image

        nbtiles["isprs/train"] = (20, 100)  # small image
        nbtiles["dfc/train"] = (20, 100)

        nbtiles["dota/train"] = (5, 25)  # small image but large dataset

        nbtiles["saclay/train"] = (100, 500)  # medium image

        nbtiles["little_xview/train"] = (
            100,
            100,
        )  # medium image with many image with no car

        XY = []
        for town in self.towns:
            XY += self.data[town].getrawrandomtiles(
                nbtiles[town][0], nbtiles[town][1], tilesize
            )
        # print(len(XY)) ~ 25000

        # pytorch
        X = torch.stack(
            [torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu() for x, y in XY]
        )
        Y = torch.stack([torch.from_numpy(y).long().cpu() for x, y in XY])
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, shuffle=True, num_workers=2
        )

        return dataloader


def largeforward(net, image, device, tilesize=128, stride=64):
    ## assume GPU is large enough -- use largeforwardCPU on very large image
    net.eval()
    with torch.no_grad():
        pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).to(device)
        image = image.float().to(device)
        i = 0
        for row in range(0, image.shape[2] - tilesize + 1, stride):
            for col in range(0, image.shape[3] - tilesize + 1, stride):
                tmp = net(image[:, :, row : row + tilesize, col : col + tilesize])
                pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]

                # if i % 500 == 499:
                #    print("forward in progress", row, image.shape[2])
                i += 1

    return pred


def largeforwardCPU(net, image, device, tilesize=128, stride=32):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cpu()

    net.eval()
    with torch.no_grad():
        i = 0
        for row in range(0, image.shape[2] - tilesize + 1, stride):
            for col in range(0, image.shape[3] - tilesize + 1, stride):
                tmp = net(
                    image[:, :, row : row + tilesize, col : col + tilesize]
                    .float()
                    .to(device)
                ).cpu()
                pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]

                if i % 500 == 499:
                    print("forward in progress", row, image.shape[2])
                i += 1

    return pred


def convertIn3class(y, size=2):
    yy = torch.nn.functional.max_pool2d(
        y.float(), kernel_size=size * 2 + 1, stride=1, padding=size
    )
    yy = yy * 2 - y.float()
    return yy.long()


def convertIn3classNP(y):
    yy = convertIn3class(torch.Tensor(y).cuda().unsqueeze(0))
    return np.uint8(yy[0].cpu().numpy())
