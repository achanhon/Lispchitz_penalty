import numpy as np


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


import os
import PIL
from PIL import Image

import torch
import random


def getindexeddata():
    whereIam = os.uname()[1]

    root = None
    availabledata = ["dfc", "vedai", "saclay", "isprs", "little_xview", "dota"]

    if whereIam in ["super", "wdtim719z"]:
        root = "/data/CIA/"

    if whereIam in ["ldtis706z"]:
        root = "/media/achanhon/bigdata/data/CIA/"

    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        root = "/scratchf/CIA/"

    return root, availabledata


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
            label = np.asarray(label, dtype=np.uint8)  # warning wh swapping
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
        image = np.asarray(image, dtype=np.uint8)  # warning wh swapping

        label = PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
        label = np.asarray(label, dtype=np.uint8)  # warning wh swapping
        label = np.uint8(label != 0)
        return image, label

    ###
    ### get randomcrops + symetrie
    ### get train usage
    def getrawrandomtiles(self, nbtiles, tilesize):
        XY = []
        nbtilesperimage = int(nbtiles / self.nbImages + 1)

        # crop
        for name in range(self.nbImages):
            # nbtilesperimage*probaOnImage==nbtiles
            if (
                nbtiles < self.nbImages
                and random.randint(0, int(nbtiles + 1)) > nbtilesperimage
            ):
                continue

            image, label = self.getImageAndLabel(name)

            row = np.random.randint(
                0, image.shape[0] - tilesize - 2, size=nbtilesperimage
            )
            col = np.random.randint(
                0, image.shape[1] - tilesize - 2, size=nbtilesperimage
            )

            for i in range(nbtilesperimage):
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
        for town in self.towns:
            self.data[town] = SegSemDataset(self.root + town + "/")
            self.nbImages += self.data[town].nbImages

        print(
            "indexing cia (mode",
            flag,
            "):",
            len(self.towns),
            "towns found (",
            self.towns,
            ") with a total of",
            self.nbImages,
            "images",
        )

    def getrandomtiles(self, nbtiles, tilesize, batchsize):
        XY = []
        nbtilesperTown = 1.0 * nbtiles / len(self.towns)

        for town in self.towns:
            XY += self.data[town].getrawrandomtiles(nbtilesperTown, tilesize)

        # pytorch
        X = torch.stack(
            [torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu() for x, y in XY]
        )
        Y = torch.stack([torch.from_numpy(y).cpu() for x, y in XY])
        Y = Y.unsqueeze(0).float()
        Y = torch.nn.functional.max_pool2d(Y, kernel_size=9, stride=1, padding=4)
        Y = Y[0].long()

        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, shuffle=True, num_workers=2
        )

        return dataloader


def largeforward(net, image, device="cuda", tilesize=128, stride=64):
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


def distanceToBorder(y, size=4):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]
