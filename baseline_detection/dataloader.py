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
    availabledata = ["dfc", "vedai", "saclay", "isprs", "little_xview", "dota"]

    if whereIam in ["super", "wdtim719z"]:
        root = "/data/CIA/"

    if whereIam in ["ldtis706z"]:
        root = "/media/achanhon/bigdata/data/CIA/"

    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        root = "/scratchf/CIA/"

    return root, availabledata


from skimage import measure


def getsomecenters(label):
    blobs_image = measure.label(label, background=0)
    blobs = measure.regionprops(blobs_image)

    output = []
    for blob in blobs:
        r, c = blob.centroid
        r, c = int(r), int(c)
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
    def getrawrandomtiles(self, posperimage, negperimage, tilesize):
        XY = []

        # crop
        for name in range(self.nbImages):
            image, label = self.getImageAndLabel(name)

            # positive crop
            l = getsomecenters(label)
            random.shuffle(l)
            l = l[0 : min(len(l), posperimage)]
            noise = np.random.randint(-tilesize, tilesize, size=(len(l), 2))

            for i, (r, c) in enumerate(l):
                R, C = r + noise[i][0], c + noise[i][1]

                R = max(0, min(R, image.shape[0] - tilesize - 2))
                C = max(0, min(C, image.shape[1] - tilesize - 2))

                im = image[R : R + tilesize, C : C + tilesize, :]
                mask = label[R : R + tilesize, C : C + tilesize]
                XY.append((im.copy(), mask.copy()))

            # random crop
            R = np.random.randint(0, image.shape[0] - tilesize - 2, size=negperimage)
            C = np.random.randint(0, image.shape[1] - tilesize - 2, size=negperimage)
            for i in range(negperimage):
                im = image[R[i] : R[i] + tilesize, C[i] : C[i] + tilesize, :].copy()
                mask = label[R[i] : R[i] + tilesize, C[i] : C[i] + tilesize].copy()
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

    def getrandomtiles(self, batchsize, tilesize=128):
        nbtiles = {}
        nbtiles["vedai"] = (1, 5)  # very small image

        nbtiles["isprs"] = (20, 100)  # small image
        nbtiles["dfc"] = (20, 100)

        nbtiles["dota"] = (5, 25)  # small image but large dataset

        nbtiles["saclay"] = (100, 500)  # medium image

        nbtiles["little_xview"] = (100, 100)  # medium image with many image with no car

        tmp = {}
        for key in nbtiles:
            tmp[key + "/train"] = nbtiles[key]
            tmp[key + "/test"] = nbtiles[key]
        nbtiles = tmp.copy()
        del tmp

        XY = []
        for town in self.towns:
            XY += self.data[town].getrawrandomtiles(
                nbtiles[town][0], nbtiles[town][1], tilesize
            )

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


def largeforward(net, image, device="cuda", tilesize=128, stride=64):
    net.eval()
    with torch.no_grad():
        pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).to(device)
        image = image.float().to(device)
        for row in range(0, image.shape[2] - tilesize + 1, stride):
            for col in range(0, image.shape[3] - tilesize + 1, stride):
                tmp = net(image[:, :, row : row + tilesize, col : col + tilesize])
                pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]

    return pred


def distancetransform(y, size=6):
    yy = y.unsqueeze(0).float()
    yyy = yy.clone()
    for i in range(size):
        yy = torch.nn.functional.max_pool2d(yy, kernel_size=3, stride=1, padding=1)
        yyy += yy
    yyy /= size

    D = (yyy - y.unsqueeze(0)).abs()
    D = torch.exp(-D)

    return D[0]


class SoftNMS(torch.nn.Module):
    def __init__(self):
        super(SoftNMS, self).__init__()

        self.conv = torch.nn.Conv2d(1, 6, kernel_size=7, padding=3)
        self.merge = torch.nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x = x[:, 1, :, :] - x[:, 0, :, :]
        x = x.unsqueeze(1)

        learnednms = torch.nn.functional.leaky_relu(self.conv(x))

        x5 = torch.nn.functional.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        expertnms = torch.nn.functional.relu(x * 10 - 9 * x5)

        x = torch.cat([x, learnednms, expertnms], dim=1)
        x += self.merge(x)
        x = torch.cat([-x, x], dim=1)
        return x
