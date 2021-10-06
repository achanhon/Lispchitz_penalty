import numpy as np
import os


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

    availabledata = ["dfc", "vedai", "saclay", "isprs", "little_xview", "dota"]

    if whereIam in ["super", "wdtim719z"]:
        root = "/data/CIA/"
    if whereIam in ["ldtis706z"]:
        root = "/media/achanhon/bigdata/data/CIA/"
    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        root = "/scratchf/CIA/"

    return root, availabledata


def computeperf(cm):
    if len(cm.shape) == 1:
        good, fa, miss = cm[0], cm[1], cm[2]
        if good == 0:
            precision = 0
            recall = 0
        else:
            precision = good / (good + fa + hardfa)
            recall = good / (good + miss + hardmiss)
        return torch.Tensor([precision * recall, precision, recall])
    else:
        out = torch.zeros(cm.shape[0], 3)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


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

        image = PIL.Image.open(self.pathTOdata + str(i) + "_x.png").convert("RGB")
        image = np.uint8(np.asarray(image).copy())  # warning wh swapping

        label = PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L")
        label = np.asarray(label).copy()  # warning wh swapping

        ###HACK degeu parce que dans le dataset c'est du 3x3
        label = -torch.nn.functional.max_pool2d(
            -torch.Tensor(label).unsqueeze(0), kernel_size=3, stride=1, padding=1
        )

        label = np.uint8(label[0].cpu().numpy() != 0)

        return image, label

    ###
    ### get randomcrops + symetrie
    ### get train usage
    def getrawrandomtiles(self, nbtiles, tilesize, nbpos=0):
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
                im = image[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize, :]
                mask = label[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize]
                XY.append((im.copy(), mask.copy()))

            if nbpos != 0 and np.sum(label) > 1:
                row, col = np.nonzero(label)
                l = [(row[i], col[i]) for i in range(row.shape[0])]
                random.shuffle(l)
                l = l[0 : min(len(l), nbpos)]
                noise = np.random.randint(-tilesize, tilesize, size=(len(l), 2))

                for i, (r, c) in enumerate(l):
                    R, C = r + noise[i][0], c + noise[i][1]

                    R = max(0, min(R, image.shape[0] - tilesize - 2))
                    C = max(0, min(C, image.shape[1] - tilesize - 2))

                    im = image[R : R + tilesize, C : C + tilesize, :]
                    mask = label[R : R + tilesize, C : C + tilesize]
                    XY.append((im.copy(), mask.copy()))

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

    def getrandomtiles(self, tilesize, batchsize):
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
                nbtiles[town][1], tilesize, nbpos=nbtiles[town][0]
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
