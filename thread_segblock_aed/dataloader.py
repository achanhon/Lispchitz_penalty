import os
import threading
import queue
import PIL
from PIL import Image
import numpy
import torch
import random
import csv


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    return x.copy(), y.copy()


def computeperf(yz=None, stats=None):
    assert yz is not None or stats is not None
    if stats is not None:
        good, fa, miss = stats[0], stats[1], stats[2]
        if good == 0:
            precision = 0
            recall = 0
            count = 0
            real = miss
        else:
            precision = good / (good + fa)
            recall = good / (good + miss)
            count = good + fa
            real = good + miss
        return torch.Tensor(
            [precision * recall, precision, recall, (count - real).abs(), count, real]
        ).cuda()
    else:
        y, z = yz
        good = torch.sum((y == 1).float() * (z > 0).float())
        fa = torch.sum((y == 0).float() * (z > 0).float())
        miss = torch.sum((y == 1).float() * (z <= 0).float())
        return torch.Tensor([good, fa, miss]).cuda()


class AED(threading.Thread):
    def __init__(self, flag, maxsize=10000, tilesize=256):
        assert flag in ["train", "test"]

        whereIam = os.uname()[1]
        if whereIam == "ldtis706z":
            self.root = "/media/achanhon/bigdata/data"
        else:
            self.root = "/scratchf"
        if flag == "train":
            pathtovt = self.root + "/AED/training_elephants.csv"
            self.root = self.root + "/AED/training_images/"
        else:
            pathtovt = self.root + "/AED/test_elephants.csv"
            self.root = self.root + "/AED/test_images/"

        print("reading csv file", pathtovt)
        self.labels = {}
        with open(pathtovt, "r") as csvfile:
            text = csv.reader(csvfile, delimiter=",")
            for line in text:
                if line[0] not in self.labels:
                    self.labels[line[0]] = []
                self.labels[line[0]].append((float(line[1]), float(line[2])))

        self.names = list(self.labels.keys())

        threading.Thread.__init__(self)

        if maxsize > 0:
            self.tilesize = tilesize
            self.q = queue.Queue(maxsize=maxsize)
        else:
            self.tilesize = None
        print("data successfully loaded")

    def getImageAndLabel(self, name, torchformat=False):
        assert name in self.labels

        image = PIL.Image.open(self.root + name + ".jpg").convert("RGB")
        image = numpy.uint8(numpy.asarray(image).copy())

        h, w = image.shape[0], image.shape[1]
        h2, w2 = (h // 64) * 64, (w // 64) * 64
        globalresize = torch.nn.AdaptiveAvgPool2d((h2, w2))

        tmp = numpy.transpose(image, axes=(2, 0, 1))
        tmp = torch.Tensor(tmp)
        tmp = globalresize(tmp)
        image = numpy.transpose(tmp.cpu().numpy(), axes=(1, 2, 0))

        mask = numpy.zeros((image.shape[0], image.shape[1]))
        points = self.labels[name]
        for c, r in points:
            r, c = int(r * h2 / h), int(c * w2 / w)
            r, c = max(0, r), max(0, c)
            r, c = min(r, mask.shape[0] - 1), min(c, mask.shape[1] - 1)
            mask[r][c] = 1

        tmp = torch.Tensor(mask).float().unsqueeze(0)
        tmp = torch.nn.functional.max_pool2d(tmp, kernel_size=16, stride=16, padding=0)
        mask = tmp[0].cpu().numpy()

        assert image.shape[2] == 3 and image.shape[0] == h2 and image.shape[1] == w2
        assert mask.shape[0] == h2 // 16 and mask.shape[1] == w2 // 16

        if torchformat:
            x = torch.Tensor(numpy.transpose(image, axes=(2, 0, 1)))
            return x.unsqueeze(0), torch.Tensor(mask)
        else:
            return image, mask

    def getbatch(self, batchsize=32):
        X = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        Y = torch.zeros(batchsize, self.tilesize // 16, self.tilesize // 16)
        for i in range(batchsize):
            x, y = self.q.get(block=True)
            X[i] = x
            Y[i] = y
        return X, Y.long()

    def run(self):
        NB, nbpos = 10, 2
        tilesize = self.tilesize
        tile16 = tilesize // 16

        print("start collecting tiles")
        while True:
            for name in self.names:
                XY = []
                image, label = self.getImageAndLabel(name, torchformat=False)

                # random crop
                row = numpy.random.randint(0, label.shape[0] - tile16 - 2, size=NB)
                col = numpy.random.randint(0, label.shape[1] - tile16 - 2, size=NB)

                for i in range(NB):
                    im = image[
                        row[i] * 16 : row[i] * 16 + tilesize,
                        col[i] * 16 : col[i] * 16 + tilesize,
                        :,
                    ]
                    mask = label[row[i] : row[i] + tile16, col[i] : col[i] + tile16]
                    XY.append((im.copy(), mask.copy()))

                # positive crop
                if nbpos != 0 and numpy.sum(label) > 1:
                    row, col = numpy.nonzero(label)
                    l = [(row[i], col[i]) for i in range(row.shape[0])]
                    random.shuffle(l)
                    l = l[0 : min(len(l), nbpos)]
                    noise = numpy.random.randint(-tile16, tile16, size=(len(l), 2))

                    for i, (r, c) in enumerate(l):
                        R, C = r + noise[i][0], c + noise[i][1]

                        R = max(0, min(R, label.shape[0] - tile16 - 2))
                        C = max(0, min(C, label.shape[1] - tile16 - 2))

                        im = image[
                            R * 16 : R * 16 + tilesize, C * 16 : C * 16 + tilesize, :
                        ]
                        mask = label[R : R + tile16, C : C + tile16]
                        XY.append((im.copy(), mask.copy()))

                # symetrie
                symetrieflag = numpy.random.randint(0, 2, size=(len(XY), 3))
                XY = [(symetrie(x, y, symetrieflag[i])) for i, (x, y) in enumerate(XY)]

                # pytorch
                for x, y in XY:
                    x = torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))
                    y = torch.Tensor(y)
                    self.q.put((x, y), block=True)


import torchvision
import math

if __name__ == "__main__":
    aed = AED(flag="test")

    i = 0
    while aed.labels[aed.names[i]] == []:
        i += 1

    image, mask = aed.getImageAndLabel(aed.names[i], torchformat=True)
    debug = image[0].cpu().numpy()
    debug = numpy.transpose(debug, axes=(1, 2, 0))
    debug = PIL.Image.fromarray(numpy.uint8(debug))
    debug.save("build/image.png")
    globalresize = torch.nn.AdaptiveAvgPool2d((image.shape[2], image.shape[3]))
    debug = globalresize(mask.unsqueeze(0).float())
    debug = debug[0].cpu().numpy() * 255
    debug = PIL.Image.fromarray(numpy.uint8(debug))
    debug.save("build/label.png")

    aed.start()
    x, y = aed.getbatch(batchsize=16)

    torchvision.utils.save_image(x / 255, "build/cropimage.png")
    globalresize = torch.nn.AdaptiveAvgPool2d((x.shape[2], x.shape[3]))
    y = globalresize(y.float())
    torchvision.utils.save_image(y.unsqueeze(1), "build/croplabel.png")

    os._exit(0)
