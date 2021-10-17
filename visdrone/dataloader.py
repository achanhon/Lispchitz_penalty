import numpy
import os
import threading
import queue
import sys
import PIL
from PIL import Image
import torch
import random
import scipy
import scipy.io


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    return x.copy(), y.copy()


def computeperf(stats):
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


class VISDRONE(threading.Thread):
    def __init__(self, flag, maxsize=10000, tilesize=256):
        assert flag in ["train", "test"]
        if flag == "train":
            self.root = "/scratchf/VisDrone/train_data/"
        else:
            self.root = "/scratchf/VisDrone/test_data/"

        tmp = os.listdir(self.root + "/ground_truth/")
        tmp = [s for s in tmp if "GT_" in s and ".mat" in s]
        self.names = [s[3:-4] for s in tmp]

        threading.Thread.__init__(self)

        if maxsize > 0:
            self.tilesize = tilesize
            self.q = queue.Queue(maxsize=maxsize)
        else:
            self.tilesize = None

    def getImageAndLabel(self, name, torchformat=False):
        assert name in self.names

        image = PIL.Image.open(self.root + "images/" + name + ".jpg").convert("RGB")
        image = numpy.uint8(numpy.asarray(image).copy())

        mask = numpy.zeros((image.shape[0], image.shape[1]))
        points = scipy.io.loadmat(self.root + "ground_truth/GT_" + name + ".mat")
        points = points["image_info"][0][0][0][0][0]
        if len(points.shape) == 2 and points.shape[1] == 3:
            I = points.shape[0]
            for i in range(I):
                mask[int(points[i][1])][int(points[i][0])] = 1

        if torchformat:
            x = torch.Tensor(numpy.transpose(image, axes=(2, 0, 1)))
            return x, torch.Tensor(mask)
        else:
            return image, mask

    def getbatch(self, batchsize=32):
        X = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        Y = torch.zeros(batchsize, self.tilesize, self.tilesize)
        for i in range(batchsize):
            x, y = self.q.get(block=True)
            X[i] = x
            Y[i] = y
        return X, Y.long()

    def run(self):
        NB, nbpos = 10, 2
        tilesize = self.tilesize

        print("start collecting tiles")
        while True:
            for name in self.names:
                XY = []
                image, label = self.getImageAndLabel(name, torchformat=False)

                # random crop
                row = numpy.random.randint(0, image.shape[0] - tilesize - 2, size=NB)
                col = numpy.random.randint(0, image.shape[1] - tilesize - 2, size=NB)

                for i in range(NB):
                    im = image[
                        row[i] : row[i] + tilesize, col[i] : col[i] + tilesize, :
                    ]
                    mask = label[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize]
                    XY.append((im.copy(), mask.copy()))

                # positive crop
                if nbpos != 0 and numpy.sum(label) > 1:
                    row, col = numpy.nonzero(label)
                    l = [(row[i], col[i]) for i in range(row.shape[0])]
                    random.shuffle(l)
                    l = l[0 : min(len(l), nbpos)]
                    noise = numpy.random.randint(-tilesize, tilesize, size=(len(l), 2))

                    for i, (r, c) in enumerate(l):
                        R, C = r + noise[i][0], c + noise[i][1]

                        R = max(0, min(R, image.shape[0] - tilesize - 2))
                        C = max(0, min(C, image.shape[1] - tilesize - 2))

                        im = image[R : R + tilesize, C : C + tilesize, :]
                        mask = label[R : R + tilesize, C : C + tilesize]
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
    visdrone = VISDRONE(flag="test")

    image, mask = visdrone.getImageAndLabel(visdrone.names[0], torchformat=True)
    debug = image[0].cpu().numpy()
    debug = numpy.transpose(debug, axes=(1, 2, 0))
    debug = PIL.Image.fromarray(numpy.uint8(debug))
    debug.save("build/image.png")
    debug = mask.cpu().numpy()
    debug = PIL.Image.fromarray(numpy.uint8(debug) * 255)
    debug.save("build/label.png")

    visdrone.start()
    x, y = visdrone.getbatch(batchsize=16)

    torchvision.utils.save_image(x / 255, "build/cropimage.png")
    torchvision.utils.save_image(y.unsqueeze(1).float(), "build/croplabel.png")

    os._exit(0)
