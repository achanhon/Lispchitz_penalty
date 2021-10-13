import numpy as np
import os
import sys
import PIL
from PIL import Image
import torch
import random
import csv


def symetrie(x, y, i, j, k):
    if i == 1:
        x, y = np.transpose(x, axes=(1, 0, 2)), np.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    if k == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    return x.copy(), y.copy()


def computeperf(cm):
    good, fa, miss = cm[0], cm[1], cm[2]
    if good == 0:
        precision = 0
        recall = 0
    else:
        precision = good / (good + fa)
        recall = good / (good + miss)
    return torch.Tensor([precision * recall, precision, recall])


class AED:
    def __init__(self, flag):
        assert flag in ["train", "test"]
        if flag == "train":
            self.root = "/data/AED/training_images/"
            pathtovt = "/data/AED/training_elephants.csv"
        else:
            self.root = "/data/AED/test_images/"
            pathtovt = "/data/AED/test_elephants.csv"

        print("reading csv file", pathtovt)
        self.labels = {}
        with open(pathtovt, "r") as csvfile:
            text = csv.reader(csvfile, delimiter=",")
            for line in text:
                if line[0] not in self.labels:
                    self.labels[line[0]] = []
                self.labels[line[0]].append((int(line[1]), int(line[2])))

        self.names = list(self.labels.keys())
        print("data successfully loaded")

    def getImageAndLabel(self, name, torchformat=False):
        assert name in self.labels

        image = PIL.Image.open(self.root + name + ".jpg").convert("RGB")
        image = np.uint8(np.asarray(image).copy())

        mask = np.zeros((image.shape[0], image.shape[1]))
        points = self.labels[name]
        for c, r in points:
            mask[r][c] = 1

        if torchformat:
            x = torch.Tensor(np.transpose(image, axes=(2, 0, 1)))
            return x.unsqueeze(0), torch.Tensor(mask)
        else:
            return image, mask

    def getbatchloader(self, nbtiles=10000, tilesize=128, batchsize=32, nbpos=1):
        NB = nbtiles // len(self.names) + 1

        # crop
        XY = []
        for name in self.names:
            image, label = self.getImageAndLabel(name, torchformat=False)

            # random crop
            row = np.random.randint(0, image.shape[0] - tilesize - 2, size=NB)
            col = np.random.randint(0, image.shape[1] - tilesize - 2, size=NB)

            for i in range(NB):
                im = image[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize, :]
                mask = label[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize]
                XY.append((im.copy(), mask.copy()))

            # positive crop
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


import torchvision
import math

if __name__ == "__main__":
    aed = AED(flag="test")

    distance = np.zeros(129)
    for name in aed.names:
        points = aed.labels[name]
        if points == []:
            continue
        I = len(points)
        for i in range(I):
            for j in range(I):
                if i < j:
                    dr = abs(points[i][0] - points[j][0])
                    dc = abs(points[i][1] - points[j][1])
                    d = int(max(dr, dc))
                    if d < 128:
                        distance[int(d)] += 1
    for i in range(120):
        if distance[i] != 0:
            print(i, distance[i])

    i = 0
    while aed.labels[aed.names[i]] == []:
        i += 1

    image, mask = aed.getImageAndLabel(aed.names[i], torchformat=True)
    debug = image[0].cpu().numpy()
    debug = np.transpose(debug, axes=(1, 2, 0))
    debug = PIL.Image.fromarray(np.uint8(debug))
    debug.save("build/image.png")
    debug = mask.cpu().numpy()
    debug = PIL.Image.fromarray(np.uint8(debug) * 255)
    debug.save("build/label.png")

    beforeafter = torch.zeros(2).cuda()
    for name in aed.names:
        _, mask = aed.getImageAndLabel(name, torchformat=True)
        mask = mask.cuda()
        beforeafter[0] += torch.sum(mask)
        mask = torch.nn.functional.max_pool2d(
            mask.unsqueeze(0), kernel_size=16, stride=16, padding=0
        )
        beforeafter[1] += torch.sum(mask)
    print(beforeafter)

    batchloader = aed.getbatchloader(nbtiles=0, batchsize=8)
    x, y = next(iter(batchloader))

    torchvision.utils.save_image(x / 255, "build/cropimage.png")
    torchvision.utils.save_image(y.unsqueeze(1).float(), "build/croplabel.png")
