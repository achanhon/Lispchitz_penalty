from __future__ import print_function

import os
import sys
import os.path
import numpy as np
import PIL
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import torch.autograd.variable

from calcul_precision_recall import computeprecisionrappel


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.prob = nn.Conv2d(512, 1, kernel_size=1, bias=True)

        self.modelpooling = 16

    def forward(self, x):
        pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        x = (x - 127.5) / 127.5
        x = F.leaky_relu(self.conv1_1(x))
        x = F.leaky_relu(self.conv1_2(x))
        x = pool(x)

        x = F.leaky_relu(self.conv2_1(x))
        x = F.leaky_relu(self.conv2_2(x))
        x = pool(x)

        x = F.leaky_relu(self.conv3_1(x))
        x = F.leaky_relu(self.conv3_2(x))
        x = F.leaky_relu(self.conv3_3(x))
        x = pool(x)

        x = F.leaky_relu(self.conv4_1(x))
        x = F.leaky_relu(self.conv4_2(x))
        x = F.leaky_relu(self.conv4_3(x))
        x = pool(x)

        feature = x

        x = F.leaky_relu(self.conv5_1(x))
        x = F.leaky_relu(self.conv5_2(x))
        x = F.leaky_relu(self.conv5_3(x))
        x = self.prob(x)

        convpool5 = nn.MaxPool2d(
            kernel_size=5, stride=1, padding=2, return_indices=False
        )
        xdilated = convpool5(x + 10) - 10  # zeros padding is problematic at this point

        xnms = 100 * x - 99 * F.relu(xdilated)

        xnms = torch.cat([-xnms, xnms], dim=1)
        x = torch.cat([-x, x], dim=1)
        return xnms, x, xdilated, feature

    def loadimage(self, path):
        image3D = np.asarray(PIL.Image.open(path).convert("RGB").copy(), dtype=float)
        H, W = image3D.shape[0], image3D.shape[1]
        H, W = (H // self.modelpooling) * self.modelpooling, (
            W // self.modelpooling
        ) * self.modelpooling
        image3D = image3D[0:H, 0:W, :]
        return np.transpose(image3D, (2, 0, 1))

    def loadcenters(self, path):
        if os.stat(path).st_size == 0:
            return np.empty((0, 2), dtype=int)
        else:
            colrowcenters = np.loadtxt(path, ndmin=2, dtype=int) // self.modelpooling
            rowcolcenters = colrowcenters[:, ::-1]
            return rowcolcenters

    def formbinaryraster(self, centers, shape, k):
        out = np.zeros(shape, dtype=int)
        for i in range(centers.shape[0]):
            row = centers[i][0]
            col = centers[i][1]
            mincol = max(col - k, 0)
            maxcol = min(col + k, shape[1] - 1)
            minrow = max(row - k, 0)
            maxrow = min(row + k, shape[0] - 1)
            out[minrow : maxrow + 1, mincol : maxcol + 1] = 1
        return out

    def forwardnumpy(self, x):
        inputtensor = torch.autograd.Variable(
            torch.Tensor(np.expand_dims(x, axis=0)).cuda(), requires_grad=False
        )
        with torch.no_grad():
            outputtensor, _, _, _ = self.forward(inputtensor)
            proba = outputtensor.cpu().data.numpy()
            return proba[0]

    def computemetric(self, prob, centers):
        pred = np.argmax(prob, axis=0)

        nbposit = centers.shape[0]
        nbpropo = np.sum(pred)

        prediction = np.transpose(np.nonzero(pred))
        nbGoodmatch, goodalarme, catcheddetection, match = computeprecisionrappel(
            prediction, centers, 4
        )

        largemask = self.formbinaryraster(centers, pred.shape, 4)
        clearfalsealarm = pred * (1 - largemask)
        nbclearfalsealarm = np.sum(clearfalsealarm)

        coloredimage = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=int)
        goodalarme = set(goodalarme)
        catcheddetection = set(catcheddetection)
        for i in range(nbpropo):
            row = prediction[i][0]
            col = prediction[i][1]
            if i in goodalarme:
                coloredimage[row][col][2] = 1
            else:
                coloredimage[row][col][0] = 1
                coloredimage[row][col][2] = 1
        for i in range(nbposit):
            row = centers[i][0]
            col = centers[i][1]
            if i in catcheddetection:
                coloredimage[row][col][1] = 1
            else:
                coloredimage[row][col][1] = 1
                coloredimage[row][col][0] = 1

        return (
            pred,
            np.asarray([nbposit, nbpropo, nbGoodmatch, nbclearfalsealarm]),
            coloredimage,
        )

    def forwardnumpywithvt(self, x, centers):
        return self.computemetric(self.forwardnumpy(x), centers)

    def stdtest(self, foldertest, folderout):
        self.eval()
        imagename = os.listdir(foldertest + "/images/")
        metric = np.zeros(4, dtype=float)
        for name in imagename:
            image = self.loadimage(foldertest + "/images/" + name)
            centers = self.loadcenters(foldertest + "/centers/" + name + ".txt")

            pred, lmetric, coloredimage = self.forwardnumpywithvt(image, centers)
            metric += lmetric

            coloredimagescaled = np.zeros(
                (
                    coloredimage.shape[0] * self.modelpooling,
                    coloredimage.shape[1] * self.modelpooling,
                    3,
                ),
                dtype=int,
            )
            for r in range(coloredimage.shape[0]):
                for c in range(coloredimage.shape[1]):
                    for i in range(3):
                        coloredimagescaled[
                            r * self.modelpooling : (r + 1) * self.modelpooling,
                            c * self.modelpooling : (c + 1) * self.modelpooling,
                            i : i + 1,
                        ] = coloredimage[r][c][i]
            image = np.transpose(image, (1, 2, 0))
            im = PIL.Image.fromarray(np.uint8(image))
            im.save(folderout + name + "_image.png.jpg")
            im = PIL.Image.fromarray(np.uint8(coloredimagescaled * 255))
            im.save(folderout + name + "_match.png.jpg")

        nbposit, nbpropo, nbcorrect, nbclearfalsealarm = tuple(metric)
        if nbposit == 0:
            nbposit = 1
        if nbpropo == 0:
            nbpropo = 1
        return (
            nbposit,
            nbpropo,
            nbcorrect,
            100 * nbcorrect / nbposit,
            100 * nbcorrect / nbpropo,
            100 * nbcorrect / nbpropo * nbcorrect / nbposit,
            nbclearfalsealarm,
        )
