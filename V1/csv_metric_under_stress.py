from __future__ import print_function

import os
import sys
import random
import numpy as np
import PIL
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import torch.autograd.variable

import mynetwork

model = torch.load("model.pth")

imagename = os.listdir("test/images")
with torch.no_grad():
    print("#####################################################################")
    print("######################   standard evaluation   ######################")
    print("#####################################################################")
    metric = np.zeros(4, dtype=float)
    for name in imagename:
        image = model.loadimage("test/images/" + name)
        centers = model.loadcenters("test/centers/" + name + ".txt")

        pred, lmetric, coloredimage = model.forwardnumpywithvt(image, centers)
        metric += lmetric

        # coloredimagescaled = np.zeros((coloredimage.shape[0]*model.modelpooling,coloredimage.shape[1]*model.modelpooling,3),dtype=int)
        # for r in range(coloredimage.shape[0]):
        #    for c in range(coloredimage.shape[1]):
        #        for i in range(3):
        #            coloredimagescaled[r*model.modelpooling:(r+1)*model.modelpooling,c*model.modelpooling:(c+1)*model.modelpooling,i:i+1]=coloredimage[r][c][i]
        # image = np.transpose(image,(1,2,0))
        # im = PIL.Image.fromarray(np.uint8(image))
        # im.save("understresspreddiff/raw_"+name+"_image.png.jpg")
        # im = PIL.Image.fromarray(np.uint8(coloredimagescaled*255))
        # im.save("understresspreddiff/raw_"+name+"_match.png.jpg")

    nbposit, nbpropo, nbcorrect, nbclearfalsealarm = tuple(metric)
    if nbposit == 0:
        nbposit = 1
    if nbpropo == 0:
        nbpropo = 1

    print("@@@@@@@@@@@@@@@@@ no stress ===>")
    print("recall=", 100 * nbcorrect / nbposit)
    print("precision=", 100 * nbcorrect / nbpropo)
    print("gscore=", 100 * nbcorrect / nbpropo * nbcorrect / nbposit)
    print("nb hard false alarms=", nbclearfalsealarm)


print("#####################################################################")
print("######################  adversarial evaluation ######################")
print("#####################################################################")


def flatTensor(mytensor):
    mytensor = mytensor.view(mytensor.size(0), mytensor.size(1), -1)
    mytensor = torch.transpose(mytensor, 1, 2).contiguous()
    mytensor = mytensor.view(-1, mytensor.size(2))
    return mytensor


def flatGT(mygt):
    target = torch.autograd.Variable(torch.from_numpy(mygt).cuda().long())
    target = target.view(-1)
    return target


X = []
Y = []
Name = []
for name in imagename:
    X.append(model.loadimage("test/images/" + name))
    Y.append(model.loadcenters("test/centers/" + name + ".txt"))
    Name.append(name)

stress = "adversarial"
for intensity in range(10):
    print(
        "@@@@@@@@@@@@@@@@@ stress = adversarial with intensity ", intensity + 1, " ===>"
    )
    metric = np.zeros(4, dtype=float)

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        variableimage = torch.autograd.Variable(
            torch.from_numpy(np.expand_dims(x, axis=0)).cuda().float(),
            requires_grad=True,
        )
        optimizer = optim.SGD([variableimage], lr=1, momentum=0)
        optimizer.zero_grad()

        variableoutput, _, _, _ = model(variableimage)
        pred = np.argmax(variableoutput.cpu().data.numpy()[0], axis=0)

        losslayer = nn.CrossEntropyLoss()
        mask = model.formbinaryraster(y, pred.shape, 0)
        loss = losslayer(flatTensor(variableoutput), flatGT(mask))

        optimizer.zero_grad()
        loss.backward()
        xgrad = np.sign(variableimage.grad.cpu().data.numpy())[0]

        X[i] = X[i] + xgrad
        X[i] = np.maximum(np.zeros(X[i].shape, dtype=float), X[i])
        X[i] = np.minimum(255 * np.ones(X[i].shape, dtype=float), X[i])
        X[i] = np.uint8(X[i])

    with torch.no_grad():
        for i in range(len(X)):
            image = X[i]
            centers = Y[i]
            name = Name[i]

            pred, lmetric, coloredimage = model.forwardnumpywithvt(image, centers)
            metric += lmetric

            # coloredimagescaled = np.zeros((coloredimage.shape[0]*model.modelpooling,coloredimage.shape[1]*model.modelpooling,3),dtype=int)
            # for r in range(coloredimage.shape[0]):
            #    for c in range(coloredimage.shape[1]):
            #        for i in range(3):
            #            coloredimagescaled[r*model.modelpooling:(r+1)*model.modelpooling,c*model.modelpooling:(c+1)*model.modelpooling,i:i+1]=coloredimage[r][c][i]
            # image = np.transpose(image,(1,2,0))
            # im = PIL.Image.fromarray(np.uint8(image))
            # im.save("understresspreddiff/"+stress+"_"+str(intensity)+"_"+name+"_image.png.jpg")
            # im = PIL.Image.fromarray(np.uint8(coloredimagescaled*255))
            # im.save("understresspreddiff/"+stress+"_"+str(intensity)+"_"+name+"_match.png.jpg")

        nbposit, nbpropo, nbcorrect, nbclearfalsealarm = tuple(metric)
        if nbposit == 0:
            nbposit = 1
        if nbpropo == 0:
            nbpropo = 1

        print("recall=", 100 * nbcorrect / nbposit)
        print("precision=", 100 * nbcorrect / nbpropo)
        print("gscore=", 100 * nbcorrect / nbpropo * nbcorrect / nbposit)
        print("nb hard false alarms=", nbclearfalsealarm)

with torch.no_grad():
    print("#####################################################################")
    print("###################### blind stress evaluation ######################")
    print("#####################################################################")
    allstress = [
        ("hard_sparse_noise", list(range(1, 15))),
        ("illumination_change", list(range(1, 15))),
        ("gaussian_noise", list(range(1, 15))),
    ]
    for stress, possibleintensity in allstress:
        for intensity in possibleintensity:
            metric = np.zeros(4, dtype=float)

            for name in imagename:
                image = model.loadimage("test/images/" + name)
                centers = model.loadcenters("test/centers/" + name + ".txt")

                if stress == "illumination_change":
                    if random.randint(0, 1) == 0:
                        image += intensity * 10
                    else:
                        image -= intensity * 10
                    image = np.maximum(np.zeros(image.shape, dtype=float), image)
                    image = np.minimum(255 * np.ones(image.shape, dtype=float), image)

                if stress == "gaussian_noise":
                    image += np.random.standard_normal(image.shape) * intensity * 3
                    image = np.maximum(np.zeros(image.shape, dtype=float), image)
                    image = np.minimum(255 * np.ones(image.shape, dtype=float), image)

                if stress == "hard_sparse_noise":
                    nbRand = (
                        image.shape[0]
                        * image.shape[1]
                        * image.shape[2]
                        * intensity
                        // 100
                        + 1
                    )
                    randindex = np.stack(
                        [
                            np.random.randint(0, image.shape[0], size=nbRand),
                            np.random.randint(0, image.shape[1], size=nbRand),
                            np.random.randint(0, image.shape[2], size=nbRand),
                        ],
                        axis=0,
                    )
                    for i in range(randindex[0].shape[0]):
                        image[randindex[0][i]][randindex[1][i]][
                            randindex[2][i]
                        ] = 255 * random.randint(0, 1)

                pred, lmetric, coloredimage = model.forwardnumpywithvt(image, centers)
                metric += lmetric

                # coloredimagescaled = np.zeros((coloredimage.shape[0]*model.modelpooling,coloredimage.shape[1]*model.modelpooling,3),dtype=int)
                # for r in range(coloredimage.shape[0]):
                #    for c in range(coloredimage.shape[1]):
                #        for i in range(3):
                #            coloredimagescaled[r*model.modelpooling:(r+1)*model.modelpooling,c*model.modelpooling:(c+1)*model.modelpooling,i:i+1]=coloredimage[r][c][i]
                # image = np.transpose(image,(1,2,0))
                # im = PIL.Image.fromarray(np.uint8(image))
                # im.save("understresspreddiff/"+stress+"_"+str(intensity)+"_"+name+"_image.png.jpg")
                # im = PIL.Image.fromarray(np.uint8(coloredimagescaled*255))
                # im.save("understresspreddiff/"+stress+"_"+str(intensity)+"_"+name+"_match.png.jpg")

            nbposit, nbpropo, nbcorrect, nbclearfalsealarm = tuple(metric)
            if nbposit == 0:
                nbposit = 1
            if nbpropo == 0:
                nbpropo = 1

            print(
                "@@@@@@@@@@@@@@@@@ stress = ",
                stress,
                " with intensity ",
                intensity,
                " ===>",
            )
            print("recall=", 100 * nbcorrect / nbposit)
            print("precision=", 100 * nbcorrect / nbpropo)
            print("gscore=", 100 * nbcorrect / nbpropo * nbcorrect / nbposit)
            print("nb hard false alarms=", nbclearfalsealarm)
