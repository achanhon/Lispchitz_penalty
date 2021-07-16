import os
import sys
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

outputname = "build/model.pth"
if len(sys.argv) > 1:
    outputname = sys.argv[1]
os.system("cat train3.py")

whereIam = os.uname()[1]

print("define model")
if whereIam == "super":
    sys.path.append("/home/achanhon/github/segmentation_models/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/segmentation_models/pytorch-image-models")
    sys.path.append(
        "/home/achanhon/github/segmentation_models/pretrained-models.pytorch"
    )
    sys.path.append(
        "/home/achanhon/github/segmentation_models/segmentation_models.pytorch"
    )
if whereIam == "ldtis706z":
    sys.path.append("/home/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/pytorch-image-models")
    sys.path.append("/home/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/home/achanhon/github/segmentation_models.pytorch")
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import collections
import random

if whereIam == "super":
    net = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )
else:
    net = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )
net = net.cuda()
net.train()


print("load data")
import dataloader

cia = dataloader.CIA("train")

earlystopping = cia.getrandomtiles(128, 16)
weights = torch.Tensor([1, 10, 0.000001]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

criterionbis = smp.losses.dice.DiceLoss(mode="multiclass", ignore_index=[2])

print("train")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def f1(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


def trainaccuracy():
    cm = np.zeros((3, 3), dtype=int)
    net.eval()
    with torch.no_grad():
        for inputs, targets in earlystopping:
            inputs, targets = inputs.to(device), targets.to(device)

            targets = dataloader.convertIn3class(targets)

            outputs = net(inputs)
            _, pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(
                    pred[i].cpu().numpy().flatten(),
                    targets[i].cpu().numpy().flatten(),
                    labels=[0, 1, 2],
                )
    return cm[0:2, 0:2]


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 300
batchsize = 32
changecrops = 3

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    if epoch % changecrops == 0:
        XY = cia.getrandomtiles(128, batchsize)

    for x, y in XY:
        x, y = x.to(device), y.to(device)

        preds = net(x)
        tmp = torch.zeros(preds.shape[0], 1, preds.shape[2], preds.shape[3])
        tmp = tmp.to(device)
        preds = torch.cat([preds, tmp], dim=1)

        yy = dataloader.convertIn3class(y)

        loss = criterion(preds, yy) + criterionbis(preds, yy)

        meanloss.append(loss.cpu().data.numpy())

        if epoch > 30:
            loss = loss * 0.5
        if epoch > 60:
            loss = loss * 0.5
        if epoch > 90:
            loss = loss * 0.5
        if epoch > 120:
            loss = loss * 0.5
        if epoch > 260:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("backup model")
    torch.save(net, outputname)
    cm = trainaccuracy()
    print("accuracy and IoU", accu(cm), f1(cm))

print("training stops after reaching time limit")

