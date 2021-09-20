import os
import sys

if len(sys.argv) > 1:
    outputname = sys.argv[1]
else:
    outputname = "build/model.pth"

whereIam = os.uname()[1]
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")
else:
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")

import torch
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
import dataloader

print("define model")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()
tmp = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net = torch.nn.Sequential()
net.add_module("core", tmp)
net.add_module("softnms", dataloader.SoftNMS())
net = net.cuda()
net.train()


print("load data")
cia = dataloader.CIA("custom", custom=["isprs/train"])
batchsize = 32

print("train")
import collections
import random


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / torch.sum(cm)


def iou(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
meanloss = collections.deque(maxlen=200)
nbepoch = 800
criterionbis = smp.losses.dice.DiceLoss(mode="multiclass")

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = cia.getrandomtiles(batchsize)
    cm = torch.zeros((2, 2)).cuda()

    for x, y in XY:
        x, y = x.cuda(), y.cuda()
        D = dataloader.distancetransform(y)

        nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")

        z = net(x)

        CE = criterion(z, y)
        if nb1 != 0:
            dice = criterionbis(z, y)
            loss = torch.mean(CE * D) + dice
        else:
            loss = torch.mean(CE * D)

        meanloss.append(loss.cpu().data.numpy())

        if epoch > 30:
            loss = loss * 0.5
        if epoch > 90:
            loss = loss * 0.5
        if epoch > 160:
            loss = loss * 0.5
        if epoch > 400:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

        _, z = z.max(1)
        cm[0][0] += torch.sum((z == 0).float() * (y == 0).float() * D)
        cm[1][1] += torch.sum((z == 1).float() * (y == 1).float() * D)
        cm[1][0] += torch.sum((z == 1).float() * (y == 0).float() * D)
        cm[0][1] += torch.sum((z == 0).float() * (y == 1).float() * D)

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    torch.save(net, outputname)
    print("perf", iou(cm), accu(cm))

    if iou(cm) > 92:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
