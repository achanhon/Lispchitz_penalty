import os
import sys

if len(sys.argv) > 1:
    outputname = sys.argv[1]
else:
    outputname = "build/model.pth"

whereIam = os.uname()[1]
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
net.add_module("softnmsbis", dataloader.SoftNMS())
net = net.cuda()
net.train()


print("load data")
if whereIam == "wdtim719z":
    cia = dataloader.CIA("custom", custom=["isprs/train", "saclay/train"])
else:
    cia = dataloader.CIA("train")
batchsize = 32

print("train")
import collections
import random


class Gscore(torch.nn.Module):
    def __init__(self):
        super(Gscore, self).__init__()
        self.tot = torch.zeros(1).cuda()
        self.correct = torch.zeros(1).cuda()
        self.good = torch.zeros(1).cuda()
        self.fa = torch.zeros(1).cuda()
        self.miss = torch.zeros(1).cuda()

    def forward(self, y, z, D):
        with torch.no_grad():
            zs = z[:, 1, :, :] - z[:, 0, :, :]
            self.tot += torch.sum(D)

            goodP = torch.sum((zs > 0).float() * (y == 1).float())
            goodN = torch.sum((zs < 0).float() * (y == 0).float() * D)
            self.correct += goodP + goodN
            self.good += goodP

            self.fa += torch.sum((zs > 0).float() * (y == 0).float() * D)
            self.miss += torch.sum((zs < 0).float() * (y == 1).float())

    def getPerf(self):
        precision = self.good / (self.good + self.fa + 0.0001)
        recall = self.good / (self.good + self.miss + 0.0001)

        return precision * recall * 100.0, self.correct / self.tot * 100


class GscoreLoss(torch.nn.Module):
    def __init__(self):
        super(GscoreLoss, self).__init__()

    def forward(self, y, z, D):
        zs = z[:, 1, :, :] - z[:, 0, :, :]
        zp = torch.nn.functional.relu(zs)
        zm = torch.nn.functional.relu(-zs)

        good = zp * (y == 1).float()
        good = torch.min(good, 0.9 + 0.1 * good)
        good = torch.sum(good)
        # being very confident on trivial correct ones should not be sufficient

        fa = torch.sum(zp * D * (y == 0).float())
        miss = torch.sum(zm * (y == 1).float())

        precision = good / (good + fa + 0.0001)
        recall = good / (good + miss + 0.0001)

        loss = 1.0 - precision * recall
        return loss


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
meanloss = collections.deque(maxlen=200)
gscore = Gscore()
gscoreloss = GscoreLoss()
nbepoch = 800
criterionbis = smp.losses.dice.DiceLoss(mode="multiclass")

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = cia.getrandomtiles(batchsize)

    for x, y in XY:
        x, y = x.cuda(), y.cuda()
        D = dataloader.distancetransform(y)

        z = net(x)

        nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        CE = criterion(z, y)
        G = gscoreloss(y, z, D)
        dice = criterionbis(z, y)
        loss = torch.mean(CE * D) + 10.0 * G + dice

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

        gscore(y, z, D)

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    torch.save(net, outputname)
    g, accuracy = gscore.getPerf()
    print("perf", g, accuracy)

    if g > 92:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
