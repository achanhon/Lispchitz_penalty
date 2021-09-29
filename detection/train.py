import os
import sys
import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

whereIam = os.uname()[1]
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
import dataloader

print("define model")
net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
headNMS = dataloader.HardNMS()
net = net.cuda()
net.eval()


print("load data")
cia = dataloader.CIA(flag="custom", custom=["isprs/train", "saclay/train"])

print("train")
import collections
import random


class DistanceVT(torch.nn.Module):
    def __init__(self):
        super(DistanceVT, self).__init__()

        self.hackconv = torch.nn.Conv2d(1, 1, kernel_size=13, padding=6, bias=False)
        w = torch.zeros(1, 1, 13, 13)
        for i in range(13):
            for j in range(13):
                w[0][0][i][j] = (i - 6) * (i - 6) + (j - 6) * (j - 6)
        w = torch.sqrt(w)
        w = 1.0 / (1 + w)
        self.hackconv.weight.data = w
        self.hackconv = self.hackconv.cuda()

    def forward(self, y):
        yy = self.hackconv(y.unsqueeze(1)).detach()
        return torch.clamp(yy[:, 0, :, :], 0, 1)


criteriondice = smp.losses.dice.DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 800
batchsize = 32
distanceVT = DistanceVT()
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = cia.getrandomtiles(128, batchsize)
    miss, fa, good = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()

    for x, y in XY:
        x, y = x.cuda(), y.cuda().float()
        z = net(x)
        zNMS = headNMS(z)

        # coarse loss
        ycoarse = dataloader.etendre(y, 3)
        DT = dataloader.distancetransform(ycoarse)
        nb0, nb1 = torch.sum((ycoarse == 0).float()), torch.sum((ycoarse == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1) * 2]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        CE = criterion(z, ycoarse.long())
        CE = torch.mean(CE * DT)
        dice = criteriondice(z, ycoarse.long())

        if epoch < 10:
            loss = CE + dice * 0.1 + 1
        else:
            # fine loss
            DVT = distanceVT(y)
            softgood = torch.mean(zNMS * DVT)
            softfa = torch.mean(zNMS * (DVT == 0).float())
            zm = torch.nn.functional.relu(z[:, 0, :, :] - z[:, 1, :, :])
            softmiss = torch.sum(zm * y)
            softprecision = softgood / (softgood + fa + 0.00001)
            softrecall = softgood / (softgood + softmiss + 0.00001)
            softG = softprecision * softrecall

            loss = CE + dice * 0.1 + 1.0 - softG

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

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

        y5 = dataloader.etendre(y, 1)
        zNMS5 = dataloader.etendre(zNMS.unsqueeze(0), 1)
        good += torch.sum((zNMS > 0).float() * (y == 1).float())
        fa += torch.sum((zNMS > 0).float() * (y5 == 0).float())
        miss += torch.sum((zNMS5 == 0).float() * (y == 1).float())

    torch.save(net, "build/model.pth")
    precision = good / (good + fa + 1)
    rappel = good / (good + miss + 1)
    gs = precision * rappel
    print("perf", gs * 100, precision * 100, rappel * 100)

    if gs * 100 > 92:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
