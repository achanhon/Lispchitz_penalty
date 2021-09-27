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
head = dataloader.HardNMS()
net = net.cuda()
net.eval()


print("load data")
cia = dataloader.CIA(flag="custom", custom=["isprs/train"])

print("train")
import collections
import random

criteriondice = smp.losses.dice.DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 800
batchsize = 32
distanceVT = dataloader.DistanceVT()
headNMS = dataloader.HardNMS()
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = cia.getrandomtiles(128, batchsize)
    miss, fa, good = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()

    for x, y in XY:
        x, y = x.cuda(), y.cuda().float()
        z = net(x)

        # coarse loss (emphasis recall)
        DT = dataloader.distancetransform(y)
        y5 = torch.nn.functional.max_pool2d(y, kernel_size=5, stride=1, padding=2)
        nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        CE = criterion(z, y5.long())
        CE = torch.mean(CE * DT)
        dice = criteriondice(z, y5.long())

        # fine loss (emphasis precision)
        DVT = distanceVT(y)
        zNMS = headNMS(z)
        softgood = torch.mean(zNMS * DVT)
        softfa = torch.mean(zNMS * (DVT == 0).float())

        loss = CE * 0.5 + dice * 0.1  # + 1.0 - softgood / (softgood + fa + 0.00001)
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

        good += torch.sum((zNMS > 0).float() * (y == 1).float())
        fa += torch.sum((zNMS > 0).float() * (y == 0).float() * (1 - DVT) / 9)
        miss += torch.sum((zNMS == 0).float() * (y == 1).float())

    torch.save(net, "build/model.pth")
    precision = good / (good + fa + 1)
    rappel = good / (good + miss + 1)
    gs = precision * rappel
    print("perf", gs * 100, precision * 100, recall * 100)

    if gs * 100 > 92:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
