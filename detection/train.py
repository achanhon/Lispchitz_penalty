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

net = torch.nn.Sequential()
net.add_module(
    "encoder",
    smp.encoders.get_encoder(
        "efficientnet-b7",
        weights="imagenet",
        in_channels=3,
        depth=5,
    ),
)
net.add_module("decoder", dataloader.PartialDecoder())
net.add_module("softNMS", dataloader.SoftNMS())
net.add_module("softNMS", dataloader.SoftNMS())
net.add_module("head", dataloader.OneToTwo())
net = net.cuda()
net.train()


print("load data")
if whereIam == "wdtim719z" or True:
    cia = dataloader.CIA("custom", custom=["isprs/train"])
else:
    cia = dataloader.CIA("train")
batchsize = 32

print("train")
import collections
import random


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
meanloss = collections.deque(maxlen=200)
nbepoch = 800
criterionbis = smp.losses.dice.DiceLoss(mode="multiclass")

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = cia.getrandomtiles(batchsize)
    cm = torch.zeros((2, 2)).cuda()

    for x, y in XY:
        x, y = x.cuda(), y.cuda().float().unsqueeze(0)
        y = torch.nn.functional.max_pool2d(y, kernel_size=8, stride=8)
        y = y[0].long()

        nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        z = net(x)

        CE = criterion(z, y)
        if nb1 != 0:
            dice = criterionbis(z, y)
            loss = CE + dice
        else:
            loss = CE

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

        z = (z[:, 1, :, :] > z[:, 0, :, :]).long()
        cm[0][0] += torch.sum((z == 0).float() * (y == 0).float())
        cm[1][1] += torch.sum((z == 1).float() * (y == 1).float())
        cm[1][0] += torch.sum((z == 1).float() * (y == 0).float())
        cm[0][1] += torch.sum((z == 0).float() * (y == 1).float())

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    torch.save(net, outputname)
    g, pre, rec = dataloader.perf(cm)
    print("perf", g, pre, rec)

    if g > 92:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
