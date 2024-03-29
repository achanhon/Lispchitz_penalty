import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

print("define model")
import dataloader

net = torchvision.models.vgg16()
net = net.features
net._modules["30"] = torch.nn.Identity()
dummy = torch.zeros(1, 3, 16 * 5, 16 * 5)
dummy = net(dummy)
assert dummy.shape == (1, 512, 5, 5)
net.add_module("31", torch.nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1))
net.add_module("32", torch.nn.LeakyReLU())
net.add_module("33", torch.nn.Conv2d(1024, 2, kernel_size=1, padding=0, stride=1))
net = net.cuda()
net.train()


print("load data")
cia = dataloader.CIA(flag="custom", custom=["isprs/train", "vedai/train", "dfc/train"])

print("train")
import collections
import random

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 800
batchsize = 32
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = cia.getrandomtiles(192, batchsize)
    stats = torch.zeros(3).cuda()

    for x, y in XY:
        x, y = x.cuda(), y.cuda()
        z = net(x)

        nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        loss = criterion(z, y)
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

        with torch.no_grad():
            z = z[:, 1, :, :] - z[:, 0, :, :]
            stats[0] += torch.sum((z > 0).float() * (y == 1).float())
            stats[1] += torch.sum((z > 0).float() * (y == 0).float())
            stats[2] += torch.sum((z <= 0).float() * (y == 1).float())

    torch.save(net, "build/model.pth")
    perfs = dataloader.computeperf(stats)
    print("perf", perfs)

    if perfs[0] * 100 > 92:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
