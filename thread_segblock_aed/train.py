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

net = torchvision.models.vgg16(pretrained=True)
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
aed = dataloader.AED(flag="train")
aed.start()

print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = torch.zeros(1).cuda()
stats = torch.zeros(3).cuda()
nbbatch = 10000
batchsize = 32


def dice_loss(preds, targets):
    preds = preds[:, 1, :, :] - preds[:, 0, :, :]
    targets = targets.float()
    I = (preds * targets).sum()
    U = preds.sum() + targets.sum()
    return 1.0 - I / (U + 1)


for batch in range(nbbatch):
    if batch % 25 == 24:
        print("batch=", batch, "/", nbbatch)

    x, y = aed.getbatch(batchsize=batchsize)
    x, y = x.cuda(), y.cuda()
    z = net(x)

    nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
    weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    loss = criterion(z, y) + 0.5 * dice_loss(z, y)
    meanloss += loss.clone().detach()

    if batch > 1000:
        loss = loss * 0.5
    if batch > 3000:
        loss = loss * 0.5
    if batch > 6000:
        loss = loss * 0.5
    if batch > 9000:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

    if batch % 50 == 49:
        print("loss=", meanloss / 50)
        meanloss = torch.zeros(1).cuda()

    with torch.no_grad():
        z = z[:, 1, :, :] - z[:, 0, :, :]
        stats += dataloader.computeperf(yz=(y, z))

    if batch % 200 == 199:
        torch.save(net, "build/model.pth")
        perfs = dataloader.computeperf(stats=stats)
        stats = torch.zeros(3).cuda()
        print("perf", perfs)

        if perfs[0] * 100 > 92:
            print("training stops after reaching high training accuracy")
            os._exit(0)

print("training stops after reaching time limit")
os._exit(0)
