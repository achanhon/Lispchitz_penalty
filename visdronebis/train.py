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

sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/d/achanhon/github/pytorch-image-models")
sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import detectionhead
import dataloader

print("define model")
net = detectionhead.DetectionHead(
    smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )
)
net = net.cuda()
net.train()

print("load data")
visdrone = dataloader.VISDRONE(flag="train")
visdrone.start()

print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
nbbatch = 100000
batchsize = 8
stats = torch.zeros(3).cuda()
meanloss = torch.zeros(1).cuda()
for batch in range(nbbatch):
    if batch % 25 == 24:
        print("batch=", batch, "/", nbbatch)

    x, y = visdrone.getbatch(batchsize=batchsize)
    x, y = x.cuda(), y.cuda().float()
    s = net(x)

    coarseloss = net.lossSegmentation(s, y)
    fineloss = net.lossDetection(s, y)

    if nbbatch < 601:
        loss = 0.1 * fineloss + 0.9 * coarseloss
    else:
        loss = 0.9 * fineloss + 0.1 * coarseloss

    meanloss += loss.clone().detach()
    if batch > 10000:
        loss = loss * 0.5
    if batch > 30000:
        loss = loss * 0.5
    if batch > 60000:
        loss = loss * 0.5
    if batch > 90000:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

    with torch.no_grad():
        z = net.headforward(s[:, 1, :, :] - s[:, 0, :, :])
        stats += net.computegscore(z, y)

    if batch % 50 == 49:
        print("loss=", meanloss / 50)
        meanloss = torch.zeros(1).cuda()

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
