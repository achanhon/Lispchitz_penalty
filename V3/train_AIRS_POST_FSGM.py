print("TRAIN AIRS + POST WITH FSGM")
import torch
import segsemdata
import numpy as np
import torch.backends.cudnn as cudnn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

print("load data")
datatrain = segsemdata.MergedSegSemDataset(
    [
        segsemdata.makeTrainAIRSdataset(),
        segsemdata.makeISPRS_POSTDAM(),
        segsemdata.makeTrainAIRSdataset(),
    ]
)
earlystopping = datatrain.getrandomtiles(25, 128, 128, 16)

print("load model")
net = segsemdata.Unet(2)
net.loadpretrained("/data/vgg16-00b39a1b.pth")
net = net.to(device)
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("train setting")
import torch.nn as nn
import torch.optim as optim
import collections
from sklearn.metrics import confusion_matrix

weights = torch.tensor([1.0, 10.0]).to(device)
criterion_weighted = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

meanloss = collections.deque(maxlen=200)
nbepoch = 90


def trainaccuracy():
    net.eval()
    cm = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for inputs, targets in earlystopping:
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(
                    pred[i].cpu().numpy().flatten(),
                    targets[i].cpu().numpy().flatten(),
                    [0, 1],
                )
    return np.sum(cm.diagonal()) / (np.sum(cm) + 1)


print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    trainloader = datatrain.getrandomtiles(100, 128, 128, 16)
    net.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        grad = torch.zeros(inputs.shape).to(device)
        if epoch != 0:
            inputsfsgm = inputs.clone().detach().requires_grad_()
            optimizerbis = optim.Adam([inputsfsgm], lr=0.0001)
            z = net(inputsfsgm)
            lossbis = criterion_weighted(z, targets)
            optimizerbis.zero_grad()
            lossbis.backward()
            grad = inputsfsgm.grad.data.sign().detach().clone()
            del inputsfsgm, optimizerbis, z, lossbis

        loss, _ = net.computeloss(inputs + grad, targets, criterion_weighted)

        if epoch > 30:
            loss = loss * 0.5
        if epoch > 60:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meanloss.append(loss.cpu().data.numpy())
        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    torch.save(net, "build/model.pth")
    acc = trainaccuracy()
    print("acc=", acc)
    if acc > 0.97:
        quit()
