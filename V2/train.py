

print("TRAIN.PY")
import torch
import segsemdata
import numpy as np
import torch.backends.cudnn as cudnn
device = "cuda" if torch.cuda.is_available() else "cpu"

print("load data")
dfc2015train = segsemdata.makeISPRStrainVAIHINGEN(datasetpath="/data/ISPRS_VAIHINGEN",normalize=False,color=True)
nbclasses = len(dfc2015train.getcolors())
earlystopping = dfc2015train.getrandomtiles(100,128,128,16)

print("load model")
net = segsemdata.Unet(len(dfc2015train.getcolors()))
net.loadpretrained("/data/vgg16-00b39a1b.pth")
net = net.to(device)
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("train setting")
import torch.nn as nn
import torch.optim as optim
import collections
import random
from sklearn.metrics import confusion_matrix
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

meanloss = collections.deque(maxlen=200)
nbepoch = 90

def trainaccuracy():
    net.eval()
    cm = np.zeros((nbclasses,nbclasses),dtype=int)
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(earlystopping):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _,pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(pred[i].cpu().numpy().flatten(),targets[i].cpu().numpy().flatten(),list(range(nbclasses)))
    return np.sum(cm.diagonal())/(np.sum(cm)+1)

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    trainloader = dfc2015train.getrandomtiles(200,128,128,16)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        #loss = criterion(torch.flatten(outputs,start_dim=2), torch.flatten(targets,start_dim=1))
        loss,_ = net.computeloss(inputs,targets,criterion)
        
        if epoch>30:
            loss = loss*0.5
        if epoch>60:
            loss = loss*0.5
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meanloss.append(loss.cpu().data.numpy())
        if random.randint(0,30)==0:
            print("loss=",(sum(meanloss)/len(meanloss)))
    
    torch.save(net, "build/model.pth")
    acc=trainaccuracy()
    print("acc=", acc)
    if acc>0.97:
        quit()

