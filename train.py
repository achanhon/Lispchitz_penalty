from __future__ import print_function

import os
import sys
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import torch.autograd.variable

import mynetwork
from calcul_precision_recall import computeprecisionrappel

if not torch.cuda.is_available():
    print("no cuda")
    quit()
seed = 1
torch.cuda.manual_seed(seed)    
LR = 0.01   
POS_W = 10
LR_EASY = 0.005
LR_RAW = 0.0005
LR_GRAD = 0.00005

print("load network")
model = mynetwork.NET()
model.cuda()
model.train()

print("index data")
imagename = os.listdir("train/images")

print("init network with pretrained weight")
correspondance=[]
correspondance.append(("features.0","conv1_1"))
correspondance.append(("features.2","conv1_2"))
correspondance.append(("features.5","conv2_1"))
correspondance.append(("features.7","conv2_2"))
correspondance.append(("features.10","conv3_1"))
correspondance.append(("features.12","conv3_2"))
correspondance.append(("features.14","conv3_3"))
correspondance.append(("features.17","conv4_1"))
correspondance.append(("features.19","conv4_2"))
correspondance.append(("features.21","conv4_3"))
correspondance.append(("features.24","conv5_1"))
correspondance.append(("features.26","conv5_2"))
correspondance.append(("features.28","conv5_3"))       

pretrained_dict = torch.load("vgg16-00b39a1b.pth")
model_dict = model.state_dict()
        
for name1,name2 in correspondance:
    fw,fb = False,False
    for name, param in pretrained_dict.items():
        if name==name1+".weight" :
            model_dict[name2+".weight"].copy_(param)
            fw=True
        if name==name1+".bias" :
            model_dict[name2+".bias"].copy_(param)
            fb=True
    if (not fw) or (not fb):
        print(name2+" not found")
        quit()
model.load_state_dict(model_dict)   

print("define solver parameter")
lr = 1
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
losslayer = nn.CrossEntropyLoss()
losslayerbis = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,POS_W]).cuda())

def flatTensor(mytensor):
    mytensor = mytensor.view(mytensor.size(0),mytensor.size(1), -1)
    mytensor = torch.transpose(mytensor,1,2).contiguous()
    mytensor = mytensor.view(-1,mytensor.size(2))
    return mytensor
def flatGT(mygt):
    target = torch.autograd.Variable(torch.from_numpy(mygt).long()).cuda()                    
    target = target.view(-1)
    return target

allloss = collections.deque(maxlen=200)
allrawloss = collections.deque(maxlen=200)
alllossMiss = collections.deque(maxlen=200)
alllossfalse = collections.deque(maxlen=200)
alllossgrad = collections.deque(maxlen=200)
        
print("forward-backward data")
nbepoch = 10
for epoch in range(nbepoch):
    print(epoch)
    random.shuffle(imagename)
    optimizer.zero_grad()
    model.train()
    
    for name in imagename:
        image = model.loadimage("train/images/"+name)
        centers = model.loadcenters("train/centers/"+name+".txt")
        
        inputtensor = torch.autograd.Variable(torch.Tensor(np.expand_dims(image,axis=0)).cuda(),requires_grad=True)        
        outputtensornms,outputtensor,outputdilated,feature = model(inputtensor)

        pred = np.argmax(outputtensor.cpu().data.numpy()[0], axis=0)
        prednms = np.argmax(outputtensornms.cpu().data.numpy()[0], axis=0)
        
        #penalize pred(i,j)!=gt(i,j)
        rawmask = model.formbinaryraster(centers,pred.shape,0)
        outputtensornms = flatTensor(outputtensornms)
        rawloss = losslayer(outputtensornms,flatGT(rawmask))
        
        #penalize pred(i,j)==1 and gt(around i,j)==0
        dilatedmask = model.formbinaryraster(centers,pred.shape,2)
        dilatedmaskAroundPred = dilatedmask*pred 
        lossMiss = losslayerbis(flatTensor(outputtensor),flatGT(dilatedmaskAroundPred))
        
        #penalize pred(around i,j)==0 and gt(i,j)==1
        outputdilatedgated = torch.autograd.Variable(torch.Tensor(np.expand_dims(np.expand_dims(rawmask,axis=0),axis=0)).cuda())*outputdilated
        outputdilatedgated = torch.cat([-outputdilatedgated,outputdilatedgated],dim=1)
        lossFalse = losslayerbis(flatTensor(outputdilatedgated),flatGT(rawmask))
        
        #approximate the real detection loss - typically penalize double detections
        detectionmask = np.zeros(pred.shape,dtype=int)
        prediction = np.transpose(np.nonzero(prednms))
        nbprop = prediction.shape[0]
        nbgt = centers.shape[0]
        nbclearfalsealarm = np.sum((1-dilatedmask)*prednms)
        
        nbGoodmatch,goodalarme,catcheddetection,_ = computeprecisionrappel(prediction,centers,3)
        catcheddetection = set(catcheddetection)
        for i in range(centers.shape[0]):
            if i not in catcheddetection:
                row = centers[i][0]
                col = centers[i][1]
                detectionmask[min(row,detectionmask.shape[0]-1)][min(col,detectionmask.shape[1]-1)]=1
                
            for j in goodalarme:
                row = prediction[j][0]
                col = prediction[j][1]
                detectionmask[row][col]=1
        loss = losslayer(outputtensornms,flatGT(detectionmask))
        
        #penalize variation of features regarding variation of input
        feature = feature.contiguous()
        feature = feature[0].view(512,-1)
        feature = feature.contiguous()
        feature = torch.transpose(feature,0,1)
        centralfeature = feature.mean(0)
        diff = feature - centralfeature
        gradientFeatureInput = torch.autograd.grad(diff.abs().sum(), inputtensor, allow_unused=True, create_graph=True, retain_graph=True)[0]
        gradientFeatureInput = gradientFeatureInput.contiguous()
        gradientFeatureInput = gradientFeatureInput.view(1,-1)
        gradnorm = torch.norm(gradientFeatureInput,2,1)
        onelipsdisagreement = F.relu(gradnorm-1)
        lossgradient = onelipsdisagreement.sum()
        
        allloss.append(loss.cpu().data.numpy())
        alllossfalse.append(lossFalse.cpu().data.numpy())
        alllossMiss.append(lossMiss.cpu().data.numpy())
        allrawloss.append(rawloss.cpu().data.numpy())
        alllossgrad.append(lossgradient.cpu().data.numpy())
        if random.randint(0,20)==0 or epoch==0:
            print(sum(allloss)/len(allloss),sum(alllossfalse)/len(alllossfalse),sum(alllossMiss)/len(alllossMiss),sum(allrawloss)/len(allrawloss),sum(alllossgrad)/len(alllossgrad),nbgt,nbprop,nbGoodmatch,nbclearfalsealarm)
        
        optimizer.zero_grad()
        losslr = loss*LR + lossFalse*LR_EASY + lossMiss*LR_EASY + rawloss*LR_RAW + lossgradient*LR_GRAD
        losslr.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        
    if epoch%1==0:
        print("train eval",model.stdtest("train","trainpreddiff/"))
        print("test eval",model.stdtest("test","preddiff/"))
        torch.save(model,"backup"+str(epoch)+".pth")

torch.save(model,"model.pth")
