

print("TEST AIRS FSGM")
import torch
import segsemdata
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True
import PIL
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def accuracy(cm):
    return np.sum(cm.diagonal())/(np.sum(cm)+1)
    
def iou(cm):
    return 0.5*(cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+1) + cm[1][1]/(cm[1][1]+cm[0][1]+cm[1][0]+1))

print("load data")
datatest = segsemdata.makeTestAIRSdataset(size=33)

print("load model")
net = torch.load("build/model.pth")
net = net.to(device)
net.eval()

print("test")
cm,cmfsgm = np.zeros((2,2),dtype=int),np.zeros((2,2),dtype=int)
for name in datatest.names:
    print(name)
    im = PIL.Image.open(datatest.root+"/"+datatest.images[name]).convert("RGB").copy()
    im = np.asarray(im,dtype=np.uint8)#warning wh swapping
    imagetensor = torch.Tensor(np.transpose(im,axes=(2, 0, 1))).cpu().to(device).unsqueeze(0)
    size = (imagetensor.shape[2]//64)*64,(imagetensor.shape[3]//64)*64
    x = F.interpolate(imagetensor,size)
    
    realmask = PIL.Image.open(datatest.root+"/"+datatest.masks[name]).convert("RGB").copy()
    realmask = realmask.resize(((imagetensor.shape[3]//64)*64,(imagetensor.shape[2]//64)*64),PIL.Image.NEAREST)
    y = np.asarray(realmask,dtype=np.uint8) #warning wh swapping
    y = torch.from_numpy(datatest.colorvtTOvt(y)).long().to(device).unsqueeze(0)
    
    with torch.no_grad():
        z = net(x).cpu()
        _,z = z[0].max(0)
    
    localcm = confusion_matrix(z.cpu().numpy().flatten(),y.cpu().numpy().flatten(),[0,1])
    #print(localcm)
    print(accuracy(localcm),iou(localcm))
    cm += localcm
    
    for a,b,c,d in [(0,x.shape[2]//2,0,x.shape[3]//2),(x.shape[2]//2,x.shape[2],0,x.shape[3]//2),(0,x.shape[2]//2,x.shape[3]//2,x.shape[3]),(x.shape[2]//2,x.shape[2],x.shape[3]//2,x.shape[3])]:
        xfsgm = x[:,:,a:b,c:d].clone().detach().requires_grad_()
        optimizer = optim.Adam([xfsgm], lr=0.0001)
        z = net(xfsgm)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(z,y[:,a:b,c:d])
        optimizer.zero_grad()
        loss.backward()
        grad = xfsgm.grad.data.sign().detach().clone()
        del xfsgm, optimizer,z, loss,criterion
        
        with torch.no_grad():
            z = net(x[:,:,a:b,c:d]+grad).cpu()
            _,z = z[0].max(0)
        
        localcmfsgm = confusion_matrix(z.cpu().numpy().flatten(),y[:,a:b,c:d].cpu().numpy().flatten(),[0,1])
        #print(localcmfsgm)
        print(accuracy(localcmfsgm),iou(localcmfsgm))
        cmfsgm += localcmfsgm

print("global results")
#print(cm)
print(accuracy(cm),iou(cm))
print(accuracy(cmfsgm),iou(cmfsgm))

