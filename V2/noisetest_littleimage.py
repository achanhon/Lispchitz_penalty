
print("NOISE LITTLE TEST.PY")
import torch
import torch.nn as nn
import torch.nn.functional as F
import segsemdata
import mynoisegenerator
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import PIL
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"

print("load data")
dfc2015test = segsemdata.makeISPRStestVAIHINGEN(datasetpath="/data/ISPRS_VAIHINGEN",normalize=False,color=True)
nbclasses = len(dfc2015test.getcolors())

print("load model")
net = torch.load("build/model.pth")
net = net.to(device)
net.eval()
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("do test")
cm = np.zeros((nbclasses,nbclasses))
stesscm = np.zeros((nbclasses,nbclasses))
for imageindex,name in enumerate(dfc2015test.names):
    im = PIL.Image.open(dfc2015test.root+"/"+dfc2015test.images[name]).convert("RGB").copy()
    im = np.asarray(im,dtype=np.uint8)#warning wh swapping
    imagetensor = torch.Tensor(np.transpose(im,axes=(2, 0, 1))).cpu().to(device).unsqueeze(0)
    size = (imagetensor.shape[2]//32)*32,(imagetensor.shape[3]//32)*32
    x = F.interpolate(imagetensor,size)
    
    realmask = PIL.Image.open(dfc2015test.root+"/"+dfc2015test.masks[name]).convert("RGB").copy()
    realmask = realmask.resize(((imagetensor.shape[3]//32)*32,(imagetensor.shape[2]//32)*32),PIL.Image.NEAREST)
    y = np.asarray(realmask,dtype=np.uint8) #warning wh swapping
    y = torch.from_numpy(dfc2015test.colorvtTOvt(y)).long().to(device).unsqueeze(0)
    
    z = net(x).cpu()
    _,z = z[0].max(0)
    
    normalcm = confusion_matrix(z.cpu().numpy().flatten(),y.cpu().numpy().flatten(),list(range(len(dfc2015test.setofcolors))))
    normalaccuracy = np.sum(normalcm.diagonal())/(np.sum(normalcm)+1)
    cm += normalaccuracy
    
    xe,_,pred,worsecm,worseaccuracy, d1,d2,mode = mynoisegenerator.worseperturbation(x,y,net,2,len(dfc2015test.setofcolors))
    stesscm += worseaccuracy
    
    rahh = segsemdata.safeuint8(x[0].cpu().numpy())
    rahh = np.transpose(rahh,axes=(1, 2, 0))
    image = PIL.Image.fromarray(rahh)
    image.save("build/stress/"+name+"_x.jpg")
    
    rahh = segsemdata.safeuint8(xe[0])
    rahh = np.transpose(rahh,axes=(1, 2, 0))
    image = PIL.Image.fromarray(rahh)
    image.save("build/stress/"+name+"_xe.jpg")
    
    image = PIL.Image.fromarray(segsemdata.safeuint8(dfc2015test.vtTOcolorvt(z.numpy())))
    image.save("build/stress/"+name+"_z.jpg")
    image = PIL.Image.fromarray(segsemdata.safeuint8(dfc2015test.vtTOcolorvt(y[0].cpu().numpy())))
    image.save("build/stress/"+name+"_y.jpg")
    image = PIL.Image.fromarray(segsemdata.safeuint8(dfc2015test.vtTOcolorvt(pred[0])))
    image.save("build/stress/"+name+"_ze.jpg")
    
    print(name,": normal accuracy=",normalaccuracy," accuracy under",mode,"stress=",worseaccuracy)    

print("normal accuracy=",np.sum(cm.diagonal())/(np.sum(cm)+1))
print("stress accuracy=",np.sum(stesscm.diagonal())/(np.sum(stesscm)+1))
print(cm,stesscm)

