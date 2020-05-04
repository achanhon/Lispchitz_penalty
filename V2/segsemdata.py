

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def safeuint8(x):
    x0 = np.zeros(x.shape,dtype=float)
    x255 = np.ones(x.shape,dtype=float)*255
    x = np.maximum(x0,np.minimum(x.copy(),x255))
    return np.uint8(x)
    
def symetrie(x,y,i,j,k):
    if i==1:
        x,y = np.transpose(x,axes=(1,0,2)),np.transpose(y,axes=(1,0,2))
    if j==1:
        x,y = np.flip(x,axis=1),np.flip(y,axis=1)
    if k==1:
        x,y = np.flip(x,axis=1),np.flip(y,axis=1)
    return x.copy(),y.copy()

def getsize(path):
    with Image.open(path) as img:
        return img.size
        
def normalizehistogram(im):
    allvalues = list(im.flatten())
    allvalues = sorted(allvalues)
    n = len(allvalues)
    allvalues = allvalues[0:int(98*n/100)]
    allvalues = allvalues[int(2*n/100):]
    
    n = len(allvalues)
    k = n//255
    pivot = [0]+[allvalues[i] for i in range(0,n,k)]
    assert(len(pivot)>=255)
    
    out = np.zeros(im.shape,dtype = int)
    for i in range(1,255):
        out=np.maximum(out,np.uint8(im>pivot[i])*i)
        
    return np.uint8(out)

import PIL
from PIL import Image

class SegSemDataset:
    def __init__(self):
        self.names = []
        self.root = ""
        self.images = {}
        self.masks = {}
        self.sizes = {}
        self.setofcolors = []
    
    def getcolors(self):
        return self.setofcolors.copy()
        
    def getrandomtiles(self,nbtilesperimage,h,w,batchsize):
        #crop
        XY = []
        for name in self.names:
            col = np.random.randint(0,self.sizes[name][0]-w-2,size = nbtilesperimage)
            row = np.random.randint(0,self.sizes[name][1]-h-2,size = nbtilesperimage)
                   
            image = PIL.Image.open(self.root+"/"+self.images[name]).convert("RGB").copy()
            image = np.asarray(image,dtype=np.uint8) #warning wh swapping

            label = PIL.Image.open(self.root+"/"+self.masks[name]).convert("RGB").copy()
            label = np.asarray(label,dtype=np.uint8) #warning wh swapping
            for i in range(nbtilesperimage):
                im = image[row[i]:row[i]+h,col[i]:col[i]+w,:].copy()
                mask = label[row[i]:row[i]+h,col[i]:col[i]+w,:].copy()
                XY.append((im,mask))
                        
        #symetrie
        symetrieflag = np.random.randint(0,2,size = (len(XY),3))
        XY = [(symetrie(x,y,symetrieflag[i][0],symetrieflag[i][1],symetrieflag[i][2])) for i,(x,y) in enumerate(XY)]

        #pytorch
        X = torch.stack([torch.Tensor(np.transpose(x,axes=(2, 0, 1))).cpu() for x,y in XY])
        tmp = [torch.from_numpy(self.colorvtTOvt(y)).long().cpu() for x,y in XY]
        Y = torch.stack(tmp)
        dataset = torch.utils.data.TensorDataset(X,Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
        
        return dataloader
        
    def apply(self,model, tilesize, stride, pathout=""):
        out = []
        with torch.no_grad():
            for imageindex,name in enumerate(self.names):
                model.eval()
                
                im = PIL.Image.open(self.root+"/"+self.images[name]).convert("RGB").copy()
                im = np.asarray(im,dtype=np.uint8)#warning wh swapping

                realmask = PIL.Image.open(self.root+"/"+self.masks[name]).convert("RGB").copy()
                realmask = np.asarray(realmask,dtype=np.uint8)#warning wh swapping
                
                imagetensor = torch.Tensor(np.transpose(im,axes=(2, 0, 1))).cpu().unsqueeze(0)
                if torch.cuda.is_available():
                    model.cuda()
                    imagetensor = imagetensor.cuda()
                
                if imagetensor.shape[2]<=3*tilesize and imagetensor.shape[2]<=3*tilesize:
                    #only for small image
                    size = (imagetensor.shape[2]//32)*32,(imagetensor.shape[3]//32)*32
                    imagetensor = F.interpolate(imagetensor,size)
                    mask = model(imagetensor).cpu()
                else:
                    #resize and cut in tile
                    size = (imagetensor.shape[2]//tilesize)*tilesize,(imagetensor.shape[3]//tilesize)*tilesize
                    imagetensor = F.interpolate(imagetensor,size)
                    
                    mask = torch.zeros(1,len(self.setofcolors),imagetensor.shape[2],imagetensor.shape[3]).cpu()
                    for row in range(0,size[0]-tilesize+1,stride):
                        for col in range(0,size[1]-tilesize+1,stride):
                            mask[:,:,row:row+tilesize,col:col+tilesize] += model(imagetensor[:,:,row:row+tilesize,col:col+tilesize]).cpu()
                
                mask = F.interpolate(mask,im.shape[0:2])
                mask = mask[0]
                _,mask = mask.max(0)
                
                if pathout!="":
                    image = PIL.Image.fromarray(self.vtTOcolorvt(mask.numpy()))
                    image.save(pathout+"/"+name+"_z.jpg")
                    
                    if imageindex%10==0:
                        image = PIL.Image.fromarray(im)
                        image.save(pathout+"/"+name+"_x.jpg")
                        image = PIL.Image.fromarray(self.vtTOcolorvt(self.colorvtTOvt(realmask)))
                        image.save(pathout+"/"+name+"_y.jpg")
            
            out.append((mask.numpy(),self.colorvtTOvt(realmask)))
        return out
            
    def vtTOcolorvt(self,mask):
        maskcolor = np.zeros((mask.shape[0],mask.shape[1],3),dtype=int)
        for i in range(len(self.setofcolors)):
            for ch in range(3):
                maskcolor[:,:,ch]+=((mask == i).astype(int))*self.setofcolors[i][ch]
        return safeuint8(maskcolor)

    def colorvtTOvt(self,maskcolor):
        mask = np.zeros((maskcolor.shape[0],maskcolor.shape[1]),dtype=int)
        for i in range(len(self.setofcolors)):
            mask+=i*(maskcolor[:,:,0]==self.setofcolors[i][0]).astype(int)*(maskcolor[:,:,1]==self.setofcolors[i][1]).astype(int)*(maskcolor[:,:,2]==self.setofcolors[i][2]).astype(int)
        return mask


def resizeDataset(XY,path,setofcolors,nativeresolution,resolution,color,normalize,pathTMP="build"):
    XYout = {}
    for name,(x,y) in XY.items():
        image = PIL.Image.open(path+"/"+x).convert("RGB").copy()
        image = image.resize((int(image.size[0]*nativeresolution/resolution),int(image.size[1]*nativeresolution/resolution)), PIL.Image.BILINEAR)
        if not color:
            image = torchvision.transforms.functional.to_grayscale(image,num_output_channels=1)
        if normalize:
            image = np.asarray(image,dtype=np.uint8)
            image = normalizehistogram(image)
            image = PIL.Image.fromarray(np.stack([image,image,image],axis=-1))
        image.save(pathTMP+"/"+name+"_x.png")
        
        maskc = PIL.Image.open(path+"/"+y).convert("RGB").copy()
        maskc = maskc.resize((int(maskc.size[0]*nativeresolution/resolution),int(maskc.size[1]*nativeresolution/resolution)), PIL.Image.NEAREST)
        maskc.save(pathTMP+"/"+name+"_y.png")
        
        XYout[name] = (name+"_x.png",name+"_y.png")
    
    out = SegSemDataset()
    out.root = pathTMP
    out.setofcolors = setofcolors
    for name,(x,y) in XYout.items():
        out.names.append(name)
        out.sizes[name] = getsize(pathTMP+"/"+x)

        out.images[name] = x
        out.masks[name] = y
        
    return out
    

class Unet(nn.Module):
    def __init__(self,nbclasses):
        super(Unet, self).__init__()
        
        self.nbClasses = nbclasses
        self.transform=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.gradientdoor16 = nn.Conv2d(512, self.nbClasses, kernel_size=1)

        self.conv43d = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        self.gradientdoor8 = nn.Conv2d(256, self.nbClasses, kernel_size=1)

        self.conv33d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        
        self.gradientdoor4 = nn.Conv2d(128, self.nbClasses, kernel_size=1)

        self.conv22d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.gradientdoor2 = nn.Conv2d(64, self.nbClasses, kernel_size=1)

        self.conv12d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv11d = nn.Conv2d(64, self.nbClasses, kernel_size=3, padding=1)
      
      
    def forwardaux(self, x):
        x = x/255
        x = torch.stack([x[:,0,:,:]-self.transform[0][0],x[:,1,:,:]-self.transform[0][1],x[:,2,:,:]-self.transform[0][2]],dim=1)
        x = torch.stack([x[:,0,:,:]/self.transform[1][0],x[:,1,:,:]/self.transform[1][1],x[:,2,:,:]/self.transform[1][2]],dim=1)
        
        x1 = F.leaky_relu(self.conv11(x))
        x1 = F.leaky_relu(self.conv12(x1))
        x1p = F.max_pool2d(x1, kernel_size=2, stride=2)

        x2 = F.leaky_relu(self.conv21(x1p))
        x2 = F.leaky_relu(self.conv22(x2))
        x2p = F.max_pool2d(x2, kernel_size=2, stride=2)

        x3 = F.leaky_relu(self.conv31(x2p))
        x3 = F.leaky_relu(self.conv32(x3))
        x3 = F.leaky_relu(self.conv33(x3))
        x3p = F.max_pool2d(x3, kernel_size=2, stride=2)

        x4 = F.leaky_relu(self.conv41(x3p))
        x4 = F.leaky_relu(self.conv42(x4))
        x4 = F.leaky_relu(self.conv43(x4))
        x4p = F.max_pool2d(x4, kernel_size=2, stride=2)

        x5 = F.leaky_relu(self.conv51(x4p))
        x5 = F.leaky_relu(self.conv52(x5))
        x5 = F.leaky_relu(self.conv53(x5))
        
        x_grad_16 = self.gradientdoor16(x5)

        x5u = F.upsample_nearest(x5, scale_factor=2)
        x4 = torch.cat((x5u, x4), 1)
        
        x4 = F.leaky_relu(self.conv43d(x4))
        x4 = F.leaky_relu(self.conv42d(x4))
        x4 = F.leaky_relu(self.conv41d(x4))
        
        x_grad_8 = self.gradientdoor8(x4)

        x4u = F.upsample_nearest(x4, scale_factor=2)
        x3 = torch.cat((x4u, x3), 1)
        
        x3 = F.leaky_relu(self.conv33d(x3))
        x3 = F.leaky_relu(self.conv32d(x3))
        x3 = F.leaky_relu(self.conv31d(x3))
        
        x_grad_4 = self.gradientdoor4(x3)
        
        x3u = F.upsample_nearest(x3, scale_factor=2)
        x2 = torch.cat((x3u, x2), 1)
        
        x2 = F.leaky_relu(self.conv22d(x2))
        x2 = F.leaky_relu(self.conv21d(x2))
        
        x_grad_2 = self.gradientdoor2(x2)

        x2u = F.upsample_nearest(x2, scale_factor=2)
        x1 = torch.cat((x2u, x1), 1)
        
        x1 = F.leaky_relu(self.conv12d(x1))
        x = self.conv11d(x1)
        return x, (x_grad_2,x_grad_4,x_grad_8,x_grad_16)
        
    def forward(self, x):
        return self.forwardaux(x)[0]

    def pool2GT(self,targets):
        resizedtarget = torch.zeros(targets.shape[0],targets.shape[1]//2,targets.shape[2]//2)
        for i in range(targets.shape[0]):
            maskc = PIL.Image.fromarray(np.uint8(targets[i].cpu().numpy()))
            maskc = maskc.resize((maskc.size[0]//2,maskc.size[1]//2), PIL.Image.NEAREST)
            resizedtarget[i] = torch.from_numpy(np.asarray(maskc,dtype=np.uint8))
        if torch.cuda.is_available():
            return resizedtarget.cuda().long()
        else:
            return resizedtarget.long()
            
    def computeloss(self, inputs, targets, criterion):
        self.train()
        prob,gradientdoor = self.forwardaux(inputs)
        
        prob2,prob4,prob8,prob16 = gradientdoor
        allprob = [prob,prob2,prob4,prob8,prob16]
        
        targets2 = self.pool2GT(targets)
        targets4 = self.pool2GT(targets2)
        targets8 = self.pool2GT(targets4)
        target16 = self.pool2GT(targets8)
        alltarget = [targets,targets2,targets4,targets8,target16]
        
        losses = [criterion(torch.flatten(allprob[i],start_dim=2),torch.flatten(alltarget[i],start_dim=1).long())*pow(0.8,i) for i in range(5)]
        loss=sum(losses)
        return loss,prob

    def loadpretrained(self,path):
        correspondance=[]
        correspondance.append(("features.0","conv11"))
        correspondance.append(("features.2","conv12"))
        correspondance.append(("features.5","conv21"))
        correspondance.append(("features.7","conv22"))
        correspondance.append(("features.10","conv31"))
        correspondance.append(("features.12","conv32"))
        correspondance.append(("features.14","conv33"))
        correspondance.append(("features.17","conv41"))
        correspondance.append(("features.19","conv42"))
        correspondance.append(("features.21","conv43"))
        correspondance.append(("features.24","conv51"))
        correspondance.append(("features.26","conv52"))
        correspondance.append(("features.28","conv53"))       

        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
                
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
        self.load_state_dict(model_dict)   


class Deeplabv3(nn.Module):
    def __init__(self,nbclasses):
        super(Deeplabv3, self).__init__()
        
        self.net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        self.net.classifier._modules["4"] = nn.Conv2d(256, nbclasses, 1)
        
    def forward(self,x):
        return self.net["out"]


def DFC2015color(mode):
    if mode=="normal":
        return [[255,255,255]
        ,[0,0,128]
        ,[255,0,0]
        ,[0,255,255]
        ,[0,0,255]
        ,[0,255,0]
        ,[255,0,255]
        ,[255,255,0]]
    else:#lod0
        return [[255,255,255],[0,0,255]]

def maketrainDFC2015(resolution=50,datasetpath="/data/DFC2015",mode="normal",color = True,normalize = False):
    XY = {}
    XY["1"]=("BE_ORTHO_27032011_315130_56865.tif","label_315130_56865.tif")
    XY["2"]=("BE_ORTHO_27032011_315130_56870.tif","label_315130_56870.tif")
    XY["3"]=("BE_ORTHO_27032011_315135_56870.tif","label_315135_56870.tif")
    XY["4"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    
    return resizeDataset(XY,datasetpath,DFC2015color(mode),5,resolution,color,normalize)
    
def maketestDFC2015(resolution=50,datasetpath="/data/DFC2015",mode="normal",color = True,normalize = False):
    XY = {}
    XY["5"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    XY["6"]=("BE_ORTHO_27032011_315145_56865.tif","label_315145_56865.tif")
    
    return resizeDataset(XY,datasetpath,DFC2015color(mode),5,resolution,color,normalize)
    

def ISPRScolor(mode):
    if mode=="normal":
        return [[255, 255, 255]
        ,[0, 0, 255]
        ,[0, 255, 255]
        ,[ 0, 255, 0]
        ,[255, 255, 0]
        ,[255, 0, 0]]
    else:#lod0
        return [[255,255,255],[0,0,255]]

def makeISPRStrainVAIHINGEN(resolution=50,datasetpath,mode="lod0",color = False,normalize = True):
    names = ["top_mosaic_09cm_area5.tif",
    "top_mosaic_09cm_area17.tif",
    "top_mosaic_09cm_area21.tif",
    "top_mosaic_09cm_area23.tif",
    "top_mosaic_09cm_area26.tif",
    "top_mosaic_09cm_area28.tif",
    "top_mosaic_09cm_area30.tif",
    "top_mosaic_09cm_area32.tif",
    "top_mosaic_09cm_area34.tif",
    "top_mosaic_09cm_area37.tif"]
    
    XY = {}
    for name in names:
        XY[name]=("top/"+name,"gts_for_participants/"+name)
    
    return resizeDataset(XY,datasetpath,ISPRScolor(mode),9,resolution,color,normalize)
    
def makeISPRStestVAIHINGEN(resolution=50,datasetpath,mode="lod0",color = False,normalize = True):
    names = ["top_mosaic_09cm_area1.tif",
    "top_mosaic_09cm_area3.tif",
    "top_mosaic_09cm_area7.tif",
    "top_mosaic_09cm_area11.tif",
    "top_mosaic_09cm_area13.tif",
    "top_mosaic_09cm_area15.tif"]
    
    XY = {}
    for name in names:
        XY[name]=("top/"+name,"gts_for_participants/"+name)
    
    return resizeDataset(XY,datasetpath,ISPRScolor(mode),9,resolution,color,normalize)
        
import os   

def makeAIRSdataset(datasetpath,resolution=50,color = False,normalize = True):
    allfile = os.listdir(datasetpath+"/image")
    XY = {}
    for name in allfile:
        XY[name] = ("image/"+name,"label/"+name[0:-4]+"_vis.tif")
    return resizeDataset(XY,datasetpath, [[0,0,0],[255,255,255]],7.5,resolution,color,normalize)

def makeINRIAdataset(mode,resolution=50,color = False,normalize = True):
    datasetpath = "TODO"
    allfile = os.listdir(datasetpath+"/images")
    allfile = sorted(allfile)
    XY = {}
    for name in allfile:
        XY[name] = ("images/"+name,"gt/"+name)
    if mode=="Train":
        allfile = allfile[0:200]
    if mode=="Test":
        allfile = allfile[200:250]
    if mode not in ["Train","Test"]:
        allfile = allfile
    return resizeDataset(XY,datasetpath, [[0,0,0],[255,255,255]],9,resolution,color,normalize)
    
