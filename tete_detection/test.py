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

whereIam = os.uname()[1]
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

import segmentation_models_pytorch as smp
import dataloader

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    headNMS = dataloader.HardNMS()
    net = net.cuda()
    net.eval()

print("load data")
cia = dataloader.CIA(flag="custom", custom=["isprs/test", "saclay/test"])

print("test")
import numpy
import PIL
from PIL import Image

cm = torch.zeros((len(cia.towns), 3)).cuda()
with torch.no_grad():
    for k, town in enumerate(cia.towns):
        print(k, town)
        for i in range(cia.data[town].nbImages):
            imageraw, label = cia.data[town].getImageAndLabel(i)

            x = torch.Tensor(numpy.transpose(imageraw, axes=(2, 0, 1))).cuda()
            y = torch.Tensor(label).cuda()

            z, s = net(x)
            good, fa, miss = net.computegscore(z, y)

            cm[k][0] += good
            cm[k][1] += fa
            cm[k][2] += miss

            if town in ["isprs/test", "saclay/test"]:
                nextI = len(os.listdir("build"))
                debug = numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                debug = y.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = (s[0, 1, :, :] > s[0, 0, :, :]).float()
                debug = debug.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_Z.png")
                debug = (z > 0).float()
                debug = debug.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")

        print("perf=", dataloader.computegscore(cm[k]))
        numpy.savetxt("build/logtest.txt", perf(cm).cpu().numpy())

print("-------- results ----------")
for k, town in enumerate(cia.towns):
    print(town, perf(cm[k]))

cm = torch.sum(cm, dim=0)
print("cia", perf(cm))
