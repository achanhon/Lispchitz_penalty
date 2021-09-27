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
    net = net.cuda()
    net.eval()

print("load data")
cia = dataloader.CIA(flag="custom", custom=["isprs/test"])

print("test")
import numpy
import PIL
from PIL import Image


def perf(cm):
    if len(cm.shape) == 1:
        precision = cm[0] / (cm[0] + cm[1] + 1)
        recall = cm[0] / (cm[0] + cm[2] + 1)
        g = precision * recall
        return torch.Tensor((g * 100, precision * 100, recall * 100))
    else:
        out = torch.zeros(cm.shape[0], 3)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


cm = torch.zeros((len(cia.towns), 3)).cuda()
distanceVT = dataloader.DistanceVT()
headNMS = dataloader.HardNMS()
with torch.no_grad():
    for k, town in enumerate(cia.towns):
        print(k, town)
        for i in range(cia.data[town].nbImages):
            imageraw, label = cia.data[town].getImageAndLabel(i)

            y = torch.Tensor(label).cuda().float()
            h, w = y.shape[0], y.shape[1]
            DVT = distanceVT(y.unsqueeze(0))[0][0]

            x = torch.Tensor(numpy.transpose(imageraw, axes=(2, 0, 1))).unsqueeze(0)
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = dataloader.largeforward(net, x)

            z = globalresize(z)
            zNMS = headNMS(z)[0][0]

            cm[k][0] += torch.sum((zNMS > 0).float() * (y == 1).float())
            cm[k][1] += torch.sum((zNMS > 0).float() * (y == 0).float() * (1 - DVT) / 9)
            cm[k][2] += torch.sum((zNMS == 0).float() * (y == 1).float())

            if town in ["isprs/test", "saclay/test"]:
                nextI = len(os.listdir("build"))
                debug = globalresize(x)[0].cpu().numpy()
                debug = numpy.transpose(debug, axes=(1, 2, 0))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                debug = torch.nn.functional.max_pool2d(
                    y.unsqueeze(0), kernel_size=3, stride=1, padding=1
                )
                debug = (debug[0] * 255 * DVT).cpu().numpy()
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = (z[0, 1, :, :] > z[0, 0, :, :]).float()
                debug = debug.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")
                debug = (zNMS > 0).float()
                debug = debug.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_{.png")

        print("perf=", perf(cm[k]))
        numpy.savetxt("build/logtest.txt", perf(cm).cpu().numpy())

print("-------- results ----------")
for k, town in enumerate(cia.towns):
    print(town, perf(cm[k]))

cm = torch.sum(cm, dim=0)
print("cia", perf(cm))
