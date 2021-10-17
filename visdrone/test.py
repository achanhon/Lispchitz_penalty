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
import dataloader
import detectionhead

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
visdrone = dataloader.VISDRONE(flag="test")

print("test")
import numpy
import PIL
from PIL import Image

stats = torch.zeros(3).cuda()
with torch.no_grad():
    for name in visdrone.names:
        x, y = visdrone.getImageAndLabel(name, torchformat=True)

        z, s = net(x)
        stats += net.computegscore(z, y.cuda())

        if True:
            nextI = len(os.listdir("build"))
            debug = numpy.transpose(x[0].cpu().numpy(), axes=(1, 2, 0))
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_x.png")
            debug = y.cpu().numpy() * 255
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_y.png")
            debug = (s > 0).float()
            debug = debug.cpu().numpy() * 255
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_Z.png")
            debug = (z[0] > 0).float()
            debug = debug.cpu().numpy() * 255
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_z.png")

        print("perf=", dataloader.computeperf(stats))

os._exit(0)
