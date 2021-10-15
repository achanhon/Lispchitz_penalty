import os
import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
import dataloader

aed = dataloader.AED(flag="test")

print("test")
import numpy
import PIL
from PIL import Image

stats = torch.zeros(3).cuda()
with torch.no_grad():
    for name in aed.names:
        x, y = aed.getImageAndLabel(name, torchformat=True)
        z = net(x.cuda())
        z = z[0, 1, :, :] - z[0, 0, :, :]

        stats += dataloader.computeperf(yz=(y.cuda(), z))

        if True:
            nextI = len(os.listdir("build"))
            debug = numpy.transpose(x[0].cpu().numpy(), axes=(1, 2, 0))
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_x.png")
            globalresize = torch.nn.AdaptiveAvgPool2d((x.shape[2], x.shape[3]))
            debug = globalresize(y.unsqueeze(0).float())
            debug = debug[0].cpu().numpy() * 255
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_y.png")
            debug = globalresize((z > 0).float().unsqueeze(0))
            debug = debug[0].cpu().numpy() * 255
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_z.png")

    print("perf=", dataloader.computeperf(stats=stats))

os._exit(0)
