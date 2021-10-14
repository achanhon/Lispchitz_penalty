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

print("load model")
import dataloader

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

stats = torch.zeros((len(cia.towns), 3)).cuda()
with torch.no_grad():
    for k, town in enumerate(cia.towns):
        print(k, town)
        for i in range(cia.data[town].nbImages):
            imageraw, label = cia.data[town].getImageAndLabel(i)

            x = torch.Tensor(numpy.transpose(imageraw, axes=(2, 0, 1))).cuda()
            y = torch.Tensor(label).cuda()

            z = net(x.unsqueeze(0))
            z = z[0, 1, :, :] - z[0, 0, :, :]

            stats[k][0] += torch.sum((z > 0).float() * (y == 1).float())
            stats[k][1] += torch.sum((z > 0).float() * (y == 0).float())
            stats[k][2] += torch.sum((z <= 0).float() * (y == 1).float())

            if town in ["isprs/test", "saclay/test"]:
                nextI = len(os.listdir("build"))
                debug = numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                globalresize = torch.nn.AdaptiveAvgPool2d((x.shape[1], x.shape[2]))
                debug = globalresize(y.unsqueeze(0).float())
                debug = debug[0].cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = globalresize((z > 0).float().unsqueeze(0))
                debug = debug[0].cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")

        print("perf=", dataloader.computeperf(stats[k]))
        numpy.savetxt("build/logtest.txt", dataloader.computeperf(stats).cpu().numpy())

print("-------- results ----------")
for k, town in enumerate(cia.towns):
    print(town, dataloader.computeperf(stats[k]))

stats = torch.sum(stats, dim=0)
print("cia", dataloader.computeperf(stats))
