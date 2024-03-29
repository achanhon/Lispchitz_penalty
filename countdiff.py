import os
import sys
import numpy
import PIL
from PIL import Image
import torch

if len(sys.argv) >= 2:
    path, denom = sys.argv[1], int(sys.argv[2])
else:
    path, denom = "build", 1

l = os.listdir(path)
names = [s[0:-5] for s in l if "y.png" in s]
l = set(l)

names = [s for s in names if s + "z.png" in l and s + "y.png" in l]
print(len(names))

totPred, totVT, totdiff, totdiff2 = 0, 0, 0, 0
for name in names:
    label = PIL.Image.open(path + "/" + name + "y.png").convert("L")
    label = torch.Tensor(numpy.asarray(label))
    label = (label > 0).float().sum() / denom

    pred = PIL.Image.open(path + "/" + name + "z.png").convert("L")
    pred = torch.Tensor(numpy.asarray(pred))
    pred = (pred > 0).float().sum() / denom

    diff = abs(label - pred)
    totPred += pred
    totVT += label
    totdiff += diff
    totdiff2 += diff * diff

totVT /= len(names)
totPred /= len(names)
totdiff /= len(names)
totdiff2 /= len(names)
print(totdiff, totdiff2, totVT, totPred)
