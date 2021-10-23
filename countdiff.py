import os
import sys
import numpy
import PIL
from PIL import Image

if len(sys.argv) >= 1:
    path = sys.argv[1]
else:
    path = "build"

l = os.listdir(path)
names = [s[0:-5] for s in l if "y.png" in s]
l = set(l)

names = [s for s in names if s + "z.png" in l and s + "y.png" in l]

totPred, totVT, totdiff, totdiff2 = 0, 0, 0, 0
for name in names:
    label = PIL.Image.open(path + "/" + name + "y.png").convert("L")
    label = numpy.uint8(numpy.asarray(label) > 0)
    label = numpy.sum(label)

    pred = PIL.Image.open(path + "/" + name + "z.png").convert("L")
    pred = numpy.uint8(numpy.asarray(pred) > 0)
    pred = numpy.sum(pred)

    diff = abs(label - pred)
    totPred += pred
    totVT += label
    totdiff += diff
    totdiff2 += diff * diff

totVT /= len(name)
totPred /= len(name)
totdiff /= len(name)
totdiff2 /= len(name)
print(totdiff, totdiff2, totVT, totPred)
