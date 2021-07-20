import os
import sys
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

whereIam = os.uname()[1]

print("load model")
if whereIam == "ldtis706z":
    sys.path.append("/home/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/pytorch-image-models")
    sys.path.append("/home/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/home/achanhon/github/segmentation_models.pytorch")
if whereIam == "super":
    sys.path.append("/home/achanhon/github/segmentation_models/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/segmentation_models/pytorch-image-models")
    sys.path.append(
        "/home/achanhon/github/segmentation_models/pretrained-models.pytorch"
    )
    sys.path.append(
        "/home/achanhon/github/segmentation_models/segmentation_models.pytorch"
    )
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
import segmentation_models_pytorch

with torch.no_grad():
    if len(sys.argv) > 1:
        net = torch.load(sys.argv[1])
    else:
        net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()


print("load data")
import dataloader

cia = dataloader.CIA("train")

earlystopping = cia.getrandomtiles(128, 16)

print("test on training crop")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def f1(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


def trainaccuracy():
    cm = np.zeros((3, 3), dtype=int)
    net.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(earlystopping):
            inputs, targets = inputs.to(device), targets.to(device)

            targets = dataloader.convertIn3class(targets)

            outputs = net(inputs)
            _, pred = outputs.max(1)

            rahh = torch.transpose(inputs, 1, 2)
            rahh = torch.transpose(rahh, 2, 3)

            for j in range(pred.shape[0]):
                imageraw = PIL.Image.fromarray(np.uint8(rahh[j].cpu().numpy()))
                imageraw.save("build/" + str(i * 16 + j) + "_x.png")
                labelim = PIL.Image.fromarray(np.uint8(targets[j].cpu().numpy()) * 125)
                labelim.save("build/" + str(i * 16 + j) + "_y.png")
                predim = PIL.Image.fromarray(np.uint8(pred[j].cpu().numpy()) * 125)
                predim.save("build/" + str(i * 16 + j) + "_z.png")

                cm += confusion_matrix(
                    pred[j].cpu().numpy().flatten(),
                    targets[j].cpu().numpy().flatten(),
                    labels=[0, 1, 2],
                )
    return cm[0:2, 0:2]


cm = trainaccuracy()
print("accuracy and IoU", accu(cm), f1(cm))
