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
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()


print("massif benchmark")
import dataloader

if True:
    miniworld = dataloader.MiniWorld(flag="custom", custom=["toulouse/test"])
else:
    miniworld = dataloader.MiniWorld("test")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def f1(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


from skimage import measure

CROPSIZE = 64


def select_rootcentredpatch(image, label):
    blobs_image = measure.label(label, background=0)

    blobs = measure.regionprops(blobs_image)
    blobs = [c for c in blobs if 64 <= c.area and c.area <= 256]

    tmp = []
    for blob in blobs:
        r, c = blob.centroid
        r, c = int(r), int(c)
        if (
            r <= CROPSIZE + 1
            or r + CROPSIZE + 1 >= label.shape[0]
            or c <= CROPSIZE + 1
            or c + CROPSIZE + 1 >= label.shape[1]
        ):
            continue
        else:
            tmp.append(blob)
    blobs = tmp

    if blobs == []:
        return None

    if len(blobs) > 100:
        random_shuffle(blobs)
        blobs = blobs[0:100]

    XYA = []
    for blob in blobs:
        r, c = blob.centroid
        r, c = int(r), int(c)
        x, y = (
            image[r - CROPSIZE : r + CROPSIZE, c - CROPSIZE : c + CROPSIZE, :].copy(),
            label[r - CROPSIZE : r + CROPSIZE, c - CROPSIZE : c + CROPSIZE].copy(),
        )
        a = blobs_image[r - CROPSIZE : r + CROPSIZE, c - CROPSIZE : c + CROPSIZE].copy()

        a = np.uint8(a == blob.label)
        XYA.append((x, y, a))

    return XYA


weights = torch.Tensor([1, miniworld.balance, 0.00001]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)


def do_a_very_good_adversarial_attack(x, y, a):
    xa = x.clone()

    for i in range(100):
        xaa = xa.clone().detach().requires_grad_()
        optimizer = torch.optim.SGD([xaa], lr=1)
        preds = net(xaa)

        tmp = torch.zeros(preds.shape[0], 1, preds.shape[2], preds.shape[3])
        tmp = tmp.to(device)
        preds = torch.cat([preds, tmp], dim=1)

        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()

        grad = xaa.grad
        grad = torch.sign(grad * 1000)
        a3 = torch.stack([a, a, a], dim=1)
        xaa = xaa + grad * a3

        x0 = torch.zeros(xa.shape).cuda()
        x255 = torch.ones(xa.shape).cuda() * 255
        xa = torch.max(x0, torch.min(xaa, x255)).clone()

    tmp = torch.sum((x - xa).abs()) / x.shape[0]
    print(tmp)
    return xa


cm = {}
cmattack = {}
if True:
    for town in miniworld.towns:
        print(town)
        XYA = []
        for i in range(miniworld.data[town].nbImages):
            imageraw, label = miniworld.data[town].getImageAndLabel(i)
            XYA += select_rootcentredpatch(imageraw, label)

        # pytorch
        X = torch.stack(
            [torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu() for x, _, _ in XYA]
        )
        Y = torch.stack([torch.from_numpy(y).long().cpu() for _, y, _ in XYA])
        Y = dataloader.convertIn3class(Y)
        A = torch.stack([torch.from_numpy(a).long().cpu() for _, _, a in XYA])
        del XYA

        XYAtensor = torch.utils.data.TensorDataset(X, Y, A)
        pytorchloader = torch.utils.data.DataLoader(
            XYAtensor, batch_size=16, shuffle=False, num_workers=2
        )

        ZXaZa = []
        for inputs, targets, masks in pytorchloader:
            inputs, targets, masks = inputs.cuda(), targets.cuda(), masks.cuda()
            with torch.no_grad():
                preds = net(inputs)
                _, preds = torch.max(preds, 1)

            # with grad
            adversarial = do_a_very_good_adversarial_attack(inputs, targets, masks)

            with torch.no_grad():
                predsa = net(adversarial)
                _, predsa = torch.max(predsa, 1)

            ZXaZa.append((preds, adversarial, predsa))

        Z = torch.cat([z for z, _, _ in ZXaZa], dim=0)
        Xa = torch.cat([xa for _, xa, _ in ZXaZa], dim=0)
        Za = torch.cat([za for _, _, za in ZXaZa], dim=0)

        cm[town] = np.zeros((3, 3), dtype=int)
        cmattack[town] = np.zeros((3, 3), dtype=int)
        for i in range(X.shape[0]):
            x = np.transpose(X[i].cpu().numpy(), axes=(1, 2, 0))
            im = PIL.Image.fromarray(np.uint8(x))
            im.save("build/" + town[0:-5] + "_" + str(i) + "_x.png")

            im = PIL.Image.fromarray(np.uint8(Y[i].cpu().numpy() * 125))
            im.save("build/" + town[0:-5] + "_" + str(i) + "_y.png")

            im = PIL.Image.fromarray(np.uint8(A[i].cpu().numpy() * 125))
            im.save("build/" + town[0:-5] + "_" + str(i) + "_a.png")

            im = PIL.Image.fromarray(np.uint8(Z[i].cpu().numpy() * 125))
            im.save("build/" + town[0:-5] + "_" + str(i) + "_z.png")

            im = PIL.Image.fromarray(np.uint8(Za[i].cpu().numpy() * 125))
            im.save("build/" + town[0:-5] + "_" + str(i) + "_p.png")

            xa = np.transpose(Xa[i].detach().cpu().numpy(), axes=(1, 2, 0))
            im = PIL.Image.fromarray(np.uint8(xa))
            im.save("build/" + town[0:-5] + "_" + str(i) + "_q.png")

            cm[town] += confusion_matrix(
                Y[i].cpu().numpy().flatten(),
                Z[i].cpu().numpy().flatten(),
                labels=[0, 1, 2],
            )
            cmattack[town] += confusion_matrix(
                Y[i].cpu().numpy().flatten(),
                Za[i].cpu().numpy().flatten(),
                labels=[0, 1, 2],
            )

        cm[town] = cm[town][0:2, 0:2]
        print(cm[town][0][0], cm[town][0][1], cm[town][1][0], cm[town][1][1])
        print(
            accu(cm[town]),
            f1(cm[town]),
        )
        cmattack[town] = cmattack[town][0:2, 0:2]
        print(
            cmattack[town][0][0],
            cmattack[town][0][1],
            cmattack[town][1][0],
            cmattack[town][1][1],
        )
        print(
            accu(cmattack[town]),
            f1(cmattack[town]),
        )
        print(
            "number of modified pixel",
            torch.sum(A),
            "/",
            Y.shape[0] * Y.shape[1] * Y.shape[2],
            "=",
            torch.sum(A) / (Y.shape[0] * Y.shape[1] * Y.shape[2]),
        )
