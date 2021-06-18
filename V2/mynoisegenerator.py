import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from sklearn.metrics import confusion_matrix


def implementedperturbation():
    return [
        "FSGM",
        "randomsign",
        "gaussian",
        "pepperandsalt",
        "uniform",
        "blur",
        "center",
        "identity",
    ]


def ensureimage(x, device="cuda"):
    return torch.min(
        255 * torch.ones(x.shape).to(device),
        torch.max(torch.round(x), torch.zeros(x.shape).to(device)),
    )


def perturbinput(x, y, f, level, mode, nbclasses):
    assert mode in implementedperturbation()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        f, x, y = f.to(device), x.to(device), y.to(device)

    if mode == "uniform":
        xe = x + (torch.rand(x.shape).to(device) * 2 - 0.5) * level
        xe = ensureimage(xe)

    if mode == "gaussian":
        xe = x + torch.randn(x.shape).to(device) * level
        xe = ensureimage(xe)

    if mode == "center":
        tmp = torch.sign(x - 125)
        xe = x + level * tmp
        xe = ensureimage(xe)

    if mode == "blur":
        xe = x.clone()
        seuil = x.shape[3] * x.shape[2] * 3 * level
        while torch.sum((xe - x).abs()) < seuil:
            xe = F.avg_pool2d(xe, kernel_size=3, padding=1, stride=1)
        xe = ensureimage(xe)

    if mode == "pepperandsalt":
        nbpixel = x.shape[3] * x.shape[2]
        nbbreak = int(level / 255 / 3 * nbpixel)
        row = np.random.randint(0, x.shape[2], size=nbbreak)
        col = np.random.randint(0, x.shape[3], size=nbbreak)
        pepperorsalt = np.random.randint(0, 2, size=nbbreak)
        xe = x.clone()
        for i in range(nbbreak):
            xe[:, :, row[i], col[i]] = torch.ones(1).to(device) * float(
                255 * pepperorsalt[i]
            )
        xe = ensureimage(xe)

    if mode == "randomsign":
        best = None
        bestaccuracy = 2
        with torch.no_grad():
            for i in range(500):
                xe = x + (torch.rand(x.shape).to(device) * 2 - 0.5) * level
                xe = ensureimage(xe)
                ze = f(xe)
                _, pred = ze.max(1)
                cm = confusion_matrix(
                    pred.cpu().detach().numpy().flatten(),
                    y.cpu().numpy().flatten(),
                    list(range(nbclasses)),
                )
                accuracy = np.sum(cm.diagonal()) / (np.sum(cm) + 1)
                if accuracy < bestaccuracy:
                    best = xe
                    bestaccuracy = accuracy
                    if accuracy == 0:
                        break
        xe = best

    if mode == "FSGM":
        criterion = nn.CrossEntropyLoss()
        xe = x.clone()

        for i in range(level):
            xe = xe.detach().requires_grad_()
            optimizer = optim.Adam([xe], lr=0.0001)
            loss = criterion(
                torch.flatten(f(xe), start_dim=2), torch.flatten(y, start_dim=1)
            )
            optimizer.zero_grad()
            loss.backward()

            grad = xe.grad.data
            grad = grad.sign()
            xe = xe + grad

    if mode == "identity":
        xe = x.copy()

    with torch.no_grad():
        ze = f(xe)
        _, pred = ze.max(1)
        d1, d2 = torch.sum((xe - x).abs()), torch.norm(xe - x)

    xe, ze, pred, y, d1, d2 = (
        xe.cpu().detach().numpy(),
        ze.cpu().detach().numpy(),
        pred.cpu().numpy(),
        y.cpu().numpy(),
        d1.cpu().detach().numpy(),
        d2.cpu().detach().numpy(),
    )
    cm = confusion_matrix(pred.flatten(), y.flatten(), list(range(nbclasses)))
    accuracy = np.sum(cm.diagonal()) / (np.sum(cm) + 1)
    return xe, ze, pred, cm, accuracy, d1 / x.size(), d2 / x.size()


def worseperturbation(x, y, f, level, nbclasses):
    bestaccuracy = 2
    best = None
    for mode in [
        "FSGM",
        "randomsign",
        "gaussian",
        "pepperandsalt",
        "uniform",
        "blur",
        "center",
    ]:
        xe, ze, pred, cm, accuracy, d1, d2 = perturbinput(
            x, y, f, level, mode, nbclasses
        )
        if accuracy < bestaccuracy:
            best = xe, ze, pred, cm, accuracy, d1, d2, mode
            bestaccuracy = accuracy
            if accuracy == 0:
                break
    return best
