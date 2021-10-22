import os
import sys
import torch

sys.path.append("/scratchm/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/scratchm/achanhon/github/pytorch-image-models")
sys.path.append("/scratchm/achanhon/github/pretrained-models.pytorch")
sys.path.append("/scratchm/achanhon/github/segmentation_models.pytorch")


class PoolWithHole(torch.nn.Module):
    def __init__(self):
        super(PoolWithHole, self).__init__()

    def forward(self, x):
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        if torch.cuda.is_available():
            Xm = torch.zeros(B, H + 2, W + 2).cuda()
        else:
            Xm = torch.zeros(B, H + 2, W + 2)
        X = [Xm.clone() for i in range(9)]
        for i in range(3):
            for j in range(3):
                if i != 1 or j != 1:
                    X[i * 3 + j][:, i : i + H, j : j + W] = x[:, :, :]

        for i in range(9):
            Xm = torch.max(Xm, X[i])
        return Xm[:, 1 : 1 + H, 1 : 1 + W]


class Detector(torch.nn.Module):
    def __init__(self, backbone):
        super(Detector, self).__init__()
        self.backbone = backbone
        self.pool = PoolWithHole()

    def segmentationforward(self, x):
        tile, stride = 256, 64
        h, w = x.shape[2], x.shape[3]
        h64, w64 = ((h // stride) * stride, (w // stride) * stride)

        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d((h64, w64))
        x = power2resize(x)

        with torch.no_grad():
            if torch.cuda.is_available():
                pred = torch.zeros(x.shape[0], 2, h64, w64).cuda()
            else:
                pred = torch.zeros(x.shape[0], 2, h64, w64)
            for row in range(0, h64 - tile + 1, stride):
                for col in range(0, w64 - tile + 1, stride):
                    tmp = self.backbone(x[:, :, row : row + tile, col : col + tile])
                    pred[:, :, row : row + tile, col : col + tile] += tmp[0]
        return globalresize(pred)

    def forward(self, x):
        segmentation = self.segmentationforward(x)
        x = segmentation[:, 1, :, :] - segmentation[:, 0, :, :]
        xp = torch.nn.functional.relu(x)
        xm = self.pool(xp)
        localmax = (x > xm).float()
        return xp * localmax


import segmentation_models_pytorch as smp

tmp = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights=None,
    in_channels=3,
    classes=2,
)
if torch.cuda.is_available():
    weights = torch.load("build/state_dict.pth")
else:
    weights = torch.load("build/state_dict.pth", map_location=torch.device("cpu"))
tmp.load_state_dict(weights)

net = Detector(tmp)
net.eval()

if torch.cuda.is_available():
    net = net.cuda()
    tmp = torch.zeros(5, 3, 512, 512).cuda()
else:
    tmp = torch.zeros(5, 3, 512, 512)
print(net(tmp).shape)
