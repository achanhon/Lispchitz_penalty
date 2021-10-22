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
        Xm = torch.zeros(B, H + 2, W + 2).cuda()
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

    def forward(self, x):
        segmentation = self.backbone(x)
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
weights = torch.load("build/state_dict.pth")
tmp.load_state_dict(weights)

net = Detector(tmp)

tmp = torch.zeros(1, 3, 256, 256)
print(net(tmp).shape)
