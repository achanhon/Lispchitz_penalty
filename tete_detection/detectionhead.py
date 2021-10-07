import torch


def etendre(x, size):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=2 * size + 1, stride=1, padding=size
    )


def distancetransform(y, size=6):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]


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


class DetectionHead(torch.nn.Module):
    def __init__(self, backbone):
        super(DetectionHead, self).__init__()
        self.backbone = backbone
        self.pool = PoolWithHole()

    def headforward(self, x):
        xp = torch.nn.functional.relu(x)
        xm = self.pool(xp)

        localmax = (x > xm).float()
        return xp * localmax

    def largeforward(self, x):
        tile, stride = 128, 32
        h, w = x.shape[1], x.shape[2]
        h32, w32 = ((h // stride) * stride, (w // stride) * stride)

        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d((h32, w32))
        x = power2resize(x.unsqueeze(0))

        with torch.no_grad():
            pred = torch.zeros(1, 2, h32, w32).cuda()
            for row in range(0, h32 - tile + 1, stride):
                for col in range(0, w32 - tile + 1, stride):
                    tmp = self.backbone(x[:, :, row : row + tile, col : col + tile])
                    pred[0, :, row : row + tile, col : col + tile] += tmp[0]

        return globalresize(pred)

    def forward(self, x):
        if len(x.shape) == 3:
            segmentation = self.largeforward(x)
            x = segmentation[0, 1, :, :] - segmentation[0, 0, :, :]
            return self.headforward(x.unsqueeze(0)), x
        else:
            return self.backbone(x)

    def computeiou(self, s, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        y = etendre(y.float(), 10)
        s = s[:, 1, :, :] - s[:, 0, :, :]
        cm00 = torch.sum((s <= 0).float() * (y == 0).float())
        cm11 = torch.sum((s > 0).float() * (y == 1).float())
        cm01 = torch.sum((s <= 0).float() * (y == 1).float())
        cm10 = torch.sum((s > 0).float() * (y == 0).float())

        accu = (cm00 + cm11) / (cm00 + cm11 + cm01 + cm10)
        iou = cm00 / (cm00 + cm01 + cm10) + cm11 / (cm11 + cm01 + cm10)
        return iou / 2, accu

    def computepairing(self, z, y):
        if torch.sum(z) == 0 or torch.sum(y) == 0:
            return [], None, None
        else:
            z, y = torch.nonzero(z).float(), torch.nonzero(y).float()

            Z2 = torch.stack([torch.sum(z * z, dim=1)] * y.shape[0], dim=1)
            Y2 = torch.stack([torch.sum(y * y, dim=1)] * z.shape[0], dim=0)
            D = Z2 + Y2 - 2 * torch.matmul(z, y.t())

            _, Dz = torch.min(D, dim=1)
            _, Dy = torch.min(D, dim=0)
            pair = [(i, Dz[i]) for i in range(z.shape[0]) if Dy[Dz[i]] == i]
            return pair, z.long(), y.long()

    def computegscore(self, z, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        z, y = (z > 0).float(), y.float()
        z10, y10 = etendre(z, 10), etendre(y, 10)
        hardfa = torch.sum((z == 1).float() * (y10 == 0).float())
        hardmiss = torch.sum((y == 1).float() * (z10 == 0).float())

        z, y = z * (y10 == 1).float(), y * (z10 == 1).float()

        pair, _, _ = self.computepairing(z, y)

        good = len(pair)
        fa = torch.sum(z) - len(pair)
        miss = torch.sum(y) - len(pair)

        return good, fa + hardfa, miss + hardmiss

    def lossSegmentation(self, s, y):
        y10 = etendre(y.float(), 10)
        D = distancetransform(y10)

        nb0, nb1 = torch.sum((y10 == 0).float()), torch.sum((y10 == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")

        CE = criterion(s, y10.long())
        CE = torch.mean(CE * D)
        return CE

    def lossDetection(self, s, y):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        Y = torch.zeros(y.shape).cuda()
        with torch.no_grad():
            z = (s[:, 1, :, :] - s[:, 0, :, :]).clone()
            zNMS = self.headforward(z)
            y10 = etendre(y, 10)
            z10 = etendre((zNMS > 0).float(), 10)

            candidateZ = (zNMS > 0).float() * (y10 == 1).float()
            candidateY = y * (z10 > 0).float()

            ## good
            pair, ouZ, ouY = self.computepairing(candidateZ, candidateY)
            for (i, j) in pair:
                Y[ouZ[i][0]][ouZ[i][1]][ouZ[i][2]] = 1

            ## miss: multiple vt for 1 detection
            if ouY is not None:
                J = set([j for _, j in pair])
                J = [j for j in range(ouY.shape[0]) if j not in J]
                for j in J:
                    Y[ouY[j][0]][ouY[j][1]][ouY[j][2]] = 0.1

            ## hard miss detection
            hardmiss = y * (z10 == 0).float()
            Y += 1.1 * hardmiss

            ## hard false alarm
            hardfa = (zNMS > 0).float() * (y10 == 0).float()
            Y -= 1.1 * hardfa

            ## double detection
            if ouZ is not None:
                I = set([i for i, _ in pair])
                I = [i for i in range(ouZ.shape[0]) if i not in I]
                for i in I:
                    Y[ouZ[i][0]][ouZ[i][1]][ouZ[i][2]] = -0.1

        # entropy on hard pixel only
        CE = criterion(s, (Y > 0).long())
        loss = torch.sum(CE * Y.abs()) / (torch.sum(Y.abs()) + 1)

        return faloss + recallloss
