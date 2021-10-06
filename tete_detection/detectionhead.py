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
        eps = 0.01
        xp = torch.nn.functional.relu(x - eps)
        xm = self.pool(xp)

        localmax = (x > xm).float()
        # localmaxwithgrad = torch.nn.functional.relu(x - xm)
        return xp * localmax  # , xp * localmaxwithgrad

    def largeforward(self, x):
        tile, stride = 128, 32
        h, w = x.shape[1], x.shape[2]
        x = x.unsqueeze(0)

        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d(
            ((h // stride) * stride, (w // stride) * stride)
        )
        x = power2resize(x)

        with torch.no_grad():
            pred = torch.zeros(1, 2, x.shape[2], x.shape[3]).cuda()
            for row in range(0, x.shape[2] - tile + 1, stride):
                for col in range(0, x.shape[3] - tile + 1, stride):
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

    def computeiou(self, x, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        y = etendre(y.float(), 10)
        s = x[:, 1, :, :] - x[:, 0, :, :]
        cm00 = torch.sum((s <= 0).float() * (y == 0).float())
        cm11 = torch.sum((s > 0).float() * (y == 1).float())
        cm01 = torch.sum((s <= 0).float() * (y == 1).float())
        cm10 = torch.sum((s > 0).float() * (y == 0).float())

        accu = (cm00 + cm11) / (cm00 + cm11 + cm01 + cm10)
        iou = cm00 / (cm00 + cm01 + cm10) + cm11 / (cm11 + cm01 + cm10)
        return iou / 2, accu

    def computepairing(self, x, y):
        if torch.sum(x) == 0 or torch.sum(y) == 0:
            return [], None, None
        else:
            x, y = torch.nonzero(x).float(), torch.nonzero(y).float()

            X2 = torch.stack([torch.sum(x * x, dim=1)] * y.shape[0], dim=1)
            Y2 = torch.stack([torch.sum(y * y, dim=1)] * x.shape[0], dim=0)
            XY = X2 + Y2 - 2 * torch.matmul(x, y.t())

            _, Dx = torch.min(XY, dim=1)
            _, Dy = torch.min(XY, dim=0)
            pair = [(i, Dx[i]) for i in range(x.shape[0]) if Dy[Dx[i]] == i]
            return pair, x.long(), y.long()

    def computegscore(self, x, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        x, y = (x > 0).float(), y.float()
        x10, y10 = etendre(x, 10), etendre(y, 10)
        hardfa = torch.sum((x == 1).float() * (y10 == 0).float())
        hardmiss = torch.sum((x10 == 0).float() * (y == 1).float())

        x, y = x * (y10 == 1).float(), y * (x10 == 1).float()

        pair, x, y = self.computepairing(x, y)

        good = len(pair)
        fa = torch.sum(x) - len(pair)
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
        x = s[:, 1, :, :] - s[:, 0, :, :]
        xNMS = self.headforward(x)

        ### improve recall
        Y = torch.zeros(y.shape).cuda()

        ## enforce correct pair
        pair, ouX, ouY = self.computepairing(xNMS > 0, y)
        for (i, j) in pair:
            Y[ouX[i][0]][ouX[i][1]][ouX[i][2]] = 1

        ## recall in area with positif
        if ouY is not None:
            J = set([j for _, j in pair])
            J = [j for j in range(ouY.shape[0]) if j not in J]
            for j in J:
                Y[ouY[j][0]][ouY[j][1]][ouY[j][2]] = 0.5

        ## recall in area without positif
        x10 = etendre((x > 0).float(), 10)
        y1 = torch.nonzero(y * (x10 == 0).float())
        for i in range(y1.shape[0]):
            im, row, col = y1[i][0], y1[i][1], y1[i][2]
            rm = max(row - 10, 0)
            rM = min(row + 10, y.shape[1])
            cm = max(col - 10, 0)
            cM = min(col + 10, y.shape[2])

            best = torch.amax(x[y1[i][0], rm:rM, cm:cM])
            ou = torch.nonzero(x[y1[i][0], rm:rM, cm:cM] == best)

            Y[im][rm + ou[0][0]][cm + ou[0][1]] = 1.5

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        recallloss = criterion(s, torch.ones(y.shape).long().cuda())
        recallloss = torch.sum(recallloss * Y) / (torch.sum(Y) + 1)

        ### improve precision
        Y = torch.zeros(y.shape).cuda()

        ## precision in area without y
        y10 = etendre(y, 10)
        Y += 2 * (xNMS > 0).float() * (y10 == 0).float()

        ## precision in area with y
        if ouX is not None:
            I = set([i for i, _ in pair])
            I = [i for i in range(ouX.shape[0]) if i not in I]
            for i in I:
                Y[ouX[i][0]][ouX[i][1]][ouX[i][2]] = 0.5

        faloss = criterion(s, torch.zeros(y.shape).long().cuda())
        faloss = torch.sum(faloss * Y) / (torch.sum(Y) + 1)

        return faloss + recallloss
