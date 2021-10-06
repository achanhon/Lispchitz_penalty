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

    def headforward(self, segmentation):
        eps = 0.01
        x = segmentation[:, 1, :, :] - segmentation[:, 0, :, :] - eps
        xp = torch.nn.functional.relu(x)
        xm = self.pool(xp)

        localmax = (x > xm).float()
        return x * localmax

    def largeforward(self, x):
        tilesize, stride = 128, 32
        x = x.unsqueeze(0)
        with torch.no_grad():
            pred = torch.zeros(1, 2, x.shape[2], x.shape[3]).to(device)
            for row in range(0, image.shape[2] - tilesize + 1, stride):
                for col in range(0, image.shape[3] - tilesize + 1, stride):
                    tmp = self.backbone(
                        x[0, :, row : row + tilesize, col : col + tilesize]
                    )
                    pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]

        return pred

    def forward(self, x):
        if len(x.shape) == 3:
            segmentation = self.largeforward(x)
        else:
            segmentation = self.backbone(x)

        xnms = self.headforward(segmentation)

        return xnms, segmentation

    def computeiou(self, x, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        y = etendre(y.float(), 10)
        s = x[:, 1, :, :] - x[:, 0, :, :]
        cm00 = torch.sum((s <= 0).float() * (y == 0).float())
        cm11 = torch.sum((s > 0).float() * (y == 1).float())
        cm01 = torch.sum((s <= 0).float() * (y == 1).float())
        cm10 = torch.sum((s > 0).float() * (y == 0).float())

        accu = (cm00 + cm11) / (cm00 + cm11 + cm01 + cm10 + 1)
        iou = cm00 / (cm00 + cm01 + cm10 + 1) + cm11 / (cm11 + cm01 + cm10 + 1)
        return iou / 2, accu

    def computegscore(self, x, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        x, y = (x > 0).float(), y.float()
        x10, y10 = etendre(x, 10), etendre(y, 10)
        hardfa = torch.sum((x > 0).float() * (y10 == 0).float())
        hardmiss = torch.sum((x10 <= 0).float() * (y == 1).float())

        x, y = x * (y10 == 1).float(), y * (x10 > 0).float()
        if torch.sum(x) == 0 or torch.sum(y) == 0:
            good, fa, miss = 0, 0, 0
        else:
            D = torch.zeros((x.shape[0], y.shape[0]))
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    D[i][j] = (x[i] - y[j]).norm()

            _, Dx = torch.min(D, dim=1)
            _, Dy = torch.min(D, dim=0)
            pair = [i for i in range(x.shape[0]) if Dy[Dx[i]] == i]

            good = len(pair)
            fa = len(x) - len(pair)
            miss = len(y) - len(pair)

        if good == 0:
            precision = 0
            recall = 0
        else:
            precision = good / (good + fa + hardfa)
            recall = good / (good + miss + hardmiss)

        gscrore = precision * recall
        tmp = (good, fa, miss, hardfa, hardmiss)
        return gscrore, precision, recall, tmp

        def lossSegmentation(self, x, y, auxcriterions=[]):
            y10 = etendre(y.float(), 10)
            D = distancetransform(y10)

            nb0, nb1 = torch.sum((y10 == 0).float()), torch.sum((y10 == 1).float())
            weights = torch.Tensor([1, nb0 / (nb1 + 1) * 2]).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
            CE = criterion(x, y10.long())
            CE = torch.mean(CE * D)

            auxlosses = []
            for criterion in auxcriterions:
                auxlosses.append(criterion(x, y10))

            if auxlosses == []:
                return CE
            else:
                return CE, auxlosses

        def lossDetection(self, xNMS, segmentation, y):
            # si y=1 et qu'il y a pas de x autour -> faut favoriser le max autour - pas forcément y
            # si un couple x,y est bon, faut mettre le point blanc sur le x et non sur le y

            # miss =
            fa = (x > 0).float() * (y == 1).float()
            # D = D*

            nb0, nb1 = torch.sum((y10 == 0).float()), torch.sum((y10 == 1).float())
            weights = torch.Tensor([1, nb0 / (nb1 + 1) * 2]).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
            CE = criterion(x, y10.long())
            CE = torch.mean(CE * D)

            auxlosses = []
            for criterion in auxcriterions:
                auxlosses.append(criterion(x, y10))

            if auxlosses == []:
                return CE
            else:
                return CE, auxlosses
