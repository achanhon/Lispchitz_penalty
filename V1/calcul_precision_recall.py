from __future__ import print_function

import os
import sys
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
from skimage.draw import line_aa


def computeprecisionrappel(prediction, vt, distance):
    if prediction.shape[0] == 0 or vt.shape[0] == 0:
        return 0, [], [], []

    i_j_dist = []
    for i in range(len(vt)):
        for j in range(len(prediction)):
            dist = (prediction[j][0] - vt[i][0]) * (prediction[j][0] - vt[i][0]) + (
                prediction[j][1] - vt[i][1]
            ) * (prediction[j][1] - vt[i][1])
            if dist <= distance * distance:
                i_j_dist.append((i, j, dist))

    i_j_dist_sorted = sorted(i_j_dist, key=lambda tup: tup[2])
    goodalarme = set()
    catcheddetection = set()
    match = []
    for i, j, d in i_j_dist_sorted:
        if i in catcheddetection or j in goodalarme:
            continue
        goodalarme.add(j)
        catcheddetection.add(i)
        match.append((j, i))

    return len(match), list(goodalarme), list(catcheddetection), match


if __name__ == "__main__":
    prediction = np.random.randint(0, 100, size=(50, 2))
    vt = np.random.randint(0, 100, size=(64, 2))

    npvisu = 255 * np.ones((100, 100, 3), dtype=int)
    for i in range(prediction.shape[0]):
        npvisu[prediction[i][0]][prediction[i][1]][0] = 0
    for i in range(vt.shape[0]):
        npvisu[vt[i][0]][vt[i][1]][2] = 0

    res = computeprecisionrappel(prediction, vt, 7)
    print(res)
    _, _, _, match = res

    for i, j in match:
        rr, cc, val = line_aa(prediction[i][0], prediction[i][1], vt[j][0], vt[j][1])
        for k in range(rr.shape[0]):
            npvisu[rr[k]][cc[k]][1] = 0

    fig, ax = plt.subplots(figsize=(200, 200))

    imgplot = ax.imshow(np.uint8(npvisu))
    # plt.ion()
    plt.show()
    # plt.pause(0)
