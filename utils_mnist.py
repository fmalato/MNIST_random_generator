import PIL
import os, re
import PIL.ImageOps

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def goodPos(positions, current, baseRangeWidth):

    for el in positions:
        if current[0] in range(el[0], int(el[0] + baseRangeWidth * el[2])) or \
                current[1] in range(el[1], int(el[1] + baseRangeWidth * el[2])):
            return False

    return True


def convertMNIST(src_dir, dst_dir):

    for i in range(0, 10):
        for j in range(len(os.listdir(src_dir + "/" + str(i)))):
            numImg = Image.open(src_dir + str(i) + "/img" + str(i) + "_" + str(j) + ".jpg")
            numImg.convert("RGBA")
            numImgInv = PIL.ImageOps.invert(numImg)
            numImgInv.save(dst_dir + "/" + str(i) + "/img" + str(i) + "_" + str(j) + ".jpg")


def plotAccuracy(filepath, loss=True):

    if loss:
        losses = []
        with open(filepath + 'loss.txt', 'r+') as f:
            for line in f.readlines():
                line = re.sub('\n', '', line)
                losses.append(float(line))
            f.close()

    accuracies = []
    with open(filepath + 'accuracy.txt', 'r+') as f:
        for line in f.readlines():
            line = re.sub('\n', '', line)
            accuracies.append(float(line))
        f.close()

    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over time')
    plt.show()

    if loss:
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss over time')
        plt.show()


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

