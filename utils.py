import PIL
import os
import PIL.ImageOps

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
