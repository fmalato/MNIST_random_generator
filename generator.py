import random as rand
import os
import PIL.ImageOps

from PIL import Image

import utils

class ImageGenerator:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def setWidth(self, width):
        self.width = width

    def setHeight(self, height):
        self.height = height

    def generateBlankImage(self, numNumbers):

        positions = []
        img = Image.new("RGB", (self.width, self.height), color=(0, 255, 0))

        for x in range(numNumbers):
            numClass = rand.randint(0, 9)
            num = rand.randint(0, len(os.listdir("MNIST/" + str(numClass) + "/")))
            numImg = Image.open("MNIST/" + str(numClass) + "/img" + str(numClass) + "_" + str(num) + ".jpg")
            numImg.convert("RGBA")
            numImgInv = PIL.ImageOps.invert(numImg)
            scale = rand.choice((0.5, 1, 1.5, 2))
            numImg = numImg.resize((int(numImg.size[0] * scale), int(numImg.size[1] * scale)), PIL.Image.ANTIALIAS)
            numImgInv = numImgInv.resize((int(numImgInv.size[0] * scale), int(numImgInv.size[1] * scale)), PIL.Image.ANTIALIAS)

            posX = rand.randint(0, img.size[0] - 28 * scale)
            posY = rand.randint(0, img.size[1] - 28 * scale)

            while not utils.goodPos(positions, (posX, posY), 28):
                posX = rand.randint(0, img.size[0] - 28 * scale)
                posY = rand.randint(0, img.size[1] - 28 * scale)

            img.paste(numImgInv, (posX, posY), numImg)
            positions.append((posX, posY, scale))

        return img

    def generateBackgroudImage(self, numNumbers, bgPath):

        positions = []
        img = Image.open(bgPath)

        for x in range(numNumbers):
            numClass = rand.randint(0, 9)
            num = rand.randint(0, len(os.listdir("MNIST/" + str(numClass) + "/")))
            numImg = Image.open("MNIST/" + str(numClass) + "/img" + str(numClass) + "_" + str(num) + ".jpg")
            numImg.convert("RGBA")
            numImgInv = PIL.ImageOps.invert(numImg)
            scale = rand.choice((0.5, 1, 1.5, 2))
            numImg = numImg.resize((int(numImg.size[0] * scale), int(numImg.size[1] * scale)), PIL.Image.ANTIALIAS)
            numImgInv = numImgInv.resize((int(numImgInv.size[0] * scale), int(numImgInv.size[1] * scale)), PIL.Image.ANTIALIAS)

            posX = rand.randint(0, img.size[0] - 28)
            posY = rand.randint(0, img.size[1] - 28)

            while not utils.goodPos(positions, (posX, posY), 28):
                posX = rand.randint(0, img.size[0] - 28)
                posY = rand.randint(0, img.size[1] - 28)

            img.paste(numImgInv, (posX, posY), numImg)
            positions.append((posX, posY, scale))

        return img


