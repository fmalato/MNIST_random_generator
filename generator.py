import cv2
import random as rand
import os
import PIL.ImageOps

from PIL import Image


class ImageGenerator:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def setWidth(self, width):
        self.width = width

    def setHeiht(self, height):
        self.height = height

    def generateBlankImage(self, numNumbers):


        positions = []
        img = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))

        for x in range(numNumbers):
            numClass = rand.randint(0, 9)
            num = rand.randint(0, len(os.listdir("MNIST/" + str(numClass) + "/")))
            numImg = Image.open("MNIST/" + str(numClass) + "/img" + str(numClass) + "_" + str(num) + ".jpg")
            numImg.convert("RGBA")
            numImg = PIL.ImageOps.invert(numImg)
            pixeldata = list(img.getdata())

            for i, pixel in enumerate(pixeldata):
                if pixel[:3] == (255, 255, 255):
                    pixeldata[i] = (255, 255, 255, 0)

            posX = rand.randint(0, self.width - 28)
            posY = rand.randint(0, self.height - 28)

            img.paste(numImg, (posX, posY))
            positions.append((posX, posY))

        img.save("lol.jpg")
        return img
