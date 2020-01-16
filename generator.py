import random as rand
import os

from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader

import PIL.ImageOps
import utils
import pandas as pd
import torch
import numpy as np

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

        return img, positions

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


class datasetGenerator(ImageGenerator):

    def __init__(self, height, width, path):
        super().__init__(height, width)
        self.path = path

    def generateDataset(self, numElements, numNumbers):

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.path + "imgs/"):
            os.mkdir(self.path + "imgs/")

        with open("{x}".format(x=self.path) + "annotations.csv", "a+") as f:
            firstLine = "name,"
            pos = ['pos{j}x,pos{j}y,scale{j},'.format(j=x) for x in range(1, numNumbers + 1)]
            for s in pos:
                firstLine += s
            firstLine += "\n"
            f.write(firstLine)

        for i in range(numElements):
            image, positions = self.generateBlankImage(numNumbers)
            image.save(self.path + "imgs/" + str(i) + ".jpg")
            with open("{x}".format(x=self.path) + "annotations.csv", "a+") as f:
                f.write("{x}.jpg,".format(x=i))
                for j in range(numNumbers):
                    f.write(str(positions[j][0]) + "," + str(positions[j][1]) + "," + str(positions[j][2]) + ",")
                f.write("\n")
            f.close()
        print("Annotations have been created correctly.")


class MNISTDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:10]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 3)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
