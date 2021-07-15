import utils
import numpy as np

from PIL import Image
from generator import ImageGenerator

gen = ImageGenerator(200, 200)

sizes = [(112, 112)]
sal = True
for el in sizes:
    gen.setWidth(el[0])
    gen.setHeight(el[1])
    img, _, saliency = gen.generateBlankImage(numNumbers=1, saliency=sal)
    img = Image.fromarray(img)
    saliency = Image.fromarray(np.uint8(saliency * 255), mode="L")
    img.save("generated/no_bg_" + str(el[0]) + "x" + str(el[1]) + ".jpg")
    if sal:
        saliency.save("generated/no_bg_" + str(el[0]) + "x" + str(el[1]) + "_saliency.jpg")

bgImages = ["sand", "crowded1", "crowded2", "landscape"]

for pic in bgImages:
    img = gen.generateBackgroudImage(18, "test_background_images/" + pic + ".jpg")
    img.save("generated/bg_" + pic + ".jpg")

# utils.convertMNIST("MNIST/", "MNIST_inv/")

