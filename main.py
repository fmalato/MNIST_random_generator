import utils
import numpy as np

from PIL import Image
from MNIST_random_generator.generator import ImageGenerator

gen = ImageGenerator(200, 200)
saliency = True

sizes = [(112, 112)]
for el in sizes:
    gen.setWidth(el[0])
    gen.setHeight(el[1])
    img, _, saliency_map = gen.generateBlankImage(numNumbers=3, saliency=saliency)
    img = Image.fromarray(img)
    img.save("generated/no_bg_" + str(el[0]) + "x" + str(el[1]) + ".jpg")
    if saliency:
        saliency_map = Image.fromarray(np.uint8(saliency_map*255), mode="L")
        saliency_map.save("generated/no_bg_" + str(el[0]) + "x" + str(el[1]) + "_saliency.jpg")

"""bgImages = ["sand", "crowded1", "crowded2", "landscape"]

for pic in bgImages:
    img = gen.generateBackgroudImage(18, "test_background_images/" + pic + ".jpg")
    img.save("generated/bg_" + pic + ".jpg")"""

# utils.convertMNIST("MNIST/", "MNIST_inv/")

