import utils

from generator import ImageGenerator

gen = ImageGenerator(200, 200)

sizes = [(500, 400), (600, 600), (800, 600), (1000, 1000), (1200, 1200)]
for el in sizes:
    gen.setWidth(el[0])
    gen.setHeight(el[1])
    img = gen.generateBlankImage(17)
    img.save("generated/no_bg_" + str(el[0]) + "x" + str(el[1]) + ".jpg")

bgImages = ["sand", "crowded1", "crowded2", "landscape"]

for pic in bgImages:
    img = gen.generateBackgroudImage(18, "test_background_images/" + pic + ".jpg")
    img.save("generated/bg_" + pic + ".jpg")

# utils.convertMNIST("MNIST/", "MNIST_inv/")

