from generator import ImageGenerator
from PIL import Image
import matplotlib.pyplot as plt
import os


gen = ImageGenerator(200, 200)
img = gen.generateBlankImage(7)
