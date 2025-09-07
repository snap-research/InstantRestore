import math

import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeLargeAxis:

    def __init__(self, max_scale: float, interpolation=Image.BICUBIC):
        self.max_scale = max_scale
        self.interpolation = interpolation

    def __call__(self, image):
        min_dim = np.argmax(image.size)
        scale_factor = float(self.max_scale) / image.size[min_dim]
        w = int(math.floor(image.size[0] * scale_factor))
        h = int(math.floor(image.size[1] * scale_factor))
        image = image.resize((w, h), self.interpolation)
        return image
