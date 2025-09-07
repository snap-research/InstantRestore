import random

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import math

class CustomGaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_x, sigma_y):
        super(CustomGaussianBlur, self).__init__()
        self.sigma_x = sigma_x
        self.simga_y = sigma_y
        self.kernel_size = kernel_size
    def kernel_creator(self, type):
        if type=='iso':
            rotation = 0

            sigma_matrix = np.array([[self.sigma_x**2, 0], [0, self.sigma_x**2]])
        else:
            rotation = np.random.uniform(-math.pi, math.pi)
            d_matrix = np.array([[self.sigma_x**2, 0], [0, self.simga_y**2]])
            u_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
            sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        ax = np.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        grid = np.hstack((xx.reshape((self.kernel_size * self.kernel_size, 1)), yy.reshape(self.kernel_size * self.kernel_size,
                                                                           1))).reshape(self.kernel_size, self.kernel_size, 2)

        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        
        kernel = kernel / np.sum(kernel)
        return kernel


    def forward(self, image):
        np_image = image.numpy()
        np_image = np_image.astype('float32')
        np_image = np.transpose(np_image, (1, 2, 0))

        if random.random() < 0.0:
            kernel = self.kernel_creator('iso')
        else:
            kernel = self.kernel_creator('aniso')
        np_image = cv2.filter2D(np_image, -1, kernel)

        np_image = np.transpose(np_image, (2, 0, 1))
        image = torch.tensor(np_image)
        return image

class GaussianNoise(torch.nn.Module):
    def __init__(self, p):
        super(GaussianNoise, self).__init__()
        self.p = p
        self.random_seed = np.random.randint(0,2**32-1)
    def forward(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
            np.random.seed(self.random_seed)
            noise = np.random.randn(*img.shape) * self.p
            out = img + noise
            out = np.clip(out, 0, 255).astype(np.uint8)
            out = Image.fromarray(out)
            return out
        else:
            np.random.seed(self.random_seed)
            noise = torch.tensor(np.random.randn(*img.shape) * self.p/255).float().to(img.device)
            out = img + noise
            out = torch.clip(out, 0, 1)
            # print(out.max())
            # print(out.min())
            # print(out.shape)
            return out
        

class JPEGCompress(torch.nn.Module):
    def __init__(self, p):
        super(JPEGCompress, self).__init__()
        self.p = p

    def __call__(self, image):
        if isinstance(image, Image.Image):
            np_image = np.array(image)
            np_image = np_image.astype('float32')
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.p)]
            _, img_encoded = cv2.imencode('.jpg', np_image, encode_param)
            np_image = cv2.imdecode(img_encoded, 1)
            image = Image.fromarray(np_image)
            return image
        else:
            np_image = image.numpy()
            np_image = np_image.astype('float32')
            np_image = np.transpose(np_image, (1, 2, 0))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.p)]
            _, img_encoded = cv2.imencode('.jpg', np_image*255, encode_param)
            np_image = np.float32(cv2.imdecode(img_encoded, 1)) / 255
            np_image = np.transpose(np_image, (2, 0, 1))
            image = torch.tensor(np_image)
            return image

class RandomBlur:

    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, image):
        if np.random.uniform() < self.p:
            kernel_radius = np.random.randint(1, 6)
            filter_type = random.choice(['GAUSSIAN', 'BOX'])
            kernel = ImageFilter.BoxBlur(radius=0)
            if filter_type == 'GAUSSIAN':
                kernel = ImageFilter.GaussianBlur(radius=kernel_radius)
            elif filter_type == 'BOX':
                kernel = ImageFilter.BoxBlur(radius=kernel_radius)
            image = image.filter(kernel)
        return image
