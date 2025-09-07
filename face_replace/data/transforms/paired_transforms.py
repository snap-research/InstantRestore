from typing import List, Tuple, Union
import numpy as np
import torch
from PIL import Image
import random
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
import cv2


class PairedTransform:
    def __init__(self, transforms: List, probabilities: List[float]):
        self.transforms = []
        for transform, probability in zip(transforms, probabilities):
            self.transforms.append((transform, probability))

    def __call__(self, img1: Image.Image, img2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        result_img1, result_img2 = img1, img2
        for transform, probability in self.transforms:
            if random.random() < probability:
                if type(transform) in [PairedColorJitter, PairedRandomBlur, PairedCompress]:
                    result_img1, result_img2 = transform(result_img1, result_img2)
                else:
                    result_img1 = transform(result_img1)
                    result_img2 = transform(result_img2)
        return result_img1, result_img2


class PairedColorJitter(ColorJitter):

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, img1, img2):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img1 = F.adjust_brightness(img1, brightness_factor)
                img2 = F.adjust_brightness(img2, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img1 = F.adjust_contrast(img1, contrast_factor)
                img2 = F.adjust_contrast(img2, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img1 = F.adjust_saturation(img1, saturation_factor)
                img2 = F.adjust_saturation(img2, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img1 = F.adjust_hue(img1, hue_factor)
                img2 = F.adjust_hue(img2, hue_factor)

        return img1, img2


class PairedRandomBlur:

    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img1, img2):
        if np.random.uniform() < self.p:
            kernel_radius = np.random.randint(1, 6)
            filter_type = random.choice(['GAUSSIAN', 'BOX'])
            kernel = ImageFilter.BoxBlur(radius=0)
            if filter_type == 'GAUSSIAN':
                kernel = ImageFilter.GaussianBlur(radius=kernel_radius)
            elif filter_type == 'BOX':
                kernel = ImageFilter.BoxBlur(radius=kernel_radius)
            img1 = img1.filter(kernel)
            img2 = img2.filter(kernel)
        return img1, img2


class PairedCompress(object):
    def __init__(self, p: float = 0.4, qual_mean: int = 30, qual_std: int = 5):
        self.p = p
        self.qual_mean = qual_mean
        self.qual_std = qual_std

    def __call__(self, img1: Image, img2: Image):
        if np.random.uniform() < self.p:
            np_img1 = np.array(img1)
            np_img2 = np.array(img2)
            np_img1 = np_img1.astype('float32')
            np_img2 = np_img2.astype('float32')
            assert np_img1.dtype == np.float32
            assert np_img2.dtype == np.float32
            compression_quality = np.random.normal(loc=self.qual_mean, scale=self.qual_std)
            compression_quality = min(max(compression_quality, 1), 100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(compression_quality)]
            _, img1_encoded = cv2.imencode('.jpg', np_img1, encode_param)
            _, img2_encoded = cv2.imencode('.jpg', np_img2, encode_param)
            np_img1 = cv2.imdecode(img1_encoded, 1)
            np_img2 = cv2.imdecode(img2_encoded, 1)
            img1 = Image.fromarray(np_img1)
            img2 = Image.fromarray(np_img2)
        return img1, img2
