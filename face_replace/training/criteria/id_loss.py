from typing import Tuple

import numpy as np
import torch
from facenet_pytorch import MTCNN
from torch import nn
from torchvision import transforms

from face_replace.training.criteria.arcface import Backbone
from face_replace.training.criteria.utils import extract_faces_and_landmarks


class IDLoss(nn.Module):
    """
    Use pretrained facenet model to extract features from the face of the predicted image and target image.
    Facenet expects 112x112 images, so we crop the face using MTCNN and resize it to 112x112.
    Then we use the cosine similarity between the features to calculate the loss. (The cosine similarity is 1 - cosine distance).
    Also notice that the outputs of facenet are normalized so the dot product is the same as cosine distance.
    """
    def __init__(self, pretrained_arcface_path: str, device, dtype, using_curricular=False, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.dtype = dtype
        self.mtcnn = MTCNN(device=self.device)
        self.mtcnn.forward = self.mtcnn.detect
        self.mtcnn.eval()
        self.facenet_input_size = 112  # Has to be 112, can't find weights for 224 size.

        if using_curricular:
            self.facenet = Backbone(112, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)
        else:
            self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')

        self.facenet.load_state_dict(torch.load(pretrained_arcface_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((self.facenet_input_size, self.facenet_input_size))
        self.facenet.requires_grad_(False)
        self.facenet.eval()
        self.facenet.to(device=self.device, dtype=self.dtype)  # not implemented for half precision
        self.face_pool.to(device=self.device, dtype=self.dtype)  # not implemented for half precision
        self.visualization_resize = transforms.Resize((self.facenet_input_size, self.facenet_input_size),
                                                      interpolation=transforms.InterpolationMode.BICUBIC)
        self.reference_facial_points = np.array([[38.29459953, 51.69630051],
                                                 [72.53179932, 51.50139999],
                                                 [56.02519989, 71.73660278],
                                                 [41.54930115, 92.3655014],
                                                 [70.72990036, 92.20410156]
                                                 ])  # Original points are 112 * 96 added 8 to the x axis to make it 112 * 112

    def extract_feats(self, x: torch.Tensor):
        """ Extract features from the face of the image using facenet model. """
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self,
                predicted_pixel_values: torch.Tensor,
                target_pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        target_pixel_values = target_pixel_values.to(dtype=self.dtype)
        predicted_pixel_values = predicted_pixel_values.to(dtype=self.dtype)

        predicted_pixel_values_face, predicted_invalid_indices = extract_faces_and_landmarks(predicted_pixel_values,
                                                                                             mtcnn=self.mtcnn)
        with torch.no_grad():
            encoder_pixel_values_face, source_invalid_indices = extract_faces_and_landmarks(target_pixel_values,
                                                                                            mtcnn=self.mtcnn)

        # from PIL import Image
        # Image.fromarray(((target_pixel_values[0] * 0.5 + 0.5) * 255).type(torch.uint8).detach().cpu().numpy().transpose(1, 2, 0)).save('/nfs/private/yuval/test.png')

        valid_indices = []
        for i in range(predicted_pixel_values.shape[0]):
            if i not in predicted_invalid_indices and i not in source_invalid_indices:
                valid_indices.append(i)
        valid_indices = torch.tensor(valid_indices).to(device=predicted_pixel_values.device)

        if valid_indices.shape[0] == 0:
            return torch.tensor(0.0).to(device=predicted_pixel_values.device, dtype=self.dtype), \
                   torch.tensor(0.0).to(device=predicted_pixel_values.device, dtype=self.dtype)

        predicted_pixel_values_feats = self.extract_feats(predicted_pixel_values_face[valid_indices])
        with torch.no_grad():
            pixel_values_feats = self.extract_feats(encoder_pixel_values_face[valid_indices])

        # Calculate norm, similarity and loss is the same as 1 - similarity
        similarity = pixel_values_feats @ predicted_pixel_values_feats.T
        loss = 1 - torch.einsum("bi,bi->b", pixel_values_feats, predicted_pixel_values_feats)

        return loss.mean(), similarity.mean()
