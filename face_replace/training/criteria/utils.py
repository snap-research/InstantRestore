import numpy as np
import torch
from einops import rearrange
from kornia.geometry.transform.crop2d import warp_affine

from face_replace.training.criteria.matlab_cp2tform import get_similarity_transform_for_cv2

REFERNCE_FACIAL_POINTS_RELATIVE = np.array([[38.29459953, 51.69630051],
                                            [72.53179932, 51.50139999],
                                            [56.02519989, 71.73660278],
                                            [41.54930115, 92.3655014],
                                            [70.72990036, 92.20410156]
                                            ]) / 112  # Original points are 112 * 96 added 8 to the x axis to make it 112 * 112


def verify_load(missing_keys, unexpected_keys):
    if unexpected_keys:
        RuntimeError(f"Found unexpected keys in state dict while loading the encoder:\n{unexpected_keys}")

    filtered_missing = [key for key in missing_keys if not "extract_kv1" in key]
    if filtered_missing:
        RuntimeError(f"Missing keys in state dict while loading the encoder:\n{filtered_missing}")


def detect_face(images: torch.Tensor, mtcnn: torch.nn.Module) -> torch.Tensor:
    """ Detect faces in the images using MTCNN. If no face is detected, use the whole image. """
    with torch.no_grad():
        images = rearrange(images, "b c h w -> b h w c")
        if images.dtype != torch.uint8:
            images = ((images * 0.5 + 0.5) * 255).type(torch.uint8)  # Unnormalize
        _, _, landmarks = mtcnn(images, landmarks=True)
        return landmarks


def extract_faces_and_landmarks(images: torch.Tensor,
                                output_size: int = 112,
                                mtcnn: torch.nn.Module = None,
                                reference_points=REFERNCE_FACIAL_POINTS_RELATIVE):
    """
    detect faces in the images and crop them (in a differentiable way) to 112x112 using MTCNN.
    """
    landmarks_batched = detect_face(images, mtcnn=mtcnn)
    affine_transformations = []
    invalid_indices = []
    for i, landmarks in enumerate(landmarks_batched):
        if landmarks is None:
            invalid_indices.append(i)
            affine_transformations.append(np.eye(2, 3).astype(np.float32))
        else:
            affine_transformations.append(
                get_similarity_transform_for_cv2(
                    landmarks[0].astype(np.float32), reference_points.astype(np.float32) * output_size
                )
            )
    affine_transformations = torch.from_numpy(
        np.stack(affine_transformations).astype(np.float32)
    ).to(device=images.device, dtype=torch.float32)

    invalid_indices = torch.tensor(invalid_indices).to(device=images.device)

    fp_images = images.to(torch.float32)
    return warp_affine(fp_images, affine_transformations,
                       dsize=(output_size, output_size)).to(dtype=images.dtype), invalid_indices
