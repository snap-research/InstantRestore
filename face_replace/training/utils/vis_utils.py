from typing import Tuple, List

from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from torch import Tensor
import math

from face_replace.training.utils.types import BatchResults


def tensor2im(var: Tensor, unnorm=False) -> Image.Image:
    var = var.clone()
    if unnorm:
        var *= torch.Tensor([0.5, 0.5, 0.5]).to(var.device).reshape(3, 1, 1)
        var += torch.Tensor([0.5, 0.5, 0.5]).to(var.device).reshape(3, 1, 1)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var[var < 0] = 0
    var[var > 1] = 1
    var *= 255
    return Image.fromarray(var.astype('uint8'))


def tensor2np(var: Tensor, unnorm=False) -> np.ndarray:
    var = var.clone()
    if unnorm:
        var *= torch.Tensor([0.5, 0.5, 0.5]).to(var.device).reshape(3, 1, 1)
        var += torch.Tensor([0.5, 0.5, 0.5]).to(var.device).reshape(3, 1, 1)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var[var < 0] = 0
    var[var > 1] = 1
    var *= 255
    return var.astype('uint8')


def vis_data(results: BatchResults, display_count: int = 4) -> Tuple[Image.Image, List[wandb.Image]]:
    batch_size = results.pred.shape[0] if results.pred is not None else results.pred_mix_id.shape[0]
    display_count = min(batch_size, display_count)
    x_src = results.batch["conditioning_pixel_values"]
    x_tgt = results.batch["output_pixel_values"]
    x_pred = results.pred if results.pred is not None else results.pred_mix_id
    source_images = [tensor2im(x_src[idx], unnorm=True) for idx in range(display_count)]
    target_images = [tensor2im(x_tgt[idx], unnorm=True) for idx in range(display_count)]
    pred_images = [tensor2im(x_pred[idx], unnorm=True) for idx in range(display_count)]
    # Concatenate images side by side based on the index
    images = [np.concatenate([source_images[idx], target_images[idx], pred_images[idx]], axis=0)
              for idx in range(display_count)]
    images = [Image.fromarray(image) for image in images]
    image = Image.fromarray(np.concatenate(images, axis=1))
    wandb_images = [wandb.Image(image)]
    return image, wandb_images


def vis_attn_probs(results: BatchResults, attn_probs: torch.Tensor, display_count: int = 4) -> Tuple[Image.Image, List[wandb.Image]]: 
    display_count = min(results.pred.shape[0], display_count)
    target_images = [tensor2im(img) for img in results.batch["output_pixel_values"]]
    cond_images = results.conditions.reshape(results.pred.shape[0], -1, 3, 512, 512)
    joined_images = []
    for idx in range(cond_images.shape[0]):
        images = [tensor2im(cond_images[idx, i]) for i in range(cond_images.shape[1])]
        joined_images.append(Image.fromarray(np.concatenate([target_images[idx]] + images, axis=1)))
    
    attention_images = []
    for attn in attn_probs:
        image = get_visualization_image(results, attn, display_count=1)
        attention_images.append(image)
        
    final_attention_images = []
    for image_idx in range(1):  # only do one image to save some time
        img_attentions = [attn[image_idx] for attn in attention_images]
        original_imgs = joined_images[image_idx]
        image_results = []
        for att_idx in range(len(img_attentions)):
            img = img_attentions[att_idx].resize(original_imgs.size)
            # Paste the attention on top of the original image with alpha 0.75
            image_results.append(Image.blend(original_imgs, img, alpha=0.75))
        # Join the image results row-wise
        final_attention_images.append(Image.fromarray(np.concatenate([np.array(img) for img in image_results], axis=0)))
    
    image = final_attention_images[0]  # return only the first image since this is a big image
    wandb_images = [wandb.Image(image)]     
        
    return image, wandb_images


def get_visualization_image(results: BatchResults,
                            attn: torch.Tensor,
                            display_count: int = 1):
    # Average over all the heads
    attn = attn.mean(dim=1).numpy()
    images = []
    for idx in range(1):
        img = attn[idx]
        landmarks = results.batch['landmarks_dict'][idx]['target']  # We will use these for our queries
        source_shape = 512
        target_shape = int(math.sqrt(img.shape[0]))
        # Rescale the landmarks so that they are between [0, target_shape]
        landmarks_scaled = np.round((landmarks * target_shape / source_shape)).astype(int)
        # For each landmark, get the index between [0, target_shape^2]
        landmark_indices = landmarks_scaled[:, 1] * target_shape + landmarks_scaled[:, 0]
        # Get the maps for each landmark
        landmark_maps = img[landmark_indices]
        # Sum them up
        landmark_maps_summed = np.sum(landmark_maps, axis=0)
        # Reshape to be [X, X * 5] since we have 5 images (the target and the 4 conditioning images)
        landmark_maps_summed_reshape = landmark_maps_summed.reshape(target_shape, target_shape * 5)
        landmark_maps_summed_reshape = Image.fromarray(landmark_maps_summed_reshape).resize((256 * 5, 256))
        landmark_maps_summed_reshape = np.array(landmark_maps_summed_reshape)
        # Visualize this using plt
        plt.imshow(landmark_maps_summed_reshape)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = plt
        ax.axis('off')
        fig = plt.gcf()
        fig.canvas.draw()
        plt.tight_layout()
        pil_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        # Crop this to remove the white border
        width, height = pil_img.size
        threshold = 250
        top = 0
        for y in range(height):
            average = sum(pil_img.getpixel((x, y))[0] for x in range(width)) / width
            if average < threshold:
                top = y
                break

        # Check bottom padding
        bottom = height
        for y in range(height - 1, -1, -1):
            average = sum(pil_img.getpixel((x, y))[0] for x in range(width)) / width
            if average < threshold:
                bottom = y
                break

        # Crop the image to remove padding
        cropped_img = pil_img.crop((0, top, width, bottom))
        images.append(cropped_img)
    
    return images
