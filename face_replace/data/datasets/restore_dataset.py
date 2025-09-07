import random
from typing import Callable, List, Dict, Any
import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import cv2

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as F
import torch.nn.functional as functional
from torchvision import transforms

import sys
sys.path.append('.')
sys.path.append('..')

from face_replace.data.datasets.coach_dataset import CoachDataset
from torch.utils.data import DataLoader

from face_replace.data.transforms.paired_transforms import PairedTransform, PairedColorJitter
from face_replace.data.transforms.augmentations import GaussianNoise, JPEGCompress,CustomGaussianBlur
from face_replace.data.transforms.DiffJPEG import DiffJPEG

from torchvision.transforms.v2 import GaussianBlur, Resize

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class RestoreDataset(CoachDataset):

    def __init__(self, dataset_folder,
                 identity_names = None,
                 norm_transform: Callable = None,
                 to_tensor_transform: Callable = None,
                 max_conditioning_images: int = 4,
                 joined_transforms=None,
                 conditioning_transforms=None,
                 condition_on_face_embeds=False,
                 return_degrade_transforms=False,
                 get_gt_attn_probs=False,
                 train_input=True,
                 get_attn_pos_reg=False,
                 get_attn_neg_reg=False,
                 get_facial_comps=False,
                 resolution=512):
        super().__init__()

        if type(dataset_folder) != list: 
            dataset_folder = [dataset_folder]

        self.norm_transform = norm_transform
        self.to_tensor_transform = to_tensor_transform
        self.joined_transforms = joined_transforms
        self.condition_on_face_embeds = condition_on_face_embeds
        self.max_conditioning_images = max_conditioning_images
        self.conditioning_transforms = conditioning_transforms
        self.resolution = resolution
        self.return_degrade_transforms = return_degrade_transforms
        self.get_gt_attn_probs = get_gt_attn_probs
        self.train_input = train_input
        self.get_attn_pos_reg = get_attn_pos_reg
        self.get_attn_neg_reg = get_attn_neg_reg
        self.get_facial_comps = get_facial_comps

        # layer info to help with attn maps
        self.layer_stat_list = [{'num_heads': 20, 'size': 16}, {'num_heads': 20, 'size': 16}, 
                           {'num_heads': 20, 'size': 16}, {'num_heads': 10, 'size': 32}, 
                           {'num_heads': 10, 'size': 32}, {'num_heads': 10, 'size': 32}, 
                           {'num_heads': 5, 'size': 64}, {'num_heads': 5, 'size': 64}, 
                           {'num_heads': 5, 'size': 64}]

        all_identity_names = [] 
        self.output_folders = []
        for folder in dataset_folder: 
            old_dataset_naming_convention = False
            # if 'tfhq_v5' in str(folder):
            #     old_dataset_naming_convention = True
            if identity_names is None:
                # if old_dataset_naming_convention:
                if self.get_gt_attn_probs:
                    folder_identities = [
                        identity.name for identity in folder.glob("*") 
                        if identity.is_dir() and len(list(identity.glob("cropped_images/*"))) > 1 and len(list(identity.glob("new_landmarks/*"))) > 1
                    ]
                else:
                    folder_identities = [
                        identity.name for identity in folder.glob("*") 
                        if identity.is_dir() and len(list(identity.glob("cropped_images/*"))) > 1
                    ]
                self.output_folders.extend([
                    folder / identity / "cropped_images" for identity in folder_identities
                ])
                all_identity_names.extend(folder_identities)
                # else:
                #     folder_identities = [
                #         identity.name for identity in folder.glob("*") 
                #         if identity.is_dir() and len(list(identity.glob("*.png"))) > 1
                #     ]
                #     self.output_folders.extend([
                #         folder / identity for identity in folder_identities
                #     ])
                #     all_identity_names.extend(folder_identities)
            else:
                self.output_folders.extend([
                    folder / identity for identity in identity_names
                ])
                all_identity_names.extend(identity_names)
        
        print(f"Total number of identities: {len(all_identity_names)}")

        self.paths = []
        for folder in self.output_folders:
            self.paths += [img for img in folder.glob("*") if img.suffix in [".jpg", ".png", ".jpeg"]]
        self.paths = self.paths[::-1]

    def __len__(self):
        return len(self.paths)

    def shuffle(self):
        random.shuffle(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]

        input_img = Image.open(image_path).convert("RGB")
        output_img = Image.open(image_path).convert("RGB")
            
        input_img, output_img = self.joined_transforms(input_img, output_img)

        if random.random() < -1:  # Need to apply the flip to everything together so we know how to paste the mask back
            flip_transform = transforms.RandomHorizontalFlip(p=1.0)
            input_img = flip_transform(input_img)
            output_img = flip_transform(output_img)        

        if random.random() < -1:
            # Apply random rotation to images and the landmarks
            angle = np.random.randint(-45, 45)
            input_img = F.rotate(input_img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            output_img = F.rotate(output_img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        
        blur_sigma_x = np.random.uniform(0.1, 12)
        blur_sigma_y = np.random.uniform(0.1, 12)
        downsample_factor = np.random.randint(1, 13)
        noise = np.random.uniform(10, 20)
        quality = np.random.randint(10, 20)

        # degrade_transforms = transforms.Compose([
        #     GaussianBlur(41, blur_sigma),
        #     GaussianNoise(noise),
        #     Resize(self.resolution//downsample_factor, transforms.InterpolationMode.BILINEAR),
        #     Resize(self.resolution, transforms.InterpolationMode.BILINEAR),
        #     DiffJPEG.DiffJPEG(self.resolution, self.resolution, differentiable=True, quality=quality)
        # ])

        degrade_transforms = transforms.Compose([
            CustomGaussianBlur(41, blur_sigma_x, blur_sigma_y),
            Resize(512//downsample_factor, transforms.InterpolationMode.BILINEAR),
            GaussianNoise(noise),
            JPEGCompress(quality),
            Resize(512, transforms.InterpolationMode.BILINEAR),
        ])

        input_t = self.to_tensor_transform(input_img)
        input_t = degrade_transforms(input_t)
        input_t = self.norm_transform(input_t)
        input_t = input_t.detach()
        output_t = self.to_tensor_transform(output_img)
        output_t = self.norm_transform(output_t)

        caption = "A high-quality photo of a person; professional, 8k"

        identity_dir = image_path.parent.parent
        if self.condition_on_face_embeds:
            conditioning_images, _, face_embed = self._get_conditioning_images_and_face_embeds(identity_dir, image_path)
        else:
            conditioning_images, conditioning_image_paths = self._get_conditioning_images(identity_dir, image_path, 
                get_gt_attn_probs = self.get_gt_attn_probs)
            face_embed = None
        chosen_replace_pos = None
        if self.get_attn_pos_reg and random.random() < 0.25:
            chosen_replace_pos = random.randint(0,len(conditioning_images)-1)
            conditioning_images[chosen_replace_pos] = Image.open(image_path).convert("RGB")
        chosen_replace_neg = None
        if self.get_attn_neg_reg and random.random() < 0.25:
            chosen_neg_identity_idx = random.randint(0, len(self.output_folders)-1)
            chosen_neg_identity = self.output_folders[chosen_neg_identity_idx].parent
            if str(chosen_neg_identity) == str(image_path.parent.parent):
                chosen_neg_identity_idx = len(self.output_folders) - 1 - chosen_neg_identity_idx
            neg_paths = [img for img in self.output_folders[chosen_neg_identity_idx].glob("*") if img.suffix in [".jpg", ".png", ".jpeg"]]
            chosen_neg_path_idx = random.randint(0, len(neg_paths)-1)
            neg_path = neg_paths[chosen_neg_path_idx]
            chosen_replace_neg = random.randint(0, len(conditioning_images)-1)
            if chosen_replace_neg == chosen_replace_pos:
                chosen_replace_neg = len(conditioning_images) - 1 - chosen_replace_pos
            conditioning_images[chosen_replace_neg] = Image.open(neg_path).convert("RGB")

        conditioning_images = [self.conditioning_transforms(img) for img in conditioning_images]
        if self.get_gt_attn_probs:
            chosen_layer = random.randint(0, 8)
            cond_max = self.max_conditioning_images if self.train_input else (self.max_conditioning_images - 1)
            chosen_cond = random.randint(0,cond_max)
            new_attn_probs, masks, success = self._get_gt_attn_probs_per_layer_per_cond(identity_dir, image_path, conditioning_image_paths, chosen_layer, chosen_cond)
            if success:
                gt_attn_probs = (new_attn_probs, masks, chosen_layer, chosen_cond)
            else:
                gt_attn_probs = None
        else:
            gt_attn_probs = None
        
        if self.get_facial_comps:
            facial_comps = self._get_facial_comps(identity_dir, image_path)
        else:
            facial_comps = None

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": input_t,
            "caption": caption,
            "face_embed": face_embed,
            "conditioning_images": conditioning_images,
            "degrade_transforms": degrade_transforms if self.return_degrade_transforms else None,
            "gt_attn_probs": gt_attn_probs if self.get_gt_attn_probs else None,
            "gt_attn_pos_reg_idx": chosen_replace_pos,
            "gt_attn_neg_reg_idx": chosen_replace_neg,
            "gt_facial_comps": facial_comps
        }
    
    def _get_facial_comps(self, identity_dir: Path, original_image_path: Path):
        gt_filename = original_image_path.stem + '.npy'
        if os.path.exists(f"{identity_dir}/new_landmarks/{gt_filename}"):
            gt_landmark = np.load(f"{identity_dir}/new_landmarks/{gt_filename}")
        else:
            return None
        leye_mask = np.zeros((self.resolution, self.resolution)).astype(np.uint8)
        reye_mask = np.zeros((self.resolution, self.resolution)).astype(np.uint8)
        mouth_mask = np.zeros((self.resolution, self.resolution)).astype(np.uint8)

        lm_leye_idx = 626
        lm_leye_x = int(gt_landmark[lm_leye_idx][0])
        lm_leye_y = int(gt_landmark[lm_leye_idx][1])
        cv2.rectangle(leye_mask, (lm_leye_x - 50, lm_leye_y-50), (lm_leye_x + 50, lm_leye_y+20), 1, thickness=cv2.FILLED)

        lm_reye_idx = 590
        lm_reye_x = int(gt_landmark[lm_reye_idx][0])
        lm_reye_y = int(gt_landmark[lm_reye_idx][1])
        cv2.rectangle(reye_mask, (lm_reye_x - 50, lm_reye_y-50), (lm_reye_x + 50, lm_reye_y+20), 1, thickness=cv2.FILLED)


        lm_mouth_idx = 0
        lm_mouth_x = int(gt_landmark[lm_mouth_idx][0])
        lm_mouth_y = int(gt_landmark[lm_mouth_idx][1])
        cv2.rectangle(mouth_mask, (lm_mouth_x - 80, lm_mouth_y-30), (lm_mouth_x + 80, lm_mouth_y+60), 1, thickness=cv2.FILLED)

        leye_mask = torch.tensor(leye_mask).bool()
        reye_mask = torch.tensor(reye_mask).bool()
        mouth_mask = torch.tensor(mouth_mask).bool()

        return (leye_mask, reye_mask, mouth_mask)
        

    def _get_gt_attn_probs_per_layer_per_cond(self, identity_dir: Path, original_image_path: Path, conditioning_image_paths: List[Path], chosen_layer: int, chosen_cond: int):
        def gaussian_2d(x, y, x0, y0, sigma):
            return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        gt_filename = original_image_path.stem + '.npy'
        if os.path.exists(f"{identity_dir}/new_landmarks/{gt_filename}"):
            gt_landmark = np.load(f"{identity_dir}/new_landmarks/{gt_filename}")
        else:
            return None, None, False
        
        if self.train_input:
            if chosen_cond == 0:
                cond_landmark = gt_landmark
            else:
                conditioning_image_path = conditioning_image_paths[chosen_cond-1]
                cond_filename = conditioning_image_path.stem + '.npy'
                if os.path.exists(f"{identity_dir}/new_landmarks/{cond_filename}"):
                    cond_landmark = np.load(f"{identity_dir}/new_landmarks/{cond_filename}")
                else:
                    return None, None, False
        else:
            conditioning_image_path = conditioning_image_paths[chosen_cond]
            cond_filename = conditioning_image_path.stem + '.npy'
            if os.path.exists(f"{identity_dir}/new_landmarks/{cond_filename}"):
                cond_landmark = np.load(f"{identity_dir}/new_landmarks/{cond_filename}")
            else:
                return None, None, False
            cond_landmark = np.load(f"{identity_dir}/new_landmarks/{cond_filename}")

        num_landmarks = gt_landmark.shape[0]
        new_attn_probs = []
        masks = []
        layer_stat = self.layer_stat_list[chosen_layer]
        size = layer_stat['size']
        downsample_factor = self.resolution//size
        num_heads = layer_stat['num_heads']

        x_coord = torch.arange(0, size, 1)
        y_coord = torch.arange(0, size, 1)
        x_coord, y_coord = torch.meshgrid(x_coord, y_coord, indexing='ij')

        mask_layers = torch.zeros(size*size).bool()
        attn_prob_layer = torch.zeros((size * size, size * size))
        for i in range(num_landmarks):
            x_up, y_up = gt_landmark[i]
            x = (x_up//downsample_factor).astype(np.int32)
            y = (y_up//downsample_factor).astype(np.int32)
            pos_gt = y*size + x
            cond_x_up, cond_y_up = cond_landmark[i]
            cond_x = (cond_x_up//downsample_factor).astype(np.int32)
            cond_y = (cond_y_up//downsample_factor).astype(np.int32)
            # pos_cond = cond_y*size + cond_x
            if x >= size or y >= size or cond_x >= size or cond_y >= size:
                continue
            # attn_prob_layer[pos_gt, pos_cond] += 1
            sigma = .03125*size  # Adjust the spread as needed
            # Apply the Gaussian function to each point in the grid
            gaussian = gaussian_2d(x_coord, y_coord, cond_x, cond_y, sigma)
            attn_prob_layer[pos_gt, :] += gaussian.flatten()

            mask_layers[pos_gt] = True
        masks.append(mask_layers)
        new_attn_probs.append(attn_prob_layer.unsqueeze(0).repeat((num_heads, 1, 1)))
        return new_attn_probs, masks, True            
    
    def _get_gt_attn_probs(self, identity_dir: Path, original_image_path: Path, conditioning_image_paths: List[Path]):
        gt_filename = original_image_path.stem + '.npy'
        if os.path.exists(f"{identity_dir}/new_landmarks/{gt_filename}"):
            gt_landmark = np.load(f"{identity_dir}/new_landmarks/{gt_filename}")
        else:
            return None, None, False
        
        if self.train_input:
            cond_landmarks = [gt_landmark]
            num_conditioning = 5
        else:
            cond_landmarks = []
            num_conditioning = 4
        for conditioning_image_path in conditioning_image_paths:
            cond_filename = conditioning_image_path.stem + '.npy'
            cond_landmarks.append(np.load(f"{identity_dir}/new_landmarks/{cond_filename}"))

        num_landmarks = gt_landmark.shape[0]
        new_attn_probs = []
        masks = []
        for layer_stat in self.layer_stat_list:
            size = layer_stat['size']
            downsample_factor = self.resolution//size
            num_heads = layer_stat['num_heads']
            attn_prob_conds = []

            mask_layers = torch.zeros((num_conditioning, size*size))
            for cond_idx, cond_landmark in enumerate(cond_landmarks):
                attn_prob_layer = torch.zeros((size * size, size * size))
                for i in range(num_landmarks):
                    x_up, y_up = gt_landmark[i]
                    x = (x_up//downsample_factor).astype(np.int32)
                    y = (y_up//downsample_factor).astype(np.int32)
                    pos_gt = y*size + x
                    cond_x_up, cond_y_up = cond_landmark[i]
                    cond_x = (cond_x_up//downsample_factor).astype(np.int32)
                    cond_y = (cond_y_up//downsample_factor).astype(np.int32)
                    # try:
                    pos_cond = cond_y*size + cond_x
                    if x >= size or y >= size or cond_x >= size or cond_y >= size:
                        continue
                    attn_prob_layer[pos_gt, pos_cond] += 1
                    # except:
                    #     print("----------------------")
                    #     print(f"pos_cond: {pos_cond}")
                    #     print(f"attn_prob_layer shape: {attn_prob_layer.shape}")
                    #     print(f"x: {x}")
                    #     print(f"y: {y}")
                    #     print(f"x_up: {x_up}")
                    #     print(f"y_up: {y_up}")
                    #     print(f"cond_x: {cond_x}")
                    #     print(f"cond_y: {cond_y}")
                    #     print(f"cond_x_up: {cond_x_up}")
                    #     print(f"cond_y_up: {cond_y_up}")
                    #     print(f"downsample_factor: {downsample_factor}")
                    #     print(f"size: {size}")
                    #     raise Exception
                    mask_layers[cond_idx, pos_gt] = 1
                attn_prob_conds.append(attn_prob_layer)
            masks.append(mask_layers)
            new_attn_probs.append(torch.cat(attn_prob_conds, dim=1).unsqueeze(0).repeat((num_heads, 1, 1)))
        return new_attn_probs, masks, True


    def _get_conditioning_images_and_face_embeds(self, identity_dir: Path, original_image_path: Path):
        conditioning_images, conditioning_image_paths = self._get_conditioning_images(identity_dir, original_image_path)
        # Sample a random face embed for now. Later we will change this to be from the conditioning images
        embed_results = [self.canonical_face_processor.app.get(np.array(face_image)) for face_image in conditioning_images]
        face_embed = []
        for result in embed_results:
            if result is not None and len(result) == 1:
                embed = torch.from_numpy(result[0].normed_embedding)
                face_embed.append(embed)
            else:
                face_embed.append(torch.zeros(512))
        face_embed = torch.stack(face_embed)
        return conditioning_images, conditioning_image_paths, face_embed

    def _get_conditioning_images(self, identity_dir: Path, original_image_path: Path, get_gt_attn_probs: bool = False):
        conditioning_image_paths = None
        # if 'tfhq_v5' in str(identity_dir):
            # if get_gt_attn_probs:
            #     conditioning_image_paths = []
            #     for img in identity_dir.glob("cropped_images/*"):
            #         landmark_filename = img.stem + '.npy'
            #         if img != original_image_path and img.suffix in [".jpg", ".png", ".jpeg"] and os.path.exists(f"{img.parent.parent}/new_landmarks/{landmark_filename}"):
            #             conditioning_image_paths.append(img)
            # else:
        conditioning_image_paths = [img for img in identity_dir.glob("cropped_images/*") 
                                if img != original_image_path and img.suffix in [".jpg", ".png", ".jpeg"]]
        # else:
        #     conditioning_image_paths = [img for img in identity_dir.glob("*") 
        #                             if img != original_image_path and img.suffix in [".jpg", ".png", ".jpeg"]]
        conditioning_image_paths = np.random.choice(
            conditioning_image_paths,
            size=min(len(conditioning_image_paths), np.random.randint(1, self.max_conditioning_images + 1)),
            replace=False
        ).tolist()
        conditioning_images = [Image.open(path).convert("RGB") for path in conditioning_image_paths]

        # If there are less than 4 conditioning images, duplicate the images that exist and augment them using horizontal flipping
        n_images_to_add = self.max_conditioning_images - len(conditioning_images)
        added_idxs = []
        all_conditioning_images = conditioning_images.copy()
        all_conditioning_image_paths = conditioning_image_paths.copy()
        for i in range(n_images_to_add):
            idx_to_add = i % len(conditioning_images)
            if added_idxs.count(idx_to_add) % 2 == 0:
                new_image = transforms.RandomHorizontalFlip(p=0.0)(conditioning_images[idx_to_add])
            else:
                new_image = conditioning_images[idx_to_add]
            all_conditioning_images.append(new_image)
            added_idxs.append(idx_to_add)
            all_conditioning_image_paths.append(conditioning_image_paths[idx_to_add])

        return all_conditioning_images, all_conditioning_image_paths

class RestoreDatasetTest(CoachDataset):

    def __init__(self, dataset_folder,
                 norm_transform: Callable = None,
                 to_tensor_transform: Callable = None,
                 max_conditioning_images: int = 4,
                 joined_transforms=None,
                 conditioning_transforms=None,
                 condition_on_face_embeds=False):
        super().__init__()

        if type(dataset_folder) != list: 
            dataset_folder = [dataset_folder]

        self.norm_transform = norm_transform
        self.to_tensor_transform = to_tensor_transform
        self.joined_transforms = joined_transforms
        self.condition_on_face_embeds = condition_on_face_embeds
        self.max_conditioning_images = max_conditioning_images
        self.conditioning_transforms = conditioning_transforms

        all_identity_names = [] 
        self.paths = []
        for folder in dataset_folder: 
            folder_identities = [
                identity.name for identity in folder.glob("*") 
                if identity.is_dir() and len(list(identity.glob("*.png"))) >= 3
            ]
            # print("IDENTITIES: " + folder.glob('*'))
            # for identity in folder.glob("*"):
            #     print(len(list(identity.glob("*.png"))))
            self.paths.extend([
                    folder / identity / "degraded.png" for identity in folder_identities
            ])
            all_identity_names.extend(folder_identities)
        
        print(f"Total number of identities: {len(all_identity_names)}")
        self.paths = self.paths[::-1]

    def __len__(self):
        return len(self.paths)

    def shuffle(self):
        random.shuffle(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        output_path = image_path.parent / 'gt.png'

        input_img = Image.open(image_path).convert("RGB")
        output_img = Image.open(output_path).convert("RGB")
            
        input_img, output_img = self.joined_transforms(input_img, output_img)

        input_t = self.to_tensor_transform(input_img)
        input_t = self.norm_transform(input_t)
        output_t = self.to_tensor_transform(output_img)
        output_t = self.norm_transform(output_t)

        caption = "A high-quality photo of a person; professional, 8k"

        identity_dir = image_path.parent
        if self.condition_on_face_embeds:
            conditioning_images, _, face_embed = self._get_conditioning_images_and_face_embeds(identity_dir, image_path)
        else:
            conditioning_images, _ = self._get_conditioning_images(identity_dir, image_path)
            face_embed = None

        conditioning_images = [self.conditioning_transforms(img) for img in conditioning_images]

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": input_t,
            "caption": caption,
            "face_embed": face_embed,
            "conditioning_images": conditioning_images,
            "degrade_transforms": None,
            "gt_attn_probs": None,
            "gt_attn_pos_reg_idx": None,
            "gt_attn_neg_reg_idx": None,
            "gt_facial_comps": None,
        }

    def _get_conditioning_images_and_face_embeds(self, identity_dir: Path, original_image_path: Path):
        conditioning_images, conditioning_image_paths = self._get_conditioning_images(identity_dir, original_image_path)
        # Sample a random face embed for now. Later we will change this to be from the conditioning images
        embed_results = [self.canonical_face_processor.app.get(np.array(face_image)) for face_image in conditioning_images]
        face_embed = []
        for result in embed_results:
            if result is not None and len(result) == 1:
                embed = torch.from_numpy(result[0].normed_embedding)
                face_embed.append(embed)
            else:
                face_embed.append(torch.zeros(512))
        face_embed = torch.stack(face_embed)
        return conditioning_images, conditioning_image_paths, face_embed

    def _get_conditioning_images(self, identity_dir: Path, original_image_path: Path):
        conditioning_image_paths = [img for img in identity_dir.glob("conditioning/*") 
                                    if img.suffix in [".jpg", ".png", ".jpeg"]]
        conditioning_image_paths = np.random.choice(
            conditioning_image_paths,
            size=min(len(conditioning_image_paths), np.random.randint(1, self.max_conditioning_images + 1)),
            replace=False
        ).tolist()
        conditioning_images = [Image.open(path).convert("RGB") for path in conditioning_image_paths]

        # If there are less than 4 conditioning images, duplicate the images that exist and augment them using horizontal flipping
        n_images_to_add = self.max_conditioning_images - len(conditioning_images)
        added_idxs = []
        all_conditioning_images = conditioning_images.copy()
        all_conditioning_image_paths = conditioning_image_paths.copy()
        for i in range(n_images_to_add):
            idx_to_add = i % len(conditioning_images)
            if added_idxs.count(idx_to_add) % 2 == 0:
                new_image = transforms.RandomHorizontalFlip(p=1.0)(conditioning_images[idx_to_add])
            else:
                new_image = conditioning_images[idx_to_add]
            all_conditioning_images.append(new_image)
            added_idxs.append(idx_to_add)
            all_conditioning_image_paths.append(conditioning_image_paths[idx_to_add])

        return all_conditioning_images, all_conditioning_image_paths


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    joined_batch = {
        "output_pixel_values": torch.stack([b["output_pixel_values"] for b in batch]),
        "conditioning_pixel_values": torch.stack([b["conditioning_pixel_values"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
    
    # For face embeds, we'll pad the extra indices with zeros to make it the same dimensions
    if batch[0]['face_embed'] == None:
        joined_batch["face_embed"] = None
    else:
        joined_batch['face_embed'] = torch.stack([b["face_embed"] for b in batch])

    # For conditioning images, we need to pad the images to the same size
    conditioning_images = [b["conditioning_images"] for b in batch]
    max_len = max([len(images) for images in conditioning_images])
    padded_conditioning_images = torch.zeros(len(conditioning_images), max_len, 3, 512, 512)
    for i, images in enumerate(conditioning_images):
        for j, img in enumerate(images):
            padded_conditioning_images[i, j] = img

    # Also store the valid lengths of the face embeds and conditioning images
    valid_indices = [len(images) for images in conditioning_images]
    valid_indices = torch.tensor(valid_indices)

    joined_batch["conditioning_images"] = padded_conditioning_images
    joined_batch["valid_indices"] = valid_indices

    if batch[0]['degrade_transforms'] == None:
        joined_batch["degrade_transforms"] = None
    else:
        joined_batch["degrade_transforms"] = [b["degrade_transforms"] for b in batch]
    
    if batch[0]['gt_attn_probs'] == None:
        joined_batch['gt_attn_probs'] = None
    else:
        attn_probs = []
        masks = []
        chosen_layer = [b["gt_attn_probs"][2] for b in batch]
        chosen_cond = [b["gt_attn_probs"][3] for b in batch]
        num_layers = len(batch[0]['gt_attn_probs'][0])
        for layer_idx in range(num_layers):
            attn_probs.append(torch.stack([b["gt_attn_probs"][0][layer_idx] for b in batch]))
            masks.append(torch.stack([b["gt_attn_probs"][1][layer_idx] for b in batch]))
        joined_batch['gt_attn_probs'] = (attn_probs, masks, chosen_layer, chosen_cond)
    joined_batch['gt_attn_pos_reg_idx'] = [b["gt_attn_pos_reg_idx"] for b in batch]
    joined_batch['gt_attn_neg_reg_idx'] = [b["gt_attn_neg_reg_idx"] for b in batch]

    if batch[0]['gt_facial_comps'] == None:
        joined_batch['gt_facial_comps'] = None
    else:
        joined_leye_mask = torch.stack([b["gt_facial_comps"][0] for b in batch])
        joined_reye_mask = torch.stack([b["gt_facial_comps"][1] for b in batch])
        joined_mouth_mask = torch.stack([b["gt_facial_comps"][2] for b in batch])
        joined_batch['gt_facial_comps'] = (joined_leye_mask, joined_reye_mask, joined_mouth_mask)
    
    return joined_batch