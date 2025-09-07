from typing import Tuple
from torchvision import transforms

from face_replace.configs.train_config import DataConfig, TrainConfig
from face_replace.data.datasets.coach_dataset import CoachDataset
from face_replace.data.datasets.paired_dataset import PairedDataset
from face_replace.data.datasets.restore_dataset import RestoreDataset, RestoreDatasetTest
from face_replace.data.transforms.paired_transforms import PairedColorJitter, PairedCompress, PairedRandomBlur, PairedTransform

import numpy as np



def get_dataset(cfg: DataConfig, full_cfg: TrainConfig) -> Tuple[CoachDataset, CoachDataset]:
    if cfg.dataset_type == 'debug':
        train_dataset, test_dataset = get_debug_dataset(cfg, full_cfg=full_cfg)
    elif cfg.dataset_type == 'augmentations':
        train_dataset, test_dataset = get_augmentation_dataset(cfg, full_cfg=full_cfg)
    elif cfg.dataset_type == 'face_restore':
        train_dataset, test_dataset = get_restore_dataset(cfg, full_cfg=full_cfg)
    else:
        raise ValueError(f"No such dataset type: {cfg.dataset_type}!")
    return train_dataset, test_dataset


def get_infer_transforms(cfg: DataConfig):
    transform = transforms.Compose([
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg.resolution),
        transforms.ToTensor(),
    ])
    return transform


def get_debug_dataset(cfg: DataConfig, full_cfg: TrainConfig):
    train_transforms = transforms.Compose([
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg.resolution),
        transforms.ToTensor(),
    ])
    joined_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
    ])

    train_dataset = PairedDataset(dataset_folder=cfg.data_root,
                                  identity_names=None,
                                  transform=train_transforms,
                                  max_conditioning_images=cfg.max_conditioning_images,
                                  joined_transforms=joined_transforms,
                                  condition_on_face_embeds=full_cfg.model.condition_on_face_embeds,
                                  store_landmarks=cfg.store_landmarks,
                                  augment_masks=cfg.augment_masks)
    test_dataset = PairedDataset(dataset_folder=cfg.val_data_root,
                                 identity_names=None,
                                 transform=get_infer_transforms(cfg),
                                 max_conditioning_images=cfg.max_conditioning_images,
                                 condition_on_face_embeds=full_cfg.model.condition_on_face_embeds,
                                 store_landmarks=cfg.store_landmarks,
                                 augment_masks=False)
    return train_dataset, test_dataset

def get_restore_dataset(cfg: DataConfig, full_cfg: TrainConfig):
    joined_transforms = [
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg.resolution),
        transforms.Grayscale(num_output_channels=3),
        PairedColorJitter(brightness=0.3 , contrast=0.3, saturation=0.3),
    ]
    probabilities = [
        1.0,
        1.0,
        0.1, 
        1.0, 
    ]
    joined_transforms = PairedTransform(transforms=joined_transforms, probabilities=probabilities)

    test_joined_transforms = [
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg.resolution),
    ]
    test_probabilities = [
        1.0,
        1.0,
    ]
    test_joined_transforms = PairedTransform(transforms=test_joined_transforms, probabilities=test_probabilities)

    final_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    to_tensor = transforms.ToTensor()

    get_gt_attn_probs = full_cfg.optim.lambda_landmark > 0
    get_attn_pos_reg = full_cfg.optim.lambda_pos_reg > 0
    get_attn_neg_reg = full_cfg.optim.lambda_neg_reg > 0
    return_degrade_transforms = full_cfg.optim.lambda_cycle > 0
    train_input = full_cfg.model.train_input
    facial_components = full_cfg.optim.lambda_facial_comp > 0

    train_dataset = RestoreDataset(dataset_folder=cfg.data_root,
                                   identity_names=None,
                                    norm_transform=final_normalize,
                                    to_tensor_transform=to_tensor,
                                    max_conditioning_images=cfg.max_conditioning_images,
                                    joined_transforms=joined_transforms,
                                    conditioning_transforms=get_infer_transforms(cfg),
                                    condition_on_face_embeds=full_cfg.model.condition_on_face_embeds,
                                    resolution=cfg.resolution,
                                    return_degrade_transforms=return_degrade_transforms,
                                    get_gt_attn_probs=get_gt_attn_probs,
                                    train_input=train_input,
                                    get_attn_pos_reg=get_attn_pos_reg,
                                    get_attn_neg_reg=get_attn_neg_reg,
                                    get_facial_comps=facial_components)
    test_dataset = RestoreDatasetTest(dataset_folder=cfg.val_data_root,
                                 norm_transform=final_normalize,
                                 to_tensor_transform=to_tensor,
                                 conditioning_transforms=get_infer_transforms(cfg),
                                 joined_transforms=test_joined_transforms,
                                 max_conditioning_images=cfg.max_conditioning_images,
                                 condition_on_face_embeds=full_cfg.model.condition_on_face_embeds)
    return train_dataset, test_dataset


def get_augmentation_dataset(cfg: DataConfig, full_cfg: TrainConfig):
    joined_transforms = [
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg.resolution),
        transforms.Grayscale(num_output_channels=3),
        PairedColorJitter(brightness=0.3 , contrast=0.3, saturation=0.3),
    ]
    probabilities = [
        1.0,
        1.0,
        0.1, 
        1.0, 
    ]
    joined_transforms = PairedTransform(transforms=joined_transforms, probabilities=probabilities)

    test_joined_transforms = [
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg.resolution),
    ]
    test_probabilities = [
        1.0,
        1.0,
    ]
    test_joined_transforms = PairedTransform(transforms=test_joined_transforms, probabilities=test_probabilities)

    final_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = PairedDataset(dataset_folder=cfg.data_root,
                                  identity_names=None,
                                  transform=final_transforms,
                                  max_conditioning_images=cfg.max_conditioning_images,
                                  joined_transforms=joined_transforms,
                                  conditioning_transforms=get_infer_transforms(cfg),
                                  condition_on_face_embeds=full_cfg.model.condition_on_face_embeds,
                                  augment_masks=cfg.augment_masks,
                                  mode='train')
    test_dataset = PairedDataset(dataset_folder=cfg.val_data_root,
                                 identity_names=None,
                                 transform=final_transforms,
                                 conditioning_transforms=get_infer_transforms(cfg),
                                 joined_transforms=test_joined_transforms,
                                 max_conditioning_images=cfg.max_conditioning_images,
                                 condition_on_face_embeds=full_cfg.model.condition_on_face_embeds,
                                 augment_masks=cfg.augment_masks,
                                 mode='test')
    return train_dataset, test_dataset
