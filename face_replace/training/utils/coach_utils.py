import time
from pathlib import Path
from typing import List, Dict, Callable, Any

import torch
from loguru import logger
from torch import Tensor
from torchvision import transforms

PROJECT_NAME = 'CVBoiler'


def nameit(func: Callable) -> Callable:
    def named(*args, **kwargs):
        logger.info(f"Running {func.__name__}...")
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        logger.info(f"{func.__name__} Done! {func.__name__} took {t1 - t0:0.2f} seconds to run.")
        return result

    return named


def aggregated_loss_dict(agg_loss_dict: List[Dict[str, float]]) -> Dict[str, float]:
    mean_values = {}
    for output in agg_loss_dict:
        for key, value in output.items():
            mean_values[key] = mean_values.setdefault(key, []) + [value]
    for key, vals in mean_values.items():
        if len(vals) > 0:
            mean_values[key] = sum(vals) / len(vals)
        else:
            logger.info(f"{key} has no value")
            mean_values[key] = 0
    return mean_values


def create_dir(dir_path: Path):
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path


def tensor2list(x: Tensor) -> List[Any]:
    return x.cpu().detach().tolist()


def clip_normalization():
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
    return t_clip_renorm


def perfect_shuffle(tensor: torch.Tensor) -> torch.Tensor:
    B = tensor.size(0)
    idx = torch.randperm(B)
    while (idx == torch.arange(B)).any():
        idx = torch.randperm(B)
    return tensor[idx]


VAE_ENCODER_LAYERS_TO_TRAIN = [
    'encoder.conv_in',
    'encoder.conv_in',
    'encoder.down_blocks.0.resnets.0.conv1',
    'encoder.down_blocks.0.resnets.0.conv1',
    'encoder.down_blocks.0.resnets.0.conv2',
    'encoder.down_blocks.0.resnets.0.conv2',
    'encoder.down_blocks.0.resnets.1.conv1',
    'encoder.down_blocks.0.resnets.1.conv1',
    'encoder.down_blocks.0.resnets.1.conv2',
    'encoder.down_blocks.0.resnets.1.conv2',
    'encoder.down_blocks.0.downsamplers.0.conv',
    'encoder.down_blocks.0.downsamplers.0.conv',
    'encoder.down_blocks.1.resnets.0.conv1',
    'encoder.down_blocks.1.resnets.0.conv1',
    'encoder.down_blocks.1.resnets.0.conv2',
    'encoder.down_blocks.1.resnets.0.conv2',
    'encoder.down_blocks.1.resnets.0.conv_shortcut',
    'encoder.down_blocks.1.resnets.0.conv_shortcut',
    'encoder.down_blocks.1.resnets.1.conv1',
    'encoder.down_blocks.1.resnets.1.conv1',
    'encoder.down_blocks.1.resnets.1.conv2',
    'encoder.down_blocks.1.resnets.1.conv2',
    'encoder.down_blocks.1.downsamplers.0.conv',
    'encoder.down_blocks.1.downsamplers.0.conv',
    'encoder.down_blocks.2.resnets.0.conv1',
    'encoder.down_blocks.2.resnets.0.conv1',
    'encoder.down_blocks.2.resnets.0.conv2',
    'encoder.down_blocks.2.resnets.0.conv2',
    'encoder.down_blocks.2.resnets.0.conv_shortcut',
    'encoder.down_blocks.2.resnets.0.conv_shortcut',
    'encoder.down_blocks.2.resnets.1.conv1',
    'encoder.down_blocks.2.resnets.1.conv1',
    'encoder.down_blocks.2.resnets.1.conv2',
    'encoder.down_blocks.2.resnets.1.conv2',
    'encoder.down_blocks.2.downsamplers.0.conv',
    'encoder.down_blocks.2.downsamplers.0.conv',
    'encoder.down_blocks.3.resnets.0.conv1',
    'encoder.down_blocks.3.resnets.0.conv1',
    'encoder.down_blocks.3.resnets.0.conv2',
    'encoder.down_blocks.3.resnets.0.conv2',
    'encoder.down_blocks.3.resnets.1.conv1',
    'encoder.down_blocks.3.resnets.1.conv1',
    'encoder.down_blocks.3.resnets.1.conv2',
    'encoder.down_blocks.3.resnets.1.conv2',
    'encoder.mid_block.attentions.0.to_q',
    'encoder.mid_block.attentions.0.to_q',
    'encoder.mid_block.attentions.0.to_k',
    'encoder.mid_block.attentions.0.to_k',
    'encoder.mid_block.attentions.0.to_v',
    'encoder.mid_block.attentions.0.to_v',
    'encoder.mid_block.attentions.0.to_out.0',
    'encoder.mid_block.attentions.0.to_out.0',
    'encoder.mid_block.resnets.0.conv1',
    'encoder.mid_block.resnets.0.conv1',
    'encoder.mid_block.resnets.0.conv2',
    'encoder.mid_block.resnets.0.conv2',
    'encoder.mid_block.resnets.1.conv1',
    'encoder.mid_block.resnets.1.conv1',
    'encoder.mid_block.resnets.1.conv2',
    'encoder.mid_block.resnets.1.conv2',
    'encoder.conv_out',
    'encoder.conv_out'
 ]