import sys
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import pyrallis
from git import GitError
from loguru import logger
import torch
from torch.utils.tensorboard import SummaryWriter

from face_replace.configs.train_config import TrainConfig
from face_replace.training.utils import git_utils
from face_replace.training.utils.coach_utils import create_dir, PROJECT_NAME
from face_replace.training.utils.types import BatchResults
from face_replace.training.utils import vis_utils


class CoachLogger:

    def __init__(self, cfg: TrainConfig, im_display_count: int = 4):
        self.cfg = cfg
        self.step = 0
        self.im_display_count = im_display_count

        self.wandb_task = self.init_wandb()
        self.log_dir = create_dir(self.cfg.log.exp_dir / 'logs')
        self.configure_loguru()
        self.log_config()

    def init_wandb(self):
        pass

    def log_config(self):
        with (self.cfg.log.exp_dir / 'config.yaml').open('w') as f:
            pyrallis.dump(self.cfg, f)
        self.log_message('\n' + pyrallis.dump(self.cfg))

    def configure_loguru(self):
        logger.remove()
        format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>'
        logger.add(sys.stdout, colorize=True, format=format)
        logger.add(self.log_dir / 'log.txt', colorize=False, format=format)

    def log_message(self, msg: str):
        logger.info(msg)

    def log_metrics(self, metrics_dict: Dict[str, float], prefix: str):
        self.log_message(f'Metrics for {prefix}, step {self.step}')
        for key, value in metrics_dict.items():
            self.log_message(f'\t{key} = {value:0.4f}')

    def update_step(self, step: int):
        self.step = step

    def vis_batch(self, batch_results: BatchResults, title: str, subscript: Optional[str] = None):
        joined_image, wandb_images = vis_utils.vis_data(results=batch_results)
        out_dir = self.log_dir / title
        out_dir.mkdir(exist_ok=True, parents=True)
        if subscript:
            out_name = f'{subscript}_{self.step:04d}.jpg'
            out_mixed_name = f'mixed_{subscript}_{self.step:04d}.jpg'
        else:
            out_name = f'{self.step:04d}.jpg'
            out_mixed_name = f'mixed_{self.step:04d}.jpg'

        joined_image.save(out_dir / out_name)
        return wandb_images
    
    def vis_attn_batch(self, batch_results: BatchResults, attn_probs: List[torch.Tensor], title: str, subscript: Optional[str] = None):
        joined_attn_image, wandb_attn_images = vis_utils.vis_attn_probs(results=batch_results, attn_probs=attn_probs)
        out_dir = self.log_dir / title
        out_dir.mkdir(exist_ok=True, parents=True)
        if subscript:
            out_name = f'attn_{subscript}_{self.step:04d}.jpg'
        else:
            out_name = f'attn_{self.step:04d}.jpg'
        joined_attn_image.save(out_dir / out_name)
        return wandb_attn_images
