import dataclasses
from typing import List, Tuple, Optional, Dict, Any

import diffusers
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs
from datetime import timedelta
import random
import time
from pytorch_msssim import ms_ssim

from face_replace.configs.train_config import TrainConfig
from face_replace.data import data_setups
from face_replace.data.datasets.coach_dataset import CoachDataset
from face_replace.data.datasets.restore_dataset import custom_collate_fn
from face_replace.models.attn_processors import SharedAttnProcessor
from face_replace.models.face_replace_model import FaceReplaceModel
from face_replace.training.criteria.id_loss import IDLoss
from face_replace.training.logging.coach_logger import CoachLogger
from face_replace.training.utils import coach_utils
from face_replace.training.utils.coach_utils import perfect_shuffle
from face_replace.training.utils.types import BatchResults
from face_replace.training.criteria.lpips import lpips
from face_replace.training import vision_aided_loss


class Coach:
    """
    coach /koCH/ (noun) - someone whose job is to train
    Think of it as your main script, written as a class
    This gives you flexibility while keeping things relatively structured
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = 'cuda'
        self.best_val_loss = None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.optim.gradient_accumulation_steps,
            mixed_precision='bf16' if self.cfg.optim.mixed_precision else 'no',
            log_with='tensorboard' if self.cfg.log.log2wandb else None,
            project_dir=cfg.log.exp_dir,
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(seconds=3600)),
                DistributedDataParallelKwargs(broadcast_buffers=False)
            ]
        )

        if self.accelerator.is_main_process:
            self.create_exp_dir()
        
        self.logger = CoachLogger(cfg=self.cfg)
        self.checkpoint_dir = coach_utils.create_dir(self.cfg.log.exp_dir / 'checkpoints')

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.cfg.compute.seed is not None:
            set_seed(self.cfg.compute.seed)

        # Initialize modules
        self.model = self.init_model()

        if self.cfg.model.net_type == "pix2pix_turbo":
            if self.cfg.optim.enable_xformers_memory_efficient_attention and is_xformers_available():
                self.model.net.unet.enable_xformers_memory_efficient_attention()
            elif not is_xformers_available():
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")
            else:
                print("Not enabling xformers memory efficient attention")

            if self.cfg.optim.gradient_checkpointing:
                self.model.net.unet.enable_gradient_checkpointing()

        self.net_disc = self.init_discriminator()
        self.net_lpips = self.init_loss_networks()
        self.id_loss = IDLoss(pretrained_arcface_path='path/to/external/models/model_ir_se50.pth',
                              device=self.accelerator.device,
                              dtype=torch.float32)

        self.optimizer, self.lr_scheduler, self.optimizer_disc, self.lr_scheduler_disc, self.layers_to_opt = self.init_optimizer()
        if self.cfg.model.checkpoint_path is not None:
            checkpoint_dict = torch.load(self.cfg.model.checkpoint_path, map_location='cpu')
            if 'optimizer' in checkpoint_dict:
                print(f"Loading optimizer state from path: {self.cfg.model.checkpoint_path}...")
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            checkpoint_dict = None
            print("Emptying cache")
            torch.cuda.empty_cache()

        self.train_dataset, self.test_dataset = self.init_datasets()
        self.train_dataloader, self.test_dataloader = self.init_dataloaders()

        components_to_prepare = [self.model.net,
                                 self.net_disc,
                                 self.optimizer,
                                 self.optimizer_disc,
                                 self.train_dataloader,
                                 self.lr_scheduler,
                                 self.lr_scheduler_disc]
        (
            self.model.net,
            self.net_disc,
            self.optimizer,
            self.optimizer_disc,
            self.train_dataloader,
            self.lr_scheduler,
            self.lr_scheduler_disc
        ) = self.accelerator.prepare(*components_to_prepare)

        self.net_lpips, self.id_loss = self.accelerator.prepare(self.net_lpips, self.id_loss)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        print(f"Using weight type: {self.weight_dtype}")

        self.model.net.to(self.accelerator.device, dtype=self.weight_dtype)
        self.net_disc.to(self.accelerator.device, dtype=self.weight_dtype)
        self.net_lpips.to(self.accelerator.device, dtype=self.weight_dtype)

        if self.accelerator.is_main_process:
            tracker_config = dataclasses.asdict(self.cfg)
            self.accelerator.init_trackers('Face-Replace')

    def forward_batch(self, batch: Dict[str, Any]) -> BatchResults:
        x_src = batch["conditioning_pixel_values"].to(self.device, self.weight_dtype)
        x_tgt = batch["output_pixel_values"].to(self.device, self.weight_dtype)
        
        face_embeds = None
        if batch["face_embed"] is not None:
            face_embeds = batch["face_embed"].to(self.device, self.weight_dtype) if 'face_embed' in batch else None
        
        gt_attn_probs_masks_layer = None
        if batch["gt_attn_probs"] is not None:
            num_layers = len(batch['gt_attn_probs'][0])
            gt_attn_probs = []
            mask_attn_probs = []
            chosen_layer = batch["gt_attn_probs"][2]
            chosen_cond = batch["gt_attn_probs"][3]
            for i in range(num_layers):
                gt_attn_probs.append(batch['gt_attn_probs'][0][i].to(self.device, self.weight_dtype))
                mask_attn_probs.append(batch['gt_attn_probs'][1][i].to(self.device, torch.bool))
            gt_attn_probs_masks_layer = (gt_attn_probs, mask_attn_probs, chosen_layer, chosen_cond)
        
        gt_attn_pos_reg_idx = batch["gt_attn_pos_reg_idx"]
        gt_attn_neg_reg_idx = batch["gt_attn_neg_reg_idx"]

        gt_facial_comps = batch['gt_facial_comps']
        
        conditioning_images = None
        if 'conditioning_images' in batch:
            conditioning_images = batch["conditioning_images"].to(self.device, self.weight_dtype)
        
        valid_indices = batch["valid_indices"] if 'valid_indices' in batch else None
        
        with torch.cuda.amp.autocast():
            x_tgt_pred, _, shared_attn_probs = self.model.net(
                x_src,
                face_embeds=face_embeds,
                conditioning_images=conditioning_images,
                valid_indices=valid_indices,
                mask=batch.get("mask", None),
                return_self_attention_maps=self.cfg.optim.lambda_attn_reg > 0 or 
                    self.cfg.optim.lambda_landmark > 0 or self.cfg.optim.lambda_pos_reg > 0 or 
                    self.cfg.optim.lambda_neg_reg > 0,
            )
        loss, loss_dict = self.calc_loss(batch=batch, 
                                            gts=x_tgt, 
                                            pred=x_tgt_pred,
                                            inp=x_src,
                                            shared_attn_probs=shared_attn_probs,
                                            gt_attn_probs_masks=gt_attn_probs_masks_layer,
                                            gt_attn_pos_reg_idx=gt_attn_pos_reg_idx,
                                            gt_attn_neg_reg_idx=gt_attn_neg_reg_idx,
                                            gt_facial_comps=gt_facial_comps)
        batch_results = BatchResults(batch, 
                                        pred=x_tgt_pred, 
                                        pred_mix_id=None,
                                        conditions=None,
                                        loss=loss, 
                                        loss_dict=loss_dict)
        
        return loss, batch_results
    
    def crop_image_with_mask(self, img, mask, gt_img=None):
        nonzero_indices = torch.nonzero(mask)
        y_min, x_min = nonzero_indices[:, 0].min(), nonzero_indices[:, 1].min()
        y_max, x_max = nonzero_indices[:, 0].max(), nonzero_indices[:, 1].max()

        if not (gt_img is None):
            cropped_gt_img = gt_img[:, :, y_min:y_max+1, x_min:x_max+1]
        cropped_img = img[:, :, y_min:y_max+1, x_min:x_max+1]
        
        if gt_img is None:
            return cropped_img, None
        return cropped_img, cropped_gt_img



    @coach_utils.nameit
    def train(self):
        self.model.net.train()
        if self.accelerator.num_processes > 1: 
            self.model.net.module.set_train()
        else:
            self.model.net.set_train()
        
        while self.train_step < self.cfg.steps.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                l_acc = [self.model.net, self.net_disc]
                with self.accelerator.accumulate(*l_acc):
                    
                    loss, batch_results = self.forward_batch(batch)

                    """ Generator loss: fool the discriminator """
                    x_tgt_pred = batch_results.pred
                    loss_g = self.net_disc(x_tgt_pred, for_G=True).mean() * self.cfg.optim.lambda_gan
                    batch_results.loss_dict['loss_g'] = loss_g.item()
                    loss += loss_g
                    
                    # Facial comp loss
                    if self.cfg.optim.lambda_facial_comp > 0 and batch['gt_facial_comps'] is not None:
                        leye_mask, reye_mask, mouth_mask = batch['gt_facial_comps']
                        leye_img, _ = self.crop_image_with_mask(x_tgt_pred, leye_mask)
                        reye_img, _ = self.crop_image_with_mask(x_tgt_pred, reye_mask)
                        mouth_img, _ = self.crop_image_with_mask(x_tgt_pred, mouth_mask)
                        leye_loss_g = self.net_disc(leye_img, for_G=True).mean()
                        reye_loss_g = self.net_disc(reye_img, for_G=True).mean()
                        mouth_loss_g = self.net_disc(mouth_img, for_G=True).mean()
                        total_fc_g_loss = leye_loss_g + reye_loss_g + mouth_loss_g
                        batch_results.loss_dict['fc_loss_g'] = total_fc_g_loss.item()
                        loss += total_fc_g_loss * self.cfg.optim.lambda_gan * self.cfg.optim.lambda_facial_comp
                        
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.layers_to_opt, self.cfg.optim.clip_grad_max_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    """ Discriminator loss: fake image vs real image """
                    # Real Image
                    x_tgt = batch["output_pixel_values"]
                    loss_disc_real = self.net_disc(x_tgt, for_real=True).mean() * self.cfg.optim.lambda_gan
                    # Fake image
                    x_tgt_pred = batch_results.pred.detach()
                    loss_disc_fake = self.net_disc(x_tgt_pred, for_real=False).mean() * self.cfg.optim.lambda_gan
                    loss_disc = 0.5 * (loss_disc_real + loss_disc_fake)

                    if self.cfg.optim.lambda_facial_comp > 0 and batch['gt_facial_comps'] is not None:
                        leye_mask, reye_mask, mouth_mask = batch['gt_facial_comps']
                        leye_img, leye_real = self.crop_image_with_mask(x_tgt_pred, leye_mask, gt_img=x_tgt)
                        reye_img, reye_real = self.crop_image_with_mask(x_tgt_pred, reye_mask, gt_img=x_tgt)
                        mouth_img, mouth_real = self.crop_image_with_mask(x_tgt_pred, mouth_mask, gt_img=x_tgt)
                        leye_loss_d = self.net_disc(leye_real, for_real=True).mean()
                        reye_loss_d = self.net_disc(reye_real, for_real=True).mean()
                        mouth_loss_d = self.net_disc(mouth_real, for_real=True).mean()
                        leye_loss_df = self.net_disc(leye_img, for_real=False).mean()
                        reye_loss_df = self.net_disc(reye_img, for_real=False).mean()
                        mouth_loss_df = self.net_disc(mouth_img, for_real=False).mean()

                        total_fc_d_loss = leye_loss_d + reye_loss_d + mouth_loss_d + leye_loss_df + reye_loss_df + mouth_loss_df
                        batch_results.loss_dict['fc_loss_d'] = total_fc_d_loss.item()
                        loss_disc += total_fc_d_loss * self.cfg.optim.lambda_gan * self.cfg.optim.lambda_facial_comp

                    self.accelerator.backward(loss_disc)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.net_disc.parameters(), self.cfg.optim.clip_grad_max_norm)
                    self.optimizer_disc.step()
                    self.optimizer_disc.zero_grad()
                    batch_results.loss_dict['loss_d'] = loss_disc.item()

                if self.accelerator.sync_gradients:
                    self.logger.update_step(step=self.train_step)

                    if self.accelerator.is_main_process:

                        logs = batch_results.loss_dict.copy()

                        # Logging related
                        if self.time_to(self.cfg.steps.image_interval):
                            wandb_images = self.logger.vis_batch(batch_results=batch_results, title='vis_batch/train')
                            if self.cfg.log.log2wandb:
                                logs.update({'images': wandb_images})

                            # Visualize the attention probabilities
                            if self.cfg.log.vis_attention:
                                self_attn_processors = [p for p in self.model.net.unet.attn_processors.values() 
                                                        if type(p) == SharedAttnProcessor and p.self_attn_idx is not None]
                                attn_probs = [p.attention_probs for p in self_attn_processors]
                                wandb_attn_images = self.logger.vis_attn_batch(batch_results=batch_results, 
                                                                            attn_probs=attn_probs,
                                                                            title='vis_batch/train')
                                if self.cfg.log.log2wandb:
                                    logs.update({'attn_images': wandb_attn_images})
                                del self_attn_processors, attn_probs

                        if self.time_to(self.cfg.steps.metric_interval):
                            batch_results.loss_dict.update({'lr': self.lr_scheduler.get_last_lr()[0]})
                            self.logger.log_metrics(metrics_dict=batch_results.loss_dict, prefix='train')

                        if self.time_to(self.cfg.steps.save_interval):
                            self.checkpoint_me(is_best=False)

                        # Validation related
                        if self.time_to(self.cfg.steps.val_interval):
                            val_loss = self.validate()
                            if val_loss and (self.best_val_loss is None or val_loss < self.best_val_loss):
                                self.best_val_loss = val_loss
                                self.checkpoint_me(is_best=True)

                        logs = {f'train/{k}': v for k, v in logs.items()}
                        self.accelerator.log(logs, step=self.train_step)

                        self.train_step += 1

                        if self.is_final_step():
                            self.logger.log_message("OMG! Finished Training!")
                            return

    @coach_utils.nameit
    def validate(self) -> Optional[float]:
        self.model.net.eval()
        if self.accelerator.num_processes > 1: 
            self.model.net.module.set_eval()
        else:
            self.model.net.set_eval()
        
        agg_loss_dict = []
        val_images, val_attn_images = [], []
        for batch_idx, batch in tqdm(enumerate(self.test_dataloader)):

            with torch.no_grad():

                _, batch_results = self.forward_batch(batch)
                agg_loss_dict.append(batch_results.loss_dict)

            # Logging related
            if batch_idx <= self.cfg.log.val_vis_count:
                wandb_images = self.logger.vis_batch(batch_results=batch_results,
                                                     title='vis_batch/val',
                                                     subscript=f"{batch_idx:04d}")
                val_images.append(wandb_images[0])

                # Visualize the attention probabilities (just do this for a few batches since it takes a long time)
                if self.cfg.log.vis_attention and batch_idx <= 5:
                    self_attn_processors = [p for p in self.model.net.unet.attn_processors.values() 
                                            if type(p) == SharedAttnProcessor and p.self_attn_idx is not None]
                    attn_probs = [p.attention_probs for p in self_attn_processors]
                    wandb_attn_images = self.logger.vis_attn_batch(batch_results=batch_results, 
                                                                    attn_probs=attn_probs,
                                                                    title='vis_batch/val',
                                                                    subscript=f"{batch_idx:04d}")
                    val_attn_images.append(wandb_attn_images[0])
                    del self_attn_processors, attn_probs

        self.model.net.train()
        if self.accelerator.num_processes > 1: 
            self.model.net.module.set_train()
        else:
            self.model.net.set_train()
        logs = coach_utils.aggregated_loss_dict(agg_loss_dict)
        logs = {f'val/{k}': v for k, v in logs.items()}
        self.logger.log_metrics(metrics_dict=logs, prefix='val')
        if self.cfg.log.log2wandb:
            logs.update({'val/images': val_images})
            if len(val_attn_images) > 0:
                logs.update({'val/attn_images': val_attn_images})
        self.accelerator.log(logs, step=self.train_step)
        return logs['val/loss']

    def time_to(self, interval: int) -> bool:
        time_to_by_interval = self.train_step > 0 and self.train_step % interval == 0
        time_to_by_final_step = self.is_final_step()
        return time_to_by_interval or time_to_by_final_step

    def is_final_step(self):
        return self.train_step == self.cfg.steps.max_steps - 1

    def checkpoint_me(self, is_best: bool = False):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.train_step}.pt'
        self.logger.log_message(f"Saving checkpoint {save_name} to {self.checkpoint_dir}")
        checkpoint_path = self.checkpoint_dir / save_name
        torch.save(self.get_save_dict(), checkpoint_path)
        if is_best:
            with (self.checkpoint_dir / "timestep.txt").open('a') as f:
                f.write(f"Step - {self.train_step}, Loss - {self.best_val_loss:0.3f}\n")

    def init_model(self) -> nn.Module:
        model = FaceReplaceModel(cfg=self.cfg.model, full_cfg=self.cfg)
        model.to(self.device)
        return model

    def init_discriminator(self) -> nn.Module:
        if self.cfg.optim.gan_disc_type == "clip":
            net_disc = vision_aided_loss.Discriminator(cv_type='clip',
                                                       loss_type=self.cfg.optim.gan_loss_type,
                                                       device="cuda")
        elif self.cfg.optim.gan_disc_type == 'dinov2': 
            net_disc = vision_aided_loss.Discriminator(cv_type='dinov2',
                                                       loss_type=self.cfg.optim.gan_loss_type,
                                                       device="cuda")
        else:
            raise ValueError(f"No such discriminator type: {self.cfg.optim.gan_disc_type}!")
            
        net_disc = net_disc.cuda()
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()
        # turn off efficient attention for the discriminator
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False
        
        return net_disc

    def init_loss_networks(self):
        net_lpips = lpips.LPIPS(net='vgg').cuda()
        net_lpips.requires_grad_(False)
        return net_lpips

    def init_optimizer(self) -> torch.optim.Optimizer:
        if "pix2pix_turbo" not in self.cfg.model.net_type:
            raise ValueError("Current only support optimizers for pix2pix turbo!")

        layers_to_opt = []
        for n, _p in self.model.net.unet.named_parameters():
            if "lora" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)
        layers_to_opt += list(self.model.net.unet.conv_in.parameters())
        for n, _p in self.model.net.vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)

        if self.cfg.model.use_shortcuts:
            layers_to_opt = layers_to_opt + \
                            list(self.model.net.vae.decoder.skip_conv_1.parameters()) + \
                            list(self.model.net.vae.decoder.skip_conv_2.parameters()) + \
                            list(self.model.net.vae.decoder.skip_conv_3.parameters()) + \
                            list(self.model.net.vae.decoder.skip_conv_4.parameters())

        optimizer = torch.optim.AdamW(layers_to_opt,
                                      lr=self.cfg.optim.learning_rate,
                                      betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
                                      weight_decay=self.cfg.optim.adam_weight_decay,
                                      eps=self.cfg.optim.adam_epsilon)
        lr_scheduler = get_scheduler(name=self.cfg.optim.scheduler_type.value,
                                     optimizer=optimizer,
                                     num_warmup_steps=self.cfg.optim.lr_warmup_steps * self.accelerator.num_processes,
                                     num_training_steps=self.cfg.steps.max_steps * self.accelerator.num_processes,
                                     num_cycles=self.cfg.optim.lr_num_cycles,
                                     power=self.cfg.optim.lr_power)

        optimizer_disc = torch.optim.AdamW(self.net_disc.parameters(),
                                           lr=self.cfg.optim.learning_rate,
                                           betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
                                           weight_decay=self.cfg.optim.adam_weight_decay,
                                           eps=self.cfg.optim.adam_epsilon)
        lr_scheduler_disc = get_scheduler(self.cfg.optim.scheduler_type.value,
                                          optimizer=optimizer_disc,
                                          num_warmup_steps=self.cfg.optim.lr_warmup_steps * self.accelerator.num_processes,
                                          num_training_steps=self.cfg.steps.max_steps * self.accelerator.num_processes,
                                          num_cycles=self.cfg.optim.lr_num_cycles,
                                          power=self.cfg.optim.lr_power)
        return optimizer, lr_scheduler, optimizer_disc, lr_scheduler_disc, layers_to_opt

    def init_scheduler(self) -> torch.optim.lr_scheduler:
        raise NotImplementedError("Scheduler not implemented yet!")

    def init_datasets(self) -> Tuple[CoachDataset, CoachDataset]:
        train_dataset, test_dataset = data_setups.get_dataset(self.cfg.data, full_cfg=self.cfg)
        train_dataset.tokenizer = self.model.net.tokenizer
        test_dataset.tokenizer = self.model.net.tokenizer

        # Try to overfit
        if self.cfg.data.overfit:
            self.logger.log_message("WARNING: Running in overfit mode!")
            train_dataset.shuffle()  # So that no all samples have same label
            train_dataset.paths = train_dataset.paths[:self.cfg.compute.batch_size]
            test_dataset = train_dataset

        self.logger.log_message(f"Number of training samples: {len(train_dataset)}")
        self.logger.log_message(f"Number of testing samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def init_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.cfg.compute.batch_size,
                                      shuffle=True,
                                      num_workers=int(self.cfg.compute.workers),
                                      drop_last=True,
                                      worker_init_fn=worker_init_fn,
                                      collate_fn=custom_collate_fn)
        self.test_dataset.shuffle()  # Just shuffle so visualization will be fixed but interesting
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.cfg.compute.test_batch_size,
                                     shuffle=False,
                                     num_workers=int(self.cfg.compute.test_workers),
                                     drop_last=False,
                                     worker_init_fn=worker_init_fn,
                                     collate_fn=custom_collate_fn)
        return train_dataloader, test_dataloader
    
    def calc_landmark_loss(self, attn_prob_pred, attn_prob_gt, mask_layer, chosen_cond):
        total_loss = 0

        pred_min_val = attn_prob_pred.min()
        pred_max_val = attn_prob_pred.max()
        attn_prob_pred = (attn_prob_pred - pred_min_val)/(pred_max_val - pred_min_val)

        gt_min_val = attn_prob_gt.min()
        gt_max_val = attn_prob_gt.max()
        attn_prob_gt = (attn_prob_gt - gt_min_val)/(gt_max_val - gt_min_val)

        _, _, q_pix, k_pix = attn_prob_pred.shape
        start_idx = q_pix * chosen_cond
        end_idx = q_pix * (chosen_cond + 1)
        attn_prob_pred_cond = attn_prob_pred[:,:,:,start_idx:end_idx]
        # attn_prob_gt_cond = attn_prob_gt[:,:,:,start_idx:end_idx]
        mask_cond = mask_layer.unsqueeze(1).unsqueeze(3).expand_as(attn_prob_pred_cond)
        reduced_tensor_pred = attn_prob_pred_cond[mask_cond]
        reduced_tensor_gt = attn_prob_gt[mask_cond]
        # print("------------------")
        # print(f"attn_prob_pred: {attn_prob_pred.shape}")
        # print(f"attn_prob_gt: {attn_prob_gt.shape}")
        # print(f"mask_cond: {mask_cond.shape}")
        # print(f"start_idx: {start_idx}")
        # print(f"end_idx: {end_idx}")
        # print(f"reduced_tensor_pred: {reduced_tensor_pred.shape}")
        # attn_prob_pred[:,:,:,start_idx:end_idx] = mask_cond.unsqueeze(2) * attn_prob_pred[:,:,:,start_idx:end_idx]
        # attn_prob_gt[:,:,:,start_idx:end_idx] = mask_cond.unsqueeze(2) * attn_prob_gt[:,:,:,start_idx:end_idx]
        total_loss += F.mse_loss(reduced_tensor_pred.to(self.weight_dtype), reduced_tensor_gt)
        return total_loss

    def calc_loss(self, 
                  batch: Dict[str, Any], 
                  gts: Tensor, 
                  pred: Tensor, 
                  inp: Tensor,
                  shared_attn_probs: List[Tensor] = None,
                  gt_attn_probs_masks: Tuple[List[Tensor],List[Tensor],List[int],List[int]] = None,
                  gt_attn_pos_reg_idx: List[int] = None,
                  gt_attn_neg_reg_idx: List[int] = None,
                  gt_facial_comps: Tuple[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
        loss_dict = {}
        if self.cfg.optim.lambda_l1 > 0:
            loss_rec = F.l1_loss(pred.float(), gts.float(), reduction="mean")
            loss_dict['loss_l1'] = loss_rec.item()
            loss_lambda = self.cfg.optim.lambda_l1
        else:
            loss_rec = F.mse_loss(pred.float(), gts.float(), reduction="mean")
            loss_dict['loss_l2'] = loss_rec.item()
            loss_lambda = self.cfg.optim.lambda_l2

        loss_lpips = self.net_lpips(pred.float(), gts.float()).mean()
        loss_dict['loss_lpips'] = loss_lpips.item()
        loss = loss_rec * loss_lambda + loss_lpips * self.cfg.optim.lambda_lpips

        if self.cfg.optim.lambda_ssim > 0:
            pred_norm = (pred.float() + 1) / 2
            gt_norm = (gts.float() + 1) / 2
            loss_ssim = 1 - ms_ssim(pred_norm, gt_norm, data_range=1, size_average=True)
            loss_dict['loss_ssim'] = loss_ssim.item()
            loss += self.cfg.optim.lambda_ssim * loss_ssim

        if self.cfg.optim.lambda_id_loss > 0:
            loss_id, sim_id = self.id_loss(predicted_pixel_values=pred.float(), target_pixel_values=gts.float())
            loss_dict['loss_id'] = loss_id.item()
            loss_dict['sim_id'] = sim_id.item()
            loss += loss_id * self.cfg.optim.lambda_id_loss

        if self.cfg.optim.lambda_attn_reg > 0 and shared_attn_probs is not None:
            reg_losses = []
            for attn_probs in shared_attn_probs:
                bs, n_heads, n_tokens, _ = attn_probs.size()
                attn_probs_ = attn_probs.view(bs, n_heads, n_tokens, 5, n_tokens)
                attn_probs_ = attn_probs_[:, :, :, 1:, :]
                mean_activations = attn_probs_.mean(dim=-1)
                max_indices = mean_activations.argmax(dim=-1)
                one_hot_indices = torch.nn.functional.one_hot(max_indices, num_classes=5).to(dtype=torch.float32)
                average_one_hot = one_hot_indices.mean(dim=2)
                uniform_dist = torch.full_like(average_one_hot, fill_value=0.2)
                # Compute cross-entropy
                log_probs = (average_one_hot + 1e-8).log()
                reg_loss = -torch.sum(log_probs * uniform_dist) / log_probs.size(0)
                reg_losses.append(reg_loss)
            loss_attn_reg = sum(reg_losses) / len(reg_losses)
            loss_dict['loss_attn_reg'] = loss_attn_reg.item()
            loss += loss_attn_reg * self.cfg.optim.lambda_attn_reg
        
        if self.cfg.optim.lambda_cycle > 0 and batch["degrade_transforms"] is not None:
            degrade_transforms = batch["degrade_transforms"]
            loss_cycle = 0
            for idx, degrade_transform in enumerate(degrade_transforms):
                degraded_pred = degrade_transform(pred[idx].float())
                loss_cycle += F.mse_loss(degraded_pred, inp[idx].detach().float(), reduction="mean")
            lambda_cycle = self.cfg.optim.lambda_cycle
            loss_cycle /= len(degrade_transforms)
            loss += lambda_cycle * loss_cycle
            loss_dict['loss_cycle'] = loss_cycle.item()

        if self.cfg.optim.lambda_landmark > 0 and shared_attn_probs is not None and gt_attn_probs_masks is not None:
            (gt_attn_probs, masks, chosen_layer, chosen_cond) = gt_attn_probs_masks
            # TODO: This only works for 1 batch
            shared_attn_prob_layer = shared_attn_probs[chosen_layer[0]]
            gt_attn_probs = gt_attn_probs[0]
            masks = masks[0]
            loss_landmarks = self.calc_landmark_loss(shared_attn_prob_layer, gt_attn_probs, masks, chosen_cond[0])
            lambda_landmark = self.cfg.optim.lambda_landmark
            loss += lambda_landmark * loss_landmarks
            loss_dict['loss_landmark'] = loss_landmarks.item()
        
        if self.cfg.optim.lambda_pos_reg > 0 or self.cfg.optim.lambda_neg_reg > 0:
            softmax = nn.Softmax(dim=1)
            nll_loss = nn.NLLLoss()
            num_layers = len(shared_attn_probs)
            attn_size = shared_attn_probs[0].shape[2]
            num_conds = shared_attn_probs[0].shape[-1]//attn_size
            attn_reg_chosen_layer = random.randint(0,num_layers-1)

            if self.cfg.optim.lambda_pos_reg > 0 and gt_attn_pos_reg_idx[0] is not None:
                # TODO: This only works for 1 batch
                gt_attn_pos_reg_idx = torch.tensor(gt_attn_pos_reg_idx, device = self.device)
                attn_prob = shared_attn_probs[attn_reg_chosen_layer]
                means = torch.zeros(num_conds, device=self.device, dtype=self.weight_dtype)
                for cond_img_index in range(num_conds):
                    attn_selection = attn_prob[:, :, :, attn_size*(cond_img_index):attn_size*(cond_img_index + 1)]
                    means[cond_img_index] = attn_selection.sum()
                means = means/means.max()
                pos_reg_softmax = softmax(means.unsqueeze(0))
                pos_reg_log_softmax = torch.log(pos_reg_softmax)
                total_pos_reg_loss = nll_loss(pos_reg_log_softmax, gt_attn_pos_reg_idx)
                lambda_pos_reg = self.cfg.optim.lambda_pos_reg
                loss += lambda_pos_reg * total_pos_reg_loss
                loss_dict['loss_attn_pos_reg'] = total_pos_reg_loss.item()
            
            if self.cfg.optim.lambda_neg_reg > 0 and gt_attn_neg_reg_idx[0] is not None:
                # TODO: This only works for 1 batch
                gt_attn_neg_reg_idx = torch.tensor(gt_attn_neg_reg_idx, device = self.device)
                attn_prob = shared_attn_probs[attn_reg_chosen_layer]
                means = torch.zeros(num_conds, device=self.device, dtype=self.weight_dtype)
                for cond_img_index in range(num_conds):
                    attn_selection = attn_prob[:, :, :, attn_size*(cond_img_index):attn_size*(cond_img_index + 1)]
                    means[cond_img_index] = attn_selection.sum()
                means = means/means.max()
                neg_reg_softmax = 1 - softmax(means.unsqueeze(0))
                neg_reg_log_softmax = torch.log(neg_reg_softmax)
                total_neg_reg_loss = nll_loss(neg_reg_log_softmax, gt_attn_neg_reg_idx)
                lambda_neg_reg = self.cfg.optim.lambda_neg_reg
                loss += lambda_neg_reg * total_neg_reg_loss
                loss_dict['loss_attn_neg_reg'] = total_neg_reg_loss.item()
            
        if self.cfg.optim.lambda_facial_comp > 0 and gt_facial_comps is not None:
            leye_mask, reye_mask, mouth_mask = gt_facial_comps

            # Left eye
            loss_leye_l2 = F.mse_loss(pred.float() * leye_mask, gts.float() * leye_mask, reduction="mean")
            loss_reye_l2 = F.mse_loss(pred.float() * reye_mask, gts.float() * reye_mask, reduction="mean")
            loss_mouth_l2 = F.mse_loss(pred.float() * mouth_mask, gts.float() * mouth_mask, reduction="mean")
            total_fc_l2 = loss_leye_l2 + loss_reye_l2 + loss_mouth_l2
            loss_dict['loss_facial_comp_l2'] = total_fc_l2.item()

            loss_leye_lpips = self.net_lpips(pred.float() * leye_mask, gts.float() * leye_mask).mean()
            loss_reye_lpips = self.net_lpips(pred.float() * reye_mask, gts.float() * reye_mask).mean()
            loss_mouth_lpips = self.net_lpips(pred.float() * mouth_mask, gts.float() * mouth_mask).mean()
            total_fc_lpips = loss_leye_lpips + loss_reye_lpips + loss_mouth_lpips
            loss_dict['loss_facial_comp_lpips'] = total_fc_lpips.item()

            total_fc_loss = total_fc_l2 * self.cfg.optim.lambda_l2 + total_fc_lpips * self.cfg.optim.lambda_lpips
            loss += self.cfg.optim.lambda_facial_comp * total_fc_loss

        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def create_exp_dir(self):
        if self.cfg.log.exp_dir.exists():
            if self.cfg.log.allow_overwrite:
                print(f"Note... {self.cfg.log.exp_dir} already exists")
            else:
                raise Exception(f"Ooops... {self.cfg.log.exp_dir} already exists. "
                                f"Use `--log.allow_overwrite True` if you really want to overwrite.")
        else:
            self.cfg.log.exp_dir.mkdir(exist_ok=True, parents=True)

    def get_save_dict(self) -> Dict[str, Any]:
        save_dict = {
            "state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "cfg": pyrallis.encode(self.cfg),
            "optimizer": self.optimizer.state_dict(),
        }
        return save_dict
