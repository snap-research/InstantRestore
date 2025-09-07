from torch import nn

from face_replace.configs.train_config import ModelConfig, TrainConfig
from face_replace.models.pix2pix_turbo import Pix2Pix_Turbo
import torch


class FaceReplaceModel(nn.Module):

    def __init__(self, cfg: ModelConfig, full_cfg: TrainConfig, evaluating: bool = False):
        super().__init__()
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.net = self.set_network()
        if self.cfg.checkpoint_path is not None and not evaluating: 
            print(f"Loading checkpoint from path: {self.cfg.checkpoint_path}")
            checkpoint_dict = torch.load(self.cfg.checkpoint_path, map_location='cpu')
            checkpoint_dict['state_dict'] = {k.replace('net.', '', 1) : v for k, v in checkpoint_dict['state_dict'].items()} 
            checkpoint_dict['state_dict'] = {k.replace('module.', '', 1) : v for k, v in checkpoint_dict['state_dict'].items()} 
            # print(checkpoint_dict['state_dict'].keys())
            out = self.net.load_state_dict(checkpoint_dict['state_dict'], strict=True)
            print(out)
            checkpoint_dict = None
            print("Emptying cache")
            torch.cuda.empty_cache()

    def set_network(self):
        if self.cfg.net_type == 'pix2pix_turbo':
            net = Pix2Pix_Turbo(lora_rank_unet=self.cfg.lora_rank_unet,
                                lora_rank_vae=self.cfg.lora_rank_vae,
                                condition_on_face_embeds=self.cfg.condition_on_face_embeds,
                                concat_mask_and_landmarks=self.cfg.concat_mask_and_landmarks,
                                save_self_attentions=self.full_cfg.log.vis_attention or self.full_cfg.optim.lambda_attn_reg > 0 or 
                                    self.full_cfg.optim.lambda_landmark > 0 or self.full_cfg.optim.lambda_pos_reg > 0 or self.full_cfg.optim.lambda_neg_reg > 0,
                                train_reference_networks=self.cfg.train_reference_networks,
                                cfg=self.cfg)
            net.set_train()
        else:
            raise ValueError(f"Invalid encoder type: {self.cfg.net_type}")
        return net

    def forward(self, x):
        return self.encoder(x)
