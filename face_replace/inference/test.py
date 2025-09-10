import logging
from enum import Enum, auto
from pathlib import Path
from typing import List, Any, Optional

import pyrallis
import torch
from PIL import Image
from torch import Tensor
import numpy as np
from torchvision import transforms
from insightface.app import FaceAnalysis

import sys
sys.path.append(".")
sys.path.append("..")

from face_replace.configs.train_config import TrainConfig
from face_replace.models.face_replace_model import FaceReplaceModel
from face_replace.training.utils.vis_utils import tensor2im
from face_replace.models.attn_processors import SharedAttnProcessor

import time
import pickle

import warnings
import glob
from natsort import natsorted
warnings.simplefilter(action='ignore', category=FutureWarning)


class GPUMode(Enum):
    SINGLE = auto()
    MULTI = auto()
    CPU = auto()


class Predictor:
    logging.basicConfig(level=logging.INFO)

    def __init__(self, checkpoint_path: Path):
        checkpoint_dict = torch.load(checkpoint_path)
        self.cfg = pyrallis.decode(TrainConfig, checkpoint_dict['cfg'])

        self.face_replace_model = FaceReplaceModel(cfg=self.cfg.model, full_cfg=self.cfg)
        try:
            out = self.face_replace_model.load_state_dict(checkpoint_dict['state_dict'], strict=True)
        except:
            checkpoint_dict['state_dict'] = {k.replace('.module.', '.') : v for k, v in checkpoint_dict['state_dict'].items()} 
            out = self.face_replace_model.load_state_dict(checkpoint_dict['state_dict'], strict=True)

        self.face_replace_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.max_conditioning_images = self.cfg.data.max_conditioning_images
        self.face_replace_model.net.noise_timesteps = [249]
        self.dtype = torch.float16

        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._set_gpu()

    def _set_gpu(self):
        logging.info("Moving model to GPU")
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda')
        self.face_replace_model = self.face_replace_model.to(self.device)
    
    def _apply_transforms_on_image_list(self, images: List[Image.Image]) -> List[Tensor]:
        images = [self.transform(image) for image in images]
        return images

    def _forward_batch(self, input_images: Tensor, conditioning_images: Tensor, face_embeds: Tensor = None, calc_attn_probs: bool = False) -> Tensor:
        # Everything is valid since we pad the conditioning images if we have less than the max
        valid_indices = torch.ones(input_images.size(0), dtype=int) * self.max_conditioning_images
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if len(self.face_replace_model.net.noise_timesteps) > 1:
                    x_pred, _ = self.face_replace_model.net.multi_step_forward(
                        input_images.to(self.device, self.dtype),
                        conditioning_images=conditioning_images.to(self.device, self.dtype),
                        valid_indices=valid_indices,
                        face_embeds=face_embeds.to(self.device, self.dtype) if self.cfg.model.condition_on_face_embeds else None
                    )
                else:
                    # start_time = time.time()
                    if calc_attn_probs:
                        self_attn_processors = [p for p in self.face_replace_model.net.unet.attn_processors.values() 
                                                            if type(p) == SharedAttnProcessor and p.self_attn_idx is not None]
                        for self_attn_processor in self_attn_processors:
                            self_attn_processor.save_self_attentions = True
                    # print("TIME")
                    # start_time = time.time()
                    x_pred, x_conds, shared_attn_maps = self.face_replace_model.net.forward(
                        input_images.to(self.device, self.dtype),
                        conditioning_images=conditioning_images.to(self.device, self.dtype),
                        valid_indices=valid_indices,
                        face_embeds=face_embeds.to(self.device, self.dtype) if self.cfg.model.condition_on_face_embeds else None
                    )
                    # print(time.time() - start_time)
                    if calc_attn_probs:
                        attn_probs = [p.attention_probs.float().cpu().detach() for p in self_attn_processors]
        if calc_attn_probs:
            return x_pred, attn_probs
        return x_pred, None

    def prepare_conditioning_images(self, cond_imgs: List[Image.Image]) -> Tensor:
        faceid_embeds = None
        if self.cfg.model.condition_on_face_embeds: 
            # Compute face embeddings on all conditioning images
            faceid_embeds = []
            for image in cond_imgs:
                faces = self.app.get(np.array(image))
                try:
                    embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0)
                except: 
                    embed = torch.zeros((1, 1, 512))
                faceid_embeds.append(embed)
            faceid_embeds = torch.cat(faceid_embeds, dim=1)
            
        cond_imgs = self._apply_transforms_on_image_list(cond_imgs)

        cond_imgs = torch.stack(cond_imgs, dim=0)
        if self.cfg.model.condition_on_face_embeds:
            faceid_embeds = torch.stack(faceid_embeds, dim=0).squeeze(1)

        return cond_imgs, None, faceid_embeds

    def parse_results(self, outputs: Tensor, 
                      input_img: Image.Image, 
                      target_img: Optional[Image.Image] = None) -> List[Any]:
        # Get the output images
        pred_images = [tensor2im(out, unnorm=True) for out in outputs][0]
        
        # Join all the images together
        images_to_join = [input_img]
        images_to_join.append(pred_images)
        if target_img is not None: 
            images_to_join.append(target_img)
        images = Image.fromarray(np.concatenate(images_to_join, axis=0))
        return pred_images, images

    def predict(self, input_img: Image.Image, cond_imgs: Optional[List[Image.Image]] = None, target_img: Optional[Image.Image] = None, calc_attn_probs: bool = False) -> Any:
        input_t = self.transform(input_img)
        input_t = input_t.unsqueeze(0)
        conds_t, _, face_embeds_t = self.prepare_conditioning_images(cond_imgs)
        conds_t = conds_t.unsqueeze(0)
        if self.cfg.model.condition_on_face_embeds:
            face_embeds_t = face_embeds_t.unsqueeze(0)
        outputs, attn_probs = self._forward_batch(input_t, conditioning_images=conds_t, face_embeds=face_embeds_t, calc_attn_probs=calc_attn_probs)

        pred_images, visualization_results = self.parse_results(
            outputs, 
            input_img=input_img, 
            target_img=target_img
        )
        return pred_images, visualization_results, attn_probs

if __name__ == '__main__':
    calc_attn_probs = False
    checkpoint_path = 'path/to/checkpoint/ckpt.pt'
    predictor = Predictor(checkpoint_path=checkpoint_path)

    results_dir = Path(f'path/to/resultsdir')
    results_dir.mkdir(exist_ok=True)
    data_root = Path('path/to/data')
    identities = sorted(data_root.glob(f'*'))
    for identity in identities:
        # input_path = identity / 'degraded.png'
        input_path = identity / 'degraded.png'
        input_img = Image.open(input_path).convert('RGB')
        cond_paths = natsorted([p for p in (identity / 'conditioning').glob('*.png')])[:4]
        cond_imgs = [Image.open(p).convert('RGB') for p in cond_paths]

        target_path = identity / 'gt.png'
        target_img = Image.open(target_path).convert('RGB')
        pred_images, visualization_results, attn_probs = predictor.predict(input_img, 
                                                                cond_imgs = cond_imgs,
                                                                target_img = target_img,
                                                                calc_attn_probs=calc_attn_probs)
        pred_images.save(results_dir / f'{identity.name}.png')