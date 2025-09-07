import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.utils import is_xformers_available
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from face_replace.configs.train_config import ModelConfig
from face_replace.models.attn_processors import SharedAttnProcessor, register_attention_processor, register_attention_processor_kv_unet, \
    AttnProcessor
from face_replace.models.model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd
from face_replace.models.unet_2d_condition.unet import UNet2DConditionModel

import time


MODEL_NAME = 'stabilityai/sd-turbo'


class Pix2Pix_Turbo(torch.nn.Module):

    def __init__(self,
                 pretrained_name: str = None,
                 pretrained_path: str = None,
                 lora_rank_unet: int = 8,
                 lora_rank_vae: int = 4,
                 condition_on_face_embeds: bool = False,
                 concat_mask_and_landmarks: bool = False,
                 save_self_attentions: bool = False,
                 train_reference_networks: bool = False,
                 cfg: ModelConfig = None):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
        self.sched = make_1step_sched(model_name=MODEL_NAME)

        self.train_vae = self.cfg.train_vae
        self.train_only_vae_encoder = self.cfg.train_only_vae_encoder

        # vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        # add the skip connection convs
        if self.cfg.use_shortcuts:
            vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.ignore_skip = False
        else:
            vae.decoder.ignore_skip = True

        unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

        original_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

        original_unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

        b1 = 1.4
        b2 = 1.6
        s1 = 0.9
        s2 = 0.2

        unet.enable_freeu(s1, s2, b1, b2)
        original_unet.enable_freeu(s1, s2, b1, b2)

        unet.to("cuda")
        vae.to("cuda")

        original_unet.to("cuda")
        original_vae.to("cuda")

        self.unet, self.vae, self.original_unet, self.original_vae = unet, vae, original_unet, original_vae

        self.train_reference_networks = train_reference_networks

        # We can use xformers here
        if is_xformers_available():
            self.original_unet.enable_xformers_memory_efficient_attention()

        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()

        self._init_models(pretrained_path=pretrained_path,
                          pretrained_name=pretrained_name,
                          lora_rank_vae=lora_rank_vae,
                          lora_rank_unet=lora_rank_unet)

        self.condition_on_face_embeds = condition_on_face_embeds

        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

        register_attention_processor_kv_unet(self.original_unet)
        register_attention_processor(self.unet, cfg=cfg, save_self_attentions=save_self_attentions)

        prompt = "A high-quality photo of a person; professional, 8k"
        caption_tokens = self.tokenizer(prompt,
                                        max_length=self.tokenizer.model_max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt").input_ids.cuda()
        self.caption_enc = self.text_encoder(caption_tokens)[0]
        self.noise_timesteps = [249, 499, 749]

    def _init_models(self,
                     pretrained_path: str = None,
                     pretrained_name: str = None,
                     lora_rank_vae: int = 4,
                     lora_rank_unet: int = 8):
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"],
                                          init_lora_weights="gaussian",
                                          target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"],
                                         init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = self.vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            self.vae.load_state_dict(_sd_vae)
            self.unet.add_adapter(unet_lora_config)
            _sd_unet = self.unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_unet)
            # Load the weights for the original unet and vae
            _sd_original_unet = self.original_unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_original_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_original_unet)
            _sd_original_vae = self.original_vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_original_vae[k] = sd["state_dict_vae"][k]
            self.original_vae.load_state_dict(_sd_original_vae)

        elif pretrained_name is None and pretrained_path is None:
            if self.cfg.use_shortcuts:
                print("Initializing model with random weights")
                torch.nn.init.constant_(self.vae.decoder.skip_conv_1.weight, 1e-5)
                torch.nn.init.constant_(self.vae.decoder.skip_conv_2.weight, 1e-5)
                torch.nn.init.constant_(self.vae.decoder.skip_conv_3.weight, 1e-5)
                torch.nn.init.constant_(self.vae.decoder.skip_conv_4.weight, 1e-5)
            
            if self.train_vae:
                target_modules_vae = [
                    "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                    "to_k", "to_q", "to_v", "to_out.0",
                ]
                if self.cfg.use_shortcuts:
                    target_modules_vae.extend(["skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4"])

                vae_lora_config = LoraConfig(r=lora_rank_vae,
                                            lora_alpha=lora_rank_vae // 2,
                                            init_lora_weights="gaussian",
                                            target_modules=target_modules_vae)
                self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                self.target_modules_vae = target_modules_vae
                if self.train_reference_networks:
                    original_vae_lora_config = LoraConfig(r=16,
                                                          lora_alpha=8,
                                                          init_lora_weights="gaussian",
                                                          target_modules=target_modules_vae)
                    self.original_vae.add_adapter(original_vae_lora_config, adapter_name="vae_skip")

            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet,
                                          lora_alpha=lora_rank_unet // 2,
                                          init_lora_weights="gaussian",
                                          target_modules=target_modules_unet)
            self.unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_unet = target_modules_unet
            if self.train_reference_networks:
                original_unet_lora_config = LoraConfig(r=16,
                                                       lora_alpha=8,
                                                       init_lora_weights="gaussian",
                                                       target_modules=target_modules_unet)
                self.original_unet.add_adapter(original_unet_lora_config)

    def set_eval(self):
        self.unet.eval()
        self.original_unet.eval()
        self.vae.eval()
        self.original_vae.eval()
        self.unet.requires_grad_(False)
        self.original_unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.original_vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        if self.train_vae:
            self.vae.train()
            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.vae.eval()
            for n, _p in self.vae.named_parameters():
                _p.requires_grad = False

        # Set the new cross-attention layers to trainable
        if self.condition_on_face_embeds:
            for n, _p in self.unet.named_parameters():
                if "_face_embed" in n or "face_projection" in n:
                    _p.requires_grad = True
        
        # Freeze the weights of the original UNet and VAE
        if self.train_reference_networks:
            self.original_unet.train()
            for n, _p in self.original_unet.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            self.original_unet.conv_in.requires_grad_(True)
            self.original_vae.train()
            for n, _p in self.original_vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.original_unet.eval()
            for n, _p in self.original_unet.named_parameters():
                _p.requires_grad = False
            self.original_vae.eval()
            for n, _p in self.original_vae.named_parameters():
                _p.requires_grad = False

    def get_conditioning_keys_values(self, conditioning_images, valid_indices):
        # Extract the keys and values from the conditioning images and use this to inject into the unet
        cond = conditioning_images.reshape(-1, 3, 512, 512)
        encoded_condition = self.original_vae.encode(cond).latent_dist.sample() * self.vae.config.scaling_factor

        t = torch.tensor([1], device="cuda")
        noise = torch.randn_like(encoded_condition)
        timesteps = t.long().repeat(encoded_condition.shape[0])
        noisy_encoded_condition = self.sched.add_noise(encoded_condition, noise, timesteps)
        model_input = self.sched.scale_model_input(noisy_encoded_condition, timesteps)

        extended_caption_enc = self.caption_enc.repeat(model_input.shape[0], 1, 1)

        model_pred_condition = self.original_unet(model_input,
                                                  t,
                                                  encoder_hidden_states=extended_caption_enc).sample

        # Get all the keys and values from the forward pass
        self_attn_processors = [p for p in self.original_unet.attn_processors.values() if type(p) in [AttnProcessor]]
        keys = [p.keys for p in self_attn_processors]
        values = [p.values for p in self_attn_processors]

        # Split then back to be back to the batch dimension
        keys_ = [k.reshape(-1, conditioning_images[0].shape[0], k.shape[1], k.shape[2]) for k in keys]
        values_ = [v.reshape(-1, conditioning_images[0].shape[0], v.shape[1], v.shape[2]) for v in values]

        # Using the valid_indices, we can zero out the invalid keys and values so we don't use them
        for k, v in zip(keys_, values_):
            for sample_idx in range(k.shape[0]):
                idx = valid_indices[sample_idx]  # zero out the entries greater than valid_idx
                k[sample_idx, idx:] = 0
                v[sample_idx, idx:] = 0

        for p in self_attn_processors: p.reset()

        x_denoised = self.sched.step(model_pred_condition, t, noisy_encoded_condition, return_dict=True).pred_original_sample
        output_image_conditions = (self.original_vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return keys_, values_, output_image_conditions

    def forward(self, c_t,
                face_embeds=None,
                conditioning_images: torch.Tensor = None,
                valid_indices: torch.Tensor = None,
                mask: torch.Tensor = None,
                return_self_attention_maps: bool = False):

        # print("------------")
        # start_time = time.time()

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # print(f"VAE Encoding: {time.time() - start_time}")
        # start_time = time.time()

        # Extract the keys and values from the conditioning images and use this to inject into the unet
        if conditioning_images is not None and self.cfg.use_shared_attention:
            keys_, values_, output_image_conditions = self.get_conditioning_keys_values(conditioning_images, valid_indices)
        else:
            keys_, values_ = None, None
            output_image_conditions = None

        # print(f"Get Keys and Values: {time.time() - start_time}")
        # start_time = time.time()

        step = np.random.choice(self.noise_timesteps, 1)[0]
        t = torch.tensor([step], device="cuda")
        noise = torch.randn_like(encoded_control)
        timesteps = t.long().repeat(encoded_control.shape[0])
        noisy_encoded_condition = self.sched.add_noise(encoded_control, noise, timesteps)
        model_input = self.sched.scale_model_input(noisy_encoded_condition, timesteps)

        # print(f"Preprocessing: {time.time() - start_time}")
        # start_time = time.time()

        if self.condition_on_face_embeds and face_embeds is not None:
            model_pred = self.unet(model_input,
                                    t,
                                    encoder_hidden_states=face_embeds,
                                    cross_attention_kwargs={'ref_keys': keys_, 'ref_values': values_}).sample
        else:
            extended_caption_enc = self.caption_enc.repeat(model_input.shape[0], 1, 1)
            model_pred = self.unet(model_input,
                                    t,
                                    encoder_hidden_states=extended_caption_enc,
                                    cross_attention_kwargs={'ref_keys': keys_, 'ref_values': values_}).sample
        
        # print(f"UNet: {time.time() - start_time}")
        # start_time = time.time()

        x_denoised = self.sched.step(model_pred, t, noisy_encoded_condition, return_dict=True).pred_original_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        # print(f"Post-processing + VAE Decode: {time.time() - start_time}")
        # start_time = time.time()

        if return_self_attention_maps:
            shared_attn_maps = [p.attention_probs for p in self.unet.attn_processors.values() 
                                if type(p) == SharedAttnProcessor and p.self_attn_idx is not None]
            return output_image, output_image_conditions, shared_attn_maps
        else:
            return output_image, output_image_conditions, None

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae if self.train_vae else None
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)
