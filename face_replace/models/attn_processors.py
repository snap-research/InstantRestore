import torch
from torch import nn
import torch.nn.functional as F

from face_replace.configs.train_config import ModelConfig

def adain(content_features, style_mean, style_std):
    # Compute mean and standard deviation of content features
    content_mean = content_features.mean(dim=(1), keepdim=True)
    content_std = content_features.std(dim=(1), keepdim=True) + 1e-5

    # Normalize content features
    normalized_content = (content_features - content_mean) / content_std

    # Scale and shift content features with style statistics
    stylized_features = normalized_content * style_std + style_mean

    return stylized_features



class AttnProcessor(nn.Module):
    r""" Default processor for performing attention-related computations. """

    def __init__(self):
        super().__init__()
        self.keys, self.values = None, None
        self.is_self_attn = None

    def reset(self):
        self.keys, self.values = None, None
        self.is_self_attn = None

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        self.is_self_attn = encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Store the keys and values to be used later for shared attention
        self.keys, self.values = key, value

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FaceIDAttnProcessor(nn.Module):

    def __init__(self, hidden_size, self_attn_idx=None, cross_attention_dim=None, embed_dim: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.face_projection = nn.Linear(embed_dim, cross_attention_dim or hidden_size)
        self.to_k_face_embed = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_face_embed = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.self_attn_idx = self_attn_idx
        self.keys, self.values = None, None
        self.is_self_attn = None

    def reset(self):
        self.keys, self.values = None, None
        self.is_self_attn = None

    def forward(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            ref_keys=None,  # These are ignored here and only used in the self-attention layers
            ref_values=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        self.is_self_attn = encoder_hidden_states is None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        encoder_hidden_states = self.face_projection(encoder_hidden_states)
        key = self.to_k_face_embed(encoder_hidden_states)
        value = self.to_v_face_embed(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SharedAttnProcessor(nn.Module):
    r""" Attention processor for shared attention computation """

    def __init__(self, self_attn_idx: int = None, save_self_attentions: bool = False, use_adain: bool = False, train_input: bool = True):
        super().__init__()
        self.self_attn_idx = self_attn_idx
        self.save_self_attentions = save_self_attentions
        self.use_adain = use_adain
        self.train_input = train_input

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        ref_keys=None,   # We'll rely on this to pass the keys and values
        ref_values=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if self.self_attn_idx is not None and ref_keys is not None and ref_values is not None:
            # Get the corresponding keys and values from the reference images
            ref_key = ref_keys[self.self_attn_idx]
            ref_value = ref_values[self.self_attn_idx]
            ref_key_reshaped = [attn.head_to_batch_dim(ref_key[:, idx]) for idx in range(ref_key.shape[1])]
            ref_value_reshaped = [attn.head_to_batch_dim(ref_value[:, idx]) for idx in range(ref_value.shape[1])]
            if self.use_adain:
                    # Compute mean and standard deviation of style features
                style_mean = value.mean(dim=(1), keepdim=True)
                style_std = value.std(dim=(1), keepdim=True) + 1e-5
                ref_value_reshaped = [adain(ref_value_reshaped[idx], style_mean, style_std) for idx in range(len(ref_value_reshaped))]
            if self.train_input:
                extended_keys = torch.cat([key] + ref_key_reshaped, dim=1)
                extended_values = torch.cat([value] + ref_value_reshaped, dim=1)
            else:
                extended_keys = torch.cat(ref_key_reshaped, dim=1)
                extended_values = torch.cat(ref_value_reshaped, dim=1)
        else:
            extended_keys = key
            extended_values = value

        attention_probs = attn.get_attention_scores(query, extended_keys, attention_mask)
        if self.save_self_attentions:
            # Convert the attention_probas back into batch dimension
            self.attention_probs = attention_probs.reshape(batch_size, attn.heads, query.shape[1], extended_keys.shape[1])
            # self.attention_probs = self.attention_probs.float().cpu().detach()
        
        hidden_states = torch.bmm(attention_probs, extended_values)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_attention_processor(unet, cfg: ModelConfig, save_self_attentions: bool = False):
    attn_procs = {}
    self_attn_idx = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is not None:
            if cfg.condition_on_face_embeds:
                attn_procs[name] = FaceIDAttnProcessor(
                    self_attn_idx=None, hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, embed_dim=512,
                ).to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = SharedAttnProcessor(self_attn_idx=None, use_adain=cfg.use_adain, train_input=cfg.train_input).to(unet.device, dtype=unet.dtype)

        elif name.startswith("up_blocks") and 'attn1' in name:
            attn_procs[name] = SharedAttnProcessor(
                self_attn_idx=self_attn_idx,
                save_self_attentions=save_self_attentions,
                use_adain=cfg.use_adain,
                train_input=cfg.train_input
            ).to(unet.device, dtype=unet.dtype)
            self_attn_idx += 1

        else:
            attn_procs[name] = SharedAttnProcessor(
                self_attn_idx=None, 
                save_self_attentions=save_self_attentions,
                use_adain=cfg.use_adain,
                train_input=cfg.train_input
            ).to(unet.device, dtype=unet.dtype)

    unet.set_attn_processor(attn_procs)


def register_attention_processor_kv_unet(unet):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if name.startswith("up_blocks") and 'attn1' in name:
            attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
        else:
            attn_procs[name] = unet.attn_processors[name]
    unet.set_attn_processor(attn_procs)
