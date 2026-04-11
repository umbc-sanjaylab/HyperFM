# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
import math
from model_arch.maxvitblock import MaxViTBlock
from model_arch.HypoformerBlock import HypoformerBlock
from model_arch.model_decoder_conv import GroupViTRegressionDecoder

################################################################################

class HyperFMConv(nn.Module):
    """ HyperFM backbone with Conv Decoder for regression tasks.
    Args:
        img_size (int): Input image size. Default: 96
        patch_size (int): Patch size. Default: 8
        in_chans (int): Number of input image channels. Default: 291
        channel_groups (tuple): Tuple of channel groups. 
        beta (float): HyperFM compression ratio parameter. Default: 0.5
        tt_rank (int): HyperFM TT rank. Default: 3
        mf_rank (int): HyperFM MF rank. Default: 128
        decoder_ch (int): Number of output channels for decoder. Default: 1
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=291,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
                 decoder_ch = 1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 beta=0.5, tt_rank=3, mf_rank=128, proj_ratio=4, gating=False, selected_groups=3):
        super().__init__()

        self.in_c = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.channel_groups = channel_groups
        tt_ranks = [1,tt_rank,tt_rank,1]
        patch_shape = (patch_size,patch_size,in_chans)
        num_groups = len(channel_groups)
        channels_per_group = in_chans//num_groups

        ######################################################
        # MAE Pre-embedding specifics
        self.group_band_attn = nn.ModuleList(
            [
                # Regular groups
                *[
                    MaxViTBlock(
                        dim=channels_per_group,num_heads=8,window_size=8
                    )
                    for _ in channel_groups[:-1]
                ],

                # Last (larger) group
                MaxViTBlock(
                    dim=channels_per_group + in_chans % num_groups,
                    num_heads=7,
                    window_size=8
                )
            ]
        )

        if gating:
            self.gating_func = TopKGroupRouter(num_groups, in_channels_list, k=selected_groups)
            self.global_band_attn = MaxViTBlock(dim=selected_groups*channels_per_group, num_heads=1, window_size=8, dropout=0.)
        else:
            self.global_band_attn = MaxViTBlock(dim=in_chans, num_heads=1, window_size=8)

        ######################################################
        # Encoder specifics
        self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                          for group in channel_groups])
        num_patches = self.patch_embed[0].num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            HypoformerBlock(embed_dim, num_heads, mlp_ratio, beta, tt_rank, mf_rank)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        ######################################################
        # Decoder specifics
        self.decoder = GroupViTRegressionDecoder(
            img_size=img_size,
            patch_size=patch_size,
            num_groups=num_groups,
            embed_dim=embed_dim,
            out_ch=decoder_ch
        )

        self.decoder.apply(self._init_weights) # initialize decoder weights only

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # same style as JAX ViT: Xavier uniform
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            # Kaiming normal works well for conv layers with GELU/ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # x is (N, C, H, W)
        b, c, h, w = x.shape

        x_c_gba = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_gba.append(self.group_band_attn[i](x_c))  # (N, L, H, W)

        x = torch.cat(x_c_gba, dim=1)  # (N, C, H, W)

        # pass Thorugh global band attn
        x = self.global_band_attn(x)

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)

        # change shape to N, G*L, D
        x = x.view(-1,G*L, D)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, G*L + 1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self,imgs):
        latent = self.forward_encoder(imgs)
        pred = self.decoder(latent)  
        return pred

    def no_weight_decay(self):
        no_decay = set()

        # Positional embeddings
        if hasattr(self, 'pos_embed'):
            no_decay.add('pos_embed')

        # Class token
        if hasattr(self, 'cls_token'):
            no_decay.add('cls_token')

        # Patch embedding biases
        if hasattr(self, 'patch_embed') and hasattr(self.patch_embed, 'proj'):
            if hasattr(self.patch_embed.proj, 'bias') and self.patch_embed.proj.bias is not None:
                no_decay.add('patch_embed.proj.bias')

        # LayerNorm / normalization layers
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay.add(f'{name}.weight')
                no_decay.add(f'{name}.bias')

        return no_decay

    def freeze_encoder_update_decoder(self):
        """
        Freeze all parameters except the decoder.
        Call this before creating the optimizer to fine-tune decoder only.
        """
        for name, param in self.named_parameters():
            if name.startswith("decoder."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def decoder_parameters(self):
        """Return only the decoder parameters (trainable ones)."""
        return (p for n, p in self.named_parameters() if n.startswith("decoder.") and p.requires_grad)

################################################################################


def hyperFM_conv_small_patch8_decConv(**kwargs):
    model = HyperFMConv(
        embed_dim=768, depth=4, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def hyperFM_conv_base_patch8_decConv(**kwargs):
    model = HyperFMConv(
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def hyperFM_conv_large_patch8_decConv(**kwargs):
    model = HyperFMConv(
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
hyperFM_enc4 = hyperFM_conv_small_patch8_decConv  # encoder: 768 dim, 4 blocks, decoder: 512 dim, 8 blocks
hyperFM_enc12 = hyperFM_conv_base_patch8_decConv  # encoder: 768 dim, 12 blocks, decoder: 512 dim, 8 blocks
hyperFM_enc24 = hyperFM_conv_large_patch8_decConv  # encoder: 1024 dim, 24 blocks, decoder: 512 dim, 8 blocks