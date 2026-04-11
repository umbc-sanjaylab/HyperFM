import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# Decoder Layers
class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        # GroupNorm(1, C) is a nice LayerNorm-like choice for 2D maps
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class UpBlock(nn.Module):
    """Upsample by 2x (bilinear) + conv refinement."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = ConvNormAct(in_ch, out_ch, 3, 1, 1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.refine(x)
        return x

class ViTRegressionDecoder(nn.Module):
    """
    Turn ViT token embeddings into full-res predictions.
    - Drops CLS token if present.
    - Reshapes tokens -> patch grid -> upsample to HxW.
    - Total upsampling is patch_size (e.g., 16 for ViT-B/16).
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,     # 768 for ViT-B/16
        out_ch: int,        # controls decoder output channels
        mid_ch: int = 128,  # width of the decoder
        num_refine: int = 1,# extra 3x3 conv blocks at full res
        has_cls_token: bool = True,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.has_cls_token = has_cls_token

        # Project ViT embed_dim -> decoder width
        self.proj = nn.Linear(embed_dim, mid_ch)

        # We'll upsample from (H/ps, W/ps) to (H, W) by factors of 2
        total_scale = patch_size
        num_x2 = int(math.floor(math.log2(total_scale)))
        remainder = total_scale / (2 ** num_x2)

        blocks = []
        in_ch = mid_ch
        for _ in range(num_x2):
            blocks.append(UpBlock(in_ch, in_ch))  # keep width constant
        self.ups = nn.Sequential(*blocks)

        # Handle any non power-of-two remainder cleanly (rare; e.g., ps=12)
        self.remainder_scale = remainder if remainder != 1.0 else None
        if self.remainder_scale is not None:
            assert self.remainder_scale > 1.0, "Unexpected remainder scale"
            self.remainder_refine = ConvNormAct(in_ch, in_ch, 3, 1, 1)

        # Optional refinement at full res
        self.refine = nn.Sequential(
            *[ConvNormAct(in_ch, in_ch, 3, 1, 1) for _ in range(num_refine)]
        )

        # Final prediction head
        self.head = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, H=None, W=None):
        """
        x: (B, N, D) ViT tokens (w/ or w/o CLS).
        H, W: full image size if you need dynamic sizes (else uses constructor).
        """
        B, N, D = x.shape
        img_h = self.img_size if H is None else H
        img_w = self.img_size if W is None else W
        assert img_h % self.patch_size == 0 and img_w % self.patch_size == 0

        grid_h, grid_w = img_h // self.patch_size, img_w // self.patch_size
        expected_tokens = grid_h * grid_w + (1 if self.has_cls_token else 0)
        if N != expected_tokens:
            raise ValueError(
                f"Token count mismatch: got N={N}, expected {expected_tokens} "
                f"(grid {grid_h}x{grid_w}, has_cls={self.has_cls_token})."
            )

        # Drop CLS if present
        if self.has_cls_token:
            x = x[:, 1:, :]  # (B, N-1, D)

        # Project and reshape to (B, C, grid_h, grid_w)
        x = self.proj(x)                     # (B, N_grid, mid_ch)
        x = x.transpose(1, 2)                # (B, mid_ch, N_grid)
        x = x.view(B, -1, grid_h, grid_w)    # (B, mid_ch, gh, gw)

        # Upsample by powers of 2
        if len(self.ups) > 0:
            x = self.ups(x)

        # Remainder upscale if needed (non power-of-two patch size)
        if self.remainder_scale is not None:
            x = F.interpolate(
                x,
                scale_factor=self.remainder_scale,
                mode='bilinear',
                align_corners=False
            )
            x = self.remainder_refine(x)

        # Full-res refinement and prediction
        x = self.refine(x)
        x = self.head(x)                     # (B, out_ch, H, W)
        return x

class GroupViTRegressionDecoder(nn.Module):
    """
    Turn ViT token embeddings into full-res predictions.
    - Drops CLS token if present.
    - Reshapes tokens -> patch grid -> upsample to HxW.
    - Total upsampling is patch_size (e.g., 16 for ViT-B/16).
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_groups: int,
        embed_dim: int,     # 768 for ViT-B/16
        out_ch: int,        # controls decoder output channels
        mid_ch: int = 128,  # width of the decoder
        num_refine: int = 1,# extra 3x3 conv blocks at full res
        has_cls_token: bool = True,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_groups = num_groups
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.has_cls_token = has_cls_token

        # Project ViT embed_dim -> decoder width
        self.proj = nn.Linear(embed_dim*num_groups, mid_ch)

        # We'll upsample from (H/ps, W/ps) to (H, W) by factors of 2
        total_scale = patch_size
        num_x2 = int(math.floor(math.log2(total_scale)))
        remainder = total_scale / (2 ** num_x2)

        blocks = []
        in_ch = mid_ch
        for _ in range(num_x2):
            blocks.append(UpBlock(in_ch, in_ch))  # keep width constant
        self.ups = nn.Sequential(*blocks)

        # Handle any non power-of-two remainder cleanly (rare; e.g., ps=12)
        self.remainder_scale = remainder if remainder != 1.0 else None
        if self.remainder_scale is not None:
            assert self.remainder_scale > 1.0, "Unexpected remainder scale"
            self.remainder_refine = ConvNormAct(in_ch, in_ch, 3, 1, 1)

        # Optional refinement at full res
        self.refine = nn.Sequential(
            *[ConvNormAct(in_ch, in_ch, 3, 1, 1) for _ in range(num_refine)]
        )

        # Final prediction head
        self.head = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, H=None, W=None):
        """
        x: (B, N, D) ViT tokens (w/ or w/o CLS).
        H, W: full image size if you need dynamic sizes (else uses constructor).
        """
        # Drop CLS if present
        if self.has_cls_token:
            x = x[:, 1:, :]  # (B, N-1, D)

        B, N, D = x.shape
        img_h = self.img_size if H is None else H
        img_w = self.img_size if W is None else W
        assert img_h % self.patch_size == 0 and img_w % self.patch_size == 0

        grid_h, grid_w = img_h // self.patch_size, img_w // self.patch_size
        tokens_per_group = (grid_h * grid_w)
        assert N == tokens_per_group * self.num_groups
       
        x = x.view(B, self.num_groups, tokens_per_group, D)
        x = x.permute(0, 2, 1, 3).reshape(B, tokens_per_group, D * self.num_groups)
        
        # Project and reshape to (B, C, grid_h, grid_w)
        x = self.proj(x)                     # (B, N_grid, mid_ch)
        x = x.transpose(1, 2)                # (B, mid_ch, N_grid)
        x = x.view(B, -1, grid_h, grid_w)    # (B, mid_ch, gh, gw)

        # Upsample by powers of 2
        if len(self.ups) > 0:
            x = self.ups(x)

        # Remainder upscale if needed (non power-of-two patch size)
        if self.remainder_scale is not None:
            x = F.interpolate(
                x,
                scale_factor=self.remainder_scale,
                mode='bilinear',
                align_corners=False
            )
            x = self.remainder_refine(x)

        # Full-res refinement and prediction
        x = self.refine(x)
        x = self.head(x)                     # (B, out_ch, H, W)
        return x
