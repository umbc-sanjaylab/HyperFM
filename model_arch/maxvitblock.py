'''
PyTorch Implementation of Maxvitblock
'''
import torch
import torch.nn as nn
from einops import rearrange

# Feed-forward MLP
class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Multi-head self-attention on spatial tokens
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).contiguous()
        out = rearrange(out, "b h n d -> b n (h d)").contiguous()
        return self.proj(out)

# Attention applied on local windows
def block_attention(x, attn, window_size):
    B, C, H, W = x.shape
    ws = window_size
    x = rearrange(x, 'b c (nh ws1) (nw ws2) -> (b nh nw) (ws1 ws2) c', ws1=ws, ws2=ws)
    x = attn(x)
    x = rearrange(x, '(b nh nw) (ws1 ws2) c -> b c (nh ws1) (nw ws2)',
                  b=B, nh=H // ws, nw=W // ws, ws1=ws, ws2=ws)
    return x

# Attention applied on global grid
def grid_attention(x, attn, grid_size):
    B, C, H, W = x.shape
    gs = grid_size
    # Split into grid partitions: (B * gs * gs, (H/gs * W/gs), C)
    x = rearrange(x, 'b c (gh gs1) (gw gs2) -> (b gs1 gs2) (gh gw) c', gh=H // gs, gw=W // gs)
    x = attn(x)
    # Reverse reshape
    x = rearrange(x, '(b gs1 gs2) (gh gw) c -> b c (gh gs1) (gw gs2)',
                  b=B, gh=H // gs, gw=W // gs, gs1=gs, gs2=gs)
    return x

# --- Squeeze-and-Excitation (SE) block ---
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(1, dim // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


# --- MBConv block ---
class MBConv(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0., reduction=4):
        super().__init__()
        hidden_dim = int(dim * expansion)

        self.conv = nn.Sequential(
            # 1x1 expand
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            # depthwise 3x3
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            # extra 1x1 conv after depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            # SE block
            SEBlock(hidden_dim, reduction=reduction),

            # project back
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.conv(x))


# Full MaxViT block
class MaxViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7, dropout=0.):
        super().__init__()
        self.mbconv = MBConv(dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.block_attn = MultiHeadSelfAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.grid_attn = MultiHeadSelfAttention(dim, num_heads)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

        self.window_size = window_size

    def forward(self, x):
        # Step 1: MBConv
        x = self.mbconv(x)

        # Step 2: Block attention (local)
        residual = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm1(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = block_attention(x, self.block_attn, self.window_size)
        x = x + residual

        # Step 3: Grid attention (global)
        residual = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm2(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = grid_attention(x, self.grid_attn, self.window_size)
        x = x + residual

        # Step 4: FFN
        residual = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm3(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.ffn(x)
        x = x + residual

        return x


if __name__ == "__main__":
    x = torch.randn(2, 35, 96, 96)
    block = MaxViTBlock(dim=35, num_heads=5, window_size=8)
    y = block(x)
    print(y.shape)  # [2, 64, 56, 56]
