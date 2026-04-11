
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
################################################################################
# Hybrid Tensor Train Block Modules
################################################################################
from tltorch import FactorizedLinear


def compute_out_tensorized_features(out_features, fixed_dim):
    target = out_features // fixed_dim
    factors = []
    for i in range(1, int(math.sqrt(target)) + 1):
        if target % i == 0:
            factors.append((i, target // i))
    # Choose the pair with minimal difference for better balance
    a, b = min(factors, key=lambda x: abs(x[0] - x[1]))
    return (fixed_dim, a, b)

# --------- HTT Self-Attention --------- #
class HypoAttention(nn.Module):
    def __init__(self, dim, num_heads=8, beta=0.5, tt_rank=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.beta = beta
        dense_dim = int(3 * beta * dim)
        tt_dim = 3 * dim - dense_dim

        self.w_dense = nn.Linear(dim, dense_dim)
        # self.w_tt = TTLinear(dim, tt_dim, tt_rank)
        self.w_tt = FactorizedLinear(in_tensorized_features=compute_out_tensorized_features(dim,8),
                                       out_tensorized_features=compute_out_tensorized_features(tt_dim,8),
                                       factorization='blocktt', rank=0.5)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        x_dense = self.w_dense(x)
        x_tt = self.w_tt(x)

        q1, k1, v1 = x_dense.chunk(3, dim=-1)
        q2, k2, v2 = x_tt.chunk(3, dim=-1)

        q = torch.cat([q1, q2], dim=-1)
        k = torch.cat([k1, k2], dim=-1)
        v = torch.cat([v1, v2], dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.out_proj(out)


# --------- LMF Feed-Forward Network --------- #


class LMFFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, rank):
        super().__init__()
        self.u1 = nn.Linear(dim, rank, bias=False)
        self.v1 = nn.Linear(rank, hidden_dim, bias=False)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.u2 = nn.Linear(hidden_dim, rank, bias=False)
        self.v2 = nn.Linear(rank, dim, bias=False)
        self.b2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.relu(self.v1(self.u1(x)) + self.b1)
        x = self.v2(self.u2(x)) + self.b2
        return x


# --------- Hypoformer Block --------- #
class HypoformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, beta=0.5, tt_rank=3, mf_rank=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HypoAttention(dim, num_heads, beta, tt_rank)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LMFFeedForward(dim, int(dim * mlp_ratio), mf_rank)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

