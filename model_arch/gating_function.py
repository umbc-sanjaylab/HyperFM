import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utility: distributed all-reduce safe mean
# ----------------------------
def dist_mean(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return tensor


# ----------------------------
# Gating MLP per group
# ----------------------------
class PerGroupGate(nn.Module):
    def __init__(self, num_groups, in_channels_list, hidden=64, use_group_embedding=False, embed_dim=16):
        super().__init__()
        self.num_groups = num_groups
        self.use_group_embedding = use_group_embedding

        self.group_mlps = nn.ModuleList()
        for c in in_channels_list:
            in_dim = c + (embed_dim if use_group_embedding else 0)
            self.group_mlps.append(nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1)  # scalar logit per group
            ))

        if use_group_embedding:
            self.group_embedding = nn.Parameter(torch.randn(num_groups, embed_dim) * 0.02)
        else:
            self.group_embedding = None

    def forward(self, groups):
        B = groups[0].shape[0]
        logits = []
        for i, x in enumerate(groups):
            pooled = x.mean(dim=[2,3])  # (B, C_i)
            if self.use_group_embedding:
                emb = self.group_embedding[i].unsqueeze(0).expand(B, -1)
                inp = torch.cat([pooled, emb], dim=1)
            else:
                inp = pooled
            logit = self.group_mlps[i](inp)  # (B,1)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)  # (B, num_groups)
        return logits

# ----------------------------
# Differentiable Top-K Router
# ----------------------------
class TopKGroupRouter(nn.Module):
    """
    Router options:
      - mode='hard' : return concatenation of top-k groups per sample (deterministic eval)
      - mode='soft' : return concatenation of top-k groups weighted by soft probs (training-friendly)
      - straight_through=True : use hard forward and soft backward (straight-through trick)
    Works efficiently when all group channels are equal. If channels differ, falls back to safe per-sample concat.
    """
    def __init__(self, num_groups, in_channels_list, k=3, temp=1.0,
                 use_group_embedding=False, load_balance_coef=0.01,
                 mode='hard', straight_through=True):
        super().__init__()
        assert mode in ('hard', 'soft'), "mode must be 'hard' or 'soft'"
        self.num_groups = num_groups
        self.in_channels_list = in_channels_list
        self.k = min(k, num_groups)
        self.temp = temp
        self.load_balance_coef = load_balance_coef
        self.mode = mode
        self.straight_through = straight_through

        self.gate = PerGroupGate(num_groups, in_channels_list, use_group_embedding=use_group_embedding)

        # quick flags
        self._uniform_channels = len(set(in_channels_list)) == 1
        self._group_channels = in_channels_list[0] if self._uniform_channels else None

    def forward(self, groups):
        """
        groups : list of length G of tensors shape (B, C_i, H, W)
        returns:
          out : (B, k * C, H, W)  when using concatenation of groups
          aux : dict with gate_logits, gate_mask (hard), gate_probs (soft), load_balance_loss
        """
        B = groups[0].shape[0]
        device = groups[0].device
        H, W = groups[0].shape[2], groups[0].shape[3]

        logits = self.gate(groups)  # (B, G)

        # soft probabilities for gradients / soft routing
        soft_probs = F.softmax(logits / self.temp, dim=1)  # (B, G)

        # hard top-k indices
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=1)  # (B, k)

        # build hard mask (B, G)
        hard_mask = torch.zeros_like(logits, device=device)
        hard_mask.scatter_(1, topk_idx, 1.0)

        # gate_mask final: if training and straight_through, use ST trick
        if self.training and self.straight_through and self.mode == 'hard':
            # forward: hard, backward: soft_probs
            gate_mask = hard_mask - soft_probs.detach() + soft_probs
        else:
            # training soft or eval hard
            if self.mode == 'soft':
                gate_mask = soft_probs
            else:
                gate_mask = hard_mask

        # load-balance loss: use soft_probs (more stable)
        if self.load_balance_coef > 0:
            importance = soft_probs.mean(dim=0)  # (G,)
            importance = dist_mean(importance)
            load_loss = self.load_balance_coef * self.num_groups * torch.sum(importance**2)
        else:
            load_loss = torch.tensor(0.0, device=device)

        # Now select top-k groups per sample
        if self._uniform_channels:
            # Efficient path: stack and gather
            # x: (B, G, C, H, W)
            x = torch.stack(groups, dim=1)
            C = self._group_channels

            # Gather selected groups: create index for gather
            idx = topk_idx.view(B, self.k, 1, 1, 1).expand(B, self.k, C, H, W)
            selected = torch.gather(x, 1, idx)  # (B, k, C, H, W)

            if self.mode == 'soft':
                # apply soft weights only for the selected groups
                # collect the soft probs for the selected indices -> (B, k)
                sel_probs = torch.gather(soft_probs, 1, topk_idx)  # (B, k)
                sel_probs = sel_probs.view(B, self.k, 1, 1, 1)
                selected = selected * sel_probs  # weighted groups
            # reshape to (B, k*C, H, W)
            out = selected.reshape(B, self.k * C, H, W)
        else:
            # Fallback path for variable channels: build per-sample list then concat
            # This is slower but correct.
            out_list = []
            for b in range(B):
                chosen = topk_idx[b]  # (k,)
                parts = []
                if self.mode == 'soft':
                    sel_probs = soft_probs[b, chosen]  # (k,)
                    for j, gi in enumerate(chosen):
                        g = groups[gi]
                        # scale only sample b
                        part = g[b:b+1] * sel_probs[j]  # shape (1, C_i, H, W)
                        parts.append(part)
                else:
                    for gi in chosen:
                        parts.append(groups[gi][b:b+1])  # keep batch dim
                # concat along channel, result (1, sum_Ci, H, W)
                parts_concat = torch.cat(parts, dim=1)
                out_list.append(parts_concat)
            # stack back to (B, sum_Ci, H, W)
            out = torch.cat(out_list, dim=0)

        aux = {
            'gate_logits': logits,
            'gate_mask': hard_mask,       # hard mask for inspection
            'gate_probs': soft_probs,     # soft probs
            'load_balance_loss': load_loss
        }
        return out, aux

# Example usage
if __name__ == "__main__":
    G = 9
    B = 4
    H = W = 32
    in_channels_list = [4] * G
    groups = [torch.randn(B, c, H, W) for c in in_channels_list]

    router = TopKGroupRouter(G, in_channels_list, k=3, mode='soft', straight_through=True)
    router.train()
    out, aux = router(groups)
    print("Train output shape:", out.shape)  # should be (B, 3*4, H, W)
    print("Gate mask sample:", aux['gate_mask'][0])
    print("Load balance loss:", aux['load_balance_loss'])

    router.eval()
    out_eval, aux_eval = router(groups)
    print("Eval output shape:", out_eval.shape)
    print("Gate mask sample (eval):", aux_eval['gate_mask'][0])
