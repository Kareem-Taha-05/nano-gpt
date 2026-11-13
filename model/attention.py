import torch
import torch.nn as nn
from torch.nn import functional as F


def rope_cos_sin(positions, head_size, base=10000.0):
    # rotary embeddings: instead of ADDING a position vector to the input, we ROTATE the queries and keys by an angle that depends on the position
    # every pair of channels (i, i + head_size/2) gets its own rotation speed, positions: (T,) absolute positions
    freqs = 1.0 / (base ** (torch.arange(0, head_size, 2, device=positions.device).float() / head_size))  # (head_size/2,)
    angles = positions.float()[:, None] * freqs[None, :]  # (T, head_size/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x is (B, H, T, head_size), we rotate each (x1, x2) pair, the dot product q.k then only depends on the RELATIVE distance between the two positions
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)  # cos/sin are float32, under mixed precision we must give back the dtype we got


class CausalSelfAttention(nn.Module):
    """ all the heads of self-attention in one module, same idea as Head + MultiHeadAttention but computed in parallel """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # one big matrix instead of 3 * n_head tiny ones, the heads are just slices of it
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout
        self.flash = cfg.flash
        self.register_buffer('tril', torch.tril(torch.ones(cfg.chunk_size, cfg.chunk_size)))
        self.last_attn = None  # only filled when we ask for it, this is what the visualization reads

    def forward(self, x, rope=None, past=None, keep_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # note that v is NOT rotated, only the things used to compute the affinities

        if past is not None:
            # kv cache: the keys and values of the tokens we already processed are glued in front of the new ones
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        Tk = k.size(2)  # number of keys = cached tokens + new tokens

        if self.flash and not keep_attn and (past is None or T == 1):
            # fused kernel, never builds the (T, T) matrix in memory, with a single new token no mask is needed since it may look at everything before it
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=(past is None))
        else:
            wei = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5  # (B, H, T, Tk), scaled by head_size, the size of the vectors being dotted
            wei = wei.masked_fill(self.tril[Tk - T:Tk, :Tk] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            if keep_attn:
                self.last_attn = wei.detach()
            wei = self.attn_dropout(wei)
            out = wei @ v  # (B, H, T, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together, this is the torch.cat we used to do
        return self.resid_dropout(self.proj(out)), present
