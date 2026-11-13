import torch.nn as nn

from model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, cfg):
        super().__init__()
        act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            act,
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, cfg):
        super().__init__()
        self.sa = CausalSelfAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.pre_ln = cfg.pre_ln

    def forward(self, x, rope=None, past=None, keep_attn=False):
        if self.pre_ln:
            a, present = self.sa(self.ln1(x), rope, past, keep_attn)  # normalize BEFORE the sublayer, the residual stream itself stays untouched
            x = x + a
            x = x + self.ffwd(self.ln2(x))
        else:
            a, present = self.sa(x, rope, past, keep_attn)  # normalize AFTER adding, like in "attention is all you need"
            x = self.ln1(x + a)
            x = self.ln2(x + self.ffwd(x))
        return x, present  # present is the (keys, values) of this layer for the kv cache
