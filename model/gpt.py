import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.attention import rope_cos_sin
from model.transformer import Block


class GPT(nn.Module):
    """ the final model, this replaces the whole BigramLanguageModel family, it hasn't been a bigram model for a while now """

    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.chunk_size, cfg.n_embd) if cfg.pos_encoding == "learned" else None  # rope has no table at all
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])  # ModuleList instead of Sequential, since each block now also takes and returns the kv cache
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # final layer norm
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if cfg.weight_tying:
            # the embedding maps token -> vector, the lm_head maps vector -> token, the same matrix can do both jobs
            self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('sa.proj.weight') or name.endswith('ffwd.net.2.weight'):
                # every block adds two things to the residual stream, so we shrink the layers that write to it, otherwise its variance grows with depth
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past=None, use_cache=False, keep_attn=False):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        past_len = past[0][0].size(2) if past is not None else 0
        assert past_len + T <= self.cfg.chunk_size, "sequence is longer than the context the model was trained with"
        positions = torch.arange(past_len, past_len + T, device=idx.device)  # with a cache the new token isn't at position 0 anymore

        x = self.token_embedding_table(idx)  # (B,T,C)
        rope = None
        if self.cfg.pos_encoding == "learned":
            x = x + self.position_embedding_table(positions)  # (B,T,C) + (T,C)
        else:
            rope = rope_cos_sin(positions, self.cfg.n_embd // self.cfg.n_head)
        x = self.drop(x)

        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, rope, past[i] if past is not None else None, keep_attn)
            presents.append(present)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        if use_cache:
            return logits, loss, presents
        return logits, loss  # same output as before, so train loops written for the old models still work

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=True, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        was_training = self.training
        self.eval()  # generation shouldn't use dropout
        chunk_size = self.cfg.chunk_size
        past = None
        for _ in range(max_new_tokens):
            if not use_cache:
                # the old way: crop to the last chunk_size tokens and recompute everything from scratch, every single step
                logits, _ = self(idx[:, -chunk_size:])
            else:
                if past is None:
                    logits, _, past = self(idx[:, -chunk_size:], use_cache=True)  # first step: process the whole prompt once and fill the cache
                elif past[0][0].size(2) >= chunk_size:
                    # the window is full, positions would go out of range, so we rebuild the cache from the last half of the context
                    # (rebuilding from half and not from scratch means we only pay for it once every chunk_size/2 tokens)
                    logits, _, past = self(idx[:, -(chunk_size // 2):], use_cache=True)
                else:
                    logits, _, past = self(idx[:, -1:], past=past, use_cache=True)  # only the newest token goes in, everything else is in the cache
            logits = logits[:, -1, :]  # becomes (B, C)
            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # temperature -> 0 is the limit of "always take the most likely token" (greedy), dividing by 0 would give nan
            else:
                logits = logits / temperature  # <1 sharpens the distribution (safer, more repetitive), >1 flattens it (more surprising, more typos)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # the k best logits per row, sorted
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))  # everything below the k-th best can never be sampled
                probs = F.softmax(logits, dim=-1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train(was_training)
        return idx

    @torch.no_grad()
    def attention_maps(self, idx):
        # returns the attention weights of every layer and head for the first sequence in idx: (n_layer, n_head, T, T)
        was_training = self.training
        self.eval()
        self(idx[:, -self.cfg.chunk_size:], keep_attn=True)
        maps = torch.stack([b.sa.last_attn[0] for b in self.blocks])
        self.train(was_training)
        return maps.cpu()
