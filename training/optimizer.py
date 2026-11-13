import math
import torch


def configure_optimizer(model, cfg):
    # weight decay should only shrink the big weight matrices (dim >= 2: linear layers, embeddings)
    # biases and layernorm gains (dim < 2) are left alone, decaying them only hurts
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{'params': decay, 'weight_decay': cfg.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.learning_rate)  # decay is applied directly to the weights, separate from the gradient
    return torch.optim.Adam(groups, lr=cfg.learning_rate)  # here the same number becomes an L2 penalty added to the gradient, which adam then rescales


def get_lr(it, cfg):
    # we compute the lr by hand every step instead of using torch schedulers, a plain function of the step is easy to plot and to reason about
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    warmup = cfg.warmup_iters if cfg.lr_schedule == "warmup_cosine" else 0
    if it < warmup:
        return cfg.learning_rate * (it + 1) / warmup  # linear warmup, adam's statistics are garbage in the first steps so we start gently
    progress = min(1.0, (it - warmup) / max(1, cfg.max_iters - warmup))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.learning_rate - min_lr)  # cosine from learning_rate down to min_lr
