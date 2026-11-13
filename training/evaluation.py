import math
import torch

from data.prepare import get_batch


@torch.no_grad()  # no gradients needed here, this saves memory and time
def estimate_loss(model, train_data, val_data, cfg, device, amp_ctx=None):
    # the loss of a single batch is very noisy, so we average over eval_iters batches, for both splits
    # eval mode turns dropout off, so train and val are measured under the same conditions (the running train loss we print inside the loop has dropout on)
    amp_ctx = amp_ctx or (lambda: torch.autocast(device_type='cpu', enabled=False))  # a context factory, so evaluation uses the same precision as training
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg, device)
            with amp_ctx():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def perplexity(loss):
    # ppl = e^(cross entropy), "the model is as unsure as if it had to pick uniformly between ppl tokens"
    return math.exp(loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())  # parameters() already skips duplicates, so tied weights are counted once


def bits_per_char(loss, n_tokens, n_chars):
    # perplexity is per TOKEN, so a char model and a bpe model can't be compared with it (a bpe token is worth ~4 chars)
    # total nats = loss * n_tokens, we divide by the characters those tokens cover, and by ln(2) to get bits
    return loss * n_tokens / n_chars / math.log(2)


class OverfitDetector:
    """ flags overfitting when val loss stops improving for `patience` evals in a row while the train loss is still going down """

    def __init__(self, patience=3, min_delta=1e-3):
        self.patience, self.min_delta = patience, min_delta
        self.best_val = float('inf')
        self.prev_train = float('inf')
        self.bad_evals = 0

    def update(self, train_loss, val_loss):
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.bad_evals = 0
        elif train_loss < self.prev_train:  # val is stuck/worse but train still improves -> the gap is growing
            self.bad_evals += 1
        self.prev_train = train_loss
        return self.bad_evals >= self.patience
