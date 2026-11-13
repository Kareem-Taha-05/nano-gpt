import os
import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt

from configs import Config
from data.prepare import load_text, prepare
from training.train import train_model, get_device
from inference.generate import generate_text

# every study is just a list of (label, config overrides), the runs differ in ONE thing so the comparison means something
STUDIES = {
    "lr_schedule": [
        ("constant", dict(lr_schedule="constant")),
        ("cosine", dict(lr_schedule="cosine")),
        ("warmup+cosine", dict(lr_schedule="warmup_cosine")),
    ],
    "optimizer": [
        ("adam", dict(optimizer="adam")),
        ("adamw", dict(optimizer="adamw")),
        ("adamw, no clip", dict(optimizer="adamw", grad_clip=0.0)),
        ("adamw, lr 1e-3", dict(optimizer="adamw", learning_rate=1e-3)),
    ],
    "tokenizer": [
        ("char", dict(tokenizer="char")),
        ("word", dict(tokenizer="word")),
        ("bpe", dict(tokenizer="bpe")),
    ],
    "architecture": [
        ("2 layers / 2 heads / 64", dict(n_layer=2, n_head=2, n_embd=64)),
        ("4 layers / 4 heads / 128", dict(n_layer=4, n_head=4, n_embd=128)),
        ("6 layers / 4 heads / 256", dict(n_layer=6, n_head=4, n_embd=256)),
    ],
    "components": [
        ("relu + learned pos (baseline)", dict(activation="relu", pos_encoding="learned", weight_tying=False)),
        ("gelu", dict(activation="gelu", pos_encoding="learned", weight_tying=False)),
        ("gelu + weight tying", dict(activation="gelu", pos_encoding="learned", weight_tying=True)),
        ("gelu + tying + rope", dict(activation="gelu", pos_encoding="rope", weight_tying=True)),
        ("post-ln", dict(pre_ln=False)),
    ],
}


def run_study(study, base_cfg, text, device, verbose=True):
    results = []
    for label, overrides in STUDIES[study]:
        cfg = base_cfg.but(**overrides)
        tokenizer, train_data, val_data, val_chars = prepare(cfg, text)
        model, history = train_model(cfg, tokenizer, train_data, val_data, val_chars, device, name=f"{study}: {label}", verbose=verbose)
        history['label'] = label
        history['sample'] = generate_text(model, tokenizer, "\n", 200, device)
        results.append(history)
    return results


def summarize(results):
    # the table from the ideas list, but with numbers that come from real runs
    header = f"{'run':32s} {'params':>9s} {'val loss':>9s} {'val ppl':>9s} {'val bpc':>8s} {'tok/s':>9s} {'time(s)':>8s}  overfit@"
    lines = [header, '-' * len(header)]
    for h in results:
        best = min(range(len(h['val_loss'])), key=lambda i: h['val_loss'][i])
        tps = [t for t in h['tokens_per_sec'] if t > 0]
        lines.append(f"{h['label']:32s} {h['n_params']:9,d} {h['val_loss'][best]:9.4f} {h['val_ppl'][best]:9.2f} {h['val_bpc'][best]:8.3f} "
                     f"{sum(tps) / max(1, len(tps)):9,.0f} {h['train_time']:8.1f}  {h['overfit_step']}")
    return '\n'.join(lines)


def plot_results(results, title="", save_to=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for h in results:
        axes[0].plot(h['step'], h['train_loss'], label=h['label'])
        axes[1].plot(h['step'], h['val_loss'], label=h['label'])
        axes[2].plot(h['step'], h['val_bpc'], label=h['label'])
    for ax, t in zip(axes, ['train loss', 'val loss', 'val bits per char']):
        ax.set_title(f"{title} - {t}")
        ax.set_xlabel('step')
        ax.legend(fontsize=7)
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120)
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=list(STUDIES), required=True)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()
    device = get_device()
    base = Config(max_iters=args.iters, eval_interval=max(1, args.iters // 10), warmup_iters=args.iters // 25)
    results = run_study(args.study, base, load_text(), device)
    table = summarize(results)
    print(table)
    os.makedirs("experiments/results", exist_ok=True)
    with open(f"experiments/results/{args.study}.txt", "w") as f:
        f.write(table)
    plot_results(results, args.study, f"experiments/results/{args.study}.png")
