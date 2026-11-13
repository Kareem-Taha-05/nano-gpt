import torch
import matplotlib.pyplot as plt


def token_labels(tokenizer, ids):
    return [tokenizer.decode([i]).replace('\n', '\\n').replace(' ', '·') for i in ids]  # make newlines and spaces visible on the axes


def plot_attention(model, tokenizer, sentence, layer=0, device='cpu'):
    # one heatmap per head: row = the token that is looking, column = the token it is looking at
    ids = tokenizer.encode(sentence)[-model.cfg.chunk_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    maps = model.attention_maps(idx)  # (n_layer, n_head, T, T)
    n_head = maps.size(1)
    labels = token_labels(tokenizer, ids)
    fig, axes = plt.subplots(1, n_head, figsize=(4.2 * n_head, 4.4), squeeze=False)
    for h in range(n_head):
        ax = axes[0][h]
        ax.imshow(maps[layer, h].numpy(), cmap='viridis')
        ax.set_title(f"layer {layer}, head {h}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    plt.tight_layout()
    return fig
