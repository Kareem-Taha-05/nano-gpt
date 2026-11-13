import time
import argparse
import torch

from configs import Config
from model.gpt import GPT


def load_checkpoint(path, device='cpu'):
    # the checkpoint holds the weights, the config AND the tokenizer, so a model can always be rebuilt from one file
    # (weights_only=False because we pickle the tokenizer, only load checkpoints you made yourself)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt['config'])
    tokenizer = ckpt['tokenizer']
    model = GPT(cfg, tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.metrics = ckpt.get('metrics', {})  # val loss / bits per char / params saved at training time, the app shows them
    return model, tokenizer, cfg


def generate_text(model, tokenizer, prompt="\n", max_new_tokens=500, device='cpu', use_cache=True, temperature=1.0, top_k=None):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)  # the default prompt is a newline, which is token 0 for the char model, just like torch.zeros before
    out = model.generate(idx, max_new_tokens, use_cache=use_cache, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


@torch.no_grad()
def verify_cache(model, device='cpu', atol=1e-4):
    # same idea as our torch.allclose checks at the start: the cache must change the speed, never the answer
    # we feed a sequence token by token through the cache, and compare against one full forward pass
    model.eval()
    T = model.cfg.chunk_size
    idx = torch.randint(0, model.token_embedding_table.num_embeddings, (1, T), device=device)
    full, _ = model(idx)
    past, steps = None, []
    for t in range(T):
        logits, _, past = model(idx[:, t:t + 1], past=past, use_cache=True)
        steps.append(logits)
    return torch.allclose(full, torch.cat(steps, dim=1), atol=atol)


@torch.no_grad()
def benchmark_kv_cache(model, new_tokens=None, device='cpu'):
    # the cached version is exactly the same math, so the only thing that changes is how long it takes
    model.eval()
    new_tokens = new_tokens or model.cfg.chunk_size - 1  # stay inside one context window so the comparison is clean
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    results = {}
    for use_cache in (False, True):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        model.generate(idx, new_tokens, use_cache=use_cache)
        if device == 'cuda':
            torch.cuda.synchronize()
        results[use_cache] = time.time() - t0
        print(f"use_cache={use_cache}: {new_tokens} tokens in {results[use_cache]:.2f}s ({new_tokens / results[use_cache]:.0f} tok/s)")
    print(f"speedup: {results[False] / results[True]:.2f}x")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, cfg = load_checkpoint(args.ckpt, device)
    if args.benchmark:
        print("cache matches full forward pass:", verify_cache(model, device))
        benchmark_kv_cache(model, device=device)
    else:
        print(generate_text(model, tokenizer, args.prompt, args.tokens, device, temperature=args.temperature, top_k=args.top_k))
