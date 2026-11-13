import os
from glob import glob
from urllib.parse import urlparse
import urllib.request
import numpy as np
import torch

from model.tokenizer import get_tokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_PATH = "data/input.txt"
_cache = {}  # building a tokenizer can take a while, so experiments that reuse one shouldn't pay for it twice


def _download(url, local):
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    if not os.path.exists(local):
        urllib.request.urlretrieve(url, local)
    return local


def load_text(path=None):
    # the training data can now be: a single text file, a folder (every .txt file inside, subfolders included, is joined), or an http(s) url
    path = path or DEFAULT_PATH
    if path.startswith(("http://", "https://")):
        path = _download(path, os.path.join("data", os.path.basename(urlparse(path).path) or "download.txt"))
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "**", "*.txt"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .txt files found inside {path}")
        parts = []
        for fp in files:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:  # real-world corpora always contain a few broken bytes, we replace them instead of crashing
                parts.append(f.read())
        print(f"loaded {len(files)} files from {path}")
        return "\n\n".join(parts)
    if not os.path.exists(path):
        if path == DEFAULT_PATH:
            _download(URL, path)  # the original tinyshakespeare, same as the !wget in the notebook
        else:
            raise FileNotFoundError(f"data path not found: {path}")
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def encode_tensor(tokenizer, text):
    # int32 instead of int64 halves the memory of the dataset (matters at 100M+ tokens), get_batch converts the small batches back to long
    if hasattr(tokenizer, 'encode_np'):
        arr = tokenizer.encode_np(text)  # the char and huggingface tokenizers have a fast numpy path
    else:
        arr = np.array(tokenizer.encode(text), dtype=np.int32)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32))


def prepare(cfg, text):
    # we split the TEXT 90/10 first and only then tokenize, so every tokenizer is validated on exactly the same characters
    # (splitting the token stream would give each tokenizer a slightly different validation set, and then we couldn't compare them)
    key = (cfg.tokenizer, cfg.bpe_backend, cfg.bpe_merges, cfg.word_min_freq, len(text))
    if key not in _cache:
        split = int(0.9 * len(text))
        train_text, val_text = text[:split], text[split:]
        # the vocab/merges are learned from the train part only, no peeking at val
        # except for the char tokenizer: its "vocabulary" is just the alphabet, and on a big corpus a rare character that only shows up in val would crash the encoding
        tokenizer = get_tokenizer(cfg, text if cfg.tokenizer == 'char' else train_text)
        train_data = encode_tensor(tokenizer, train_text)
        val_data = encode_tensor(tokenizer, val_text)
        _cache[key] = (tokenizer, train_data, val_data, len(val_text))
    return _cache[key]  # tokenizer, train_data, val_data, number of characters in val


def get_batch(data, cfg, device):
    ix = torch.randint(low=0, high=len(data) - cfg.chunk_size - 1, size=(cfg.batch_size,))
    x = torch.stack([data[i: i + cfg.chunk_size] for i in ix])
    y = torch.stack([data[i + 1: i + cfg.chunk_size + 1] for i in ix])
    return x.to(device).long(), y.to(device).long()  # the tensors have to live on the same device as the model, and embeddings want int64


if __name__ == "__main__":
    import time
    from configs import Config
    text = load_text()
    for t, backend in (("char", "python"), ("word", "python"), ("bpe", "python"), ("bpe", "hf")):
        t0 = time.time()
        try:
            tok, tr, va, nchars = prepare(Config(tokenizer=t, bpe_backend=backend), text)
        except ImportError:
            print(f"{t:5s} ({backend}) skipped, pip install tokenizers")
            continue
        print(f"{t:5s} ({backend:6s}) vocab {tok.vocab_size:6d}  train tokens {len(tr):8d}  chars/token {nchars / len(va):.2f}  {time.time() - t0:.1f}s")
