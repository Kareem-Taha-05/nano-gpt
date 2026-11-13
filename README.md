# nano-gpt

A small GPT language model built from scratch in PyTorch, following Andrej Karpathy's "let's build GPT" lecture, then extended into a configurable training and experiment project with a demo app.

It started as a single notebook (character-level, trained on tinyshakespeare). This repository keeps that notebook as the explanation layer and wraps the same ideas in a proper Python project: configurable models, three tokenizers, real evaluation, optimizer experiments, resumable checkpoints, optional GPU speedups, attention visualization, and a Streamlit app.

## What this project is (and isn't)

**It is a text continuer.** You give the model a starting text (the prompt) and it keeps writing, one token at a time, in the style of whatever it was trained on. Trained on Shakespeare it writes Shakespeare-shaped text. Trained on code it writes code-shaped text.

**It is not a chatbot.** It has no concept of a question or an instruction and it will not answer you. Getting a conversational model needs far more data and compute plus a further instruction-tuning stage, which is outside this project. A model trained on text laid out as `User: ... / Bot: ...` turns will imitate that format, but replies at this scale are shallow.

**The topic is decided in two places:**
- the training data decides what the model *can* write about (`--data`);
- the prompt steers it within that (`ROMEO:` starts a play script).

Expect fluent-looking local text (spelling, line structure, style) and weak long-range meaning. Larger data, a larger model and longer training on a GPU improve this, but it stays a small model.

## Project structure

```
nano-gpt/
├── configs/
│   ├── __init__.py          Config dataclass: every hyperparameter, validation, PRESETS
│   ├── small.yaml           trains in minutes on a CPU
│   ├── medium.yaml          the default, wants a GPU
│   └── large.yaml           for a real GPU (Kaggle / Colab)
├── data/
│   └── prepare.py           load text (file / folder / URL), 90/10 split, tokenize, get_batch
├── model/
│   ├── tokenizer.py         Char, Word and BPE tokenizers (BPE: our own or Hugging Face)
│   ├── attention.py         causal self-attention, RoPE, fused kernel, KV cache support
│   ├── transformer.py       feed-forward + transformer block (pre-LN or post-LN)
│   └── gpt.py               the GPT class: forward, generate, attention_maps
├── training/
│   ├── train.py             training loop, checkpoints, resume, command line
│   ├── optimizer.py         Adam / AdamW, weight-decay groups, LR schedules
│   └── evaluation.py        loss, perplexity, bits per char, overfitting detector
├── inference/
│   ├── generate.py          load a checkpoint, generate text, KV cache check + benchmark
│   └── attention_viz.py     attention heatmaps
├── experiments/
│   └── compare_models.py    five ready-made comparison studies
├── app/
│   └── streamlit_app.py     the demo app
├── checkpoints/             trained models are saved here
├── notebooks/
│   └── nanogpt_continued.ipynb   the notebook continuation, generated from the files above
├── requirements.txt
└── README.md
```

The notebook cells are generated from the same module files, so the notebook and the project cannot drift apart.

## How it works

### Data flow

1. `data/prepare.py` loads the text (a file, a folder of `.txt` files, or a URL; tinyshakespeare is downloaded if you use the default path).
2. The **text** is split 90% train / 10% validation, then tokenized. Splitting before tokenizing means every tokenizer is validated on exactly the same characters, which is what makes tokenizer comparisons fair. Vocabulary and merges are learned from the train part only (the char tokenizer uses the whole alphabet, so a rare character that only appears in validation cannot crash it).
3. `training/train.py` builds a `GPT` from a `Config`, trains it with periodic evaluation, and saves a checkpoint.
4. A checkpoint is one `.pt` file holding the weights, the config, the tokenizer and the validation metrics. The app and the generation script need nothing else.

### Tokenizers

| Tokenizer | Each token is | Notes |
|---|---|---|
| `char` | one character | the original notebook approach, vocabulary of ~65 for tinyshakespeare |
| `word` | a word with its leading space, or one punctuation mark | words seen fewer than `word_min_freq` times become `<unk>`, large vocabulary |
| `bpe` | a learned subword unit | byte-level, so nothing is ever unknown; `bpe_merges` sets the vocabulary size (256 bytes + merges) |

BPE has two backends, selected by `bpe_backend`:
- `python`: our own implementation, easy to read, fine for small and medium data;
- `hf`: Hugging Face's Rust `tokenizers` library (`pip install tokenizers`), much faster on large corpora.

### The model

One configurable `GPT` class replaces the notebook's `BigramLanguageModel1-7`:
- token embeddings plus learned positions, or rotary position embeddings (`pos_encoding: rope`);
- `n_layer` transformer blocks with multi-head causal self-attention, a feed-forward layer (ReLU or GELU), residual connections and LayerNorm (pre-LN by default, post-LN optional);
- optional weight tying between the embedding and the output layer;
- all attention heads computed in one fused module, using PyTorch's fused attention kernel when attention weights aren't needed;
- a KV cache for generation: keys and values of already-processed tokens are stored, so each new token costs one step instead of recomputing the whole context. The cached output is verified to match the full forward pass exactly. When the context window fills up, the cache is rebuilt from the last half of the context.

Generation supports `temperature` (0 = greedy, below 1 safer and more repetitive, above 1 more adventurous) and `top_k` (sample only from the k most likely tokens).

### Training and evaluation

- Optimizer: AdamW (or Adam for comparison), with weight decay applied only to weight matrices, not biases or LayerNorm gains.
- Learning rate: constant, cosine decay, or linear warmup followed by cosine decay (the default).
- Gradient clipping (default 1.0).
- Evaluation averages many batches with dropout off, for both train and validation. It reports validation loss, perplexity, **bits per character** and tokens/second.
- Bits per character is the metric that compares tokenizers. Perplexity is per token, and a BPE token covers about two characters, so perplexity alone can't compare a char model with a BPE model.
- An overfitting detector flags when validation loss stops improving for `patience` evaluations while training loss keeps dropping. Early stopping on that flag is optional.
- The best-validation weights are kept and saved as the final checkpoint.

### GPU speedups (both optional)

- `mixed_precision`: runs the forward pass in 16 bit. It uses bf16 on Ampere or newer GPUs and fp16 with loss scaling on older GPUs such as the Kaggle T4. Weights and optimizer stay in fp32.
- `compile_model`: `torch.compile`, CUDA only (skipped with a message otherwise). Costs about a minute of compile time at the start, then faster steps.

### Checkpoints

- The final checkpoint (`--ckpt`) holds the best-validation weights. It is small and inference-only.
- With `ckpt_every_epochs: N`, a resumable checkpoint is also written every N epochs (`name-epoch5.pt`, ...). These additionally hold the optimizer state and step, so they are about 3x bigger and can be resumed.
- One "epoch" means as many tokens as the training set holds. Batches are sampled randomly, so this is a count of tokens seen, not a strict pass over the data.

## Quick start

```
pip install -r requirements.txt

python -m data.prepare                                    # downloads tinyshakespeare, compares the tokenizers
python -m training.train --config configs/small.yaml --ckpt checkpoints/char-small.pt
python -m inference.generate --ckpt checkpoints/char-small.pt --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top_k 40
python -m inference.generate --ckpt checkpoints/char-small.pt --benchmark      # KV cache check + speed
streamlit run app/streamlit_app.py
```

Run everything from the project root. `small` trains in minutes on a CPU; `medium` and `large` want a GPU.

## Configuration

Settings live in a `Config` dataclass. You can set them three ways:
- a YAML file (`--config configs/medium.yaml`), a plain text file of settings that saves you from retyping flags;
- command-line flags: **every Config field is also a flag**, generated automatically;
- in code, with `cfg.but(n_layer=6)`, which returns a modified copy.

These are decisions you make at **training time**. They are saved inside the checkpoint, and the app only displays them.

| Flag | What it does |
|---|---|
| `--data PATH` | text file, folder of `.txt` files, or http(s) URL (alias of `--data_path`) |
| `--tokenizer char\|word\|bpe` | which tokenizer |
| `--bpe_backend python\|hf` | own BPE or Hugging Face BPE |
| `--bpe_merges N` | BPE vocabulary size is 256 + N |
| `--n_embd / --n_head / --n_layer / --chunk_size / --dropout` | model size and context length (`chunk_size` is the context length) |
| `--activation relu\|gelu`, `--pos_encoding learned\|rope`, `--pre_ln`, `--weight_tying` | architecture options |
| `--learning_rate / --lr_schedule / --warmup_iters / --optimizer / --weight_decay / --grad_clip` | optimization |
| `--iters N` | training steps (alias of `--max_iters`) |
| `--mixed_precision true` | 16-bit forward pass |
| `--compile_model true` | `torch.compile` (CUDA only) |
| `--ckpt_every_epochs N` | resumable checkpoint every N epochs (0 = off, fractions allowed) |
| `--eval_interval / --eval_iters / --patience / --early_stop` | evaluation and overfitting detection |
| `--use_wandb true` | log to Weights & Biases |

Example:

```
python -m training.train --config configs/medium.yaml --data my_books/ --tokenizer bpe --bpe_backend hf --mixed_precision true --ckpt_every_epochs 5 --ckpt checkpoints/books-bpe.pt
```

Resume a stopped run from an epoch checkpoint (the config comes from the checkpoint; keep `max_iters` unchanged or the learning-rate schedule will jump, and best-weights tracking restarts at the resume point):

```
python -m training.train --resume checkpoints/books-bpe-epoch5.pt --ckpt checkpoints/books-bpe.pt
```

## Experiments

`experiments/compare_models.py` runs studies where every run differs from the base config in exactly one thing, and prints a table (parameters, validation loss, perplexity, bits per char, tokens/s, time, overfitting step) plus loss curves saved to `experiments/results/`.

```
python -m experiments.compare_models --study lr_schedule     # constant vs cosine vs warmup + cosine
python -m experiments.compare_models --study optimizer       # Adam vs AdamW, clipping on/off, higher LR
python -m experiments.compare_models --study tokenizer       # char vs word vs BPE
python -m experiments.compare_models --study architecture    # 2/4/6 layers, heads and widths
python -m experiments.compare_models --study components      # ReLU/GELU, weight tying, RoPE, post-LN
```

Use `--iters` to change the run length (default 2000). Notes for reading the tokenizer study: the word model has a much larger embedding matrix, and its bits-per-char is optimistic because every `<unk>` hides the characters of a rare word. With the same context length in tokens, BPE and word models also see more characters of context than the char model.

## The app

```
streamlit run app/streamlit_app.py
```

- **Sidebar:** a model dropdown (every `.pt` in `checkpoints/`) and read-only facts about the selected model: tokenizer and vocabulary size, layers/heads/dimensions/context length, parameter count, validation loss and bits per character.
- **Generate tab:** prompt box, max-new-tokens slider, temperature slider (0-2), top-k input, and the generated text.
- **Attention tab:** type a sentence, pick a layer, and see one heatmap per attention head showing which earlier tokens each token looks at.

Each checkpoint carries its own tokenizer and config, so models trained on different datasets, with different tokenizers or sizes, can sit side by side in `checkpoints/` and be switched from the dropdown. Name the files descriptively (`shakespeare-char.pt`, `python-code-bpe.pt`), since the dropdown shows file names.

Characters outside a char model's vocabulary produce an error message in the app rather than a crash.

## Training on Kaggle (or any GPU machine)

1. Create a notebook with a GPU accelerator and internet on. Add your text as a dataset if it isn't downloadable.
2. In a cell:
   ```
   !git clone https://github.com/<you>/nano-gpt
   !pip install -q pyyaml tokenizers
   !cd nano-gpt && python -m training.train --config configs/large.yaml --data /kaggle/input/<your-dataset> --tokenizer bpe --bpe_backend hf --mixed_precision true --ckpt_every_epochs 2 --ckpt /kaggle/working/books-bpe.pt
   ```
3. Download the `.pt` file from the notebook's output tab into your local `checkpoints/`; the app picks it up.

Sessions are time-limited. To continue in a new session, upload the last epoch checkpoint as a dataset and pass it to `--resume`.

## Deployment notes

- The app runs on CPU at this model size, so Streamlit Community Cloud or Hugging Face Spaces both work.
- Commit or upload only the final small checkpoints. Epoch checkpoints are large and `.gitignore` excludes all `.pt` files by default (`git add -f` for the ones you want).
- Checkpoints contain a pickled tokenizer, so **only load checkpoints you trained yourself or trust**.

## Status and limitations

- Tested on CPU: tokenizers (round trips, both BPE backends), model variants, KV cache equivalence, sampling controls, checkpoints and resume, the command line, the app (headless), and the full notebook end to end. The CUDA-specific paths (real mixed precision, `torch.compile` inside training, speedups) have not been run on a GPU.
- Weights & Biases logging has not been run.
- No trained results are included, the tiny CPU runs were only smoke tests. Run the studies yourself to get real numbers.
- Requires PyTorch 2.3 or newer.

## Credits

Built following Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" lecture and nanoGPT, using the tinyshakespeare dataset.
