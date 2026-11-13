from dataclasses import dataclass, asdict, replace

# before, every hyperparameter was a loose global (n_embd, n_head, chunk_size...) that the classes silently read from
# now they all live in one object that we pass around explicitly, so two models with different sizes can live side by side


@dataclass
class Config:
    # data
    data_path: str = "data/input.txt"  # a text file, a folder of .txt files, or an http(s) url (tinyshakespeare is downloaded if the default path is missing)

    # tokenizer
    tokenizer: str = "char"        # char | word | bpe
    bpe_backend: str = "python"    # bpe only: python (our own implementation) | hf (huggingface `tokenizers`, written in rust, much faster on big data)
    bpe_merges: int = 256          # bpe only: vocab size will be 256 bytes + this many merges
    word_min_freq: int = 3         # word only: words seen fewer times than this become <unk>

    # model
    chunk_size: int = 128          # the context length (called block_size in karpathy's code)
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2
    activation: str = "gelu"       # relu | gelu
    pos_encoding: str = "learned"  # learned | rope
    pre_ln: bool = True            # True: x + sa(ln(x))  (what we had)   False: ln(x + sa(x))  (original transformer paper)
    weight_tying: bool = True      # share the token embedding matrix with the output layer
    flash: bool = True             # use pytorch's fused attention kernel when we don't need the attention weights

    # training
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1      # the decayed lr ends at learning_rate * min_lr_ratio
    max_iters: int = 5000
    lr_schedule: str = "warmup_cosine"  # constant | cosine | warmup_cosine
    warmup_iters: int = 200
    optimizer: str = "adamw"       # adam | adamw
    weight_decay: float = 0.1
    grad_clip: float = 1.0         # 0 disables clipping
    mixed_precision: bool = False  # gpu speedup 1: run the forward pass in 16 bit (bf16 on ampere+ gpus, fp16 + loss scaling on older ones like the kaggle T4)
    compile_model: bool = False    # gpu speedup 2: torch.compile fuses the model into faster kernels, costs a minute of compile time at the start (cuda only)

    # checkpoints
    ckpt_every_epochs: float = 0.0  # save a resumable checkpoint every N epochs (0 = off), can be fractional (0.5), "epoch" = as many tokens as the train set holds

    # evaluation
    eval_interval: int = 500
    eval_iters: int = 100
    patience: int = 3              # how many evals in a row val loss may stall (while train loss keeps dropping) before we flag overfitting
    early_stop: bool = False

    # misc
    seed: int = 1337
    use_wandb: bool = False

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.tokenizer in ("char", "word", "bpe")
        assert self.bpe_backend in ("python", "hf")
        assert self.activation in ("relu", "gelu")
        assert self.pos_encoding in ("learned", "rope")
        assert self.lr_schedule in ("constant", "cosine", "warmup_cosine")
        assert self.optimizer in ("adam", "adamw")
        if self.pos_encoding == "rope":
            assert (self.n_embd // self.n_head) % 2 == 0, "rope rotates pairs of channels, so head_size must be even"

    def to_dict(self):
        return asdict(self)

    def but(self, **changes):
        # a copy of this config with a few fields changed, this is what makes experiments one-liners
        return replace(self, **changes)


# the model sizes we talked about in the ideas list
PRESETS = {
    "small":  dict(n_embd=64,  n_head=4, n_layer=2, chunk_size=64,  batch_size=32, dropout=0.1),
    "medium": dict(n_embd=128, n_head=4, n_layer=4, chunk_size=128, batch_size=64, dropout=0.2),
    "large":  dict(n_embd=256, n_head=8, n_layer=6, chunk_size=256, batch_size=64, dropout=0.2),
}


def load_config(path):
    import yaml  # imported here so the rest of the project works without pyyaml
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))
