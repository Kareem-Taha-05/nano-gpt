import os
import time
import argparse
import dataclasses
import torch

from configs import Config, load_config
from data.prepare import load_text, prepare, get_batch
from model.gpt import GPT
from training.evaluation import estimate_loss, perplexity, count_parameters, bits_per_char, OverfitDetector
from training.optimizer import configure_optimizer, get_lr


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def amp_settings(cfg, device):
    # which 16 bit format to use, and whether we need loss scaling
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # bf16 has the same range as fp32 so it needs nothing special, but only ampere gpus (compute capability 8+) run it natively, the kaggle T4/P100 don't
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.bfloat16  # cpu autocast only really makes sense in bf16, and only pays off on newer cpus
    use_scaler = cfg.mixed_precision and dtype == torch.float16  # fp16 can't represent tiny gradients (they become 0), so we multiply the loss by a big number first and divide the gradients afterwards
    return device_type, dtype, use_scaler


def save_checkpoint(path, weights, cfg, tokenizer, metrics, optimizer=None, step=None, history=None):
    # the final checkpoint is inference-only (small), the periodic ones also carry the optimizer state and the step so training can be resumed from them
    ckpt = {'model': weights, 'config': cfg.to_dict(), 'tokenizer': tokenizer, 'metrics': metrics}
    if optimizer is not None:
        ckpt.update(optimizer=optimizer.state_dict(), step=step, history=history)
    torch.save(ckpt, path)


def train_model(cfg, tokenizer, train_data, val_data, val_chars, device, name="run", verbose=True, ckpt_path=None, resume_from=None):
    torch.manual_seed(cfg.seed)
    model = GPT(cfg, tokenizer.vocab_size).to(device)
    n_params = count_parameters(model)
    optimizer = configure_optimizer(model, cfg)
    detector = OverfitDetector(cfg.patience)
    history = {k: [] for k in ['step', 'train_loss', 'val_loss', 'val_ppl', 'val_bpc', 'lr', 'tokens_per_sec', 'elapsed']}
    history.update(name=name, n_params=n_params, overfit_step=None, best_val=float('inf'))

    start_it = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        assert 'optimizer' in ckpt, "this checkpoint has no optimizer state, only the periodic epoch checkpoints can be resumed"
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history = ckpt['history']
        history['best_val'] = float('inf')  # the best weights are tracked from the resume point on
        start_it = ckpt['step'] + 1
        assert start_it <= cfg.max_iters, f"checkpoint is already at step {ckpt['step']}, raise max_iters to keep training"
    torch.manual_seed(cfg.seed + start_it)  # a resumed run must not replay the same batches it already saw

    # how many training steps are "one epoch", batches are random so this is the number of tokens in the train set divided by the tokens per step
    iters_per_epoch = len(train_data) / (cfg.batch_size * cfg.chunk_size)
    ckpt_every = max(1, round(cfg.ckpt_every_epochs * iters_per_epoch)) if (cfg.ckpt_every_epochs > 0 and ckpt_path) else 0

    # gpu speedups, both are opt-in through the config
    device_type, amp_dtype, use_scaler = amp_settings(cfg, device)
    amp_ctx = lambda: torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=cfg.mixed_precision)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)  # when it is disabled, scale/unscale/step/update are plain pass-throughs, so there is one code path for everything
    fwd = model  # evaluation and generation always use the plain model, only the training steps use the compiled one
    if cfg.compile_model:
        if device_type != 'cuda':
            print("compile_model is only used on cuda, continuing without it")
        elif torch.cuda.get_device_capability()[0] < 7:
            # the compiler (triton) needs a volta/turing/newer gpu, and the error would only appear at the first training step, so we check up front (kaggle's P100 is too old)
            print(f"compile_model needs a gpu with compute capability 7+, this one has {torch.cuda.get_device_capability()[0]}.x, continuing without it")
        else:
            try:
                fwd = torch.compile(model)  # the compiled wrapper shares its parameters with `model`, so checkpoints stay clean
            except Exception as e:
                print(f"torch.compile failed ({e}), continuing without it")

    if verbose:
        prec = str(amp_dtype).replace('torch.', '') if cfg.mixed_precision else 'fp32'
        print(f"[{name}] {n_params/1e6:.3f}M parameters, vocab {tokenizer.vocab_size}, device {device}, precision {prec}, compile {fwd is not model}, "
              f"{iters_per_epoch:.0f} steps/epoch ({cfg.max_iters / iters_per_epoch:.1f} epochs total)" + (f", resuming at step {start_it}" if start_it else ""))

    if cfg.use_wandb:
        import wandb
        wandb.init(project="nano-gpt", name=name, config={**cfg.to_dict(), 'n_params': n_params, 'vocab_size': tokenizer.vocab_size})

    best_state, tokens_since = None, 0
    sync = torch.cuda.synchronize if device_type == 'cuda' else (lambda: None)
    t_start = time.time() - (history['elapsed'][-1] if history['elapsed'] else 0.0)
    t_last = time.time()
    for it in range(start_it, cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            sync()
            interval = time.time() - t_last
            tps = tokens_since / interval if tokens_since > 0 else 0.0  # throughput of the training steps only, the evaluation below isn't counted
            losses = estimate_loss(model, train_data, val_data, cfg, device, amp_ctx)
            bpc = bits_per_char(losses['val'], len(val_data), val_chars)
            for k, v in zip(history.keys(), [it, losses['train'], losses['val'], perplexity(losses['val']), bpc, lr, tps, time.time() - t_start]):
                history[k].append(v)
            if losses['val'] < history['best_val']:
                history['best_val'] = losses['val']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # keep the best weights, the last ones may already be overfit
            overfit = detector.update(losses['train'], losses['val'])
            if overfit and history['overfit_step'] is None:
                history['overfit_step'] = it
                if verbose:
                    print(f"  -> overfitting detected at step {it}: val loss stopped improving while train loss kept dropping (gap {losses['val'] - losses['train']:.3f})")
            if verbose:
                print(f"step {it} (epoch {it / iters_per_epoch:.1f}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val ppl {perplexity(losses['val']):.2f}, val bpc {bpc:.3f}, lr {lr:.2e}, {tps:,.0f} tok/s")
            if cfg.use_wandb:
                wandb.log({'train_loss': losses['train'], 'val_loss': losses['val'], 'val_ppl': perplexity(losses['val']), 'val_bpc': bpc,
                           'lr': lr, 'tokens_per_sec': tps, 'gap': losses['val'] - losses['train'], 'epoch': it / iters_per_epoch}, step=it)
            if overfit and cfg.early_stop:
                break
            tokens_since, t_last = 0, time.time()

        if it == cfg.max_iters:
            break

        x, y = get_batch(train_data, cfg, device)
        with amp_ctx():  # only the forward pass runs in 16 bit, the weights and the optimizer stay in fp32
            logits, loss = fwd(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)  # clipping must see the real gradients, not the scaled ones
            # if one unlucky batch produces a huge gradient, we rescale it so the update can't blow up the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_since += cfg.batch_size * cfg.chunk_size

        if ckpt_every and (it + 1) % ckpt_every == 0:
            epoch = (it + 1) / iters_per_epoch
            path = ckpt_path.replace('.pt', f'-epoch{epoch:.2f}'.rstrip('0').rstrip('.') + '.pt')
            last = {'val_loss': history['val_loss'][-1], 'val_ppl': history['val_ppl'][-1], 'val_bpc': history['val_bpc'][-1], 'n_params': n_params, 'step': it + 1}  # from the latest evaluation
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(path, weights, cfg, tokenizer, last, optimizer, it, history)
            if verbose:
                print(f"  -> saved {path} (epoch {epoch:.2f})")

    history['train_time'] = time.time() - t_start
    if ckpt_path:
        b = min(range(len(history['val_loss'])), key=lambda i: history['val_loss'][i])
        metrics = {'val_loss': history['val_loss'][b], 'val_ppl': history['val_ppl'][b], 'val_bpc': history['val_bpc'][b], 'n_params': n_params, 'step': history['step'][b]}
        save_checkpoint(ckpt_path, best_state if best_state is not None else model.state_dict(), cfg, tokenizer, metrics)
    if cfg.use_wandb:
        sample = tokenizer.decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 300)[0].tolist())
        wandb.log({'sample': wandb.Html(f"<pre>{sample}</pre>")})
        wandb.finish()
    return model, history


if __name__ == "__main__":
    # every field of Config is also a command line flag, generated automatically so the two can never get out of sync
    # e.g. python -m training.train --config configs/medium.yaml --tokenizer bpe --bpe_backend hf --mixed_precision true --data /kaggle/input/my-corpus
    def str2bool(v):
        return str(v).lower() in ('1', 'true', 'yes', 'y')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="yaml file with the base config")
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    parser.add_argument("--resume", default=None, help="path of an epoch checkpoint to continue training from (its own config is used)")
    for f in dataclasses.fields(Config):
        flags = ["--" + f.name] + (["--data"] if f.name == "data_path" else []) + (["--iters"] if f.name == "max_iters" else [])
        parser.add_argument(*flags, dest=f.name, default=None, type=str2bool if f.type is bool else f.type)
    args = parser.parse_args()

    if args.resume:
        cfg = Config(**torch.load(args.resume, map_location='cpu', weights_only=False)['config'])
    else:
        cfg = load_config(args.config) if args.config else Config()
    overrides = {f.name: getattr(args, f.name) for f in dataclasses.fields(Config) if getattr(args, f.name) is not None}
    cfg = cfg.but(**overrides)

    text = load_text(cfg.data_path)
    print(f"{len(text):,} characters from {cfg.data_path}")
    tokenizer, train_data, val_data, val_chars = prepare(cfg, text)
    print(f"{len(train_data):,} train tokens, {len(val_data):,} val tokens, vocab {tokenizer.vocab_size}")
    device = get_device()
    os.makedirs(os.path.dirname(args.ckpt) or '.', exist_ok=True)
    model, history = train_model(cfg, tokenizer, train_data, val_data, val_chars, device, name=args.config or "default", ckpt_path=args.ckpt, resume_from=args.resume)
    idx = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
