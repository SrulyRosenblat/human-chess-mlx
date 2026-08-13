"""Train the human-chess next-move model on Apple Silicon with MLX."""

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare import (
    DATA_DIR,
    LENGTH_BUCKETS,
    MAX_SEQ_LEN,
    TEST_GAMES,
    TIME_BUDGET,
    Tokenizer,
    VAL_GAMES,
    evaluate_move_bits,
    make_dataloader,
)
from checkpointing import load_checkpoint, save_checkpoint, save_rotating_snapshot

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = norm(self.rope(q))
        k = norm(self.rope(k))

        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config.n_layer,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }
        self._mask_cache = {}

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-scale, scale, block.attn.c_q.weight.shape).astype(mx.bfloat16)
            block.attn.c_k.weight = mx.random.uniform(-scale, scale, block.attn.c_k.weight.shape).astype(mx.bfloat16)
            block.attn.c_v.weight = mx.random.uniform(-scale, scale, block.attn.c_v.weight.shape).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)
            block.mlp.c_fc.weight = mx.random.uniform(-scale, scale, block.mlp.c_fc.weight.shape).astype(mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight).astype(mx.bfloat16)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(mx.bfloat16)

        self.resid_lambdas = mx.ones((self.config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1, dtype=mx.float32)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-scale, scale, ve.weight.shape).astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        # Build masks in the model's dtype (bfloat16). scaled_dot_product_attention
        # requires the mask to promote to the q/k/v dtype; now that the residual
        # stream stays bf16 (issue #3), a float32 mask no longer promotes.
        dtype = self.wte.weight.dtype
        unique_windows = set(self.window_sizes)
        for window_size in unique_windows:
            key = (seq_len, window_size, dtype)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len, dtype=dtype)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size, dtype=dtype)
        return [self._mask_cache[(seq_len, window_size, dtype)] for window_size in self.window_sizes]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            # Cast the fp32 scalars to the hidden-state dtype so the residual
            # stream stays bfloat16. Without this, float32 * bfloat16 promotes
            # to float32 and the whole network silently runs in fp32 after the
            # first block, defeating the bf16 init (issue #3).
            x = self.resid_lambdas[i].astype(x.dtype) * x + self.x0_lambdas[i].astype(x.dtype) * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        x = norm(x)

        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is None:
            return logits

        valid = targets != -1
        targets_safe = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
        ce = ce * valid
        if reduction == "none":
            return ce
        denom = mx.maximum(mx.sum(valid), 1)
        return mx.sum(ce) / denom


class AdamW:
    def __init__(self, model, unembedding_lr, embedding_lr, matrix_lr, weight_decay, adam_betas, scalar_lr):
        self.param_config = {}
        self.adam_state = {}

        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                self.param_config[path] = {
                    "lr": matrix_lr,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                }
            elif "wte" in path:
                self.param_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "value_embeds" in path:
                self.param_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.param_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "resid_lambdas" in path:
                self.param_config[path] = {
                    "lr": scalar_lr * 0.01,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "x0_lambdas" in path:
                self.param_config[path] = {
                    "lr": scalar_lr,
                    "betas": (0.96, 0.95),
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            else:
                self.param_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }

        self.initial_lrs = {path: config["lr"] for path, config in self.param_config.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        weight_decay = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * weight_decay)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            param = flat_params[path]
            new_param = self._step(path, grad, param, config)
            self._set_path_value(model, path, new_param)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

# v0.1: AdamW only. Muon port is future work.
TOTAL_BATCH_SIZE = 2**13
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 6
DEVICE_BATCH_SIZE = 16
FINAL_EVAL_BATCH_SIZE = 64
STARTUP_EXCLUDE_STEPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train the human-chess MLX model")
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument(
        "--time-budget",
        type=float,
        help="Total cumulative training seconds (default: 300 when no schedule is supplied).",
    )
    schedule.add_argument(
        "--epochs",
        type=int,
        help="Train each game exactly once per epoch; learning-rate progress follows games seen.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save a resumable, Hugging Face-ready checkpoint here.",
    )
    parser.add_argument("--resume", type=Path, help="Resume weights, AdamW state, and schedule progress.")
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=1800,
        help="Checkpoint interval in training seconds (default: 30 minutes).",
    )
    parser.add_argument(
        "--snapshot-every",
        type=float,
        default=21600,
        help="Rotating snapshot interval in training seconds (default: 6 hours).",
    )
    parser.add_argument(
        "--keep-snapshots",
        type=int,
        default=12,
        help="Number of rotating snapshots to retain (default: 12).",
    )
    parser.add_argument(
        "--eval-every",
        type=float,
        default=0,
        help="Run fixed validation after this many training seconds; 0 disables periodic evaluation.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        help="Append periodic and final validation records as JSON Lines.",
    )
    return parser.parse_args()


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


args = parse_args()
if args.time_budget is None and args.epochs is None:
    args.time_budget = TIME_BUDGET
if args.resume and not args.output_dir:
    args.output_dir = args.resume
if args.time_budget is not None and args.time_budget <= 0:
    raise SystemExit("--time-budget must be positive")
if args.epochs is not None and args.epochs <= 0:
    raise SystemExit("--epochs must be positive")
if args.checkpoint_every < 0:
    raise SystemExit("--checkpoint-every cannot be negative")
if args.snapshot_every < 0:
    raise SystemExit("--snapshot-every cannot be negative")
if args.keep_snapshots <= 0:
    raise SystemExit("--keep-snapshots must be positive")
if args.eval_every < 0:
    raise SystemExit("--eval-every cannot be negative")

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
target_games = train_loader.corpus.train_games * args.epochs if args.epochs is not None else None
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=model_dim // HEAD_DIM,
    n_kv_head=model_dim // HEAD_DIM,
    n_embd=model_dim,
    window_pattern=WINDOW_PATTERN,
)

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

optimizer = AdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    scalar_lr=SCALAR_LR,
)

training_config = {
    "total_batch_size": TOTAL_BATCH_SIZE,
    "device_batch_size": DEVICE_BATCH_SIZE,
    "embedding_lr": EMBEDDING_LR,
    "unembedding_lr": UNEMBEDDING_LR,
    "matrix_lr": MATRIX_LR,
    "scalar_lr": SCALAR_LR,
    "weight_decay": WEIGHT_DECAY,
    "adam_betas": list(ADAM_BETAS),
    "warmup_ratio": WARMUP_RATIO,
    "warmdown_ratio": WARMDOWN_RATIO,
    "final_lr_frac": FINAL_LR_FRAC,
    "data_dir": str(DATA_DIR.resolve()),
    "length_buckets": list(LENGTH_BUCKETS),
    "validation_games": VAL_GAMES,
    "test_games": TEST_GAMES,
    "schedule": (
        {"mode": "epochs", "epochs": args.epochs, "target_games": target_games}
        if args.epochs is not None
        else {"mode": "time", "target_seconds": args.time_budget}
    ),
}

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
processed_tokens = 0
processed_games = 0
last_checkpoint_time = 0.0
last_snapshot_time = 0.0
last_validation_time = 0.0
if args.resume:
    resume_state = load_checkpoint(args.resume, model, optimizer)
    if resume_state["model_config"] != config.__dict__:
        raise SystemExit("Checkpoint model configuration does not match train.py")
    if resume_state.get("training_config", training_config) != training_config:
        raise SystemExit("Checkpoint optimizer or batch configuration does not match train.py")
    step = int(resume_state["step"])
    processed_tokens = int(resume_state.get("total_tokens", step * TOTAL_BATCH_SIZE))
    processed_games = int(resume_state.get("total_games", 0))
    total_training_time = float(resume_state["training_seconds"])
    smooth_train_loss = float(resume_state.get("smooth_train_loss", 0.0))
    if resume_state.get("dataloader_state") is not None:
        train_loader.load_state_dict(resume_state["dataloader_state"])
    last_checkpoint_time = total_training_time
    if args.snapshot_every > 0:
        last_snapshot_time = math.floor(total_training_time / args.snapshot_every) * args.snapshot_every
    last_validation_time = total_training_time
    print(f"Resumed {args.resume} at step {step:,} ({total_training_time:.1f}s)")

loss_sum_grad_fn = nn.value_and_grad(
    model,
    lambda model, inputs, targets: mx.sum(model(inputs, targets=targets, reduction="none")),
)

if target_games is not None:
    print(f"Epoch target: {args.epochs} ({target_games:,} games total)")
else:
    print(f"Time budget: {args.time_budget}s total")
print(f"Target padded tokens per optimizer step: {TOTAL_BATCH_SIZE:,}")

t_compiled = None


def current_training_state():
    return {
        "step": step,
        "training_seconds": total_training_time,
        "total_tokens": processed_tokens,
        "total_games": processed_games,
        "smooth_train_loss": smooth_train_loss,
        "training_config": training_config,
        "dataloader_state": train_loader.state_dict(),
    }


def record_validation(move_bits, phase):
    if not args.history_file:
        return
    args.history_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "phase": phase,
        "step": step,
        "training_seconds": total_training_time,
        "total_tokens": processed_tokens,
        "total_games": processed_games,
        "num_params": num_params,
    }
    entry["test_move_bits" if phase == "test" else "val_move_bits"] = move_bits
    with args.history_file.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

def training_complete():
    if target_games is not None:
        return processed_games >= target_games
    return step >= STARTUP_EXCLUDE_STEPS and total_training_time >= args.time_budget


while not training_complete():
    t0 = time.time()
    accum_grads = None
    step_loss_sum = 0.0
    step_valid_moves = 0
    step_padded_tokens = 0
    step_games = 0

    while step_padded_tokens < TOTAL_BATCH_SIZE:
        x, y, epoch = next(train_loader)
        loss_sum, grads = loss_sum_grad_fn(model, x, y)
        mx.eval(loss_sum, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Model compiled in {t_compiled - t_data:.1f}s")
        valid_moves = int(mx.sum(y != -1).item())
        step_loss_sum += float(loss_sum.item())
        step_valid_moves += valid_moves
        step_padded_tokens += int(x.shape[0] * x.shape[1])
        step_games += int(x.shape[0])
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda lhs, rhs: lhs + rhs, accum_grads, grads)
        if target_games is not None and processed_games + step_games >= target_games:
            break

    accum_grads = tree_map(lambda grad: grad * (1.0 / max(step_valid_moves, 1)), accum_grads)

    progress = min(
        processed_games / target_games
        if target_games is not None
        else total_training_time / args.time_budget,
        1.0,
    )
    lrm = get_lr_multiplier(progress)
    optimizer.set_lr_multiplier(lrm)
    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), *optimizer.state)

    train_loss_f = step_loss_sum / max(step_valid_moves, 1)
    if train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt
    processed_tokens += step_padded_tokens
    processed_games += step_games

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(step_padded_tokens / dt) if dt > 0 else 0
    remaining = (
        f"{max(0, target_games - processed_games):,} games"
        if target_games is not None
        else f"{max(0.0, args.time_budget - total_training_time):.0f}s"
    )

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining}    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if (
        args.eval_every > 0
        and total_training_time - last_validation_time >= args.eval_every
        and not training_complete()
    ):
        print("\nStarting periodic eval...")
        periodic_bits = evaluate_move_bits(model, tokenizer, FINAL_EVAL_BATCH_SIZE)
        last_validation_time = total_training_time
        record_validation(periodic_bits, "periodic")
        print(f"Periodic val_move_bits at {total_training_time:.1f}s: {periodic_bits:.6f}")
    if (
        args.output_dir
        and args.checkpoint_every > 0
        and total_training_time - last_checkpoint_time >= args.checkpoint_every
    ):
        save_checkpoint(args.output_dir, model, optimizer, config, current_training_state())
        last_checkpoint_time = total_training_time
        print(f"\nCheckpoint saved to {args.output_dir}")
    if (
        args.output_dir
        and args.snapshot_every > 0
        and total_training_time - last_snapshot_time >= args.snapshot_every
    ):
        snapshot_dir = save_rotating_snapshot(
            args.output_dir,
            model,
            optimizer,
            config,
            current_training_state(),
            keep=args.keep_snapshots,
        )
        last_snapshot_time = total_training_time
        print(f"\nRotating snapshot saved to {snapshot_dir}")
print()
t_train = time.time()
training_started = t_compiled if t_compiled is not None else t_train
print(f"Training completed in {t_train - training_started:.1f}s")

total_tokens = processed_tokens
print("Starting final eval...")
print(f"Final eval batch size: {FINAL_EVAL_BATCH_SIZE}")
val_move_bits = evaluate_move_bits(model, tokenizer, FINAL_EVAL_BATCH_SIZE, split="val")
record_validation(val_move_bits, "final")
test_move_bits = evaluate_move_bits(model, tokenizer, FINAL_EVAL_BATCH_SIZE, split="test")
record_validation(test_move_bits, "test")
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

if args.output_dir:
    save_checkpoint(
        args.output_dir,
        model,
        optimizer,
        config,
        current_training_state(),
        metrics={"val_move_bits": val_move_bits, "test_move_bits": test_move_bits},
    )
    print(f"Publishable checkpoint saved to {args.output_dir}")

steady_state_mfu = 0.0
peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_move_bits:    {val_move_bits:.6f}")
print(f"test_move_bits:   {test_move_bits:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"total_games_M:    {processed_games / 1e6:.3f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
