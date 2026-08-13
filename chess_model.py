"""Standalone MLX model definition shipped with checkpoints."""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 2075
    n_layer: int = 4
    n_head: int = 2
    n_kv_head: int = 2
    n_embd: int = 256
    window_pattern: str = "SSSL"


class ChessTokenizer:
    """Minimal exact-token tokenizer for metadata and atomic UCI move tokens."""

    def __init__(self, vocabulary):
        self.id_to_token = vocabulary["id_to_token"]
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.pad_token_id = self.token_to_id["PAD"]
        self.bos_token_id = self.token_to_id["BOS"]
        self.eos_token_id = self.token_to_id["EOS"]

    @classmethod
    def from_pretrained(cls, checkpoint_dir):
        import json
        from pathlib import Path

        vocabulary = json.loads((Path(checkpoint_dir) / "vocab.json").read_text())
        return cls(vocabulary)

    def encode_tokens(self, tokens):
        return [self.token_to_id[token] for token in tokens]

    def decode_ids(self, token_ids):
        return [self.id_to_token[int(token_id)] for token_id in token_ids]


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
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
        self.wte.weight = mx.random.normal(self.wte.weight.shape).astype(mx.bfloat16)
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
        mapping = {"L": config.sequence_len, "S": config.sequence_len // 2}
        windows = [mapping[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        windows[-1] = config.sequence_len
        return windows

    def _get_masks(self, seq_len):
        dtype = self.wte.weight.dtype
        for window_size in set(self.window_sizes):
            key = (seq_len, window_size, dtype)
            if key not in self._mask_cache:
                self._mask_cache[key] = (
                    create_additive_causal_mask(seq_len, dtype)
                    if window_size >= seq_len
                    else create_sliding_window_mask(seq_len, window_size, dtype)
                )
        return [self._mask_cache[(seq_len, size, dtype)] for size in self.window_sizes]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)
        x = norm(self.wte(idx))
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i].astype(x.dtype) * x + self.x0_lambdas[i].astype(x.dtype) * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        logits = self.lm_head(norm(x)).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)
        if targets is None:
            return logits
        valid = targets != -1
        safe_targets = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, safe_targets, reduction="none") * valid
        if reduction == "none":
            return ce
        return mx.sum(ce) / mx.maximum(mx.sum(valid), 1)


def load_model(checkpoint_dir):
    """Load a published MLX checkpoint and its configuration."""
    import json
    from pathlib import Path

    checkpoint_dir = Path(checkpoint_dir)
    raw = json.loads((checkpoint_dir / "config.json").read_text())
    config = GPTConfig(**raw["model_config"])
    model = GPT(config)
    load_weights(model, checkpoint_dir / "model.safetensors")
    mx.eval(model.parameters())
    return model


def load_weights(model, weights_path):
    """Load flattened weights without MLX confusing numeric dict keys for list indices."""
    current = dict(tree_flatten(model.parameters()))
    loaded = mx.load(str(weights_path))
    if current.keys() != loaded.keys():
        missing = sorted(current.keys() - loaded.keys())
        extra = sorted(loaded.keys() - current.keys())
        raise ValueError(f"Checkpoint key mismatch; missing={missing}, extra={extra}")
    for path, value in loaded.items():
        if current[path].shape != value.shape:
            raise ValueError(f"Shape mismatch for {path}: {value.shape} != {current[path].shape}")
        parts = path.split(".")
        target = model
        for part in parts[:-1]:
            if isinstance(target, list):
                target = target[int(part)]
            elif isinstance(target, dict):
                target = target[part]
            else:
                target = getattr(target, part)
        final = parts[-1]
        if isinstance(target, list):
            target[int(final)] = value
        elif isinstance(target, dict):
            target[final] = value
        else:
            setattr(target, final, value)
    return model
