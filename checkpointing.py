"""Resumable MLX checkpoints and Hugging Face repository packaging."""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from chess_model import load_weights
from prepare import VOCAB_PATH


ROOT = Path(__file__).resolve().parent


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def refresh_runtime_files(output_dir):
    """Refresh loader and legal-inference code without touching trained weights."""
    output_dir = Path(output_dir)
    shutil.copy2(VOCAB_PATH, output_dir / "vocab.json")
    shutil.copy2(ROOT / "chess_model.py", output_dir / "chess_model.py")
    shutil.copy2(ROOT / "legal_inference.py", output_dir / "legal_inference.py")
    shutil.copy2(ROOT / "NOTICE", output_dir / "NOTICE")
    (output_dir / "requirements.txt").write_text("mlx>=0.30.0\npython-chess>=1.999\n")
    state_path = output_dir / "training_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        metrics_path = output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        (output_dir / "README.md").write_text(_model_card(state, metrics))


def _model_card(state, metrics):
    score = metrics.get("val_move_bits") if metrics else None
    metric_lines = f"- Validation bits per human move: `{score:.6f}`\n" if score is not None else ""
    if "test_move_bits" in metrics:
        metric_lines += f"- Out-of-time test bits per human move: `{metrics['test_move_bits']:.6f}`\n"
    if "legal_top1_accuracy" in metrics:
        metric_lines += f"- Legal-masked top-1 human-move accuracy: `{metrics['legal_top1_accuracy']:.2%}`\n"
    if "legal_top5_accuracy" in metrics:
        metric_lines += f"- Legal-masked top-5 human-move accuracy: `{metrics['legal_top5_accuracy']:.2%}`\n"
    if "raw_top1_accuracy" in metrics:
        metric_lines += f"- Raw exact next-move accuracy: `{metrics['raw_top1_accuracy']:.2%}`\n"
    if "raw_top1_legal_rate" in metrics:
        metric_lines += f"- Raw top-1 legal-move rate: `{metrics['raw_top1_legal_rate']:.2%}`\n"
    if "legal_masked_top1_accuracy" in metrics:
        metric_lines += f"- Legal-masked top-1 accuracy: `{metrics['legal_masked_top1_accuracy']:.2%}`\n"
    if "legal_masked_top5_accuracy" in metrics:
        metric_lines += f"- Legal-masked top-5 accuracy: `{metrics['legal_masked_top5_accuracy']:.2%}`\n"
    return f"""---
license: agpl-3.0
library_name: mlx
pipeline_tag: text-generation
tags:
- chess
- mlx
- autoregressive
- custom-code
datasets:
- Lichess/chess-games
---

# Human Chess MLX

An autoregressive MLX model trained on complete human chess-game histories. Moves are atomic UCI tokens. Metadata and padding may be input context, but training and validation loss are calculated only for human move targets.

The checkpoint was trained on 27,971,437 chronological January 2025 Lichess games (45.6% of one shuffled epoch). Lichess database exports are CC0.

## Configuration

- Context length: `{state['model_config']['sequence_len']}` tokens
- Vocabulary size: `{state['model_config']['vocab_size']}`
- Transformer layers: `{state['model_config']['n_layer']}`
- Embedding width: `{state['model_config']['n_embd']}`
{metric_lines}
Metrics use fixed 8,192-game chronological holdouts. The test split was evaluated only after training stopped.

## Loading

```python
import mlx.core as mx
from chess_model import ChessTokenizer, load_model

model = load_model(".")
tokenizer = ChessTokenizer.from_pretrained(".")
tokens = mx.array([tokenizer.encode_tokens(["BOS"])])
logits = model(tokens)
```

This is a custom MLX architecture, not a Transformers `AutoModelForCausalLM` checkpoint. The repository includes `chess_model.py` for loading.

Legal-move-masked inference is available through `legal_inference.py`; it reconstructs the board from the full supplied move history before scoring only legal continuations.

Source code: https://github.com/SrulyRosenblat/human-chess-mlx

## License

AGPL-3.0. See `LICENSE`.
"""


def save_checkpoint(output_dir, model, optimizer, config, training_state, metrics=None):
    """Save weights, optimizer state, metadata, tokenizer, and loader code."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tmp = output_dir / "model.tmp.safetensors"
    model.save_weights(str(model_tmp))
    os.replace(model_tmp, output_dir / "model.safetensors")

    optimizer_arrays = {}
    optimizer_steps = {}
    for path, value in optimizer.adam_state.items():
        optimizer_arrays[f"{path}.m"] = value["m"]
        optimizer_arrays[f"{path}.v"] = value["v"]
        optimizer_steps[path] = int(value["t"])
    optimizer_tmp = output_dir / "optimizer.tmp.safetensors"
    mx.save_safetensors(str(optimizer_tmp), optimizer_arrays)
    os.replace(optimizer_tmp, output_dir / "optimizer.safetensors")

    state = dict(training_state)
    state["format_version"] = 1
    state["model_config"] = asdict(config)
    state["optimizer_steps"] = optimizer_steps
    _write_json(output_dir / "training_state.json", state)

    hf_config = {
        "architectures": ["GPT"],
        "library_name": "mlx",
        "model_type": "human-chess-mlx",
        "model_config": asdict(config),
        "torch_dtype": "bfloat16",
    }
    _write_json(output_dir / "config.json", hf_config)
    _write_json(output_dir / "metrics.json", metrics or {})
    _write_json(
        output_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "ChessTokenizer",
            "model_max_length": config.sequence_len,
            "pad_token": "PAD",
            "bos_token": "BOS",
            "eos_token": "EOS",
        },
    )
    _write_json(
        output_dir / "special_tokens_map.json",
        {"pad_token": "PAD", "bos_token": "BOS", "eos_token": "EOS"},
    )
    refresh_runtime_files(output_dir)
    (output_dir / "README.md").write_text(_model_card(state, metrics or {}))
    return output_dir


def prune_snapshots(snapshot_root, keep):
    """Remove older numbered snapshots and return the retained paths."""
    if keep <= 0:
        raise ValueError("snapshot retention must be positive")
    snapshot_root = Path(snapshot_root)
    snapshots = sorted(
        (path for path in snapshot_root.glob("step-*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("step-")),
    )
    for expired in snapshots[:-keep]:
        shutil.rmtree(expired)
    return snapshots[-keep:]


def save_rotating_snapshot(
    output_dir, model, optimizer, config, training_state, metrics=None, keep=12
):
    """Save a numbered recovery point and retain the newest ``keep`` snapshots."""
    snapshot_root = Path(output_dir) / "checkpoints"
    snapshot_dir = snapshot_root / f"step-{int(training_state['step']):09d}"
    save_checkpoint(snapshot_dir, model, optimizer, config, training_state, metrics=metrics)
    prune_snapshots(snapshot_root, keep)
    return snapshot_dir


def load_checkpoint(checkpoint_dir, model, optimizer):
    """Restore model and AdamW state, returning scalar training state."""
    checkpoint_dir = Path(checkpoint_dir)
    state = json.loads((checkpoint_dir / "training_state.json").read_text())
    load_weights(model, checkpoint_dir / "model.safetensors")
    arrays = mx.load(str(checkpoint_dir / "optimizer.safetensors"))
    optimizer.adam_state = {}
    for path, step in state["optimizer_steps"].items():
        optimizer.adam_state[path] = {
            "m": arrays[f"{path}.m"],
            "v": arrays[f"{path}.v"],
            "t": int(step),
        }
    mx.eval(model.parameters(), *optimizer.state)
    return state
