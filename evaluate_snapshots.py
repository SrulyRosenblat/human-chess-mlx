"""Evaluate recovery snapshots without pausing or mutating the trainer.

The validation sample is deterministic.  Results are written beside each
snapshot as ``snapshot_validation.json`` so a watcher can safely skip work it
has already completed.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import chess
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chess_model import ChessTokenizer, load_model
from prepare import MAX_SEQ_LEN, make_dataloader


MOVE_VOCAB_SIZE = 1968
RESULT_NAME = "snapshot_validation.json"
REQUIRED_FILES = ("config.json", "model.safetensors", "training_state.json", "vocab.json")


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _score_legal_predictions(logits, inputs, targets, tokenizer, totals):
    """Accumulate board-aware metrics for one host-side batch."""
    move_start = len(tokenizer.id_to_token) - MOVE_VOCAB_SIZE

    for row_logits, row_inputs, row_targets in zip(logits, inputs, targets):
        board = chess.Board()
        for position, target in enumerate(row_targets):
            target = int(target)
            if target < move_start:
                continue

            legal_ids = []
            for move in board.legal_moves:
                token_id = tokenizer.token_to_id.get(move.uci())
                if token_id is not None:
                    legal_ids.append(token_id)

            if target not in legal_ids:
                totals["invalid_target_positions"] += 1
                # Do not let one corrupt game invalidate the rest of the sample.
                target_move = chess.Move.from_uci(tokenizer.id_to_token[target])
                if target_move not in board.legal_moves:
                    break
            else:
                scores = row_logits[position]
                raw_prediction = int(np.argmax(scores))
                move_prediction = move_start + int(np.argmax(scores[move_start:]))
                legal_scores = scores[legal_ids]
                legal_prediction = legal_ids[int(np.argmax(legal_scores))]
                top_k = min(5, len(legal_ids))
                top_legal_ids = {
                    legal_ids[index]
                    for index in np.argpartition(legal_scores, -top_k)[-top_k:]
                }

                totals["raw_top1_correct"] += raw_prediction == target
                totals["move_top1_correct"] += move_prediction == target
                totals["raw_top1_legal"] += raw_prediction in legal_ids
                totals["move_top1_legal"] += move_prediction in legal_ids
                totals["legal_top1_correct"] += legal_prediction == target
                totals["legal_top5_correct"] += target in top_legal_ids
                totals["board_positions"] += 1

            board.push(chess.Move.from_uci(tokenizer.id_to_token[target]))


def evaluate_snapshot(checkpoint, games, batch_size, split="val"):
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / "training_state.json").read_text())
    tokenizer = ChessTokenizer.from_pretrained(checkpoint)
    move_start = len(tokenizer.id_to_token) - MOVE_VOCAB_SIZE
    model = load_model(checkpoint)
    loader = make_dataloader(
        type("TokenizerMetadata", (), {"move_start": move_start})(),
        batch_size,
        MAX_SEQ_LEN,
        split,
    )

    totals = {
        "loss_nats": 0.0,
        "move_targets": 0,
        "board_positions": 0,
        "invalid_target_positions": 0,
        "raw_top1_correct": 0,
        "move_top1_correct": 0,
        "raw_top1_legal": 0,
        "move_top1_legal": 0,
        "legal_top1_correct": 0,
        "legal_top5_correct": 0,
    }

    started = time.time()
    evaluated_games = 0
    while evaluated_games < games:
        inputs, targets, _ = next(loader)
        take = min(batch_size, games - evaluated_games)
        inputs = inputs[:take]
        targets = targets[:take]
        logits = model(inputs)
        valid = targets != -1
        safe_targets = mx.where(valid, targets, mx.zeros_like(targets))
        losses = nn.losses.cross_entropy(logits, safe_targets, reduction="none") * valid
        mx.eval(logits, losses, valid)

        totals["loss_nats"] += float(mx.sum(losses).item())
        totals["move_targets"] += int(mx.sum(valid).item())
        _score_legal_predictions(
            np.asarray(logits),
            np.asarray(inputs),
            np.asarray(targets),
            tokenizer,
            totals,
        )
        evaluated_games += take
        mx.clear_cache()

    denominator = max(totals["board_positions"], 1)
    result = {
        "checkpoint": checkpoint.name,
        "step": int(state["step"]),
        "training_games": int(state["total_games"]),
        "training_seconds": float(state["training_seconds"]),
        "split": split,
        "evaluated_games": games,
        "move_targets": totals["move_targets"],
        "invalid_target_positions": totals["invalid_target_positions"],
        "val_bits_per_move": totals["loss_nats"]
        / (math.log(2) * max(totals["move_targets"], 1)),
        "raw_top1_accuracy": totals["raw_top1_correct"] / denominator,
        "move_only_top1_accuracy": totals["move_top1_correct"] / denominator,
        "raw_top1_legal_rate": totals["raw_top1_legal"] / denominator,
        "move_only_top1_legal_rate": totals["move_top1_legal"] / denominator,
        "legal_masked_top1_accuracy": totals["legal_top1_correct"] / denominator,
        "legal_masked_top5_accuracy": totals["legal_top5_correct"] / denominator,
        "evaluation_seconds": time.time() - started,
    }
    result_name = RESULT_NAME if split == "val" else f"snapshot_{split}.json"
    _atomic_json(checkpoint / result_name, result)
    return result


def ready_snapshots(root):
    for checkpoint in sorted(root.glob("step-*")):
        if checkpoint.is_dir() and all((checkpoint / name).is_file() for name in REQUIRED_FILES):
            yield checkpoint


def print_result(result):
    print(
        f"{result['checkpoint']}: bits={result['val_bits_per_move']:.6f}, "
        f"raw_top1={result['raw_top1_accuracy']:.2%}, "
        f"raw_legal={result['raw_top1_legal_rate']:.2%}, "
        f"legal_top1={result['legal_masked_top1_accuracy']:.2%}, "
        f"legal_top5={result['legal_masked_top5_accuracy']:.2%}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.games <= 0 or args.batch_size <= 0 or args.poll_seconds <= 0:
        raise SystemExit("games, batch size, and poll seconds must be positive")

    while True:
        for checkpoint in ready_snapshots(args.snapshot_root):
            result_name = RESULT_NAME if args.split == "val" else f"snapshot_{args.split}.json"
            result_path = checkpoint / result_name
            if result_path.exists():
                continue
            print(f"Evaluating {checkpoint} on {args.games:,} fixed validation games...", flush=True)
            try:
                print_result(evaluate_snapshot(checkpoint, args.games, args.batch_size, args.split))
            except Exception as error:
                # A snapshot can briefly be visible while its atomic files are still being saved.
                print(f"Deferring {checkpoint.name}: {error}", flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
