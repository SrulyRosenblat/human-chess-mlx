"""Legal-move-masked inference for the published human-chess MLX model."""

import argparse
from pathlib import Path

import chess
import mlx.core as mx

from chess_model import ChessTokenizer, load_model


MOVE_VOCAB_SIZE = 1968


def board_from_history(tokenizer, token_ids):
    """Reconstruct the exact board while ignoring metadata tokens."""
    board = chess.Board()
    move_start = len(tokenizer.id_to_token) - MOVE_VOCAB_SIZE
    for token_id in token_ids:
        if int(token_id) < move_start:
            continue
        move = chess.Move.from_uci(tokenizer.id_to_token[int(token_id)])
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move in supplied history: {move.uci()}")
        board.push(move)
    return board


def predict_legal_moves(model, tokenizer, token_ids, top_k=10):
    """Return `(uci, logit)` pairs after masking every illegal move."""
    if not token_ids:
        raise ValueError("At least one context token is required")
    board = board_from_history(tokenizer, token_ids)
    context = token_ids[-model.config.sequence_len :]
    logits = model(mx.array([context]))[0, -1]
    mx.eval(logits)

    candidates = []
    for move in board.legal_moves:
        token_id = tokenizer.token_to_id.get(move.uci())
        if token_id is not None:
            candidates.append((move.uci(), float(logits[token_id].item())))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "tokens",
        nargs="+",
        help="Full token history, including metadata and UCI moves.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    tokenizer = ChessTokenizer.from_pretrained(args.checkpoint)
    token_ids = tokenizer.encode_tokens(args.tokens)
    model = load_model(args.checkpoint)
    for move, score in predict_legal_moves(model, tokenizer, token_ids, args.top_k):
        print(f"{move}\t{score:.6f}")


if __name__ == "__main__":
    main()
