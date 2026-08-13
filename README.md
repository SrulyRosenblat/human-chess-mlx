# Human Chess MLX

A 14.6M-parameter language model that predicts how humans play chess. It reads
the full game history plus both players' Elo buckets and time control, then
scores the next move. Moves are atomic UCI tokens and a deterministic legality
mask restricts inference to moves that can actually be played.

The trained weights are available on
[Hugging Face](https://huggingface.co/sruly/human-chess-mlx).

## Results

The released checkpoint was trained on 27,971,437 chronological January 2025
Lichess games. Metrics use fixed 8,192-game holdouts from later in the month;
the test split was evaluated only after training stopped.

| Metric | Validation | Out-of-time test |
|---|---:|---:|
| Bits per human move | 4.099 | 4.115 |
| Raw exact next-move accuracy | 27.68% | 27.39% |
| Raw top-1 move is legal | 87.29% | 87.20% |
| Legal-masked top-1 accuracy | 30.97% | 30.73% |
| Legal-masked top-5 accuracy | 66.82% | 66.52% |

Exact next-move prediction is intentionally harder than finding a good chess
move: several moves can be reasonable, while the target is the particular move
one human selected.

## Model

- 6 transformer layers, width 384, 3 attention heads
- 14.6M parameters
- 256-token context window
- 2,075-token vocabulary, including 1,968 atomic UCI moves
- Alternating 128-token sliding and 256-token full causal attention
- Elo, time-control, result, and boundary tokens
- Move-only loss: metadata and padding provide context but are not targets
- Complete games per sample, with no attention across game boundaries

Each stored game has this form:

```text
BOS WHITE_ELO_1625 BLACK_ELO_1675 TC_BLITZ
e2e4 e7e5 g1f3 b8c6 ... RESULT_WHITE EOS
```

## Run the model

Requirements: Apple Silicon, Python 3.10–3.13, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SrulyRosenblat/human-chess-mlx.git
cd human-chess-mlx
uv sync

hf download sruly/human-chess-mlx --local-dir model

uv run legal_inference.py model \
  BOS WHITE_ELO_1625 BLACK_ELO_1675 TC_BLITZ \
  e2e4 e7e5 g1f3 b8c6 \
  --top-k 10
```

`legal_inference.py` reconstructs the board from the supplied history, evaluates
the model once, removes illegal candidates, and returns the highest-scoring
legal continuations.

To call it from Python:

```python
from chess_model import ChessTokenizer, load_model
from legal_inference import predict_legal_moves

checkpoint = "model"
tokenizer = ChessTokenizer.from_pretrained(checkpoint)
model = load_model(checkpoint)

history = [
    "BOS", "WHITE_ELO_1625", "BLACK_ELO_1675", "TC_BLITZ",
    "e2e4", "e7e5", "g1f3", "b8c6",
]
token_ids = tokenizer.encode_tokens(history)
print(predict_legal_moves(model, tokenizer, token_ids, top_k=10))
```

## Train

`prepare.py` expects a tokenized corpus containing `tokens.bin`, `offsets.bin`,
`vocab.json`, and `splits.json`. Set `CHESS_DATA_DIR` to its directory. The
split manifest must define chronological train, validation, and test ranges.

```bash
CHESS_DATA_DIR=/path/to/tokenized-games uv run train.py \
  --epochs 1 \
  --output-dir artifacts/human-chess-mlx \
  --checkpoint-every 1800 \
  --snapshot-every 21600 \
  --keep-snapshots 12
```

Training is deterministic and shuffled without replacement. Length bucketing
reduces padding, checkpoints preserve the exact next-game cursor, and rotating
snapshots provide recovery points. Resume without repeating or skipping games:

```bash
CHESS_DATA_DIR=/path/to/tokenized-games uv run train.py \
  --epochs 1 \
  --resume artifacts/human-chess-mlx \
  --checkpoint-every 1800 \
  --snapshot-every 21600 \
  --keep-snapshots 12
```

Evaluate saved snapshots independently from training:

```bash
uv run evaluate_snapshots.py artifacts/human-chess-mlx/checkpoints \
  --games 8192 --batch-size 16
```

## Code map

- `chess_model.py` — standalone model, tokenizer wrapper, and checkpoint loader
- `legal_inference.py` — board reconstruction and legal-move-masked prediction
- `train.py` — model training and optimizer schedule
- `prepare.py` — memory-mapped corpus, splits, batching, and move-only evaluation
- `checkpointing.py` — resumable and Hugging Face-ready checkpoints
- `evaluate_snapshots.py` — loss, exact accuracy, and legality metrics
- `upload_hf.py` — filtered model upload without optimizer or training data
- `test_training_reliability.py` — split, bucketing, resume, and snapshot tests

This implementation started from Trevin Peterson's MLX port of Andrej
Karpathy's autoresearch training harness and was adapted into the chess model
and data pipeline here.

The model weights are AGPL-3.0 on Hugging Face. This source repository has no
declared license.
