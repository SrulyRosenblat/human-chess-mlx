"""Memory-mapped human-chess corpus, batching, splits, and evaluation."""

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Corpus and evaluation constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256
TIME_BUDGET = 300
EVAL_GAMES = 8192

ROOT = Path(__file__).resolve().parent.parent
NEW_DATA_DIR = ROOT / "data" / "processed" / "lichess_hf_2025-01_temporal"
LEGACY_DATA_DIR = ROOT / "data" / "processed" / "lichess_2016-06"
new_data_ready = all(
    (NEW_DATA_DIR / name).is_file()
    for name in ("tokens.bin", "offsets.bin", "vocab.json", "splits.json")
)
DATA_DIR = Path(os.environ.get("CHESS_DATA_DIR", NEW_DATA_DIR if new_data_ready else LEGACY_DATA_DIR))
TOKENS_PATH = DATA_DIR / "tokens.bin"
OFFSETS_PATH = DATA_DIR / "offsets.bin"
VOCAB_PATH = DATA_DIR / "vocab.json"
SPLITS_PATH = DATA_DIR / "splits.json"
SPLIT_METADATA = json.loads(SPLITS_PATH.read_text())["splits"] if SPLITS_PATH.is_file() else None
VAL_GAMES = int(SPLIT_METADATA["val"]["games"]) if SPLIT_METADATA else 100_000
TEST_GAMES = int(SPLIT_METADATA["test"]["games"]) if SPLIT_METADATA else 100_000
LENGTH_BUCKETS = (64, 96, 128, 160, 192, 224, 256)


class Tokenizer:
    """Vocabulary metadata wrapper for the already-tokenized chess corpus."""

    def __init__(self, vocab_size: int, move_start: int):
        self.vocab_size = vocab_size
        self.move_start = move_start

    @classmethod
    def from_directory(cls, _tokenizer_dir=None):
        vocab = json.loads(VOCAB_PATH.read_text())
        vocab_size = int(vocab["size"])
        move_start = vocab_size - 1968
        labels = vocab["id_to_token"]
        assert labels[0] == "PAD"
        assert labels[move_start] == "a1a2"
        return cls(vocab_size, move_start)

    def get_vocab_size(self):
        return self.vocab_size


class ChessCorpus:
    """Memory-mapped games with chronological train/validation/test splits."""

    def __init__(self):
        self.tokens = np.memmap(TOKENS_PATH, dtype="<u2", mode="r")
        self.offsets = np.memmap(OFFSETS_PATH, dtype="<u8", mode="r")
        self.games = len(self.offsets) - 1
        assert self.games > VAL_GAMES + TEST_GAMES
        assert int(self.offsets[-1]) == len(self.tokens)
        if SPLIT_METADATA:
            self.train_games = int(SPLIT_METADATA["train"]["games"])
            self.val_start = int(SPLIT_METADATA["val"]["start_game"])
            self.test_start = int(SPLIT_METADATA["test"]["start_game"])
            assert self.val_start == self.train_games
            assert self.test_start == self.val_start + VAL_GAMES
            assert int(SPLIT_METADATA["test"]["end_game_exclusive"]) == self.games
        else:
            self.train_games = self.games - VAL_GAMES - TEST_GAMES
            self.val_start = self.train_games
            self.test_start = self.train_games + VAL_GAMES

    def game(self, index: int):
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.tokens[start:end]


_CORPUS = None


def get_corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = ChessCorpus()
    return _CORPUS


def _make_row(game, move_start, seq_len=MAX_SEQ_LEN):
    """Pad one complete game and mask every non-move prediction target."""
    row = np.zeros(seq_len + 1, dtype=np.int32)
    usable = min(len(game), seq_len + 1)
    row[:usable] = game[:usable]
    inputs = row[:-1]
    targets = row[1:]
    valid = (targets >= move_start) & (np.arange(seq_len) < usable - 1)
    targets = np.where(valid, targets, -1).astype(np.int32, copy=False)
    return inputs, targets


def _build_train_buckets(corpus, seq_len, chunk_size=1_000_000):
    """Index every training game once using bounded temporary memory."""
    bucket_lengths = tuple(length for length in LENGTH_BUCKETS if length <= seq_len)
    if not bucket_lengths or bucket_lengths[-1] != seq_len:
        bucket_lengths += (seq_len,)
    chunks = [[] for _ in bucket_lengths]

    for start in range(0, corpus.train_games, chunk_size):
        end = min(start + chunk_size, corpus.train_games)
        lengths = np.asarray(corpus.offsets[start + 1 : end + 1] - corpus.offsets[start:end])
        input_lengths = np.minimum(np.maximum(lengths - 1, 1), seq_len)
        bucket_ids = np.searchsorted(bucket_lengths, input_lengths, side="left")
        for bucket_id in range(len(bucket_lengths)):
            local = np.flatnonzero(bucket_ids == bucket_id)
            if len(local):
                chunks[bucket_id].append((local + start).astype(np.uint32))

    buckets = tuple(
        np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)
        for parts in chunks
    )
    assert sum(map(len, buckets)) == corpus.train_games
    return bucket_lengths, buckets


class ChessDataLoader:
    """Deterministic whole-game loader with resumable no-replacement epochs."""

    def __init__(self, tokenizer, batch_size, seq_len, split, corpus=None):
        assert seq_len == MAX_SEQ_LEN
        assert split in {"train", "val", "test"}
        self.tokenizer = tokenizer
        self.base_batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.corpus = corpus or get_corpus()
        self.last_indices = None

        if split == "train":
            self.bucket_lengths, self.buckets = _build_train_buckets(self.corpus, seq_len)
            max_padded_tokens = batch_size * seq_len
            self.bucket_batch_sizes = tuple(
                max(1, math.ceil(max_padded_tokens / length)) for length in self.bucket_lengths
            )
            self.epoch = 1
            self._start_epoch(self.epoch)
        else:
            self.rng = np.random.default_rng(314159 if split == "val" else 271828)

    def __iter__(self):
        return self

    def _start_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = np.random.default_rng(np.random.SeedSequence([42, self.epoch]))
        self.orders = tuple(rng.permutation(bucket) for bucket in self.buckets)
        batch_counts = [
            math.ceil(len(order) / batch_size)
            for order, batch_size in zip(self.orders, self.bucket_batch_sizes)
        ]
        self.schedule = np.repeat(np.arange(len(self.orders), dtype=np.uint8), batch_counts)
        rng.shuffle(self.schedule)
        self.bucket_positions = [0] * len(self.orders)
        self.batch_position = 0

    def _next_train_indices(self):
        if self.batch_position >= len(self.schedule):
            self._start_epoch(self.epoch + 1)
        bucket_id = int(self.schedule[self.batch_position])
        self.batch_position += 1
        start = self.bucket_positions[bucket_id]
        end = min(start + self.bucket_batch_sizes[bucket_id], len(self.orders[bucket_id]))
        self.bucket_positions[bucket_id] = end
        return bucket_id, self.orders[bucket_id][start:end]

    def __next__(self):
        if self.split == "train":
            bucket_id, indices = self._next_train_indices()
            batch_seq_len = self.bucket_lengths[bucket_id]
            epoch = self.epoch
        else:
            split_start = self.corpus.val_start if self.split == "val" else self.corpus.test_start
            split_games = VAL_GAMES if self.split == "val" else TEST_GAMES
            indices = split_start + self.rng.integers(0, split_games, size=self.base_batch_size)
            batch_seq_len = self.seq_len
            epoch = 0

        self.last_indices = np.asarray(indices)
        inputs = np.empty((len(indices), batch_seq_len), dtype=np.int32)
        targets = np.empty_like(inputs)
        for row_index, game_index in enumerate(indices):
            inputs[row_index], targets[row_index] = _make_row(
                self.corpus.game(int(game_index)), self.tokenizer.move_start, batch_seq_len
            )
        return mx.array(inputs), mx.array(targets), epoch

    def state_dict(self):
        if self.split != "train":
            return None
        return {
            "format_version": 1,
            "epoch": self.epoch,
            "batch_position": self.batch_position,
            "bucket_positions": list(self.bucket_positions),
            "bucket_lengths": list(self.bucket_lengths),
        }

    def load_state_dict(self, state):
        if self.split != "train":
            raise ValueError("validation loader state is not resumable")
        if state.get("format_version") != 1:
            raise ValueError("unsupported dataloader checkpoint format")
        if tuple(state["bucket_lengths"]) != self.bucket_lengths:
            raise ValueError("checkpoint length buckets do not match the loader")
        self._start_epoch(int(state["epoch"]))
        positions = [int(value) for value in state["bucket_positions"]]
        if len(positions) != len(self.orders):
            raise ValueError("checkpoint bucket count does not match the loader")
        if any(position < 0 or position > len(order) for position, order in zip(positions, self.orders)):
            raise ValueError("checkpoint contains an invalid bucket position")
        batch_position = int(state["batch_position"])
        if not 0 <= batch_position <= len(self.schedule):
            raise ValueError("checkpoint contains an invalid batch position")
        self.bucket_positions = positions
        self.batch_position = batch_position


def make_dataloader(tokenizer, batch_size, seq_len, split, corpus=None):
    return ChessDataLoader(tokenizer, batch_size, seq_len, split, corpus=corpus)


def evaluate_move_bits(model, tokenizer, batch_size, split="val"):
    """Cross-entropy in bits per human move on a fixed holdout sample."""
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, split)
    steps = math.ceil(EVAL_GAMES / batch_size)
    total_nats = 0.0
    total_moves = 0

    for _ in range(steps):
        x, y, _ = next(loader)
        losses = model(x, y, reduction="none")
        valid = y != -1
        total_nats += float(mx.sum(losses).item())
        total_moves += int(mx.sum(valid).item())
        mx.clear_cache()

    if total_moves == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_moves)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_directory()
    corpus = get_corpus()
    print(f"Data: {DATA_DIR}")
    print(f"Games: {corpus.games:,}")
    print(f"Train games: {corpus.train_games:,}")
    print(f"Validation games: {VAL_GAMES:,}")
    print(f"Test games: {TEST_GAMES:,}")
    print(f"Evaluation games per run: {EVAL_GAMES:,}")
    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(f"Move token start: {tokenizer.move_start}")
    print(f"Context: {MAX_SEQ_LEN}")
    loader = make_dataloader(tokenizer, 4, MAX_SEQ_LEN, "train")
    x, y, _ = next(loader)
    assert x.shape == y.shape
    assert x.shape[1] in loader.bucket_lengths
    assert x.shape[0] * x.shape[1] <= 5 * MAX_SEQ_LEN
    assert int(mx.sum(y != -1).item()) > 0
    print("Harness check passed.")
