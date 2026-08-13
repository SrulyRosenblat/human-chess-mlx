import tempfile
import unittest
from pathlib import Path

import numpy as np

from checkpointing import prune_snapshots
from prepare import MAX_SEQ_LEN, TEST_GAMES, VAL_GAMES, Tokenizer, make_dataloader


class SyntheticCorpus:
    def __init__(self, lengths):
        games = []
        for index, length in enumerate(lengths):
            game = np.full(length, 107 + index, dtype=np.uint16)
            game[:4] = [1, 7, 57, 104]
            game[-1] = 2
            games.append(game)
        self.tokens = np.concatenate(games)
        self.offsets = np.array(
            [0, *np.cumsum([len(game) for game in games])], dtype=np.uint64
        )
        self.games = len(games)
        self.train_games = self.games

    def game(self, index):
        return self.tokens[self.offsets[index] : self.offsets[index + 1]]


class SplitBoundaryCorpus:
    train_games = 10
    val_start = train_games
    test_start = train_games + VAL_GAMES

    def game(self, _index):
        return np.array([1, 7, 57, 104, 107, 2], dtype=np.uint16)


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(vocab_size=2075, move_start=107)

    def test_each_game_appears_exactly_once_per_epoch(self):
        corpus = SyntheticCorpus([6 + index % 240 for index in range(509)])
        loader = make_dataloader(
            self.tokenizer, batch_size=8, seq_len=MAX_SEQ_LEN, split="train", corpus=corpus
        )
        seen = []
        while True:
            _, _, epoch = next(loader)
            if epoch != 1:
                break
            seen.extend(map(int, loader.last_indices))
        self.assertEqual(sorted(seen), list(range(corpus.train_games)))
        self.assertEqual(len(seen), len(set(seen)))

    def test_resume_continues_at_exact_next_batch(self):
        corpus = SyntheticCorpus([10 + index % 180 for index in range(257)])
        original = make_dataloader(
            self.tokenizer, batch_size=4, seq_len=MAX_SEQ_LEN, split="train", corpus=corpus
        )
        for _ in range(5):
            next(original)
        state = original.state_dict()
        next(original)
        expected = original.last_indices.copy()

        resumed = make_dataloader(
            self.tokenizer, batch_size=4, seq_len=MAX_SEQ_LEN, split="train", corpus=corpus
        )
        resumed.load_state_dict(state)
        next(resumed)
        np.testing.assert_array_equal(resumed.last_indices, expected)

    def test_bucket_batches_use_less_than_the_max_padding_budget(self):
        corpus = SyntheticCorpus([30] * 100)
        loader = make_dataloader(
            self.tokenizer, batch_size=8, seq_len=MAX_SEQ_LEN, split="train", corpus=corpus
        )
        inputs, _, _ = next(loader)
        self.assertEqual(inputs.shape[1], 64)
        self.assertLessEqual(inputs.shape[0] * inputs.shape[1], 8 * MAX_SEQ_LEN)

    def test_validation_and_test_ranges_are_disjoint_and_chronological(self):
        corpus = SplitBoundaryCorpus()
        validation = make_dataloader(
            self.tokenizer, batch_size=32, seq_len=MAX_SEQ_LEN, split="val", corpus=corpus
        )
        test = make_dataloader(
            self.tokenizer, batch_size=32, seq_len=MAX_SEQ_LEN, split="test", corpus=corpus
        )
        next(validation)
        next(test)
        self.assertTrue(np.all(validation.last_indices >= corpus.val_start))
        self.assertTrue(np.all(validation.last_indices < corpus.test_start))
        self.assertTrue(np.all(test.last_indices >= corpus.test_start))
        self.assertTrue(np.all(test.last_indices < corpus.test_start + TEST_GAMES))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_retention_keeps_newest_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in [30, 10, 40, 20]:
                (root / f"step-{step:09d}").mkdir()
            retained = prune_snapshots(root, keep=2)
            self.assertEqual([path.name for path in retained], ["step-000000030", "step-000000040"])
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["step-000000030", "step-000000040"],
            )


if __name__ == "__main__":
    unittest.main()
