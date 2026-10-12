"""Frozen specification checks for the Round 14B suffix-write task."""
import unittest

import numpy as np

try:
    from .round14b_task import (
        ANSWER_MASK_POSITIONS,
        BLOCK_LENGTH,
        BLOCKS,
        FILL_ID,
        IGNORE,
        INPUT_LENGTH,
        KEY_IDS,
        PREFIX_LENGTH,
        READ_ID,
        START_ID,
        SUFFIX_LENGTH,
        VALUE_IDS,
        WRITE_ID,
        answer_mask,
        make_episodes,
        sample_batch,
        score_suffix,
    )
except ImportError:
    from round14b_task import (
        ANSWER_MASK_POSITIONS,
        BLOCK_LENGTH,
        BLOCKS,
        FILL_ID,
        IGNORE,
        INPUT_LENGTH,
        KEY_IDS,
        PREFIX_LENGTH,
        READ_ID,
        START_ID,
        SUFFIX_LENGTH,
        VALUE_IDS,
        WRITE_ID,
        answer_mask,
        make_episodes,
        sample_batch,
        score_suffix,
    )


class TestRound14BTask(unittest.TestCase):
    def test_episode_roles_causality_and_prompt_has_no_values(self):
        episodes = make_episodes(12, np.random.default_rng(14017))
        self.assertEqual(len(episodes), 12)
        for episode in episodes:
            prompt = episode["prompt"]
            suffix = episode["suffix"]
            self.assertEqual(len(prompt), PREFIX_LENGTH)
            self.assertEqual(len(suffix), SUFFIX_LENGTH)
            self.assertEqual(prompt[0], START_ID)
            self.assertEqual(set(prompt[1:5]), set(KEY_IDS))
            self.assertEqual(prompt[5:], [FILL_ID, FILL_ID, FILL_ID])
            self.assertTrue(set(prompt).isdisjoint(VALUE_IDS))

            for start in range(0, SUFFIX_LENGTH, BLOCK_LENGTH):
                block = suffix[start:start + BLOCK_LENGTH]
                self.assertEqual(block[0], WRITE_ID)
                self.assertIn(block[1], KEY_IDS)
                self.assertIn(block[2], KEY_IDS)
                self.assertNotEqual(block[1], block[2])
                self.assertIn(block[3], VALUE_IDS)
                self.assertIn(block[4], VALUE_IDS)
                self.assertEqual(block[5:7], [FILL_ID, FILL_ID])
                self.assertEqual(block[7], READ_ID)
                self.assertIn(block[8], (block[1], block[2]))
                self.assertEqual(block[9], block[3] if block[8] == block[1] else block[4])
                self.assertEqual(block[10], FILL_ID)

    def test_training_arrays_align_full_suffix_and_answer_mask(self):
        count = 5
        inputs, labels = sample_batch(count, np.random.default_rng(20261015))
        self.assertEqual(inputs.shape, (count, INPUT_LENGTH))
        self.assertEqual(labels.shape, inputs.shape)
        self.assertTrue(np.all(labels[:, :PREFIX_LENGTH - 1] == IGNORE))
        self.assertEqual(int((labels[:, PREFIX_LENGTH - 1:] != IGNORE).sum()), count * SUFFIX_LENGTH)
        mask = answer_mask()
        self.assertEqual(mask.shape, (INPUT_LENGTH,))
        self.assertEqual(np.flatnonzero(mask).tolist(), list(ANSWER_MASK_POSITIONS))
        self.assertEqual(int(mask.sum()), BLOCKS)
        for row_inputs, row_labels in zip(inputs, labels):
            suffix_targets = row_labels[PREFIX_LENGTH - 1:]
            self.assertEqual(len(suffix_targets), SUFFIX_LENGTH)
            self.assertEqual(int(row_inputs[0]), START_ID)
            self.assertEqual(set(row_inputs[1:5].tolist()), set(KEY_IDS))
            self.assertTrue(set(row_inputs[5:PREFIX_LENGTH].tolist()).issubset({FILL_ID}))
            np.testing.assert_array_equal(row_inputs[PREFIX_LENGTH:], suffix_targets[:-1])
            for block_index in range(BLOCKS):
                block = suffix_targets[block_index * BLOCK_LENGTH:(block_index + 1) * BLOCK_LENGTH]
                self.assertEqual(int(block[0]), WRITE_ID)
                self.assertEqual(int(block[7]), READ_ID)

    def test_score_suffix_applies_only_valid_completed_writes(self):
        blocks = [
            [WRITE_ID, 0, 1, 4, 5, FILL_ID, FILL_ID, READ_ID, 0, 4, FILL_ID],
            # Duplicate keys invalidate this write; key 2 is still absent.
            [WRITE_ID, 2, 2, 6, 7, FILL_ID, FILL_ID, READ_ID, 2, 6, FILL_ID],
            # A later valid write overwrites key 0's value from 4 to 7.
            [WRITE_ID, 0, 3, 7, 4, FILL_ID, FILL_ID, READ_ID, 0, 7, FILL_ID],
            # An invalid marker retains earlier memory; malformed slots stay in syntax denominator.
            [0, 1, 2, 4, 5, FILL_ID, FILL_ID, READ_ID, 1, 5, 99],
        ]
        metrics = score_suffix(np.asarray(blocks, dtype=np.int64).reshape(-1))
        self.assertEqual(metrics, {
            "syntax_correct": 42,
            "syntax_count": 4 * BLOCK_LENGTH,
            "query_valid": 2,
            "query_count": 4,
            "answer_correct": 2,
            "answer_count": 4,
        })

    def test_old_key_after_invalid_write_is_an_invalid_query(self):
        blocks = [
            [WRITE_ID, 0, 1, 4, 5, FILL_ID, FILL_ID, READ_ID, 0, 4, FILL_ID],
            # Key 0 remains in memory, but this malformed current write has no
            # query support, so its old-key query and answer both fail.
            [WRITE_ID, 2, 2, 6, 7, FILL_ID, FILL_ID, READ_ID, 0, 4, FILL_ID],
        ]
        metrics = score_suffix(np.asarray(blocks, dtype=np.int64).reshape(-1))
        self.assertEqual(metrics, {
            "syntax_correct": 2 * BLOCK_LENGTH,
            "syntax_count": 2 * BLOCK_LENGTH,
            "query_valid": 1,
            "query_count": 2,
            "answer_correct": 1,
            "answer_count": 2,
        })

    def test_score_suffix_rejects_partial_blocks(self):
        with self.assertRaises(ValueError):
            score_suffix([WRITE_ID] * (BLOCK_LENGTH + 1))


if __name__ == "__main__":
    unittest.main()
