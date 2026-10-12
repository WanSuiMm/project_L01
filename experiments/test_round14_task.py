"""Specification checks for the Round 14 associative-recall episodes."""
import unittest

import numpy as np

try:
    from .round14_train import (
        IGNORE,
        KEY_IDS,
        PREFIX_LENGTH,
        START_ID,
        VALUE_IDS,
        sample_batch,
        task_spec,
    )
except ImportError:
    from round14_train import (
        IGNORE,
        KEY_IDS,
        PREFIX_LENGTH,
        START_ID,
        VALUE_IDS,
        sample_batch,
        task_spec,
    )


class TestRound14Task(unittest.TestCase):
    def test_samples_preserve_episode_mappings_and_causal_boundaries(self):
        ids, labels = sample_batch(
            12, np.random.default_rng(14017), suffix_length=130
        )
        spec = task_spec()
        self.assertEqual(spec["vocab_size"], 17)
        self.assertEqual(spec["start_id"], 2)
        self.assertEqual(ids.shape, (12, PREFIX_LENGTH + 130))
        self.assertTrue(np.all(labels[:, :PREFIX_LENGTH] == IGNORE))
        self.assertTrue(np.all(ids[:, PREFIX_LENGTH] == START_ID))

        for row_ids, row_labels in zip(ids, labels):
            mapping = {
                int(key): int(value)
                for key, value in row_ids[:PREFIX_LENGTH].reshape(8, 2)
            }
            self.assertEqual(set(mapping), set(KEY_IDS))
            self.assertTrue(set(mapping.values()).issubset(set(VALUE_IDS)))

            suffix = row_labels[PREFIX_LENGTH:]
            self.assertTrue(set(suffix[0::2]).issubset(set(KEY_IDS)))
            self.assertTrue(set(suffix[1::2]).issubset(set(VALUE_IDS)))
            for offset in range(0, len(suffix), 2):
                query = int(suffix[offset])
                answer = int(suffix[offset + 1])
                self.assertEqual(answer, mapping[query])

            # START predicts the first query; that query is then the input that
            # predicts the first mapped answer at target position 17.
            self.assertIn(int(row_labels[PREFIX_LENGTH]), KEY_IDS)
            self.assertEqual(row_ids[PREFIX_LENGTH + 1], row_labels[PREFIX_LENGTH])
            self.assertIn(int(row_labels[PREFIX_LENGTH + 1]), VALUE_IDS)

    def test_same_token_bag_can_require_different_first_answers(self):
        keys = np.asarray(KEY_IDS, dtype=np.int64)
        values = np.asarray(VALUE_IDS, dtype=np.int64)
        prefix_a = np.stack((keys, values), axis=-1).reshape(-1)
        shifted_values = np.roll(values, 1)
        prefix_b = np.stack((keys, shifted_values), axis=-1).reshape(-1)

        # The two prefixes contain the same multiset of key and value tokens,
        # yet the shared first query (key 0) has a different correct answer.
        self.assertEqual(sorted(prefix_a.tolist()), sorted(prefix_b.tolist()))
        self.assertNotEqual(prefix_a.tolist(), prefix_b.tolist())
        first_query = int(keys[0])
        answer_a = int(values[0])
        answer_b = int(shifted_values[0])
        self.assertNotEqual(answer_a, answer_b)

        labels_a = np.concatenate((np.full(PREFIX_LENGTH, IGNORE), [first_query, answer_a]))
        labels_b = np.concatenate((np.full(PREFIX_LENGTH, IGNORE), [first_query, answer_b]))
        self.assertEqual(labels_a[PREFIX_LENGTH], labels_b[PREFIX_LENGTH])
        self.assertNotEqual(labels_a[PREFIX_LENGTH + 1], labels_b[PREFIX_LENGTH + 1])


if __name__ == "__main__":
    unittest.main()
