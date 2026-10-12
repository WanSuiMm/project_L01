"""Frozen suffix-write recall task and episode sampler for L01 Round 14B."""
from __future__ import annotations

from typing import Any

import numpy as np


KEY_IDS = (0, 1, 2, 3)
VALUE_IDS = (4, 5, 6, 7)
START_ID = 8
WRITE_ID = 9
READ_ID = 10
FILL_ID = 11
PREFIX_LENGTH = 8
BLOCK_LENGTH = 11
BLOCKS = 16
SUFFIX_LENGTH = BLOCK_LENGTH * BLOCKS
INPUT_LENGTH = PREFIX_LENGTH + SUFFIX_LENGTH - 1
IGNORE = -100

MODEL_SPEC = {
    "vocab_size": 12,
    "d_model": 128,
    "n_layers": 2,
    "max_context": 184,
    "ff_multiplier": 4,
}
FIXED_SEEDS = [11, 23, 37, 51, 71]

# A target suffix token at offset j is predicted by the logit at input offset
# PREFIX_LENGTH - 1 + j. Answers occupy block offset 9.
ANSWER_SUFFIX_OFFSETS = tuple(block * BLOCK_LENGTH + 9 for block in range(BLOCKS))
ANSWER_MASK_POSITIONS = tuple(
    PREFIX_LENGTH - 1 + offset for offset in ANSWER_SUFFIX_OFFSETS
)
RANDOM_ROLE_SUFFIX_OFFSETS = {
    "key_a": tuple(block * BLOCK_LENGTH + 1 for block in range(BLOCKS)),
    "key_b": tuple(block * BLOCK_LENGTH + 2 for block in range(BLOCKS)),
    "value_a": tuple(block * BLOCK_LENGTH + 3 for block in range(BLOCKS)),
    "value_b": tuple(block * BLOCK_LENGTH + 4 for block in range(BLOCKS)),
    "query": tuple(block * BLOCK_LENGTH + 8 for block in range(BLOCKS)),
}
RANDOM_ROLE_LOGIT_POSITIONS = {
    role: tuple(PREFIX_LENGTH - 1 + offset for offset in offsets)
    for role, offsets in RANDOM_ROLE_SUFFIX_OFFSETS.items()
}


def answer_mask(sequence_length: int = INPUT_LENGTH) -> np.ndarray:
    """Return the logit-position mask for teacher-forced answer validation."""
    sequence_length = int(sequence_length)
    if sequence_length != INPUT_LENGTH:
        raise ValueError(f"Expected input length {INPUT_LENGTH}, got {sequence_length}")
    mask = np.zeros(sequence_length, dtype=np.bool_)
    mask[list(ANSWER_MASK_POSITIONS)] = True
    return mask


def task_spec() -> dict[str, Any]:
    """Describe the fixed token roles, sequence layout, and model dimensions."""
    return {
        "name": "suffix_write_delayed_recall",
        "key_ids": list(KEY_IDS),
        "value_ids": list(VALUE_IDS),
        "start_id": START_ID,
        "write_id": WRITE_ID,
        "read_id": READ_ID,
        "fill_id": FILL_ID,
        "prefix_length": PREFIX_LENGTH,
        "block_length": BLOCK_LENGTH,
        "blocks": BLOCKS,
        "suffix_length": SUFFIX_LENGTH,
        "input_length": INPUT_LENGTH,
        "vocab_size": MODEL_SPEC["vocab_size"],
    }


def _sample_arrays(
    count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    key_ids = np.asarray(KEY_IDS, dtype=np.int64)
    value_ids = np.asarray(VALUE_IDS, dtype=np.int64)

    # Every prefix has a fresh uniform permutation and contains no value tokens.
    key_ranks = rng.permuted(
        np.tile(np.arange(len(key_ids), dtype=np.int64), (count, 1)), axis=1
    )
    prompts = np.full((count, PREFIX_LENGTH), FILL_ID, dtype=np.int64)
    prompts[:, 0] = START_ID
    prompts[:, 1:1 + len(key_ids)] = key_ids[key_ranks]

    block_shape = (count, BLOCKS)
    key_a_rank = rng.integers(0, len(key_ids), size=block_shape)
    # A uniform nonzero cyclic offset samples each distinct ordered pair equally.
    key_b_rank = (key_a_rank + rng.integers(1, len(key_ids), size=block_shape)) % len(key_ids)
    key_a = key_ids[key_a_rank]
    key_b = key_ids[key_b_rank]
    value_a = value_ids[rng.integers(0, len(value_ids), size=block_shape)]
    value_b = value_ids[rng.integers(0, len(value_ids), size=block_shape)]
    query_b = rng.integers(0, 2, size=block_shape).astype(np.bool_)
    query = np.where(query_b, key_b, key_a)
    answer = np.where(query_b, value_b, value_a)

    blocks = np.empty((count, BLOCKS, BLOCK_LENGTH), dtype=np.int64)
    blocks[:, :, 0] = WRITE_ID
    blocks[:, :, 1] = key_a
    blocks[:, :, 2] = key_b
    blocks[:, :, 3] = value_a
    blocks[:, :, 4] = value_b
    blocks[:, :, 5] = FILL_ID
    blocks[:, :, 6] = FILL_ID
    blocks[:, :, 7] = READ_ID
    blocks[:, :, 8] = query
    blocks[:, :, 9] = answer
    blocks[:, :, 10] = FILL_ID
    suffixes = blocks.reshape(count, SUFFIX_LENGTH)
    return prompts, suffixes


def make_episodes(
    count: int, rng: np.random.Generator
) -> list[dict[str, list[int]]]:
    """Sample complete prompts and suffixes for evaluation or reproducibility."""
    prompts, suffixes = _sample_arrays(count, rng)
    return [
        {"prompt": prompt.tolist(), "suffix": suffix.tolist()}
        for prompt, suffix in zip(prompts, suffixes)
    ]


def sample_batch(
    count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal inputs and full-suffix targets with prompt logits ignored.

    Inputs are ``prompt + suffix[:-1]``. The first seven logits are masked;
    the last prompt token predicts suffix[0], then each suffix token predicts
    the next one. Thus the 183 input positions carry exactly 176 training labels.
    """
    prompts, suffixes = _sample_arrays(count, rng)
    input_ids = np.concatenate((prompts, suffixes[:, :-1]), axis=1)
    labels = np.full(input_ids.shape, IGNORE, dtype=np.int64)
    labels[:, PREFIX_LENGTH - 1:] = suffixes
    if input_ids.shape[1] != INPUT_LENGTH:
        raise AssertionError("Unexpected Round 14B input length")
    return input_ids, labels


def score_suffix(suffix: list[int] | np.ndarray) -> dict[str, int]:
    """Count role syntax and strict online write/read correctness.

    Each block contributes all eleven syntax slots and one query/answer pair
    to the denominators, including malformed slots and invalid queries. A
    write updates both keys only when its marker, distinct keys, and values are
    valid. The update takes effect after vb, before that block's READ. A read
    query is valid only when it uses that block's valid write pair.
    """
    tokens = np.asarray(suffix)
    if tokens.ndim != 1:
        raise ValueError("suffix must be a one-dimensional token sequence")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError("suffix tokens must be integers")
    if len(tokens) % BLOCK_LENGTH != 0:
        raise ValueError(f"suffix length must be a multiple of {BLOCK_LENGTH}")

    memory: dict[int, int] = {}
    syntax_correct = 0
    query_valid = 0
    answer_correct = 0
    blocks = tokens.reshape(-1, BLOCK_LENGTH)
    key_set, value_set = set(KEY_IDS), set(VALUE_IDS)

    for row in blocks:
        write, key_a, key_b, value_a, value_b, fill_a, fill_b, read, query, answer, fill_c = (
            int(token) for token in row
        )
        role_checks = (
            write == WRITE_ID,
            key_a in key_set,
            key_b in key_set,
            value_a in value_set,
            value_b in value_set,
            fill_a == FILL_ID,
            fill_b == FILL_ID,
            read == READ_ID,
            query in key_set,
            answer in value_set,
            fill_c == FILL_ID,
        )
        syntax_correct += sum(role_checks)

        valid_write = (
            write == WRITE_ID
            and key_a in key_set
            and key_b in key_set
            and key_a != key_b
            and value_a in value_set
            and value_b in value_set
        )
        if valid_write:
            memory[key_a] = value_a
            memory[key_b] = value_b

        is_valid_query = (
            read == READ_ID
            and valid_write
            and query in (key_a, key_b)
            and query in memory
        )
        if is_valid_query:
            query_valid += 1
            if answer == memory[query]:
                answer_correct += 1

    block_count = len(blocks)
    return {
        "syntax_correct": int(syntax_correct),
        "syntax_count": int(len(tokens)),
        "query_valid": int(query_valid),
        "query_count": int(block_count),
        "answer_correct": int(answer_correct),
        "answer_count": int(block_count),
    }
