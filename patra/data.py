"""Dataset loading.

PATRA-EVAL: four task files (``comprehension.jsonl`` ...), rows with ``input`` (containing ``<ts><ts/>``),
``timeseries``, ``question_format`` and ``answer``.

PATRA-TRAIN: ``sft.jsonl`` (input, output, *normalised* series) and ``grpo.jsonl`` (same 27,906 prompts in the same
order, *raw* series, gold answer). The raw series are needed for the offset/scale prefix and the text-only model, so
``load_sft_rows`` can attach them from ``grpo.jsonl`` after checking the two files line up row by row.
"""
import os
import random

from .scoring import TASKS
from .utils import read_jsonl


def load_eval_samples(data_dir, tasks=TASKS, limit=None, shard=(0, 1)):
    """``[(task, idx, sample)]`` in task order. ``shard=(i, n)`` keeps every n-th item starting at i."""
    si, sn = shard
    samples = []
    for task in tasks:
        rows = read_jsonl(os.path.join(data_dir, f'{task}.jsonl'))
        for idx, s in enumerate(rows[:limit]):
            if idx % sn == si:
                samples.append((task, idx, s))
    return samples


def load_sft_rows(sft_path, raw_path=None):
    """Rows of ``sft.jsonl``; with ``raw_path`` (``grpo.jsonl``) each row also gets ``raw_ts``."""
    rows = read_jsonl(sft_path)
    if raw_path:
        raw = read_jsonl(raw_path)
        assert len(raw) == len(rows), (len(raw), len(rows))
        for r, g in zip(rows, raw):
            assert r['input'] == g['input'] and len(r['timeseries']) == len(g['timeseries']), 'files do not line up'
            r['raw_ts'] = g['timeseries']
    return rows


def split_train_val(rows, n_val, seed):
    """The shuffle used by every run: the first ``n_val`` rows of the shuffled list are the validation set."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[n_val:], rows[:n_val]
