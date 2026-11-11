"""Small helpers: reproducibility, JSONL I/O and surviving OOM on shared GPUs."""
import json
import os
import random
import time

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def append_jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def atomic_torch_save(obj, path):
    """Write to a temporary file first so an interrupted save never leaves a truncated checkpoint."""
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in str(err).lower()


def run_split_on_oom(fn, items, wait_s=120):
    """``fn(items) -> list`` with one output per item. On CUDA OOM, retry on the two halves (recursively); a single
    item that still does not fit waits for memory to free up. Results are returned in input order."""
    try:
        return list(fn(items))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) == 1:
            print(f'OOM at batch size 1, waiting {wait_s}s', flush=True)
            time.sleep(wait_s)
            return run_split_on_oom(fn, items, wait_s)
        h = len(items) // 2
        print(f'OOM -> split {len(items)} into {h}+{len(items) - h}', flush=True)
        return run_split_on_oom(fn, items[:h], wait_s) + run_split_on_oom(fn, items[h:], wait_s)
