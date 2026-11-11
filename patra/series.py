"""Time-series preprocessing shared by training, evaluation and analysis.

PATRA prompts contain one ``<ts><ts/>`` placeholder per series. The model receives each series through the TS
encoder after ``sp_encoding`` (mean-shift, then scale so that |x| <= 3). That normalisation throws away the absolute
level and scale; ChatTS (from which PATRA's code is derived) restores them with a text prefix that the PATRA release
has commented out. This module implements the normalisation, the prefix, the text-only serialisation used by the
no-encoder baseline, and the series ablations used to test whether a model actually reads the series.
"""
import random

import numpy as np
import torch

TS_PLACEHOLDER = '<ts><ts/>'


# ----------------------------------------------------------------------------- normalisation
def offset_scale_stats(ts):
    """Return ``(offset, scale)`` exactly as ``processing_qwen2_ts.sp_encoding`` computes them (offset = -mean)."""
    x = np.asarray(ts, dtype=np.float64)
    centred = x - x.mean()
    scale = float(np.max(np.abs(centred)) / 3.0) if np.any(np.abs(centred) >= 3.0) else 1.0
    return float(-x.mean()), scale


def sp_encode(ts):
    """Normalise one series and interleave the all-ones mask channel -> flat array of length 2 * len(ts)."""
    x = np.asarray(ts, dtype=np.float64)
    offset, scale = offset_scale_stats(x)
    x = (x + offset) / scale
    return np.stack([x, np.ones_like(x)], -1).reshape(-1)


def ts_tensor(ts_list, pre_encoded=False):
    """Stack series into the processor's layout ``[n_series, max_len, 1]`` (zero padded, float32).

    ``pre_encoded=True`` means the values are already normalised and interleaved with the mask channel
    (the format stored in PATRA-TRAIN's ``sft.jsonl``).
    """
    enc = [np.asarray(t, dtype=np.float64).reshape(-1) if pre_encoded else sp_encode(t) for t in ts_list]
    arr = np.zeros((len(enc), max(len(e) for e in enc), 1), dtype=np.float32)
    for i, e in enumerate(enc):
        arr[i, :len(e), 0] = e
    return torch.from_numpy(arr)


# ----------------------------------------------------------------------------- prompt rewriting
def _rewrite_placeholders(text, ts_list, render):
    parts = text.split(TS_PLACEHOLDER)
    assert len(parts) - 1 == len(ts_list), f'{len(parts) - 1} placeholders vs {len(ts_list)} series'
    return parts[0] + ''.join(render(ts) + nxt for ts, nxt in zip(ts_list, parts[1:]))


def offset_scale_prefix(ts):
    """ChatTS' value prefix: ``[Value Offset: {-mean:.4f}|Value Scaling: {scale:.4f}]``."""
    offset, scale = offset_scale_stats(ts)
    return f'[Value Offset: {offset:.4f}|Value Scaling: {scale:.4f}]'


def add_offset_scale(text, ts_list):
    """Put the offset/scale prefix of ``ts_list[i]`` in front of the i-th placeholder (placeholders are kept)."""
    return _rewrite_placeholders(text, ts_list, lambda ts: offset_scale_prefix(ts) + TS_PLACEHOLDER)


def serialize_series(ts):
    return '[' + ', '.join(f'{v:g}' for v in ts) + ']'


def serialize_text(text, ts_list):
    """Text-only input: replace each placeholder by the raw series written as numbers, e.g. ``[1.5, 2, 3.25]``."""
    return _rewrite_placeholders(text, ts_list, serialize_series)


def hide_series(text):
    """Question-only input: the series is not shown at all."""
    return text.replace(TS_PLACEHOLDER, '[series not shown]')


# ----------------------------------------------------------------------------- ablations
def shuffled_series(samples, seed=0):
    """For every sample, the series of a random sample with the same number of series.

    ``samples`` is a list of dicts with a ``timeseries`` field. The draw order is fixed (one ``rng.choice`` per sample,
    in order), so every evaluation script that uses this function gets identical replacements.
    """
    rng = random.Random(seed)
    pool = {}
    for s in samples:
        pool.setdefault(len(s['timeseries']), []).append(s['timeseries'])
    return [rng.choice(pool[len(s['timeseries'])]) for s in samples]


def constant_series(ts_list):
    """Replace every series by a flat line at its mean (same length): the shape information is removed."""
    return [[float(sum(t) / len(t))] * len(t) for t in ts_list]
