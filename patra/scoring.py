"""Answer extraction and metrics.

* Official (paper, Table 1): Acc = exact match of the first ``\\boxed{...}`` with the gold label over multiple-choice
  and true/false items; ROUGE-L F1 over open-ended items.
* Lenient Acc ("AccLen"): ``\\boxed{}`` if present, else an ``<answer>`` tag, else the first option letter /
  True|False word in the text. SFT models trained on PATRA-TRAIN answer as "B) description", often inside the box,
  which the official metric scores as wrong; AccLen separates format from content.
"""
import glob
import gzip
import json
import os
import re

from .prompts import BOXED_RE, CLOSED_FORMATS

TASKS = ['comprehension', 'recognition', 'reasoning', 'prescience']
PAPER_RESULTS = {'comprehension': (56.03, 25.67), 'recognition': (64.69, 25.46), 'reasoning': (44.59, 27.36),
                 'prescience': (52.78, 27.06)}  # PATRA (Acc, ROUGE-L), Table 1
ANSWER_TAG_RE = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.S)


def gold_label(sample):
    return sample['answer'] if sample['question_format'] == 'multiple_choice' else \
        ('True' if sample['answer'] == 'T' else 'False')


def score_response(sample, response, rouge):
    """Official per-item score: ``('acc', 0|1)`` for closed questions, ``('rougeL', f1)`` for open-ended ones."""
    if sample['question_format'] in CLOSED_FORMATS:
        m = BOXED_RE.search(response)
        pred = m.group(1) if m else None
        return 'acc', int(pred is not None and pred.lower() == gold_label(sample).lower())
    return 'rougeL', rouge.score(sample['answer'], response)['rougeL'].fmeasure


def lenient_pred(response, question_format):
    """Predicted label (``'A'``-``'D'`` / ``'True'`` / ``'False'``) or None, see module docstring."""
    m = BOXED_RE.search(response) or ANSWER_TAG_RE.search(response)
    text = m.group(1) if m else response
    if question_format == 'multiple_choice':
        mm = re.search(r'\b([ABCD])\b', text, re.I)
        return mm.group(1).upper() if mm else None
    mm = re.search(r'\b(true|false)\b', text, re.I)
    return mm.group(1).capitalize() if mm else None


def lenient_hit(row):
    """``row``: a prediction record (``answer``, ``question_format``, ``response``)."""
    pred = lenient_pred(row['response'], row['question_format'])
    return int(pred is not None and pred.lower() == gold_label(row).lower())


# ----------------------------------------------------------------------------- prediction files
def load_predictions(run_dir):
    """All ``predictions*.jsonl[.gz]`` of an evaluation run -> ``{(task, idx): record}``; None if the run is missing."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'predictions*.jsonl*'))):
        with (gzip.open(f, 'rt') if f.endswith('.gz') else open(f)) as fh:
            for line in fh:
                r = json.loads(line)
                rows[(r['task'], r['idx'])] = r
    return rows or None


def task_metrics(rows):
    """Per-task (and 'all') Acc, AccLen, ROUGE-L in %, plus format diagnostics, from ``load_predictions`` output."""
    out = {}
    for t in TASKS + ['all']:
        rs = [r for (task, _), r in rows.items() if t == 'all' or task == t]
        if not rs:
            continue
        closed = [r for r in rs if r['metric'] == 'acc']
        rouge = [r['score'] for r in rs if r['metric'] == 'rougeL']
        ntok = [r['n_gen_tokens'] for r in rs]
        mean = lambda xs: 100 * sum(xs) / len(xs) if xs else float('nan')
        out[t] = {'n': len(rs), 'n_acc': len(closed), 'n_rouge': len(rouge),
                  'acc': mean([r['score'] for r in closed]),
                  'acc_lenient': mean([lenient_hit(r) for r in closed]),
                  'rougeL': mean(rouge),
                  'no_boxed_pct': mean([int('\\boxed{' not in r['response']) for r in closed]),
                  'no_answer_lenient_pct': mean([int(lenient_pred(r['response'], r['question_format']) is None)
                                                 for r in closed]),
                  'avg_gen_tokens': sum(ntok) / len(ntok),
                  'hit_max_tokens_pct': mean([int(n >= 512) for n in ntok])}
    return out
