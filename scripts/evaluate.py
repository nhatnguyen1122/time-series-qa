"""Batched evaluation of a PATRA-architecture checkpoint on PATRA-EVAL, with series ablations.

Each prompt's input embeddings are built one at a time through the checkpoint's own processor (official
preprocessing) and only the LLM is batched (see ``patra.modeling``); decoding is greedy with the checkpoint's
repetition penalty, which reproduces the official ``eval/patra_eval.py`` outputs. Scoring is the official one.

Series ablations (did the model use the series?)
  --ts_ablation shuffle   every sample gets the series of a random other sample with the same number of series
  --ts_ablation constant  every series becomes a flat line at its mean
For models trained with the value prefix (``--offset_scale``), ``--os_series`` picks which series the prefix
describes, so the TS-token channel and the text-prefix channel can be ablated separately:
  fed (default) = the series given to the TS encoder, original = the sample's own series,
  shuffled = the shuffled series (same draws as --ts_ablation shuffle), none = no prefix.

Output: OUT_DIR/predictions.jsonl (one record per item); summarise with scripts/aggregate.py.
"""
import argparse
import json
import os
import sys
import time

import torch
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.data import load_eval_samples  # noqa: E402
from patra.generation import generate  # noqa: E402
from patra.modeling import load_model, load_processor, processor_embedding  # noqa: E402
from patra.prompts import eval_prompt  # noqa: E402
from patra.scoring import TASKS, score_response  # noqa: E402
from patra.series import TS_PLACEHOLDER, add_offset_scale, constant_series, shuffled_series  # noqa: E402
from patra.utils import run_split_on_oom  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--data_dir', default=os.path.join(REPO, 'data', 'PATRA-EVAL'))
    ap.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    ap.add_argument('--max_memory', default=None, help='e.g. "0:20GiB" or "0:6GiB,1:8GiB" (one replica over GPUs)')
    ap.add_argument('--device', default=None, help='single device, e.g. cuda:0 (instead of --max_memory)')
    ap.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--repetition_penalty', type=float, default=None, help='default: from generation_config')
    ap.add_argument('--limit', type=int, default=None, help='first N items per task (debug)')
    ap.add_argument('--shard', default='0/1', help='i/n: every n-th item starting at i (multi-process)')
    ap.add_argument('--ts_ablation', choices=['none', 'shuffle', 'constant'], default='none')
    ap.add_argument('--offset_scale', action='store_true', help='add the value prefix (models trained with it)')
    ap.add_argument('--os_series', choices=['fed', 'original', 'shuffled', 'none'], default='fed')
    return ap.parse_args()


def prepare_inputs(samples, ts_ablation, offset_scale, os_series):
    """Apply the series ablation and the offset/scale prefix. ``samples``: [(task, idx, sample)]."""
    rows = [s for _, _, s in samples]
    shuffled = shuffled_series(rows) if ts_ablation == 'shuffle' or os_series == 'shuffled' else None
    out = []
    for i, (task, idx, s) in enumerate(samples):
        fed = {'none': s['timeseries'], 'shuffle': shuffled and shuffled[i],
               'constant': constant_series(s['timeseries'])}[ts_ablation]
        new = dict(s, timeseries=fed)
        if offset_scale:
            described = {'fed': fed, 'original': s['timeseries'], 'shuffled': shuffled and shuffled[i],
                         'none': None}[os_series]
            if described is not None:
                new['input'] = add_offset_scale(s['input'], described)
        assert new['input'].count(TS_PLACEHOLDER) == len(fed)
        out.append((task, idx, new))
    return out


def main():
    args = parse_args()
    model, tok = load_model(args.model_path, args.max_memory, args.device, dtype=getattr(torch, args.dtype))
    model.eval()
    processor = load_processor(args.model_path, tok)
    gcfg = model.generation_config
    rep = args.repetition_penalty if args.repetition_penalty is not None else (gcfg.repetition_penalty or 1.0)
    eos_ids = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
    print(f'device_map={getattr(model, "hf_device_map", None)}\nrepetition_penalty={rep} eos={eos_ids}', flush=True)

    shard = tuple(map(int, args.shard.split('/')))
    samples = prepare_inputs(load_eval_samples(args.data_dir, args.tasks, args.limit, shard),
                             args.ts_ablation, args.offset_scale, args.os_series)
    t0 = time.time()
    with torch.no_grad():
        prepped = [(task, idx, s, *processor_embedding(model, processor, eval_prompt(s), s['timeseries']))
                   for task, idx, s in samples]
    print(f'prepared {len(prepped)} samples in {time.time() - t0:.1f}s', flush=True)

    # length-sorted batches; prompt length is a decent proxy for answer length
    order = sorted(range(len(prepped)), key=lambda i: (prepped[i][3].shape[0], prepped[i][2]['question_format']))
    decode = lambda items: generate(model, [(p[3], p[4]) for p in items], args.max_new_tokens, eos_ids,
                                    tok.pad_token_id, repetition_penalty=rep)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = '' if shard[1] == 1 else f'.shard{shard[0]}of{shard[1]}'
    t1, done = time.time(), 0
    with open(os.path.join(args.out_dir, f'predictions{tag}.jsonl'), 'w', encoding='utf-8') as fout:
        for b in range(0, len(order), args.batch_size):
            batch = [prepped[i] for i in order[b:b + args.batch_size]]
            for (task, idx, s, _, _), g in zip(batch, run_split_on_oom(decode, batch)):
                resp = tok.decode(g, skip_special_tokens=True)
                kind, score = score_response(s, resp, rouge)
                fout.write(json.dumps({'task': task, 'idx': idx, 'question_format': s['question_format'],
                                       'metric': kind, 'score': score,
                                       'n_gen_tokens': int((g != tok.pad_token_id).sum()),
                                       'answer': s['answer'], 'response': resp}, ensure_ascii=False) + '\n')
            fout.flush()
            done += len(batch)
            el = time.time() - t1
            print(f'[{done}/{len(order)}] {el:.0f}s elapsed, ETA {el / done * (len(order) - done):.0f}s', flush=True)
    print(f'total generation time {time.time() - t1:.1f}s', flush=True)


if __name__ == '__main__':
    main()
