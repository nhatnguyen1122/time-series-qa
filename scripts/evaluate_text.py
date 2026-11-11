"""Evaluate a text-only LLM (Qwen2.5-7B-Instruct or the text-only SFT model) on PATRA-EVAL.

Each ``<ts><ts/>`` is replaced by the series written as numbers; prompt, greedy decoding, token budget and scoring
are the same as for ``evaluate.py``.
  --ts_ablation shuffle  numbers of a random other sample's series (same draws as evaluate.py)
  --ts_ablation remove   the placeholder becomes "[series not shown]" (question-only prior)
Rows already in OUT_DIR/predictions.jsonl are kept, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import sys
import time

import torch
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.data import load_eval_samples  # noqa: E402
from patra.modeling import parse_max_memory  # noqa: E402
from patra.prompts import eval_prompt  # noqa: E402
from patra.scoring import score_response  # noqa: E402
from patra.series import hide_series, serialize_text, shuffled_series  # noqa: E402
from patra.utils import run_split_on_oom  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_prompts(samples, ablation):
    rows = [s for _, _, s in samples]
    if ablation == 'remove':
        return [eval_prompt(dict(s, input=hide_series(s['input']))) for s in rows]
    series = shuffled_series(rows) if ablation == 'shuffle' else [s['timeseries'] for s in rows]
    return [eval_prompt(dict(s, input=serialize_text(s['input'], ts))) for s, ts in zip(rows, series)]


def load_done(path):
    """Complete rows of an interrupted run (the file is rewritten without a possibly truncated last line)."""
    if not os.path.exists(path):
        return set()
    good = []
    for line in open(path):
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    with open(path, 'w') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in good)
    return {(r['task'], r['idx']) for r in good}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--data_dir', default=os.path.join(REPO, 'data', 'PATRA-EVAL'))
    ap.add_argument('--max_memory', default='0:20GiB')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--ts_ablation', choices=['none', 'shuffle', 'remove'], default='none')
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path)
    cfg.base_model_tp_plan = {}  # transformers 4.52 + torch < 2.5: the tensor-parallel style check crashes otherwise
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=cfg, device_map='auto',
                                                 max_memory=parse_max_memory(args.max_memory),
                                                 torch_dtype=torch.float16).eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = 'left'

    samples = load_eval_samples(args.data_dir)
    prompts = build_prompts(samples, args.ts_ablation)
    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, 'predictions.jsonl')
    done = load_done(pred_path)
    order = [i for i in sorted(range(len(samples)), key=lambda i: len(prompts[i]))
             if (samples[i][0], samples[i][1]) not in done]
    print(f'{len(done)} rows already done, {len(order)} to go', flush=True)

    def decode(idx):
        enc = tok([prompts[i] for i in idx], return_tensors='pt', padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return out[:, enc['input_ids'].shape[1]:].cpu()

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    t0 = time.time()
    with open(pred_path, 'a') as f:
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            for i, g in zip(idx, run_split_on_oom(decode, idx)):
                task, k, s = samples[i]
                resp = tok.decode(g, skip_special_tokens=True)
                kind, score = score_response(s, resp, rouge)
                f.write(json.dumps({'task': task, 'idx': k, 'question_format': s['question_format'], 'metric': kind,
                                    'score': score, 'n_gen_tokens': int((g != tok.pad_token_id).sum()),
                                    'answer': s['answer'], 'response': resp}, ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{len(done) + b + len(idx)}/{len(done) + len(order)}] {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
