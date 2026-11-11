"""Check that the batched evaluator reproduces the official one.

For N items, compares the response of the official path (batch size 1, ``model.generate(do_sample=False)`` on the
processor output, as in the authors' ``eval/patra_eval.py``) with ours (per-sample embeddings + batched greedy
decoding with the repetition penalty). Reports exact-match count and mean character similarity.

With ``--official_task T`` it instead runs only the official path over every multiple-choice / true-false item of
task T and reports the official accuracy (to compare with the batched evaluator's number for that task).
"""
import argparse
import difflib
import json
import os
import sys

import torch
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.data import load_eval_samples  # noqa: E402
from patra.generation import generate  # noqa: E402
from patra.modeling import load_model, load_processor, processor_embedding  # noqa: E402
from patra.prompts import CLOSED_FORMATS, eval_prompt  # noqa: E402
from patra.scoring import score_response  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def official_response(model, tok, processor, sample):
    inputs = processor(text=[eval_prompt(sample)], timeseries=sample['timeseries'], padding=True, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs['attention_mask'][0].sum().item():], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--data_dir', default=os.path.join(REPO, 'data', 'PATRA-EVAL'))
    ap.add_argument('--max_memory', default='0:20GiB')
    ap.add_argument('--n', type=int, default=16, help='items, spread over recognition and reasoning')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--official_task', default=None, help='official accuracy on all MC/TF items of this task')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model_path, args.max_memory, dtype=torch.float16)
    model.eval()
    processor = load_processor(args.model_path, tok)
    if args.official_task:
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rows = [(idx, s) for _, idx, s in load_eval_samples(args.data_dir, [args.official_task])
                if s['question_format'] in CLOSED_FORMATS]
        res = [{'idx': idx, 'score': score_response(s, official_response(model, tok, processor, s), rouge)[1]}
               for idx, s in rows]
        print(f'official acc on {args.official_task}: {sum(r["score"] for r in res)}/{len(res)}')
        json.dump(res, open(args.out, 'w'))
        return
    samples = []
    for task in ['recognition', 'reasoning']:  # a mix of MC / TF / open-ended
        rows = [s for _, _, s in load_eval_samples(args.data_dir, [task])]
        samples += rows[::max(1, len(rows) // (args.n // 2))][:args.n // 2]

    official = [official_response(model, tok, processor, s) for s in samples]
    with torch.no_grad():
        items = [processor_embedding(model, processor, eval_prompt(s), s['timeseries']) for s in samples]
    gcfg = model.generation_config
    eos = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
    batched = []
    for b in range(0, len(items), args.batch_size):
        out = generate(model, items[b:b + args.batch_size], 512, eos, tok.pad_token_id,
                       repetition_penalty=gcfg.repetition_penalty or 1.0)
        batched += [tok.decode(x, skip_special_tokens=True) for x in out]

    sims = [difflib.SequenceMatcher(None, x, y).ratio() for x, y in zip(official, batched)]
    print(f'exact match {sum(x == y for x, y in zip(official, batched))}/{len(official)}; '
          f'mean char similarity {sum(sims) / len(sims):.4f}; min {min(sims):.4f}')
    json.dump([{'q': s['input'], 'format': s['question_format'], 'official': x, 'batched': y, 'similarity': z}
               for s, x, y, z in zip(samples, official, batched, sims)], open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
