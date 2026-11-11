"""Per question-category lenient accuracy, true/false bias and multiple-choice answer distribution.

Optionally, paired bootstrap CIs per category for base -> other pairs (e.g. the effect of an ablation on 'trend').

    python scripts/analysis/category_breakdown.py --runs_dir runs --data_dir data/PATRA-EVAL \
        --runs qwen_text sft_fixed sft_text --pairs sft_fixed_res:sft_fixed_res_tsshuf_keepos --out cat.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from patra.data import load_eval_samples  # noqa: E402
from patra.scoring import lenient_hit, lenient_pred, load_predictions  # noqa: E402
from patra.stats import paired_bootstrap, question_category  # noqa: E402


def closed_rows_with_category(run_dir, category_of):
    rows = load_predictions(run_dir) or {}
    return {k: dict(r, cat=category_of[k]) for k, r in rows.items() if r['metric'] == 'acc'}


def summarise(rows):
    cats = collections.defaultdict(list)
    for r in rows.values():
        cats[r['cat']].append(lenient_hit(r))
    tf = [r for r in rows.values() if r['question_format'] == 'true/false']
    mc = [r for r in rows.values() if r['question_format'] == 'multiple_choice']
    pred_true = sum(lenient_pred(r['response'], 'true/false') == 'True' for r in tf)
    return {'cat_acc_len': {c: (round(100 * sum(v) / len(v), 1), len(v)) for c, v in sorted(cats.items())},
            'tf_pred_true_pct': round(100 * pred_true / len(tf), 1),
            'tf_gold_true_pct': round(100 * sum(r['answer'] == 'T' for r in tf) / len(tf), 1),
            'mc_pred_dist': dict(collections.Counter(str(lenient_pred(r['response'], 'multiple_choice')) for r in mc)),
            'mc_gold_dist': dict(collections.Counter(r['answer'] for r in mc))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='run names (eval_<name> directories)')
    ap.add_argument('--pairs', nargs='*', default=[], help='base:other run pairs for per-category bootstrap')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    category_of = {(t, i): question_category(s['input']) for t, i, s in load_eval_samples(args.data_dir)}
    load = lambda n: closed_rows_with_category(os.path.join(args.runs_dir, f'eval_{n}'), category_of)
    res = {n: summarise(rows) for n in args.runs if (rows := load(n))}

    cats = sorted({c for v in res.values() for c in v['cat_acc_len']})
    print('| category (n) | ' + ' | '.join(res) + ' |')
    for c in cats:
        n = next(v['cat_acc_len'][c][1] for v in res.values() if c in v['cat_acc_len'])
        print(f'| {c} ({n}) | ' + ' | '.join(str(v['cat_acc_len'].get(c, ('-',))[0]) for v in res.values()) + ' |')
    print('| TF pred True % (gold %) | ' + ' | '.join(f"{v['tf_pred_true_pct']} ({v['tf_gold_true_pct']})"
                                                   for v in res.values()) + ' |')

    out = {'runs': res, 'paired_by_category': {}}
    for pair in args.pairs:
        a, b = pair.split(':')
        ra, rb = load(a), load(b)
        out['paired_by_category'][pair] = {c: paired_bootstrap(ra, rb, 'acc_len', lambda k, r, c=c: r['cat'] == c,
                                                               n_boot=4000) for c in cats}
        for c, v in out['paired_by_category'][pair].items():
            print(f"{pair} {c:24s} n={v['n']:4d} d={v['diff']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]")
    json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
