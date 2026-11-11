"""Compare evaluation runs: per-task metrics, series sensitivity of every ablation, paired bootstrap CIs.

Expects RUNS_DIR/eval_<model>[_<ablation>]/predictions*.jsonl (the layout written by experiments/*.sh).

    python scripts/analysis/compare_runs.py --runs_dir runs --out runs/analysis/compare_runs.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from patra.scoring import TASKS, load_predictions, task_metrics  # noqa: E402
from patra.stats import paired_bootstrap, series_sensitivity  # noqa: E402

MODELS = ['patra7b', 'qwen_text', 'sft_ours', 'grpo_ours', 'sft_fixed', 'sft_fixed_res', 'sft_text']
ABLATIONS = ['tsshuffle', 'noseries', 'tsshuf_keepos', 'osshuf', 'noos', 'tsconst_keepos']
# model-vs-model comparisons reported in the round-2 report (base -> ablation pairs are added automatically)
MODEL_PAIRS = [('sft_ours', 'sft_fixed'), ('sft_fixed', 'sft_fixed_res'), ('sft_ours', 'sft_fixed_res'),
               ('sft_text', 'sft_fixed_res'), ('sft_text', 'sft_fixed'), ('qwen_text', 'sft_text'),
               ('qwen_text', 'sft_fixed_res'), ('patra7b', 'sft_fixed_res'), ('sft_ours', 'grpo_ours')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--n_boot', type=int, default=5000)
    args = ap.parse_args()

    runs = {}
    for m in args.models:
        for suffix in [''] + [f'_{a}' for a in ABLATIONS]:
            rows = load_predictions(os.path.join(args.runs_dir, f'eval_{m}{suffix}'))
            if rows:
                runs[m + suffix] = rows
    pairs = [p for p in MODEL_PAIRS if p[0] in runs and p[1] in runs]
    pairs += [(m, f'{m}_{a}') for m in args.models for a in ABLATIONS if m in runs and f'{m}_{a}' in runs]

    res = {'metrics': {k: task_metrics(v) for k, v in runs.items()},
           'sensitivity': {b: series_sensitivity(runs[a], runs[b]) for a, b in pairs if b.startswith(a + '_')},
           'paired': {}}
    task_filter = {t: (None if t == 'all' else (lambda k, r, t=t: k[0] == t)) for t in TASKS + ['all']}
    for a, b in pairs:
        res['paired'][f'{a} -> {b}'] = {
            metric: {t: paired_bootstrap(runs[a], runs[b], metric, keep, n_boot=args.n_boot)
                     for t, keep in task_filter.items()}
            for metric in ('acc_len', 'acc', 'rougeL')}

    print('| run | ' + ' | '.join(f'{t[:6]} Acc/Len/R-L' for t in TASKS) + ' | all AccLen |')
    for k, m in res['metrics'].items():
        print(f'| {k} | ' + ' | '.join(f"{m[t]['acc']:.1f}/{m[t]['acc_lenient']:.1f}/{m[t]['rougeL']:.1f}"
                                      for t in TASKS) + f" | {m['all']['acc_lenient']:.1f} |")
    print('\nseries sensitivity (base -> ablation)')
    for k, s in res['sensitivity'].items():
        print(f'{k:32s}', {kk: round(v, 1) for kk, v in s.items()})
    print('\npaired bootstrap, all tasks: lenient Acc | ROUGE-L')
    for k, v in res['paired'].items():
        a, r = v['acc_len']['all'], v['rougeL']['all']
        print(f"{k:45s} dAccLen {a['diff']:+.2f} [{a['ci95'][0]:+.2f},{a['ci95'][1]:+.2f}] p={a['p_two_sided']:.3f} | "
              f"dR-L {r['diff']:+.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
