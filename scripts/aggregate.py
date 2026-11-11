"""Summarise evaluation runs: per-task Acc / lenient Acc / ROUGE-L next to the paper's numbers.

    python scripts/aggregate.py runs/eval_sft_fixed runs/eval_sft_text ...
Writes RUN_DIR/summary.json for every run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.scoring import PAPER_RESULTS, TASKS, load_predictions, task_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dirs', nargs='+')
    for d in ap.parse_args().run_dirs:
        rows = load_predictions(d)
        if rows is None:
            print(f'== {d}: no predictions')
            continue
        m = task_metrics(rows)
        print(f'== {d}  ({len(rows)} predictions)')
        print(f"{'task':14s} {'n':>5s} {'Acc':>7s} {'(paper)':>8s} {'ROUGE-L':>8s} {'(paper)':>8s} "
              f"{'noBoxed%':>8s} {'AccLen':>7s} {'avgTok':>7s} {'hit512%':>7s}")
        for t in TASKS:
            if t in m:
                r = m[t]
                print(f"{t:14s} {r['n']:5d} {r['acc']:7.2f} {PAPER_RESULTS[t][0]:8.2f} {r['rougeL']:8.2f} "
                      f"{PAPER_RESULTS[t][1]:8.2f} {r['no_boxed_pct']:8.1f} {r['acc_lenient']:7.2f} "
                      f"{r['avg_gen_tokens']:7.1f} {r['hit_max_tokens_pct']:7.1f}")
        json.dump(m, open(os.path.join(d, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
