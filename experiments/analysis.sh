#!/bin/bash
# All analysis tables from the evaluation runs (CPU only; the weight comparison needs ~16 GB of RAM).
#   RUNS=<dir>         read the eval_* runs from <dir> instead of runs/
#   ONLY_PREDICTIONS=1 stop after the analyses that need nothing but predictions + data
source "$(dirname "$0")/common.sh"
RUNS=${RUNS:-runs}
A=runs/analysis
mkdir -p $A

$PY scripts/analysis/compare_runs.py --runs_dir $RUNS --out $A/compare_runs.json
$PY scripts/analysis/category_breakdown.py --runs_dir $RUNS --data_dir $EVAL_DATA --out $A/categories.json \
  --runs qwen_text qwen_text_noseries patra7b sft_ours sft_fixed sft_fixed_tsshuffle sft_fixed_res \
         sft_fixed_res_tsshuf_keepos sft_text sft_text_tsshuffle sft_text_noseries \
  --pairs sft_fixed_res:sft_fixed_res_tsshuf_keepos sft_fixed:sft_fixed_tsshuf_keepos \
          sft_fixed:sft_fixed_osshuf sft_text:sft_text_tsshuffle
$PY scripts/analysis/question_only_probe.py --train $TRAIN_DATA/grpo.jsonl --data_dir $EVAL_DATA \
  --out $A/question_only_probe.json
[ "${ONLY_PREDICTIONS:-0}" = 1 ] && exit 0

DIAG=scripts/analysis/alignment_diagnostics.py
$PY $DIAG models/PATRA-7B          $A/aligndiag_patra7b.json
$PY $DIAG runs/sft_ours/merged     $A/aligndiag_sft_ours.json
$PY $DIAG runs/grpo_ours/merged    $A/aligndiag_grpo_ours.json
$PY $DIAG runs/sft_fixed/merged    $A/aligndiag_sft_fixed.json --offset_scale
$PY $DIAG runs/sft_fixed_res/merged $A/aligndiag_sft_fixed_res.json --offset_scale

$PY scripts/analysis/weight_delta.py models/PATRA-7B models/Qwen2.5-7B-Instruct $A/weight_delta.json
