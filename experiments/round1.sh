#!/bin/bash
# Round 1: evaluate the released model and a text-only baseline, then re-train the paper's pipeline
# (SFT -> GRPO) with the released code. About 2 h SFT + 2 h GRPO + ~10 min per evaluation on one 24 GB GPU.
source "$(dirname "$0")/common.sh"

# --- baselines
eval_patra patra7b            models/PATRA-7B
eval_patra patra7b_tsshuffle  models/PATRA-7B --ts_ablation shuffle
eval_text  qwen_text          models/Qwen2.5-7B-Instruct

# --- stage 1: SFT (normalised series from sft.jsonl, released alignment code, fresh TS modules)
train train_sft.py sft_ours --model_path models/init-released --data $TRAIN_DATA/sft.jsonl --reinit_ts \
  --save_every 50
eval_patra sft_ours            runs/sft_ours/merged
eval_patra sft_ours_tsshuffle  runs/sft_ours/merged --ts_ablation shuffle

# --- stage 2: GRPO with the task-aware balanced reward
train train_grpo.py grpo_ours --model_path runs/sft_ours/merged --data $TRAIN_DATA/grpo.jsonl
eval_patra grpo_ours            runs/grpo_ours/merged
eval_patra grpo_ours_tsshuffle  runs/grpo_ours/merged --ts_ablation shuffle
