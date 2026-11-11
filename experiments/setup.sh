#!/bin/bash
# Download data and checkpoints from the Hugging Face Hub and build the three init checkpoints.
# Needs ~35 GB of disk (PATRA-7B + Qwen2.5-7B-Instruct); the init checkpoints only symlink Qwen's weights.
source "$(dirname "$0")/common.sh"

huggingface-cli download DecisionIntelligence/PATRA-EVAL --repo-type dataset --local-dir "$EVAL_DATA"
huggingface-cli download DecisionIntelligence/PATRA-TRAIN --repo-type dataset --local-dir "$TRAIN_DATA"
huggingface-cli download DecisionIntelligence/PATRA-7B --local-dir models/PATRA-7B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

INIT="$PY scripts/build_init.py --patra models/PATRA-7B --qwen models/Qwen2.5-7B-Instruct"
$INIT --out models/init-released                                        # round 1: code as released
$INIT --out models/init-fixed --fixes noncausal_align_attn trend_seasonal_pairing per_series_decomp
$INIT --out models/init-fixed-res --fixes noncausal_align_attn trend_seasonal_pairing per_series_decomp \
  residual_alignment
