#!/bin/bash
# Round 2: same SFT configuration as round 1, three input/architecture variants, plus ablations that separate the
# two ways a series reaches the model (TS tokens vs the offset/scale text prefix). 1.5-2.2 h per SFT run.
#   A  sft_fixed      bug fixes + offset/scale prefix + raw series
#   C  sft_fixed_res  A + residual connection in the alignment module
#   B  sft_text       no TS encoder: the series is written into the prompt as numbers
source "$(dirname "$0")/common.sh"
RAW="--raw_ts $TRAIN_DATA/grpo.jsonl"
SFT="--data $TRAIN_DATA/sft.jsonl --save_every 50"

train train_sft.py sft_fixed     --model_path models/init-fixed     $SFT $RAW --offset_scale --reinit_ts
train train_sft.py sft_fixed_res --model_path models/init-fixed-res $SFT $RAW --offset_scale --reinit_ts
train train_sft.py sft_text      --model_path models/init-released  $SFT $RAW --text_ts \
  --text_save_src models/Qwen2.5-7B-Instruct

for m in sft_fixed sft_fixed_res; do
  M=runs/$m/merged
  eval_patra $m                 $M --offset_scale
  eval_patra ${m}_tsshuffle     $M --offset_scale --ts_ablation shuffle                          # both channels
  eval_patra ${m}_tsshuf_keepos $M --offset_scale --ts_ablation shuffle  --os_series original    # TS tokens only
  eval_patra ${m}_osshuf        $M --offset_scale --os_series shuffled                           # prefix only
  eval_patra ${m}_noos          $M --offset_scale --os_series none                               # no prefix
  eval_patra ${m}_tsconst_keepos $M --offset_scale --ts_ablation constant --os_series original   # flat series
done

eval_text sft_text            runs/sft_text/merged
eval_text sft_text_tsshuffle  runs/sft_text/merged --ts_ablation shuffle
eval_text sft_text_noseries   runs/sft_text/merged --ts_ablation remove
eval_text qwen_text_tsshuffle models/Qwen2.5-7B-Instruct --ts_ablation shuffle
eval_text qwen_text_noseries  models/Qwen2.5-7B-Instruct --ts_ablation remove
