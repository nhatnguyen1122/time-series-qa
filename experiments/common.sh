# Shared settings and helpers for the experiment scripts (sourced, not executed).
#   GPU      GPU index to use (default 0)
#   MAX_MEM  per-GPU memory budget for one model replica (default 0:20GiB; "0:6GiB,1:8GiB" splits over two GPUs)
# Every step is skipped when its output already exists, so a script can simply be re-run after an interruption.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAX_MEM=${MAX_MEM:-0:20GiB}
PY=${PY:-python}
EVAL_DATA=data/PATRA-EVAL
TRAIN_DATA=data/PATRA-TRAIN
mkdir -p runs logs

# eval_patra NAME MODEL_DIR [evaluate.py options...]  -> runs/eval_NAME
eval_patra() {
  local name=$1 model=$2; shift 2
  [ -f "runs/eval_$name/summary.json" ] && { echo "skip eval_$name"; return; }
  $PY scripts/evaluate.py --model_path "$model" --out_dir "runs/eval_$name" --max_memory "$MAX_MEM" "$@" \
    > "logs/eval_$name.log" 2>&1
  $PY scripts/aggregate.py "runs/eval_$name"
}

# eval_text NAME MODEL_DIR [evaluate_text.py options...]  -> runs/eval_NAME (resumes a partial run)
eval_text() {
  local name=$1 model=$2; shift 2
  [ -f "runs/eval_$name/summary.json" ] && { echo "skip eval_$name"; return; }
  $PY scripts/evaluate_text.py --model_path "$model" --out_dir "runs/eval_$name" --max_memory "$MAX_MEM" "$@" \
    >> "logs/eval_$name.log" 2>&1
  $PY scripts/aggregate.py "runs/eval_$name"
}

# train SCRIPT NAME [options...]  -> runs/NAME/merged (resumes from the last checkpoint)
train() {
  local script=$1 name=$2; shift 2
  [ -f "runs/$name/merged/model.safetensors.index.json" ] && { echo "skip $name"; return; }
  $PY "scripts/$script" --out_dir "runs/$name" --max_memory "$MAX_MEM" --resume "$@" >> "logs/$name.log" 2>&1
}
