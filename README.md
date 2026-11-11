# PATRA

Re-implementation of the training and evaluation pipeline of [PATRA](https://arxiv.org/abs/2602.23161)
(time-series question answering). It adds the GRPO trainer missing from the official release, optional fixes to the
released modelling code, and series ablations for evaluation.

## Layout

```
patra/            shared library: series preprocessing, prompts, model loading, decoding, rewards, scoring
patra_model/      PATRA modelling code (fixes switched on by `patra_fixes` in config.json)
scripts/
  build_init.py     init checkpoint: PATRA config + Qwen2.5-7B-Instruct weights
  train_sft.py      stage 1: LoRA on the LLM + fully trained TS encoder / alignment
  train_grpo.py     stage 2: GRPO with the task-aware reward
  evaluate.py       batched evaluation of PATRA checkpoints, with series ablations
  evaluate_text.py  evaluation of text-only LLMs (series written as numbers)
  analysis/         run comparison, per-category breakdown, alignment diagnostics
experiments/      end-to-end pipelines
tests/            unit tests (CPU)
```

## Setup

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash experiments/setup.sh     # data -> data/, PATRA-7B + Qwen2.5-7B-Instruct -> models/, builds models/init-*
pytest
```

## Usage

```bash
GPU=0 bash experiments/round1.sh     # baselines, SFT -> GRPO with the released code
GPU=0 bash experiments/round2.sh     # SFT with code fixes / value prefix / text-only input, ablations
bash experiments/analysis.sh         # comparison tables -> runs/analysis/
```

Finished steps are skipped and training resumes from its last checkpoint. `MAX_MEM=0:8GiB,1:8GiB` splits a model
over two GPUs. Single steps can also be run directly, e.g.

```bash
python scripts/evaluate.py --model_path runs/sft_ours/merged --out_dir runs/eval_sft_ours_tsshuffle --ts_ablation shuffle
python scripts/aggregate.py runs/eval_sft_ours_tsshuffle
```

Every script lists its options with `--help`. Training runs are written to `runs/<name>/` (merged model in
`merged/`), evaluations to `runs/eval_<name>/predictions.jsonl`, logs to `logs/`.
