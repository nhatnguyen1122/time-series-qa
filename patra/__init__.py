"""Reproduction and analysis toolkit for PATRA (time-series question answering, ICML 2026).

Modules
-------
series      time-series preprocessing: normalisation, offset/scale prefix, text serialisation, ablations
prompts     chat templates used for training and the official evaluation prompt
modeling    loading PATRA checkpoints across GPUs and building the merged (text + TS) input embeddings
generation  batched greedy / sampled decoding over precomputed input embeddings
training    teacher-forced log-probabilities, LR schedule, LoRA targets, checkpoint saving
rewards     GRPO reward functions (copied from the authors' workflow) and the "original reward" ablation
scoring     official and lenient answer extraction, per-task metrics
stats       paired bootstrap and question categorisation used by the analysis scripts
utils       small I/O and robustness helpers
"""
