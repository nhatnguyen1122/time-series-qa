"""Pieces shared by SFT and GRPO: LoRA targets, LR schedule, teacher-forced log-probs and checkpoint saving.

The official pipeline (a LLaMA-Factory fork with DeepSpeed) keeps a full fp16 replica per GPU and depends on files
that are not released (PATRA-init, the GRPO trainer). Here one replica is split over GPUs, LoRA is applied to the
LLM, and the TS modules are trained in fp32.
"""
import json
import math
import os
import shutil

import torch
from torch.utils.checkpoint import checkpoint

from .modeling import embedding_device, unwrap

LLM_LORA_REGEX = r'model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
# The authors' lora_target list ("q_proj, ..., down_proj, mlp.0, mlp.2, mlp.4, mlp.6, mlp.8, complete_query,
# trend_query, seasonal_query") also matches the TS-encoder MLP, the alignment queries and the alignment attentions.
ALL_LORA_REGEX = (r'(' + LLM_LORA_REGEX + r')|(ts_encoder\.mlp\.(0|2|4|6|8))|(align\.(complete|trend|seasonal)_query)'
                  r'|(align\.\w+_attn\.(q|k|v|o)_proj)')

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json',
                   'added_tokens.json', 'processor_config.json', 'generation_config.json')


def warmup_cosine(total_steps, warmup_steps):
    """LR multiplier: linear warm-up to 1, then cosine decay to 0 (for ``LambdaLR``)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return f


# ----------------------------------------------------------------------------- forward
def _token_logprobs(lm_head, hidden, targets):
    logits = lm_head(hidden).float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)


def completion_logprobs(model, prompt_embeds, completion_ids, chunk=1024):
    """Teacher-forced ``log p(completion | prompt)`` for a right-padded batch.

    prompt_embeds: list of ``[Lp_i, D]`` (may carry gradients through the TS path); completion_ids: list of 1-D ids.
    Returns ``(logp [B, Tc] float32, mask [B, Tc] bool)`` on the embedding device. The vocabulary log-softmax is
    computed in checkpointed chunks so ``[B, Tc, 152k]`` logits are never materialised.
    """
    base = unwrap(model)
    dev = embedding_device(base)
    dtype = base.model.embed_tokens.weight.dtype
    seqs, prompt_lens, comp_lens = [], [], []
    for pe, c in zip(prompt_embeds, completion_ids):
        ce = model.get_input_embeddings()(c.to(dev)[None])[0]
        seqs.append(torch.cat([pe.to(dev, dtype), ce.to(dtype)], 0))
        prompt_lens.append(pe.shape[0])
        comp_lens.append(c.shape[0])
    B, L, D = len(seqs), max(s.shape[0] for s in seqs), seqs[0].shape[-1]
    x = torch.zeros(B, L, D, dtype=dtype, device=dev)
    attn = torch.zeros(B, L, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = s
        attn[i, :s.shape[0]] = 1
    pos = torch.arange(L, device=dev)[None].expand(B, -1)
    hidden = base.model(inputs_embeds=x, attention_mask=attn, position_ids=pos, use_cache=False,
                        return_dict=True).last_hidden_state

    hdev, Tc = hidden.device, max(comp_lens)
    mask = torch.zeros(B, Tc, dtype=torch.bool, device=hdev)
    targets = torch.zeros(B, Tc, dtype=torch.long, device=hdev)
    rows = []
    for i, (p, c) in enumerate(zip(prompt_lens, comp_lens)):
        mask[i, :c] = True
        targets[i, :c] = completion_ids[i].to(hdev)
        rows.append(torch.arange(p - 1, p - 1 + c, device=hdev) + i * L)  # position p-1+j predicts token j
    h_sel = hidden.reshape(B * L, D).index_select(0, torch.cat(rows))
    t_sel = targets[mask]
    parts = []
    for a in range(0, h_sel.shape[0], chunk):
        args = (base.lm_head, h_sel[a:a + chunk], t_sel[a:a + chunk])
        parts.append(checkpoint(_token_logprobs, *args, use_reentrant=False) if torch.is_grad_enabled()
                     else _token_logprobs(*args))
    logp = torch.zeros(B, Tc, dtype=torch.float32, device=hdev).masked_scatter(mask, torch.cat(parts))
    return logp.to(dev), mask.to(dev)


# ----------------------------------------------------------------------------- saving
def save_trainable(model, path, extra=None):
    """Only the trainable tensors (LoRA, plus the TS modules when they are trained)."""
    sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save({'state_dict': sd, 'extra': extra or {}}, path)


def save_full_fp16(model, out_dir, src_dir, shard_bytes=5 * 1024 ** 3, drop_prefixes=(), config_from_src=False):
    """Save a merged model as fp16 safetensors shards + index, with the config, modelling code and tokenizer files
    of ``src_dir``. (``save_pretrained`` needs torch>=2.5 under transformers 4.52.)

    drop_prefixes: parameters to leave out, e.g. the unused TS modules of the text-only model, which is then saved
    as a plain Qwen2 checkpoint with ``config_from_src=True``.
    """
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k, v in model.state_dict().items():
        if k.startswith(tuple(drop_prefixes)):
            continue
        t = v.detach().to('cpu', torch.float16).contiguous()
        if cur and size + t.numel() * 2 > shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = t
        size += t.numel() * 2
    shards.append(cur)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        fn = f'model-{i + 1:05d}-of-{len(shards):05d}.safetensors'
        save_file(shard, os.path.join(out_dir, fn), metadata={'format': 'pt'})
        for k, t in shard.items():
            weight_map[k] = fn
            total += t.numel() * 2
    if config_from_src:
        shutil.copy(os.path.join(src_dir, 'config.json'), os.path.join(out_dir, 'config.json'))
    else:
        model.config.torch_dtype = torch.float16
        model.config.save_pretrained(out_dir)
    for f in os.listdir(src_dir):
        if f.endswith(('.py', '.jinja', '.txt')) or f in TOKENIZER_FILES:
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
    # the index is written last: its presence marks a complete checkpoint
    with open(os.path.join(out_dir, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {'total_size': total}, 'weight_map': weight_map}, f, indent=1)
