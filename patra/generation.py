"""Batched decoding over precomputed input embeddings (KV cache, left padding).

``model.generate`` cannot be used with batches of PATRA inputs (see ``modeling``), so each prompt's embeddings are
built separately and only the LLM is batched here. Greedy decoding with the checkpoint's repetition penalty reproduces
the official ``model.generate(do_sample=False)`` output (checked by ``scripts/verify_batched_eval.py``); sampling with
temperature / top-p is used for GRPO rollouts.
"""
import torch
from transformers.cache_utils import DynamicCache

from .modeling import embedding_device, unwrap


def _apply_repetition_penalty(logits, seen, penalty):
    """HF-style penalty on every token already in the prompt or the generated prefix."""
    penalised = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalised, logits)


def _sample(logits, temperature, top_p, generator):
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sorted_p, sorted_idx = probs.sort(-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_p * keep)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, items, max_new_tokens, eos_ids, pad_id, repetition_penalty=1.0, temperature=0.0, top_p=1.0,
             generator=None):
    """Decode a batch of prompts.

    items: list of ``(prompt_embeds [L, D], prompt_ids [L])``. ``temperature == 0`` means greedy decoding.
    Returns a CPU LongTensor ``[B, T]``; positions after a sequence's EOS hold ``pad_id``.
    """
    base = unwrap(model)
    dev = embedding_device(base)
    dtype = base.model.embed_tokens.weight.dtype
    B, lens = len(items), [e.shape[0] for e, _ in items]
    L, D = max(lens), items[0][0].shape[-1]

    embeds = torch.zeros(B, L, D, dtype=dtype, device=dev)
    attn = torch.zeros(B, L, dtype=torch.long, device=dev)
    for i, (e, _) in enumerate(items):  # left padding
        embeds[i, L - lens[i]:] = e.to(dev, dtype)
        attn[i, L - lens[i]:] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)

    seen = None
    if repetition_penalty != 1.0:
        seen = torch.zeros(B, base.lm_head.out_features, dtype=torch.bool, device=dev)
        for i, (_, ids) in enumerate(items):
            seen[i, ids.to(dev)] = True
    eos = torch.tensor(eos_ids, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    out = torch.full((B, max_new_tokens), pad_id, dtype=torch.long, device=dev)

    h = base.model(inputs_embeds=embeds, attention_mask=attn, position_ids=pos, past_key_values=DynamicCache(),
                   use_cache=True, return_dict=True)
    cache, last, cur = h.past_key_values, h.last_hidden_state[:, -1], pos[:, -1]
    for t in range(max_new_tokens):
        logits = base.lm_head(last).float().to(dev)
        if seen is not None:
            logits = _apply_repetition_penalty(logits, seen, repetition_penalty)
        nxt = _sample(logits, temperature, top_p, generator) if temperature > 0 else logits.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        out[:, t] = nxt
        if seen is not None:
            seen[torch.arange(B, device=dev), nxt] = True
        finished |= torch.isin(nxt, eos)
        if finished.all():
            return out[:, :t + 1].cpu()
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=dev)], 1)
        cur = cur + 1
        h = base.model(inputs_embeds=base.model.embed_tokens(nxt[:, None]), attention_mask=attn,
                       position_ids=cur[:, None], past_key_values=cache, use_cache=True, return_dict=True)
        cache, last = h.past_key_values, h.last_hidden_state[:, -1]
    return out.cpu()
