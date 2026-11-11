"""What does the alignment module inject into the LLM? CPU diagnostics of a checkpoint's TS path.

Loads only the TS encoder, the alignment module and the token embeddings (fp32), using the checkpoint's own modelling
code, and runs them on a grid of Q multiple-choice/true-false eval questions x S eval series (single series of 24
points = 2 patches; MC/TF only, so all prompts share one system prompt and the question effect is the question text).

  1. text-query attention: mass on the first 10 prompt tokens, normalised entropy (1 = uniform), top tokens
  2. two-way variance decomposition of the aligned TS token (mean over patches): question / series / interaction
  3. constant share: fraction of ||token||^2 explained by the grid mean, i.e. by a vector that depends on neither
     the series nor the question; reported for the TS-encoder output and for the aligned output
  4. relative change of the token when the series (or the question) is swapped; token norm vs text-embedding norm

    python scripts/analysis/alignment_diagnostics.py MODEL_DIR OUT_JSON [--offset_scale] [--ckpt runs/x/ckpt.pt]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from patra.data import load_eval_samples  # noqa: E402
from patra.prompts import CLOSED_FORMATS, eval_prompt  # noqa: E402
from patra.series import TS_PLACEHOLDER, add_offset_scale, ts_tensor  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_ts_path(model_dir, ckpt=None):
    """(TS encoder, alignment module, token embedding matrix, tokenizer), fp32 on CPU."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    M = sys.modules[get_class_from_dynamic_module('modeling_qwen2.Qwen2TSForCausalLM', model_dir).__module__]
    align, enc = M.MultiModalAlignment(cfg).float().eval(), M.TimeSeriesEmbedding(cfg.ts).float().eval()
    weight_map = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))['weight_map']
    sd = {}
    for k, f in weight_map.items():
        if k.startswith(('align.', 'ts_encoder.')) or k == 'model.embed_tokens.weight':
            with safe_open(os.path.join(model_dir, f), 'pt') as fh:
                sd[k] = fh.get_tensor(k).float()
    if ckpt:  # an intermediate SFT checkpoint: override the TS-module weights
        ck = torch.load(ckpt, map_location='cpu')
        for k, v in ck.get('params', ck.get('state_dict')).items():
            k = k.replace('base_model.model.', '')
            if k.startswith(('align.', 'ts_encoder.')):
                sd[k] = v.float()
    align.load_state_dict({k[len('align.'):]: v for k, v in sd.items() if k.startswith('align.')})
    enc.load_state_dict({k[len('ts_encoder.'):]: v for k, v in sd.items() if k.startswith('ts_encoder.')})
    return enc, align, sd['model.embed_tokens.weight'], AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def pick_grid(data_dir, n_questions=20, n_series=20, seed=0):
    samples = [s for _, _, s in load_eval_samples(data_dir)
               if len(s['timeseries']) == 1 and len(s['timeseries'][0]) == 24 and s['input'].count(TS_PLACEHOLDER) == 1]
    pick = np.random.RandomState(seed).choice(len(samples), 120, replace=False)
    questions = [samples[i] for i in pick if samples[i]['question_format'] in CLOSED_FORMATS][:n_questions]
    series = [samples[i]['timeseries'][0] for i in pick[-n_series:]]
    return questions, series


def constant_share(X):
    """Share of the mean squared norm explained by the mean vector, for rows X [N, D]."""
    mu = X.reshape(-1, X.shape[-1]).mean(0)
    return float((mu ** 2).sum() / (X ** 2).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('model_dir')
    ap.add_argument('out_json')
    ap.add_argument('--offset_scale', action='store_true', help='the model was trained with the value prefix')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--data_dir', default=os.path.join(REPO, 'data', 'PATRA-EVAL'))
    args = ap.parse_args()
    torch.manual_seed(0)
    enc, align, E, tok = load_ts_path(args.model_dir, args.ckpt)
    Qs, Ss = pick_grid(args.data_dir)

    def run(q, ts):
        text = add_offset_scale(q['input'], [ts]) if args.offset_scale else q['input']
        ids = tok(eval_prompt(dict(q, input=text)), return_tensors='pt', add_special_tokens=False)['input_ids']
        with torch.no_grad():
            feat, pc = enc(ts_tensor([ts]).reshape(1, -1, 1))
            aligned, _ = align(ids, E[ids], feat, pc)
        return ids, feat, aligned  # [n_patches, D]

    # 1. text-query attention
    mass10, entropy, top = [], [], Counter()
    for q in Qs:
        ids, _, _ = run(q, Ss[0])
        te = E[ids]
        for name in ('complete', 'trend', 'seasonal'):
            queries = getattr(align, f'{name}_query')(torch.eye(align.num_align_tokens)).unsqueeze(0)
            with torch.no_grad():
                _, w = getattr(align, f'{name}_text_attn')(queries, te, te, need_weights=True)
            w = w[0].clamp_min(1e-12)  # [10, L]
            mass10.append(w[:, :10].sum(-1).mean().item())
            entropy.append((-(w * w.log()).sum(-1)).mean().item() / np.log(w.shape[-1]))
            top.update(tok.decode(ids[0, j]) for j in w.mean(0).topk(3).indices.tolist())

    # 2.-4. grid of aligned tokens (and encoder outputs) for every (question, series)
    G = np.zeros((len(Qs), len(Ss), E.shape[1]))
    enc_out = np.zeros((len(Ss), E.shape[1]))
    for j, s in enumerate(Ss):
        for i, q in enumerate(Qs):
            _, feat, aligned = run(q, s)
            G[i, j] = aligned.mean(0).numpy()
        enc_out[j] = feat.mean(0).numpy()
    mu = G.mean((0, 1), keepdims=True)
    row, col = G.mean(1, keepdims=True) - mu, G.mean(0, keepdims=True) - mu
    total = ((G - mu) ** 2).sum()
    variance = {'question': float((row ** 2).sum() * len(Ss) / total),
                'series': float((col ** 2).sum() * len(Qs) / total),
                'interaction': float(((G - mu - row - col) ** 2).sum() / total)}
    live = E[:151643][E[:151643].norm(dim=-1) > 0]  # text-token embeddings that are actually used
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a))
    res = {'model': args.model_dir, 'offset_scale_prompt': args.offset_scale,
           'patra_fixes': getattr(align.config, 'patra_fixes', None),
           'text_attn_mass_first10_tokens': float(np.mean(mass10)), 'text_attn_norm_entropy': float(np.mean(entropy)),
           'text_attn_top_tokens': top.most_common(12), 'variance_fraction': variance,
           'constant_share_encoder_output': constant_share(enc_out), 'constant_share_aligned': constant_share(G),
           'encoder_output_norm': float(np.linalg.norm(enc_out, axis=-1).mean()),
           'aligned_token_norm': float(np.linalg.norm(G, axis=-1).mean()),
           'text_embedding_norm': float(live[torch.randint(0, live.shape[0], (2000,))].norm(dim=-1).mean()),
           'rel_change_other_series': float(np.mean([rel(G[i, 0], G[i, 1]) for i in range(len(Qs))])),
           'rel_change_other_question': float(np.mean([rel(G[0, j], G[1, j]) for j in range(len(Ss))]))}
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
