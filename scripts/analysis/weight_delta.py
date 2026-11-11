"""What is the released PATRA-7B LLM? Compare its weights with Qwen2.5-7B-Instruct.

Reports the relative difference of embeddings / lm_head / final norm and, for a sample of decoder layers, the
effective rank (99% / 99.9% of the energy) of the weight difference: a merged LoRA of rank r leaves a rank-r delta.

    python scripts/analysis/weight_delta.py models/PATRA-7B models/Qwen2.5-7B-Instruct runs/analysis/weight_delta.json
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

LAYERS = [0, 7, 14, 21, 27]
MATRICES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


class Weights:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))['weight_map']

    def __getitem__(self, key):
        with safe_open(os.path.join(self.dir, self.map[key]), 'pt') as f:
            return f.get_tensor(key).float()

    def numel(self, key):
        with safe_open(os.path.join(self.dir, self.map[key]), 'pt') as f:
            return torch.Size(f.get_slice(key).get_shape()).numel()


def effective_rank(sv, energy):
    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    return int((e < energy).sum().item()) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('patra')
    ap.add_argument('qwen')
    ap.add_argument('out')
    args = ap.parse_args()
    A, B = Weights(args.patra), Weights(args.qwen)
    extra = [k for k in A.map if k not in B.map]
    res = {'only_in_patra': sorted({'.'.join(k.split('.')[:2]) for k in extra}),
           'n_params_only_in_patra': sum(A.numel(k) for k in extra), 'rows': []}
    for layer in LAYERS:
        for m in MATRICES:
            k = f'model.layers.{layer}.{m}.weight'
            wa, wb = A[k], B[k]
            sv = torch.linalg.svdvals(wa - wb)
            row = {'param': k, 'rel_delta': ((wa - wb).norm() / wb.norm()).item(), 'rank99': effective_rank(sv, 0.99),
                   'rank999': effective_rank(sv, 0.999), 'top_sv': [round(x, 4) for x in sv[:20].tolist()]}
            res['rows'].append(row)
            print(k, f"rel={row['rel_delta']:.2e} rank99={row['rank99']} rank999={row['rank999']}", flush=True)
    for k in ['model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight']:
        res['rows'].append({'param': k, 'rel_delta': ((A[k] - B[k]).norm() / B[k].norm()).item()})
        print(k, res['rows'][-1]['rel_delta'])
    print('only in PATRA:', res['only_in_patra'], 'params:', res['n_params_only_in_patra'])
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
