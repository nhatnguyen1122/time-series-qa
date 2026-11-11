"""Rebuild base model + LoRA from a ``trainable.pt`` (written by the training scripts), merge, save as fp16.

Only needed when a run was trained with ``--no_merge`` or its merged copy was deleted.
"""
import argparse
import os
import sys

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.modeling import load_model, unwrap  # noqa: E402
from patra.training import ALL_LORA_REGEX, LLM_LORA_REGEX, save_full_fp16  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base', required=True, help='the --model_path the run was trained from')
    ap.add_argument('--trainable', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--targets', choices=['llm', 'all'], required=True, help='llm = SFT runs, all = GRPO runs')
    ap.add_argument('--train_ts_full', action='store_true', help='SFT runs: TS modules were trained in fp32')
    ap.add_argument('--max_memory', default='0:20GiB')
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    args = ap.parse_args()

    model, _ = load_model(args.base, args.max_memory, dtype=torch.bfloat16)
    targets = LLM_LORA_REGEX if args.targets == 'llm' else ALL_LORA_REGEX
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias='none',
                                             task_type='CAUSAL_LM', target_modules=targets))
    if args.train_ts_full:
        unwrap(model).ts_encoder.float()
        unwrap(model).align.float()
    sd = torch.load(args.trainable, map_location='cpu')['state_dict']
    params = dict(model.named_parameters())
    missing = [k for k in sd if k not in params]
    assert not missing, f'unknown tensors in {args.trainable}: {missing[:5]}'
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))
    print(f'loaded {len(sd)} tensors')
    save_full_fp16(model.merge_and_unload(), args.out, args.base)
    print('saved', args.out)


if __name__ == '__main__':
    main()
