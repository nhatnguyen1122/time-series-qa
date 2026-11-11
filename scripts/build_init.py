"""Build a stand-in for the unreleased PATRA-init checkpoint.

The result has PATRA's architecture, config and tokenizer (copied from the released PATRA-7B) and the LLM weights of
Qwen2.5-7B-Instruct (symlinked, no copy). The TS encoder and alignment weights are absent and are initialised at
load time (``train_sft.py --reinit_ts``). ``--fixes`` selects code fixes in ``patra_model/modeling_qwen2.py``; they are
recorded in config.json as ``patra_fixes`` and read by the modelling code.

    python scripts/build_init.py --patra models/PATRA-7B --qwen models/Qwen2.5-7B-Instruct --out models/init-released
    python scripts/build_init.py ... --out models/init-fixed --fixes noncausal_align_attn trend_seasonal_pairing \
        per_series_decomp
"""
import argparse
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXES = ['noncausal_align_attn', 'trend_seasonal_pairing', 'per_series_decomp', 'residual_alignment']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--patra', required=True, help='released PATRA-7B directory (config, code, tokenizer)')
    ap.add_argument('--qwen', required=True, help='Qwen2.5-7B-Instruct directory (LLM weights)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fixes', nargs='*', default=[], choices=FIXES)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.patra):  # config, modelling/processing code, tokenizer
        if f.endswith(('.py', '.json', '.jinja', '.txt')) and not f.startswith('model.safetensors'):
            shutil.copy(os.path.join(args.patra, f), os.path.join(args.out, f))
    # the modelling code of this repository (identical to the release when no fix is selected)
    shutil.copy(os.path.join(REPO, 'patra_model', 'modeling_qwen2.py'), os.path.join(args.out, 'modeling_qwen2.py'))
    cfg_path = os.path.join(args.out, 'config.json')
    cfg = json.load(open(cfg_path))
    cfg['patra_fixes'] = args.fixes
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    for f in os.listdir(args.qwen):  # LLM weights
        if f.endswith('.safetensors') or f == 'model.safetensors.index.json':
            dst = os.path.join(args.out, f)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(os.path.join(args.qwen, f)), dst)
    print(f'{args.out}: fixes={args.fixes}')


if __name__ == '__main__':
    main()
