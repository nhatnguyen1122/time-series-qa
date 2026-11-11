"""Stage 1 (alignment): supervised fine-tuning of a PATRA-architecture model on PATRA-TRAIN ``sft.jsonl``.

Follows the authors' ``train_patra_sft.sh`` where it is specified: qwen chat template with the default system prompt,
loss on the response (+ ``<|im_end|>``) only, LoRA r=8 / alpha=16 / dropout 0.05 on every LLM linear layer, cosine
schedule with warm-up. Because PATRA-init is not released, the TS encoder and alignment module start from scratch and
are trained fully (fp32) instead of with LoRA.

Input variants
  default         normalised series from sft.jsonl through the TS encoder (round 1)
  --raw_ts F      take the raw series from F (grpo.jsonl, row-aligned) and normalise them on the fly
  --offset_scale  prefix each ``<ts><ts/>`` with ``[Value Offset: ..|Value Scaling: ..]`` (needs --raw_ts)
  --text_ts       no TS encoder: the raw series is written into the prompt as numbers (needs --raw_ts); the merged
                  model is saved as a plain Qwen2 checkpoint taken from --text_save_src

The run checkpoints every --save_every steps and continues with --resume; a micro-batch that hits CUDA OOM is split
in halves, which leaves the gradient unchanged.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.data import load_sft_rows, split_train_val  # noqa: E402
from patra.modeling import load_model, prompt_embedding, unwrap  # noqa: E402
from patra.prompts import chat_prompt  # noqa: E402
from patra.series import add_offset_scale, serialize_text, ts_tensor  # noqa: E402
from patra.training import (LLM_LORA_REGEX, completion_logprobs, save_full_fp16, save_trainable,  # noqa: E402
                                  warmup_cosine)
from patra.utils import append_jsonl, atomic_torch_save, seed_all  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model_path', required=True, help='PATRA-architecture init checkpoint (see build_init.py)')
    ap.add_argument('--data', required=True, help='sft.jsonl')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_memory', default='0:20GiB', help='per-GPU budget for the decoder layers')
    ap.add_argument('--ts_device', type=int, default=None, help='GPU for TS encoder/alignment/embeddings')
    ap.add_argument('--reinit_ts', action='store_true', help='fresh TS encoder + alignment (no PATRA-init)')
    g = ap.add_argument_group('input variant')
    g.add_argument('--raw_ts', default=None, help='row-aligned jsonl with raw series (grpo.jsonl)')
    g.add_argument('--offset_scale', action='store_true')
    g.add_argument('--text_ts', action='store_true')
    g.add_argument('--text_save_src', default=None, help='plain Qwen2 dir: config/tokenizer for the text-only model')
    g = ap.add_argument_group('optimisation')
    g.add_argument('--n_train', type=int, default=None)
    g.add_argument('--n_val', type=int, default=256)
    g.add_argument('--micro_bs', type=int, default=8)
    g.add_argument('--grad_accum', type=int, default=4)
    g.add_argument('--epochs', type=float, default=1.0)
    g.add_argument('--max_steps', type=int, default=None)
    g.add_argument('--lr_lora', type=float, default=1e-4)
    g.add_argument('--lr_ts', type=float, default=1e-4)
    g.add_argument('--warmup_ratio', type=float, default=0.03)
    g.add_argument('--lora_r', type=int, default=8)
    g.add_argument('--lora_alpha', type=int, default=16)
    g.add_argument('--lora_dropout', type=float, default=0.05)
    g.add_argument('--cutoff_len', type=int, default=3000)
    g.add_argument('--seed', type=int, default=42)
    g = ap.add_argument_group('bookkeeping')
    g.add_argument('--eval_every', type=int, default=100)
    g.add_argument('--save_every', type=int, default=0)
    g.add_argument('--resume', action='store_true')
    g.add_argument('--stop_after', type=int, default=None, help='debug: exit after this many steps')
    g.add_argument('--no_merge', action='store_true')
    args = ap.parse_args()
    if (args.offset_scale or args.text_ts) and not args.raw_ts:
        ap.error('--offset_scale/--text_ts need --raw_ts')
    if args.text_ts and not args.text_save_src and not args.no_merge:
        ap.error('--text_ts needs --text_save_src (or --no_merge)')
    return args


class Encoder:
    """Turns a training row into ``(prompt, series tensor or None, completion ids)`` for the chosen input variant."""

    def __init__(self, tok, args):
        self.tok, self.args = tok, args
        self.im_end = tok.convert_tokens_to_ids('<|im_end|>')

    def __call__(self, row):
        a = self.args
        if a.text_ts:
            prompt, ts = chat_prompt(serialize_text(row['input'], row['raw_ts'])), None
        elif a.raw_ts:
            text = add_offset_scale(row['input'], row['raw_ts']) if a.offset_scale else row['input']
            prompt, ts = chat_prompt(text), (ts_tensor(row['raw_ts']) if row['raw_ts'] else None)
        else:
            prompt = chat_prompt(row['input'])
            ts = ts_tensor(row['timeseries'], pre_encoded=True) if row['timeseries'] else None
        return prompt, ts, torch.tensor(self.completion_ids(row))

    def completion_ids(self, row):
        return self.tok(row['output'], add_special_tokens=False)['input_ids'][:self.args.cutoff_len] + [self.im_end]


def sum_logprob(model, tok, encoded):
    """Sum of completion log-probs and number of completion tokens for a list of encoded rows."""
    embeds, comps = [], []
    for prompt, ts, comp in encoded:
        embeds.append(prompt_embedding(model, tok, prompt, ts)[0])
        comps.append(comp)
    logp, mask = completion_logprobs(model, embeds, comps)
    return logp.sum(), int(mask.sum())


def setup_model(args):
    model, tok = load_model(args.model_path, args.max_memory, dtype=torch.bfloat16, ts_device=args.ts_device,
                            reinit_ts=args.reinit_ts, seed=args.seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=LLM_LORA_REGEX, bias='none', task_type='CAUSAL_LM'))
    ts_params = []
    if not args.text_ts:  # TS modules: trained fully, in fp32
        for module in (unwrap(model).ts_encoder, unwrap(model).align):
            module.float()
            for p in module.parameters():
                p.requires_grad_(True)
                ts_params.append(p)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    print(f'trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.1f}M, '
          f'TS encoder+alignment {sum(p.numel() for p in ts_params) / 1e6:.1f}M', flush=True)
    return model, tok, lora_params, ts_params


def main():
    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'args.json'), 'w'), indent=1)
    log_path = os.path.join(args.out_dir, 'train_log.jsonl')
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    resuming = args.resume and os.path.exists(ckpt_path)
    if os.path.exists(log_path) and not resuming:
        os.remove(log_path)

    train, val = split_train_val(load_sft_rows(args.data, args.raw_ts), args.n_val, args.seed)
    train = train[:args.n_train] if args.n_train else train
    print(f'train {len(train)}  val {len(val)}', flush=True)

    model, tok, lora_params, ts_params = setup_model(args)
    groups = [{'params': lora_params, 'lr': args.lr_lora, 'weight_decay': 0.0}]
    if ts_params:
        groups.append({'params': ts_params, 'lr': args.lr_ts, 'weight_decay': 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    total = args.max_steps or max(1, int(len(train) // (args.micro_bs * args.grad_accum) * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, max(1, int(total * args.warmup_ratio))))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    encode = Encoder(tok, args)

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(val), args.micro_bs):
            s, k = sum_logprob(model, tok, [encode(r) for r in val[i:i + args.micro_bs]])
            tot, n = tot - s.item(), n + k
        model.train()
        return tot / n

    step, ptr, epoch = 0, 0, 0
    if resuming:
        ck = torch.load(ckpt_path, map_location='cpu')
        named = dict(trainable)
        for k, v in ck['params'].items():
            named[k].data.copy_(v.to(named[k].device, named[k].dtype))
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        if 'torch_rng' in ck:  # dropout masks continue exactly where they stopped
            torch.set_rng_state(ck['torch_rng'])
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        step, ptr, epoch = ck['step'], ck['ptr'], ck['epoch']
        for e in range(1, epoch + 1):  # replay the per-epoch reshuffles
            random.Random(args.seed + e).shuffle(train)
        print(f'resumed from step {step}', flush=True)
        del ck

    def save_ckpt():
        atomic_torch_save({'params': {n: p.detach().cpu() for n, p in trainable}, 'opt': opt.state_dict(),
                           'sched': sched.state_dict(), 'step': step, 'ptr': ptr, 'epoch': epoch,
                           'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, ckpt_path)

    def backward_microbatch(batch, ntok):
        """Accumulate ``-sum(logp) / (ntok * grad_accum)``; on OOM split the failing piece in two and retry it
        (pieces already done keep their gradients, so the result equals the unsplit micro-batch)."""
        pieces, i, total_nll = [batch], 0, 0.0
        while i < len(pieces):
            try:
                s, _ = sum_logprob(model, tok, [encode(r) for r in pieces[i]])
                (-s / (ntok * args.grad_accum)).backward()
                total_nll -= s.item()
                i += 1
            except torch.cuda.OutOfMemoryError:
                s = None
                torch.cuda.empty_cache()
                if len(pieces[i]) == 1:
                    print('OOM at micro-batch size 1; waiting 120s', flush=True)
                    time.sleep(120)
                else:
                    h = len(pieces[i]) // 2
                    pieces[i:i + 1] = [pieces[i][:h], pieces[i][h:]]
                    print(f'OOM -> micro-batch split into {len(pieces)} pieces', flush=True)
        return total_nll

    model.train()
    t0 = time.time()
    if step == 0:
        v = val_loss()
        print(f'step 0 val_loss {v:.4f}', flush=True)
        append_jsonl(log_path, {'step': 0, 'val_loss': v})
    ntok_window, t_window = 0, time.time()
    while step < total:
        micro = []  # the grad_accum micro-batches of this optimizer step (fixed data order)
        for _ in range(args.grad_accum):
            if ptr + args.micro_bs > len(train):
                ptr, epoch = 0, epoch + 1
                random.Random(args.seed + epoch).shuffle(train)
            micro.append(train[ptr:ptr + args.micro_bs])
            ptr += args.micro_bs
        loss = 0.0
        for batch in micro:  # loss = token-mean within a micro-batch, averaged over micro-batches
            ntok = sum(len(encode.completion_ids(r)) for r in batch)
            loss += backward_microbatch(batch, ntok) / ntok / args.grad_accum
            ntok_window += ntok
        gn_lora = torch.nn.utils.clip_grad_norm_(lora_params, 1.0).item()
        gn_ts = torch.nn.utils.clip_grad_norm_(ts_params, 1.0).item() if ts_params else 0.0
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        row = {'step': step, 'loss': loss, 'lr_lora': sched.get_last_lr()[0], 'grad_norm_lora': gn_lora,
               'grad_norm_ts': gn_ts, 'epoch': epoch + ptr / len(train), 'elapsed_s': time.time() - t0,
               'max_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30}
        if step % 10 == 0:
            row['tok_per_s'] = ntok_window / (time.time() - t_window)
            ntok_window, t_window = 0, time.time()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if step % args.eval_every == 0 or step == total:
            row['val_loss'] = val_loss()
            print(f'step {step} val_loss {row["val_loss"]:.4f}', flush=True)
        append_jsonl(log_path, row)
        if args.save_every and step % args.save_every == 0 and step < total:
            save_ckpt()
        if args.stop_after and step >= args.stop_after:
            print('stop_after reached', flush=True)
            return

    save_trainable(model, os.path.join(args.out_dir, 'trainable.pt'), {'args': vars(args)})
    if not args.no_merge:
        merged_dir = os.path.join(args.out_dir, 'merged')
        model = model.merge_and_unload()
        if args.text_ts:
            save_full_fp16(model, merged_dir, args.text_save_src, drop_prefixes=('ts_encoder.', 'align.'),
                           config_from_src=True)
        else:
            save_full_fp16(model, merged_dir, args.model_path)
        print('saved merged model to', merged_dir, flush=True)


if __name__ == '__main__':
    main()
