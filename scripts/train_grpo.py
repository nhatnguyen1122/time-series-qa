"""Stage 2 (reasoning): GRPO with the task-aware balanced reward, starting from a merged SFT checkpoint.

The authors' trainer (``train/grpo/trainer.py``) is imported by their workflow but not released; this is a
re-implementation that follows their workflow and the vendored-TRL ``GRPOConfig`` defaults where they are specified:
  * prompt = chat template + question + the format suffix for its question type; raw series -> sp_encoding
  * reward = xmlcount (format, [0, 0.75]) + correctness (task, [0, 2]), see ``patra.rewards``
  * LoRA on the authors' ``lora_target`` list (LLM + TS-encoder MLP + alignment queries/attentions)
  * temperature 1.0, beta 0 (no KL term; the KL to the SFT model is still logged), one policy update per batch
    (ratio = 1), group-normalised advantages, token-level 'bnpo' loss
  * max prompt / completion length 512
Budget-driven choices: G = 4 generations per prompt and 200 steps x 8 prompts.
``--reward original`` runs the paper's "Original Reward" ablation.
"""
import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from patra.data import split_train_val  # noqa: E402
from patra.generation import generate  # noqa: E402
from patra.modeling import embedding_device, load_model, prompt_embedding  # noqa: E402
from patra.prompts import chat_prompt  # noqa: E402
from patra.rewards import CORRECT_THRESHOLD, PROMPT_SUFFIX, TASK_REWARDS, xmlcount_reward_func  # noqa: E402
from patra.series import ts_tensor  # noqa: E402
from patra.training import (ALL_LORA_REGEX, completion_logprobs, save_full_fp16, save_trainable,  # noqa: E402
                                  warmup_cosine)
from patra.utils import append_jsonl, atomic_torch_save, is_oom, read_jsonl, seed_all  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model_path', required=True, help='merged SFT checkpoint')
    ap.add_argument('--data', required=True, help='grpo.jsonl')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_memory', default='0:20GiB')
    ap.add_argument('--ts_device', type=int, default=None)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--prompts_per_step', type=int, default=8)
    ap.add_argument('--num_generations', type=int, default=4)
    ap.add_argument('--micro_bs', type=int, default=8, help='sequences per backward pass (halved on OOM)')
    ap.add_argument('--gen_chunk', type=int, default=32, help='sequences per generate() call (halved on OOM)')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--warmup_ratio', type=float, default=0.1)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=0.0, help='KL coefficient (0 = as in the authors\' config)')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max_completion_length', type=int, default=512)
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--reward', choices=sorted(TASK_REWARDS), default='balanced')
    ap.add_argument('--skip_rows', type=int, default=256, help='rows of the shuffled set held out as SFT validation')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--kl_log_every', type=int, default=5)
    ap.add_argument('--sample_log_every', type=int, default=10)
    ap.add_argument('--save_every', type=int, default=10, help='resumable checkpoint interval (steps)')
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args()


def rollout(model, tok, batch, args, gen, gen_chunk, eos_ids):
    """Sample G completions per prompt. Returns prompts, series, completion ids (up to and including EOS), texts."""
    prompts, series, items = [], [], []
    with torch.no_grad():
        for r in batch:
            p = chat_prompt(r['input'] + PROMPT_SUFFIX[r['question_format']])
            ts = ts_tensor(r['timeseries'])
            emb, ids = prompt_embedding(model, tok, p, ts)
            prompts.append(p)
            series.append(ts)
            items += [(emb, ids)] * args.num_generations
        outs = [generate(model, items[c:c + gen_chunk], args.max_completion_length, eos_ids, tok.pad_token_id,
                         temperature=args.temperature, generator=gen) for c in range(0, len(items), gen_chunk)]
    T = max(o.shape[1] for o in outs)
    out = torch.cat([torch.nn.functional.pad(o, (0, T - o.shape[1]), value=tok.pad_token_id) for o in outs])
    eos = torch.tensor(eos_ids)
    comps, texts = [], []
    for row in out:  # completion = tokens up to and including the first EOS (as TRL trains on it); else truncated
        hit = torch.isin(row, eos).nonzero()
        ids = row[:int(hit[0]) + 1] if len(hit) else row
        comps.append(ids)
        texts.append(tok.decode(ids, skip_special_tokens=True))
    return prompts, series, comps, texts


def group_advantages(rewards, G):
    """(r - group mean) / (group std + 1e-4), groups = the G completions of one prompt."""
    grp = rewards.reshape(-1, G)
    adv = (grp - grp.mean(1, keepdims=True)) / (grp.std(1, ddof=1, keepdims=True) + 1e-4)
    return torch.tensor(adv.reshape(-1), dtype=torch.float32), grp


def policy_loss_backward(model, tok, prompts, series, comps, adv, args, G, micro_bs, with_kl):
    """Token-level policy-gradient loss (ratio = 1, so the gradient is -A * grad log p), normalised by the total
    number of completion tokens ('bnpo'); optionally the k3 KL estimate to the adapter-free (SFT) model.
    Returns (KL sum, KL token count)."""
    n_tok_total = sum(int(c.shape[0]) for c in comps)
    kl_sum, kl_n = 0.0, 0
    for a in range(0, len(comps), micro_bs):
        idx = list(range(a, min(a + micro_bs, len(comps))))
        embeds = [prompt_embedding(model, tok, prompts[i // G], series[i // G])[0] for i in idx]
        lp, mask = completion_logprobs(model, embeds, [comps[i] for i in idx])
        A = adv[idx].to(lp.device)[:, None]
        per_tok = -(torch.exp(lp - lp.detach()) * A)
        if with_kl:
            with torch.no_grad(), model.disable_adapter():
                model.eval()
                ref_embeds = [prompt_embedding(model, tok, prompts[i // G], series[i // G])[0] for i in idx]
                ref_lp, _ = completion_logprobs(model, ref_embeds, [comps[i] for i in idx])
                model.train()
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1
            kl_sum += float((kl.detach() * mask).sum())
            kl_n += int(mask.sum())
            if args.beta > 0:
                per_tok = per_tok + args.beta * kl
        ((per_tok * mask).sum() / n_tok_total).backward()
    return kl_sum, kl_n


def step_log(step, rew, r_fmt, r_task, grp, comps, ttypes, args, gn, lr, timings, kl):
    lens = np.array([c.shape[0] for c in comps])
    row = {'step': step, 'reward': float(rew.mean()), 'reward_std': float(rew.std()),
           'format_reward': float(r_fmt.mean()), 'task_reward': float(r_task.mean()),
           'zero_std_groups': float((grp.std(1) < 1e-6).mean()), 'completion_len': float(lens.mean()),
           'clipped_frac': float((lens >= args.max_completion_length).mean()), 'grad_norm': gn, 'lr': lr, **timings}
    for f in PROMPT_SUFFIX:
        m = np.array([t == f for t in ttypes])
        if m.any():
            row[f'task_reward/{f}'] = float(r_task[m].mean())
            row[f'n/{f}'] = int(m.sum())
            if f != 'open_ended_question':
                row[f'acc/{f}'] = float((r_task[m] >= CORRECT_THRESHOLD[args.reward] - 1e-9).mean())
    if kl[1]:
        row['kl_to_sft'] = kl[0] / kl[1]
    return row


def main():
    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'args.json'), 'w'), indent=1)
    log_path, samp_path, ckpt_path = (os.path.join(args.out_dir, f) for f in
                                      ('train_log.jsonl', 'samples.jsonl', 'ckpt.pt'))
    if not args.resume:
        for p in (log_path, samp_path, ckpt_path):
            if os.path.exists(p):
                os.remove(p)

    rows, _ = split_train_val(read_jsonl(args.data), args.skip_rows, args.seed)  # same shuffle as SFT, minus its val
    data = rows[:args.steps * args.prompts_per_step]
    print(f'GRPO prompts used: {len(data)} / {len(rows)}', flush=True)

    model, tok = load_model(args.model_path, args.max_memory, dtype=torch.bfloat16, ts_device=args.ts_device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=ALL_LORA_REGEX, bias='none', task_type='CAUSAL_LM'))
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    params = list(named.values())
    print(f'trainable LoRA params {sum(p.numel() for p in params) / 1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    warmup = max(1, int(args.steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.steps, warmup))
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    gen = torch.Generator(device=embedding_device(model)).manual_seed(args.seed)
    G = args.num_generations
    t_start, step, micro_bs, gen_chunk = time.time(), 0, args.micro_bs, args.gen_chunk

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for n, v in ck['lora'].items():
                named[n].copy_(v.to(named[n].device, named[n].dtype))
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        gen.set_state(ck['gen'])
        torch.set_rng_state(ck['torch_rng'])
        np.random.set_state(ck['np_rng'])
        random.setstate(ck['py_rng'])
        step, t_start = ck['step'], t_start - ck['elapsed_s']
        if os.path.exists(log_path):  # drop log rows written after the checkpoint
            keep = [line for line in open(log_path) if json.loads(line)['step'] <= step]
            open(log_path, 'w').writelines(keep)
        print(f'resumed from step {step}', flush=True)

    def save_ckpt():
        atomic_torch_save({'lora': {n: p.detach().cpu() for n, p in named.items()}, 'opt': opt.state_dict(),
                           'sched': sched.state_dict(), 'gen': gen.get_state(), 'torch_rng': torch.get_rng_state(),
                           'np_rng': np.random.get_state(), 'py_rng': random.getstate(), 'step': step,
                           'elapsed_s': time.time() - t_start}, ckpt_path)

    oom_retries = 0
    while step < args.steps:
        batch = data[step * args.prompts_per_step:(step + 1) * args.prompts_per_step]
        gen_state = gen.get_state()
        try:
            t0 = time.time()
            model.eval()  # rollouts without dropout
            prompts, series, comps, texts = rollout(model, tok, batch, args, gen, gen_chunk, eos_ids)
            t_gen = time.time() - t0

            expanded = [r for r in batch for _ in range(G)]
            completions = [[{'content': t}] for t in texts]
            answers, ttypes = [r['answer'] for r in expanded], [r['question_format'] for r in expanded]
            r_fmt = np.array(xmlcount_reward_func(completions))
            r_task = np.array(TASK_REWARDS[args.reward](expanded, completions, answers, ttypes))
            rew = r_fmt + r_task
            adv, grp = group_advantages(rew, G)

            model.train()
            t1 = time.time()
            kl = policy_loss_backward(model, tok, prompts, series, comps, adv, args, G, micro_bs,
                                      with_kl=args.beta > 0 or step % args.kl_log_every == 0)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            oom = False
        except Exception as e:  # shared GPUs: other jobs may grab memory mid-run -> retry the step with smaller chunks
            if not is_oom(e):
                raise
            oom = True
        if oom:  # outside the except block, so the traceback (and the activations it references) is already freed
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            gen.set_state(gen_state)
            oom_retries += 1
            micro_bs, gen_chunk = max(2, micro_bs // 2), max(4, gen_chunk // 2)
            print(f'OOM at step {step + 1} (retry {oom_retries}); micro_bs={micro_bs} gen_chunk={gen_chunk}',
                  flush=True)
            if oom_retries > 30:
                raise RuntimeError('persistent OOM')
            if oom_retries > 3:
                time.sleep(120)
            continue

        row = step_log(step + 1, rew, r_fmt, r_task, grp, comps, ttypes, args, gn, sched.get_last_lr()[0],
                       {'t_gen': t_gen, 't_train': time.time() - t1, 'elapsed_s': time.time() - t_start}, kl)
        append_jsonl(log_path, row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if step % args.sample_log_every == 0:
            for i in range(0, len(texts), G):
                append_jsonl(samp_path, {'step': step + 1, 'input': expanded[i]['input'], 'format': ttypes[i],
                                         'answer': answers[i], 'completions': texts[i:i + G],
                                         'rewards': rew[i:i + G].tolist()})
        step += 1
        oom_retries = 0
        if step % args.save_every == 0:
            save_ckpt()

    save_trainable(model, os.path.join(args.out_dir, 'trainable.pt'), {'args': vars(args)})
    model = model.merge_and_unload()
    save_full_fp16(model, os.path.join(args.out_dir, 'merged'), args.model_path)
    print('saved merged model to', os.path.join(args.out_dir, 'merged'), flush=True)


if __name__ == '__main__':
    main()
