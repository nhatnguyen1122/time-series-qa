"""GRPO prompts and rewards.

The prompt suffixes, ``correctness_reward_func`` and ``xmlcount_reward_func`` are copied verbatim from the authors'
``train/src/llamafactory/train/grpo/workflow.py`` (the trainer itself is not released). The "task-aware balanced"
reward is their sum: a format reward in [0, 0.75] plus a task reward in [0, 2] (+0.5 for a valid label, +1.5 for the
correct label, or 2 x ROUGE-L for open-ended questions). ``original_task_reward`` is the paper's "Original Reward"
ablation (label correctness in {0, 1}, ROUGE-L in [0, 1]).
"""
import re

from rouge_score import rouge_scorer

# ---- verbatim from workflow.py -------------------------------------------------------------------------------------
MC_PROMPT = '''
You must first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think>
and </think> tags. Once you've finished your reasoning, you must choose ONE from A, B, C and D and then present it within
<answer> and </answer> tags. Finally, name the task and present it within <task_type> and </task_type> tags.
'''

TF_PROMPT = '''
You must first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think>
and </think> tags. Once you've finished your reasoning, you must choose ONE from True and False and then present it within
<answer> and </answer> tags. Finally, name the task and present it within <task_type> and </task_type> tags.
'''

OE_PROMPT = '''
You must first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think>
and </think> tags. Once you've finished your reasoning, you must give the final answer then present it within
<answer> and </answer> tags. Finally, name the task and present it within <task_type> and </task_type> tags.
'''
PROMPT_SUFFIX = {'multiple_choice': MC_PROMPT, 'true/false': TF_PROMPT, 'open_ended_question': OE_PROMPT}
_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def correctness_reward_func(prompts, completions, answer, task_type, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    res_answer = []
    for response in responses:
        matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
        res_answer.append(matches[-1].strip().lower() if matches else None)
    reward = []
    for i in range(len(prompts)):
        reward.append(0.0)
        if task_type[i] == 'true/false':
            reward[i] = reward[i] + 0.5 if res_answer[i] in ['true', 'false'] else reward[i]
        if task_type[i] == 'multiple_choice':
            reward[i] = reward[i] + 0.5 if res_answer[i] in ['a', 'b', 'c', 'd'] else reward[i]
        if task_type[i] == 'true/false' or task_type[i] == 'multiple_choice':
            flag = ('true' if answer[i] == 'T' else 'false') if task_type[i] == 'true/false' else answer[i].lower()
            reward[i] = reward[i] + 1.5 if res_answer[i] is not None and res_answer[i].lower() == flag else reward[i]
        else:
            if res_answer[i] is not None:
                reward[i] = reward[i] + 2 * _SCORER.score(answer[i], res_answer[i])['rougeL'].fmeasure
    return reward


def count_xml(text):
    count = 0.0
    if text.count("<task_type>") == 1:
        count += 0.125
    if text.count("</task_type>") == 1:
        count += 0.125
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>\n") == 1:
        count += 0.125
    if text.count("<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("</answer>\n")[-1]) * 0.001
    if text.count("</answer>\n") == 1:
        count += 0.125
        count -= (len(text.split("</answer>\n")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs):
    return [count_xml(c[0]["content"]) for c in completions]
# --------------------------------------------------------------------------------------------------------------------


def original_task_reward(prompts, completions, answer, task_type, **kwargs):
    """'Original Reward' ablation (paper, Table 5): label in {0, 1}, ROUGE-L in [0, 1], no credit for a valid label."""
    out = []
    for c, a, t in zip(completions, answer, task_type):
        m = re.findall(r'<answer>(.*?)</answer>', c[0]['content'], re.DOTALL)
        r = m[-1].strip().lower() if m else None
        if t in ('true/false', 'multiple_choice'):
            gold = ('true' if a == 'T' else 'false') if t == 'true/false' else a.lower()
            out.append(1.0 if r == gold else 0.0)
        else:
            out.append(_SCORER.score(a, r)['rougeL'].fmeasure if r is not None else 0.0)
    return out


TASK_REWARDS = {'balanced': correctness_reward_func, 'original': original_task_reward}
# task reward at or above which a closed-form answer counts as correct (for logging accuracy during training)
CORRECT_THRESHOLD = {'balanced': 1.5, 'original': 1.0}
