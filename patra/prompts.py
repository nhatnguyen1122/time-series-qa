"""Chat templates.

Two templates are needed because the released pipeline uses two:
  * training (LLaMA-Factory ``qwen`` template): default system prompt, newline after every ``<|im_end|>``;
  * evaluation (the authors' ``eval/patra_eval.py``, copied verbatim): a ``\\boxed{}`` instruction for multiple-choice
    and true/false questions, and no newline between turns.
"""
import re

DEFAULT_SYSTEM = 'You are a helpful assistant.'

EVAL_SYSTEM = '''
You are an assistant for evaluating time series questions.

Task: Answer the question concisely in the required format.
- Your thinking steps should be in <think></think> tags.
- Always output **exactly one answer**, using the following formats:

  1. For True/False questions:
     \\boxed{True}  or  \\boxed{False}

  2. For Multiple Choice questions (options A, B, C, D):
     \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}

'''

BOXED_RE = re.compile(r'\\boxed\{([^\{\}]+)\}')
CLOSED_FORMATS = ('multiple_choice', 'true/false')


def chat_prompt(user, system=DEFAULT_SYSTEM):
    """Training-time prompt (LLaMA-Factory ``qwen`` template), ending with the assistant header."""
    return f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'


def eval_prompt(sample):
    """Evaluation prompt, identical to ``build_prompt`` in the authors' ``eval/patra_eval.py``."""
    system = EVAL_SYSTEM if sample['question_format'] in CLOSED_FORMATS else DEFAULT_SYSTEM
    return (f"<|im_start|>system\n{system}<|im_end|>"
            f"<|im_start|>user\n{sample['input']}<|im_end|>"
            f"<|im_start|>assistant\n")
