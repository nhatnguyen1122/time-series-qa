from rouge_score import rouge_scorer

from patra.prompts import eval_prompt
from patra.rewards import correctness_reward_func, count_xml, original_task_reward
from patra.scoring import lenient_hit, lenient_pred, score_response
from patra.stats import paired_bootstrap, question_category

ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def test_official_score_needs_exact_boxed_label():
    mc = {'question_format': 'multiple_choice', 'answer': 'B'}
    assert score_response(mc, '<think>..</think> \\boxed{B}', ROUGE) == ('acc', 1)
    assert score_response(mc, '\\boxed{B) upward trend}', ROUGE) == ('acc', 0)  # right answer, wrong format
    tf = {'question_format': 'true/false', 'answer': 'F'}
    assert score_response(tf, '\\boxed{false}', ROUGE) == ('acc', 1)


def test_lenient_extraction():
    assert lenient_pred('\\boxed{B) upward trend}', 'multiple_choice') == 'B'
    assert lenient_pred('<answer>\nc\n</answer>', 'multiple_choice') == 'C'
    assert lenient_pred('B) The series rises', 'multiple_choice') == 'B'
    assert lenient_pred('The statement is false.', 'true/false') == 'False'
    assert lenient_hit({'question_format': 'true/false', 'answer': 'T', 'response': '\\boxed{True}'}) == 1


def test_eval_prompt_system_depends_on_format():
    closed = eval_prompt({'question_format': 'true/false', 'input': 'Q'})
    open_ = eval_prompt({'question_format': 'open_ended_question', 'input': 'Q'})
    assert '\\boxed{True}' in closed and 'helpful assistant' in open_
    assert closed.endswith('<|im_start|>user\nQ<|im_end|><|im_start|>assistant\n')


def test_rewards():
    good = '<think>\nr\n</think>\n<answer>\nb\n</answer>\n<task_type>x</task_type>'
    assert abs(count_xml(good) - 0.75) < 0.05
    comp = [[{'content': good}], [{'content': '<answer>e</answer>'}]]
    assert correctness_reward_func([0, 1], comp, ['B', 'B'], ['multiple_choice'] * 2) == [2.0, 0.0]
    assert original_task_reward([0, 1], comp, ['B', 'B'], ['multiple_choice'] * 2) == [1.0, 0.0]


def test_paired_bootstrap_and_categories():
    rows = lambda hits: {('t', i): {'metric': 'acc', 'question_format': 'true/false', 'answer': 'T',
                                    'response': '\\boxed{True}' if h else '\\boxed{False}', 'score': h}
                         for i, h in enumerate(hits)}
    r = paired_bootstrap(rows([0] * 50 + [1] * 50), rows([1] * 100), 'acc_len', n_boot=500)
    assert abs(r['diff'] - 50) < 1e-9 and r['ci95'][0] > 0 and r['p_two_sided'] == 0
    assert question_category('Is there an upward trend?') == 'trend'
    assert question_category('What is the mean value?') == 'magnitude/statistic'
