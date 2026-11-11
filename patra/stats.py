"""Statistics used by the analysis scripts: paired bootstrap, series-sensitivity and question categories."""
import re

import numpy as np

from .scoring import lenient_hit, lenient_pred

# keyword rules, first match wins (a coarse but transparent grouping of the eval questions)
CATEGORIES = [
    ('magnitude/statistic', r'\b(mean|median|average|maximum|minimum|max|min|highest|lowest|sum|standard deviation|'
                            r'variance|range|value at|how many|count|percent|percentage|exceed|above|below)\b'),
    ('forecast/next value', r'\b(next|predict|forecast|future|upcoming|expect)\w*'),
    ('anomaly/outlier/break', r'\b(anomal|outlier|spike|sudden|abnormal|structural break|break|shift|drop)\w*'),
    ('seasonality/cycle', r'\b(season|cycl|period|repeat|recurr|oscillat)\w*'),
    ('trend', r'\b(trend|increas|decreas|upward|downward|rise|rising|declin|growth)\w*'),
    ('volatility', r'\b(volatil|fluctuat|stabl|stationar|noise|smooth)\w*'),
]


def question_category(question):
    q = question.lower()
    for name, pattern in CATEGORIES:
        if re.search(pattern, q):
            return name
    return 'other'


def item_scores(rows, keys, metric):
    """Per-item scores for ``metric`` in {'acc_len', 'acc', 'rougeL'}."""
    if metric == 'acc_len':
        return np.array([lenient_hit(rows[k]) for k in keys], dtype=float)
    return np.array([rows[k]['score'] for k in keys], dtype=float)


def paired_keys(a, b, metric, keep=None):
    """Items present in both runs that the metric applies to (optionally filtered by ``keep(key, row)``)."""
    kind = 'rougeL' if metric == 'rougeL' else 'acc'
    return sorted(k for k in set(a) & set(b) if a[k]['metric'] == kind and (keep is None or keep(k, a[k])))


def paired_bootstrap(a, b, metric='acc_len', keep=None, n_boot=5000, seed=0):
    """Mean difference ``b - a`` (in points) over shared items with a 95% bootstrap CI and two-sided p-value."""
    keys = paired_keys(a, b, metric, keep)
    d = item_scores(b, keys, metric) - item_scores(a, keys, metric)
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {'diff': 100 * d.mean(), 'ci95': [100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)],
            'p_two_sided': p, 'n': int(len(d))}


def series_sensitivity(base, ablated):
    """How much the answers move when the series input is changed (base run vs ablated run, same items)."""
    keys = sorted(set(base) & set(ablated))
    closed = [k for k in keys if base[k]['metric'] == 'acc']
    same = lambda k: lenient_pred(base[k]['response'], base[k]['question_format']) == \
        lenient_pred(ablated[k]['response'], ablated[k]['question_format'])
    return {'identical_response_pct': 100 * np.mean([base[k]['response'] == ablated[k]['response'] for k in keys]),
            'same_choice_pct': 100 * np.mean([same(k) for k in closed]),
            'd_acc_len': 100 * (item_scores(ablated, closed, 'acc_len').mean()
                                - item_scores(base, closed, 'acc_len').mean())}
