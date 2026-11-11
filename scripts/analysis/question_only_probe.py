"""How much of the multiple-choice / true-false answer can be predicted without reading the series?

Classifiers are trained on the MC/TF rows of PATRA-TRAIN (``grpo.jsonl``: raw series + gold label) and tested on the
MC/TF items of PATRA-EVAL:
  majority          most frequent training label (TF) / answer position (MC)
  text_only         TF-IDF + logistic regression on the question text -> the "shortcut" ceiling
  stats_only        gradient boosting on 13 hand-crafted statistics per series
  text_plus_stats   gradient boosting on SVD(text) + statistics -> a lower bound on what the series adds
For MC, every (stem, option) pair is scored and the best option is picked.

    python scripts/analysis/question_only_probe.py --train data/PATRA-TRAIN/grpo.jsonl --data_dir data/PATRA-EVAL \
        --out runs/analysis/question_only_probe.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from patra.scoring import TASKS  # noqa: E402
from patra.series import TS_PLACEHOLDER  # noqa: E402
from patra.utils import read_jsonl  # noqa: E402

OPTION_RE = re.compile(r'(?:^|(?<=[\s:.,;]))([A-Da-d])[\]\)]\s*')
N_STATS = 13
PADDED_LEN = 28


def split_options(text):
    """'stem A) x B) y ...' -> (stem, [x, y, ...]); ([], stem) if the options are not listed in the text."""
    ms = list(OPTION_RE.finditer(text))
    if len(ms) < 2 or [m.group(1).upper() for m in ms] != [chr(65 + i) for i in range(len(ms))]:
        return text, []
    ends = [m.start() for m in ms[1:]] + [len(text)]
    return text[:ms[0].start()], [text[m.end():e].strip(' .;,') for m, e in zip(ms, ends)]


def label(answer, fmt):
    a = str(answer).strip()
    return ('T' if a[:1].upper() == 'T' else 'F') if fmt == 'true/false' else a[:1].upper()


def series_stats(ts_list):
    """13 statistics for each of the first two series, concatenated and zero-padded to 28, plus the number of series:
    length, level, spread, linear trend and its R^2, largest jump, lag-1 autocorrelation of the residual,
    turning-point rate, max z-score, first-to-last change, half-to-half shift, position of the largest jump."""
    out = []
    for ts in ts_list[:2]:
        x = np.asarray(ts, dtype=float)
        n = len(x)
        if n < 2:
            out += [n] + [0.0] * N_STATS
            continue
        m, sd = x.mean(), x.std() + 1e-9
        t = np.arange(n)
        b, a = np.polyfit(t, x, 1)
        resid = x - (a + b * t)
        r2 = 1 - (resid ** 2).sum() / (((x - m) ** 2).sum() + 1e-12)
        d = np.diff(x)
        ac1 = np.corrcoef(resid[:-1], resid[1:])[0, 1] if n > 3 and resid.std() > 0 else 0.0
        half = n // 2
        out += [n, np.sign(m) * np.log1p(abs(m)), np.log(sd), sd / (abs(m) + 1e-9), b * n / sd, r2,
                np.abs(d).max() / sd, ac1, (np.diff(np.sign(d)) != 0).mean() if n > 2 else 0,
                (np.abs(x - m) / sd).max(), (x[-1] - x[0]) / sd, (x[half:].mean() - x[:half].mean()) / sd,
                np.argmax(np.abs(d)) / n]
    out += [0.0] * (PADDED_LEN - len(out))
    return np.nan_to_num(np.array(out[:PADDED_LEN] + [len(ts_list)], dtype=float))


def question_text(r):
    return r['input'].replace(TS_PLACEHOLDER, ' TS ')


def closed_rows(rows):
    return [r for r in rows if r['question_format'] in ('multiple_choice', 'true/false')]


def gbm():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)


class TrueFalseProbes:
    def __init__(self, train):
        text = [question_text(r) for r in train]
        y = np.array([label(r['answer'], 'true/false') for r in train])
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text)
        X = self.vec.transform(text)
        self.lr = LogisticRegression(C=2.0, max_iter=3000).fit(X, y)
        self.svd = TruncatedSVD(64, random_state=0).fit(X)
        F = np.stack([series_stats(r['timeseries']) for r in train])
        self.gb = gbm().fit(np.hstack([self.svd.transform(X), F]), y)
        self.gb_stats = gbm().fit(F, y)
        self.majority = 'T' if (y == 'T').mean() >= 0.5 else 'F'
        self.p_true = float((y == 'T').mean())

    def correct(self, rows):
        y = np.array([label(r['answer'], 'true/false') for r in rows])
        X = self.vec.transform([question_text(r) for r in rows])
        F = np.stack([series_stats(r['timeseries']) for r in rows])
        return {'majority': int((y == self.majority).sum()), 'text_only': int((self.lr.predict(X) == y).sum()),
                'stats_only': int((self.gb_stats.predict(F) == y).sum()),
                'text_plus_stats': int((self.gb.predict(np.hstack([self.svd.transform(X), F])) == y).sum())}


class MultipleChoiceProbes:
    @staticmethod
    def pairs(rows):
        text, pos, feats, group, y = [], [], [], [], []
        for gi, r in enumerate(rows):
            stem, opts = split_options(r['input'])
            f = series_stats(r['timeseries'])
            for k, o in enumerate(opts):
                text.append(stem.replace(TS_PLACEHOLDER, ' TS ') + ' || ' + o)
                pos.append(k)
                feats.append(f)
                group.append(gi)
                y.append(int(chr(65 + k) == label(r['answer'], 'multiple_choice')))
        return text, np.array(pos), np.array(feats), np.array(group), np.array(y)

    @staticmethod
    def position(pos):
        return np.eye(4)[np.minimum(pos, 3)]

    def __init__(self, train):
        text, pos, feats, _, y = self.pairs(train)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text)
        X = self.vec.transform(text)
        self.lr = LogisticRegression(C=2.0, max_iter=3000).fit(hstack([X, csr_matrix(self.position(pos))]), y)
        self.svd = TruncatedSVD(64, random_state=0).fit(X)
        self.gb = gbm().fit(np.hstack([self.svd.transform(X), self.position(pos), feats]), y)
        prior = np.bincount([ord(label(r['answer'], 'multiple_choice')) - 65 for r in train
                             if label(r['answer'], 'multiple_choice') in 'ABCD'], minlength=4)
        self.position_prior = prior / prior.sum()
        self.majority = chr(65 + int(np.argmax(prior)))

    def correct(self, rows):
        text, pos, feats, group, _ = self.pairs(rows)
        X = self.vec.transform(text)
        s_lr = self.lr.predict_proba(hstack([X, csr_matrix(self.position(pos))]))[:, 1]
        s_gb = self.gb.predict_proba(np.hstack([self.svd.transform(X), self.position(pos), feats]))[:, 1]
        pick = lambda s: {g: chr(65 + int(np.argmax(s[group == g]))) for g in np.unique(group)}
        p_lr, p_gb = pick(s_lr), pick(s_gb)
        gold = [label(r['answer'], 'multiple_choice') for r in rows]
        # items whose options are not in the text get 'A' from the text probes and the prior from the others
        return {'majority': sum(g == self.majority for g in gold),
                'text_only': sum(p_lr.get(i, 'A') == g for i, g in enumerate(gold)),
                'stats_only': sum(g == self.majority for g in gold),
                'text_plus_stats': sum(p_gb.get(i, 'A') == g for i, g in enumerate(gold)),
                'n_options_parsed': len(p_lr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train', required=True, help='grpo.jsonl')
    ap.add_argument('--data_dir', required=True, help='PATRA-EVAL directory')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    train = closed_rows(read_jsonl(args.train))
    tf = TrueFalseProbes([r for r in train if r['question_format'] == 'true/false'])
    mc = MultipleChoiceProbes([r for r in train if r['question_format'] == 'multiple_choice'])
    print(f'train: MC answer positions {np.round(mc.position_prior, 3)}, TF P(True) = {tf.p_true:.3f}')
    res = {}
    for t in TASKS:
        rows = closed_rows(read_jsonl(os.path.join(args.data_dir, f'{t}.jsonl')))
        r_tf = [r for r in rows if r['question_format'] == 'true/false']
        r_mc = [r for r in rows if r['question_format'] == 'multiple_choice']
        c_tf = tf.correct(r_tf) if r_tf else {}
        c_mc = mc.correct(r_mc) if r_mc else {}
        n = len(rows)
        res[t] = {'n': n, 'n_tf': len(r_tf), 'n_mc': len(r_mc), 'n_mc_options_parsed': c_mc.get('n_options_parsed', 0),
                  **{k: round(100 * (c_tf.get(k, 0) + c_mc.get(k, 0)) / n, 2)
                     for k in ('majority', 'text_only', 'stats_only', 'text_plus_stats')}}
        print(t, res[t])
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
