"""The optional fixes in patra_model/modeling_qwen2.py, on a tiny random alignment module (CPU)."""
import types

import pytest
import torch

from patra_model.modeling_qwen2 import MultiModalAlignment

D, TS_START = 32, 5
ALL = ['noncausal_align_attn', 'trend_seasonal_pairing', 'per_series_decomp', 'residual_alignment']


def make(fixes, state=None):
    cfg = types.SimpleNamespace(ts={'hidden_size': D, 'alignment_num_heads': 4, 'alignment_dropout': 0.0},
                                ts_token_start_index=TS_START, patra_fixes=fixes)
    m = MultiModalAlignment(cfg).eval()
    if state is not None:
        m.load_state_dict(state)
    return m


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    ids = torch.tensor([[1, 2, 3, 4, TS_START, 9, 6, 7, TS_START, 9, 6, 8, 3, 2, 1, 4, 4, 2]])  # two series
    return ids, torch.randn(1, ids.shape[1], D), torch.randn(5, D), torch.tensor([2, 3])  # 2 + 3 patches


def run(m, inputs):
    with torch.no_grad():
        return m(*inputs)[0]


def test_released_text_queries_only_see_the_first_tokens(inputs):
    """Released code: the output does not depend on anything after the 10th prompt token."""
    ids, text, feats, pc = inputs
    m = make([])
    changed = text.clone()
    changed[:, 10:] += 5.0
    assert torch.equal(run(m, (ids, text, feats, pc)), run(m, (ids, changed, feats, pc)))
    fixed = make(['noncausal_align_attn'], m.state_dict())
    assert not torch.allclose(run(fixed, (ids, text, feats, pc)), run(fixed, (ids, changed, feats, pc)))


def test_per_series_decomposition_isolates_batch_items():
    """Two prompts with one series each in one batch: with the fix, changing the second prompt's series leaves the
    first prompt's TS tokens untouched; the released moving average runs across both and leaks."""
    torch.manual_seed(0)
    ids = torch.tensor([[1, 2, 3, TS_START, 9, 6, 7, 3], [4, 2, 1, TS_START, 9, 6, 8, 2]])
    text, feats, pc = torch.randn(2, ids.shape[1], D), torch.randn(5, D), torch.tensor([2, 3])
    other = feats.clone()
    other[2:] += 3.0  # patches of the second prompt's series
    fixed = make(ALL)
    leaky = make([f for f in ALL if f != 'per_series_decomp'], fixed.state_dict())
    a, b = run(fixed, (ids, text, feats, pc)), run(fixed, (ids, text, other, pc))
    assert torch.allclose(a[:2], b[:2], atol=1e-6)
    a, b = run(leaky, (ids, text, feats, pc)), run(leaky, (ids, text, other, pc))
    assert not torch.allclose(a[:2], b[:2], atol=1e-6)


def test_residual_adds_encoder_output(inputs):
    plain = make(ALL[:3])
    residual = make(ALL, plain.state_dict())
    assert torch.allclose(run(residual, inputs) - run(plain, inputs), inputs[2], atol=1e-6)


def test_every_fix_changes_the_output(inputs):
    base = make([])
    for k in range(1, len(ALL) + 1):
        assert not torch.allclose(run(base, inputs), run(make(ALL[:k], base.state_dict()), inputs))


def test_unknown_fix_is_rejected():
    with pytest.raises(AssertionError):
        make(['typo'])
