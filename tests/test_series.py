import numpy as np

from patra_model.processing_qwen2_ts import sp_encoding
from patra.series import (add_offset_scale, constant_series, hide_series, offset_scale_stats, serialize_text,
                                shuffled_series, sp_encode, ts_tensor)


def test_sp_encode_matches_released_processor():
    rng = np.random.RandomState(0)
    for ts in [rng.randn(24) * 50 + 10, rng.randn(30) * 0.1, np.arange(16.0)]:
        ours = sp_encode(ts).reshape(-1, 2)
        ref, _, _ = sp_encoding(np.asarray(ts, dtype=np.float64).copy())
        np.testing.assert_allclose(ours[:, 0], np.asarray(ref).reshape(len(ts), -1)[:, 0])
        assert (ours[:, 1] == 1).all()  # mask channel
        assert np.abs(ours[:, 0]).max() <= 3 + 1e-9


def test_offset_scale_prefix():
    ts = [10.0, 20.0, 30.0]  # mean 20, max |x - mean| = 10 -> scale 10 / 3
    offset, scale = offset_scale_stats(ts)
    assert offset == -20.0 and abs(scale - 10 / 3) < 1e-12
    out = add_offset_scale('a <ts><ts/> b <ts><ts/>', [ts, [1.0, 2.0]])
    assert out == 'a [Value Offset: -20.0000|Value Scaling: 3.3333]<ts><ts/> b ' \
                  '[Value Offset: -1.5000|Value Scaling: 1.0000]<ts><ts/>'


def test_serialize_and_hide():
    assert serialize_text('x=<ts><ts/>?', [[1.0, 2.5, -3.0]]) == 'x=[1, 2.5, -3]?'
    assert hide_series('x=<ts><ts/>?') == 'x=[series not shown]?'


def test_ts_tensor_layout():
    t = ts_tensor([[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
    assert t.shape == (2, 8, 1)
    assert (t[0, 4:] == 0).all()  # zero padding after the shorter series


def test_ablations():
    samples = [{'timeseries': [[i, i + 1.0]]} for i in range(10)] + [{'timeseries': [[0.0], [1.0]]}]
    a, b = shuffled_series(samples), shuffled_series(samples)
    assert a == b  # deterministic: every script gets the same replacement series
    assert all(len(x) == len(s['timeseries']) for x, s in zip(a, samples))  # same number of series
    assert constant_series([[1.0, 2.0, 6.0]]) == [[3.0, 3.0, 3.0]]
