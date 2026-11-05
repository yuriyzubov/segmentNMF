"""Tests for add_residual, which adds the residual of the fit back to each trace.

WHAT THE CHOICES ARE
    'cnmf': for each component, the least-squares trace for its final footprint,
    with the fitted light of every other component taken away. 'split': the same,
    but the residual of a shared voxel is split between its components. The fit
    does not change.

THE TOY
    One cell with a slow dip below its usual level and a few calcium events. The
    calcium model cannot go below its baseline, so its fitted trace loses part of
    the dip.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import nmf_pytorch  # noqa: E402


def toy(n_t=600, seed=0):
    yy, xx = np.mgrid[0:12, 0:12]
    S = np.exp(-0.5 * ((yy - 6.0) ** 2 + (xx - 6.0) ** 2) / 3.0 ** 2).reshape(-1, 1).astype(np.float32)
    t = np.arange(n_t)
    true = 40.0 - 6.0 * np.exp(-0.5 * ((t - 300) / 40.0) ** 2)     # slow dip
    s = np.zeros(n_t)
    s[[80, 200, 450]] = 15.0
    true = true + np.convolve(s, np.exp(-np.arange(30) / 4.0))[:n_t]
    rng = np.random.default_rng(seed)
    V = (S @ true[None, :] + rng.normal(0.0, 2.0, (S.shape[0], n_t))).astype(np.float32)
    return V, S, np.full((1, n_t), 40.0, np.float32), true


def fit(V, S, H, residual):
    return nmf_pytorch(V, S, H, np.ones_like(S), num_iterations=300, objective_rtol=None,
                       footprint_rtol=None, update_int=10000, H_lr=0.5, S_lr=0.2,
                       calcium_rule=dict(decay=4.0, penalty='noise'),
                       add_residual=residual)


def test_cnmf_is_the_least_squares_trace_of_the_footprint():
    """The fit does not change, and a lone component's 'cnmf' trace is S . V / |S|^2.

    If the option changed the fit, a comparison of the two traces would compare
    two different footprints.
    """
    V, S, H, _ = toy()
    S0, _, _ = fit(V, S, H, 'none')
    S1, H1, _ = fit(V, S, H, 'cnmf')
    np.testing.assert_allclose(S1, S0, atol=1e-5)
    s = S1[:, 0]
    np.testing.assert_allclose(H1[0], s @ V / (s @ s), rtol=1e-4, atol=1e-3)


def test_the_residual_keeps_a_fall_below_the_baseline():
    """With the residual, the trace keeps the depth of a slow dip that the calcium fit cuts.

    This is the detail the option exists for. If the trace with the residual also
    cut the dip, the option would add nothing.
    """
    V, S, H, true = toy()
    dip = slice(260, 340)
    out = {}
    for residual in ('none', 'cnmf'):
        So, Ho, _ = fit(V, S, H, residual)
        out[residual] = Ho[0] * So[:, 0].max()
    depth = true[:250].mean() - true[dip].min()

    def seen(x):
        return np.median(x[:250]) - np.median(x[290:310])

    assert seen(out['cnmf']) == pytest.approx(depth, rel=0.15)
    assert seen(out['none']) < 0.75 * depth


def test_split_equals_cnmf_for_a_lone_component():
    """With no other component, a voxel's residual has one owner, so 'split' equals 'cnmf'.

    Otherwise 'split' would change the traces of cells that share no voxel.
    """
    V, S, H, _ = toy()
    _, H1, _ = fit(V, S, H, 'cnmf')
    _, H2, _ = fit(V, S, H, 'split')
    np.testing.assert_allclose(H2, H1, rtol=1e-4, atol=1e-3)


def test_split_counts_unexplained_light_once():
    """Light that no cell explains, in the overlap of two cells, goes half to each.

    'cnmf' gives it in full to both, so both traces copy it and
    correlate. If 'split' did the same, it would not reduce the leakage.
    """
    yy, xx = np.mgrid[0:12, 0:20]

    def blob(cx):
        return np.exp(-0.5 * ((yy - 6.0) ** 2 + (xx - cx) ** 2) / 2.5 ** 2).ravel()

    S = np.stack([blob(7.5), blob(12.5)], 1).astype(np.float32)
    n_t = 400
    rng = np.random.default_rng(2)
    flicker = rng.normal(0.0, 3.0, n_t)                       # no cell owns it
    overlap = np.exp(-0.5 * ((yy - 6.0) ** 2 + (xx - 10.0) ** 2) / 1.0 ** 2).ravel()
    V = (S @ np.full((2, n_t), 40.0) + np.outer(overlap, flicker)
         + rng.normal(0.0, 0.5, (S.shape[0], n_t))).astype(np.float32)
    H = np.full((2, n_t), 40.0, np.float32)
    got = {}
    for residual in ('cnmf', 'split'):
        So, Ho, _ = fit(V, S, H, residual)
        got[residual] = [np.polyfit(flicker, Ho[k] * So[:, k].max(), 1)[0] for k in range(2)]
    for k in range(2):
        assert got['cnmf'][k] > 0.0
        assert got['split'][k] == pytest.approx(0.5 * got['cnmf'][k], rel=0.2)


def test_bad_add_residual_fails_loudly():
    """An unknown value must raise, not fall back to one of the choices.

    The old option took True and False. Those must raise too, so an old config
    does not run with a choice it did not ask for.
    """
    V, S, H, _ = toy()
    for bad in ('shared', True, False):
        with pytest.raises(ValueError, match='add_residual must be one of'):
            nmf_pytorch(V, S, H, np.ones_like(S), num_iterations=1, add_residual=bad)
