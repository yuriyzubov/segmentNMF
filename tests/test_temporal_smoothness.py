"""Tests for temporal_smoothness in nmf_pytorch.

WHY IT EXISTS
    Where two footprints overlap, the data fix the sum of the two traces but
    only weakly their difference. Each frame's noise then goes into the
    difference: one trace up, the other down. A penalty on fast changes in each
    trace acts more strongly on that difference than on the shared signal.

WHAT IS TESTED
    smooth_in_time solves (I + a L) X = H, L the 1D Laplacian with mirror
    boundaries. In the fit: off changes nothing, on smooths the traces, and on
    lowers the noise trade between two overlapping cells.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import nmf_pytorch, smooth_in_time  # noqa: E402


def laplacian_mirror(x):
    """L x with mirror boundaries: x[-1] = x[0] and x[T] = x[T - 1]."""
    left = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    right = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    return 2 * x - left - right


def freq(n_t):
    k = torch.arange(n_t + 1, dtype=torch.float64)
    return (2 - 2 * torch.cos(np.pi * k / n_t))[None, :]


# ---------------------------------------------------------- the step itself

def test_zero_strength_changes_nothing():
    H = torch.rand(3, 50, dtype=torch.float64)
    out = smooth_in_time(H, torch.zeros(3, 1, dtype=torch.float64), freq(50))
    np.testing.assert_allclose(out.numpy(), H.numpy(), atol=1e-12)


def test_the_step_solves_the_penalised_problem():
    """X + a L X must give back H, row by row, each row with its own a."""
    rng = np.random.default_rng(0)
    H = torch.tensor(rng.random((3, 40)))
    a = torch.tensor([[0.5], [2.0], [10.0]], dtype=torch.float64)
    X = smooth_in_time(H, a, freq(40)).numpy()
    np.testing.assert_allclose(X + a.numpy() * laplacian_mirror(X), H.numpy(),
                               atol=1e-9)


def test_the_mean_is_kept():
    """The Laplacian has no constant part, so smoothing keeps each row's mean."""
    rng = np.random.default_rng(1)
    H = torch.tensor(rng.random((2, 64)))
    X = smooth_in_time(H, torch.full((2, 1), 5.0, dtype=torch.float64), freq(64))
    np.testing.assert_allclose(X.mean(dim=1).numpy(), H.mean(dim=1).numpy(),
                               atol=1e-9)


# ----------------------------------------------------------------- the fit

def overlap_toy(seed=0, n_frames=300):
    """Two cells that share half their voxels, with independent noise."""
    rng = np.random.default_rng(seed)
    n_vox = 12
    S_true = np.zeros((n_vox, 2), dtype=np.float32)
    S_true[0:8, 0] = 1.0
    S_true[4:12, 1] = 1.0
    H_true = np.full((2, n_frames), 10.0, dtype=np.float32)
    V = S_true @ H_true + rng.normal(0, 2.0, (n_vox, n_frames))
    return dict(V=np.maximum(V, 0).astype(np.float32),
                S_init=S_true.copy(), H_init=np.ones((2, n_frames), dtype=np.float32),
                B=(S_true > 0).astype(np.float32),
                num_iterations=300, update_int=10 ** 6, H_lr=0.5, S_lr=0.0,
                min_iterations=300, objective_rtol=None, footprint_rtol=None)


def test_none_and_zero_are_the_same_as_off():
    a, ha, _ = nmf_pytorch(**overlap_toy())
    b, hb, _ = nmf_pytorch(temporal_smoothness=0.0, **overlap_toy())
    np.testing.assert_array_equal(ha, hb)


def test_on_makes_each_trace_smoother():
    _, off, _ = nmf_pytorch(**overlap_toy())
    _, on, _ = nmf_pytorch(temporal_smoothness=2.0, **overlap_toy())
    rough = lambda h: np.mean(np.diff(h, axis=1) ** 2)  # noqa: E731
    assert rough(on) < 0.5 * rough(off)


def test_on_lowers_the_noise_trade_between_overlapping_cells():
    """Off, the two traces move opposite each other. On, much less so."""
    _, off, _ = nmf_pytorch(**overlap_toy())
    _, on, _ = nmf_pytorch(temporal_smoothness=2.0, **overlap_toy())
    corr = lambda h: np.corrcoef(h[0], h[1])[0, 1]  # noqa: E731
    assert corr(off) < -0.2, "the toy does not show the noise trade"
    assert corr(on) > corr(off) + 0.1


def test_a_negative_strength_fails_loudly():
    with pytest.raises(ValueError, match="temporal_smoothness must be >= 0"):
        nmf_pytorch(temporal_smoothness=-1.0, **overlap_toy())
