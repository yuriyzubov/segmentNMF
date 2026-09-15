"""Tests for NORM_FLOOR, the floor under both step-size denominators.

THE BUG THESE TESTS LOCK DOWN
    Both step sizes divide a learning rate by a column sum. The old guard was
    `norm[norm == 0] = 1`, an EXACT zero test.

    A cell with no real signal drains its trace toward zero. On the way down,
    sum(H) passed through the subnormal range: smaller than 1.18e-38, but not
    equal to 0. The guard missed it. Torch flushed the subnormal to zero inside
    the division and returned infinity, although the true quotient fits in
    float32 with room to spare. NumPy returns the correct number.

    0 * inf is NaN. The next step is computed from V - S @ H, and that product
    adds up all components. So one NaN column in S turned every component in
    the block to NaN, not only the one that drained.

    NORM_FLOOR clamps the denominator, so the infinity never appears.

WHAT IS NOT TESTED HERE
    The numpy `nmf` uses the same pattern but numpy divides correctly, so it
    cannot produce this infinity. It is left alone on purpose.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import NORM_FLOOR, nmf_pytorch  # noqa: E402


# A subnormal float32: below tiny (1.18e-38), above zero. This is the range a
# draining trace passes through, and the range `norm == 0` does not catch.
SUBNORMAL_SUM = 2.9116e-39
S_LR = 0.01


def drained_toy(n_iter=1400):
    """The toy that reproduced the failure, and the arguments to fit it.

    Three components on a 2 x 6 x 6 canvas. Two hold real signal. The third is
    a single voxel whose data is exactly zero, so nothing holds its trace up
    and it drains. That is the case that reached the subnormal range.
    """
    shape = (2, 6, 6)
    n_vox, n_frames = int(np.prod(shape)), 120
    rng = np.random.default_rng(0)

    S_true = np.zeros((n_vox, 3), dtype=np.float32)
    for k, (z, ys, xs) in enumerate([(0, slice(1, 4), slice(1, 4)),
                                     (0, slice(2, 5), slice(3, 6))]):
        b = np.zeros(shape, dtype=np.float32)
        b[z, ys, xs] = 1.0
        S_true[:, k] = b.reshape(-1)
    lonely = np.zeros(shape, dtype=np.float32)
    lonely[1, 4, 1] = 0.779
    S_true[:, 2] = lonely.reshape(-1)

    H_true = np.abs(rng.normal(20.0, 5.0, (3, n_frames))).astype(np.float32)
    H_true[2] = 0.0
    V = (S_true @ H_true
         + rng.normal(0, 1.0, (n_vox, n_frames))).astype(np.float32)
    V = np.maximum(V, 0.0)
    V[int(np.flatnonzero(S_true[:, 2] > 0)[0]), :] = 0.0

    return dict(
        V=V,
        S_init=np.where(S_true > 0, 0.779, 0.0).astype(np.float32),
        H_init=np.full((3, n_frames), 1.0, dtype=np.float32),
        B=np.ones((n_vox, 3), dtype=np.float32),
        num_iterations=n_iter,
        update_int=n_iter + 1,
        H_lr=0.1,
        S_lr=S_LR,
        min_iterations=n_iter,
        spatial_threshold=0.45,
        objective_rtol=None,
        footprint_rtol=None,
    )


def test_a_subnormal_sum_is_not_exactly_zero():
    """The premise of the bug: the old guard could not see this value."""
    x = torch.tensor([SUBNORMAL_SUM], dtype=torch.float32)
    assert not bool((x == 0).item()), "the old guard would have caught it"
    assert x.item() < torch.finfo(torch.float32).tiny, "not subnormal"


def test_torch_divides_a_subnormal_to_infinity():
    """The bug itself. numpy is correct here and torch is not."""
    x = torch.tensor([SUBNORMAL_SUM], dtype=torch.float32)
    assert not bool(torch.isfinite(S_LR / x).item()), \
        "torch no longer overflows; this test has outlived the bug"
    # the true quotient is about 3.4e36, well inside float32
    assert np.isfinite(np.float32(S_LR) / np.float32(SUBNORMAL_SUM))


def test_the_floor_stops_the_overflow():
    """The fix, on the exact value that failed."""
    x = torch.tensor([SUBNORMAL_SUM], dtype=torch.float32)
    q = S_LR / torch.clamp(x, min=NORM_FLOOR)
    assert bool(torch.isfinite(q).all()), "the floor did not stop it"
    assert q.item() == pytest.approx(S_LR / NORM_FLOOR)


def test_infinity_times_zero_is_the_nan():
    """Why an infinite step size does not merely give a huge step."""
    inf = S_LR / torch.tensor([SUBNORMAL_SUM], dtype=torch.float32)
    assert bool(torch.isnan(torch.zeros(1) * inf).item())


def test_the_floor_is_below_any_live_component():
    """The floor must never bind on a component that still holds weight.

    Checked against the default, a sum of SQUARES. The smallest live footprint
    is one voxel at spatial_threshold with the peak at 1, so at a threshold of
    0.45 its sum of squares is 0.2025. Any real trace reaches far more.
    """
    assert NORM_FLOOR < 0.45 / 1000.0, "the floor could bind on a real cell"
    assert NORM_FLOOR > torch.finfo(torch.float32).tiny * 1e6, \
        "the floor is too close to the subnormal range"


def test_a_drained_component_no_longer_makes_nan():
    """The regression test. This toy went NaN before the floor was added.

    Smoothing is OFF here on purpose. The bug predates spatial_smoothness, so
    the plain path must be the one that is proven fixed.
    """
    p = drained_toy()
    S, H, objectives = nmf_pytorch(**p)
    assert np.isfinite(S).all(), "S holds a value that is not finite"
    assert np.isfinite(H).all(), "H holds a value that is not finite"
    obj = np.array([float(o) for o in objectives])
    assert np.isfinite(obj).all(), "the objective went non finite"


def test_it_still_fits_the_two_real_components():
    """The floor must not break the fit. The two real cells must survive."""
    p = drained_toy()
    S, H, _ = nmf_pytorch(**p)
    assert (S[:, 0] > 0).sum() >= 4, "the first real cell collapsed"
    assert (S[:, 1] > 0).sum() >= 4, "the second real cell collapsed"
    assert H[0].mean() > 1.0 and H[1].mean() > 1.0, "the real traces died"


def test_an_exactly_zero_trace_is_still_handled():
    """The old guard's own case must keep working.

    A component held at exactly zero weight everywhere gives sum(S) == 0. The
    floor replaces the old `= 1` substitution, so this must stay finite.
    """
    p = drained_toy(n_iter=50)
    p["S_init"] = p["S_init"].copy()
    p["S_init"][:, 2] = 0.0
    p["H_init"] = p["H_init"].copy()
    p["H_init"][2] = 0.0
    S, H, _ = nmf_pytorch(**p)
    assert np.isfinite(S).all() and np.isfinite(H).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
