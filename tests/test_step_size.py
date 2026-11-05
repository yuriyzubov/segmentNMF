"""Tests for the two step-size denominators in nmf_pytorch.

WHAT THE OPTION DOES
    nmf_pytorch holds one factor still and steps the other. step_size_norm
    chooses the denominator of that step, one value per component.

    'sum_of_squares' divides by sum(S ** 2). With S held still the error is a
    quadratic in H, so this step lands on the minimum for a lone component.

    'sum' divides by sum(S). The step is then the mean residual over the
    footprint, weighted by S. It is larger than the least squares step by about
    the component's amplitude, and the gap is widest for a peaked factor. A
    large step drives S to its clamp at 1, where the footprint follows the mask
    B instead of the intensity. So each choice needs its own H_lr and S_lr.

WHAT EACH TEST DOES
    The first three are arithmetic on a one component problem, where the answer
    is known in closed form. They need no fit and cannot be flaky. The rest fit
    a small toy at a rate near 1, which suits the default.

WHAT IS NOT TESTED HERE
    The numpy `nmf` always divides by the plain sum and takes no keyword. It is
    left alone on purpose.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import nmf_pytorch  # noqa: E402


# ----------------------------------------------------------------- arithmetic
# One voxel column s, one component, one frame. The objective is
#     f(h) = || v - s h || ** 2
# which is a quadratic in h with its minimum at h* = (s . v) / (s . s).
# A gradient step from h = 0 with step size k lands at k * (s . v).
# So the step size that lands ON the minimum is exactly 1 / (s . s).

def _peaked_column(n=16, peak=4.0, tail=0.05):
    """A column whose sum and sum of squares are far apart."""
    s = np.full(n, tail, dtype=np.float64)
    s[0] = peak
    return s


def test_the_two_denominators_differ_for_a_peaked_factor():
    """The premise. If they were close, the choice would not matter."""
    s = _peaked_column()
    assert s.sum() == pytest.approx(4.75)
    assert (s * s).sum() == pytest.approx(16.0375)
    # the plain sum is more than three times too small here, so a step that
    # divides by it is more than three times too large
    assert (s * s).sum() / s.sum() > 3.0


def test_the_sum_of_squares_lands_on_the_minimum():
    """'sum_of_squares', in closed form. One step with lr = 1 must be exact."""
    s = _peaked_column()
    v = 7.0 * s                      # so the true h is exactly 7
    h_star = float(s @ v) / float(s @ s)
    assert h_star == pytest.approx(7.0)

    gradient = float(s @ v)          # the gradient at h = 0
    h_after = gradient / float(s @ s)
    assert h_after == pytest.approx(h_star), "the step missed the minimum"


def test_the_plain_sum_steps_past_the_minimum():
    """'sum', in closed form. Its step goes past the least squares answer."""
    s = _peaked_column()
    v = 7.0 * s
    h_star = float(s @ v) / float(s @ s)

    h_after = float(s @ v) / float(s.sum())      # the 'sum' rule
    assert h_after > h_star, "'sum' did not step past the minimum"
    # and it overshoots by the ratio the first test measured
    assert h_after / h_star == pytest.approx(float(s @ s) / float(s.sum()))


# ---------------------------------------------------------------------- a toy
# Two cells side by side, with CORRELATED traces. Correlation is what lets one
# cell explain its neighbour's voxels, so it is the case that matters. One
# trace is spiky and one is calm, because the bug is driven by peakedness.

SHAPE = (2, 8, 8)
CELL0 = (0, slice(1, 4), slice(1, 4))
CELL1 = (0, slice(1, 4), slice(4, 7))


def two_cell_toy(n_frames=200, seed=0):
    n_vox = int(np.prod(SHAPE))
    rng = np.random.default_rng(seed)

    S_true = np.zeros((n_vox, 2), dtype=np.float32)
    for k, sl in enumerate((CELL0, CELL1)):
        b = np.zeros(SHAPE, dtype=np.float32)
        b[sl] = 1.0
        S_true[:, k] = b.reshape(-1)

    # a shared component makes the two traces correlated
    shared = np.zeros(n_frames, dtype=np.float32)
    shared[rng.choice(n_frames, 12, replace=False)] = 40.0
    spiky = shared.copy()
    spiky[rng.choice(n_frames, 8, replace=False)] += 60.0
    calm = shared + np.abs(rng.normal(20.0, 2.0, n_frames)).astype(np.float32)
    H_true = np.stack([spiky, calm]).astype(np.float32)

    V = (S_true @ H_true
         + rng.normal(0, 1.0, (n_vox, n_frames))).astype(np.float32)
    V = np.maximum(V, 0.0)

    args = dict(
        V=V,
        # the seed is CORRECT. The question is whether the fit keeps it there.
        S_init=np.where(S_true > 0, 0.779, 0.0).astype(np.float32),
        H_init=np.full((2, n_frames), 1.0, dtype=np.float32),
        # a permissive mask: both cells are allowed to grow anywhere. This is
        # what the real `weights` neighbourhood does near a contact.
        B=np.ones((n_vox, 2), dtype=np.float32),
        num_iterations=400,
        update_int=401,
        # near 1.0, the natural rate for 'sum_of_squares'. 'sum' takes a much
        # larger step, so it needs a much smaller rate than these.
        H_lr=1.0,
        S_lr=1.0,
        min_iterations=400,
    )
    return args, S_true


def test_a_spiky_cell_does_not_invade_its_neighbour():
    """At the default rate, a peaked cell must stay out of its neighbour."""
    args, S_true = two_cell_toy()
    S, H, _ = nmf_pytorch(**args)
    assert np.isfinite(S).all() and np.isfinite(H).all()

    for c, other in ((0, 1), (1, 0)):
        live = S[:, c] > 0
        n_live = int(live.sum())
        assert n_live > 0, f"cell {c} died"
        invaded = int((live & (S_true[:, other] > 0)).sum())
        assert invaded / n_live < 0.2, (
            f"cell {c} put {invaded} of its {n_live} voxels inside cell "
            f"{other}")


def test_the_clamp_does_not_decide_the_footprint_shape():
    """A footprint pinned at 1 away from its OWN cell has no shape from data.

    Weight at the clamp inside a cell's own support is correct: S is capped at
    1 and the amplitude lives in H. So each column is checked against its own
    true support only, not against the union of both.

    Counted per column on purpose. A count over the union of both supports
    passes either way, because this toy has no signal free region large enough
    to pin.
    """
    args, S_true = two_cell_toy()
    S, _, _ = nmf_pytorch(**args)
    for c in range(S.shape[1]):
        away = S_true[:, c] == 0
        pinned = int(((S[:, c] >= 0.99) & away).sum())
        assert pinned == 0, (
            f"cell {c} holds {pinned} weights at the clamp outside its own "
            f"support, so the clamp and not the data set its shape")


# -------------------------------------------------------------- the keyword

def test_the_default_is_the_sum_of_squares():
    """Naming the default must change nothing, bit for bit."""
    a, _, _ = nmf_pytorch(**two_cell_toy()[0])
    b, _, _ = nmf_pytorch(step_size_norm='sum_of_squares',
                          **two_cell_toy()[0])
    np.testing.assert_array_equal(a, b)


def test_the_keyword_reaches_the_step():
    """'sum' must give a different answer, or the keyword is not connected."""
    a, _, _ = nmf_pytorch(**two_cell_toy()[0])
    b, _, _ = nmf_pytorch(step_size_norm='sum', **two_cell_toy()[0])
    assert np.abs(a - b).max() > 1e-4


def test_an_unknown_norm_fails_loudly():
    """A typo must stop the run, not pick a denominator in silence."""
    with pytest.raises(ValueError, match="step_size_norm must be one of"):
        nmf_pytorch(step_size_norm='squares', **two_cell_toy()[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
