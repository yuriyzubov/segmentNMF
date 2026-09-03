"""Tests for spatial_threshold in segmentNMF.NMF_methods.

B damps how fast a footprint grows, but it never reaches zero, so a footprint
keeps creeping outward the longer the fit runs. spatial_threshold puts a floor
under each weight: a voxel survives only if it stays above a fraction of its
own component's peak.

The threshold is PER COMPONENT, and that is the property most easily lost. A
single threshold over the whole S matrix would erase a dim component outright,
because every one of its weights sits below the peak of a bright neighbour.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import nmf, nmf_pytorch  # noqa: E402

THRESHOLD = 0.3


def two_cell_toy(n_vox=24, n_time=30, seed=0, dim_factor=1.0):
    """Two components, the second `dim_factor` times as bright as the first.

    Each component owns 6 voxels. B is all ones, so nothing but the threshold
    limits where a footprint may go.
    """
    rng = np.random.default_rng(seed)
    S_true = np.zeros((n_vox, 2), dtype=np.float32)
    S_true[0:6, 0] = 1.0
    S_true[6:12, 1] = dim_factor
    H_true = np.abs(rng.normal(10.0, 2.0, (2, n_time))).astype(np.float32)
    V = (S_true @ H_true).astype(np.float32)
    return dict(
        V=V,
        S_init=(S_true + 0.05 * rng.random((n_vox, 2))).astype(np.float32),
        H_init=H_true.copy(),
        B=np.ones((n_vox, 2), dtype=np.float32),
        num_iterations=30,
        update_int=10 ** 6,
        min_iterations=30,
    )


def test_off_by_default_changes_nothing():
    """Not passing the argument must give exactly the old result."""
    a, _, _ = nmf_pytorch(**two_cell_toy())
    b, _, _ = nmf_pytorch(spatial_threshold=None, **two_cell_toy())
    np.testing.assert_array_equal(a, b)


def test_every_surviving_weight_is_above_its_own_peak_fraction():
    """The rule itself. Nothing may sit between 0 and threshold * peak."""
    S, _, _ = nmf_pytorch(spatial_threshold=THRESHOLD, **two_cell_toy())
    for c in range(S.shape[1]):
        column = S[:, c]
        peak = column.max()
        alive = column[column > 0]
        assert alive.size > 0, f"component {c} was erased"
        assert alive.min() >= THRESHOLD * peak - 1e-6, (
            f"component {c} kept a weight below {THRESHOLD} of its peak")


def test_a_dim_component_is_not_erased():
    """The threshold is per component, so a faint cell must keep its voxels.

    The second component is 20 times fainter here. A threshold taken over the
    whole S matrix would delete all of it.
    """
    S, _, _ = nmf_pytorch(spatial_threshold=THRESHOLD,
                          **two_cell_toy(dim_factor=0.05))
    assert (S[:, 1] > 0).sum() > 0, "the dim component lost every voxel"
    assert S[:, 1].max() < S[:, 0].max(), "the toy did not stay dim"


def test_a_higher_threshold_never_keeps_more_voxels():
    """The threshold must be monotone, or the value means nothing."""
    counts = []
    for t in (0.0, 0.2, 0.5, 0.8):
        S, _, _ = nmf_pytorch(spatial_threshold=t, **two_cell_toy())
        counts.append(int((S > 0).sum()))
    assert all(a >= b for a, b in zip(counts, counts[1:])), counts


# ------------------------------------------------------------- the numpy nmf

NMF_KW = dict(num_iterations=30, update_int=10 ** 6, H_lr=1e-1, S_lr=1e-1,
              min_iterations=30, objective_threshold=None)


def numpy_toy(**kw):
    p = two_cell_toy(**kw)
    for key in ("num_iterations", "update_int", "min_iterations"):
        p.pop(key)
    return p


def test_the_numpy_nmf_applies_the_same_rule():
    """Both implementations must mean the same thing by spatial_threshold."""
    S, _, _ = nmf(spatial_threshold=THRESHOLD, estimate_noise_component=False,
                  **numpy_toy(), **NMF_KW)
    for c in range(S.shape[1]):
        column = S[:, c]
        alive = column[column > 0]
        assert alive.size > 0, f"component {c} was erased"
        assert alive.min() >= THRESHOLD * column.max() - 1e-6


def test_the_numpy_nmf_is_unchanged_when_the_threshold_is_off():
    a, _, _ = nmf(spatial_threshold=None, estimate_noise_component=False,
                  **numpy_toy(), **NMF_KW)
    b, _, _ = nmf(estimate_noise_component=False, **numpy_toy(), **NMF_KW)
    np.testing.assert_array_equal(a, b)


# NOTE: the threshold exempts the noise column, by slicing S[:, :n_components].
# That is not tested here. The noise column holds 1 / n_voxels at every voxel,
# so it is uniform, and a per-component threshold never removes any of it. The
# exemption is only visible against a GLOBAL peak, which this code never uses.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
