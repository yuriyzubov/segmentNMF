"""Tests for the noise component of segmentNMF.NMF_methods.nmf.

estimate_noise_component adds one extra column to S, H and B. It holds a flat
weight over every voxel, so the fit can put light that belongs to no segment
somewhere other than into a cell.

The extra column goes onto B as well, but the gradient keeps one column per
FITTED component. Multiplying the two therefore needs B sliced to the fitted
components. Without the slice the shapes do not broadcast and the first
iteration raises, so the documented default of nmf could not run.
"""

import numpy as np
import pytest

from segmentNMF.NMF_methods import nmf

N_VOX, N_TIME, K = 24, 30, 2
RUN_KW = dict(num_iterations=5, update_int=10 ** 6, H_lr=1e-1, S_lr=1e-1,
              min_iterations=5, objective_threshold=None)


def toy(seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros((N_VOX, K), dtype=np.float32)
    S[0:6, 0] = 1.0
    S[6:12, 1] = 1.0
    H = np.abs(rng.normal(10.0, 2.0, (K, N_TIME))).astype(np.float32)
    return dict(
        V=(S @ H).astype(np.float32),
        S_init=S.copy(),
        H_init=H.copy(),
        B=np.ones((N_VOX, K), dtype=np.float32),
    )


def test_the_default_runs():
    """estimate_noise_component is True by default, so this is the normal path."""
    S, H, objectives = nmf(**toy(), **RUN_KW)
    assert np.isfinite(S).all() and np.isfinite(H).all()
    assert len(objectives) == RUN_KW["num_iterations"]


def test_the_noise_component_adds_one_column_to_S_and_H():
    """The caller must be able to find the fitted components by position."""
    S, H, _ = nmf(estimate_noise_component=True, **toy(), **RUN_KW)
    assert S.shape == (N_VOX, K + 1)
    assert H.shape == (K + 1, N_TIME)


def test_the_noise_column_covers_every_voxel():
    """It is a flat weight over the block, so it must not be sparse."""
    S, _, _ = nmf(estimate_noise_component=True, **toy(), **RUN_KW)
    noise = S[:, K]
    assert (noise > 0).all()
    assert noise.min() == pytest.approx(noise.max()), "not flat"


def test_turning_it_off_leaves_the_fitted_components_alone():
    """The slice must be a no-op when B already has one column per component."""
    with_noise, _, _ = nmf(estimate_noise_component=True, **toy(), **RUN_KW)
    without, _, _ = nmf(estimate_noise_component=False, **toy(), **RUN_KW)
    assert without.shape == (N_VOX, K)
    np.testing.assert_allclose(with_noise[:, :K], without, rtol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
