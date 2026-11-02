"""Tests for plane_weight_profile in segmentNMF.distributed_NMF_methods.

Several segment planes fall inside one time series plane. This gives each of
them a weight, so that a segment is projected onto the time series grid by its
distance from the plane centre.

The count is the property that matters. The caller reshapes the result to
(projection_factor, 1, 1), so a profile of the wrong length stops the run. An
earlier form built the offsets with arange(-(pf // 2), pf // 2 + 1), which
holds pf + 1 values whenever pf is even.
"""

import numpy as np
import pytest

from segmentNMF.distributed_NMF_methods import plane_weight_profile

SPACING = 1.0
SIGMA = 2.0


def profile(pf, spacing=SPACING, sigma=SIGMA):
    return plane_weight_profile(pf, spacing, sigma)


def test_an_odd_factor_has_a_centre_plane_at_weight_1():
    """An odd factor puts one plane exactly at the centre, at distance 0."""
    for pf in (1, 3, 5, 7):
        w = profile(pf)
        assert len(w) == pf
        assert w[pf // 2] == pytest.approx(1.0)


def test_an_even_factor_gives_exactly_that_many_weights():
    """The case the earlier form could not build. It gave pf + 1 values."""
    for pf in (2, 4, 6):
        assert len(profile(pf)) == pf


def test_an_even_factor_has_no_plane_at_the_centre():
    """With no centre plane every offset is a half step, so no weight is 1."""
    w = profile(4)
    assert w.max() < 1.0
    # the two inner planes sit half a step away, the two outer ones 1.5 steps
    assert w[1] == pytest.approx(np.exp(-0.5 * SPACING / SIGMA))
    assert w[0] == pytest.approx(np.exp(-1.5 * SPACING / SIGMA))


def test_the_profile_is_symmetric_and_falls_from_the_centre():
    """A plane must not care which side of the centre it sits on."""
    for pf in (2, 3, 4, 5):
        w = profile(pf)
        np.testing.assert_allclose(w, w[::-1])
        half = w[:pf // 2]
        assert all(a < b for a, b in zip(half, half[1:])), "does not rise"


def test_a_larger_sigma_keeps_more_weight_far_from_the_centre():
    """sigma is the falloff length, so it must widen the profile."""
    near, far = profile(5, sigma=1.0), profile(5, sigma=4.0)
    assert (far[0] > near[0]) and (far[-1] > near[-1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
