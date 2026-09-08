"""Tests for how distributed_nmf hands spatial_smoothness to nmf_pytorch.

Two things can break in that hand off, and each has a test here.

1. Only nmf_pytorch implements the term. So use_gpu=False with the term on must
   fail immediately, not inside every block.

2. nmf_pytorch reads a voxel's neighbours by reshaping a column of S to
   (nz, ny, nx). distributed_nmf builds that column with
   comp.reshape(np.prod(ts[1:])) and passes spatial_shape=ts[1:]. If those two
   orders ever disagree, the smoothing would average the wrong voxels and no
   error would be raised. The round trip test pins the convention down.
"""

import numpy as np
import pytest

from segmentNMF.distributed_NMF_methods import distributed_nmf

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import lateral_neighbour_mean  # noqa: E402


def test_spatial_smoothness_requires_gpu():
    """The check must fire before any data is touched, so dummies are enough."""
    with pytest.raises(AssertionError, match="requires use_gpu=True"):
        distributed_nmf(
            time_series_zarr=None,
            segments=np.zeros((2, 2, 2), dtype=np.uint32),
            time_series_spacing=np.array([4.0, 1.0, 1.0]),
            segments_spacing=np.array([1.0, 1.0, 1.0]),
            spatial_smoothness=0.1,
            use_gpu=False,
        )


def test_zero_smoothness_does_not_demand_a_gpu():
    """0 means off, so it must not trip the check. It must fail later, on data."""
    with pytest.raises(Exception) as e:
        distributed_nmf(
            time_series_zarr=None,
            segments=np.zeros((2, 2, 2), dtype=np.uint32),
            time_series_spacing=np.array([4.0, 1.0, 1.0]),
            segments_spacing=np.array([1.0, 1.0, 1.0]),
            spatial_smoothness=0,
            use_gpu=False,
        )
    assert "requires use_gpu=True" not in str(e.value)


def test_flatten_order_round_trips():
    """The layout distributed_nmf writes must be the layout nmf_pytorch reads.

    distributed_nmf writes  S[:, i] = comp.reshape(np.prod(ts[1:]))
    nmf_pytorch     reads   S.T.reshape((n_components,) + spatial_shape)
    """
    ts_spatial = (2, 5, 6)
    n_vox = int(np.prod(ts_spatial))
    rng = np.random.default_rng(0)
    comps = [rng.random(ts_spatial).astype(np.float32) for _ in range(3)]

    S = np.empty((n_vox, len(comps)), dtype=np.float32)
    for i, comp in enumerate(comps):
        S[:, i] = comp.reshape(np.prod(ts_spatial))

    back = torch.tensor(S).T.reshape((len(comps),) + ts_spatial).numpy()
    for i, comp in enumerate(comps):
        np.testing.assert_array_equal(back[i], comp)


def test_a_lateral_gap_in_one_plane_stays_in_that_plane():
    """End to end on the real layout: the fill must not leak across z.

    A hole is put in plane 0 only. Plane 1 holds zeros. After the mean, the
    hole reads 1.0 from its own plane, and plane 1 is still all zero.
    """
    ts_spatial = (2, 5, 6)
    comp = np.zeros(ts_spatial, dtype=np.float32)
    comp[0] = 1.0
    comp[0, 2, 3] = 0.0                      # the hole

    S = comp.reshape(int(np.prod(ts_spatial)))[:, None]
    x = torch.tensor(S).T.reshape((1,) + ts_spatial)
    from segmentNMF.NMF_methods import lateral_neighbour_counts
    avg = lateral_neighbour_mean(
        x, lateral_neighbour_counts(ts_spatial, dtype=x.dtype))[0]

    assert float(avg[0, 2, 3]) == pytest.approx(1.0), "the hole was not filled"
    assert float(avg[1].abs().max()) == 0.0, "plane 1 picked up plane 0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
