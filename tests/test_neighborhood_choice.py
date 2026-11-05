"""Tests for the neighborhood choice in distributed_nmf.

WHY THE PARAMETER EXISTS
    The 'weights' neighborhood never reaches zero, so a footprint can grow up
    to the edge of its block, and the result depends on the block size. The
    'distance' neighborhood is exactly zero past neighborhood_radius.

WHAT THESE TESTS CHECK
    The B matrix that each block hands to the fit: exact zeros far from the
    cell for 'distance', none for 'weights'. And that bad input fails before
    any block runs.
"""

import numpy as np
import pytest

import segmentNMF.distributed_NMF_methods as dnm
from segmentNMF.distributed_NMF_methods import distributed_nmf


class SerialCluster:
    """Stands in for a ClusterWrap cluster. Runs each block in this process."""

    def __init__(self):
        self.client = self

    def map(self, func, *iterables):
        return list(map(func, *iterables))

    def gather(self, futures):
        return futures


def one_cell_volume():
    """One small cell in a wide canvas, so most voxels are far from it."""
    segments = np.zeros((1, 12, 12), dtype=np.uint32)
    segments[:, 5:7, 5:7] = 1
    rng = np.random.default_rng(0)
    trace = np.abs(rng.normal(10.0, 3.0, 20)).astype(np.float32)
    time_series = np.zeros((20,) + segments.shape, dtype=np.float32)
    time_series[:, segments == 1] = trace[:, None]
    time_series += np.abs(rng.normal(0.0, 0.5, time_series.shape))
    return segments, time_series.astype(np.float32)


def run(tmp_path, monkeypatch, **kw):
    """Run distributed_nmf and return the B of its one block."""
    seen = []
    real_nmf = dnm.nmf

    def recording_nmf(V, S, H, B, **kwargs):
        seen.append(np.array(B))
        return real_nmf(V, S, H, B, **kwargs)

    monkeypatch.setattr(dnm, "nmf", recording_nmf)
    segments, time_series = one_cell_volume()
    distributed_nmf(
        time_series_zarr=time_series,
        segments=segments,
        time_series_spacing=np.array([1.0, 1.0, 1.0]),
        segments_spacing=np.array([1.0, 1.0, 1.0]),
        block_radius=6.0,
        temporary_directory=str(tmp_path),
        num_iterations=5,
        min_iterations=5,
        cluster=SerialCluster(),
        **kw,
    )
    assert len(seen) == 1
    return seen[0]


def test_weights_never_reach_zero(tmp_path, monkeypatch):
    """The default keeps its old behaviour: no voxel is shut out."""
    B = run(tmp_path, monkeypatch)
    assert (B[:, 0] > 0).all()


def test_distance_is_exactly_zero_past_the_radius(tmp_path, monkeypatch):
    """Far voxels get exactly 0, so the footprint cannot grow there."""
    B = run(tmp_path, monkeypatch, neighborhood="distance",
            neighborhood_radius=2.0)
    assert (B[:, 0] == 0).any(), "no voxel is shut out"
    assert (B[:, 0] > 0).any(), "the cell itself is shut out"


def test_a_larger_radius_lets_more_voxels_in(tmp_path, monkeypatch):
    """The radius must reach the neighborhood, or the setting is not connected."""
    small = run(tmp_path, monkeypatch, neighborhood="distance",
                neighborhood_radius=1.0)
    large = run(tmp_path, monkeypatch, neighborhood="distance",
                neighborhood_radius=3.0)
    assert (large[:, 0] > 0).sum() > (small[:, 0] > 0).sum()


def test_an_unknown_neighborhood_fails_loudly():
    """A typo must stop the run before any block, not pick one in silence."""
    with pytest.raises(ValueError, match="must be 'weights' or 'distance'"):
        distributed_nmf(
            time_series_zarr=None,
            segments=np.zeros((1, 2, 2), dtype=np.uint32),
            time_series_spacing=np.array([1.0, 1.0, 1.0]),
            segments_spacing=np.array([1.0, 1.0, 1.0]),
            neighborhood="distanse",
        )


def test_distance_without_a_radius_fails_loudly():
    with pytest.raises(ValueError, match="requires neighborhood_radius"):
        distributed_nmf(
            time_series_zarr=None,
            segments=np.zeros((1, 2, 2), dtype=np.uint32),
            time_series_spacing=np.array([1.0, 1.0, 1.0]),
            segments_spacing=np.array([1.0, 1.0, 1.0]),
            neighborhood="distance",
        )
