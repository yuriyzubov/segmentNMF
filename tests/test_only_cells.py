"""Tests for only_cells: run only the blocks that hold some cells.

WHY THE PARAMETER EXISTS
    A spot check of a few cells must not cost a full run. A full run fits
    every block of the volume.

THE RISK THIS GUARDS
    A requested cell must get the same trace as in a full run. That holds only
    if the block layout comes from every cell, and the filter comes after it.
    A layout built from the requested cells alone would give each of them
    other neighbours, and so another trace.
"""

import numpy as np
import pytest

from segmentNMF.distributed_NMF_methods import distributed_nmf


class SerialCluster:
    """Stands in for a ClusterWrap cluster. Runs each block in this process."""

    def __init__(self):
        self.client = self
        self.n_blocks = 0

    def map(self, func, *iterables):
        results = list(map(func, *iterables))
        self.n_blocks += len(results)
        return results

    def gather(self, futures):
        return futures


def two_pair_volume():
    """Four cells in two pairs, far apart, so each pair is one block.

    Cells 1 and 2 sit close together on the left, cells 3 and 4 on the right.
    """
    segments = np.zeros((2, 6, 40), dtype=np.uint32)
    segments[:, 2:4, 2:4] = 1
    segments[:, 2:4, 6:8] = 2
    segments[:, 2:4, 30:32] = 3
    segments[:, 2:4, 34:36] = 4

    rng = np.random.default_rng(0)
    n_frames = 30
    traces = np.abs(rng.normal(10.0, 3.0, (4, n_frames))).astype(np.float32)
    time_series = np.zeros((n_frames,) + segments.shape, dtype=np.float32)
    for cell in range(4):
        time_series[:, segments == cell + 1] = traces[cell][:, None]
    time_series += np.abs(rng.normal(0.0, 0.5, time_series.shape))
    return segments, time_series.astype(np.float32)


def run(tmp_path, only_cells=None):
    segments, time_series = two_pair_volume()
    cluster = SerialCluster()
    out = distributed_nmf(
        time_series_zarr=time_series,
        segments=segments,
        time_series_spacing=np.array([1.0, 1.0, 1.0]),
        segments_spacing=np.array([1.0, 1.0, 1.0]),
        max_complete_cells_per_block=2,
        max_neighbor_distance=10.0,
        only_cells=only_cells,
        temporary_directory=str(tmp_path),
        num_iterations=20,
        min_iterations=20,
        cluster=cluster,
    )
    return out, cluster.n_blocks


def test_none_runs_every_block(tmp_path):
    """None must run the full volume: both blocks, every row filled."""
    out, n_blocks = run(tmp_path)
    assert n_blocks == 2
    assert np.isfinite(out).all()


def test_only_the_block_of_a_requested_cell_runs(tmp_path):
    """Cell 1 is in the left block. The right block must not run."""
    _, n_blocks = run(tmp_path, only_cells=[1])
    assert n_blocks == 1


def test_a_requested_cell_gets_the_same_trace_as_a_full_run(tmp_path):
    """The block of cell 1 must be the same block as in a full run.

    Cell 2 shares that block, so it gets a row too, also equal to a full run.
    """
    full, _ = run(tmp_path)
    part, _ = run(tmp_path, only_cells=[1])
    np.testing.assert_array_equal(part[0], full[0])
    np.testing.assert_array_equal(part[1], full[1])


def test_cells_of_blocks_that_did_not_run_stay_nan(tmp_path):
    """Cells 3 and 4 are in the right block, so their rows must stay NaN.

    NaN, not zero, so a reader cannot take a missing trace for a silent cell.
    """
    part, _ = run(tmp_path, only_cells=[1])
    assert np.isnan(part[2]).all()
    assert np.isnan(part[3]).all()


def test_ids_from_both_blocks_run_both_blocks(tmp_path):
    """One id in each block must run both blocks, and each block once."""
    out, n_blocks = run(tmp_path, only_cells=[2, 3])
    assert n_blocks == 2
    assert np.isfinite(out).all()


def test_an_unknown_id_fails_loudly():
    """An id that is not a seed must raise before any block runs.

    A silent skip would return all NaN, and look like a run that found nothing.
    """
    segments, time_series = two_pair_volume()
    with pytest.raises(ValueError, match="not seeds of any block"):
        distributed_nmf(
            time_series_zarr=time_series,
            segments=segments,
            time_series_spacing=np.array([1.0, 1.0, 1.0]),
            segments_spacing=np.array([1.0, 1.0, 1.0]),
            only_cells=[99],
            cluster=SerialCluster(),
        )
