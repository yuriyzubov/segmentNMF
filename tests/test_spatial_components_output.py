"""Tests for return_spatial_components in distributed_nmf.

THE RISK THIS GUARDS
    Each block fits its own crop, so its footprints are in crop coordinates.
    They are moved into one matrix over the whole canvas. A wrong offset would
    put a footprint on the wrong voxels, with no error. So each test below
    checks a cell's footprint against its own mask on the whole canvas,
    including a cell in a block far from the origin.
"""

import numpy as np
import scipy.sparse

from segmentNMF.distributed_NMF_methods import distributed_nmf


class SerialCluster:
    """Stands in for a ClusterWrap cluster. Runs each block in this process."""

    def __init__(self):
        self.client = self

    def map(self, func, *iterables):
        return list(map(func, *iterables))

    def gather(self, futures):
        return futures


def two_pair_volume():
    """Four cells in two pairs, far apart, so each pair is one block."""
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


def run(tmp_path, **kw):
    segments, time_series = two_pair_volume()
    out = distributed_nmf(
        time_series_zarr=time_series,
        segments=segments,
        time_series_spacing=np.array([1.0, 1.0, 1.0]),
        segments_spacing=np.array([1.0, 1.0, 1.0]),
        max_complete_cells_per_block=2,
        max_neighbor_distance=10.0,
        temporary_directory=str(tmp_path),
        num_iterations=20,
        min_iterations=20,
        cluster=SerialCluster(),
        **kw,
    )
    return segments, out


def test_off_by_default_returns_only_the_traces(tmp_path):
    """The default must keep the old return type."""
    _, out = run(tmp_path)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 4


def test_the_matrix_covers_the_whole_canvas(tmp_path):
    segments, (traces, S) = run(tmp_path, return_spatial_components=True)
    assert scipy.sparse.issparse(S)
    assert S.shape == (segments.size, 4)
    assert traces.shape[0] == S.shape[1]


def test_each_footprint_sits_on_its_own_mask(tmp_path):
    """The peak of each column must lie inside that cell's own mask.

    Cells 3 and 4 are in the right block, whose crop starts far from x = 0.
    A missing crop offset would put their footprints on the left.
    """
    segments, (_, S) = run(tmp_path, return_spatial_components=True)
    labels = segments.reshape(-1)
    for cell in range(1, 5):
        column = S[:, cell - 1].toarray().ravel()
        assert column.max() > 0, f"cell {cell} has an empty footprint"
        assert labels[int(np.argmax(column))] == cell, (
            f"cell {cell} peaks on another cell's voxel")


def test_cells_that_did_not_run_have_empty_columns(tmp_path):
    """With only_cells, the other block does not run, so its columns are 0."""
    _, (traces, S) = run(tmp_path, return_spatial_components=True,
                         only_cells=[1])
    assert S[:, 0].nnz > 0 and S[:, 1].nnz > 0
    assert S[:, 2].nnz == 0 and S[:, 3].nnz == 0
    assert np.isnan(traces[2]).all() and np.isnan(traces[3]).all()
