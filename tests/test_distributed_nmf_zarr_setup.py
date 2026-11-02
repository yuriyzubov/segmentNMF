"""Tests for the zarr array-creation pattern used by distributed_nmf's setup step.

distributed_NMF_methods.distributed_nmf creates several zarr arrays
(segments_zarr, reconstruction_zarr, reconstruction_weights_zarr) with
zarr.open(path, mode='w', shape=..., chunks=..., dtype=...). This mirrors
that exact call signature (minus the synchronizer=, tracked separately)
to guard against regressions in that usage under zarr v3.
"""

import numpy as np
import zarr


def _create_array_like_distributed_nmf(path, shape, chunks, dtype):
    """Reproduces the array-creation call used to set up distributed_nmf's zarr arrays."""
    return zarr.open(path, mode='w', shape=shape, chunks=chunks, dtype=dtype)


def test_array_setup_pattern_roundtrip(tmp_path):
    """The array-creation pattern should produce a writable array whose data survives a fresh read."""
    path = tmp_path / "setup.zarr"
    data = np.arange(64, dtype="float32").reshape(8, 8)

    arr = _create_array_like_distributed_nmf(str(path), data.shape, (4, 4), data.dtype)
    assert arr.chunks == (4, 4)
    assert arr.dtype == data.dtype

    arr[...] = data
    reopened = zarr.open(str(path), mode="r")
    np.testing.assert_array_equal(reopened[:], data)
