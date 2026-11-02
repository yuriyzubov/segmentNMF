"""Tests for segmentNMF.file_handling read/write behavior on zarr v3."""

import numpy as np
import pytest
import zarr

from segmentNMF.file_handling import create_temp_zarr, create_zarr, file_handler


def test_create_zarr_roundtrip(tmp_path):
    """create_zarr should produce a writable array whose data survives a fresh read."""
    path = tmp_path / "created.zarr"
    data = np.arange(400, dtype="float32").reshape(20, 20)

    arr = create_zarr(str(path), shape=data.shape, chunks=(5, 5), dtype=data.dtype)
    assert arr.shape == data.shape
    assert arr.dtype == data.dtype
    assert arr.chunks == (5, 5)

    arr[:] = data
    reopened = zarr.open(str(path), mode="r")
    np.testing.assert_array_equal(reopened[:], data)


def test_file_handler_reads_zarr(tmp_path):
    """file_handler should lazily open a .zarr array and expose its data unchanged."""
    path = tmp_path / "handled.zarr"
    data = np.arange(60, dtype="uint16").reshape(6, 10)
    zarr.create_array(str(path), shape=data.shape, dtype=data.dtype, overwrite=True)[:] = data

    result = file_handler(str(path))
    np.testing.assert_array_equal(result[:], data)


def test_file_handler_reads_nested_multiscale_zarr_path(tmp_path):
    """file_handler should resolve paths into a multiscale group, e.g. foo.zarr/s0."""
    group_path = tmp_path / "multiscale.zarr"
    data = np.arange(24, dtype="uint32").reshape(4, 6)
    zarr.create_array(f"{group_path}/s0", shape=data.shape, dtype=data.dtype, overwrite=True)[:] = data

    result = file_handler(f"{group_path}/s0")
    np.testing.assert_array_equal(result[:], data)


def test_file_handler_reads_npy_via_temp_zarr(tmp_path):
    """file_handler should load a .npy file by staging it into a temporary zarr array."""
    path = tmp_path / "data.npy"
    data = np.arange(12, dtype="int32").reshape(3, 4)
    np.save(path, data)

    result = file_handler(str(path))
    np.testing.assert_array_equal(result[:], data)


def test_file_handler_rejects_unsupported_extension(tmp_path):
    """file_handler should raise ValueError for file types it doesn't recognize."""
    path = tmp_path / "data.txt"
    path.write_text("not a supported format")

    with pytest.raises(ValueError):
        file_handler(str(path))


def test_create_temp_zarr_matches_source_array():
    """create_temp_zarr should return a zarr array whose contents match the input ndarray."""
    data = np.random.default_rng(0).random((5, 5)).astype("float64")
    arr = create_temp_zarr(data)
    np.testing.assert_array_equal(arr[:], data)
