"""Tests that concurrent writes to disjoint zarr chunks don't need a synchronizer.

distributed_NMF_methods.py writes reconstruction_zarr from multiple worker
processes, but the calling code partitions the work (via block_flags in
map_all_blocks) so that no two workers ever touch the same chunk at the same
time. This simulates that exact pattern -- separate OS processes, no
synchronizer, writing to disjoint chunk-aligned regions -- to confirm no data
is lost or corrupted, supporting removal of zarr.ThreadSynchronizer() (which
doesn't exist in zarr v3).
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import zarr


def _write_block(path, block_index, block_shape):
    """Write a distinct constant value into one disjoint chunk-aligned block."""
    arr = zarr.open(path, mode='r+')
    start = block_index * block_shape[0]
    stop = start + block_shape[0]
    arr[start:stop, :] = block_index + 1
    return block_index


def test_concurrent_disjoint_writes_without_synchronizer(tmp_path):
    """Writing disjoint, chunk-aligned blocks from separate processes with no
    synchronizer should still produce a fully correct array."""
    path = str(tmp_path / "concurrent.zarr")
    n_blocks = 8
    block_shape = (4, 6)
    shape = (n_blocks * block_shape[0], block_shape[1])

    zarr.create_array(path, shape=shape, chunks=block_shape, dtype='float32', overwrite=True)

    with ProcessPoolExecutor(max_workers=n_blocks) as pool:
        futures = [pool.submit(_write_block, path, i, block_shape) for i in range(n_blocks)]
        completed = [f.result() for f in futures]

    assert sorted(completed) == list(range(n_blocks))

    result = zarr.open(path, mode='r')[:]
    for i in range(n_blocks):
        start = i * block_shape[0]
        stop = start + block_shape[0]
        np.testing.assert_array_equal(result[start:stop, :], i + 1)
