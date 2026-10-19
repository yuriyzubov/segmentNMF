"""_requested_bytes: the size of a GPU allocation that failed, from its error message."""
import pytest
import torch

import segmentNMF.distributed_NMF_methods as dm

GIB = 2**30


def oom(gib=2.0):
    return torch.OutOfMemoryError(
        'CUDA out of memory. Tried to allocate {:.2f} GiB. GPU 0 has a total capacity '
        'of 80.00 GiB of which 1.00 GiB is free.'.format(gib))


def test_requested_bytes_reads_the_failed_allocation():
    assert dm._requested_bytes(oom(3.33)) == int(3.33 * GIB)
    assert dm._requested_bytes(RuntimeError('Tried to allocate 20.00 MiB.')) == 20 * 2**20
    assert dm._requested_bytes(RuntimeError('no size here')) == 0
