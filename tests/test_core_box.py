"""Tests for core_box: which cells get an output row.

WHY THE PARAMETER EXISTS
    A caller that works on a padded tile must keep the padding cells in the
    array. If you remove a cell, the cells next to it take over its voxels,
    and their traces pick up its activity.

    A padding cell still gets no output row. This tile does not hold all of
    its neighbours. The next tile holds them, and writes its trace.

WHY A BOX AND NOT A PADDING THICKNESS
    A caller clips the padding where the tile meets the edge of the dataset.
    The padding is then not the same on both sides. A box says the region
    directly and cannot be wrong.

THE RISK THIS GUARDS
    segment_ids[i] must describe boxes[i]. The centres, the neighbour tree and
    the block crops all use that pairing. Filtering one list without the other
    would give a cell another cell's box.
"""

import numpy as np
import pytest

from segmentNMF.distributed_NMF_methods import (as_box, distributed_nmf,
                                                filter_block_seeds)


# Four cells in a 2D array, each with an id and a bounding box.
BOXES = [(slice(0, 2), slice(0, 3)),
         (slice(4, 6), slice(1, 2)),
         (slice(7, 9), slice(5, 8)),
         (slice(9, 11), slice(2, 4))]
IDS = [3, 7, 11, 40]

# centres: 3 -> (1.0, 1.5), 7 -> (5.0, 1.5), 11 -> (8.0, 6.5), 40 -> (10.0, 3.0)


def test_none_changes_nothing():
    """None must mean no filter. Both lists come back as they went in."""
    ids, boxes = filter_block_seeds(list(IDS), list(BOXES), None)
    assert ids == IDS and boxes == BOXES


def test_a_box_keeps_each_id_with_its_own_box():
    """The two lists must stay in step.

    The box holds the centres of cells 7 and 40. Each kept id must come back
    with its own box, not with a neighbour's box.
    """
    ids, boxes = filter_block_seeds(list(IDS), list(BOXES),
                                    (slice(4, 11), slice(0, 5)))
    assert ids == [7, 40]
    assert boxes == [BOXES[1], BOXES[3]]


def test_start_stop_pairs_mean_the_same_as_slices():
    """A config file cannot hold a slice, so it holds (start, stop) pairs."""
    ids, _ = filter_block_seeds(list(IDS), list(BOXES), [(4, 11), (0, 5)])
    assert ids == [7, 40]


def test_the_box_is_half_open():
    """start is inside and stop is outside, like a python slice.

    Two tiles share a face. Exactly one of them must claim a cell on it.
    """
    ids, _ = filter_block_seeds(list(IDS), list(BOXES),
                                (slice(1, 5), slice(0, 4)))
    assert ids == [3]


def test_numpy_bounds_work():
    """Bounds read from a config file arrive as numpy integers."""
    core = np.array([[4, 11], [0, 5]], dtype=np.int64)
    ids, _ = filter_block_seeds(list(IDS), list(BOXES), core)
    assert ids == [7, 40]


def test_a_cell_is_judged_by_its_centre():
    """Only the centre counts, not any overlap with the box.

    Cell 7 starts at z 4 and touches the box, but its centre is at z 5.0. It
    is dropped. An overlap rule would let two tiles claim the same cell.
    """
    ids, _ = filter_block_seeds(list(IDS), list(BOXES),
                                (slice(0, 3), slice(0, 20)))
    assert ids == [3]


def test_an_empty_box_fails_loudly():
    """A box that holds no centre must raise, not return an empty list.

    An empty list would run the whole fit and write no trace at all.
    """
    with pytest.raises(ValueError, match="centre of no segment"):
        filter_block_seeds(list(IDS), list(BOXES), (slice(0, 1), slice(0, 1)))


def test_mismatched_lengths_fail_loudly():
    """Lists of different length must raise.

    zip() would stop at the shorter list and drop cells without a word.
    """
    with pytest.raises(ValueError, match="disagree"):
        filter_block_seeds(list(IDS), list(BOXES)[:2],
                           (slice(0, 20), slice(0, 20)))


def test_wrong_number_of_axes_fails_loudly():
    """A 3 axis box against a 2 axis array must raise.

    zip() would drop the third axis and keep the wrong cells.
    """
    with pytest.raises(ValueError, match="axes"):
        filter_block_seeds(list(IDS), list(BOXES),
                           (slice(0, 20), slice(0, 20), slice(0, 20)))


def test_an_open_ended_box_fails_loudly():
    """slice(None) has no start and no stop, so it cannot say a region."""
    with pytest.raises(ValueError, match="explicit start and stop"):
        as_box((slice(0, 20), slice(None)))


def test_the_error_reaches_a_caller_of_distributed_nmf():
    """The check must run before distributed_nmf reads any data.

    A cluster run costs hours. The dummy arguments here prove it fails first.
    """
    with pytest.raises(ValueError, match="centre of no segment"):
        distributed_nmf(
            time_series_zarr=None,
            segments=np.array([[[1, 0], [0, 2]]], dtype=np.uint32),
            time_series_spacing=np.array([4.0, 1.0, 1.0]),
            segments_spacing=np.array([1.0, 1.0, 1.0]),
            core_box=(slice(0, 1), slice(5, 6), slice(5, 6)),
            use_gpu=False,
        )
