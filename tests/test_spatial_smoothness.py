"""Tests for the lateral smoothness term in segmentNMF.NMF_methods.nmf_pytorch.

The term exists because nothing else in the objective couples a voxel to the
voxel beside it, so the fit is free to cut one footprint into many small
pieces.

Three properties matter most, and each has a test below:
  * with the term off, nothing changes at all;
  * a z weight of 0 keeps the planes apart, and a z weight above 0 mixes them;
  * a missing neighbour at a face gives its weight to the neighbours present.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import (lateral_neighbour_counts,  # noqa: E402
                                    lateral_neighbour_mean, nmf_pytorch,
                                    parse_spatial_smoothness)


def counts_of(shape):
    return lateral_neighbour_counts(shape, dtype=torch.float32)


def test_counts_are_4_inside_3_on_an_edge_2_in_a_corner():
    c = counts_of((1, 4, 5))[0, 0]
    assert c[2, 2] == 4          # inside
    assert c[0, 2] == 3          # top edge
    assert c[2, 0] == 3          # left edge
    assert c[0, 0] == 2          # corner
    assert c[-1, -1] == 2


def test_counts_never_reach_zero():
    """A single voxel plane has no neighbours, but must still be safe to divide by.

    The floor cannot be 1.0. A weight vector such as [0, 0.025, 0.025] gives an
    interior count of 0.1, and a floor of 1.0 would silently rescale it. The floor
    is only there to stop a division by zero, and the numerator is 0 there too.
    """
    c = counts_of((3, 1, 1))
    assert float(c[0, 0, 0, 0]) > 0.0
    x = torch.ones((1, 3, 1, 1))
    assert float(lateral_neighbour_mean(x, c).abs().max()) == 0.0


def test_a_single_hot_voxel_spreads_to_its_4_neighbours():
    x = torch.zeros((1, 1, 5, 5))
    x[0, 0, 2, 2] = 1.0
    avg = lateral_neighbour_mean(x, counts_of((1, 5, 5)))[0, 0]
    assert avg[2, 2] == 0.0                      # itself is excluded
    for y, z in ((1, 2), (3, 2), (2, 1), (2, 3)):
        assert avg[y, z] == pytest.approx(0.25)  # 1 of 4 neighbours is hot
    assert avg[1, 1] == 0.0                      # a diagonal is not a neighbour


def test_the_mean_is_lateral_only():
    """Two z planes must never mix. One z step is 4000 nm, one lateral step 406 nm."""
    x = torch.zeros((1, 2, 3, 3))
    x[0, 0] = 1.0                                # plane 0 full, plane 1 empty
    avg = lateral_neighbour_mean(x, counts_of((2, 3, 3)))[0]
    assert float(avg[1].abs().max()) == 0.0, "plane 1 picked up plane 0"
    assert float(avg[0, 1, 1]) == pytest.approx(1.0)


def test_a_one_voxel_gap_gets_filled():
    """The reason the term exists: a hole inside a patch must be lifted."""
    x = torch.ones((1, 1, 5, 5))
    x[0, 0, 2, 2] = 0.0
    avg = lateral_neighbour_mean(x, counts_of((1, 5, 5)))[0, 0]
    assert float(avg[2, 2]) == pytest.approx(1.0)


def test_an_isolated_speck_gets_pushed_down():
    """The same term must weaken a voxel with no neighbours."""
    x = torch.zeros((1, 1, 5, 5))
    x[0, 0, 2, 2] = 1.0
    avg = lateral_neighbour_mean(x, counts_of((1, 5, 5)))[0, 0]
    assert float(avg[2, 2]) == 0.0


def toy_problem(seed=0, spatial_shape=(2, 5, 6), n_time=40, k=2, noise=0.05):
    """A small rank k problem laid out on a real 3D canvas."""
    n_vox = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    rng = np.random.default_rng(seed)
    S_true = rng.random((n_vox, k)).astype(np.float32)
    H_true = rng.random((k, n_time)).astype(np.float32)
    V = S_true @ H_true + noise * rng.standard_normal((n_vox, n_time))
    return dict(
        V=V.astype(np.float32),
        S_init=(S_true + 0.1 * rng.random((n_vox, k))).astype(np.float32),
        H_init=(H_true + 0.1 * rng.random((k, n_time))).astype(np.float32),
        B=np.ones((n_vox, k), dtype=np.float32),
    )


RUN_KW = dict(num_iterations=40, update_int=10 ** 6, H_lr=1e-1, S_lr=1e-1,
              objective_rtol=None, footprint_rtol=None)
SHAPE = (2, 5, 6)


def run(smoothness, **extra):
    p = toy_problem()
    return nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                       spatial_smoothness=smoothness, **RUN_KW, **extra)


def test_off_by_default_changes_nothing():
    """Not passing the argument must give exactly the old result."""
    p = toy_problem()
    S_a, H_a, o_a = nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                                **RUN_KW)
    S_b, H_b, o_b = run(None, spatial_shape=SHAPE)
    np.testing.assert_array_equal(S_a, S_b)
    np.testing.assert_array_equal(H_a, H_b)
    np.testing.assert_allclose(o_a, o_b)


def test_zero_is_the_same_as_off():
    """0.0 must be treated as off, not as a smoothing of strength zero."""
    S_a, _, _ = run(None, spatial_shape=SHAPE)
    S_b, _, _ = run(0.0, spatial_shape=SHAPE)
    np.testing.assert_array_equal(S_a, S_b)


def test_on_changes_the_answer():
    S_a, _, _ = run(None, spatial_shape=SHAPE)
    S_b, _, _ = run(0.3, spatial_shape=SHAPE)
    assert not np.allclose(S_a, S_b), "the term had no effect"


def test_shape_is_required():
    with pytest.raises(ValueError, match="requires spatial_shape"):
        run(0.3)


def test_shape_must_match_the_voxel_count():
    with pytest.raises(ValueError, match="voxels, but S has"):
        run(0.3, spatial_shape=(2, 5, 7))


def test_shape_must_be_three_axes():
    with pytest.raises(ValueError, match=r"must be \(nz, ny, nx\)"):
        run(0.3, spatial_shape=(10, 6))


def test_weights_stay_inside_zero_and_one():
    """The clip must hold after the smoothing step, not only before it."""
    S, _, _ = run(0.8, spatial_shape=SHAPE)
    assert S.min() >= 0.0
    assert S.max() <= 1.0


def test_b_blocks_the_pull():
    """B scales the pull, so B all zeros must leave S at its start.

    The start is S_init clipped to [0, 1]. nmf_pytorch clips S every iteration,
    and this toy S_init holds a few values above 1.
    """
    p = toy_problem()
    p["B"] = np.zeros_like(p["B"])
    S, _, _ = nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                          spatial_smoothness=0.5, spatial_shape=SHAPE,
                          **RUN_KW)
    np.testing.assert_allclose(S, np.clip(p["S_init"], 0.0, 1.0), atol=1e-6)


def test_a_scalar_equals_the_same_value_on_all_three_axes():
    """A scalar L must be the same setting as [L, L, L], bit for bit.

    Only the ratios of the weights matter, because the mean divides by their sum.
    """
    p = toy_problem()
    a, _, _ = nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                          spatial_smoothness=0.2, spatial_shape=SHAPE, **RUN_KW)
    b, _, _ = nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                          spatial_smoothness=[0.2, 0.2, 0.2],
                          spatial_shape=SHAPE, **RUN_KW)
    np.testing.assert_array_equal(a, b)


def test_a_scalar_is_not_lateral_only():
    """A scalar pulls along z too, so it must differ from [0, L / 2, L / 2]."""
    p = toy_problem()
    a, _, _ = nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                          spatial_smoothness=0.2, spatial_shape=SHAPE, **RUN_KW)
    b, _, _ = nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                          spatial_smoothness=[0.0, 0.1, 0.1],
                          spatial_shape=SHAPE, **RUN_KW)
    assert np.abs(a - b).max() > 1e-4


def test_a_z_weight_above_zero_couples_the_planes():
    """The lateral only case must not mix planes, and the vector case must."""
    x = torch.zeros((1, 2, 3, 3))
    x[0, 0] = 1.0
    w = (0.25, 1.0, 1.0)
    avg = lateral_neighbour_mean(
        x, lateral_neighbour_counts((2, 3, 3), weights=w, dtype=x.dtype),
        weights=w)[0]
    assert float(avg[1].abs().max()) > 0.0, "plane 1 must see plane 0"
    # plane 1 centre: only neighbour is plane 0 at weight 0.25, of 4 + 0.25
    assert float(avg[1, 1, 1]) == pytest.approx(0.25 / 4.25)


def test_parse_reads_a_scalar_and_a_vector():
    # a scalar applies to all 3 axes, so the step is 3 times it
    assert parse_spatial_smoothness(0.05) == (pytest.approx(0.15),
                                              (1.0, 1.0, 1.0))
    assert parse_spatial_smoothness(None)[0] == 0.0
    step, w = parse_spatial_smoothness([0.01, 0.02, 0.02])
    assert step == pytest.approx(0.05)
    assert w == (0.5, 1.0, 1.0)                 # divided by the largest
    # a scalar means all 3 axes, so lateral only must be written out in full
    assert parse_spatial_smoothness([0.0, 0.025, 0.025]) == (0.05,
                                                             (0.0, 1.0, 1.0))
    assert parse_spatial_smoothness([0.0, 0.0, 0.0])[0] == 0.0
    for bad in ([0.1, 0.1], [0.0, -0.1, 0.1]):
        with pytest.raises(ValueError):
            parse_spatial_smoothness(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
