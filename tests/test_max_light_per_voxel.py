"""Tests for max_light_per_voxel: the most light a cell can put in each voxel.

WHY IT EXISTS
    In a voxel that two cells share, the fit cannot tell which cell gave the
    light. Without a limit, a small or dim cell can lower the error by taking a
    bright neighbour's light, and become bright on the neighbour's core. The
    limit is the cell's mask blurred by the optics: high on its own core, and
    falling to zero away from it.

WHAT IS TESTED
    The limit itself, the fit that keeps each footprint under it, and the hand
    off from distributed_nmf to the fit.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import segmentNMF.distributed_NMF_methods as dnm  # noqa: E402
from segmentNMF.distributed_NMF_methods import distributed_nmf  # noqa: E402
from segmentNMF.NMF_methods import nmf_pytorch  # noqa: E402
from segmentNMF.neighborhood import max_light_per_voxel  # noqa: E402

SPACING = (1.0, 1.0, 1.0)


def two_cell_segments():
    """Two square cells side by side on one plane, with a gap between them."""
    segments = np.zeros((1, 10, 16), dtype=np.uint32)
    segments[0, 3:7, 2:6] = 1
    segments[0, 3:7, 9:13] = 2
    return segments


# ------------------------------------------------------------- the limit

def test_shapes_follow_the_time_series_grid():
    segments = two_cell_segments()
    S_max, core = max_light_per_voxel(segments, [1, 2], 1, SPACING,
                                      sigma_xy=1.0, sigma_z=1.0,
                                      widen_xy=0.0, widen_z=0.0)
    assert S_max.shape == core.shape == (segments.size, 2)
    assert core.dtype == bool


def test_full_on_the_core_and_zero_far_away():
    """1 on the cell's own core, 0 far from it, and between 0 and 1 elsewhere."""
    segments = two_cell_segments()
    S_max, core = max_light_per_voxel(segments, [1, 2], 1, SPACING,
                                      sigma_xy=1.0, sigma_z=1.0,
                                      widen_xy=0.0, widen_z=0.0)
    flat = segments.reshape(-1)
    for i, lab in enumerate((1, 2)):
        assert core[:, i].any(), f"cell {lab} has no core"
        assert (flat[core[:, i]] == lab).all(), f"cell {lab}'s core leaves its mask"
        assert np.allclose(S_max[core[:, i], i], 1.0)
        other = 2 if lab == 1 else 1
        far = flat == other
        assert S_max[far, i].max() < 0.05, f"cell {lab} may light its neighbour"
        assert (S_max[:, i] >= 0).all() and (S_max[:, i] <= 1).all()


def test_a_wider_blur_reaches_further():
    """sigma_xy must reach the limit, or the setting is not connected."""
    segments = two_cell_segments()
    narrow, _ = max_light_per_voxel(segments, [1], 1, SPACING, sigma_xy=0.5,
                                    sigma_z=1.0, widen_xy=0.0, widen_z=0.0)
    wide, _ = max_light_per_voxel(segments, [1], 1, SPACING, sigma_xy=2.0,
                                  sigma_z=1.0, widen_xy=0.0, widen_z=0.0)
    assert (wide[:, 0] > 1e-3).sum() > (narrow[:, 0] > 1e-3).sum()


# --------------------------------------------------------------- the fit

def dim_and_bright_toy():
    """A dim cell next to a bright one, and a mask that lets both grow anywhere."""
    segments = two_cell_segments()
    n_vox, n_frames = segments.size, 60
    rng = np.random.default_rng(0)
    S_true = np.stack([(segments.reshape(-1) == lab).astype(np.float32)
                       for lab in (1, 2)], axis=1)
    H_true = np.stack([np.abs(rng.normal(2.0, 0.5, n_frames)),
                       np.abs(rng.normal(40.0, 10.0, n_frames))]).astype(np.float32)
    V = np.maximum(S_true @ H_true + rng.normal(0, 0.5, (n_vox, n_frames)), 0)
    S_max, core = max_light_per_voxel(segments, [1, 2], 1, SPACING,
                                      sigma_xy=1.0, sigma_z=1.0,
                                      widen_xy=0.0, widen_z=0.0)
    args = dict(V=V.astype(np.float32),
                S_init=np.where(S_true > 0, 0.8, 0.05).astype(np.float32),
                H_init=np.ones((2, n_frames), dtype=np.float32),
                B=np.ones((n_vox, 2), dtype=np.float32),
                num_iterations=200, update_int=10 ** 6, H_lr=0.5, S_lr=0.5,
                min_iterations=200, objective_rtol=None, footprint_rtol=None)
    return args, S_max, core


def test_every_footprint_stays_under_its_limit():
    """After the fit, S <= S_max * (peak of S on its own core), for every voxel."""
    args, S_max, core = dim_and_bright_toy()
    S, _, _ = nmf_pytorch(S_max=S_max, S_max_core=core, **args)
    for i in range(2):
        m = S[core[:, i], i].max()
        assert (S[:, i] <= S_max[:, i] * m + 1e-6).all(), f"cell {i} passed its limit"


def test_the_limit_keeps_the_dim_cell_off_its_neighbour():
    """Off, the dim cell may spread onto the bright cell. On, it must not."""
    args, S_max, core = dim_and_bright_toy()
    bright = two_cell_segments().reshape(-1) == 2
    S_on, _, _ = nmf_pytorch(S_max=S_max, S_max_core=core, **args)
    assert S_on[bright, 0].max() < 0.05 * S_on[:, 0].max()


def test_the_two_inputs_go_together():
    args, S_max, core = dim_and_bright_toy()
    with pytest.raises(ValueError, match="go together"):
        nmf_pytorch(S_max=S_max, **args)


def test_a_wrong_shape_fails_loudly():
    args, S_max, core = dim_and_bright_toy()
    with pytest.raises(ValueError, match="must have the shape of S_init"):
        nmf_pytorch(S_max=S_max[:-1], S_max_core=core[:-1], **args)


# ------------------------------------------------------------ the hand off

class SerialCluster:
    """Stands in for a ClusterWrap cluster. Runs each block in this process."""

    def __init__(self):
        self.client = self

    def map(self, func, *iterables):
        return list(map(func, *iterables))

    def gather(self, futures):
        return futures


def test_distributed_nmf_hands_the_limit_to_the_fit(tmp_path, monkeypatch):
    """Each block must build the limit and pass it to nmf_pytorch."""
    seen = []
    real = dnm.nmf_pytorch

    def recording(V, S, H, B, **kwargs):
        seen.append((kwargs.get("S_max"), kwargs.get("S_max_core"), S.shape))
        return real(V, S, H, B, **kwargs)

    monkeypatch.setattr(dnm, "nmf_pytorch", recording)
    monkeypatch.setattr(dnm, "_fit_waiting_for_gpu", lambda fit: fit())
    segments = two_cell_segments()
    rng = np.random.default_rng(0)
    time_series = np.abs(rng.normal(5.0, 1.0, (20,) + segments.shape)).astype(np.float32)
    distributed_nmf(
        time_series_zarr=time_series, segments=segments,
        time_series_spacing=np.array(SPACING), segments_spacing=np.array(SPACING),
        max_light_per_voxel=dict(sigma_xy=1.0, sigma_z=1.0, widen_xy=0.0,
                                 widen_z=0.0, core_level=0.8),
        use_gpu=True, temporary_directory=str(tmp_path),
        num_iterations=5, min_iterations=5, cluster=SerialCluster())
    assert seen, "no block ran"
    for S_max, core, shape in seen:
        assert S_max is not None and core is not None
        assert S_max.shape == core.shape == shape


def test_the_limit_requires_the_gpu_fit():
    with pytest.raises(ValueError, match="requires use_gpu=True"):
        distributed_nmf(
            time_series_zarr=None, segments=np.zeros((1, 2, 2), dtype=np.uint32),
            time_series_spacing=np.array(SPACING), segments_spacing=np.array(SPACING),
            max_light_per_voxel=dict(sigma_xy=1.0), use_gpu=False)
