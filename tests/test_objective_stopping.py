"""Tests for the windowed relative stopping rule in segmentNMF.NMF_methods.

The rule these tests cover replaced two faults:
  * an ABSOLUTE objective difference, which cannot serve blocks of different
    size, because the objective grows with the voxel count;
  * a break on a SINGLE uphill step, which gradient descent produces routinely.

Each test below pins one of those two properties down.
"""

import numpy as np
import pytest

from segmentNMF.NMF_methods import (footprint_settled, nmf,
                                    objective_settled)

WINDOW = 5
MIN_IT = 0


def settle_step(objectives, rtol=1e-4, window=WINDOW, min_iterations=MIN_IT):
    """Run the rule over a whole objective curve; return the stopping index."""
    for i in range(len(objectives)):
        if objective_settled(objectives[:i + 1], i, rtol, window, min_iterations):
            return i
    return None


def test_flat_curve_stops():
    """A curve that no longer falls must stop at the first complete pair of windows."""
    flat = [100.0] * 40
    assert settle_step(flat) == 2 * WINDOW - 1


def test_steady_fall_does_not_stop():
    """A curve still falling by 10 percent per step must run to the end."""
    curve = [100.0 * 0.9 ** k for k in range(40)]
    assert settle_step(curve) is None


def test_single_uphill_step_does_not_stop():
    """One step upward must not end the fit. This was the old rule's main fault."""
    curve = [100.0 * 0.9 ** k for k in range(40)]
    curve[12] = curve[11] * 1.05          # a noisy step, larger than its neighbour
    assert settle_step(curve) is None


def test_sustained_rise_stops():
    """A whole window that rises is a real failure, and must still stop the fit.

    The first window is flat, so the rise in the second window is the only
    signal. A rising window gives a negative drop, below any positive rtol.
    """
    curve = [100.0] * WINDOW + [100.0 + k for k in range(1, WINDOW + 1)]
    assert settle_step(curve) == 2 * WINDOW - 1


def test_scale_free():
    """The same curve times 10**6 must stop on the same iteration.

    This is the property the absolute threshold did not have. A big block and a
    small block differ by exactly such a factor.

    The curve settles onto 50 instead of onto 0. A curve that keeps falling by a
    fixed FRACTION every step never settles, by construction, and must not stop.
    """
    curve = [50.0 + 50.0 * 0.9 ** k for k in range(200)]
    big = [v * 1e6 for v in curve]
    assert settle_step(curve) is not None
    assert settle_step(curve) == settle_step(big)


def test_only_tested_on_window_boundaries():
    """The two means must never overlap, so the test only runs every `window`."""
    flat = [100.0] * 40
    hits = [i for i in range(len(flat))
            if objective_settled(flat[:i + 1], i, 1e-4, WINDOW, MIN_IT)]
    assert all((i + 1) % WINDOW == 0 for i in hits)


def test_min_iterations_is_respected():
    """min_iterations must hold the fit open even on a flat curve."""
    flat = [100.0] * 60
    assert settle_step(flat, min_iterations=40) == 39


def test_none_disables_the_rule():
    """rtol None must never stop the fit."""
    flat = [100.0] * 40
    assert settle_step(flat, rtol=None) is None


def toy_problem(seed=0, n_vox=60, n_time=40, k=2, noise=0.05):
    """A small rank k problem, plus a start point close to the true answer.

    V carries noise on purpose. Without it the objective falls toward zero at a
    fixed fraction per step, and a relative rule then never settles. Real data
    always leaves a positive floor, and that floor is what makes the rule fire.

    nmf writes into S_init and H_init in place when the noise component is off,
    so every call must get its own copy. Returns a fresh kwargs dict each time.
    """
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


# footprint_rtol off, so these tests isolate the objective rule. The two rules
# are deliberately independent requirements; see test_both_rules_must_agree.
NMF_KW = dict(num_iterations=1000, update_int=10 ** 6, H_lr=1e-1, S_lr=1e-1,
              estimate_noise_component=False, min_iterations=0,
              footprint_rtol=None)


def test_nmf_stops_early_and_on_a_boundary():
    """End to end: the rule must cut a real nmf run short, at a window boundary."""
    p = toy_problem()
    _, _, objectives = nmf(p["V"], p["S_init"], p["H_init"], p["B"],
                           objective_rtol=1e-4, objective_window=25, **NMF_KW)
    assert len(objectives) < 1000, "the rule never fired"
    assert len(objectives) % 25 == 0, "stopped off a window boundary"


def test_nmf_runs_to_the_end_when_both_rules_are_off():
    """Both rtols None must let the run reach num_iterations.

    One None alone is not enough. It drops that test as a requirement, and the
    other test can then stop the run on its own.
    """
    p = toy_problem()
    _, _, objectives = nmf(p["V"], p["S_init"], p["H_init"], p["B"],
                           objective_rtol=None, **NMF_KW)
    assert len(objectives) == 1000


def test_footprint_rule_alone_can_stop_the_run():
    """With the objective test off, a settled footprint must still stop the fit."""
    p = toy_problem()
    kw = dict(NMF_KW)
    kw["footprint_rtol"] = 1e-3
    _, _, objectives = nmf(p["V"], p["S_init"], p["H_init"], p["B"],
                           objective_rtol=None, footprint_window=50, **kw)
    assert len(objectives) < 1000, "the footprint rule never fired"
    assert len(objectives) % 50 == 0, "stopped off a footprint window boundary"


def test_both_rules_must_agree():
    """With both rules on, the run cannot stop before EITHER is satisfied.

    The footprint settles later than the objective on real data, so requiring
    both must not stop earlier than requiring the footprint alone.
    """
    kw = dict(NMF_KW)
    kw["footprint_rtol"] = 1e-3
    p = toy_problem()
    _, _, only_fp = nmf(p["V"], p["S_init"], p["H_init"], p["B"],
                        objective_rtol=None, footprint_window=50, **kw)
    q = toy_problem()
    _, _, both = nmf(q["V"], q["S_init"], q["H_init"], q["B"],
                     objective_rtol=1e-4, objective_window=25,
                     footprint_window=50, **kw)
    assert len(both) >= len(only_fp)


def test_footprint_settled_needs_two_samples():
    """One sample cannot show a change, so it must never stop the fit."""
    assert footprint_settled([500], 1e-3) is False
    assert footprint_settled([], 1e-3) is False


def test_footprint_settled_reads_a_relative_change():
    """The test is a fraction of the current count, not an absolute count."""
    assert footprint_settled([1000, 1000], 1e-3) is True
    assert footprint_settled([1002, 1000], 1e-3) is False   # 0.2 percent
    assert footprint_settled([1000000, 1000200], 1e-3) is True
    # an empty footprint carries no information, so it must not count as settled
    assert footprint_settled([0, 0], 1e-3) is False


# NOTE: scale freeness is tested on synthetic curves, in test_scale_free, and
# not end to end. Scaling V inside nmf also scales the gradients, while H_lr and
# S_lr stay fixed. The scaled run then follows a different path and can diverge,
# so such a test would measure the learning rates, not the stopping rule.


def test_pytorch_stops_early_by_the_same_rule():
    """Both implementations obey the rule, but they no longer share a path.

    This test used to assert len(a) == len(b). That held while both step sizes
    divided by the plain sum. nmf_pytorch now defaults to a sum of squares, and
    the numpy `nmf` always uses the plain sum, so the two take different steps
    and cannot stop on the same iteration. An equal iteration count was never
    part of the rule.

    The assertion is now what the rule does promise. Each run must stop before
    the maximum, on the relative drop, and neither may run away.

    nmf_pytorch uses the GPU when one is present. The problem here is 60 by 40,
    so the run costs almost nothing on a shared card.
    """
    torch = pytest.importorskip("torch")
    del torch
    from segmentNMF.NMF_methods import nmf_pytorch

    kw = dict(objective_rtol=1e-4, objective_window=25, **NMF_KW)
    p = toy_problem()
    _, _, a = nmf(p["V"], p["S_init"], p["H_init"], p["B"], **kw)
    q = toy_problem()
    kw.pop("estimate_noise_component")       # nmf_pytorch has no such option
    _, _, b = nmf_pytorch(q["V"], q["S_init"], q["H_init"], q["B"], **kw)

    n_max = NMF_KW["num_iterations"]
    for name, curve in (("numpy", a), ("torch", b)):
        assert len(curve) < n_max, f"{name} never settled"
        # the rule is tested only every `window` iterations, so the stop must
        # land on a multiple of the window
        assert len(curve) % 25 == 0, f"{name} stopped off the window grid"
        obj = np.asarray([float(o) for o in curve])
        assert np.isfinite(obj).all(), f"{name} went non finite"
        assert obj[-1] < obj[0], f"{name} did not reduce the objective"


def test_objective_threshold_is_refused_by_nmf():
    """The removed argument must stop the run, with a message that names the
    replacement. A silent ignore would let an old config run with a stopping
    rule it did not ask for.
    """
    p = toy_problem(seed=1)
    with pytest.raises(ValueError, match="objective_threshold was removed"):
        nmf(p["V"], p["S_init"], p["H_init"], p["B"], objective_threshold=1e-6,
            num_iterations=5, estimate_noise_component=False)


def test_objective_threshold_is_refused_by_nmf_pytorch():
    """The same for the GPU fit, before any data moves to the device."""
    pytest.importorskip("torch")
    from segmentNMF.NMF_methods import nmf_pytorch
    p = toy_problem(seed=1)
    with pytest.raises(ValueError, match="objective_threshold was removed"):
        nmf_pytorch(p["V"], p["S_init"], p["H_init"], p["B"],
                    objective_threshold=1e-2, num_iterations=5)


def test_leaving_objective_threshold_unset_is_fine():
    """The default is None, so a caller that never set it sees no change."""
    p = toy_problem(seed=1)
    _, _, objectives = nmf(p["V"], p["S_init"], p["H_init"], p["B"],
                           num_iterations=5, update_int=1000,
                           estimate_noise_component=False)
    assert len(objectives) == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
