"""Tests for the windowed relative stopping rule in segmentNMF.NMF_methods.

The rule these tests cover replaced two faults:
  * an ABSOLUTE objective difference, which cannot serve blocks of different
    size, because the objective grows with the voxel count;
  * a break on a SINGLE uphill step, which gradient descent produces routinely.

Each test below pins one of those two properties down.
"""

from segmentNMF.NMF_methods import objective_settled

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
