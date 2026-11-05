"""Tests for calcium_rule, the calcium-shaped trace rule.

WHAT THE RULE IS
    Each trace is baseline + c, with c[t] = gamma * c[t-1] + s[t] and s >= 0.
    A trace can rise by any amount in one frame. It cannot fall faster than
    gamma per frame. Its penalty adds an L1 penalty on s.

    penalty='noise' sets mu for each cell from the noise of its trace,
    measured from the falls only (trace_noise).

HOW IT IS CHECKED
    The pool algorithm is compared with a general solver on the same problem,
    written in the variable s. The two methods share no code.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
optimize = pytest.importorskip("scipy.optimize")

from segmentNMF.NMF_methods import (  # noqa: E402
    _ar1_project, calcium_step, nmf_pytorch, noise_feedback, trace_noise)


GAMMA = float(np.exp(-1.0 / 4.0))


def kernel_matrix(n_t, gamma):
    """K with c = K @ s: column j is gamma^(t - j) for t >= j."""
    t = np.arange(n_t)
    lag = t[:, None] - t[None, :]
    return np.where(lag >= 0, gamma ** np.maximum(lag, 0), 0.0)


def solve_in_s(y, gamma, a):
    """argmin over s >= 0 of 1/2 |K s - y|^2 + a * sum(s), by L-BFGS-B. Returns c."""
    K = kernel_matrix(y.size, gamma)

    def f(s):
        r = K @ s - y
        return 0.5 * r @ r + a * s.sum(), K.T @ r + a

    res = optimize.minimize(f, np.zeros(y.size), jac=True, method='L-BFGS-B',
                            bounds=[(0, None)] * y.size,
                            options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000))
    return K @ res.x


def noisy_rows(n_rows=6, n_t=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n_rows, n_t)) + np.linspace(2, -2, n_t)


def test_projection_matches_a_general_solver():
    """With no penalty, the pool algorithm gives the exact projection.

    A wrong merge rule still returns a trace that obeys the shape rule, so only
    a comparison with an independent solver finds it.
    """
    Y = noisy_rows()
    out = _ar1_project(Y, GAMMA)
    for y, c in zip(Y, out):
        np.testing.assert_allclose(c, solve_in_s(y, GAMMA, 0.0), atol=1e-5)


def test_penalty_matches_a_general_solver():
    """With a penalty, calcium_step gives the exact proximal step for a fixed baseline.

    The step moves the penalty into the target by a linear weight. A wrong
    weight, at the last frame for example, changes every frame of the answer.
    """
    Y = noisy_rows(seed=1)
    a = 0.7
    H = torch.tensor(Y, dtype=torch.float64)
    baseline = torch.zeros((Y.shape[0], 1), dtype=torch.float64)
    a_t = torch.full((Y.shape[0], 1), a, dtype=torch.float64)
    weights = np.full(Y.shape[1], 1.0 - GAMMA)
    weights[-1] = 1.0
    c = _ar1_project(Y - a * weights, GAMMA)
    for y, c_row in zip(Y, c):
        np.testing.assert_allclose(c_row, solve_in_s(y, GAMMA, a), atol=1e-5)
    H_out, _ = calcium_step(H, a_t, GAMMA, baseline)
    # the trace is the baseline from this c plus this c
    b = np.maximum((Y - c).mean(axis=1, keepdims=True), 0.0)
    np.testing.assert_allclose(H_out.numpy(), b + c, atol=1e-10)


def test_output_obeys_the_shape_rule():
    """No output frame falls faster than gamma, and no output is negative.

    A trace that breaks the rule means the merge loop stopped too early.
    """
    out = _ar1_project(noisy_rows(n_rows=20, n_t=300, seed=2), GAMMA)
    assert np.all(out >= 0)
    assert np.all(out[:, 1:] >= GAMMA * out[:, :-1] - 1e-9)


def test_a_calcium_trace_is_kept():
    """With no penalty, a trace that already has the calcium shape does not change.

    The rule must block only fast falls. If it changes a true calcium trace, it
    distorts the events it is meant to keep.
    """
    n_t = 200
    s = np.zeros(n_t)
    s[[10, 50, 53, 120]] = [5.0, 3.0, 4.0, 8.0]
    c = kernel_matrix(n_t, GAMMA) @ s
    H = torch.tensor(3.0 + c[None, :])
    H_out, b = calcium_step(H, torch.zeros((1, 1), dtype=H.dtype), GAMMA,
                            torch.full((1, 1), 3.0, dtype=H.dtype))
    np.testing.assert_allclose(H_out.numpy()[0], 3.0 + c, atol=1e-9)
    assert float(b) == pytest.approx(3.0)


def test_the_baseline_has_no_penalty():
    """A flat trace keeps its level under a large penalty.

    With the baseline inside c, the penalty pulls a flat trace down every
    frame, and every dim cell loses its level.
    """
    H = torch.full((1, 100), 5.0, dtype=torch.float64)
    a = torch.full((1, 1), 2.0, dtype=torch.float64)
    baseline = torch.zeros((1, 1), dtype=torch.float64)
    for _ in range(50):
        H_out, baseline = calcium_step(H, a, GAMMA, baseline)
    np.testing.assert_allclose(H_out.numpy(), 5.0, atol=1e-6)


def one_cell(n_t=150):
    S = np.zeros((16, 1), dtype=np.float32)
    S[4:12, 0] = 1.0
    rng = np.random.default_rng(3)
    H_true = (10.0 + rng.normal(0.0, 1.0, (1, n_t))).astype(np.float32)
    V = S @ H_true
    return V, S, H_true, np.ones_like(S)


def test_nmf_traces_obey_the_shape_rule():
    """The traces that nmf_pytorch returns never fall faster than gamma.

    H = baseline + c with baseline >= 0, so H[t] >= gamma * H[t-1] holds too.
    The step must be the last change to H in each iteration. A later change
    returns traces that break the rule.
    """
    V, S, H, B = one_cell()
    _, H_out, _ = nmf_pytorch(V, S, H, B, num_iterations=30, objective_rtol=None,
                              footprint_rtol=None, update_int=1000,
                              calcium_rule=dict(decay=4.0, penalty=0.1))
    h = H_out[0]
    assert np.all(h[1:] >= GAMMA * h[:-1] - 1e-3)


@pytest.mark.parametrize('kwargs', [
    dict(calcium_rule=dict(decay=4.0), temporal_smoothness=1.0),
    dict(calcium_rule=dict(penalty=1.0)),
    dict(calcium_rule=dict(decay=0.0)),
    dict(calcium_rule=dict(decay=4.0, penalty=-1.0)),
    dict(calcium_rule=dict(penalty='noise')),
    dict(calcium_rule=dict(decay=4.0, penalty='auto')),
    dict(calcium_rule=dict(decay=4.0, penalty='noise', noise_limits=(3.0, 0.1))),
    dict(calcium_rule=dict(decay=4.0, penalty='noise', noise_limits=(-0.1, 3.0))),
    dict(calcium_rule=dict(decay=4.0, penaltty='noise')),
    dict(calcium_rule=4.0),
])
def test_bad_settings_fail_loudly(kwargs):
    """Settings that cannot work together must raise.

    Otherwise one setting is ignored without a word, and the run uses a
    penalty that nobody chose.
    """
    V, S, H, B = one_cell()
    with pytest.raises(ValueError):
        nmf_pytorch(V, S, H, B, num_iterations=1, **kwargs)


def calcium_events(n_t, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros(n_t)
    s[rng.choice(n_t - 30, n_t // 50, replace=False)] = rng.uniform(5.0, 30.0, n_t // 50)
    return kernel_matrix(n_t, GAMMA) @ s


@pytest.mark.parametrize('sd', [0.4, 2.0, 5.0])
def test_noise_is_measured_from_the_falls(sd):
    """The noise of a trace with events is measured close to the true noise.

    Events rise within one frame. A measure that counts fast changes counts
    these rises as noise. Then mu is too high, and the fit cuts real events.
    """
    rng = np.random.default_rng(4)
    Y = 40.0 + calcium_events(2000, seed=5)[None, :] + rng.normal(0.0, sd, (4, 2000))
    est = np.sqrt(trace_noise(torch.tensor(Y), GAMMA).numpy().ravel())
    np.testing.assert_allclose(est, sd, rtol=0.15)


def test_noise_rule_keeps_events_and_removes_noise():
    """With penalty 'noise', the trace is closer to the truth, and events keep their height.

    The rule must remove noise only. If it also lowers events, it repeats the
    cost of temporal_smoothness.
    """
    n_t = 600
    yy, xx = np.mgrid[0:12, 0:12]
    S = np.exp(-0.5 * ((yy - 6.0) ** 2 + (xx - 6.0) ** 2) / 3.0 ** 2).reshape(-1, 1).astype(np.float32)
    events = calcium_events(n_t, seed=6)
    rng = np.random.default_rng(7)
    V = (S @ (40.0 + events)[None, :] + rng.normal(0.0, 2.0, (S.shape[0], n_t))).astype(np.float32)
    H = np.full((1, n_t), 40.0, np.float32)
    kw = dict(num_iterations=400, objective_rtol=None, footprint_rtol=None, update_int=10000,
              H_lr=0.5, S_lr=0.2)

    def error_and_height(penalty):
        So, Ho, _ = nmf_pytorch(V, S, H, np.ones_like(S),
                                calcium_rule=dict(decay=4.0, penalty=penalty), **kw)
        light = Ho[0] * So[:, 0].max()
        err = np.sqrt(np.mean((light - light.mean() - (events - events.mean())) ** 2))
        return err, np.polyfit(events, light, 1)[0]

    err0, _ = error_and_height(0.0)
    err1, height = error_and_height('noise')
    assert err1 < 0.7 * err0
    assert height == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize('left_sd, bound', [(0.5, 3.0), (5.0, 0.1)])
def test_noise_feedback_stays_within_its_limits(left_sd, bound):
    """mu stays between 0.1 and 3 times the noise's sd, also when no mu meets the noise.

    Row 0 of the toy: the fit leaves less than the noise at every mu, as in a
    trace with no events. Without the upper limit mu doubles in every step,
    reaches inf, and the block turns NaN. Row 1: the fit leaves more than the
    noise at every mu, and without the lower limit mu falls to 0.
    """
    rng = np.random.default_rng(8)
    noise_var = torch.full((1, 1), 4.0, dtype=torch.float64)
    raw = torch.tensor(rng.normal(0.0, 1.0, (1, 500)))
    mu = torch.full((1, 1), 2.0, dtype=torch.float64)
    for _ in range(3000):
        mu = noise_feedback(mu, left_sd * raw, torch.zeros_like(raw), noise_var)
    assert float(mu) == pytest.approx(bound * 2.0)


@pytest.mark.parametrize('limits', [(0.1, 3.0), (0.1, 0.3), (1.0, 1.0)])
def test_nmf_keeps_mu_within_the_given_noise_limits(limits, monkeypatch):
    """nmf_pytorch gives noise_limits to the feedback, and mu stays inside them.

    If the limits did not reach the feedback, a run with other limits would
    give the same traces as the default, and a comparison of limits would
    compare nothing.
    """
    import segmentNMF.NMF_methods as nm
    seen = []

    def watched(mu, raw, H, noise_var, **kw):
        out = noise_feedback(mu, raw, H, noise_var, **kw)
        seen.append((out / torch.sqrt(noise_var)).flatten())
        return out

    monkeypatch.setattr(nm, 'noise_feedback', watched)
    V, S, H, B = one_cell()
    nm.nmf_pytorch(V, S, H, B, num_iterations=50, objective_rtol=None, footprint_rtol=None,
                   update_int=1000,
                   calcium_rule=dict(decay=4.0, penalty='noise', noise_limits=limits))
    ratio = torch.cat(seen)
    assert len(seen) > 0
    assert float(ratio.min()) >= limits[0] * (1 - 1e-5)
    assert float(ratio.max()) <= limits[1] * (1 + 1e-5)
