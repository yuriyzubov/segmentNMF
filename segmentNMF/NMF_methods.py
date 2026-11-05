"""functions implementing the NMF algorithm for demixing neural signals using segmentation data"""

import numba
import numpy as np
import torch
from tqdm import tqdm

# ======================================================================================================================
# Helper functions
# ======================================================================================================================

def pearsonr_mat(X, Y, axis=0):
    """Calculates the Pearson correlation coefficient between each row or columns of two matrices X and Y,
     both of which should have the exact same shape.

    Args:
        X: A 2D NumPy array.
        Y: A 2D NumPy array with the same shape as X
        axis: An integer indicating the axis along which to calculate the Pearson correlation coefficient.

    Returns:
        A NumPy array containing the Pearson correlation coefficient between each row or column of X and Y.
    """

    # Check if the dimensions of X and Y match.
    if X.shape != Y.shape:
        raise ValueError("The dimensions of X and Y must match.")

    # Check if X and Y are 2D arrays.
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")

    # Check if the axis parameter is valid.
    if axis not in range(X.ndim):
        raise ValueError("The axis parameter is out of bounds.")

    # Calculate the mean of each row or column.
    X_mean = np.nanmean(X, axis=axis)
    Y_mean = np.nanmean(Y, axis=axis)

    # Calculate the squared deviations from the mean.
    X_sq_dev = np.nansum((X - X_mean)**2, axis=axis)
    Y_sq_dev = np.nansum((Y - Y_mean)**2, axis=axis)

    # Calculate the covariance between X and Y.
    cov = np.sum((X - X_mean) * (Y - Y_mean), axis=axis)

    # Calculate the Pearson correlation coefficient.
    corr = cov / (np.sqrt(X_sq_dev * Y_sq_dev) + 1e-12)

    return corr


from scipy.stats import pearsonr

def compare_with_true(X, Y, axis=0):
    """Calculate Pearson correlation coefficient for each component

    Args:
        X: A 2D NumPy array.
        Y: A 2D NumPy array with the same shape as X
        axis: An integer indicating the axis along which to calculate the Pearson correlation coefficient.
              Note this axis parameters has the opposite definition from the axis parameter for the
              pearsonr_mat function

    Returns:
        A NumPy array containing the Pearson correlation coefficient between each row or column of X and Y.
    """
    correlations = []
    for i in range(X.shape[axis]):
        correlation, _ = pearsonr(X[i], Y[i])
        correlations.append(correlation)
    return np.array(correlations)


# ======================================================================================================================
# Step-size estimation
# ======================================================================================================================

def line_search_step_size(V, S, H, gradient, objective_function, gradient_str='S', lr=1.0, alpha=0.9999, beta=0.99):
    """Estimate the step size for a single parameter using a line search.

    Args:
        V: A PyTorch tensor representing the input data matrix.
        S: A PyTorch tensor representing the spatial components matrix.
        H: A PyTorch tensor representing the temporal components matrix.
        gradient: A PyTorch tensor representing the gradient of the objective function with respect to the parameter.
        objective_function: A function that takes the input data matrix, the spatial components matrix, and the temporal components matrix and returns the objective value.
        gradient_str: A string indicating whether gradient is for S or H. Only valid values are 'S' and 'H'
        lr: The initial learning rate.
        alpha: The strong Wolfe line search parameter.
        beta: The backtracking factor.

    Returns:
        A PyTorch tensor representing the step size for the parameter.
    """

    # Calculate the initial step size
    step_size = lr

    # Calculate the objective function at the current parameters
    objective = objective_function(V, S, H)

    grad_in = gradient.reshape(-1) @ gradient.reshape(-1)
    # print(grad_in)

    # Perform the line search
    while True:
        if gradient_str == 'S':
            S_step = S + step_size * gradient
            objective_step = objective_function(V, S_step, H)
        elif gradient_str == 'H':
            H_step = H + step_size * gradient
            objective_step = objective_function(V, S, H_step)

        # # Calculate the strong Wolfe condition
        # descent_condition = objective - objective_step <= alpha * step_size * grad_in

        # If the strong Wolfe condition is satisfied, then accept the update
        if objective_step < objective*alpha:
            break

        # Otherwise, reduce the step size and try again
        else:
            step_size *= beta
            print(gradient_str + ': new line search step size {}'.format(step_size))

    print(gradient_str + ': final step size {}'.format(step_size))
    return step_size


# ======================================================================================================================
# Numpy-based implementation
# ======================================================================================================================

def frobenius_norm(V, S, H):
    """Calculate the Frobenius norm objective.

    Args:
        V: A NumPy array representing the input data matrix.
        S: A NumPy array representing the spatial components matrix.
        H: A NumPy array representing the temporal components matrix.

    Returns:
        The Frobenius norm of the residual V - S @ H; scalar value.
    """

    # Calculate the Frobenius norm objective
    residual = V - S @ H
    objective = np.linalg.norm(residual, 'fro')
    return objective


def objective_settled(objectives, i, rtol, window, min_iterations):
    """True once the objective has stopped falling by a meaningful fraction.

    Replaces an older rule that had two faults. That rule compared an ABSOLUTE
    difference between two iterations, so one setting could not serve blocks of
    different size: the objective is a Frobenius norm, which grows with the
    voxel count. And it stopped on a single uphill step, which gradient descent
    produces routinely while it is still making progress.

    Here the mean of the last `window` objectives is compared against the mean
    of the `window` before it, and the drop is asked for as a FRACTION. The mean
    absorbs step to step noise. The fraction removes the block size, so the same
    rtol means the same thing on a 128^3 core and on a single cell block.

    A window that rises gives a negative drop, which is below any positive rtol.
    A real increase therefore still stops the fit, but a single step cannot.

    Tested only every `window` iterations, so the two windows never overlap.
    """
    n = i + 1
    if rtol is None or n < max(min_iterations, 2 * window) or n % window != 0:
        return False
    prev = float(np.mean(objectives[n - 2 * window:n - window]))
    curr = float(np.mean(objectives[n - window:n]))
    if prev <= 0:
        return False
    drop = (prev - curr) / prev
    if drop < rtol:
        print('Objective settled: relative drop {:.2e} over {} iterations '
              '< {:.2e}'.format(drop, window, rtol))
        return True
    return False


def footprint_settled(nnz_history, rtol):
    """True once the number of surviving spatial voxels has stopped changing.

    The reconstruction error ||V - S @ H|| is not enough on its own. It can go
    flat while voxels still drop out of the footprints one by one, because
    each lost voxel changes the error very little. A run stopped on the error
    alone can leave a footprint that is still changing.

    `nnz_history` must be sampled on a fixed interval, so that two neighbouring
    entries are always the same number of iterations apart.
    """
    if len(nnz_history) < 2:
        return False
    prev, curr = nnz_history[-2], nnz_history[-1]
    if curr <= 0:
        return False
    change = abs(prev - curr) / curr
    if change < rtol:
        print('Footprint settled: {} voxels, {:.2e} relative change'.format(
            curr, change))
        return True
    return False


def check_objective_threshold(objective_threshold):
    """Raise if a caller still sets the removed objective_threshold.

    It compared an absolute objective difference between two neighbouring
    iterations, which cannot serve blocks of different size. A silent ignore
    would let an old config run with a stopping rule it did not ask for.
    """
    if objective_threshold is not None:
        raise ValueError(
            'objective_threshold was removed. Use objective_rtol: the fit '
            'stops when the error falls by less than this fraction over '
            'objective_window steps.')


def nmf(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr=1,
                S_lr=1e-6, objective_threshold=None, estimate_noise_component=True,
                min_iterations: int = 50, objective_rtol: float = 1e-4,
                objective_window: int = 50, footprint_rtol: float = 1e-3,
                footprint_window: int = 100):
    """Compute NMF using gradient descent

    Args:
        V: A NumPy array representing the input data matrix.
        S_init: A NumPy array representing the initial spatial components matrix.
        H_init: A NumPy array representing the initial temporal components matrix.
        B: A NumPy array representing the neighborhood matrix.
        H_true: A NumPy array representing the true temporal components matrix (optional).
        num_iterations: The number of iterations to run the algorithm for (default: 100).
        update_int: The interval at which to display progress updates (default: 10).
        H_lr: The learning rate for the H matrix (default: 1).
        S_lr: The learning rate for the S matrix (default: 1e-6).
        objective_threshold: Removed. Setting it raises an error. Use objective_rtol.
        estimate_noise_component: Whether to estimate a noise component (default: True).
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).
        objective_rtol: Stop once the objective's mean falls by less than this FRACTION over
            objective_window iterations, against the window before it. Scale free, so one
            value fits every block size. None disables it (default: 1e-4).
        objective_window: How many iterations each of the two compared means covers
            (default: 50). The test runs every objective_window iterations.
        footprint_rtol: Stop once the count of surviving spatial voxels changes by less
            than this FRACTION over footprint_window iterations. Needed because the
            objective can settle before the footprint does. None disables it
            (default: 1e-3).
        footprint_window: How many iterations between two voxel counts (default: 100).

        Early stopping needs EVERY enabled test to pass. Setting one of the two rtols
        to None removes that test as a requirement, it does not switch early stopping
        off. To run all num_iterations, set objective_rtol AND footprint_rtol to None.

    Returns:
        A tuple of NumPy arrays representing the learned spatial components matrix (S), temporal components matrix (H), and objective values (objectives).
    """

    check_objective_threshold(objective_threshold)
    n_components = H_init.shape[0]

    # If specified, add an additional 'noise' component
    if estimate_noise_component:
        noise_component = np.ones_like(S_init[:, 0]) / S_init.shape[0]
        noise_time_component = np.zeros_like(H_init[0, :])
        neighborhood_component = np.ones_like(B[:, 0])
        S_init = np.concatenate((S_init, noise_component[:, None]), axis=1)
        H_init = np.concatenate((H_init, noise_time_component[None, :]), axis=0)
        B = np.concatenate((B, neighborhood_component[:, None]), axis=1)

    objectives = []
    correlations = []
    H_gradients = []
    S_gradients = []
    # sampled every footprint_window iterations, so two neighbouring entries are
    # always the same distance apart
    nnz_history = []
    objective_done = False

    S = S_init
    H = H_init

    print('Run gradient NMF | max iterations: {}'.format(num_iterations))
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):

        # Update H matrix with dynamically estimated step size
        H_gradient = S.T @ (V - S @ H)
        norm = np.sum(S, axis=0)[:, None]
        norm[norm == 0] = 1
        H_step_size = H_lr / norm
        H_gradient *= H_step_size
        H += H_gradient
        H = np.maximum(0, H)

        # Update S matrix with spatial constraint
        S_gradient = (V - S[:, :n_components] @ H[:n_components]) @ H[:n_components].T
        norm = np.sum(H[:n_components], axis=1)[None, :]
        norm[norm == 0] = 1
        S_step_size = S_lr / norm
        # B gains a column when the noise component is on, and the gradient
        # does not. The step is only ever taken on the fitted components.
        S_gradient *= S_step_size * B[:, :n_components]
        S[:, :n_components] += S_gradient
        S = np.maximum(0, S)
        S = np.minimum(1, S)  # we want amplitude in temporal not spatial components

        # Save gradient steps
        S_gradients.append(np.mean(S_gradient))
        H_gradients.append(np.mean(H_gradient))

        # Calculate Frobenius norm objective
        objective = frobenius_norm(V, S, H)
        objectives.append(objective)

        # Display progress update
        if i % update_int == 0:
            progress_str = 'Iteration {} | Objective: {:.6f} | avg H step {:.10f} | avg S step {:.10f}'.format(
                i, objectives[i], H_gradients[i], S_gradients[i])
            if H_true is not None:
                correlations.append(pearsonr_mat(H.roll(-1, axis=0)[:H_true.shape[0]], H_true, axis=0))
                progress_str += ' | avg corr {:.5f} | min corr {:.5f}'.format(np.mean(correlations[-1]),
                                                                              np.min(correlations[-1]))
            tqdm.write(progress_str)

        # Check stopping criterion.
        #
        # Both tests must pass, because they can settle at different times.
        # See footprint_settled.
        if footprint_window and (i + 1) % footprint_window == 0:
            nnz_history.append(int(np.count_nonzero(S[:, :n_components])))
        if objective_rtol is None and footprint_rtol is None:
            continue
        if objective_rtol is not None:
            objective_done = objective_done or objective_settled(
                objectives, i, objective_rtol, objective_window, min_iterations)
            if not objective_done:
                continue
        if footprint_rtol is not None and not footprint_settled(
                nnz_history, footprint_rtol):
            continue
        break

    if H_true is not None:
        return S, H, objectives, correlations
    else:
        return S, H, objectives

# ======================================================================================================================
# Pytorch-based implementation
# ======================================================================================================================

# THE TWO STEP SIZE DENOMINATORS
#
# A step size answers one question: how far must this component move to remove
# the whole error? sum(S[:, c] ** 2) is the answer for a component that sits
# alone, so a learning rate of 1 then moves it exactly onto the best value.
#
# Components that share voxels move at the same time. Each one still moves as
# if it were alone, so the two moves add up and go too far. Use a rate below 1
# when footprints overlap.
#
# 'sum' divides by the plain sum. The step is then the mean residual over the
# footprint, weighted by S, and it stays in data units. It is larger than the
# least squares step, so each choice needs its own H_lr and S_lr.
STEP_SIZE_NORMS = ('sum_of_squares', 'sum')

# Smallest value allowed as a step-size denominator. An exact zero test, such
# as `norm[norm == 0] = 1`, does not catch a subnormal sum. A component whose
# trace drains toward zero passes through that range. Torch flushes a subnormal
# to zero inside the division and returns infinity. The infinity meets a zero
# gradient entry, and 0 * inf is NaN, which spreads to every value in the
# block. 1e-6 sits far above the subnormal range and far below a live component.
NORM_FLOOR = 1e-6


def parse_spatial_smoothness(value):
    """Read spatial_smoothness as (step, (wz, wy, wx)).

    A sequence [z, y, x] is the pull strength along each axis. The step is
    their sum. Only the ratios matter as weights, so they are divided by the
    largest one here.

    A scalar L is the same as [L, L, L]: step 3 * L, and equal weights. To keep
    the smoothing lateral, put 0 in the z slot.

    None, 0, and a vector of zeros all mean "no smoothing" and give step 0.

    >>> parse_spatial_smoothness([0, 0.01, 0.01])   # lateral only
    (0.02, (0.0, 1.0, 1.0))
    >>> parse_spatial_smoothness(0.01)              # all 3 axes
    (0.03, (1.0, 1.0, 1.0))
    >>> parse_spatial_smoothness(None)              # off
    (0.0, (1.0, 1.0, 1.0))
    """
    if value is None:
        return 0.0, (1.0, 1.0, 1.0)
    if np.isscalar(value):
        value = (value, value, value)
    w = tuple(float(v) for v in value)
    if len(w) != 3:
        raise ValueError('spatial_smoothness must be a scalar or 3 values '
                         '[z, y, x], got {}'.format(value))
    if any(v < 0.0 for v in w):
        raise ValueError('spatial_smoothness values must not be negative, '
                         'got {}'.format(value))
    top = max(w)
    if top == 0.0:
        return 0.0, (1.0, 1.0, 1.0)
    return sum(w), tuple(v / top for v in w)


def _add_neighbours(dst, src, axis, weight):
    """Add `weight` times each neighbour along `axis` of `src` into `dst`.

    Both arrays are (n_components, nz, ny, nx), so `axis` is 1, 2 or 3. The
    two narrow views are one position apart, and that offset is the shift. A
    face has no neighbour outside the array, so nothing is added there.
    """
    if not weight:
        return
    n = src.shape[axis]
    torch.narrow(dst, axis, 1, n - 1).add_(
        weight * torch.narrow(src, axis, 0, n - 1))
    torch.narrow(dst, axis, 0, n - 1).add_(
        weight * torch.narrow(src, axis, 1, n - 1))


def lateral_neighbour_counts(spatial_shape, weights=(0.0, 1.0, 1.0),
                             device=None, dtype=None):
    """The divisor for lateral_neighbour_mean, one value per position.

    `weights` is (wz, wy, wx). Each present neighbour along an axis adds that
    axis weight. A missing neighbour at a face adds nothing, so its weight
    goes to the neighbours that are present. Never returns 0, so it is safe
    to divide by.

    The result is (1, 1, ny, nx) when wz is 0, and (1, nz, ny, nx) otherwise.
    Both broadcast against (n_components, nz, ny, nx).

    Computed once, outside the iteration loop.
    """
    nz, ny, nx = spatial_shape[0], spatial_shape[1], spatial_shape[2]
    o = torch.ones((1, nz if weights[0] else 1, ny, nx),
                   device=device, dtype=dtype)
    c = torch.zeros_like(o)
    for axis, weight in enumerate(weights, start=1):
        _add_neighbours(c, o, axis, weight)
    return torch.clamp(c, min=1e-12)


def lateral_neighbour_mean(x, counts, weights=(0.0, 1.0, 1.0)):
    """The weighted mean of a voxel's neighbours, excluding the voxel itself.

    `x` is (n_components, nz, ny, nx). `counts` must come from
    lateral_neighbour_counts for the same shape AND the same `weights`.

    `weights` is (wz, wy, wx). A z step is often much longer than a lateral
    step, and wz must then be smaller than wy and wx. At wz 0 nothing links
    the planes, and each plane is fitted on its own.
    """
    s = torch.zeros_like(x)
    for axis, weight in enumerate(weights, start=1):
        _add_neighbours(s, x, axis, weight)
    return s / counts


def frobenius_norm_pytorch(V, S, H):
    """Calculate the Frobenius norm objective.

    Args:
        V: A PyTorch tensor representing the input data matrix.
        S: A PyTorch tensor representing the spatial components matrix.
        H: A PyTorch tensor representing the temporal components matrix.

    Returns:
        The Frobenius norm of the residual V - S @ H; scalar value.
    """

    # Calculate the Frobenius norm objective
    residual = V - S @ H
    objective = torch.norm(residual, 'fro').item()
    return objective


def smooth_in_time(H, a, freq):
    """(I + a L)^-1 H row by row: L the 1D Laplacian along time with mirror
    boundaries, a of shape (n_rows, 1), freq L's eigenvalues for the rfft of the
    mirrored rows. The proximal step of (a / 2) * sum_t (H[t+1] - H[t])^2."""
    n_t = H.shape[1]
    ext = torch.cat([H, torch.flip(H, dims=[1])], dim=1)
    F = torch.fft.rfft(ext, dim=1) / (1 + a * freq)
    return torch.fft.irfft(F, n=2 * n_t, dim=1)[:, :n_t]


@numba.njit(cache=False)
def _ar1_project_row(y, gamma, out):
    """out = argmin |out - y|^2 over traces with out[0] >= 0 and
    out[t] >= gamma * out[t - 1]. The pool algorithm of OASIS (Friedrich et al.
    2017): a pool is a run of frames that decays at gamma from its first value."""
    n_t = y.shape[0]
    v = np.empty(n_t)                   # pool value at its first frame
    w = np.empty(n_t)                   # pool weight, sum of gamma^(2k)
    first = np.empty(n_t, np.int64)
    length = np.empty(n_t, np.int64)
    p = -1
    pinned = False                      # pool 0 held at 0 by out[0] >= 0
    for t in range(n_t):
        p += 1
        v[p] = y[t]
        w[p] = 1.0
        first[p] = t
        length[p] = 1
        while True:
            if p == 0:
                if v[0] < 0.0:
                    pinned = True
                    v[0] = 0.0
                break
            g = gamma ** length[p - 1]
            if v[p] >= g * v[p - 1]:
                break
            # pool p falls too fast after pool p - 1: merge pool p into pool p - 1
            if not (p == 1 and pinned):
                g2 = g * g
                v[p - 1] = (w[p - 1] * v[p - 1] + g * w[p] * v[p]) / (w[p - 1] + g2 * w[p])
                w[p - 1] += g2 * w[p]
            length[p - 1] += length[p]
            p -= 1
    for q in range(p + 1):
        g = 1.0
        for k in range(length[q]):
            out[first[q] + k] = v[q] * g
            g *= gamma


@numba.njit(parallel=True, cache=False)
def _ar1_project(Y, gamma):
    """_ar1_project_row for every row of Y, in parallel."""
    out = np.empty_like(Y)
    for i in numba.prange(Y.shape[0]):
        _ar1_project_row(Y[i], gamma, out[i])
    return out


def calcium_step(H, a, gamma, baseline):
    """The proximal step of a * sum_t s[t], row by row, with H = baseline + c,
    c[t] = gamma * c[t-1] + s[t] and s >= 0. a and baseline have shape (n_rows, 1).

    One round of block coordinate descent: the exact c for the given baseline,
    then the baseline for that c. Returns (H, baseline). Run every iteration with
    the baseline it returned, the pair converges to the joint solution.

    sum_t s[t] = (1 - gamma) * sum_t c[t] + gamma * c[-1], so the penalty is
    linear in c, and the step is a projection of H - baseline - a * weights.
    The last frame has no frame after it, so its weight is 1, not 1 - gamma."""
    n_t = H.shape[1]
    weights = torch.full((1, n_t), 1.0 - gamma, device=H.device, dtype=H.dtype)
    weights[0, -1] = 1.0
    target = (H - baseline - a * weights).cpu().numpy().astype(np.float64)
    c = torch.from_numpy(_ar1_project(target, gamma)).to(device=H.device, dtype=H.dtype)
    baseline = torch.clamp(torch.mean(H - c, dim=1, keepdim=True), min=0.0)
    return baseline + c, baseline


def trace_noise(Y, gamma):
    """Noise variance of each row of Y, shape (n_rows, 1), measured from the falls only.

    Under the calcium model, Y[t] - gamma * Y[t-1] is (1 - gamma) * baseline + s[t]
    + n[t] - gamma * n[t-1], with s >= 0 and n the noise. Events only add to it, so
    the values below its median hold noise only. Their spread (median distance
    below the median / 0.6745, as for a normal distribution) is the noise of
    n[t] - gamma * n[t-1], which has variance (1 + gamma^2) times the noise.

    WHY NOT THE POWER SPECTRUM. CNMF measures the noise from the power at periods
    of 2 to 4 frames. That assumes events change slowly from frame to frame. When
    an event rises within one frame, its rise has power there too, and the noise
    is measured too high in active cells."""
    e = Y[:, 1:] - gamma * Y[:, :-1]
    m = torch.median(e, dim=1, keepdim=True).values
    below = torch.where(e < m, m - e, torch.full_like(e, float('nan')))
    spread = torch.nanmedian(below, dim=1, keepdim=True).values / 0.6745
    return torch.nan_to_num(spread) ** 2 / (1.0 + gamma ** 2)


def noise_feedback(mu, raw, H, noise_var, rate=0.5, limits=(0.1, 3.0)):
    """One step of the feedback for calcium_rule penalty 'noise', row by row.

    mu goes up where the fit leaves less than the noise between the raw trace and
    H, and down where it leaves more. A step changes mu by a factor of 0.5 to 2.
    The fixed point leaves exactly the noise. mu then stays between limits[0] and
    limits[1] times the noise's standard deviation.

    WHY THE LIMITS. The fixed point does not always exist. In a trace with no
    events, a large mu already removes every rise, and a larger mu changes nothing.
    A noise measure a little above what the fit leaves then raises mu by the same
    factor in every iteration, until mu is inf and the trace and every trace that
    shares voxels with it are NaN. In a trace with large events the fit can leave
    more than the noise at any mu, and mu falls to 0. The limits stop both."""
    left = torch.mean((raw - H) ** 2, dim=1, keepdim=True)
    factor = (noise_var / torch.clamp(left, min=1e-30)) ** rate
    sd = torch.sqrt(noise_var)
    return torch.clamp(mu * torch.clamp(factor, 0.5, 2.0), limits[0] * sd, limits[1] * sd)


CALCIUM_RULE_KEYS = ('decay', 'penalty', 'noise_limits')


def parse_calcium_rule(rule):
    """Read calcium_rule as (decay, penalty, noise_limits).

    None is off: (None, 0.0, (0.1, 3.0)). A dict must hold 'decay', and may hold
    'penalty' and 'noise_limits'. Any other key raises, so a typo cannot switch a
    setting off in silence.
    """
    if rule is None:
        return None, 0.0, (0.1, 3.0)
    if not isinstance(rule, dict):
        raise ValueError('calcium_rule must be a dict or None, got {!r}'.format(rule))
    unknown = sorted(set(rule) - set(CALCIUM_RULE_KEYS))
    if unknown:
        raise ValueError('calcium_rule has unknown keys {}; known keys are {}'
                         .format(unknown, CALCIUM_RULE_KEYS))
    if rule.get('decay') is None:
        raise ValueError("calcium_rule needs 'decay'")
    return (rule['decay'], rule.get('penalty', 0.0),
            rule.get('noise_limits', (0.1, 3.0)))


def nmf_pytorch(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr: float = 1e-1,
                S_lr: float = 1e-1, step_size_norm: str = 'sum_of_squares',
                objective_threshold: float = None, min_iterations: int = 50,
                objective_rtol: float = 1e-4, objective_window: int = 50,
                footprint_rtol: float = 1e-3, footprint_window: int = 100,
                spatial_smoothness=None, spatial_shape=None, S_max=None,
                S_max_core=None, temporal_smoothness=None, calcium_rule=None):

    """Compute NMF using gradient descent

    Args:
        V: A NumPy array or PyTorch tensor representing the input data matrix.
        S_init: A NumPy array or PyTorch tensor representing the initial spatial components matrix.
        H_init: A NumPy array or PyTorch tensor representing the initial temporal components matrix.
        B: A NumPy array or PyTorch tensor representing the neighborhood matrix.
        H_true: A PyTorch tensor representing the true temporal components matrix (optional).
        num_iterations: The number of iterations to run the algorithm for (default: 100).
        update_int: The interval at which to display progress updates (default: 10).
        H_lr: The learning rate for the H matrix (default: 1).
        S_lr: The learning rate for the S matrix (default: 1e-6).
        step_size_norm: 'sum_of_squares' or 'sum', the denominator of both step
            sizes (default: 'sum_of_squares'). It decides how the voxels of one
            footprint combine into one step. 'sum' takes the mean residual over
            the footprint, weighted by S, and stays in data units.
            'sum_of_squares' takes the least squares step, which is exact for a
            component that sits alone. Each choice needs its own H_lr and S_lr.
        objective_threshold: Removed. Setting it raises an error. Use objective_rtol.
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).
        objective_rtol: Stop once the objective's mean falls by less than this FRACTION over
            objective_window iterations, against the window before it. Scale free, so one
            value fits every block size. None disables it (default: 1e-4).
        objective_window: How many iterations each of the two compared means covers
            (default: 50). The test runs every objective_window iterations.
        footprint_rtol: Stop once the count of surviving spatial voxels changes by less
            than this FRACTION over footprint_window iterations. Needed because the
            objective can settle before the footprint does. None disables it
            (default: 1e-3).
        footprint_window: How many iterations between two voxel counts (default: 100).

        Early stopping needs EVERY enabled test to pass. Setting one of the two rtols
        to None removes that test as a requirement, it does not switch early stopping
        off. To run all num_iterations, set objective_rtol AND footprint_rtol to None.
        spatial_smoothness: A scalar, or 3 values [z, y, x]. Move each spatial weight
            toward the mean of its neighbours, once per iteration. 0 or None disables
            it (default: None). Requires spatial_shape.

            A vector gives the pull strength along each axis, and the step is their
            sum. A scalar L is the same as [L, L, L]. To keep the smoothing lateral,
            put 0 in the z slot. Give z a small weight when one z step is much larger
            than one lateral step, or the smoothing joins cells in different planes.

            WHY. Nothing else in the objective couples a voxel to the voxel beside it,
            so the fit is free to delete single voxels out of a smooth patch. That
            cuts one footprint into many small pieces.

            This term is the force that the gradient of a roughness penalty gives, sum
            over neighbour pairs of (S[a] - S[b])**2, but taken with a step of its own
            instead of the data step. That keeps the value readable: 0.1 moves a
            weight 10 percent of the way to the local mean each iteration. It is
            scaled by B, so it cannot push a footprint outward where B is small.

            Too large a value merges neighbouring cells. Watch the fraction of voxels
            with 2 or more cells, and the trace correlation between neighbour cells.
            Neither may rise.
        spatial_shape: The (nz, ny, nx) shape of the spatial canvas, needed to find a
            voxel's neighbours. Its product must equal S_init.shape[0]. Only used when
            spatial_smoothness is on (default: None).
        S_max, S_max_core: A per-voxel upper limit on each footprint, relative to its
            own core, from neighborhood.max_light_per_voxel. Both shaped like S_init.
            Every iteration, S[v, i] <= S_max[v, i] * m_i, where m_i is S[:, i]'s peak
            over the voxels where S_max_core[:, i] is True. In the core region S_max
            is 1, so the limit only binds outside it, where it is the blur tail. None
            disables it (default: None).

            WHY. Without it nothing stops a component from becoming bright on a
            neighbour's core: a small or dim cell can lower the error by taking a
            bright neighbour's light. Relative to m_i, because a footprint's scale
            is free (the trace absorbs it), so a fixed limit could be dodged by
            dimming the whole footprint.
        temporal_smoothness: lam >= 0. Adds (lam / 2) * |S_i|^2 * sum_t (H[i, t+1] - H[i, t])^2
            for every component to the objective. 0 or None disables it (default: None).

            Weighted by |S_i|^2, the component's own data weight, so lam is unitless:
            for a component that sits alone the result is its least-squares trace
            filtered by 1 / (1 + lam * (2 - 2 cos w)), w the frequency in radians per
            frame. lam = 1 cuts frame-to-frame (Nyquist) noise to 1/5 in amplitude and
            a 6-frame period to 1/2.

            WHY. Where two footprints overlap, the data pin the sum of the two traces
            but only weakly their difference, so each frame's noise goes into the
            difference: one trace up, the other down (zero-lag anticorrelation of close
            pairs, near minus the footprint overlap). The difference has less data
            weight, |S|^2 (1 - overlap), against the same penalty, so it is smoothed
            more than the sum: about (1 + overlap) / (1 - overlap) times, 3x at 0.5.
            Smoothing the traces afterwards cannot do this, it treats both alike.

            Taken as a proximal step after each H gradient step (an FFT along time,
            mirror boundaries), so it is stable at any lam and does not slow the
            data step. Its cost: real changes faster than about one frame are
            smoothed too. Check the recovery of a known signal.
        calcium_rule: dict, or None (default, off). The calcium shape rule, with these keys:

            decay: tau > 0 in frames, required. Each trace becomes baseline + c, with
                c[t] = gamma * c[t-1] + s[t], s >= 0 and gamma = exp(-1 / tau). One tau
                holds for every component. A trace can rise by any amount in one frame.
                It cannot fall faster than gamma per frame. The baseline is one value
                per component, >= 0, with no penalty.
            penalty: mu >= 0, or 'noise' (default: 0). Adds mu * |S_i|^2 * sum_t s[t]
                for every component to the objective. mu has the units of H. An
                isolated event loses about mu * (1 - gamma^2) of its height. With mu = 0
                only the shape rule acts, and no event shrinks.

                'noise' sets mu for every component in every iteration, as CNMF does
                (Pnevmatikakis et al. 2016, Neuron): the fit may remove as much from a
                component's raw trace as the noise of that trace, and no more. The raw
                trace is the least-squares trace, H_i + S_i . R / |S_i|^2. Its noise is
                measured from its falls, see trace_noise. mu follows noise_feedback,
                from a start of the noise's standard deviation.
            noise_limits: (low, high), 0 <= low <= high (default: (0.1, 3.0)). With
                penalty 'noise', each component's mu stays between low and high times
                the standard deviation of its noise. Without the high limit, mu grows
                to inf on a trace where the fit leaves less than the noise at every
                mu. A lower high limit keeps more of the small events.

            Cannot be combined with temporal_smoothness.

            WHY. temporal_smoothness stops the trade of noise between overlapping
            footprints, but it also lowers and widens short events. A calcium trace
            rises fast and decays slowly. A frame of noise traded between two
            traces puts a fast fall in both: one trace falls, the other rises and
            falls back. The shape rule blocks these falls, and it keeps the rise.

            Taken as a proximal step after each H gradient step, see calcium_step.
            Its cost: a cell that decays faster than tau is slowed to tau, so do not
            choose tau longer than the indicator's decay. Noise can still rise.

    Returns:
        A tuple of PyTorch tensors representing the learned spatial components matrix (S), temporal components matrix (H), and objective values (objectives).
    """

    check_objective_threshold(objective_threshold)
    if step_size_norm not in STEP_SIZE_NORMS:
        raise ValueError('step_size_norm must be one of {}, got {}'
                         .format(STEP_SIZE_NORMS, step_size_norm))
    square_norm = step_size_norm == 'sum_of_squares'

    # float32: inputs are uint16 data, and most GPUs run FP64 far slower
    V = torch.tensor(V, dtype=torch.float32)
    S_init = torch.tensor(S_init, dtype=torch.float32)
    H_init = torch.tensor(H_init, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))

    # Move tensors to the GPU if available
    V = V.to(device)
    S = S_init.to(device)
    H = H_init.to(device)
    B = B.to(device)
    zero = torch.zeros((), device=device, dtype=S.dtype)
    if (S_max is None) != (S_max_core is None):
        raise ValueError('S_max and S_max_core go together')
    if S_max is not None:
        S_max = torch.as_tensor(S_max, dtype=torch.float32).to(device)
        S_max_core = torch.as_tensor(S_max_core, dtype=torch.bool).to(device)
        if S_max.shape != S.shape or S_max_core.shape != S.shape:
            raise ValueError('S_max and S_max_core must have the shape of S_init, '
                             '{}'.format(tuple(S.shape)))

    t_smooth = float(temporal_smoothness or 0.0)
    if t_smooth < 0:
        raise ValueError('temporal_smoothness must be >= 0, got {}'.format(temporal_smoothness))
    calcium_decay, calcium_penalty, noise_limits = parse_calcium_rule(calcium_rule)
    ca_noise = isinstance(calcium_penalty, str)
    if ca_noise:
        if calcium_penalty != 'noise':
            raise ValueError("calcium_rule penalty must be a number >= 0 or 'noise', got {}"
                             .format(calcium_penalty))
        ca_penalty = None           # set from the noise at the first iteration
    else:
        ca_penalty = float(calcium_penalty or 0.0)
        if ca_penalty < 0:
            raise ValueError('calcium_rule penalty must be >= 0, got {}'.format(calcium_penalty))
    noise_limits = tuple(float(v) for v in noise_limits)
    if len(noise_limits) != 2 or not 0.0 <= noise_limits[0] <= noise_limits[1]:
        raise ValueError('calcium_rule noise_limits must be (low, high) with 0 <= low <= high, got {}'
                         .format(noise_limits))
    if calcium_decay is not None:
        if not calcium_decay > 0:
            raise ValueError('calcium_rule decay must be > 0, got {}'.format(calcium_decay))
        if t_smooth:
            raise ValueError('use temporal_smoothness or calcium_rule, not both')
        ca_gamma = float(np.exp(-1.0 / float(calcium_decay)))
        ca_baseline = torch.clamp(torch.amin(H, dim=1, keepdim=True), min=0.0)
    if t_smooth:
        # eigenvalues of the 1D Laplacian with mirror boundaries (rfft of the
        # mirrored trace, length 2T)
        n_t = H.shape[1]
        k = torch.arange(n_t + 1, device=device, dtype=H.dtype)
        t_freq = (2 - 2 * torch.cos(np.pi * k / n_t))[None, :]

    objectives = []
    correlations = []
    H_gradients = []
    S_gradients = []
    # sampled every footprint_window iterations, so two neighbouring entries are
    # always the same distance apart
    nnz_history = []
    objective_done = False

    # smoothing, prepared once
    counts = None
    smooth_step, smooth_w = parse_spatial_smoothness(spatial_smoothness)
    if smooth_step:
        if spatial_shape is None:
            raise ValueError('spatial_smoothness requires spatial_shape')
        spatial_shape = tuple(int(v) for v in spatial_shape)
        if len(spatial_shape) != 3:
            raise ValueError('spatial_shape must be (nz, ny, nx), got '
                             '{}'.format(spatial_shape))
        n_spatial = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        if n_spatial != S.shape[0]:
            raise ValueError('spatial_shape {} holds {} voxels, but S has {}'
                             .format(spatial_shape, n_spatial, S.shape[0]))
        counts = lateral_neighbour_counts(spatial_shape, weights=smooth_w,
                                          device=device, dtype=S.dtype)

    # messing around with line search, keeping the value of the last iteration as the starting point for the next
    # With the idea that, during training, the learning rate should keep decreasing
    S_step_size = S_lr
    H_step_size = H_lr

    print('Run gradient NMF | max iterations: {}'.format(num_iterations))
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):

        # Update H matrix with dynamically estimated step size
        H_gradient = S.T @ (V - S @ H)
        if ca_noise:
            raw = H + H_gradient / torch.clamp(torch.sum(S * S, dim=0)[:, None], min=NORM_FLOOR)
            noise_var = trace_noise(raw, ca_gamma)
            if ca_penalty is None:
                ca_penalty = torch.sqrt(noise_var)
        norm = torch.sum(S * S if square_norm else S, dim=0)[:, None]
        norm = torch.clamp(norm, min=NORM_FLOOR)
        H_step_size = H_lr / norm
        H_gradient *= H_step_size
        H += H_gradient
        # proximal step of the temporal penalty, same step size as the data step
        if t_smooth:
            H = smooth_in_time(H, H_step_size * t_smooth * torch.sum(S * S, dim=0)[:, None],
                               t_freq)
        if calcium_decay is not None:
            H, ca_baseline = calcium_step(
                H, H_step_size * ca_penalty * torch.sum(S * S, dim=0)[:, None],
                ca_gamma, ca_baseline)
            if ca_noise:
                ca_penalty = noise_feedback(ca_penalty, raw, H, noise_var, limits=noise_limits)
        H = torch.maximum(H, torch.Tensor([0]).to(H.device))

        # Update S matrix with spatial constraint
        S_gradient = (V - S @ H) @ H.T
        norm = torch.sum(H * H if square_norm else H, dim=1)[None, :]
        norm = torch.clamp(norm, min=NORM_FLOOR)
        S_step_size = S_lr / norm
        S_gradient *= S_step_size * B
        S += S_gradient
        S = torch.maximum(S, torch.Tensor([0]).to(S.device))
        S = torch.minimum(S, torch.Tensor([1]).to(S.device))  # we want amplitude in temporal not spatial components

        # Pull each weight toward the mean of its neighbours, so an isolated
        # speck is pushed down and a voxel inside a patch is held up.
        if smooth_step:
            n_comp = S.shape[1]
            avg = lateral_neighbour_mean(
                S.T.reshape((n_comp,) + spatial_shape), counts,
                weights=smooth_w)
            avg = avg.reshape(n_comp, -1).T
            S = S + smooth_step * B * (avg - S)
            S = torch.clamp(S, 0.0, 1.0)

        # per-voxel limit, relative to the component's peak in its own core region
        if S_max is not None:
            m = torch.where(S_max_core, S, zero).amax(dim=0, keepdim=True)
            S = torch.minimum(S, S_max * m)

        # Save gradient steps
        S_gradients.append(torch.mean(S_gradient).item())
        H_gradients.append(torch.mean(H_gradient).item())

        # Calculate Frobenius norm objective
        objective = frobenius_norm_pytorch(V, S, H)
        objectives.append(objective)

        # Display progress update
        if i % update_int == 0:
            progress_str = 'Iteration {} | Objective: {:.6f} | avg H step {:.10f} | avg S step {:.10f}'.format(
                i, objectives[i], H_gradients[i], S_gradients[i])
            if H_true is not None:
                # correlations.append(pearsonr_mat(np.roll(H[:H_true.shape[0]].cpu().numpy(), 0, axis=0), H_true, axis=0))
                correlations.append(compare_with_true(H.cpu().numpy(), H_true))
                progress_str += ' | avg corr {:.5f} | min corr {:.5f}'.format(np.nanmean(correlations[-1]),
                                                                              np.nanmin(correlations[-1]))
            tqdm.write(progress_str)

        # Check stopping criterion.
        #
        # Both tests must pass, because they can settle at different times.
        # See footprint_settled.
        if footprint_window and (i + 1) % footprint_window == 0:
            nnz_history.append(int(torch.count_nonzero(S).item()))
        if objective_rtol is None and footprint_rtol is None:
            continue
        if objective_rtol is not None:
            objective_done = objective_done or objective_settled(
                objectives, i, objective_rtol, objective_window, min_iterations)
            if not objective_done:
                continue
        if footprint_rtol is not None and not footprint_settled(
                nnz_history, footprint_rtol):
            continue
        break

    # Move tensors back to the CPU
    S = S.cpu().numpy()
    H = H.cpu().numpy()

    if H_true is not None:
        return S, H, objectives, correlations
    else:
        return S, H, objectives
