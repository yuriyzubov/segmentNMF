"""functions implementing the NMF algorithm for demixing neural signals using segmentation data"""

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

    The reconstruction error ||V - S @ H|| is not enough on its own. It goes
    flat as soon as the fit is good, but the footprint keeps shrinking long
    after that. A voxel is deleted only when its OWN weight decays past
    spatial_threshold, and that threshold never moves, because S is clipped at
    1 and every component's peak sits at 1. So voxels keep dropping out one by
    one while the error stands still. A run stopped on the error alone can
    leave a footprint well above its final size.

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


def nmf(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr=1,
                S_lr=1e-6, objective_threshold=None, estimate_noise_component=True,
                min_iterations: int = 50, spatial_threshold: float = None,
                objective_rtol: float = 1e-4, objective_window: int = 50,
                footprint_rtol: float = 1e-3, footprint_window: int = 100):
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
        objective_threshold: Deprecated and ignored. It compared an absolute objective
            difference between two neighbouring iterations, which cannot work across block
            sizes. Kept only so that older configs still load. Use objective_rtol.
        estimate_noise_component: Whether to estimate a noise component (default: True).
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).
        objective_rtol: Stop once the objective's mean falls by less than this FRACTION over
            objective_window iterations, against the window before it. Scale free, so one
            value fits every block size. None disables it (default: 1e-4).
        objective_window: How many iterations each of the two compared means covers
            (default: 50). The test runs every objective_window iterations.
        footprint_rtol: Stop once the count of surviving spatial voxels changes by less
            than this FRACTION over footprint_window iterations. Needed because the
            objective settles long before the footprint does. None disables it
            (default: 1e-3).
        footprint_window: How many iterations between two voxel counts (default: 100).
            The count costs about 0.02 percent of the run time at this window.

        Early stopping needs EVERY enabled test to pass. Setting one of the two rtols
        to None removes that test as a requirement, it does not switch early stopping
        off. To run all num_iterations, set objective_rtol AND footprint_rtol to None.
        spatial_threshold: Zero out spatial values below this fraction of their own component's
            peak, once per iteration. Bounds how far a footprint can spread when B never reaches
            zero. The noise component is exempt. None disables it (default: None).

    Returns:
        A tuple of NumPy arrays representing the learned spatial components matrix (S), temporal components matrix (H), and objective values (objectives).
    """

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

        # per-component threshold, so a dim component cannot be erased
        if spatial_threshold is not None:
            fit = S[:, :n_components]
            peak = fit.max(axis=0, keepdims=True)
            fit[fit < spatial_threshold * peak] = 0

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
        # Both tests must pass, because they settle at very different times.
        # The objective goes flat once the fit is good enough. The footprint
        # keeps shrinking well past that point. See footprint_settled.
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


def lateral_neighbour_counts(spatial_shape, device=None, dtype=None):
    """How many lateral neighbours each (y, x) position has.

    4 inside, 3 along an edge, 2 in a corner. Built by shifting a plane of
    ones, so a single row or a single column needs no special case. Never
    returns 0, so it is safe to divide by.

    Computed once, outside the iteration loop.
    """
    ny, nx = spatial_shape[1], spatial_shape[2]
    o = torch.ones((1, 1, ny, nx), device=device, dtype=dtype)
    c = torch.zeros_like(o)
    c[..., 1:, :] += o[..., :-1, :]
    c[..., :-1, :] += o[..., 1:, :]
    c[..., :, 1:] += o[..., :, :-1]
    c[..., :, :-1] += o[..., :, 1:]
    return torch.clamp(c, min=1.0)


def lateral_neighbour_mean(x, counts):
    """The mean of each voxel's lateral neighbours, excluding the voxel itself.

    `x` is (n_components, nz, ny, nx). `counts` comes from
    lateral_neighbour_counts for the same shape.

    LATERAL ONLY, on purpose. A z step is often much longer than a lateral
    step. A z neighbour is then much farther away, so mixing over z would join
    cells that sit in different planes.
    """
    s = torch.zeros_like(x)
    s[:, :, 1:, :] += x[:, :, :-1, :]
    s[:, :, :-1, :] += x[:, :, 1:, :]
    s[:, :, :, 1:] += x[:, :, :, :-1]
    s[:, :, :, :-1] += x[:, :, :, 1:]
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


def nmf_pytorch(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr: float = 1e-1,
                S_lr: float = 1e-1, step_size_norm: str = 'sum_of_squares',
                objective_threshold: float = None, min_iterations: int = 50,
                spatial_threshold: float = None, objective_rtol: float = 1e-4,
                objective_window: int = 50, footprint_rtol: float = 1e-3,
                footprint_window: int = 100, spatial_smoothness: float = None,
                spatial_shape=None):

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
        objective_threshold: Deprecated and ignored. See objective_rtol.
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).
        objective_rtol: Stop once the objective's mean falls by less than this FRACTION over
            objective_window iterations, against the window before it. Scale free, so one
            value fits every block size. None disables it (default: 1e-4).
        objective_window: How many iterations each of the two compared means covers
            (default: 50). The test runs every objective_window iterations.
        footprint_rtol: Stop once the count of surviving spatial voxels changes by less
            than this FRACTION over footprint_window iterations. Needed because the
            objective settles long before the footprint does. None disables it
            (default: 1e-3).
        footprint_window: How many iterations between two voxel counts (default: 100).
            The count costs about 0.02 percent of the run time at this window.

        Early stopping needs EVERY enabled test to pass. Setting one of the two rtols
        to None removes that test as a requirement, it does not switch early stopping
        off. To run all num_iterations, set objective_rtol AND footprint_rtol to None.
        spatial_threshold: Zero out spatial values below this fraction of their own component's
            peak, once per iteration. Bounds how far a footprint can spread when B never reaches
            zero. None disables it (default: None).
        spatial_smoothness: Move each spatial weight this fraction of the way toward the
            mean of its LATERAL neighbours, once per iteration. 0 or None disables it
            (default: None). Requires spatial_shape.

            WHY. Nothing else in the objective couples a voxel to the voxel beside it,
            so the fit is free to delete single voxels out of a smooth patch. That
            cuts one footprint into many small pieces. The pieces sit at the same
            distance from the seed as the legitimate halo, so a neighbourhood radius
            cannot separate them. They ARE the halo, cut apart by spatial_threshold.

            This term is the force that the gradient of a lateral roughness penalty
            gives, sum over lateral neighbour pairs of (S[a] - S[b])**2, but taken with
            a step of its own instead of the data step. That keeps the value readable:
            0.1 means "move 10 percent of the way to the local mean each iteration".
            It is scaled by B, so it cannot push a footprint outward where B is small.

            It runs AFTER the data update and BEFORE spatial_threshold, so a voxel held
            up by its neighbours survives the threshold.

            Too large a value merges neighbouring cells. Watch the fraction of voxels
            with 2 or more cells, and the trace correlation between neighbour cells.
            Neither may rise.
        spatial_shape: The (nz, ny, nx) shape of the spatial canvas, needed to find a
            voxel's neighbours. Its product must equal S_init.shape[0]. Only used when
            spatial_smoothness is on (default: None).

    Returns:
        A tuple of PyTorch tensors representing the learned spatial components matrix (S), temporal components matrix (H), and objective values (objectives).
    """

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

    objectives = []
    correlations = []
    H_gradients = []
    S_gradients = []

    # messing around with line search, keeping the value of the last iteration as the starting point for the next
    # With the idea that, during training, the learning rate should keep decreasing
    S_step_size = S_lr
    H_step_size = H_lr

    zero = torch.zeros((), device=device, dtype=S.dtype)
    # sampled every footprint_window iterations, so two neighbouring entries are
    # always the same distance apart
    nnz_history = []
    objective_done = False

    # lateral smoothing, prepared once
    counts = None
    if spatial_smoothness:
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
        counts = lateral_neighbour_counts(spatial_shape, device=device,
                                          dtype=S.dtype)

    print('Run gradient NMF | max iterations: {}'.format(num_iterations))
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):

        # Update H matrix with dynamically estimated step size
        H_gradient = S.T @ (V - S @ H)
        norm = torch.sum(S * S if square_norm else S, dim=0)[:, None]
        norm[norm == 0] = 1
        H_step_size = H_lr / norm
        H_gradient *= H_step_size
        H += H_gradient
        H = torch.maximum(H, torch.Tensor([0]).to(H.device))

        # Update S matrix with spatial constraint
        S_gradient = (V - S @ H) @ H.T
        norm = torch.sum(H * H if square_norm else H, dim=1)[None, :]
        norm[norm == 0] = 1
        S_step_size = S_lr / norm
        S_gradient *= S_step_size * B
        S += S_gradient
        S = torch.maximum(S, torch.Tensor([0]).to(S.device))
        S = torch.minimum(S, torch.Tensor([1]).to(S.device))  # we want amplitude in temporal not spatial components

        # Pull each weight toward the mean of its lateral neighbours. This runs
        # before the threshold on purpose: a voxel held up by its neighbours
        # then survives the cut, and an isolated speck is pushed down into it.
        if spatial_smoothness:
            n_comp = S.shape[1]
            avg = lateral_neighbour_mean(
                S.T.reshape((n_comp,) + spatial_shape), counts)
            avg = avg.reshape(n_comp, -1).T
            S = S + spatial_smoothness * B * (avg - S)
            S = torch.clamp(S, 0.0, 1.0)

        # per-component threshold, so a dim component cannot be erased
        if spatial_threshold is not None:
            peak = S.max(dim=0, keepdim=True).values
            S = torch.where(S < spatial_threshold * peak, zero, S)

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
        # Both tests must pass, because they settle at very different times.
        # The objective goes flat once the fit is good enough. The footprint
        # keeps shrinking well past that point. See footprint_settled.
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
