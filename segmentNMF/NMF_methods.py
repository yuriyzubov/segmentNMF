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


def nmf(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr=1,
                S_lr=1e-6, objective_threshold=1e-6, estimate_noise_component=True,
                min_iterations: int = 50):
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
        objective_threshold: The stopping criterion for the objective (default: 1e-6).
        estimate_noise_component: Whether to estimate a noise component (default: True).
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).

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

        # Check stopping criterion
        if objective_threshold is not None and i > min_iterations:
            if abs(objectives[i] - objectives[i - 1]) < objective_threshold:
                print('Reached stopping criterion (d objective < {})'.format(objective_threshold))
                break
            if objectives[i] > objectives[i - 1]:
                print('Objective increased: [i-1]: {}, [i]: {}'.format(objectives[i-1], objectives[i]))
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
                objective_threshold: float = 1e-2, min_iterations: int = 50):

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
        objective_threshold: The stopping criterion for the objective (default: 1e-2).
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).

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

    print('Run gradient NMF | max iterations: {}'.format(num_iterations))
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):

        # Update H matrix with dynamically estimated step size
        H_gradient = S.T @ (V - S @ H)
        norm = torch.sum(S * S if square_norm else S, dim=0)[:, None]
        norm = torch.clamp(norm, min=NORM_FLOOR)
        H_step_size = H_lr / norm
        H_gradient *= H_step_size
        H += H_gradient
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

        # Check stopping criterion
        if objective_threshold is not None and i > min_iterations:
            if abs(objectives[i] - objectives[i - 1]) < objective_threshold:
                print('Reached stopping criterion (d objective < {})'.format(objective_threshold))
                break
            if objectives[i] > objectives[i - 1]:
                print('Objective increased: [i-1]: {}, [i]: {}'.format(objectives[i-1], objectives[i]))
                break

    # Move tensors back to the CPU
    S = S.cpu().numpy()
    H = H.cpu().numpy()

    if H_true is not None:
        return S, H, objectives, correlations
    else:
        return S, H, objectives
