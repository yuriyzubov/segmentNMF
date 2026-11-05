"""Functions for computing neighborhoods of spatial components"""

import numpy as np
import copy

from scipy.ndimage import binary_dilation as binary_dilation_scipy
from scipy.ndimage import distance_transform_edt


def dilate_outer_layers(im, rad=[1, 1, 1], k: int = 1):
    """
    Dilate the outer layers of a binary image.

    Args:
        im (ndarray): The binary image.
        rad (list): The radius of the dilation in each dimension.
        k (int): The number of dilation iterations.

    Returns:
        ndarray: The dilated image.

    """
    for _ in range(k):
        inds = np.stack(np.where(im)).T
        im_eroded = copy.deepcopy(im)
        for ind in inds:
            window = tuple([slice(np.max([ind[i] - rad[i], 0]),
                                  np.min([ind[i] + rad[i] + 1, im.shape[i]]))
                            for i in range(len(im.shape))])
            if np.any(im[window] == 0):
                im_eroded[window] = 1
        if np.sum(im_eroded) == 0:
            im_eroded = im
        im = im_eroded
    return im


def neighborhood_by_dilation(S, segments_shape, k: int = 1, method: str = 'scipy'):
    """
    Expand the segmentation labels by dilation in each cell neighborhood.

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        k (int): The number of dilation iterations.
        method (str): The method for dilation ('scipy' or 'stand').

    Returns:
        ndarray: The expanded segmentation labels.

    """
    if method == 'scipy':
        dilation_func = binary_dilation_scipy
    elif method == 'stand':
        dilation_func = dilate_outer_layers

    B = S > 0
    dilate_radius = 3

    N_cells = S.shape[1]
    for n_i in range(N_cells):
        B_i = B[:, n_i].reshape(segments_shape)
        for z in range(segments_shape[0]):
            B_iz = B_i[z, :, :]
            for d_i in range(dilate_radius):
                B_iz = dilation_func(B_iz, iterations=k)
            B_i[z, :, :] = B_iz
        B[:, n_i] = B_i.flatten()

    return B


def distance_transform_edt_binary(arr, spacing):
    """
    Calculate the Euclidean distance transform of a binary array.

    Args:
        arr (ndarray): The binary array.
        spacing (float or tuple): The voxel spacing in each dimension.

    Returns:
        ndarray: The distance transform.

    """
    # Calculate the distance transform of the binary array
    arr = arr * -1 + 1
    dist_transform, closest_inds = distance_transform_edt(arr, spacing, return_indices=True)

    return dist_transform, closest_inds


def neighborhood_by_weights(S, segments_shape, spacing, sigma, subsample=None):

    """
    Expand the segmentation labels with weights that fall off

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        spacing (float or tuple): The voxel spacing in each dimension.
        sigma (float): Standard deviation of exponential fall off for weights
        subsample (iterable): Factors by which to skip sample each axis

    Returns:
        ndarray: The components spatial weights

    """

    # initialize empty neighborhood matrix
    B = np.zeros_like(S)

    # run over components
    for c_i in range(S.shape[1]):

        # get spatial component and initialize empty neighborhood
        Si = S[:, c_i].reshape(segments_shape)
        Bi = np.zeros_like(Si)

        # Find unique spatial component values, discard 0 (background) if present
        Si_unique = np.unique(Si)
        if Si_unique[0] == 0: Si_unique = Si_unique[1:]

        # run over unique spatial component entries and assign neighborhood values based on spatial falloff
        # this is done sequentially to make sure the largest weight value per index is assigned
        for v in Si_unique:
            dists_h, inds = distance_transform_edt_binary(Si == v, spacing)  # Within-plane (horizontal) distance
            dist_v = -np.log(v) * sigma  # Across planes (vertical) distance - Assuming same weighting calculation
            dists_tot = np.sqrt(dists_h**2 + dist_v**2)  # hypotenuse distance

            Bv_wts = np.exp(-dists_tot / sigma)
            Bi = np.maximum(Bv_wts, Bi)

        # Subsample B if applicable
        if subsample is not None:
            Bi = Bi[::subsample[0], ::subsample[1], ::subsample[2]]

        B[:, c_i] = Bi.reshape(S[:, c_i].shape)

    return B


def neighborhood_by_distance(S, segments_shape, spacing, max_dist, subsample=None):
    """
    Expand the segmentation labels by considering the Euclidean distance.

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        spacing (float or tuple): The voxel spacing in each dimension.
        max_dist (float): The maximum distance for expansion.
        subsample (iterable): Factors by which to skip sample each axis

    Returns:
        ndarray: The expanded segmentation labels.

    """

    # Initialize neighborhood with binarized S array
    B = S > 0

    if subsample is not None:
        B_res = B.reshape(*segments_shape, B.shape[1])
        B_res = B_res[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res = B_res.reshape(np.prod(B_res.shape[:-1]), B.shape[-1])
    else:
        B_res = B

    for c_i in range(B.shape[1]):
        dists, _ = distance_transform_edt_binary(B[:, c_i].reshape(segments_shape), spacing)
        Bi = (dists < max_dist)
        if subsample is not None:
            Bi = Bi[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res[:, c_i] = Bi.reshape(B_res[:, c_i].shape)

    # For each component i, rescale Bi by maximum of Si
    B_res = B_res * np.max(S, axis=0, keepdims=True)

    return B_res


def max_light_per_voxel(segments, labels, projection_factor, segments_spacing,
                        sigma_xy=2.0, sigma_z=4.0, widen_xy=1.0, widen_z=2.0,
                        core_level=0.5, plane_centres=None):
    """
    The most light each cell can physically put in each time-series voxel,
    relative to its own core, from the fine segmentation.

    For each label: its mask, widened by widen_xy (in plane) and widen_z (along
    z, both um) for mask placement errors, blurred laterally by a Gaussian of
    sigma_xy um, and integrated into each time-series plane under a Gaussian
    light sheet of sigma_z um centred on that plane's projection_factor fine
    planes. Normalised to peak 1, that is the cell's expected footprint U.

    Returns (S_max, core), both (n_voxels, n_labels) on the time-series grid of
    `segments` (z // projection_factor, y, x), flattened C-order:
        core   U >= core_level: the cell's own core region
        S_max  min(U / core_level, 1): 1 in the core, the blur tail outside
    nmf_pytorch then keeps S[v, i] <= S_max[v, i] * (S's peak inside core i), so a
    footprint may be as bright as it likes in its own core region, and outside
    it only as bright as blur allows. Relative to the core, because the scale of
    a footprint is free (the trace absorbs it).

    Args:
        segments (ndarray): fine labels (Zh, Y, X) of the block.
        labels (iterable): label values, one column each, in this order.
        projection_factor (int): fine planes per time-series plane.
        segments_spacing (3 floats): um per fine voxel (z, y, x).
        plane_centres (list of floats): the fine-plane index at the centre of
            each time-series plane's window. Default: full windows,
            j * projection_factor + (projection_factor - 1) / 2. A block at the
            first or last plane of the volume has a cut window there, and its
            caller passes the true centres.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter1d

    pf = int(projection_factor)
    sp = np.asarray(segments_spacing, dtype=float)
    Zh, Y, X = segments.shape
    if plane_centres is None:
        plane_centres = np.arange(Zh // pf) * pf + (pf - 1) / 2
    z_plane = (np.asarray(plane_centres, dtype=float) + 0.5) * sp[0]
    nz = len(z_plane)
    z_fine = (np.arange(Zh) + 0.5) * sp[0]
    sheet = np.exp(-0.5 * ((z_plane[:, None] - z_fine[None]) / sigma_z) ** 2)
    sxy = sigma_xy / sp[1]
    pad = int(np.ceil(3 * sxy + widen_xy / sp[1])) + 1
    rz = int(round(widen_z / sp[0]))

    labels = list(labels)
    S_max = np.zeros((nz * Y * X, len(labels)), dtype=np.float32)
    core = np.zeros((nz * Y * X, len(labels)), dtype=bool)
    for i, lab in enumerate(labels):
        zz, yy, xx = np.nonzero(segments == lab)
        if zz.size == 0:
            continue
        ys = slice(max(yy.min() - pad, 0), min(yy.max() + pad + 1, Y))
        xs = slice(max(xx.min() - pad, 0), min(xx.max() + pad + 1, X))
        m = segments[:, ys, xs] == lab
        if widen_xy > 0:
            # in-plane distance only: a huge z spacing keeps planes apart
            d = distance_transform_edt(~m, sampling=(1e6, sp[1], sp[2]))
            m = d <= widen_xy
        if rz > 0:
            m = maximum_filter1d(m, 2 * rz + 1, axis=0)
        img = gaussian_filter(m.astype(np.float32), (0, sxy, sxy))
        img = np.tensordot(sheet, img, axes=(1, 0))            # (nz, y, x)
        peak = img.max()
        if peak <= 0:
            continue
        U = np.zeros((nz, Y, X), dtype=np.float32)
        U[:, ys, xs] = img / peak
        U = U.ravel()
        S_max[:, i] = np.minimum(U / core_level, 1.0)
        core[:, i] = U >= core_level
    return S_max, core
