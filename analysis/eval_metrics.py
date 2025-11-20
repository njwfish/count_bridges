import numpy as np
from scipy.spatial.distance import jensenshannon

def jsd_ct(predicted: np.ndarray, true: np.ndarray, eps: float = 1e-12):
    """
    Paper Eq. (2): JSD per cell type k between Q(P_k) and Q(T_k),
    where each column is normalized across spots.

    Parameters
    ----------
    predicted, true : arrays of shape (n_spots, n_celltypes)

    Returns
    -------
    jsd_per_ct : (n_celltypes,) JSD for each cell type (in nats)
    jsd_mean   : scalar mean across cell types
    """
    if predicted.shape != true.shape:
        raise ValueError("Arrays must have the same shape.")

    # stabilize and normalize each column to sum to 1 (distribution over spots)
    P = predicted + eps
    P /= P.sum(axis=0, keepdims=True)
    Q = true + eps
    Q /= Q.sum(axis=0, keepdims=True)

    M = 0.5 * (P + Q)

    # KL terms summed over spots (axis=0), yielding one value per cell type
    jsd_per_ct = 0.5 * np.sum(P * (np.log(P) - np.log(M)), axis=0) + \
                 0.5 * np.sum(Q * (np.log(Q) - np.log(M)), axis=0)

    return jsd_per_ct, float(jsd_per_ct.mean())

def per_celltype_pearson(predicted: np.ndarray, true: np.ndarray):
    """
    Compute Pearson correlation for each cell type (column-wise).

    Parameters
    ----------
    predicted, true : (n_samples, n_celltypes)

    Returns
    -------
    corrs : np.ndarray shape (n_celltypes,)
    mean_corr : float
    """
    n_ct = predicted.shape[1]
    corrs = np.zeros(n_ct)
    for i in range(n_ct):
        x = predicted[:, i]
        y = true[:, i]
        if np.std(x) == 0 or np.std(y) == 0:
            corrs[i] = 0.0
            continue
        corr_val, _ = pearsonr(x, y)
        if np.isnan(corr_val):
            corr_val = 0.0
        corrs[i] = corr_val
    return corrs, corrs.mean()

from scipy.stats import spearmanr, pearsonr

def per_celltype_spearman(predicted: np.ndarray, true: np.ndarray):
    n_ct = predicted.shape[1]
    corrs = np.zeros(n_ct)
    for i in range(n_ct):
        x = predicted[:, i]
        y = true[:, i]
        if np.all(x == x[0]) or np.all(y == y[0]):
            corrs[i] = 0.0
            continue
        corr_val, _ = spearmanr(x, y)
        if np.isnan(corr_val):
            corr_val = 0.0
        corrs[i] = corr_val
    return corrs, corrs.mean()

def mae_per_celltype(predicted: np.ndarray, true: np.ndarray):
    """
    Returns MAE per cell type and average.
    """
    errors = np.abs(predicted - true).mean(axis=0)
    return errors, errors.mean()

def rmse_per_celltype(predicted: np.ndarray, true: np.ndarray):
    errors = np.sqrt(((predicted - true)**2).mean(axis=0))
    return errors.mean()
def mse_per_celltype(predicted: np.ndarray, true: np.ndarray):
    errors = ((predicted - true)**2).mean(axis=0)
    return errors.mean()
