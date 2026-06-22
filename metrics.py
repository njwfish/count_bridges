"""
Simple and clean metrics computation for generative models.
Dictionary-based approach for easy extensibility.
"""

import numpy as np
import scipy.stats
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


def mean_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean absolute error between means"""
    return float(np.mean(np.abs(np.mean(X, axis=0) - np.mean(Y, axis=0))))


def variance_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean absolute error between variances"""
    return float(np.mean(np.abs(np.var(X, axis=0) - np.var(Y, axis=0))))


def skewness_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean absolute error between skewness"""
    return float(np.mean(np.abs(scipy.stats.skew(X, axis=0) - scipy.stats.skew(Y, axis=0))))


def kurtosis_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean absolute error between kurtosis"""
    return float(np.mean(np.abs(scipy.stats.kurtosis(X, axis=0) - scipy.stats.kurtosis(Y, axis=0))))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two samples"""
    from scipy.spatial.distance import cdist
    
    XY_dist = np.mean(cdist(X, Y, 'euclidean'))
    XX_dist = np.mean(cdist(X, X, 'euclidean'))
    YY_dist = np.mean(cdist(Y, Y, 'euclidean'))
    
    return float(2 * XY_dist - XX_dist - YY_dist)


def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 100, scale: float = 100.0) -> float:
    """Wasserstein distance (1D per dimension, then averaged)"""

    projections = np.random.randn(X.shape[1], n_projections)
    projections = projections / np.linalg.norm(projections, axis=0)

    X = X @ projections / scale
    Y = Y @ projections / scale

    dist = np.sqrt(np.mean(np.square(np.sort(X, axis=0) - np.sort(Y, axis=0))))
    return float(dist)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0, scale: float = 100.) -> float:
    """Maximum Mean Discrepancy with RBF kernel"""
    from scipy.spatial.distance import cdist

    X = X / scale
    Y = Y / scale
    
    XX = cdist(X, X, 'sqeuclidean')
    YY = cdist(Y, Y, 'sqeuclidean')
    XY = cdist(X, Y, 'sqeuclidean')
    
    K_XX = np.exp(- gamma * XX)
    K_YY = np.exp(- gamma * YY)
    K_XY = np.exp(- gamma * XY)
    
    mmd_squared = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return float(max(0, mmd_squared) ** 0.5)


def covariance_frobenius(X: np.ndarray, Y: np.ndarray) -> float:
    """Frobenius norm of covariance difference"""
    d = X.shape[1]
    cov_X = np.cov(X.T)
    cov_Y = np.cov(Y.T)
    return float(np.linalg.norm(cov_X - cov_Y, 'fro')) / d**2


# Dictionary of all metrics: name -> function
METRICS_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    'mean_error': mean_error,
    'variance_error': variance_error,
    'skewness_error': skewness_error,
    'kurtosis_error': kurtosis_error,
    'energy_distance': energy_distance,
    'wasserstein_distance': sliced_wasserstein_distance,
    'mmd_rbf': mmd_rbf,
    'covariance_frobenius': covariance_frobenius,
}


def compute_comprehensive_metrics(eval_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all metrics by looping through the metrics dictionary.
    Super simple and clean!
    """
    x0_target = eval_data['x0_target']
    x0_generated = eval_data['x0_generated']

    if isinstance(x0_target, dict):
        logger.info("Detected dict-valued data - skipping statistical metrics")
        return None

    # Image data (N, C, H, W) or (N, H, W): flatten to (N, D) so the
    # distributional metrics below still apply pixel-wise.
    if x0_target.ndim > 2:
        x0_target = x0_target.reshape(x0_target.shape[0], -1)
        x0_generated = x0_generated.reshape(x0_generated.shape[0], -1)

    # Ensure same number of samples
    min_samples = min(len(x0_target), len(x0_generated))
    x0_target = x0_target[:min_samples]
    x0_generated = x0_generated[:min_samples]
    
    metrics = {}
    
    # Loop through all metrics functions
    for metric_name, metric_fn in METRICS_FUNCTIONS.items():
        try:
            metrics[metric_name] = metric_fn(x0_target, x0_generated)
        except Exception as e:
            logger.debug(f"Could not compute {metric_name}: {e}")
            metrics[metric_name] = float('nan')
    
    # Add basic info
    metrics['data_type'] = 'vector'
    metrics['n_samples'] = min_samples
    metrics['n_dimensions'] = x0_target.shape[1]
    
    return metrics