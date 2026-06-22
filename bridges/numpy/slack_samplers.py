import numpy as np
from scipy.special import iv


class ZeroM:
    """Zero-slack sampler: lambda = 0, no extra Bessel noise.
    Reduces the bridge to deterministic ``N_1 = |x_1 - x_0|`` thinned by the
    binomial(t)/hypergeometric draws.
    """

    def __init__(self, markov: bool = True):
        self.markov = markov

    def __call__(self, diff: np.ndarray):
        return np.zeros(diff.shape, dtype=diff.dtype)


class ConstantM:
    def __init__(self, m: int, markov: bool = True):
        self.m = m
        self.markov = markov

    def __call__(self, diff: np.ndarray):
        return np.full(diff.shape, self.m)

class PoissonM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = False):
        self.lam_p = np.asarray(lam_p)
        self.lam_m = np.asarray(lam_m)
        self.lam_star = 2.0 * np.sqrt(self.lam_p * self.lam_m)
        self.markov = markov
    
    def __call__(self, diff: np.ndarray):
        return np.random.poisson(self.lam_star, diff.shape)

class BesselM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = True):
        self.lam_p = np.asarray(lam_p)
        self.lam_m = np.asarray(lam_m)
        self.lam_star = 2.0 * np.sqrt(self.lam_p * self.lam_m)
        self.markov = markov
        
    def __call__(self, diff: np.ndarray):
        lam_p = np.broadcast_to(self.lam_p, diff.shape)
        lam_m = np.broadcast_to(self.lam_m, diff.shape)
        
        return sample_bessel_devroye(lam_p, lam_m, diff)