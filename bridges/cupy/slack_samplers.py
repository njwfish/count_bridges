import cupy as cp

from .sampling.bessel import bessel


class ZeroM:
    """Zero-slack sampler: lambda = 0, no extra Bessel/Skellam noise.

    Reduces the bridge to deterministic ``N_1 = |x_1 - x_0|`` thinned by the
    binomial(t)/hypergeometric draws. Avoids the Bessel call entirely (which
    is by far the most expensive piece of the per-step work). Allocates a
    single zero buffer and returns it.
    """

    def __init__(self, markov: bool = True):
        self.markov = markov

    def __call__(self, diff: cp.ndarray, w: cp.ndarray):
        return cp.zeros(diff.shape, dtype=diff.dtype)


class ConstantM:
    def __init__(self, m: int, markov: bool = True):
        self.m = m
        self.markov = markov

    def __call__(self, diff: cp.ndarray, w: cp.ndarray):
        return cp.round(cp.full(diff.shape, self.m) * w)

class BesselM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = True):
        self.lam_p = cp.asarray(lam_p)
        self.lam_m = cp.asarray(lam_m)
        self.markov = markov
        
    def __call__(self, diff: cp.ndarray, w: cp.ndarray):
        lam_p = cp.broadcast_to(self.lam_p, diff.shape) * w
        lam_m = cp.broadcast_to(self.lam_m, diff.shape) * w
        
        return bessel(lam_p, lam_m, diff)
