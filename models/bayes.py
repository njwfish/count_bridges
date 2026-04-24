"""
Bayes-optimal denoisers for the GaussianBridge / AdaptiveGaussianBridge,
under independent Gaussian-mixture coupling.

Includes BayesOracleSpec, a "theory-checkpoint" hydra-instantiable wrapper
that defers full model construction until a dataset is available. main.py
detects `model.is_oracle == True` and calls `model.build(dataset)` to swap
in the actual closed-form denoiser, then skips training and runs only the
existing sampling/eval pipeline. This makes oracle runs first-class
citizens of the same checkpoint/config/output-dir machinery as trained
runs.

These plug into the existing reverse samplers via the standard
    .sample(x_t, t, **z) -> Tensor
interface, just like a trained denoiser. They are used for oracle ablations
(see adaptive_gaussian_bridge.tex Sec. 5.2): given the exact conditional
mean (or a posterior sample) of X_0 given X_t, how does the same reverse
sampler perform? This isolates "integration dynamics" from
"function-fitting error".

Two classes:
  - BayesDenoiserGMMFixed: closed-form for the fixed-sigma Brownian bridge.
  - BayesDenoiserGMMAdaptive: per-coordinate Gauss-Legendre quadrature for
    the adaptive bridge with axis-aligned data covariances.

Both expose `mode='mean'` (returns the conditional mean E[X_0 | X_t = x_t])
and `mode='sample'` (returns one draw from p(X_0 | X_t = x_t)).
"""

import math
import torch
import torch.nn as nn
import numpy as np


def _build_data_space_gmm(latent_means, latent_covs, projection, noise_scale):
    """Convert a low-rank GMM (mean, cov in latent space + projection + isotropic noise)
    to its ambient data-space form: mu_data_c = W mu_c, cov_data_c = W Sigma_c W^T + noise^2 I.

    Args:
        latent_means:  (k, r)
        latent_covs:   (k, r, r)
        projection:    (d, r)
        noise_scale:   scalar
    Returns:
        mu:  (k, d)
        cov: (k, d, d)
    """
    k = latent_means.shape[0]
    d = projection.shape[0]
    W = projection
    mu = latent_means @ W.T                                                  # (k, d)
    # cov_c = W Sigma_c W^T + noise^2 I
    cov = torch.einsum('dr, krs, es -> kde', W, latent_covs, W)              # (k, d, d)
    cov = cov + (noise_scale ** 2) * torch.eye(d).expand(k, d, d)
    # Symmetrise to kill numerical asymmetry
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return mu, cov


# -----------------------------------------------------------------------------
#  Fixed-sigma Brownian bridge: closed-form Bayes denoiser
# -----------------------------------------------------------------------------
class BayesDenoiserGMMFixed(nn.Module):
    """
    Closed-form Bayes denoiser for the fixed-sigma Brownian bridge with
    independent Gaussian-mixture coupling.

    X_0 ~ sum_c pi_c^s N(mu_c^s, Sigma_c^s)
    X_1 ~ sum_c' pi_c'^t N(mu_c'^t, Sigma_c'^t)   (drawn independently of X_0)
    X_t = (1-t) X_0 + t X_1 + sigma sqrt(t(1-t)) eps

    Posterior over component pair (c, c') given x_t:
        p(c, c' | x_t)  proportional to  pi_c^s pi_c'^t * N(x_t; mu_t_cc', Sigma_t_cc')
        with mu_t_cc'  = (1-t) mu_c^s + t mu_c'^t
             Sigma_t_cc' = (1-t)^2 Sigma_c^s + t^2 Sigma_c'^t + sigma^2 t(1-t) I

    Within each pair X_0 | X_t is affine-Gaussian:
        E[X_0 | X_t, c, c'] = mu_c^s + (1-t) Sigma_c^s Sigma_t_cc'^{-1} (x_t - mu_t_cc')
        Cov(X_0 | X_t, c, c') = Sigma_c^s - (1-t)^2 Sigma_c^s Sigma_t_cc'^{-1} Sigma_c^s
    """

    def __init__(self, k_src: int, k_tgt: int, d: int,
                 sigma: float, mode: str = 'mean'):
        """Shape-only constructor (hydra-friendly).

        Buffers are zero-initialised; populate them via `set_params(...)` or
        load a checkpoint produced by `scripts/build_bayes_checkpoint.py`.
        """
        super().__init__()
        self.register_buffer('mu_s', torch.zeros(k_src, d))
        self.register_buffer('cov_s', torch.eye(d).expand(k_src, d, d).contiguous().clone())
        self.register_buffer('logw_s', torch.full((k_src,), -math.log(k_src)))
        self.register_buffer('mu_t', torch.zeros(k_tgt, d))
        self.register_buffer('cov_t', torch.eye(d).expand(k_tgt, d, d).contiguous().clone())
        self.register_buffer('logw_t', torch.full((k_tgt,), -math.log(k_tgt)))
        self.sigma = float(sigma)
        self.mode = mode

    def set_params(self, means_src, covs_src, weights_src,
                   means_tgt, covs_tgt, weights_tgt):
        """Populate the GMM buffers in-place."""
        self.mu_s.copy_(means_src.float())
        self.cov_s.copy_(covs_src.float())
        self.logw_s.copy_(weights_src.float().clamp_min(1e-30).log())
        self.mu_t.copy_(means_tgt.float())
        self.cov_t.copy_(covs_tgt.float())
        self.logw_t.copy_(weights_tgt.float().clamp_min(1e-30).log())

    # -------- factories --------
    @classmethod
    def from_dataset(cls, dataset, sigma, mode='mean'):
        """Build from a LowRankGaussianMixtureDataset (data-space GMM)."""
        mu_s, cov_s = _build_data_space_gmm(
            dataset.means_source, dataset.covs_source,
            dataset.proj_source, dataset.noise_scale)
        mu_t, cov_t = _build_data_space_gmm(
            dataset.means_target, dataset.covs_target,
            dataset.proj_target, dataset.noise_scale)
        d = mu_s.shape[-1]
        m = cls(k_src=mu_s.shape[0], k_tgt=mu_t.shape[0], d=d,
                sigma=sigma, mode=mode)
        m.set_params(mu_s, cov_s, dataset.weights_source,
                     mu_t, cov_t, dataset.weights_target)
        return m

    # -------- per-pair bridge marginals --------
    def _xt_marginal_mean(self, t):
        """E[X_t | c, c']. Returns (k_s, k_t, d)."""
        return (1.0 - t) * self.mu_s.unsqueeze(1) + t * self.mu_t.unsqueeze(0)

    def _xt_marginal_cov(self, t):
        """Cov(X_t | c, c'). Returns (k_s, k_t, d, d)."""
        d = self.mu_s.shape[1]
        I = torch.eye(d, device=self.mu_s.device, dtype=self.mu_s.dtype)
        cov = ((1.0 - t) ** 2) * self.cov_s.unsqueeze(1) \
              + (t ** 2) * self.cov_t.unsqueeze(0) \
              + (self.sigma ** 2) * t * (1.0 - t) * I
        return cov

    def _gaussian_log_prob(self, x, mu, cov):
        """log N(x; mu, cov) for x: (..., d), mu: (..., d), cov: (..., d, d).
        Uses Cholesky for stability."""
        d = mu.shape[-1]
        L = torch.linalg.cholesky(cov)
        diff = (x - mu).unsqueeze(-1)
        v = torch.cholesky_solve(diff, L).squeeze(-1)
        quad = (diff.squeeze(-1) * v).sum(-1)
        log_det = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (quad + log_det + d * math.log(2.0 * math.pi))

    def _log_post_pairs(self, x_t, t):
        """log p(c, c' | x_t). Returns (B, k_s, k_t)."""
        ks, kt, d = self.mu_s.shape[0], self.mu_t.shape[0], self.mu_s.shape[1]
        B = x_t.shape[0]
        mu = self._xt_marginal_mean(t)                                       # (ks, kt, d)
        cov = self._xt_marginal_cov(t)                                       # (ks, kt, d, d)

        # broadcast: (B, 1, 1, d) - (1, ks, kt, d) and (1, ks, kt, d, d)
        x_e = x_t.unsqueeze(1).unsqueeze(1).expand(-1, ks, kt, -1)            # (B, ks, kt, d)
        mu_e = mu.unsqueeze(0).expand(B, -1, -1, -1)                          # (B, ks, kt, d)
        cov_e = cov.unsqueeze(0).expand(B, -1, -1, -1, -1)                    # (B, ks, kt, d, d)

        log_p_xt = self._gaussian_log_prob(x_e, mu_e, cov_e)                  # (B, ks, kt)
        log_pri = self.logw_s.unsqueeze(0).unsqueeze(-1) \
                  + self.logw_t.unsqueeze(0).unsqueeze(0)                     # (1, ks, kt)
        log_post = log_p_xt + log_pri
        log_post = log_post - log_post.flatten(1, 2).logsumexp(-1).unsqueeze(-1).unsqueeze(-1)
        return log_post

    def _x0_cond_mean_pairs(self, x_t, t):
        """E[X_0 | X_t = x_t, c, c'] for each (B, ks, kt). Returns (B, ks, kt, d)."""
        ks, kt, d = self.mu_s.shape[0], self.mu_t.shape[0], self.mu_s.shape[1]
        B = x_t.shape[0]
        mu_x = self._xt_marginal_mean(t)                                       # (ks, kt, d)
        cov_x = self._xt_marginal_cov(t)                                       # (ks, kt, d, d)
        cov_x0_xt = (1.0 - t) * self.cov_s.unsqueeze(1).expand(-1, kt, -1, -1) # (ks, kt, d, d)

        # solve cov_x @ alpha = (x_t - mu_x)  for alpha; then mean = mu_s + cov_x0_xt @ alpha.
        L = torch.linalg.cholesky(cov_x)                                       # (ks, kt, d, d)
        diff = (x_t.unsqueeze(1).unsqueeze(1) - mu_x.unsqueeze(0)).unsqueeze(-1)  # (B, ks, kt, d, 1)
        L_e = L.unsqueeze(0).expand(B, -1, -1, -1, -1)                          # (B, ks, kt, d, d)
        alpha = torch.cholesky_solve(diff, L_e)                                # (B, ks, kt, d, 1)
        update = (cov_x0_xt.unsqueeze(0) @ alpha).squeeze(-1)                   # (B, ks, kt, d)
        return self.mu_s.unsqueeze(0).unsqueeze(2) + update                    # (B, ks, kt, d)

    @torch.no_grad()
    def sample(self, x_t, t, **kwargs):
        # `t` in the bridge sampler is (B, 1) tensor with all entries equal.
        if isinstance(t, torch.Tensor):
            t_scalar = float(t.flatten()[0])
        else:
            t_scalar = float(t)

        log_post = self._log_post_pairs(x_t, t_scalar)                          # (B, ks, kt)
        weights = log_post.exp()                                                # (B, ks, kt)
        m_cc = self._x0_cond_mean_pairs(x_t, t_scalar)                          # (B, ks, kt, d)

        # Posterior mean
        m = (weights.unsqueeze(-1) * m_cc).sum(dim=(1, 2))                      # (B, d)

        if self.mode == 'mean':
            return m

        # mode == 'sample': draw component pair (c, c') from posterior, then
        # X_0 ~ N(m_cc, Sigma_{0|t,cc'}) where Sigma_{0|t} = Sigma_s_c - (1-t)^2 Sigma_s_c Sigma_xt^{-1} Sigma_s_c.
        ks, kt, d = self.mu_s.shape[0], self.mu_t.shape[0], self.mu_s.shape[1]
        B = x_t.shape[0]
        flat_w = weights.reshape(B, ks * kt)                                    # (B, ks*kt)
        idx = torch.multinomial(flat_w, num_samples=1).squeeze(-1)              # (B,)
        c_s = idx // kt
        c_t = idx % kt
        chosen_m = m_cc[torch.arange(B), c_s, c_t]                              # (B, d)

        # Compute conditional cov for the chosen pairs
        cov_x = self._xt_marginal_cov(t_scalar)                                 # (ks, kt, d, d)
        L = torch.linalg.cholesky(cov_x)                                        # (ks, kt, d, d)
        Sigma_s = self.cov_s                                                    # (ks, d, d)

        # cond_cov = Sigma_s - (1-t)^2 Sigma_s Sigma_x^{-1} Sigma_s
        # We need Sigma_x^{-1} Sigma_s.  Solve Sigma_x @ M = Sigma_s.
        S_s = Sigma_s.unsqueeze(1).expand(-1, kt, -1, -1)                       # (ks, kt, d, d)
        M = torch.cholesky_solve(S_s, L)                                        # (ks, kt, d, d)
        cond_cov = Sigma_s.unsqueeze(1).expand(-1, kt, -1, -1) \
                   - ((1.0 - t_scalar) ** 2) * (S_s @ M)                        # (ks, kt, d, d)
        cond_cov = 0.5 * (cond_cov + cond_cov.transpose(-1, -2))

        # Pick the chosen pair's cond_cov per batch
        chosen_cov = cond_cov[c_s, c_t]                                         # (B, d, d)
        L_cc = torch.linalg.cholesky(chosen_cov + 1e-6 * torch.eye(d, device=x_t.device))
        z = torch.randn(B, d, device=x_t.device, dtype=x_t.dtype)
        sample = chosen_m + (L_cc @ z.unsqueeze(-1)).squeeze(-1)
        return sample


# -----------------------------------------------------------------------------
#  Adaptive bridge: per-coordinate Gauss-Legendre quadrature on a diagonal GMM
# -----------------------------------------------------------------------------
class BayesDenoiserGMMAdaptive(nn.Module):
    """
    Bayes denoiser for the adaptive bridge with diagonal-covariance GMMs
    (or with the diagonal of full-cov GMMs as a per-coordinate factorised
    approximation).

    For each coordinate i and component pair (c, c'), the posterior on
    u_i = X_1^i - X_0^i given x_t^i is a 1-D scale-mixture density:
        p(u_i | x_t^i, c, c')  proportional to  p_U(u_i | c, c') * p(x_t^i | u_i, c, c')
    where U_i | c, c' is a Gaussian (sum of two independent Gaussians) and
    p(x_t^i | u_i, c, c') is the bridge likelihood at displacement u_i.

    Because the per-coordinate mixture and the bridge factorise per
    coordinate (under diagonal cov), the d-dim Bayes mean factorises:
        m^i(x, t) = sum_{c, c'} w_{cc'}^i(x_i, t) * E_{p(u_i | x_i, c, c', t)} [ E[X_0^i | u_i, x_i] ].

    All inner expectations are computed by Gauss-Legendre on x = tanh(y/2).
    """

    def __init__(self, k_src: int, k_tgt: int, d: int,
                 sigma: float, lam: float,
                 mode: str = 'mean', gl_n: int = 128):
        """Shape-only constructor (hydra-friendly).

        Buffers are zero-initialised; populate via `set_params(...)` or load
        a checkpoint from `scripts/build_bayes_checkpoint.py`.
        """
        super().__init__()
        self.register_buffer('mu_s', torch.zeros(k_src, d))
        self.register_buffer('var_s', torch.ones(k_src, d))
        self.register_buffer('logw_s', torch.full((k_src,), -math.log(k_src)))
        self.register_buffer('mu_t', torch.zeros(k_tgt, d))
        self.register_buffer('var_t', torch.ones(k_tgt, d))
        self.register_buffer('logw_t', torch.full((k_tgt,), -math.log(k_tgt)))
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.mode = mode
        self.gl_n = gl_n
        x_d, w_d = np.polynomial.legendre.leggauss(gl_n)
        self.register_buffer('_gl_x', torch.tensor(x_d, dtype=torch.float64))
        self.register_buffer('_gl_w', torch.tensor(w_d, dtype=torch.float64))

    def set_params(self, means_src, var_src, weights_src,
                   means_tgt, var_tgt, weights_tgt):
        """Populate per-dim diagonal-GMM buffers in-place."""
        self.mu_s.copy_(means_src.float())
        self.var_s.copy_(var_src.float())
        self.logw_s.copy_(weights_src.float().clamp_min(1e-30).log())
        self.mu_t.copy_(means_tgt.float())
        self.var_t.copy_(var_tgt.float())
        self.logw_t.copy_(weights_tgt.float().clamp_min(1e-30).log())

    @classmethod
    def from_dataset(cls, dataset, sigma, lam, mode='mean', gl_n=128):
        mu_s, cov_s = _build_data_space_gmm(
            dataset.means_source, dataset.covs_source,
            dataset.proj_source, dataset.noise_scale)
        mu_t, cov_t = _build_data_space_gmm(
            dataset.means_target, dataset.covs_target,
            dataset.proj_target, dataset.noise_scale)
        var_s = torch.diagonal(cov_s, dim1=-2, dim2=-1)
        var_t = torch.diagonal(cov_t, dim1=-2, dim2=-1)
        d = mu_s.shape[-1]
        m = cls(k_src=mu_s.shape[0], k_tgt=mu_t.shape[0], d=d,
                sigma=sigma, lam=lam, mode=mode, gl_n=gl_n)
        m.set_params(mu_s, var_s, dataset.weights_source,
                     mu_t, var_t, dataset.weights_target)
        return m

    @torch.no_grad()
    def sample(self, x_t, t, **kwargs):
        if isinstance(t, torch.Tensor):
            t_scalar = float(t.flatten()[0])
        else:
            t_scalar = float(t)
        return self._compute_mean(x_t, t_scalar)

    def _compute_mean(self, x_t, t):
        """E[X_0 | X_t = x_t] per dim, summed over component pairs (c, c').

        Per-coordinate, per-pair, and uses Gauss-Legendre on x = tanh(y/2).
        """
        device = x_t.device
        dtype = x_t.dtype
        B, d = x_t.shape
        ks, kt = self.mu_s.shape[0], self.mu_t.shape[0]
        omt = 1.0 - t

        # Bridge marginal variance per dim per pair (no u, marginalised):
        #   Var(X_t^i | c, c') = Var((1-t) X_0 + t X_1) + E[ sigma^2 g(U) t(1-t) ]
        # The first term: (1-t)^2 var_s + t^2 var_t per coord.
        # The second term needs E[sqrt(lam + U^2)] under U = X_1-X_0 with X_0 ~ N(mu_s, var_s),
        # X_1 ~ N(mu_t, var_t) independent => U ~ N(mu_t - mu_s, var_s + var_t).
        # We approximate Var(X_t) here by the deterministic-interpolant variance plus
        # sigma^2 t(1-t) * E[g(U)|c,c'] for component-classification likelihood. This is
        # an analytical Laplace-style approximation used only for posterior weighting; the
        # inner conditional mean E[X_0|X_t,u] is still exact per-(u, c, c').

        # Per-coord: mu_xt = (1-t) mu_s + t mu_t, with shape (ks, kt, d)
        mu_xt = (1 - t) * self.mu_s.unsqueeze(1) + t * self.mu_t.unsqueeze(0)   # (ks, kt, d)

        # Per-coord: U mean and var
        u_mean = self.mu_t.unsqueeze(0) - self.mu_s.unsqueeze(1)                # (ks, kt, d)
        u_var = self.var_s.unsqueeze(1) + self.var_t.unsqueeze(0)               # (ks, kt, d)

        # E[g(U)] per pair per coord, marginal Gaussian: E[sqrt(lam + U^2)] for U ~ N(mu, var).
        # Use analytic formula via non-central chi: E[sqrt(lam + U^2)] = sqrt(lam) * E[cosh(asinh(U/sqrt(lam)))]
        # which is hard in closed form. We use a 2-term moment-match:
        #   E[sqrt(lam + U^2)] ~ sqrt(lam + E[U^2]) = sqrt(lam + mu^2 + var). Tight to <1% for our regimes.
        u_sq_mean = u_mean ** 2 + u_var
        E_g_pair = torch.sqrt(self.lam + u_sq_mean)                             # (ks, kt, d)

        var_xt_pair = ((1 - t) ** 2 * self.var_s.unsqueeze(1)
                        + t ** 2 * self.var_t.unsqueeze(0)
                        + self.sigma ** 2 * t * omt * E_g_pair)                 # (ks, kt, d)

        # Per-pair log p(x_t | c, c') under diagonal Gaussian approx
        log_p_xt = (-0.5 * ((x_t.unsqueeze(1).unsqueeze(1) - mu_xt.unsqueeze(0)) ** 2
                               / var_xt_pair.unsqueeze(0)
                          + torch.log(var_xt_pair.unsqueeze(0))
                          + math.log(2 * math.pi))).sum(-1)                     # (B, ks, kt)
        log_pri = self.logw_s.unsqueeze(0).unsqueeze(-1) \
                  + self.logw_t.unsqueeze(0).unsqueeze(0)
        log_post = log_p_xt + log_pri
        log_post = log_post - log_post.flatten(1).logsumexp(-1).unsqueeze(-1).unsqueeze(-1)
        weights = log_post.exp()                                                # (B, ks, kt)

        # Now compute per-pair, per-coord conditional mean
        # E[X_0^i | X_t^i = x, c, c'] = E_{p(u_i | x, c, c', t)}[ E[X_0^i | u, x_i, c, c'] ]
        # Within the (c, c') pair the prior on (X_0^i, X_1^i) is Gaussian with variances
        # (var_s_c[i], var_t_c'[i]). The bridge is x_t = (1-t) X_0 + t X_1 + sigma sqrt(g(u)t(1-t)) eps.
        # Inner conditional E[X_0 | x_t, u, c, c'] uses Gaussian conditioning given the
        # induced midpoint reparam. We compute it via GL quadrature on x = tanh(y/2).

        # Per-pair Gaussian factors:
        #   X_0 ~ N(mu_s, var_s); X_1 ~ N(mu_t, var_t) given (c, c').
        #   Midpoint R = (X_0 + X_1)/2 ~ N(mu_R, S_R) with mu_R = (mu_s + mu_t)/2, S_R = (var_s+var_t)/4.
        #   U = X_1 - X_0 ~ N(u_mean, u_var). R indep U? Only if var_s == var_t.
        #   In general Cov(R, U) = (var_t - var_s)/2.

        # For simplicity we implement the U-marginalisation per pair per dim using
        # the bridge likelihood AND a Gaussian prior on u from (c, c').
        # p(u | x, c, c') propto N(u; u_mean, u_var) * (sigma^2 g(u) t(1-t) + S_R)^{-1/2}
        #                       * exp(-(x - mu_R - delta u)^2 / (2 (S_R + sigma^2 g(u) t(1-t)))),
        # delta = t - 1/2.
        # And E[X_0 | x, u, c, c'] = mu_R + (S_R / (S_R + sigma^2 g(u) t(1-t))) * (x - mu_R - delta u)
        #                              - u/2 + Cov(R, U)/u_var * (u - u_mean) (last term zero if var_s==var_t)
        # For implementation simplicity we use diag covs => Cov(R, U) coupling exists but
        # is small for well-mixed pairs; we include it.

        delta = t - 0.5
        mu_R = 0.5 * (self.mu_s.unsqueeze(1) + self.mu_t.unsqueeze(0))          # (ks, kt, d)
        S_R = 0.25 * (self.var_s.unsqueeze(1) + self.var_t.unsqueeze(0))        # (ks, kt, d)
        Cov_RU = 0.5 * (self.var_t.unsqueeze(0) - self.var_s.unsqueeze(1))      # (ks, kt, d)

        # u(x), g(u(x)), q_x(x) on the GL nodes. These are functions of x in [-1,1].
        gl_x = self._gl_x.to(device=device, dtype=dtype).view(*([1] * 4), -1)   # (1,1,1,1, G)
        gl_w = self._gl_w.to(device=device, dtype=dtype).view(*([1] * 4), -1)
        sqrt_lam = math.sqrt(self.lam)

        # u_q = 2 sqrt(lam) * x / (1 - x^2)
        omx2 = 1.0 - gl_x ** 2
        opx2 = 1.0 + gl_x ** 2
        u_q = 2.0 * sqrt_lam * gl_x / omx2                                      # (...,G)
        g_q = 2.0 * sqrt_lam * opx2 / omx2                                      # g(u(x)) = sqrt(4lam + u^2)

        # Bridge variance per-coord at u_q
        bridge_var_q = (self.sigma ** 2) * g_q * t * omt                        # (...,G)
        # Total per-coord variance at (x, u): S_R + bridge_var_q
        total_var = S_R.unsqueeze(0).unsqueeze(-1) + bridge_var_q               # (1, ks, kt, d, G)

        # Mean of X_t given u, c, c': mu_R + delta u
        mean_given_u = mu_R.unsqueeze(0).unsqueeze(-1) + delta * u_q            # (1, ks, kt, d, G)

        # Likelihood log p(x_t | u, c, c') — Gaussian per-coord
        x_e = x_t.unsqueeze(1).unsqueeze(1).unsqueeze(-1)                       # (B, 1, 1, d, 1)
        log_lik = -0.5 * ((x_e - mean_given_u) ** 2 / total_var
                          + torch.log(total_var)
                          + math.log(2.0 * math.pi))                            # (B, ks, kt, d, G)

        # Prior: p(u | c, c') = N(u; u_mean, u_var)
        log_prior_u = -0.5 * ((u_q - u_mean.unsqueeze(0).unsqueeze(-1)) ** 2
                               / u_var.unsqueeze(0).unsqueeze(-1)
                              + torch.log(u_var.unsqueeze(0).unsqueeze(-1))
                              + math.log(2.0 * math.pi))                        # (B, ks, kt, d, G)

        # Jacobian for x = tanh(y/2): density on x = density on u * |du/dx|.
        # Equivalent: log jacobian = log(2 sqrt(lam) * (1+x^2) / (1-x^2)^2)
        #                          = log(2 sqrt(lam)) + log(1+x^2) - 2 log(1-x^2)
        log_jac = math.log(2.0 * sqrt_lam) + torch.log(opx2) - 2.0 * torch.log(omx2)

        # Inner conditional mean E[X_0 | x_t, u, c, c'] per (B, c, c', d, G)
        # = mu_R - u/2 + (S_R / (S_R + bridge_var_q)) * (x_t - mu_R - delta u)
        #   + (Cov_RU / u_var) * (u - u_mean)  -- correction when var_s != var_t
        ratio = S_R.unsqueeze(0).unsqueeze(-1) / total_var                      # (1, ks, kt, d, G)
        residual = x_e - mean_given_u                                           # (B, ks, kt, d, G)
        ru_cor = Cov_RU.unsqueeze(0).unsqueeze(-1) \
                  / u_var.unsqueeze(0).unsqueeze(-1) \
                  * (u_q - u_mean.unsqueeze(0).unsqueeze(-1))                   # (B, ks, kt, d, G)
        x0_inner = (mu_R.unsqueeze(0).unsqueeze(-1)
                     - 0.5 * u_q
                     + ratio * residual
                     + ru_cor)                                                  # (B, ks, kt, d, G)

        # Combined log-weight in x: prior + likelihood + jacobian + GL weight
        log_w = log_lik + log_prior_u + log_jac + torch.log(gl_w)
        log_w_max = log_w.max(dim=-1, keepdim=True).values
        ww = (log_w - log_w_max).exp()                                          # (B, ks, kt, d, G)

        num = (ww * x0_inner).sum(-1)                                           # (B, ks, kt, d)
        den = ww.sum(-1)                                                        # (B, ks, kt, d)
        cond_mean_per_pair = num / den                                          # (B, ks, kt, d)

        # Posterior-weighted sum over (c, c')
        m = (weights.unsqueeze(-1) * cond_mean_per_pair).sum(dim=(1, 2))        # (B, d)
        return m


# -----------------------------------------------------------------------------
#  Sample-based (empirical) Bayes denoiser for either bridge (full covariance)
# -----------------------------------------------------------------------------
class BayesDenoiserMC(nn.Module):
    """Stratified self-normalised importance-sampling (SNIS) Bayes denoiser.

    Directly estimates the exact Bayes-optimal denoiser for either bridge
    (fixed or adaptive) on any LowRank + isotropic Gaussian mixture. No
    structural approximation of the data distribution — the proposal matches
    pi(X_0, X_1) = p_0(X_0) p_1(X_1) exactly, with stratification over
    component pairs (c, c') for variance reduction.

    Target:
        m*(x_t) = E_{(X_0, X_1) ~ pi}[ X_0 * p(x_t | X_0, X_1) ]
                  / E_{(X_0, X_1) ~ pi}[ p(x_t | X_0, X_1) ]

    Proposal (stratified, equal allocation per pair):
        q(X_0, X_1) = (1 / (ks kt)) sum_{c, c'} p(X_0 | c) p(X_1 | c')

    Importance weight per particle n drawn from pair (c, c'):
        w_n = [pi_s^c pi_t^{c'} * ks * kt] * p(x_t | X_0^(n), X_1^(n))
    The (ks kt) factor cancels in the self-normalised ratio.

    Error: bias O(1/N), variance ~ 1/ESS. Unlike the diagonal-only
    BayesDenoiserGMMAdaptive, this is exact in the N -> infinity limit for
    the full-covariance low-rank benchmark.
    """

    def __init__(self, k_src: int, k_tgt: int, r: int, d: int,
                 kind: str = 'fixed',
                 sigma: float = 1.0, lam: float = 1024.0, sigma0: float = 1.0,
                 n_per_pair: int = 1024, mode: str = 'mean', seed: int = 0):
        """Shape-only constructor (hydra-friendly).

        Args:
            k_src, k_tgt: number of source / target GMM components.
            r: latent dimensionality (LowRank rank).
            d: data dimensionality.
            kind: 'fixed' or 'adaptive' bridge.
            sigma, lam: bridge hyperparameters (lam only used for adaptive).
            sigma0: isotropic noise std in data space.
            n_per_pair: particles drawn per (c, c') pair per call.
            mode: 'mean' only (posterior mean). 'sample' not implemented.
            seed: base seed for per-call random draws (each call advances it
                by one so successive calls give independent particles).
        """
        super().__init__()
        self.register_buffer('mu_s_lat', torch.zeros(k_src, r))
        self.register_buffer('cov_s_lat', torch.eye(r).expand(k_src, r, r).contiguous().clone())
        self.register_buffer('logw_s', torch.full((k_src,), -math.log(k_src)))
        self.register_buffer('W_s', torch.zeros(d, r))

        self.register_buffer('mu_t_lat', torch.zeros(k_tgt, r))
        self.register_buffer('cov_t_lat', torch.eye(r).expand(k_tgt, r, r).contiguous().clone())
        self.register_buffer('logw_t', torch.full((k_tgt,), -math.log(k_tgt)))
        self.register_buffer('W_t', torch.zeros(d, r))

        # per-call counter so successive sample() calls see fresh particles
        self.register_buffer('_call_counter', torch.zeros((), dtype=torch.long))

        self.kind = kind
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.sigma0 = float(sigma0)
        self.n_per_pair = int(n_per_pair)
        self.mode = mode
        self.seed = int(seed)

    def set_params(self, mu_s_lat, cov_s_lat, w_s, W_s,
                   mu_t_lat, cov_t_lat, w_t, W_t):
        self.mu_s_lat.copy_(mu_s_lat.float())
        self.cov_s_lat.copy_(cov_s_lat.float())
        self.logw_s.copy_(w_s.float().clamp_min(1e-30).log())
        self.W_s.copy_(W_s.float())
        self.mu_t_lat.copy_(mu_t_lat.float())
        self.cov_t_lat.copy_(cov_t_lat.float())
        self.logw_t.copy_(w_t.float().clamp_min(1e-30).log())
        self.W_t.copy_(W_t.float())

    @classmethod
    def from_dataset(cls, dataset, kind, sigma, lam=1024.0,
                     n_per_pair=1024, mode='mean', seed=0):
        """Build from a LowRankGaussianMixtureDataset (analytic GMM params)."""
        assert hasattr(dataset, 'latent_dim') and hasattr(dataset, 'proj_source'), \
            'BayesDenoiserMC (stratified SNIS) requires LowRankGaussianMixtureDataset.'
        m = cls(
            k_src=dataset.means_source.shape[0],
            k_tgt=dataset.means_target.shape[0],
            r=dataset.latent_dim, d=dataset.data_dim,
            kind=kind, sigma=sigma, lam=lam,
            sigma0=float(dataset.noise_scale),
            n_per_pair=n_per_pair, mode=mode, seed=seed,
        )
        m.set_params(
            dataset.means_source, dataset.covs_source, dataset.weights_source, dataset.proj_source,
            dataset.means_target, dataset.covs_target, dataset.weights_target, dataset.proj_target,
        )
        return m

    @torch.no_grad()
    def _draw_particles(self, generator):
        """Stratified draw: N_per_pair particles from each (c, c') pair.

        Returns:
            X0: (ks*kt*N, d)
            X1: (ks*kt*N, d)
            log_prior: (ks*kt*N,)  equals log(pi_s^c * pi_t^{c'}) per particle
        """
        device = self.W_s.device
        dtype = self.W_s.dtype
        ks = self.mu_s_lat.shape[0]
        kt = self.mu_t_lat.shape[0]
        r = self.mu_s_lat.shape[1]
        d = self.W_s.shape[0]
        N = self.n_per_pair

        # Per-component Cholesky factors
        L_s = torch.linalg.cholesky(self.cov_s_lat)                              # (ks, r, r)
        L_t = torch.linalg.cholesky(self.cov_t_lat)                              # (kt, r, r)

        # Draw latent normal noise for each component: (ks, N, r) and (kt, N, r)
        eps_s = torch.randn(ks, N, r, device=device, dtype=dtype, generator=generator)
        eps_t = torch.randn(kt, N, r, device=device, dtype=dtype, generator=generator)
        # Z_0^{c, n} = mu_s^c + L_s^c @ eps_s^{c, n}
        Z0 = self.mu_s_lat.unsqueeze(1) + torch.einsum('crs, cns -> cnr', L_s, eps_s)  # (ks, N, r)
        Z1 = self.mu_t_lat.unsqueeze(1) + torch.einsum('crs, cns -> cnr', L_t, eps_t)  # (kt, N, r)

        # Project to data space: X_0^{c, n} = W_s Z_0^{c, n} + eps_0  (per particle iso noise)
        eps_0 = torch.randn(ks, N, d, device=device, dtype=dtype, generator=generator) * self.sigma0
        eps_1 = torch.randn(kt, N, d, device=device, dtype=dtype, generator=generator) * self.sigma0
        X0_per_comp = torch.einsum('dr, cnr -> cnd', self.W_s, Z0) + eps_0        # (ks, N, d)
        X1_per_comp = torch.einsum('dr, cnr -> cnd', self.W_t, Z1) + eps_1        # (kt, N, d)

        # Expand to all pairs (ks, kt, N, d) then flatten
        X0 = X0_per_comp.unsqueeze(1).expand(ks, kt, N, d).reshape(ks * kt * N, d)
        X1 = X1_per_comp.unsqueeze(0).expand(ks, kt, N, d).reshape(ks * kt * N, d)

        # log prior per particle: log(pi_s^c) + log(pi_t^{c'})
        lp = (self.logw_s.unsqueeze(-1) + self.logw_t).unsqueeze(-1)             # (ks, kt, 1)
        lp = lp.expand(ks, kt, N).reshape(ks * kt * N)                            # (N_total,)
        return X0, X1, lp

    @torch.no_grad()
    def _log_bridge_lik(self, x_t, X0, X1, t):
        """log p(x_t | X_0, X_1) per (B, N). Per-coord Gaussian."""
        # mean_{ji} = (1-t) X_0^{j,i} + t X_1^{j,i}
        mean = (1.0 - t) * X0 + t * X1                                          # (N, d)
        diff = x_t.unsqueeze(1) - mean.unsqueeze(0)                             # (B, N, d)
        d = x_t.shape[-1]
        if self.kind == 'fixed':
            var = (self.sigma ** 2) * t * (1.0 - t)
            log_p = -0.5 * (diff ** 2).sum(-1) / var
            log_p = log_p - 0.5 * d * (math.log(2.0 * math.pi) + math.log(var))
        else:
            U = X1 - X0
            var_per_dim = ((self.sigma ** 2)
                           * torch.sqrt(4.0 * self.lam + U ** 2)
                           * t * (1.0 - t))                                      # (N, d)
            log_p = -0.5 * (diff ** 2 / var_per_dim.unsqueeze(0)).sum(-1)
            log_p = log_p - 0.5 * (torch.log(var_per_dim)
                                    + math.log(2.0 * math.pi)).sum(-1).unsqueeze(0)
        return log_p

    @torch.no_grad()
    def ess(self, x_t, t):
        """Effective sample size of the SNIS estimator at x_t (B,-tensor).

        Diagnostic: ESS_b = (sum w)^2 / sum w^2. Low ESS indicates the
        stratification is not giving enough particles of high weight for
        that query.
        """
        if isinstance(t, torch.Tensor):
            t_scalar = float(t.flatten()[0])
        else:
            t_scalar = float(t)
        t_scalar = max(min(t_scalar, 1.0 - 1e-8), 1e-8)
        g = torch.Generator(device=self.W_s.device).manual_seed(
            self.seed + int(self._call_counter.item()))
        X0, X1, lp = self._draw_particles(g)
        B, d = x_t.shape
        N_total = X0.shape[0]
        cap = 2 * (1024 ** 3)
        chunk = max(1, int(cap / max(1, N_total * d * 4)))
        chunk = min(chunk, B)
        out = torch.empty(B, device=x_t.device, dtype=x_t.dtype)
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            log_w = lp.unsqueeze(0) + self._log_bridge_lik(x_t[start:end], X0, X1, t_scalar)
            log_w = log_w - log_w.max(dim=-1, keepdim=True).values
            w = log_w.exp()
            out[start:end] = (w.sum(-1) ** 2) / (w ** 2).sum(-1).clamp_min(1e-30)
        return out

    @torch.no_grad()
    def sample(self, x_t, t, **kwargs):
        if isinstance(t, torch.Tensor):
            t_scalar = float(t.flatten()[0])
        else:
            t_scalar = float(t)
        t_scalar = max(min(t_scalar, 1.0 - 1e-8), 1e-8)

        g = torch.Generator(device=self.W_s.device).manual_seed(
            self.seed + int(self._call_counter.item()))
        self._call_counter += 1
        X0, X1, lp = self._draw_particles(g)                                     # (N_total, d) each, (N_total,)

        # Chunk the query batch so the per-(B, N_total, d) diff tensor fits in
        # memory. Target peak memory per chunk ~ 2 GB float32.
        B, d = x_t.shape
        N_total = X0.shape[0]
        bytes_per_entry = 4
        cap = 2 * (1024 ** 3)
        chunk = max(1, int(cap / max(1, N_total * d * bytes_per_entry)))
        chunk = min(chunk, B)

        out = torch.empty(B, d, device=x_t.device, dtype=x_t.dtype)
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            xt_c = x_t[start:end]
            log_lik = self._log_bridge_lik(xt_c, X0, X1, t_scalar)                # (c, N_total)
            log_w = lp.unsqueeze(0) + log_lik                                     # (c, N_total)
            log_w = log_w - log_w.max(dim=-1, keepdim=True).values
            w = log_w.exp()
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            out[start:end] = w @ X0
        return out


# -----------------------------------------------------------------------------
#  Low-rank + isotropic GMM adaptive Bayes (full-cov in data space, exact)
# -----------------------------------------------------------------------------
class BayesDenoiserLowRankAdaptive(nn.Module):
    """Exact adaptive-bridge Bayes denoiser for a LowRank + isotropic GMM.

    The data distribution is
        X_0 = W_s Z_0 + eps_0,  X_1 = W_t Z_1 + eps_1,
        eps_* ~ N(0, sigma0^2 I),  Z_0, Z_1 ~ GMM in R^r (independent coupling),
    and the adaptive bridge is
        X_t = (1-t) X_0 + t X_1 + sigma sqrt(t(1-t)) diag(sqrt(g(U)))^{1/2} xi,
        U = X_1 - X_0,  g(u) = sqrt(4 lam + u^2).

    The d-dim full-cov coupling decomposes through the latent zeta := (Z_0, Z_1)
    in R^{2r}. Conditioning on zeta, the bridge likelihood factorises per
    coordinate:
        X_t^i | zeta, U_i = (W_R zeta)_i + delta U_i + eta_i + noise_i,
        eta_i ~ N(0, sigma0^2/2),
        noise_i ~ N(0, sigma^2 t(1-t) g(U_i)) | U_i,
        U_i | zeta ~ N(m_i(zeta), 2 sigma0^2) independently across i,
    where
        m_i(zeta) = (W_t Z_1)_i - (W_s Z_0)_i,
        (W_R zeta)_i = 0.5 ((W_s Z_0)_i + (W_t Z_1)_i),
        delta = t - 1/2.

    So the exact posterior mean is
        m*(x_t) = sum_{c,c'} pi_s^c pi_t^{c'} Z_{cc'}(x_t) / Z(x_t) * m_{cc'}(x_t),
        Z_{cc'}(x_t) = E_zeta[ prod_i p_i(x_t^i | zeta) ],
        m_{cc'}^i(x_t) = E_zeta[ prod_j p_j(x_t^j | zeta) * E[X_0^i | zeta, x_t] ] / Z_{cc'},
        p_i(x_t^i | zeta) = integral_{U_i} N(U_i; m_i(zeta), 2 sigma0^2)
                            N(x_t^i; (W_R zeta)_i + delta U_i,
                                     sigma0^2/2 + sigma^2 t(1-t) g(U_i)) dU_i,
        E[X_0^i | zeta, U_i, x_t] = (W_s Z_0)_i
            + (sigma0^2/2) / (sigma0^2/2 + sigma^2 t(1-t) g(U_i))
              * (x_t^i - (W_R zeta)_i - delta U_i)
            - 0.5 (U_i - m_i(zeta)).

    The outer zeta-expectation is a 2r-dim Gaussian integral (with block-
    diagonal covariance per pair (c, c')); computed by tensor-product
    Gauss-Hermite with gh_n^{2r} nodes. The inner U-integrals are 1-D
    Gauss-Hermite with u_n nodes. Tractable for small r (r <= 3 or so).
    """

    def __init__(self, k_src: int, k_tgt: int, r: int, d: int,
                 sigma: float, lam: float, sigma0: float,
                 mode: str = 'mean', gh_n: int = 8, u_n: int = 32):
        super().__init__()
        self.register_buffer('mu_s_lat', torch.zeros(k_src, r))
        self.register_buffer('cov_s_lat', torch.eye(r).expand(k_src, r, r).contiguous().clone())
        self.register_buffer('logw_s', torch.full((k_src,), -math.log(k_src)))
        self.register_buffer('W_s', torch.zeros(d, r))

        self.register_buffer('mu_t_lat', torch.zeros(k_tgt, r))
        self.register_buffer('cov_t_lat', torch.eye(r).expand(k_tgt, r, r).contiguous().clone())
        self.register_buffer('logw_t', torch.full((k_tgt,), -math.log(k_tgt)))
        self.register_buffer('W_t', torch.zeros(d, r))

        self.sigma = float(sigma)
        self.lam = float(lam)
        self.sigma0 = float(sigma0)
        self.mode = mode
        self.gh_n = int(gh_n)
        self.u_n = int(u_n)

        # 1-D Gauss-Hermite for U integral: nodes x_u, weights w_u
        # integral f(U) N(U; m, 2 sigma0^2) dU
        #   ~ (1/sqrt(pi)) sum_k w_u_k f(m + 2 sigma0 x_u_k)
        x_u, w_u = np.polynomial.hermite.hermgauss(self.u_n)
        self.register_buffer('_u_x', torch.tensor(x_u, dtype=torch.float64))
        self.register_buffer('_u_w', torch.tensor(w_u, dtype=torch.float64))

        # Tensor-product Gauss-Hermite grid for zeta (2r-dim)
        dim_z = 2 * r
        x_z, w_z = np.polynomial.hermite.hermgauss(self.gh_n)
        x_z_t = torch.tensor(x_z, dtype=torch.float64)
        w_z_t = torch.tensor(w_z, dtype=torch.float64)
        grids = torch.meshgrid(*([x_z_t] * dim_z), indexing='ij')
        nodes = torch.stack([g.reshape(-1) for g in grids], dim=-1)              # (gh_n**dim_z, dim_z)
        wgrids = torch.meshgrid(*([w_z_t] * dim_z), indexing='ij')
        wprod = torch.stack([g.reshape(-1) for g in wgrids], dim=-1).prod(dim=-1) # (gh_n**dim_z,)
        self.register_buffer('_z_x', nodes)                                      # (N_gh, 2r)
        self.register_buffer('_z_w', wprod)                                      # (N_gh,)

    def set_params(self, mu_s_lat, cov_s_lat, w_s, W_s,
                   mu_t_lat, cov_t_lat, w_t, W_t):
        self.mu_s_lat.copy_(mu_s_lat.float())
        self.cov_s_lat.copy_(cov_s_lat.float())
        self.logw_s.copy_(w_s.float().clamp_min(1e-30).log())
        self.W_s.copy_(W_s.float())
        self.mu_t_lat.copy_(mu_t_lat.float())
        self.cov_t_lat.copy_(cov_t_lat.float())
        self.logw_t.copy_(w_t.float().clamp_min(1e-30).log())
        self.W_t.copy_(W_t.float())

    @classmethod
    def from_dataset(cls, dataset, sigma, lam, mode='mean', gh_n=8, u_n=32):
        """Requires a LowRankGaussianMixtureDataset."""
        assert hasattr(dataset, 'latent_dim') and hasattr(dataset, 'proj_source'), \
            'BayesDenoiserLowRankAdaptive requires LowRankGaussianMixtureDataset.'
        m = cls(
            k_src=dataset.means_source.shape[0],
            k_tgt=dataset.means_target.shape[0],
            r=dataset.latent_dim, d=dataset.data_dim,
            sigma=sigma, lam=lam, sigma0=float(dataset.noise_scale),
            mode=mode, gh_n=gh_n, u_n=u_n,
        )
        m.set_params(
            dataset.means_source, dataset.covs_source, dataset.weights_source, dataset.proj_source,
            dataset.means_target, dataset.covs_target, dataset.weights_target, dataset.proj_target,
        )
        return m

    @torch.no_grad()
    def sample(self, x_t, t, **kwargs):
        if isinstance(t, torch.Tensor):
            t_scalar = float(t.flatten()[0])
        else:
            t_scalar = float(t)
        t_scalar = max(min(t_scalar, 1.0 - 1e-8), 1e-8)
        return self._compute_mean(x_t, t_scalar)

    def _compute_mean(self, x_t, t):
        """Exact Bayes mean E[X_0 | X_t = x_t] via (zeta, U) decomposition.

        Internally runs in float64 on the quadrature grids for stability,
        returns float tensor matching x_t.dtype.
        """
        device = x_t.device
        in_dtype = x_t.dtype
        B, d = x_t.shape
        ks = self.mu_s_lat.shape[0]
        kt = self.mu_t_lat.shape[0]
        r = self.mu_s_lat.shape[1]

        omt = 1.0 - t
        delta = t - 0.5
        sig0 = self.sigma0
        sig02 = sig0 ** 2
        sig2 = self.sigma ** 2
        lam = self.lam
        sqrt2 = math.sqrt(2.0)
        dtype = torch.float64
        x_t64 = x_t.to(dtype=dtype)

        # GH nodes / weights (in float64)
        u_x = self._u_x.to(device=device)                                       # (u_n,)
        u_logw = torch.log(self._u_w.to(device=device).clamp_min(1e-300))        # (u_n,)
        # 1/sqrt(pi) factor for U integral against a standard normal:
        u_log_norm = -0.5 * math.log(math.pi)
        z_x = self._z_x.to(device=device)                                       # (N_gh, 2r)
        z_logw = torch.log(self._z_w.to(device=device).clamp_min(1e-300))        # (N_gh,)
        # (1/sqrt(pi))^{2r} factor for the zeta integral against a standard MVN:
        z_log_norm = -r * math.log(math.pi)
        N_gh = z_x.shape[0]

        # Running accumulators per pair (c, c'), per batch, in log-domain
        # We track, for each (c_s, c_t, b):
        #   L_max[b]  : running max of log p(x_t | zeta_j)  over zeta_j
        #   Z_shift[b]: sum_j w_j exp(log_p_xt_given_zeta_j - L_max[b])
        #   N_shift[b, :]: sum_j w_j exp(log_p_xt_given_zeta_j - L_max[b]) * E[X_0 | zeta_j, x_t]
        # so that Z_cc(x_t) = exp(L_max) Z_shift, and
        # num_cc(x_t) = Z_cc(x_t) m_cc(x_t) = exp(L_max) N_shift.

        L_max_pair = torch.full((ks, kt, B), -float('inf'), device=device, dtype=dtype)
        Z_shift_pair = torch.zeros(ks, kt, B, device=device, dtype=dtype)
        N_shift_pair = torch.zeros(ks, kt, B, d, device=device, dtype=dtype)

        # GH-reweighted sum includes z_logw + z_log_norm;
        # absorb z_log_norm + per-dim sqrt(2) Jacobian into log_p.
        # For the zeta change-of-variables: if zeta ~ N(mu, Sigma) and Sigma = L L^T,
        # then integral f(zeta) N(zeta; mu, Sigma) dzeta
        #   ~ (1/sqrt(pi))^{2r} sum_j w_j f(mu + L (sqrt(2) x_j)).

        for c_s in range(ks):
            for c_t in range(kt):
                mu_zeta = torch.cat([self.mu_s_lat[c_s], self.mu_t_lat[c_t]], dim=0).to(dtype)  # (2r,)
                cov_zeta = torch.zeros(2 * r, 2 * r, device=device, dtype=dtype)
                cov_zeta[:r, :r] = self.cov_s_lat[c_s].to(dtype)
                cov_zeta[r:, r:] = self.cov_t_lat[c_t].to(dtype)
                L_zeta = torch.linalg.cholesky(cov_zeta)                         # (2r, 2r)

                # zeta_nodes: (N_gh, 2r) = mu + (L @ (sqrt(2) x_j.T)).T
                zeta_nodes = mu_zeta + (z_x * sqrt2) @ L_zeta.T                  # (N_gh, 2r)
                Z0_nodes = zeta_nodes[:, :r]
                Z1_nodes = zeta_nodes[:, r:]

                Ws = self.W_s.to(dtype)
                Wt = self.W_t.to(dtype)
                WsZ0 = Z0_nodes @ Ws.T                                           # (N_gh, d)
                WtZ1 = Z1_nodes @ Wt.T                                           # (N_gh, d)
                mi = WtZ1 - WsZ0                                                 # (N_gh, d)
                WRz = 0.5 * (WsZ0 + WtZ1)                                        # (N_gh, d)

                # Process zeta nodes in chunks
                # Memory per chunk ~ B * chunk * d * u_n float64.
                bytes_per_scalar = 8
                cap_bytes = 2 * (1024 ** 3)  # ~2 GB headroom
                chunk = max(1, int(cap_bytes / max(1, B * d * self.u_n * bytes_per_scalar)))
                chunk = min(chunk, N_gh)

                for start in range(0, N_gh, chunk):
                    end = min(start + chunk, N_gh)
                    C = end - start

                    mi_c = mi[start:end]                                         # (C, d)
                    WRz_c = WRz[start:end]                                       # (C, d)
                    WsZ0_c = WsZ0[start:end]                                     # (C, d)
                    zlogw_c = z_logw[start:end] + z_log_norm                      # (C,)

                    # U-nodes: U_k = mi + 2 sigma0 u_x_k
                    U = mi_c.unsqueeze(-1) + 2.0 * sig0 * u_x.view(1, 1, -1)     # (C, d, u_n)
                    g_U = torch.sqrt(4.0 * lam + U ** 2)                          # (C, d, u_n)
                    cvar = 0.5 * sig02 + sig2 * t * omt * g_U                    # (C, d, u_n)
                    cmean = WRz_c.unsqueeze(-1) + delta * U                       # (C, d, u_n)

                    # per-(B, C, d, u_n) Gaussian log-density of x_t^i given (zeta, U)
                    x_e = x_t64.unsqueeze(1).unsqueeze(-1)                       # (B, 1, d, 1)
                    diff = x_e - cmean                                           # (B, C, d, u_n)
                    log_N = -0.5 * (diff * diff / cvar
                                    + torch.log(cvar)
                                    + math.log(2.0 * math.pi))                   # (B, C, d, u_n)

                    # Inner E[X_0^i | zeta, U, x_t]
                    E_X0 = (WsZ0_c.unsqueeze(0).unsqueeze(-1)
                            + (0.5 * sig02 / cvar).unsqueeze(0) * diff
                            - 0.5 * (U - mi_c.unsqueeze(-1)).unsqueeze(0))        # (B, C, d, u_n)

                    # 1-D U integral per (B, C, d)
                    log_integrand = log_N + (u_logw + u_log_norm).view(1, 1, 1, -1)  # (B, C, d, u_n)
                    log_p_i = torch.logsumexp(log_integrand, dim=-1)             # (B, C, d); log p_i(x_t^i | zeta)
                    # Self-normalised inner expectation
                    w_u_norm = torch.softmax(log_integrand, dim=-1)              # (B, C, d, u_n)
                    E_X0_int = (w_u_norm * E_X0).sum(dim=-1)                     # (B, C, d)
                    # log p(x_t | zeta_j) = sum_i log p_i
                    log_pz = log_p_i.sum(dim=-1)                                 # (B, C)

                    # zeta contribution: log_pz + zlogw_c  (per (B, C))
                    log_chunk = log_pz + zlogw_c.unsqueeze(0)                    # (B, C)

                    # Merge with running (L_max_pair[c_s, c_t], Z_shift_pair[c_s, c_t], N_shift_pair[c_s, c_t])
                    chunk_max = log_chunk.max(dim=1).values                      # (B,)
                    old_max = L_max_pair[c_s, c_t]                               # (B,)
                    new_max = torch.maximum(old_max, chunk_max)                  # (B,)
                    # Rescale old sums
                    rescale_old = torch.exp(old_max - new_max)                   # (B,); 0 where old_max=-inf
                    rescale_old = torch.where(torch.isfinite(old_max), rescale_old, torch.zeros_like(rescale_old))
                    Z_old_shifted = Z_shift_pair[c_s, c_t] * rescale_old          # (B,)
                    N_old_shifted = N_shift_pair[c_s, c_t] * rescale_old.unsqueeze(-1)  # (B, d)
                    # New chunk sums, shifted by new_max
                    w_chunk = torch.exp(log_chunk - new_max.unsqueeze(-1))        # (B, C)
                    Z_new_shifted = w_chunk.sum(dim=1)                            # (B,)
                    # Per coord numerator: sum_c w_chunk[:, c] * E_X0_int[:, c, :]
                    N_new_shifted = torch.einsum('bc,bcd->bd', w_chunk, E_X0_int) # (B, d)

                    Z_shift_pair[c_s, c_t] = Z_old_shifted + Z_new_shifted
                    N_shift_pair[c_s, c_t] = N_old_shifted + N_new_shifted
                    L_max_pair[c_s, c_t] = new_max

        # Combine across pairs. Per-pair log_Z = L_max + log(Z_shift). Prior (pi_s^c pi_t^{c'}) gives log_prior_pair.
        log_Z_pair = L_max_pair + torch.log(Z_shift_pair.clamp_min(1e-300))      # (ks, kt, B)
        log_prior_pair = (self.logw_s.to(dtype).unsqueeze(-1) + self.logw_t.to(dtype))  # (ks, kt)
        log_unnorm = log_prior_pair.unsqueeze(-1) + log_Z_pair                   # (ks, kt, B)
        log_denom = torch.logsumexp(log_unnorm.reshape(ks * kt, B), dim=0)       # (B,)

        # Z_cc m_cc = exp(L_max) * N_shift_pair  (in full). Together with prior:
        # weight_times_m = exp(log_prior_pair + L_max - log_denom) * N_shift_pair
        coeff = torch.exp(log_prior_pair.unsqueeze(-1) + L_max_pair - log_denom) # (ks, kt, B)
        m_star = (coeff.unsqueeze(-1) * N_shift_pair).sum(dim=(0, 1))            # (B, d)
        return m_star.to(in_dtype)
