"""
Bayes-optimal denoisers for the GaussianBridge and the IG-clocked
ClockedGaussianBridge, under independent Gaussian-mixture coupling.

Plugs into the existing reverse samplers via the standard
    .sample(x_t, t, **z) -> Tensor
interface, just like a trained denoiser. Used for oracle ablations: given
the exact conditional mean (or a posterior sample) of X_0 given X_t, how
does the same reverse sampler perform? This isolates "integration
dynamics" from "function-fitting error".

Two classes:
  - BayesDenoiserGMMFixed: closed-form Bayes denoiser for the fixed-sigma
    Gaussian bridge with independent Gaussian-mixture coupling.
  - BayesDenoiserMC: stratified self-normalised importance-sampling (SNIS)
    Bayes denoiser. Supports kind='fixed' (Gaussian bridge) and
    kind='clocked' (IG-clocked Gaussian bridge). Exact in the
    N_per_pair -> infinity limit for arbitrary LowRank+isotropic GMM
    couplings, with no diagonal/structural approximation.
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


class BayesDenoiserMC(nn.Module):
    """Stratified self-normalised importance-sampling (SNIS) Bayes denoiser.

    Directly estimates the exact Bayes-optimal denoiser for either bridge
    (fixed Gaussian or IG-clocked Gaussian) on any LowRank + isotropic
    Gaussian mixture. No structural approximation of the data distribution
    — the proposal matches pi(X_0, X_1) = p_0(X_0) p_1(X_1) exactly, with
    stratification over component pairs (c, c') for variance reduction.

    Target:
        m*(x_t) = E_{(X_0, X_1) ~ pi}[ X_0 * p(x_t | X_0, X_1) ]
                  / E_{(X_0, X_1) ~ pi}[ p(x_t | X_0, X_1) ]

    Proposal (stratified, equal allocation per pair):
        q(X_0, X_1) = (1 / (ks kt)) sum_{c, c'} p(X_0 | c) p(X_1 | c')

    Importance weight per particle n drawn from pair (c, c'):
        w_n = [pi_s^c pi_t^{c'} * ks * kt] * p(x_t | X_0^(n), X_1^(n))
    The (ks kt) factor cancels in the self-normalised ratio.

    Error: bias O(1/N), variance ~ 1/ESS. Exact in the N -> infinity limit
    for the full-covariance low-rank GMM benchmark.
    """

    def __init__(self, k_src: int, k_tgt: int, r: int, d: int,
                 kind: str = 'fixed',
                 sigma: float = 1.0, sigma0: float = 1.0,
                 gamma: float = 1.0, nu: float = 64.0,
                 n_per_pair: int = 1024, mode: str = 'mean', seed: int = 0):
        """Shape-only constructor (hydra-friendly).

        Args:
            k_src, k_tgt: number of source / target GMM components.
            r: latent dimensionality (LowRank rank).
            d: data dimensionality.
            kind: 'fixed' (Gaussian bridge) or 'clocked' (IG-clocked).
            sigma: bridge noise scale.
            sigma0: isotropic noise std in data space.
            gamma, nu: IG-clock parameters (only used for kind='clocked').
                Default nu=64, gamma=1 matches the legacy heuristic
                adaptive bridge at u=0 with lam=1024.
            n_per_pair: particles drawn per (c, c') pair per call.
            mode: 'mean' only (posterior mean). 'sample' not implemented.
            seed: base seed for per-call random draws (each call advances it
                by one so successive calls give independent particles).
        """
        if kind not in ('fixed', 'clocked'):
            raise ValueError(f"kind must be 'fixed' or 'clocked', got {kind!r}")
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
        self.sigma0 = float(sigma0)
        self.gamma = float(gamma)
        self.nu = float(nu)
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
    def from_dataset(cls, dataset, kind, sigma,
                     gamma=1.0, nu=64.0,
                     n_per_pair=1024, mode='mean', seed=0):
        """Build from a LowRankGaussianMixtureDataset (analytic GMM params)."""
        assert hasattr(dataset, 'latent_dim') and hasattr(dataset, 'proj_source'), \
            'BayesDenoiserMC (stratified SNIS) requires LowRankGaussianMixtureDataset.'
        m = cls(
            k_src=dataset.means_source.shape[0],
            k_tgt=dataset.means_target.shape[0],
            r=dataset.latent_dim, d=dataset.data_dim,
            kind=kind, sigma=sigma,
            sigma0=float(dataset.noise_scale),
            gamma=gamma, nu=nu,
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
        """log p(x_t | X_0, X_1) per (B, N). Per-coord factorised.

        - 'fixed':   Gaussian bridge, Var = sigma^2 t(1-t).
        - 'clocked': IG-clocked bridge, log q_t(x_t | x_0, x_1) per coord
                     = log p_t(x_t - x_0) + log p_{1-t}(x_1 - x_t)
                     - log p_1(x_1 - x_0)
                     using the closed-form IG-clocked transition density
                     p_t(y) from bridges.torch.gig.log_pt.
        """
        if self.kind == 'fixed':
            mean = (1.0 - t) * X0 + t * X1                                      # (N, d)
            diff = x_t.unsqueeze(1) - mean.unsqueeze(0)                         # (B, N, d)
            d = x_t.shape[-1]
            var = (self.sigma ** 2) * t * (1.0 - t)
            log_p = -0.5 * (diff ** 2).sum(-1) / var
            log_p = log_p - 0.5 * d * (math.log(2.0 * math.pi) + math.log(var))
            return log_p
        # kind == 'clocked'
        from bridges.torch.gig import log_pt
        y_t = x_t.unsqueeze(1) - X0.unsqueeze(0)                                # (B, N, d)
        y_1mt = X1.unsqueeze(0) - x_t.unsqueeze(1)                              # (B, N, d)
        y_1 = (X1 - X0).unsqueeze(0)                                            # (1, N, d)
        log_pt_t = log_pt(y_t, t, self.sigma, self.gamma, self.nu)
        log_pt_1mt = log_pt(y_1mt, 1.0 - t, self.sigma, self.gamma, self.nu)
        log_pt_1 = log_pt(y_1, 1.0, self.sigma, self.gamma, self.nu)
        return (log_pt_t + log_pt_1mt - log_pt_1).sum(-1)                       # (B, N)

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


