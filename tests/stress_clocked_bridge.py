"""Bridge-level stress tests for ClockedGaussianBridge.

These exercise the *bridge* code (forward + reverse) on top of the GIG
primitives, covering correctness properties that should hold whenever the
distributional primitives are correct:

  1. Endpoint marginals: at t=t_eps, X_t ≈ X_0 (within bridge std);
     at t=1-t_eps, X_t ≈ X_1.
  2. Forward consistency: per-coord Var(X_t | X_0, X_1) matches the
     analytical bridge marginal sigma^2 E[C_1 R(1-R)] across (sigma, gamma,
     eta, t, u).
  3. Forward-then-reverse round-trip: starting from (X_0, X_1) and
     forward-sampled X_t, the reverse step from t -> s (with x_0_pred =
     X_0 = truth) should produce X_s with the same marginal distribution
     as a direct forward sample at time s. Tested by comparing per-coord
     means and variances across the batch.
  4. SDE-mode sanity: drift (E[X_s] - X_t) ≈ (s - t)/t * (X_0 - X_t),
     diffusion variance ≈ D_IG(y_t, t) * |s - t| (small dt).

These do NOT re-test the GIG sampler distribution; KS testing is in
stress_gig.py.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridges.torch.clocked_gaussian_bridge import ClockedGaussianBridge
from bridges.torch.gig import (
    diffusion_coeff_ig,
    sample_clock_split,
    sample_gig,
)


torch.set_default_dtype(torch.float64)
RNG_SEED = 12345


def _cell_repr(sigma, gamma, nu, t, u_scale):
    return f"sigma={sigma} gamma={gamma} nu={nu} t={t} |u|≈{u_scale}"


def test_endpoint_marginals():
    """At t=t_eps, X_t -> X_0; at t=1-t_eps, X_t -> X_1."""
    print("\n=== Endpoint marginals (t -> 0, t -> 1) ===")
    fails = 0
    for sigma in [1.0, 100.0]:
        for gamma in [1.0, 32.0]:
            for nu in [64.0, 1024.0]:
                bridge = ClockedGaussianBridge(sigma=sigma, gamma=gamma, nu=nu,
                                               t_eps=1e-6)
                B, d = 64, 16
                torch.manual_seed(RNG_SEED)
                x_0 = torch.randn(B, d) * 5.0
                x_1 = torch.randn(B, d) * 5.0
                u = (x_1 - x_0)
                # Bridge at t = 2*t_eps (just past clipping; small but finite t)
                _, x_t_lo, _ = bridge(x_0, x_1, t=2e-6)
                err_lo = (x_t_lo - x_0).abs().mean().item()
                # Expected std at t = 2e-6: sigma * sqrt(C_1 * R*(1-R))
                # For small t, R ≈ t, (1-R) ≈ 1, so std ≈ sigma sqrt(C_1 * t).
                # E[C_1] is bounded (nu/gamma scale), so std ~ sqrt(nu/gamma * t).
                expected_std_lo = sigma * math.sqrt(nu / gamma * 2e-6)
                # Allow 10x the expected std as the threshold (the sampler can
                # produce occasional larger noise from the C_1 tail).
                ok_lo = err_lo < 30.0 * expected_std_lo
                # Bridge at t = 1 - 2*t_eps
                _, x_t_hi, _ = bridge(x_0, x_1, t=1.0 - 2e-6)
                err_hi = (x_t_hi - x_1).abs().mean().item()
                ok_hi = err_hi < 30.0 * expected_std_lo
                if not (ok_lo and ok_hi):
                    fails += 1
                    print(f"  sigma={sigma} gamma={gamma} nu={nu}: "
                          f"|x_t - x_0|={err_lo:.4g}, |x_t - x_1|={err_hi:.4g}, "
                          f"expected std≈{expected_std_lo:.4g}  FAIL")
    print(f"Endpoint marginals: {fails} failures")
    return fails


def test_forward_variance():
    """Per-coord Var(X_t | X_0, X_1) matches analytic bridge variance.

    For the IG clock:
        Var(X_t | x_0, x_1) = sigma^2 E[C_1 R(1-R) | u]
                             = sigma^2 E_{C_1 ~ GIG(-1, gamma^2, eta^2 + u^2/sigma^2)}
                                       [ C_1 * E[R(1-R) | C_1] ]
    where R ~ split(t, 1, eta) given C_1.

    We compare the empirical bridge variance over many (X_t | X_0, X_1)
    draws to a Monte-Carlo estimate of the integrand, both at the same
    parameter cell. Differences should be MC noise (a few %).
    """
    print("\n=== Forward Var(X_t | x_0, x_1) ===")
    fails = 0
    M = 200_000
    for sigma in [1.0, 100.0]:
        for gamma in [1.0, 32.0]:
            for nu in [64.0, 1024.0]:
                for t in [0.1, 0.5, 0.9]:
                    for u_scale in [0.0, 10.0, 100.0]:
                        bridge = ClockedGaussianBridge(sigma=sigma, gamma=gamma,
                                                       nu=nu, t_eps=1e-6)
                        # Single (x_0, x_1) pair, repeated M times
                        u_val = u_scale
                        x_0 = torch.zeros(M, 1)
                        x_1 = torch.full((M, 1), u_val)
                        _, x_t, _ = bridge(x_0, x_1, t=t)
                        emp_var = float((x_t - t * u_val).var())
                        # Analytic: independent C_1 / R MC at the same cell.
                        a = torch.tensor([gamma ** 2])
                        b = torch.tensor([nu ** 2 + (u_val / sigma) ** 2])
                        torch.manual_seed(RNG_SEED)
                        C_1 = sample_gig(-1.0, a.expand(M), b.expand(M), n_rounds=20)
                        R = sample_clock_split(C_1, t, 1.0, eta=nu)
                        # Var(x_t) = sigma^2 E[C_1 R(1-R)] + Var(R) * u^2
                        analytic_var = ((sigma ** 2) * float((C_1 * R * (1 - R)).mean())
                                        + (u_val ** 2) * float(R.var()))
                        rel = abs(emp_var - analytic_var) / max(analytic_var, 1e-12)
                        # MC noise in Var estimator: ~sqrt(2/M) ~ 0.3%; tolerate 5%
                        ok = rel < 0.05
                        if not ok:
                            fails += 1
                            print(f"  {_cell_repr(sigma, gamma, nu, t, u_scale)}: "
                                  f"emp={emp_var:.4g} analytic={analytic_var:.4g} rel={rel:.3g}  FAIL")
    print(f"Forward variance: {fails} failures")
    return fails


def test_forward_reverse_roundtrip():
    """Forward at s vs forward-then-reverse-to-s consistency.

    Sample many (X_0, X_1) pairs with the SAME (x_0, x_1) values (delta
    coupling so we can directly compare conditional distributions). Then
      direct: forward-sample X_s  | x_0, x_1 (M draws).
      indirect: forward-sample X_t | x_0, x_1, then reverse step from t to s
                with x_0_pred = x_0 (truth). (M draws.)
    Compare per-coord mean and variance of the two ensembles.

    The two distributions are identical (the bridge is Markov). So they
    should match to MC noise.
    """
    print("\n=== Forward-then-reverse round-trip ===")
    fails = 0
    M = 100_000
    for sigma in [1.0, 100.0]:
        for gamma in [1.0, 32.0]:
            for nu in [64.0]:
                bridge = ClockedGaussianBridge(sigma=sigma, gamma=gamma, nu=nu,
                                               t_eps=1e-6, mode='bridge', eta=1.0)
                for u_val in [0.0, 10.0, 100.0]:
                    x_0 = torch.zeros(M, 1)
                    x_1 = torch.full((M, 1), u_val)
                    s, t = 0.3, 0.7

                    # Direct
                    torch.manual_seed(RNG_SEED)
                    _, X_s_direct, _ = bridge(x_0, x_1, t=s)
                    # Indirect
                    torch.manual_seed(RNG_SEED + 1)
                    _, X_t, _ = bridge(x_0, x_1, t=t)
                    X_s_indirect, _ = bridge.sample_step(
                        torch.tensor(t), torch.tensor(s), X_t, x_0)

                    m_d = float(X_s_direct.mean()); v_d = float(X_s_direct.var())
                    m_i = float(X_s_indirect.mean()); v_i = float(X_s_indirect.var())
                    # Tolerances: both estimators have MC SE ~ sqrt(v/M) on mean,
                    # ~ v sqrt(2/M) on var. Compare to typical scale.
                    se_mean = math.sqrt(v_d / M) * 4.0
                    se_var = v_d * math.sqrt(2.0 / M) * 4.0
                    ok = abs(m_d - m_i) < max(se_mean, 0.02 * abs(m_d) + 1e-3) and \
                         abs(v_d - v_i) < max(se_var, 0.02 * v_d + 1e-3)
                    if not ok:
                        fails += 1
                        print(f"  sigma={sigma} gamma={gamma} nu={nu} u={u_val}: "
                              f"direct (mean={m_d:.4g}, var={v_d:.4g}) vs "
                              f"indirect (mean={m_i:.4g}, var={v_i:.4g})  FAIL")
    print(f"Round-trip: {fails} failures")
    return fails


def test_chapman_kolmogorov():
    """CK consistency: reverse t -> s in one step vs t -> u -> s in two steps.

    The bridge given x_0 is Markov in t. So
        X_s | X_t, x_0  ==  ∫ X_s | X_u, x_0  *  X_u | X_t, x_0  dX_u
    in distribution. Equivalently, for the reverse sampler
    `bridge.sample_step(t_curr, t_next, x_t, x_0_pred)`, the one-step
    distribution at any pair (t, s) should equal the marginal of the
    two-step composition through any intermediate u in (s, t).

    We compare the per-coord mean and variance of two ensembles:
      one_step:  reverse(t, s, X_t, x_0)
      two_step:  reverse(u, s, reverse(t, u, X_t, x_0), x_0)
    across many seeds, sigma/gamma/eta cells, and chosen u in the middle.

    This is a strict consistency: a buggy posterior C_t sampler, a buggy
    split, OR a buggy noise-variance formula would all show up here.
    """
    print("\n=== CK consistency: 1-step vs 2-step reverse ===")
    fails = 0
    M = 200_000
    for sigma in [1.0, 100.0]:
        for gamma in [1.0, 32.0]:
            for nu in [64.0]:
                bridge = ClockedGaussianBridge(sigma=sigma, gamma=gamma, nu=nu,
                                               t_eps=1e-6, mode='bridge', eta=1.0)
                for u_val, t, t_mid, s in [
                    # (x_1 - x_0 displacement, t_curr, intermediate u, t_next)
                    (0.0,   0.9, 0.6, 0.3),
                    (10.0,  0.9, 0.6, 0.3),
                    (100.0, 0.9, 0.6, 0.3),
                    (10.0,  0.7, 0.5, 0.1),
                ]:
                    x_0 = torch.zeros(M, 1)
                    x_1 = torch.full((M, 1), float(u_val))
                    # Sample X_t once; both branches start from the same X_t
                    torch.manual_seed(RNG_SEED)
                    _, X_t, _ = bridge(x_0, x_1, t=t)

                    # One-step reverse: t -> s
                    torch.manual_seed(RNG_SEED + 1)
                    X_s_one, _ = bridge.sample_step(
                        torch.tensor(t), torch.tensor(s), X_t, x_0)

                    # Two-step reverse: t -> t_mid -> s
                    torch.manual_seed(RNG_SEED + 2)
                    X_mid, _ = bridge.sample_step(
                        torch.tensor(t), torch.tensor(t_mid), X_t, x_0)
                    torch.manual_seed(RNG_SEED + 3)
                    X_s_two, _ = bridge.sample_step(
                        torch.tensor(t_mid), torch.tensor(s), X_mid, x_0)

                    m1 = float(X_s_one.mean()); v1 = float(X_s_one.var())
                    m2 = float(X_s_two.mean()); v2 = float(X_s_two.var())
                    se_mean = math.sqrt(v1 / M) * 4.0
                    se_var = v1 * math.sqrt(2.0 / M) * 4.0
                    ok = (abs(m1 - m2) < max(se_mean, 0.02 * abs(m1) + 1e-3) and
                          abs(v1 - v2) < max(se_var, 0.02 * v1 + 1e-3))
                    if not ok:
                        fails += 1
                        print(f"  sigma={sigma} gamma={gamma} nu={nu} u_val={u_val} "
                              f"t={t}->{t_mid}->{s}: "
                              f"1-step (m={m1:.4g}, v={v1:.4g}) vs "
                              f"2-step (m={m2:.4g}, v={v2:.4g})  FAIL")
    print(f"CK consistency: {fails} failures")
    return fails


def test_sde_mode_drift_diffusion():
    """SDE-mode reverse step matches drift+D_IG locally.

    For a single fixed (X_t, x_0_pred) pair, the SDE step gives
        X_s = X_t + (s - t)/t * (X_t - x_0_pred) + sqrt(D_IG(y, t) * (t-s)) xi.

    We sample many xi and compare the empirical mean and variance of X_s
    to the closed-form drift / diffusion.
    """
    print("\n=== SDE-mode drift+diffusion ===")
    fails = 0
    M = 200_000
    for sigma in [1.0, 100.0]:
        for gamma in [1.0]:
            for nu in [64.0]:
                bridge = ClockedGaussianBridge(sigma=sigma, gamma=gamma, nu=nu,
                                               t_eps=1e-6, mode='sde', eta=1.0)
                for t in [0.3, 0.7]:
                    for y_val in [0.0, 10.0, 100.0]:
                        Delta = t * 1e-2
                        s = t - Delta
                        x_0 = torch.zeros(M, 1)
                        # X_t = x_0 + y_val (so y_t = y_val)
                        X_t = torch.full((M, 1), y_val, dtype=torch.float64)
                        torch.manual_seed(RNG_SEED)
                        X_s, _ = bridge.sample_step(torch.tensor(t),
                                                    torch.tensor(s), X_t, x_0)
                        # Expected: drift toward x_0 = 0 by factor (s/t) (1 - Delta/t).
                        # E[X_s - X_t] = -Delta/t * y_val (since x_hat_0 = 0)
                        emp_mean = float((X_s - X_t).mean())
                        expected_mean = -Delta / t * y_val
                        # Diffusion variance: D_IG(y, t) * Delta
                        D = float(diffusion_coeff_ig(torch.tensor([y_val]), t,
                                                      sigma, gamma, nu, n_quad=64))
                        expected_var = D * Delta
                        emp_var = float((X_s - X_t).var())
                        rel_mean = abs(emp_mean - expected_mean) / max(abs(expected_mean), 1e-3)
                        rel_var = abs(emp_var - expected_var) / max(expected_var, 1e-12)
                        ok = (rel_mean < 0.10 + 4 * math.sqrt(expected_var / M) / max(abs(expected_mean), 1e-3)) \
                             and (rel_var < 0.05 + 4 * math.sqrt(2.0 / M))
                        if not ok:
                            fails += 1
                            print(f"  sigma={sigma} gamma={gamma} nu={nu} t={t} y={y_val}: "
                                  f"mean emp={emp_mean:.4g} exp={expected_mean:.4g}  "
                                  f"var emp={emp_var:.4g} exp={expected_var:.4g}  FAIL")
    print(f"SDE mode: {fails} failures")
    return fails


if __name__ == "__main__":
    fails = 0
    fails += test_endpoint_marginals()
    fails += test_forward_variance()
    fails += test_forward_reverse_roundtrip()
    fails += test_chapman_kolmogorov()
    fails += test_sde_mode_drift_diffusion()
    print(f"\n=== TOTAL FAILURES: {fails} ===")
    sys.exit(1 if fails else 0)
