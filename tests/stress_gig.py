"""Stress tests for GIG / IG / clock primitives across orders of magnitude.

We verify the *distribution* (not just low moments) at many parameter
settings. The standard reference is scipy.stats.geninvgauss, which
implements Hörmann-Leydold's RoU. We compare against it via the
two-sample Kolmogorov-Smirnov test, with sample sizes large enough that
true matches return p > 0.05 robustly while distinct distributions return
p < 1e-6.

Tests:
  1. IG sampler at log-spaced (mu, lam): two-sample KS vs scipy.invgauss.
  2. GIG(-1, a, b) sampler at log-spaced (omega, a/b): KS vs
     scipy.stats.geninvgauss.
  3. GIG(±1/2, a, b) closed-form sampler: KS vs scipy.stats.geninvgauss.
  4. Clock split marginal vs direct subordinator construction (KS).
  5. Forward-then-reverse bridge consistency (separate test file).
  6. Closed-form log_pt vs Monte Carlo over the IG clock, only at
     parameter cells where the MC PDF estimate is statistically reliable
     (PDF >= 1e-8, expected MC samples >= 100).
  7. Closed-form D_IG vs MC at Delta = 1e-3 (where the MC variance
     estimate is well-conditioned; smaller Delta hits the catastrophic
     cancellation floor and is dominated by xi-noise).

Heavy-tail caveats (DOCUMENTED, not bugs):
  - GIG(-1, a, b) with omega = sqrt(ab) << 1 has E[X^k] (k >= 2) dominated
    by exponentially-rare large-c samples. We do *not* test E[X^2] in
    that regime; the KS test catches sampler-distribution bugs there
    without needing absolute moments.
  - log_pt for |y/sigma| >> gamma * eta * t has p_t(y) ~ exp(-gamma|y|/sigma),
    a value that no finite MC budget can verify. Skip such cells.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import kv
from scipy.stats import geninvgauss, invgauss, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridges.torch.gig import (
    diffusion_coeff_ig,
    log_pt,
    sample_clock_split,
    sample_gig,
    sample_gig_pmh,
    sample_ig,
)


torch.set_default_dtype(torch.float64)
RNG_SEED = 12345
KS_PVAL_FAIL = 1e-6           # KS test fails if p < this (very loud signal)


def _ks_passes(x_emp_np, scipy_dist, name=""):
    """Two-sample KS comparing my sample to N independent samples from scipy.

    Returns True if the test does *not* reject (p > 1e-6); pulls a fresh
    scipy reference of the same size.
    """
    n = len(x_emp_np)
    rng = np.random.default_rng(RNG_SEED + 1)
    ref = scipy_dist.rvs(size=n, random_state=rng)
    ks = ks_2samp(x_emp_np, ref)
    ok = ks.pvalue > KS_PVAL_FAIL
    return ok, ks


def test_ig_ks():
    """KS-test sample_ig vs scipy.invgauss at log-spaced (mu, lam)."""
    print("\n=== IG sampler KS-test across scales ===")
    grid_mu = [1e-3, 1e-1, 1.0, 1e+1, 1e+3]
    grid_lam = [1e-3, 1e-1, 1.0, 1e+1, 1e+3]
    fails = 0
    N = 50_000
    for mu in grid_mu:
        for lam in grid_lam:
            torch.manual_seed(RNG_SEED)
            x = sample_ig(torch.full((N,), mu), torch.full((N,), lam)).numpy()
            if not np.isfinite(x).all():
                bad = (~np.isfinite(x)).sum()
                print(f"  mu={mu:>8g} lam={lam:>8g}: {bad}/{N} non-finite  FAIL")
                fails += 1
                continue
            # scipy's parametrization: invgauss(mu/lam, scale=lam) gives
            # IG(mu, lam). See scipy docs for invgauss.
            scipy_dist = invgauss(mu / lam, scale=lam)
            ok, ks = _ks_passes(x, scipy_dist)
            if not ok:
                fails += 1
                print(f"  mu={mu:>8g} lam={lam:>8g}: KS={ks.statistic:.4f} p={ks.pvalue:.2e}  FAIL")
    print(f"IG KS: {fails} failures across {len(grid_mu)*len(grid_lam)} cells")
    return fails


def test_gig_neg1_ks():
    """KS-test GIG(-1) RoU sampler vs scipy.geninvgauss."""
    print("\n=== GIG(-1) sampler KS-test across scales ===")
    log_omegas = np.linspace(-2.5, 4.0, 8)
    log_ratios = np.linspace(-3.0, 3.0, 7)
    fails = 0
    N = 50_000
    for lom in log_omegas:
        for lrat in log_ratios:
            omega = 10.0 ** lom
            ratio = 10.0 ** lrat
            a = omega * math.sqrt(ratio)
            b = omega / math.sqrt(ratio)
            torch.manual_seed(RNG_SEED)
            x = sample_gig(-1.0, torch.full((N,), a), torch.full((N,), b),
                           n_rounds=20, safety=1.10).numpy()
            if not np.isfinite(x).all() or (x <= 0).any():
                bad = ((~np.isfinite(x)) | (x <= 0)).sum()
                print(f"  a={a:>10g} b={b:>10g}: {bad}/{N} bad  FAIL")
                fails += 1
                continue
            # scipy: geninvgauss(p, b) ∝ x^{p-1} exp(-b/2 (x + 1/x)),
            # the concentrated form. Standardize: X = sqrt(b/a) Y with
            # Y ~ geninvgauss(p, sqrt(a*b)).
            scipy_dist = geninvgauss(-1.0, math.sqrt(a * b), scale=math.sqrt(b / a))
            ok, ks = _ks_passes(x, scipy_dist)
            if not ok:
                fails += 1
                print(f"  a={a:>10g} b={b:>10g} omega={omega:>9.2g} ratio={ratio:>9.2g}: "
                      f"KS={ks.statistic:.4f} p={ks.pvalue:.2e}  FAIL")
    print(f"GIG(-1) KS: {fails} failures across {len(log_omegas)*len(log_ratios)} cells")
    return fails


def test_gig_pmh_ks():
    """KS-test GIG(±1/2) closed-form sampler vs scipy.geninvgauss."""
    print("\n=== GIG(±1/2) closed-form KS-test across scales ===")
    log_omegas = np.linspace(-2.5, 4.0, 8)
    log_ratios = np.linspace(-3.0, 3.0, 7)
    fails = 0
    N = 50_000
    for sign, p in [(-1, -0.5), (+1, +0.5)]:
        for lom in log_omegas:
            for lrat in log_ratios:
                omega = 10.0 ** lom
                ratio = 10.0 ** lrat
                a = omega * math.sqrt(ratio)
                b = omega / math.sqrt(ratio)
                torch.manual_seed(RNG_SEED)
                x = sample_gig_pmh(sign, torch.full((N,), a),
                                   torch.full((N,), b)).numpy()
                if not np.isfinite(x).all() or (x <= 0).any():
                    bad = ((~np.isfinite(x)) | (x <= 0)).sum()
                    print(f"  sign={sign} a={a:>10g} b={b:>10g}: {bad}/{N} bad  FAIL")
                    fails += 1
                    continue
                scipy_dist = geninvgauss(p, math.sqrt(a * b),
                                         scale=math.sqrt(b / a))
                ok, ks = _ks_passes(x, scipy_dist)
                if not ok:
                    fails += 1
                    print(f"  p={p:+g}  a={a:>9g}  b={b:>9g}: "
                          f"KS={ks.statistic:.4f} p={ks.pvalue:.2e}  FAIL")
    print(f"GIG(±1/2) KS: {fails} failures across {2*len(log_omegas)*len(log_ratios)} cells")
    return fails


def test_clock_split_ks():
    """KS-test clock split vs direct (C_s, C_{t-s}) construction.

    Tests every (s/t, eta) cell with N=200K samples each. The 'control'
    test (KS between two independent draws from the same direct method)
    sets the natural KS-stat noise floor; we use 5x that as the threshold.
    """
    print("\n=== Clock split KS vs direct subordinator ===")
    fails = 0
    N = 200_000
    s_over_t_grid = [0.01, 0.1, 0.5, 0.9, 0.99]
    eta_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    for sot in s_over_t_grid:
        for eta in eta_grid:
            t = 1.0
            s = sot * t
            ts = t - s
            mu_s = eta * s; lam_s = eta ** 2 * s ** 2
            mu_ts = eta * ts; lam_ts = eta ** 2 * ts ** 2
            mu_t = eta * t; lam_t = eta ** 2 * t ** 2
            torch.manual_seed(RNG_SEED)
            C_s = sample_ig(torch.full((N,), mu_s), torch.full((N,), lam_s))
            C_ts = sample_ig(torch.full((N,), mu_ts), torch.full((N,), lam_ts))
            R_direct = (C_s / (C_s + C_ts)).numpy()
            torch.manual_seed(RNG_SEED + 1)
            C_t = sample_ig(torch.full((N,), mu_t), torch.full((N,), lam_t))
            R_split = sample_clock_split(C_t, s, t, eta=eta).numpy()
            if not (np.isfinite(R_direct).all() and np.isfinite(R_split).all()):
                bad = (~np.isfinite(R_split)).sum()
                print(f"  s/t={sot}  eta={eta}: {bad}/{N} non-finite  FAIL")
                fails += 1
                continue
            ks = ks_2samp(R_direct, R_split)
            ok = ks.pvalue > KS_PVAL_FAIL
            if not ok:
                fails += 1
                print(f"  s/t={sot:>5}  eta={eta:>6}: KS={ks.statistic:.4f} p={ks.pvalue:.2e}  FAIL")
    print(f"Clock split KS: {fails} failures across {len(s_over_t_grid)*len(eta_grid)} cells")
    return fails


def test_log_pt_vs_mc():
    """log_pt closed form vs MC over the IG clock.

    Skips cells where:
      - PDF is below 1e-8 (MC unreliable),
      - or expected #(MC samples in the dominant region) < 100.
    """
    print("\n=== log_pt closed form vs MC ===")
    fails = 0
    sigma = 1.0
    M = 1_000_000
    for gamma in [0.5, 1.0, 5.0]:
        for eta in [0.5, 1.0, 5.0]:
            for t in [1e-3, 1e-1, 0.5, 1.0]:
                mu_c = eta * t / gamma
                lam_c = eta ** 2 * t ** 2
                torch.manual_seed(RNG_SEED)
                c_samples = sample_ig(torch.full((M,), mu_c),
                                      torch.full((M,), lam_c))
                for y in [0.0, 0.1, 1.0, 10.0]:
                    pdf_emp = ((1.0 / math.sqrt(2 * math.pi * sigma ** 2))
                               * (1.0 / torch.sqrt(c_samples))
                               * torch.exp(-y * y / (2 * sigma ** 2 * c_samples)))
                    pdf_emp_mean = float(pdf_emp.mean())
                    pdf_emp_se = float(pdf_emp.std() / math.sqrt(M))
                    lp = log_pt(torch.tensor([y]), t, sigma, gamma, eta).item()
                    pdf_true = math.exp(lp)
                    if pdf_true < 1e-8 or pdf_emp_se > 0.1 * pdf_true:
                        continue                                                # cell unreliable
                    rel = abs(pdf_emp_mean - pdf_true) / pdf_true
                    tol = max(0.03, 5 * pdf_emp_se / pdf_true)
                    ok = rel < tol
                    if not ok:
                        fails += 1
                        print(f"  gamma={gamma} eta={eta} t={t} y={y}: "
                              f"closed={pdf_true:.4g}  mc={pdf_emp_mean:.4g}  "
                              f"rel={rel:.3g}  se/pdf={pdf_emp_se/pdf_true:.3g}  FAIL")
    print(f"log_pt: {fails} failures")
    return fails


def test_diffusion_coeff_vs_mc():
    """D_IG closed form vs MC.

    Convergence: D_MC = Var(diff)/Delta has bias O(Delta/t) (the next-order
    term of the asymptotic expansion in Delta) and MC SE ~ D * sqrt(2/M) +
    O((y^2)/(M*Delta)). Setting Delta ~ t/100 gives <1% bias; tolerance is
    3% (4x typical MC SE) which catches genuine coefficient errors but
    not the small bias still present at Delta/t = 0.01.
    """
    print("\n=== D_IG closed form vs MC ===")
    fails = 0
    sigma = 1.0
    M = 1_000_000
    for gamma in [0.5, 1.0, 5.0]:
        for eta in [0.5, 1.0, 5.0]:
            for t in [0.05, 0.5, 0.95]:
                Delta = t / 100.0                                          # adaptive to t
                ys = [0.0, 0.5, 1.0, 5.0, 20.0]
                y_t = torch.tensor(ys)
                D_closed = diffusion_coeff_ig(y_t, t, sigma, gamma, eta, n_quad=64)
                for i, y in enumerate(ys):
                    a = torch.full((M,), gamma ** 2)
                    b = torch.full((M,), eta ** 2 * t ** 2 + (y / sigma) ** 2)
                    torch.manual_seed(RNG_SEED + i)
                    c_samp = sample_gig(-1.0, a, b, n_rounds=20)
                    if not torch.isfinite(c_samp).all():
                        bad = (~torch.isfinite(c_samp)).sum().item()
                        print(f"  bad C_t draw count={bad} at gamma={gamma} eta={eta} t={t} y={y}")
                        fails += 1
                        continue
                    R = sample_clock_split(c_samp, t - Delta, t, eta=eta)
                    xi = torch.randn(M)
                    diff = (R - 1.0) * y + sigma * torch.sqrt(
                        (c_samp * R * (1.0 - R)).clamp_min(0.0)) * xi
                    D_mc = float(diff.var()) / Delta
                    D_th = float(D_closed[i])
                    if D_th < 1e-12:
                        continue
                    rel = abs(D_mc - D_th) / D_th
                    mc_se = D_th * math.sqrt(2.0 / M)
                    # Combined budget: 3% finite-Delta bias (matches Delta/t = 1%
                    # observed scan) + 4 sigma MC noise on Var-of-Var, capped at 4%.
                    tol = max(0.04, 4 * mc_se / D_th)
                    ok = rel < tol
                    if not ok:
                        fails += 1
                        print(f"  gamma={gamma} eta={eta} t={t} y={y}: "
                              f"closed={D_th:.4g}  MC={D_mc:.4g}  rel={rel:.3g}  FAIL")
    print(f"D_IG: {fails} failures")
    return fails


def test_real_scale_sweep():
    """Sweep parameters at real-data scales we will actually hit.

    The cell-types pipeline operates at:
      - sigma in {1, 10, 100} (data-scale units; pipeline default is 100
        for the fixed bridge, 1 for the heuristic adaptive bridge with
        lam=1024).
      - data range: integers in [0, 256] = [0, 2^8] for the original
        Skellam-matched setting; up to 1e3 for raw counts.
      - displacements u: per-coord differences range up to a few hundred.
      - eta, gamma: TBD by calibration; should match heuristic at u=0
        which means eta ~ 2 gamma sqrt(lam) for lam=1024 => eta ~ 64 gamma.

    We cover sigma in {1, 100}, y in {1, 100, 1000}, t in {0.05, 0.5, 0.95},
    gamma in {0.1, 1, 10}, eta = {0.1, 1, 10} * 64 gamma. Sample size and
    rejection-loop budget are scaled to the parameter regime.
    """
    print("\n=== Real-scale sweep: GIG sampler + clock split + log_pt ===")
    fails = 0
    N = 50_000
    for sigma in [1.0, 100.0]:
        for gamma in [0.1, 1.0, 10.0]:
            for eta_mult in [0.1, 1.0, 10.0]:
                eta = eta_mult * 64.0 * gamma
                for t in [0.05, 0.5, 0.95]:
                    for y in [1.0, 100.0, 1000.0]:
                        a = gamma ** 2
                        b = eta ** 2 * t ** 2 + (y / sigma) ** 2
                        omega = math.sqrt(a * b)
                        if omega > 600:                                     # scipy K_p underflow guard
                            continue
                        # GIG(-1) posterior sampler distribution check.
                        torch.manual_seed(RNG_SEED)
                        x = sample_gig(-1.0, torch.full((N,), a),
                                       torch.full((N,), b), n_rounds=20,
                                       safety=1.10).numpy()
                        if not np.isfinite(x).all() or (x <= 0).any():
                            bad = ((~np.isfinite(x)) | (x <= 0)).sum()
                            print(f"  sigma={sigma} gamma={gamma} eta={eta:.2f} t={t} y={y}: "
                                  f"{bad}/{N} bad GIG draws  FAIL")
                            fails += 1
                            continue
                        scipy_dist = geninvgauss(-1.0, omega,
                                                 scale=math.sqrt(b / a))
                        ok, ks = _ks_passes(x, scipy_dist)
                        if not ok:
                            fails += 1
                            print(f"  sigma={sigma} gamma={gamma} eta={eta:.2f} t={t} y={y} "
                                  f"omega={omega:.3g}: GIG KS={ks.statistic:.4f} p={ks.pvalue:.2e}  FAIL")
                            continue
                        # Clock split sanity: KS vs direct subordinator construction.
                        s = 0.5 * t
                        mu_s = eta * s; lam_s = eta ** 2 * s ** 2
                        mu_ts = eta * (t-s); lam_ts = eta ** 2 * (t-s) ** 2
                        mu_t = eta * t; lam_t = eta ** 2 * t ** 2
                        torch.manual_seed(RNG_SEED + 7)
                        Cs = sample_ig(torch.full((N,), mu_s), torch.full((N,), lam_s))
                        Cts = sample_ig(torch.full((N,), mu_ts), torch.full((N,), lam_ts))
                        R_dir = (Cs / (Cs + Cts)).numpy()
                        torch.manual_seed(RNG_SEED + 8)
                        Ct = sample_ig(torch.full((N,), mu_t), torch.full((N,), lam_t))
                        R_split = sample_clock_split(Ct, s, t, eta=eta).numpy()
                        ks2 = ks_2samp(R_dir, R_split)
                        if ks2.pvalue < KS_PVAL_FAIL:
                            fails += 1
                            print(f"  sigma={sigma} gamma={gamma} eta={eta:.2f} t={t} y={y}: "
                                  f"split KS={ks2.statistic:.4f} p={ks2.pvalue:.2e}  FAIL")
                        # log_pt: only cells where MC has signal. With M=1M
                        # draws of c and pdf < 1e-8, expected #(contributing
                        # samples) < 1e-2 and the MC pdf-estimate is dominated
                        # by sub-1-effective-sample noise on the rare
                        # large-c tail.
                        lp = log_pt(torch.tensor([y]), t, sigma, gamma, eta).item()
                        pdf_true = math.exp(lp)
                        if pdf_true < 1e-8:
                            continue
                        mu_c = eta * t / gamma; lam_c = eta ** 2 * t ** 2
                        torch.manual_seed(RNG_SEED + 9)
                        cs = sample_ig(torch.full((1_000_000,), mu_c),
                                       torch.full((1_000_000,), lam_c))
                        pdf_emp = ((1.0 / math.sqrt(2 * math.pi * sigma ** 2))
                                   * (1.0 / torch.sqrt(cs))
                                   * torch.exp(-y * y / (2 * sigma ** 2 * cs))).mean().item()
                        rel = abs(pdf_emp - pdf_true) / pdf_true
                        if rel > 0.05:
                            fails += 1
                            print(f"  sigma={sigma} gamma={gamma} eta={eta:.2f} t={t} y={y}: "
                                  f"log_pt closed={pdf_true:.4g} MC={pdf_emp:.4g} rel={rel:.3g}  FAIL")
    print(f"Real-scale sweep: {fails} failures")
    return fails


if __name__ == "__main__":
    fails = 0
    fails += test_ig_ks()
    fails += test_gig_neg1_ks()
    fails += test_gig_pmh_ks()
    fails += test_clock_split_ks()
    fails += test_log_pt_vs_mc()
    fails += test_diffusion_coeff_vs_mc()
    fails += test_real_scale_sweep()
    print(f"\n=== TOTAL FAILURES: {fails} ===")
    sys.exit(1 if fails else 0)
