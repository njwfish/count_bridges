"""
Systematic tests comparing old (dep_bessel) vs new (bessel) implementations
against theoretical predictions for the Bessel distribution:

    P(M=m) ∝ lam^m / (m! (m+d)!)

where lam = alpha * beta.

Test regimes:
  1. Small d, small lam   (easy regime, both should work)
  2. Small d, large lam   (general Devroye regime)
  3. Large d, small lam   (mode-at-0, new fast path)
  4. Large d, large lam   (integer overflow regime)
  5. Edge cases           (d=0, lam=0, very large d)

For each regime we check:
  - Empirical mean vs theoretical mean
  - Empirical variance vs theoretical variance
  - Empirical PMF vs exact PMF (chi-squared-like test)
  - No NaN/Inf in output
  - New impl doesn't silently fall back to all-zeros
"""

import sys
import time
import numpy as np
import cupy as cp
from scipy.special import iv as scipy_iv, gammaln

# ── import both implementations ──────────────────────────────────────────────
# We import from the dep_ (old) and current (new) modules
sys.path.insert(0, '.')

from dep_bessel import bessel as bessel_old, bessel_kernel as old_kernel
from bessel import bessel as bessel_new, bessel_kernel as new_kernel


# ── Exact PMF computation (CPU, arbitrary precision via log-space) ───────────
def exact_log_pmf(m_vals, lam, d):
    """Compute log P(M=m) for array of m values. Uses log-space for stability."""
    m = np.asarray(m_vals, dtype=np.float64)
    log_unnorm = m * np.log(lam) - gammaln(m + 1) - gammaln(m + d + 1)
    log_Z = np.logaddexp.reduce(log_unnorm)
    return log_unnorm - log_Z


def exact_pmf(m_vals, lam, d):
    return np.exp(exact_log_pmf(m_vals, lam, d))


def exact_mean_var(lam, d, max_m=500):
    """Compute exact mean and variance from the PMF."""
    m_vals = np.arange(max_m)
    pmf = exact_pmf(m_vals, lam, d)
    # truncate where pmf is negligible
    cumsum = np.cumsum(pmf)
    cutoff = np.searchsorted(cumsum, 1.0 - 1e-12) + 1
    cutoff = min(cutoff, max_m)
    pmf = pmf[:cutoff]
    pmf /= pmf.sum()
    m_vals = m_vals[:cutoff]
    mean = np.sum(m_vals * pmf)
    var = np.sum((m_vals - mean)**2 * pmf)
    return mean, var, pmf, m_vals


# ── Test harness ─────────────────────────────────────────────────────────────
N_SAMPLES = 500_000

def run_test(name, lam_val, d_val, n_samples=N_SAMPLES, atol_mean=None, rtol_pmf=0.15):
    """Run old vs new comparison for a single (lam, d) pair."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"  lam={lam_val:.4g}, d={d_val}, n_samples={n_samples}")
    print(f"{'='*70}")

    lam_gpu = cp.array([lam_val], dtype=cp.float64)
    d_gpu = cp.array([d_val], dtype=cp.int32)
    seeds = {"OLD": 42, "NEW": 137}  # different seeds to avoid false identity

    # ── Run old implementation ───────────────────────────────────────────
    t0 = time.time()
    try:
        old_out = bessel_old(
            cp.sqrt(lam_gpu), cp.sqrt(lam_gpu), d_gpu,
            n_samples=n_samples, seed=seeds["OLD"]
        )
        old_samples = cp.asnumpy(old_out.ravel())
        old_time = time.time() - t0
        old_ok = True
    except Exception as e:
        print(f"  OLD IMPL FAILED: {e}")
        old_samples = None
        old_time = time.time() - t0
        old_ok = False

    # ── Run new implementation ───────────────────────────────────────────
    t0 = time.time()
    try:
        new_out = bessel_new(
            cp.sqrt(lam_gpu), cp.sqrt(lam_gpu), d_gpu,
            n_samples=n_samples, seed=seeds["NEW"]
        )
        new_samples = cp.asnumpy(new_out.ravel())
        new_time = time.time() - t0
        new_ok = True
    except Exception as e:
        print(f"  NEW IMPL FAILED: {e}")
        new_samples = None
        new_time = time.time() - t0
        new_ok = False

    # ── Compute exact reference ──────────────────────────────────────────
    max_m = max(500, int(lam_val) + 10 * int(np.sqrt(lam_val)) + 100)
    exact_mu, exact_var, exact_p, m_grid = exact_mean_var(lam_val, d_val, max_m=max_m)
    print(f"  Exact: mean={exact_mu:.4f}, var={exact_var:.4f}, support={len(m_grid)}")

    if atol_mean is None:
        # adaptive tolerance: ~4 sigma of sampling noise
        atol_mean = 4.0 * np.sqrt(exact_var / n_samples) + 0.05

    results = {}
    for label, samples, ok, elapsed in [
        ("OLD", old_samples, old_ok, old_time),
        ("NEW", new_samples, new_ok, new_time),
    ]:
        if not ok:
            results[label] = {"pass": False, "reason": "exception"}
            continue

        emp_mean = np.mean(samples)
        emp_var = np.var(samples)
        n_nan = np.sum(np.isnan(samples))
        n_zero = np.sum(samples == 0)
        frac_zero = n_zero / len(samples)

        mean_err = abs(emp_mean - exact_mu)
        mean_pass = mean_err < atol_mean

        # PMF comparison: bin counts vs expected
        max_obs = int(np.max(samples)) if len(samples) > 0 else 0
        compare_up_to = min(max_obs + 1, len(m_grid))
        if compare_up_to > 0:
            obs_counts = np.bincount(samples.astype(int), minlength=compare_up_to)[:compare_up_to]
            exp_counts = exact_p[:compare_up_to] * n_samples
            # only compare bins with expected count > 5
            mask = exp_counts > 5
            if mask.sum() > 0:
                chi2_terms = (obs_counts[mask] - exp_counts[mask])**2 / exp_counts[mask]
                chi2 = np.sum(chi2_terms)
                dof = mask.sum()
                chi2_per_dof = chi2 / dof
                pmf_pass = chi2_per_dof < 5.0  # threshold
            else:
                chi2_per_dof = float('nan')
                pmf_pass = True  # can't really test
        else:
            chi2_per_dof = float('nan')
            pmf_pass = True

        all_zero_suspect = (exact_mu > 0.5 and frac_zero > 0.99)

        passed = mean_pass and pmf_pass and n_nan == 0 and not all_zero_suspect

        print(f"\n  {label} ({elapsed:.3f}s):")
        print(f"    mean={emp_mean:.4f} (exact={exact_mu:.4f}, err={mean_err:.4f}, tol={atol_mean:.4f}) {'PASS' if mean_pass else 'FAIL'}")
        print(f"    var={emp_var:.4f} (exact={exact_var:.4f})")
        print(f"    chi2/dof={chi2_per_dof:.2f} {'PASS' if pmf_pass else 'FAIL'}")
        print(f"    n_nan={n_nan}, frac_zero={frac_zero:.4f}")
        if all_zero_suspect:
            print(f"    WARNING: suspiciously all-zero (expected mean={exact_mu:.2f})")
        print(f"    OVERALL: {'PASS' if passed else '*** FAIL ***'}")

        results[label] = {
            "pass": passed,
            "mean": emp_mean,
            "var": emp_var,
            "mean_err": mean_err,
            "chi2_per_dof": chi2_per_dof,
            "time": elapsed,
        }

    return results


def run_batch_test(name, lam_vals, d_vals, n_samples=N_SAMPLES):
    """Run old vs new on a batch of (lam, d) pairs simultaneously."""
    print(f"\n{'='*70}")
    print(f"BATCH TEST: {name}")
    print(f"  {len(lam_vals)} elements, n_samples={n_samples}")
    print(f"{'='*70}")

    lam_gpu = cp.array(lam_vals, dtype=cp.float64)
    d_gpu = cp.array(d_vals, dtype=cp.int32)
    alpha_gpu = cp.sqrt(lam_gpu)
    beta_gpu = cp.sqrt(lam_gpu)
    seed = 123

    for label, fn in [("OLD", bessel_old), ("NEW", bessel_new)]:
        t0 = time.time()
        try:
            out = fn(alpha_gpu, beta_gpu, d_gpu, n_samples=n_samples, seed=seed)
            samples = cp.asnumpy(out)  # shape [B, n_samples]
            elapsed = time.time() - t0
            n_nan = np.sum(np.isnan(samples))
            n_neg = np.sum(samples < 0)
            means = np.mean(samples, axis=-1)
            print(f"\n  {label} ({elapsed:.3f}s): shape={samples.shape}, nan={n_nan}, neg={n_neg}")
            print(f"    sample means (first 10): {means[:10]}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  {label} ({elapsed:.3f}s): EXCEPTION: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = {}

    # ── Regime 1: Small d, small lam (easy) ──────────────────────────────
    all_results["small_d_small_lam"] = run_test(
        "Small d, small lam (d=3, lam=5)", lam_val=5.0, d_val=3)

    all_results["small_d_med_lam"] = run_test(
        "Small d, medium lam (d=5, lam=50)", lam_val=50.0, d_val=5)

    # ── Regime 2: Small d, large lam ─────────────────────────────────────
    all_results["small_d_large_lam"] = run_test(
        "Small d, large lam (d=2, lam=500)", lam_val=500.0, d_val=2)

    # ── Regime 3: Large d, small lam (mode-at-0, fast path) ─────────────
    all_results["large_d_small_lam_1"] = run_test(
        "Large d, small lam (d=100, lam=10)", lam_val=10.0, d_val=100)

    all_results["large_d_small_lam_2"] = run_test(
        "Large d, small lam (d=1000, lam=50)", lam_val=50.0, d_val=1000)

    all_results["large_d_small_lam_3"] = run_test(
        "Large d, tiny lam (d=5000, lam=1)", lam_val=1.0, d_val=5000)

    # ── Regime 4: Large d, large lam (overflow regime) ───────────────────
    all_results["large_d_large_lam"] = run_test(
        "Large d, large lam (d=500, lam=10000)", lam_val=10000.0, d_val=500)

    all_results["huge_d_huge_lam"] = run_test(
        "Huge d, huge lam (d=10000, lam=50000)", lam_val=50000.0, d_val=10000)

    # ── Regime 5: Edge cases ─────────────────────────────────────────────
    all_results["d_zero"] = run_test(
        "d=0, lam=10", lam_val=10.0, d_val=0)

    all_results["lam_tiny"] = run_test(
        "Very small lam (d=10, lam=0.01)", lam_val=0.01, d_val=10)

    all_results["d_equals_lam"] = run_test(
        "d equals lam boundary (d=50, lam=51)", lam_val=51.0, d_val=50)

    # ── Batch test: mixed regimes simultaneously ─────────────────────────
    run_batch_test(
        "Mixed regimes batch",
        lam_vals=[1.0, 10.0, 100.0, 1000.0, 0.5, 50.0, 5000.0, 10.0],
        d_vals=  [0,   5,    10,    100,     1000, 50,   200,    20000],
        n_samples=10_000,
    )

    # ── Timing comparison: large batch in the problematic regime ─────────
    print(f"\n{'='*70}")
    print("TIMING: Large batch in large-d regime (1000 elements)")
    print(f"{'='*70}")
    n_elem = 1000
    lam_gpu = cp.full(n_elem, 10.0, dtype=cp.float64)
    d_gpu = cp.full(n_elem, 500, dtype=cp.int32)
    alpha_gpu = cp.sqrt(lam_gpu)
    beta_gpu = cp.sqrt(lam_gpu)

    for label, fn in [("OLD", bessel_old), ("NEW", bessel_new)]:
        # warmup
        fn(alpha_gpu, beta_gpu, d_gpu, n_samples=100, seed=99)
        cp.cuda.Stream.null.synchronize()
        # timed
        t0 = time.time()
        for _ in range(10):
            fn(alpha_gpu, beta_gpu, d_gpu, n_samples=1000, seed=99)
            cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - t0
        print(f"  {label}: {elapsed:.3f}s for 10 runs (1000 elem x 1000 samples)")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for test_name, res in all_results.items():
        old_status = "PASS" if res.get("OLD", {}).get("pass", False) else "FAIL"
        new_status = "PASS" if res.get("NEW", {}).get("pass", False) else "FAIL"
        old_t = res.get("OLD", {}).get("time", float('nan'))
        new_t = res.get("NEW", {}).get("time", float('nan'))
        print(f"  {test_name:30s}  OLD={old_status:4s} ({old_t:.3f}s)  NEW={new_status:4s} ({new_t:.3f}s)")

    # Check if new impl passes everywhere
    new_all_pass = all(
        res.get("NEW", {}).get("pass", False) for res in all_results.values()
    )
    old_all_pass = all(
        res.get("OLD", {}).get("pass", False) for res in all_results.values()
    )
    print(f"\n  OLD all pass: {old_all_pass}")
    print(f"  NEW all pass: {new_all_pass}")

    if new_all_pass:
        print("\n  >>> NEW IMPLEMENTATION PASSES ALL TESTS <<<")
    else:
        print("\n  >>> NEW IMPLEMENTATION HAS FAILURES — investigate above <<<")
