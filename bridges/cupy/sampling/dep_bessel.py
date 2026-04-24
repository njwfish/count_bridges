import cupy as cp


from cupy.random import _kernels as _rk

#=== CUDA‑only preamble: RNG state, loggam, etc. ===
_preamble = (
    "#define M_PI 3.14159265358979323846\n"
  + _rk.rk_basic_definition
  + _rk.loggam_definition
  + _rk.long_min_max_definition
  + _rk.rk_standard_exponential_definition
)

#=== CUDA kernel source ===
_kernel_source = r'''
// device besselI_int: integer-order I_d(x)
extern "C" __device__
double besselI_int(int d, double x) {
    if (x == 0.0) {
        return (d == 0 ? 1.0 : 0.0);
    }
    // threshold: use series if x <= max(200, d)
    double thresh = fmax(200.0, (double)d);
    if (x <= thresh) {
        // power-series: I_d(x) = sum_{k=0..K} (x/2)^{2k+d}/(k!(k+d)!)
        int K = max(d, int(x)) + 20;
        double halfx = 0.5 * x;
        double term = exp(d * log(halfx) - loggam((double)d + 1.0));
        double sum = term;
        for (int k = 1; k <= K; ++k) {
            term *= (halfx * halfx) / double(k * (k + d));
            sum += term;
            if (term < 1e-16 * sum) break;
        }
        return sum;
    } else {
        // asymptotic expansion for large x relative to d
        double mu = double(4 * d * d);
        double inv8x = 1.0 / (8.0 * x);
        double s_asym = 1.0;
        // first three correction terms
        double t1 = -(mu - 1.0) * inv8x;                           // O(1/x)
        double t2 = (mu - 1.0) * (mu - 9.0) * inv8x * inv8x * 0.5;   // O(1/x^2)
        double t3 = -(mu - 1.0) * (mu - 9.0) * (mu - 25.0)
                     * inv8x * inv8x * inv8x / 6.0;               // O(1/x^3)
        s_asym += t1 + t2 + t3;
        double pref = exp(x) / sqrt(2.0 * M_PI * x);
        return pref * s_asym;
    }
}

extern "C" __global__
void bessel_devroye(
    const double * __restrict__ lam,    // alpha * beta per element
    const int    * __restrict__ d_arr,  // difference d per element
    unsigned long long seed,            // RNG seed
    int           n_samp,               // draws per element
    int          * __restrict__ out     // length = D*n_samp
) {
    int idx = blockIdx.x;
    if (threadIdx.x != 0) return;

    rk_state st;
    rk_seed(seed + (unsigned long long)idx, &st);

    double lam_i = lam[idx];
    int    di    = d_arr[idx];
    double ai    = 2.0 * sqrt(lam_i);
    int m0       = (int)floor((sqrt(4.0 * lam_i + double(di*di)) - di) / 2.0);

    double Iv    = besselI_int(di, ai);
    double logp0 = ((m0 + 0.5*di) * log(lam_i)
                   - (loggam(double(m0)+1.0) + loggam(double(m0+di)+1.0))
                   - log(Iv));
    double p0    = exp(logp0);
    double w     = 1.0 + p0/2.0;
    double wd    = 2.0 * w / p0;
    // printf("logp0: %f\n", logp0);

    int *row = out + idx * n_samp;
    // if log_p0 is inf or nan set row to 0 and done to True
    if (isinf(logp0) || isnan(logp0)) {
        for (int s = 0; s < n_samp; ++s) {
            row[s] = 0;
        }
        return;
    }

    for (int s = 0; s < n_samp; ++s) {
        int mp;
        bool ok = false;
        while (!ok) {
            double U = rk_double(&st);
            bool inb = (U < w / (1.0 + w));
            double Y  = inb
                       ? (rk_double(&st) - 0.5) * wd
                       : (w + rk_standard_exponential(&st)) / p0;
            if (rk_double(&st) < 0.5) Y = -Y;
            int k    = (int)rint(Y);
            mp       = m0 + k;
            if (mp < 0) continue;

            double lr = k * log(lam_i)
                      - ((loggam(double(mp)+1.0) + loggam(double(mp+di)+1.0))
                         - (loggam(double(m0)+1.0) + loggam(double(m0+di)+1.0)));
            double delta = p0 * fabs(Y) - w;
            double la    = lr - fmax(0.0, delta);
            if (rk_double(&st) < exp(la)) ok = true;
        }
        row[s] = mp;
    }
}
'''

#=== Compile RawKernel at runtime ===
bessel_kernel = cp.RawKernel(_preamble + _kernel_source.replace('bessel_devroye', 'bessel_devroye_old'), 'bessel_devroye_old')

#=== Python wrapper ===
def bessel(alpha, beta, d, n_samples=None, seed=None):
    """
    Devroye’s exact O(1)-time rejection sampler for
    P[M=m | B1−D1=d] ∝ (αβ)^{m + d/2}/(m!(m+d)!) / I_d(2√(αβ)).
    
    Args:
        alpha: [B] or [B,1] or [B,D]
        beta:  [B] or [B,1] or [B,D]
        d:     [B] or [B,1] or [B,D]
        n_samples: int, optional
        seed: int, optional
        
    Returns:
        [B, n_samples] or [B] if n_samples is None
    
    TODO: the current iv implemnetation is correct but not necessarily stable for some regimes.

    Once xsf implements the correct iv, we can use it here.

    """
    if n_samples is None:
        out_shape = alpha.shape
        n_samples = 1
    else:
        out_shape = (*alpha.shape, n_samples)
    αβ = (alpha * beta).astype(cp.float64).ravel()
    di = d.astype(cp.int32).ravel()
    D = αβ.size
    out = cp.empty((D * n_samples,), dtype=cp.int32)
    seed = seed if seed is not None else int(cp.random.randint(0, 2**63-1))

    bessel_kernel((D,), (1,), (αβ, di, seed, int(n_samples), out))
    cp.cuda.Stream.null.synchronize()
    return out.reshape(out_shape)