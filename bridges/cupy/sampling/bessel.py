import cupy as cp


from cupy.random import _kernels as _rk

#=== CUDA-only preamble: RNG state ===
_preamble = (
    "#define M_PI 3.14159265358979323846\n"
  + _rk.rk_basic_definition
  + _rk.loggam_definition
  + _rk.long_min_max_definition
  + _rk.rk_standard_exponential_definition
)

#=== CUDA kernel source ===
_kernel_source = r'''
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
    double ddi   = (double)di;

    int *row = out + idx * n_samp;

    // =================================================================
    // Exact CDF inversion sampler for
    //   P(M=m) ~ lam^m / (m! (m+d)!)
    //
    // Build unnormalized weights by recurrence from the mode,
    // expanding left and right until the tail is negligible.
    // Then sample by CDF inversion.
    //
    // No Bessel functions needed. Exact to double precision.
    // =================================================================

    // Mode: m0 = floor((sqrt(4*lam + d^2) - d) / 2)
    int m0 = (int)floor(0.5 * (sqrt(4.0 * lam_i + ddi * ddi) - ddi));
    if (m0 < 0) m0 = 0;

    const int MAXHALF = 256;
    const int MAXSUPPORT = 512;
    double cdf[MAXSUPPORT];

    // Expand right from mode: w(m0+j+1)/w(m0+j) = lam/((m0+j+1)(m0+d+j+1))
    double rw[MAXHALF];
    rw[0] = 1.0;
    int nr = 1;
    {
        double term = 1.0;
        for (int j = 0; j < MAXHALF - 1; ++j) {
            term *= lam_i / ((double)(m0 + j + 1) * (double)(m0 + di + j + 1));
            rw[j + 1] = term;
            nr = j + 2;
            double r_next = lam_i / ((double)(m0 + j + 2) * (double)(m0 + di + j + 2));
            if (r_next < 1.0) {
                double tail_bd = term * r_next / fmax(1.0 - r_next, 1e-300);
                if (tail_bd < 1e-14) break;
            }
            if (term < 1e-20) break;
        }
    }

    // Expand left from mode: w(m0-j-1)/w(m0-j) = (m0-j)(m0+d-j)/lam
    double lw[MAXHALF];
    int nl = 0;
    {
        double term = 1.0;
        for (int j = 0; j < m0 && j < MAXHALF - 1; ++j) {
            term *= (double)(m0 - j) * (double)(m0 + di - j) / lam_i;
            lw[j] = term;
            nl = j + 1;
            if (term < 1e-20) break;
        }
    }

    int total_K = nl + nr;
    if (total_K > MAXSUPPORT) total_K = MAXSUPPORT;
    int offset = m0 - nl;

    // Build CDF: left weights (reversed) then right weights
    double cumsum = 0.0;
    for (int i = 0; i < nl; ++i) {
        cumsum += lw[nl - 1 - i];
        cdf[i] = cumsum;
    }
    for (int i = 0; i < nr && (nl + i) < MAXSUPPORT; ++i) {
        cumsum += rw[i];
        cdf[nl + i] = cumsum;
    }
    double Z = cumsum;

    // Sample by CDF inversion
    for (int s = 0; s < n_samp; ++s) {
        double U = rk_double(&st) * Z;
        int k = 0;
        while (k < total_K - 1 && U > cdf[k]) ++k;
        row[s] = offset + k;
    }
}
'''

#=== Compile RawKernel at runtime ===
bessel_kernel = cp.RawKernel(_preamble + _kernel_source, 'bessel_devroye')

#=== Python wrapper ===
def bessel(alpha, beta, d, n_samples=None, seed=None):
    """
    Exact sampler for
    P[M=m | B1-D1=d] ~ (ab)^{m + d/2}/(m!(m+d)!) / I_d(2*sqrt(ab)).

    Uses CDF inversion with mode-relative weight recurrence.
    No Bessel function evaluation needed.

    Args:
        alpha: [B] or [B,1] or [B,D]
        beta:  [B] or [B,1] or [B,D]
        d:     [B] or [B,1] or [B,D]
        n_samples: int, optional
        seed: int, optional

    Returns:
        [B, n_samples] or [B] if n_samples is None
    """
    if n_samples is None:
        out_shape = alpha.shape
        n_samples = 1
    else:
        out_shape = (*alpha.shape, n_samples)
    ab = (alpha * beta).astype(cp.float64).ravel()
    di = d.astype(cp.int32).ravel()
    D = ab.size
    out = cp.empty((D * n_samples,), dtype=cp.int32)
    seed = seed if seed is not None else int(cp.random.randint(0, 2**63-1))

    bessel_kernel((D,), (1,), (ab, di, seed, int(n_samples), out))
    cp.cuda.Stream.null.synchronize()
    return out.reshape(out_shape)
