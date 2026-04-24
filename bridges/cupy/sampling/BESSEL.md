Yes. In the regime you care about, the best patch is **not** to make `besselI_int` smarter. It is to **bypass the Bessel call entirely**.

For large (d), especially when (\lambda=\alpha\beta) is fixed or only (O(d)), the pmf is
[
p(m)\propto \frac{\lambda^m}{m!(m+d)!},
\qquad
\frac{p(m+1)}{p(m)}=\frac{\lambda}{(m+1)(m+d+1)}.
]
So once (\lambda \le d+1), the mode is at (m=0), the mass is monotone decreasing, and you can sample by a tiny recurrence/CDF with no Bessel and no rejection.

That is both **more stable** and usually **faster**.

## The fast exact patch for the large-(d) regime

Add this branch before the `besselI_int` call:

```cpp
double ddi = (double)di;  // use this everywhere for safety

// exact fast path when mode is 0, i.e. lambda <= d + 1
if (lam_i <= ddi + 1.0) {
    const int MAXK = 64;      // plenty in the large-d regime
    double cdf[MAXK];

    // unnormalized weights t_m with t_0 = 1 and
    // t_{m+1} = t_m * lam / ((m+1)(m+d+1))
    double term = 1.0;
    double Z = 1.0;
    cdf[0] = 1.0;
    int K = 0;
    bool tail_ok = false;

    for (int m = 0; m < MAXK - 1; ++m) {
        term *= lam_i / (double)((m + 1) * (m + di + 1));
        Z += term;
        cdf[m + 1] = Z;
        K = m + 1;

        // rigorous geometric tail bound since ratios are decreasing here
        double r_next = lam_i / (double)((m + 2) * (m + di + 2));
        double tail_bd = (r_next < 1.0)
            ? term * r_next / fmax(1.0 - r_next, 1e-300)
            : INFINITY;

        if (tail_bd < 1e-14 * Z) {
            tail_ok = true;
            break;
        }
    }

    if (tail_ok) {
        for (int s = 0; s < n_samp; ++s) {
            double U = rk_double(&st) * Z;
            int m = 0;
            while (m < K && U > cdf[m]) ++m;
            row[s] = m;
        }
        return;
    }
    // otherwise fall through to the general sampler
}
```

### Why this is fast

In the problematic regime,
[
\frac{p(1)}{p(0)} = \frac{\lambda}{d+1},
]
so when (d) is large the support is tiny. Usually only (m=0,1,2) matter. That means:

* no `besselI_int`
* no `loggam`
* no rejection loop
* just a few multiplies/divides once per parameter, then tiny CDF lookups per draw

This should be **faster than your current code** in exactly the regime where the current code misbehaves.

## Two one-line fixes you should make regardless

You also have real overflow bugs from integer multiplication before casting.

Change:

```cpp
int m0 = (int)floor((sqrt(4.0 * lam_i + double(di*di)) - di) / 2.0);
```

to

```cpp
double ddi = (double)di;
int m0 = (int)floor(0.5 * (sqrt(4.0 * lam_i + ddi * ddi) - ddi));
```

and change:

```cpp
double mu = double(4 * d * d);
```

to

```cpp
double mu = 4.0 * ddi * ddi;
```

Without this, things go wrong for (d) in the tens of thousands no matter what else you do.

## A stronger general patch

If you want to keep Devroye for the non-large-(d) cases, the next best improvement is to remove the direct Bessel evaluation entirely and compute
[
p_0 = \Pr(M=m_0)
]
by summing weights **relative to the mode**:
[
p_0^{-1}
========

1
+
\sum_{j\ge1}\prod_{r=1}^j
\frac{\lambda}{(m_0+r)(m_0+d+r)}
+
\sum_{j=1}^{m_0}\prod_{r=1}^j
\frac{(m_0-r+1)(m_0+d-r+1)}{\lambda}.
]
That is stable and exact up to truncation, and it avoids `I_d` altogether. In the large-(d) regime it is still cheap because (m_0) is small.

## Optional ultra-fast approximation

If tiny approximation error is acceptable, then in the same regime
[
M \approx \text{Poisson}!\left(\frac{\lambda}{d+1}\right).
]
That is extremely good when (\lambda \ll d^2), and for very small (\lambda/(d+1)) you can even use a Bernoulli or “mostly zero” branch. But the exact monotone-recurrence branch above is already fast enough that I would start there.

The short recommendation is:

1. add the exact `lam_i <= d+1` monotone-CDF branch,
2. fix the integer overflows,
3. only then worry about improving the general Bessel path.

That gives you a high-leverage patch with minimal surgery and preserves speed where you care most.
