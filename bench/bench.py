#!/usr/bin/env python3

import timeit

import numpy as np
from astropy.timeseries.periodograms.lombscargle.implementations import fast_impl
import threadpoolctl

# don't let numpy do multithreading behind our back!
_limiter = threadpoolctl.threadpool_limits(1)

rand = np.random.default_rng(43)

N = 355
dtype = np.float64
df = dtype(0.1)
M = 10**6  # num freq bins

random = np.random.default_rng(5043)
t = np.sort(random.uniform(0, 10, N).astype(dtype))
y = random.normal(size=N).astype(dtype)
dy = random.uniform(0.5, 2.0, N).astype(dtype)

w = dy**-2.
w /= w.sum()  # for now, the C++ code will require normalized w
f0 = dtype(df/2)  # f0=0 yields power[0] = nan. let's use f0=df/2, from LombScargle.autofrequency

print(f'Running with {N=}, {M=}, dtype {dtype.__name__}')

if do_nufft:=True:
    import nufft_ls
    nufft_ls_compute = {np.float32: nufft_ls.baseline_compute_float,
                        np.float64: nufft_ls.baseline_compute,
                    }[dtype]

    # static void compute(size_t N, const Scalar* t, const Scalar* y, const Scalar* w, size_t M,
    #                       const Scalar f0, const Scalar df, Scalar* power) {

    power = np.empty(M, dtype=y.dtype)

    nufft_ls_compute(t, y, w, f0, df, power)

    time = timeit.timeit('nufft_ls_compute(t, y, w, f0, df, power)', number=(nloop:=1), globals=globals())
    print(f'baseline took {time/nloop:.4g} sec')

if do_astropy:=True:
    # N.B. we are using a patched astropy that will do the computation in float32 if requested
    apower = fast_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=M, use_fft=False, center_data=False, fit_mean=False)
    atime = timeit.timeit('fast_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=M, use_fft=False, center_data=False, fit_mean=False)',
                number=(nloop:=1), globals=globals(),
            )
    print(f'astropy took {atime/nloop:.4g} sec')

if dtype == np.float32:
    isclose = lambda *args: np.isclose(*args, rtol=1e-4, atol=1e-7)
else:
    isclose = lambda *args: np.isclose(*args)

if do_nufft and do_astropy:
    # currently these don't match exactly in float32, even with the relaxed tolerance
    print(f'frac isclose {isclose(power, apower).mean()*100:.4g}%')
    print(f'max frac err {(np.abs(power-apower)/apower).max():.4g}')
    #breakpoint()
