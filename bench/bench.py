#!/usr/bin/env python3
'''Benchmark Astropy's LS fast_impl versus a baseline C++ implementation
'''

import timeit
import argparse

import numpy as np
from astropy.timeseries.periodograms.lombscargle.implementations import fast_impl
import threadpoolctl

# The Gowanlock+ paper uses N_t=3554 as their single-object dataset.
# N_f=10**5 is a typical number of freq bins (we call this M)
DEFAULT_N = 3554
DEFAULT_M = 10**5
DEFAULT_DTYPE = 'f8'
DEFAULT_DF = np.dtype(DEFAULT_DTYPE).type(1e-4)

def main(N, M, dtype, df=DEFAULT_DF,
            do_baseline=True, do_astropy=True, do_winding=True):
    # process args
    dtype = np.dtype(dtype).type
    df = dtype(df)

    # don't let numpy do multithreading behind our back!
    _limiter = threadpoolctl.threadpool_limits(1)

    rand = np.random.default_rng(43)

    # Generate fake data
    random = np.random.default_rng(5043)
    t = np.sort(random.uniform(0, 10, N).astype(dtype))
    y = random.normal(size=N).astype(dtype)
    dy = random.uniform(0.5, 2.0, N).astype(dtype)

    # And some derived quantities
    w = dy**-2.
    w /= w.sum()  # for now, the C++ code will require normalized w
    f0 = dtype(df/2)  # f0=0 yields power[0] = nan. let's use f0=df/2, from LombScargle.autofrequency

    print(f'Running with {N=}, {M=}, dtype {dtype.__name__}')

    if do_baseline or do_winding:
        import nufft_ls.cpu

    all_res = {}

    if do_baseline:
        baseline_compute = {np.float32: nufft_ls.cpu.baseline_compute_float,
                            np.float64: nufft_ls.cpu.baseline_compute,
                        }[dtype]

        # static void compute(size_t N, const Scalar* t, const Scalar* y, const Scalar* w, size_t M,
        #                       const Scalar f0, const Scalar df, Scalar* power);

        power = np.zeros(M, dtype=y.dtype)

        baseline_compute(t, y, w, f0, df, power)
        all_res['baseline'] = power.copy()

        time = timeit.timeit('baseline_compute(t, y, w, f0, df, power)',
                    number=(nloop:=5), globals=globals() | locals(),
                )
        print(f'baseline took {time/nloop:.4g} sec')

    if do_winding:
        winding_compute = {np.float32: nufft_ls.cpu.baseline_compute_winding_float,
                                np.float64: nufft_ls.cpu.baseline_compute_winding,
                            }[dtype]

        power = np.zeros(M, dtype=y.dtype)

        winding_compute(t, y, w, f0, df, power)
        all_res['winding'] = power.copy()

        time = timeit.timeit('winding_compute(t, y, w, f0, df, power)',
                    number=(nloop:=20), globals=globals() | locals(),
                )
        print(f'winding baseline took {time/nloop:.4g} sec')

    if do_astropy:
        # N.B. we are using a patched astropy that will do the computation in float32 if requested
        all_res['astropy'] = fast_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=M, use_fft=False, center_data=False, fit_mean=False)
        atime = timeit.timeit('fast_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=M, use_fft=False, center_data=False, fit_mean=False)',
                    number=(nloop:=1), globals=globals() | locals(),
                )
        print(f'astropy took {atime/nloop:.4g} sec')

    # If we have more than 2 answers, compare!
    compare(all_res, dtype)
    
    
def compare(all_res, dtype):
    if dtype == np.float32:
        isclose = lambda *args: np.isclose(*args, rtol=1e-4, atol=1e-7)
    else:
        isclose = lambda *args: np.isclose(*args)

    # currently these don't match exactly in float32, even with the relaxed tolerance
    for k,j in zip(all_res, list(all_res.keys())[1:]):
        p1, p2 = all_res[k], all_res[j]
        print(f'{k} vs {j}')
        print(f'\tfrac isclose {isclose(p1, p2).mean()*100:.4g}%')
        print(f'\tmax frac err {(np.abs(p1-p2)/p2).max():.4g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-N', type=int, default=DEFAULT_N, help='N data')
    parser.add_argument('-M', type=int, default=DEFAULT_M, help='M modes')
    parser.add_argument('-dtype', default=DEFAULT_DTYPE, help='dtype', choices=('f4','f8'))

    args = vars(parser.parse_args())
    main(**args)
