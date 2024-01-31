#!/usr/bin/env python3
'''Benchmark a number of Lomb-Scargle implementations.
'''

import os
import timeit

from astropy.table import Table
import click
import matplotlib.pyplot as plt
import numpy as np
import threadpoolctl

# The Gowanlock+ paper uses N_t=3554 as their single-object dataset.
# N_f=10**5 is a typical number of freq bins (we call this M)
DEFAULT_N = 3554
DEFAULT_M = 10**4
DEFAULT_DTYPE = 'f8'
DEFAULT_METHODS = ['cufinufft', 'finufft', 'astropy_fft']
DEFAULT_NTHREAD = len(os.sched_getaffinity(0))  # only for finufft_par currently

# @profile
def do_finufft(t, y, dy, f0, df, Nf, eps='default', nthreads=1):
    if f0 != 0:
        raise NotImplementedError('f0 != 0 not yet implemented')

    import finufft

    if nthreads < 1:
        nthreads = DEFAULT_NTHREAD

    # finufft only supports C2C, but our signal is real,
    # so to reach the same Nyquist as Astropy,
    # we double the number of freq bins and take the positive half
    Nf *= 2

    if eps == 'default':
        if y.dtype == np.float32:
            eps = 1e-6
        else:
            eps = 1e-15

    cdtype = np.complex128 if y.dtype == np.float64 else np.complex64

    w = dy**-2.
    w /= w.sum()
    # FINUFFT has fixed frequency spacing, so rescale the input signal
    # to achieve the desired df
    # TODO: not working yet
    # phase_shift1 = np.exp(1j * f0 * t)
    # phase_shift2 = np.exp(1j * f0 * 2 * t)
    t = 2 * np.pi * df * t
    t2 = 2 * t
    
    t %= 2 * np.pi
    t2 %= 2 * np.pi
    
    yw = y * w

    y = y.astype(cdtype, copy=False)
    w = w.astype(cdtype, copy=False)
    yw = yw.astype(cdtype, copy=False)

    # yw *= phase_shift1
    # w *= phase_shift2

    # Not really any opportunity for batch mode here.
    # Different objects generally have different NU points.
    plan = finufft.Plan(nufft_type=1, n_modes_or_dim=(Nf,), n_trans=1, eps=eps, dtype=cdtype, nthreads=nthreads)
    plan.setpts(t)
    f1 = plan.execute(yw)

    plan.setpts(t2)
    f2 = plan.execute(w)

    f1 = f1[Nf//2:]
    f2 = f2[Nf//2:]
    
    tan_2omega_tau = f2.imag / f2.real
    S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
    Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)

    YC = f1.real * Cw + f1.imag * Sw
    YS = f1.imag * Cw - f1.real * Sw
    CC = 0.5 * (1 + f2.real * C2w + f2.imag * S2w)
    SS = 0.5 * (1 - f2.real * C2w - f2.imag * S2w)

    power = YC * YC / CC + YS * YS / SS
    norm = np.sum(w.real * y.real ** 2)
    power /= norm

    return power


def do_cuda(t, y, dy, f0, df, Nf):
    import nufft_ls.cuda
    import cupy as cp

    dtype = y.dtype.type
    cuda_compute = {np.float32: nufft_ls.cuda.cu_compute_float,
                    np.float64: nufft_ls.cuda.cu_compute,
                    }[dtype]
    d_t = cp.asarray(t)
    d_y = cp.asarray(y)
    d_dy = cp.asarray(dy)
    d_power = cp.empty(Nf, dtype=dtype)
    d_w = d_dy**-2.
    d_w /= d_w.sum()

    cuda_compute(d_t.data.ptr, d_y.data.ptr, d_w.data.ptr, len(d_t),
                f0, df,
                d_power.data.ptr, len(d_power),
                )
    power = d_power.get()
    return power

# @profile
def do_cufinufft(t, y, dy, f0, df, Nf, eps='default'):
    import cufinufft
    import cupy as cp

    Nf *= 2

    if eps == 'default':
        if y.dtype == np.float32:
            eps = 1e-6
        else:
            eps = 1e-15

    cdtype = cp.complex128 if y.dtype == np.float64 else cp.complex64

    d_t = cp.asarray(t)
    d_y = cp.asarray(y)
    d_dy = cp.asarray(dy)
    d_w = d_dy**-2
    d_w /= d_w.sum()
    del t, y, dy

    d_t = 2 * cp.pi * df * d_t
    d_t2 = 2 * d_t
    
    d_t %= 2 * cp.pi
    d_t2 %= 2 * cp.pi
    
    d_yw = d_y * d_w

    d_yw = d_yw.astype(cdtype, copy=False)
    d_w = d_w.astype(cdtype, copy=False)

    d_f1 = cufinufft.nufft1d1(d_t, d_yw, Nf, eps=eps)
    d_f2 = cufinufft.nufft1d1(d_t2, d_w, Nf, eps=eps)

    d_f1 = d_f1[Nf//2:]
    d_f2 = d_f2[Nf//2:]
    
    # Could probably fuse these kernels, if the cost were found to be noticeable
    d_tan_2omega_tau = d_f2.imag / d_f2.real
    d_S2w = d_tan_2omega_tau / cp.sqrt(1 + d_tan_2omega_tau * d_tan_2omega_tau)
    d_C2w = 1 / cp.sqrt(1 + d_tan_2omega_tau * d_tan_2omega_tau)
    d_Cw = cp.sqrt(0.5) * cp.sqrt(1 + d_C2w)
    d_Sw = cp.sqrt(0.5) * cp.sign(d_S2w) * cp.sqrt(1 - d_C2w)

    d_YC = d_f1.real * d_Cw + d_f1.imag * d_Sw
    d_YS = d_f1.imag * d_Cw - d_f1.real * d_Sw
    d_CC = 0.5 * (1 + d_f2.real * d_C2w + d_f2.imag * d_S2w)
    d_SS = 0.5 * (1 - d_f2.real * d_C2w - d_f2.imag * d_S2w)

    d_power = d_YC * d_YC / d_CC + d_YS * d_YS / d_SS
    d_norm = cp.sum(d_w.real * d_y.real ** 2)
    d_power /= d_norm

    power = d_power.get()
    return power


def do_baseline(t, y, dy, f0, df, Nf):
    import nufft_ls.cpu

    dtype = y.dtype.type

    baseline_compute = {np.float32: nufft_ls.cpu.baseline_compute_float,
                        np.float64: nufft_ls.cpu.baseline_compute,
                        }[dtype]

    power = np.zeros(Nf, dtype=y.dtype)
    w = dy**-2.
    baseline_compute(t, y, w, f0, df, power)
    return power


def do_winding(t, y, dy, f0, df, Nf):
    import nufft_ls.cpu

    dtype = y.dtype.type

    winding_compute = {np.float32: nufft_ls.cpu.baseline_compute_winding_float,
                       np.float64: nufft_ls.cpu.baseline_compute_winding,
                       }[dtype]

    power = np.zeros(Nf, dtype=y.dtype)
    w = dy**-2.
    w /= w.sum()
    winding_compute(t, y, w, f0, df, power)
    return power


def do_astropy(t, y, dy, f0, df, Nf):
    import astropy.timeseries.periodograms.lombscargle.implementations.fast_impl as astropy_impl

    # N.B. we are using a patched astropy that will do the computation in float32 if requested
    power = astropy_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=Nf, use_fft=False, center_data=False, fit_mean=False)
    return power


def do_astropy_fft(t, y, dy, f0, df, Nf):
    import astropy.timeseries.periodograms.lombscargle.implementations.fast_impl as astropy_impl

    power = astropy_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=Nf, use_fft=True, center_data=False, fit_mean=False)
    return power


def do_astropy_fft_opt(t, y, dy, f0, df, Nf):
    import astropy.timeseries.periodograms.lombscargle.implementations.fast_impl as astropy_impl

    power = astropy_impl.lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=Nf, use_fft=True, center_data=False, fit_mean=False, trig_sum_kwds=dict(opt=True))
    return power


METHODS = {'baseline': do_baseline,
           'winding': do_winding,
           'astropy': do_astropy,
           'astropy_fft': do_astropy_fft,
           'astropy_fft_opt': do_astropy_fft_opt,
           'cufinufft': do_cufinufft,
           'cuda': do_cuda,
           'finufft': do_finufft,
           'finufft_par': lambda *args, **kwds: do_finufft(*args, **kwds, nthreads=DEFAULT_NTHREAD),
           }

METHOD_LABELS = {'finufft_par': f'finufft (nthread={DEFAULT_NTHREAD})',
                 'astropy_fft_opt': 'astropy_fft (optimized)',
                 }


@click.group()
def cli():
    pass

@cli.command('bench', context_settings={'show_default': True})
@click.option('-N', 'N', type=int, default=DEFAULT_N, help='N data')
@click.option('-logNfmin', 'logNfmin', type=float, default=4, help='log10 of min number of modes')
@click.option('-logNfmax', 'logNfmax', type=float, default=7, help='log10 of max number of modes')
@click.option('-logNfdelta', 'logNfdelta', type=float, default=1, help='Spacing in log10 for Nf values')
@click.option('-dtype', default=DEFAULT_DTYPE, help='dtype', type=click.Choice(('f4','f8')))
@click.option('--method', '-m', 'methods', default=DEFAULT_METHODS, help='methods to run', multiple=True, type=click.Choice(METHODS))
def bench(N, logNfmin, logNfmax, logNfdelta, dtype,
            methods=DEFAULT_METHODS,
            ):
    
    # process args
    dtype = np.dtype(dtype).type

    # don't let numpy do multithreading behind our back!
    # _limiter = threadpoolctl.threadpool_limits(1)

    # Generate fake data
    random = np.random.default_rng(5043)
    # t = np.sort(random.uniform(0, 1, N).astype(dtype))
    t = np.sort(random.uniform(0, 2 * np.pi, N).astype(dtype))
    t[0] = 0.
    y = random.normal(size=N).astype(dtype)
    dy = random.uniform(0.5, 2.0, N).astype(dtype)

    # f0 = dtype(df/2)  # f0=0 yields power[0] = nan. let's use f0=df/2, from LombScargle.autofrequency
    f0 = dtype(0.)
    # f0 = dtype(0.5)

    # make read-only
    t.setflags(write=False)
    y.setflags(write=False)
    dy.setflags(write=False)

    print(f'Running with {N=}, {logNfmin=}, {logNfmax=}, {logNfdelta=}, dtype {dtype.__name__}')

    all_Nf = np.logspace(logNfmin, logNfmax, int((logNfmax - logNfmin) / logNfdelta) + 1, dtype=int)

    all_res = []
    for method in methods:
        for Nf in all_Nf:
            df = dtype(10 / Nf)
            Nf = int(Nf)
            res = {'method': method, 'Nf': Nf, 'dtype': dtype.__name__, 'f0': f0, 'df': df, 'N': N}
            # warmup, and get result
            func = lambda: METHODS[method](t, y, dy, f0, df, Nf)  # noqa: E731
            res['power'] = func()
            
            nrep, tot_time = timeit.Timer(func).autorange()
            time = tot_time / nrep
            res['time'] = time

            print(f'{method} took {time:.4g} sec ({Nf=})')
            all_res.append(res)

    all_res = Table(all_res)
    
    # compare(all_res)
    
    del all_res['power']
    all_res.write('bench_results.ecsv', overwrite=True)

    perf_plot(all_res)


@cli.command('plot', context_settings={'show_default': True})
@click.argument('results', type=click.Path(exists=True))
def plot(results):
    all_res = Table.read(results)
    perf_plot(all_res)


def perf_plot(all_res: Table, sort=False):
    all_res = all_res.group_by('method')
    fig, ax = plt.subplots()
    ax: plt.Axes

    dtype = all_res['dtype'][0]
    N = all_res['N'][0]
    
    if sort:
        groups = sorted(all_res.groups, key=lambda g: g['time'].max(), reverse=True)
    else:
        groups = all_res.groups
    for group in groups:
        method = group['method'][0]
        label = METHOD_LABELS.get(method, method)
        ax.plot(group['Nf'], group['time'], label=label, marker='o')
        ax.set_title(f'{dtype=}, {N=}', loc='left', fontsize='small')

    xline = all_res['Nf'].max()
    yline = all_res['time'][all_res['Nf'] == xline].min()

    ax.axline((xline, yline/2), slope=1, label='Linear scaling', linestyle='--', color='k')

    ax.tick_params(right=True, top=True, which='both')
    
    ax.legend()
    ax.set_xlabel('Number of frequencies')
    ax.set_ylabel('Time [s]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig('perf.png', bbox_inches='tight')
    

@cli.command('compare', context_settings={'show_default': True})
@click.option('-N', 'N', type=int, default=DEFAULT_N, help='N data')
@click.option('-logNf', 'logNf', type=int, default=4, help='log10 of number of modes')
@click.option('-dtype', default=DEFAULT_DTYPE, help='dtype', type=click.Choice(('f4','f8')))
@click.option('--df', default='default', help='frequency spacing')
@click.option('--method', '-m', 'methods', default=DEFAULT_METHODS, help='methods to run', multiple=True, type=click.Choice(METHODS))
def compare_cmd(N, logNf, dtype, df, methods):
    dtype = np.dtype(dtype).type
    Nf = 10**logNf
    
    if df == 'default':
        df = dtype(10 / Nf)
    df = dtype(df)

    random = np.random.default_rng(5043)
    t = np.sort(random.uniform(0, 2 * np.pi, N).astype(dtype))
    t[0] = 0.
    y = random.normal(size=N).astype(dtype)
    dy = random.uniform(0.5, 2.0, N).astype(dtype)
    f0 = dtype(0.)

    t.setflags(write=False)
    y.setflags(write=False)
    dy.setflags(write=False)
    
    all_res = []
    for method in methods:
        res = {'method': method, 'Nf': Nf, 'dtype': dtype.__name__, 'f0': f0, 'df': df, 'N': N}
        res['power'] = METHODS[method](t, y, dy, f0, df, Nf)
        all_res.append(res)

    all_res = Table(all_res)

    compare(all_res)


def compare(all_res, plot=True):
    dtype = all_res['dtype'][0]
    f0 = all_res['f0'][0]
    df = all_res['df'][0]
    all_res = {row['method']: row['power'] for row in all_res}

    if dtype == np.float32:
        def isclose(*args):
            return np.isclose(*args, rtol=0.0001, atol=1e-07)
    else:
        def isclose(*args):
            return np.isclose(*args)
        
    # currently these don't match exactly in float32, even with the relaxed tolerance
    for k,j in zip(all_res, list(all_res.keys())[1:]):
        p1, p2 = all_res[k], all_res[j]

        rms_err = np.nanmean((p2 - p1)**2, dtype=np.float64)**0.5
        rms_mean = np.nanmean(((p1 + p2) / 2)**2, dtype=np.float64)**0.5

        # peak_frac_diff = np.abs((np.nanmax(p1) - np.nanmax(p2)) / ((np.nanmax(p1) + np.nanmax(p2)) / 2))

        print(f'{k} vs {j}')
        print(f'\tisclose {isclose(p1, p2).mean()*100:.4g}%')
        denom = np.sqrt(p1**2 + p2**2)
        nz = denom != 0
        frac = (np.abs(p1-p2)/denom)
        print(f'\tmax err {np.nanmax(frac[nz])*100:.4g}%')
        print(f'\trms err / mean {rms_err/rms_mean * 100:.4g}%')
        # print(f'\terror at peak {peak_frac_diff*100:.4g}%')

    if plot:
        freq = f0 + df*np.arange(len(next(iter(all_res.values()))))
        fig, ax = plt.subplots()
        for k in all_res:
            ax.plot(freq, all_res[k], label=k)
        ax.legend()
        ax.set_xlabel('frequency')
        ax.set_ylabel('L-S power')
        # ax.set_yscale('log')
        # ax.set_ylim(top=0.005)
        fig.savefig('compare.png')


if __name__ == '__main__':
    cli()
