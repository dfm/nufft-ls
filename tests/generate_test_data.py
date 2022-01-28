#!/usr/bin/env python3

import numpy as np
from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
from astropy.timeseries.periodograms.lombscargle.implementations.fast_impl import (
    lombscargle_fast,
)

df = 0.1
N = 50
M = 12
random = np.random.default_rng(5043)
t = np.sort(random.uniform(0, 10, N))
y = random.normal(size=N)
w = random.uniform(0.5, 2.0, N)

w /= w.sum()  # for now, the C++ code will require normalized w
f0 = df/2  # f0=0 yields power[0] = nan. let's use f0=df/2, from LombScargle.autofrequency

sin, cos = trig_sum(t, y, df, M, use_fft=False)

power = lombscargle_fast(
    t,
    y,
    1 / np.sqrt(w),
    f0,
    df,
    M,
    normalization="standard",
    center_data=False,
    fit_mean=False,
    use_fft=False,
)

w /= w.sum()
Sh, Ch = trig_sum(t, w * y, df, M, f0=f0, use_fft=False)
S2, C2 = trig_sum(t, w, df, M, f0=f0, freq_factor=2, use_fft=False)
tan_2omega_tau = S2 / C2
S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)
# YY =
YC = Ch * Cw + Sh * Sw
YS = Sh * Cw - Ch * Sw
CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
SS = 0.5 * (1 - C2 * C2w - S2 * S2w)

print(YC * YC / CC + YS * YS / SS)
print(power * np.dot(w, y ** 2))


with open("data.hpp", "w") as f:
    f.write(
        f"""// Automatically generated
#pragma once

const size_t N = {N};
const size_t M = {M};
const double f0 = {f0};
const double df = {df};

const double input_t[] = {{
    {', '.join(map('{:.16e}'.format, t))}
}};
const double input_y[] = {{
    {', '.join(map('{:.16e}'.format, y))}
}};
const double input_w[] = {{
    {', '.join(map('{:.16e}'.format, w))}
}};

const double trig_sum_sin[] = {{
    {', '.join(map('{:.16e}'.format, sin))}
}};
const double trig_sum_cos[] = {{
    {', '.join(map('{:.16e}'.format, cos))}
}};

const double output_power[] = {{
    {', '.join(map('{:.16E}'.format, power))}
}};
"""
    )
