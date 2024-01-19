#pragma once

#include <cmath>
#include <cuda.h>

//#include "trig_sum.hpp"

namespace cu_periodogram {
    template <typename Scalar>
    __global__ void compute_sincos(
        Scalar* d_sin_domegat, Scalar* d_cos_domegat,
        const Scalar* d_t, size_t N, const Scalar twodf
    ) {
        const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N) return;

        // use if constexpr when we have C++17
        // to switch between sincospi and sincospif:
        if constexpr (std::is_same<Scalar, float>::value) {
            sincospif(twodf * d_t[i], &d_sin_domegat[i], &d_cos_domegat[i]);
        } else {
            sincospi(twodf * d_t[i], &d_sin_domegat[i], &d_cos_domegat[i]);
        }
    }

    template <typename Scalar,
                size_t CHUNKSIZE = 32,  // number of NU (time) points per chunk
                size_t NFREQ = 32  // number of frequencies per thread
                >
    __global__ void compute_periodogram(
        Scalar* d_power, size_t M,
        
        const Scalar twof0, const Scalar twodf,
        const Scalar* d_t,
        const Scalar* d_sin_domegat, const Scalar* d_cos_domegat,
        
        const Scalar* d_y, const Scalar* d_w, size_t N
    ) {

        const size_t tid_grid = threadIdx.x + blockIdx.x * blockDim.x;
        
        const size_t jstart = tid_grid * NFREQ;
        const size_t jend = min(jstart + NFREQ, M);
        const size_t this_nfreq = jend - jstart;

        const Scalar sqrt_half = std::sqrt(Scalar(0.5));

        // Might need to operate in chunks of N,
        // but we can loop over these chunks within
        // the kernel to avoid global reductions.
        // Need chunks of N because we want each thread
        // to do a sin/cos at a different freq
        // for every NU point in the chunk.
        // So with 32 threads, 48 KB shared memory
        // would be enough for chunks of 96 elements,
        // storing a sin and cos for each thread.

        // Note in this model every kernel loads every
        // NU point. We could change this, but then it
        // requires global reductions.

        __shared__ Scalar s_t[CHUNKSIZE];
        __shared__ Scalar s_y[CHUNKSIZE];
        __shared__ Scalar s_w[CHUNKSIZE];
        __shared__ Scalar s_sin_domegat[CHUNKSIZE];
        __shared__ Scalar s_cos_domegat[CHUNKSIZE];

        Scalar Sh[NFREQ], Ch[NFREQ];
        Scalar S2[NFREQ], C2[NFREQ];

        // loop over all NU points
        for(size_t i = 0; i < N; i += CHUNKSIZE) {
            const size_t iend = min(i + CHUNKSIZE, N);
            const size_t this_chunksize = iend - i;

            // Load a chunk of data into shared memory
            for (size_t ii = threadIdx.x; ii < this_chunksize; ii += blockDim.x) {
                s_t[ii] = d_t[i + ii];
                s_y[ii] = d_y[i + ii];
                s_w[ii] = d_w[i + ii];
                s_sin_domegat[ii] = d_sin_domegat[i + ii];
                s_cos_domegat[ii] = d_cos_domegat[i + ii];
            }
            __syncthreads();

            // loop over chunk
            for (size_t ii = 0; ii < this_chunksize; ++ii) {
                const Scalar hn = s_w[ii] * s_y[ii];
                Scalar sinomegat, cosomegat;
                const Scalar twof = twof0 + twodf * jstart;
                // bootstrap with a real sincos call
                sincospi(twof * s_t[ii], &sinomegat, &cosomegat);
                
                // loop over frequencies
                for (size_t j = 0; j < this_nfreq; ++j) {
                    Sh[j] += hn * sinomegat;
                    Ch[j] += hn * cosomegat;

                    // sin(2 x) = 2 sin(x) cos(x)
                    // cos(2 x) = cos(x) cos(x) - sin(x) sin(x)
                    S2[j] += 2 * s_w[ii] * sinomegat * sinomegat;
                    C2[j] += s_w[ii] * (cosomegat * cosomegat - sinomegat * sinomegat);

                    // angle addition formulae instead of calling sincos again
                    // sin(x + dx) = sin(x) cos(dx) + cos(x) sin(dx)
                    // cos(x + dx) = cos(x) cos(dx) - sin(x) sin(dx)
                    sinomegat = sinomegat * s_cos_domegat[ii] + cosomegat * s_sin_domegat[ii];
                    cosomegat = cosomegat * s_cos_domegat[ii] - sinomegat * s_sin_domegat[ii];
                }
            }
            __syncthreads();
        }

        for (size_t j = 0; j < this_nfreq; ++j) {
            const Scalar tan_2omega_tau = S2[j] / C2[j];

            const Scalar C2w = rsqrt(1 + tan_2omega_tau * tan_2omega_tau);
            const Scalar S2w = tan_2omega_tau * C2w;
            const Scalar Cw = sqrt_half * sqrt(1 + C2w);
            const Scalar Sw = copysign(sqrt_half * sqrt(1 - C2w), S2w);

            const Scalar YC = Ch[j] * Cw + Sh[j] * Sw;
            const Scalar YS = Sh[j] * Cw - Ch[j] * Sw;
            const Scalar CC = (Scalar) 0.5 * (1 + C2[j] * C2w + S2[j] * S2w);
            const Scalar SS = (Scalar) 0.5 * (1 - C2[j] * C2w - S2[j] * S2w);

            const Scalar norm = 1;  // TODO
            d_power[jstart + j] = norm * (YC * YC / CC + YS * YS / SS);
        }
    }
    
    template <typename Scalar>
    __host__ static void compute(
        const Scalar* d_t, const Scalar* d_y, const Scalar* d_w, size_t N,
        const Scalar f0, const Scalar df,
        Scalar* d_power, size_t M
    ) {
        // For starters, have each CUDA thread do several frequency bins
        // and all NU points.
        // But that may not expose enough parallelism.
        // We may be able to parallelize over NU points and then do
        // the reduction (probably across SMs).
        // Or we could focus on the multi-object case, possibly just by
        // launching a lot of kernels and letting CUDA handle the parallelism.

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // cudaMemsetAsync(d_power, 0, M * sizeof(Scalar), stream);

        Scalar *d_sin_domegat, *d_cos_domegat;
        cudaMallocAsync(&d_sin_domegat, N * sizeof(Scalar), stream);
        cudaMallocAsync(&d_cos_domegat, N * sizeof(Scalar), stream);

        const Scalar twof0 = 2 * f0;
        const Scalar twodf = 2 * df;

        size_t nthreads = 1024;
        size_t nblocks = (N + nthreads - 1) / nthreads;

        cu_periodogram::compute_sincos<<<nblocks, nthreads, 0, stream>>>(
            d_sin_domegat, d_cos_domegat,
            d_t, N, twodf
        );

        // Each thread does nfreq frequency bins.
        const size_t chunksize = 32;
        const size_t nfreq = 32;

        nthreads = 32;
        nblocks = (M + nthreads*nfreq - 1) / (nthreads*nfreq);
        cu_periodogram::compute_periodogram<Scalar,chunksize,nfreq><<<nblocks, nthreads, 0, stream>>>(
            d_power, M,
            twof0, twodf,
            d_t,
            d_sin_domegat, d_cos_domegat,
            d_y, d_w, N
        );

        cudaFreeAsync(d_sin_domegat, stream);
        cudaFreeAsync(d_cos_domegat, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}  // namespace cu_periodogram
