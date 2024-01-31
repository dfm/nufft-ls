#pragma once

#include <cmath>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/zip_function.h>

#include "cuda_helpers.hpp"

//#include "trig_sum.hpp"

namespace cu_periodogram {
    template <typename Scalar>
    __global__ void compute_sincos(
        Scalar* __restrict__ d_sin_domegat, Scalar* __restrict__ d_cos_domegat,
        const Scalar* __restrict__ d_t, size_t N, const Scalar twodf
    ) {
        const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N) return;

        sincospi(twodf * d_t[i], &d_sin_domegat[i], &d_cos_domegat[i]);
    }

    template <typename Scalar,
              size_t CHUNKSIZE = 32,  // number of NU (time) points per chunk
              size_t NFREQ = 32  // number of frequencies per thread
              >
    __global__ void compute_periodogram(
        Scalar* __restrict__ d_power, size_t M, const Scalar norm,
        
        const Scalar twof0, const Scalar twodf,
        const Scalar* __restrict__ d_t,
        const Scalar* __restrict__ d_sin_domegat, const Scalar* __restrict__ d_cos_domegat,
        
        const Scalar* __restrict__ d_y, const Scalar* __restrict__ d_w, size_t N
    ) {

        const size_t tid_grid = threadIdx.x + blockIdx.x * blockDim.x;
        
        const size_t jstart = min(tid_grid * NFREQ, M);
        const size_t jend = min((tid_grid + 1) * NFREQ, M);
        const size_t this_nfreq = jend - jstart;
        // assert(this_nfreq == 1);

        const Scalar sqrt_half = sqrt(Scalar(0.5));

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

        // __shared__ Scalar s_t[CHUNKSIZE];
        // __shared__ Scalar s_y[CHUNKSIZE];
        // __shared__ Scalar s_w[CHUNKSIZE];
        // __shared__ Scalar s_sin_domegat[CHUNKSIZE];
        // __shared__ Scalar s_cos_domegat[CHUNKSIZE];

        Scalar Sh[NFREQ] = {0.}, Ch[NFREQ] = {0.};
        Scalar S2[NFREQ] = {0.}, C2[NFREQ] = {0.};

        // loop over all NU points
        for(size_t i = 0; i < N; i += CHUNKSIZE) {
            const size_t iend = min(i + CHUNKSIZE, N);
            const size_t this_chunksize = iend - i;

            // // Load a chunk of data into shared memory
            // for (size_t ii = threadIdx.x; ii < this_chunksize; ii += blockDim.x) {
            //     s_t[ii] = d_t[i + ii];
            //     s_y[ii] = d_y[i + ii];
            //     s_w[ii] = d_w[i + ii];
            //     s_sin_domegat[ii] = d_sin_domegat[i + ii];
            //     s_cos_domegat[ii] = d_cos_domegat[i + ii];
            // }
            const Scalar *s_t = d_t + i;
            const Scalar *s_y = d_y + i;
            const Scalar *s_w = d_w + i;
            const Scalar *s_sin_domegat = d_sin_domegat + i;
            const Scalar *s_cos_domegat = d_cos_domegat + i;
            // __syncthreads();

            // loop over chunk
            for (size_t ii = 0; ii < this_chunksize; ++ii) {
                const Scalar hn = s_w[ii] * s_y[ii];
                Scalar sinomegat, cosomegat;
                const Scalar twof = twof0 + twodf * jstart;
                // bootstrap with a real sincos call
                // TODO: type dispatch?
                sincospi(twof * s_t[ii], &sinomegat, &cosomegat);
                
                // loop over frequencies
                // #pragma unroll NFREQ
                for (size_t j = 0; j < this_nfreq; ++j) {
                    Sh[j] += hn * sinomegat;
                    Ch[j] += hn * cosomegat;

                    // sin(2 x) = 2 sin(x) cos(x)
                    // cos(2 x) = cos(x) cos(x) - sin(x) sin(x)
                    S2[j] += 2 * s_w[ii] * sinomegat * cosomegat;
                    C2[j] += s_w[ii] * (cosomegat * cosomegat - sinomegat * sinomegat);

                    // TODO: skip freq update on last iteration
                    // angle addition formulae instead of calling sincos again
                    // sin(x + dx) = sin(x) cos(dx) + cos(x) sin(dx)
                    // cos(x + dx) = cos(x) cos(dx) - sin(x) sin(dx)
                    Scalar sinomegat_new = sinomegat * s_cos_domegat[ii] + cosomegat * s_sin_domegat[ii];
                    cosomegat = cosomegat * s_cos_domegat[ii] - sinomegat * s_sin_domegat[ii];
                    sinomegat = sinomegat_new;
                }
            }
            // __syncthreads();
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

            d_power[jstart + j] = norm * (YC * YC / CC + YS * YS / SS);
        }
    }

    template <typename Scalar>
    struct norm_op
    {
        __device__
        Scalar operator()(const Scalar& y, const Scalar& w) const { 
            return w * y * y;
        }
    };

    template<typename Scalar>
    __host__ Scalar normalization(
        const Scalar* d_y, const Scalar* d_w, const size_t N,
        const cudaStream_t stream
    ) {
        thrust::device_ptr<const Scalar> thrust_y(d_y);
        thrust::device_ptr<const Scalar> thrust_w(d_w);

        const Scalar invnorm = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            thrust::make_zip_iterator(thrust::make_tuple(d_y, d_w)),
            thrust::make_zip_iterator(thrust::make_tuple(d_y + N, d_w + N)),
            thrust::make_zip_function(norm_op<Scalar>()),
            (Scalar) 0,
            thrust::plus<Scalar>()
        );

        return Scalar(1)/invnorm;
    }
    
    template <typename Scalar>
    __host__ void compute(
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
        CHECK_CUDA(cudaStreamCreate(&stream));

        cudaMemsetAsync(d_power, 0, M * sizeof(Scalar), stream);

        Scalar norm = normalization(d_y, d_w, N, stream);

        Scalar *d_sin_domegat, *d_cos_domegat;
        CHECK_CUDA(cudaMallocAsync(&d_sin_domegat, N * sizeof(Scalar), stream));
        CHECK_CUDA(cudaMallocAsync(&d_cos_domegat, N * sizeof(Scalar), stream));

        const Scalar twof0 = 2 * f0;
        const Scalar twodf = 2 * df;

        size_t nthreads = 64;
        size_t nblocks = (N + nthreads - 1) / nthreads;

        cu_periodogram::compute_sincos<<<nblocks, nthreads, 0, stream>>>(
            d_sin_domegat, d_cos_domegat,
            d_t, N, twodf
        );
        CHECK_LAST_CUDA();

        // Each thread does nfreq frequency bins,
        // loading chunksize data points into shared memory.
        const size_t chunksize = 65536;
        const size_t nfreq = 4;

        // Double precision performance can be dramatically different than single.
        // Hence, the flop-to-byte performance can be different.
        // Double precision might prefer fewer sincos and hence higher nfreq,
        // while single precision might prefer more sincos and hence lower nfreq.
        // On the L40, nfreq 8 is close to optimal for double precision
        // while nfreq 4 is best for single (at nthread 64)
        // single-precision cuda is 3x faster than cufinufft on L40,
        // but everything else so far seems to be a tie
        // Might need to recheck CUDA sincos with smaller df, though

        nthreads = 64;
        nblocks = (M + nthreads*nfreq - 1) / (nthreads*nfreq);
        printf("nblocks = %zu\n", nblocks);
        CHECK_CUDA(cudaFuncSetCacheConfig(cu_periodogram::compute_periodogram<Scalar,chunksize,nfreq>, cudaFuncCachePreferL1));
        cu_periodogram::compute_periodogram<Scalar,chunksize,nfreq><<<nblocks, nthreads, 0, stream>>>(
            d_power, M,
            norm,
            twof0, twodf,
            d_t,
            d_sin_domegat, d_cos_domegat,
            d_y, d_w, N
        );
        CHECK_LAST_CUDA();

        CHECK_CUDA(cudaFreeAsync(d_sin_domegat, stream));
        CHECK_CUDA(cudaFreeAsync(d_cos_domegat, stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
    }
}  // namespace cu_periodogram
