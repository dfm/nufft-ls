#include <pybind11/pybind11.h>

#include "periodogram/cu_periodogram.hpp"

namespace py = pybind11;

namespace cu_periodogram_pybind {
    // TODO: we use uint64_t because pybind11 doesn't natively
    // support cupy arrays. While we could write a type converter,
    // it's not even clear that using pybind11 is important.
    // We might prefer to have cupy or pycuda build the kernels.
    template <typename Scalar>
    static void compute(const uint64_t t,
                        const uint64_t y,
                        const uint64_t w,
                        const size_t N,
                        const Scalar f0, const Scalar df,
                        uint64_t power,
                        const size_t M
                        ){
        const Scalar* d_t = reinterpret_cast<const Scalar*>(t);
        const Scalar* d_y = reinterpret_cast<const Scalar*>(y);
        const Scalar* d_w = reinterpret_cast<const Scalar*>(w);
        // const Scalar* d_t = t.cast<Scalar*>();
        // const Scalar* d_y = y.cast<Scalar*>();
        // const Scalar* d_w = w.cast<Scalar*>();
        Scalar* d_power = reinterpret_cast<Scalar*>(power);
        // Scalar* d_power = power.cast<Scalar*>();

        cu_periodogram::compute<Scalar>(d_t, d_y, d_w, N, f0, df, d_power, M);
    }
}

PYBIND11_MODULE(cuda, m) {
    m.doc() = "nufft_ls.cuda module";

    m.def("cu_compute", &cu_periodogram_pybind::compute<double>, "cu_periodogram::compute<double>");
    m.def("cu_compute_float", &cu_periodogram_pybind::compute<float>, "cu_periodogram::compute<float>");
}
