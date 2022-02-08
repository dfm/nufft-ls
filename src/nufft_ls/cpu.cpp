#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "periodogram/periodogram.hpp"
#include "periodogram/trig_sum.hpp"

namespace py = pybind11;

enum Kind { SINCOS, WINDING };  // baseline implementations

namespace periodogram_pybind {
    struct baseline {
        // TODO: do we need to disable forcecast?
        template <typename Scalar, Kind K>
        static void compute(const py::array_t<Scalar, py::array::c_style> t,
                            const py::array_t<Scalar, py::array::c_style> y,
                            const py::array_t<Scalar, py::array::c_style> w,
                            const Scalar f0, const Scalar df,
                            py::array_t<Scalar, py::array::c_style> power
                            ){
            const Scalar* tptr = t.template unchecked<1>().data(0);  // yike
            const Scalar* yptr = y.template unchecked<1>().data(0);
            const Scalar* wptr = w.template unchecked<1>().data(0);
            Scalar* pptr = power.template mutable_unchecked<1>().mutable_data(0);
            
            size_t N = t.size();
            size_t M = power.size();

            if constexpr (K == SINCOS) {
                periodogram::baseline::compute<Scalar>(N, tptr, yptr, wptr, M, f0, df, pptr);
            } else if constexpr (K == WINDING) {
                periodogram::baseline::compute_winding<Scalar>(N, tptr, yptr, wptr, M, f0, df, pptr);
            } else {
                static_assert("Unknown baseline kind?");
            }
        }
    };
}

PYBIND11_MODULE(cpu, m) {
    m.doc() = "nufft_ls.cpu module";
    //m.def("trig_sum_naive_compute", &periodogram::trig_sum_naive::compute<double>, "trig_sum_naive::compute<double>");
    //m.def("trig_sum_naive_compute_float", &periodogram::trig_sum_naive::compute<float>, "trig_sum_naive::compute<float>");

    m.def("baseline_compute", &periodogram_pybind::baseline::compute<double, SINCOS>, "periodogram::baseline::compute<double>");
    m.def("baseline_compute_float", &periodogram_pybind::baseline::compute<float, SINCOS>, "periodogram::baseline::compute<float>");

    m.def("baseline_compute_winding", &periodogram_pybind::baseline::compute<double, WINDING>, "periodogram::baseline::compute_winding<double>");
    m.def("baseline_compute_winding_float", &periodogram_pybind::baseline::compute<float, WINDING>, "periodogram::baseline::compute_winding<float>");
}
