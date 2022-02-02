#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "periodogram/periodogram.hpp"
#include "periodogram/trig_sum.hpp"

namespace py = pybind11;

namespace periodogram_pybind {
    struct baseline {
        // TODO: do we need to disable forcecast?
        template <typename Scalar>
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

            periodogram::baseline::compute<Scalar>(N, tptr, yptr, wptr, M, f0, df, pptr);
        }
    };
}

PYBIND11_MODULE(nufft_ls, m) {
    m.doc() = "nufft_ls module";
    //m.def("trig_sum_naive_compute", &periodogram::trig_sum_naive::compute<double>, "trig_sum_naive::compute<double>");
    //m.def("trig_sum_naive_compute_float", &periodogram::trig_sum_naive::compute<float>, "trig_sum_naive::compute<float>");

    m.def("baseline_compute", &periodogram_pybind::baseline::compute<double>, "periodogram::baseline::compute<double>");
    m.def("baseline_compute_float", &periodogram_pybind::baseline::compute<float>, "periodogram::baseline::compute<float>");
}


/*
namespace periodogram_pybind {
    struct baseline {
        // TODO: do we need to disable forcecast?
        static void compute(py::array_t<double> t){
                        auto tunchecked = t.unchecked<1>();
                        const double* tptr = tunchecked.data(0);
                    }
    };
}

PYBIND11_MODULE(nufft_ls, m) {
    m.doc() = "nufft_ls module";

    m.def("baseline_compute", &periodogram_pybind::baseline::compute, "periodogram::baseline::compute<double>");
}
*/