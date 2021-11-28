#pragma once

#include <cmath>

namespace periodogram {

struct trig_sum_naive {
  template <typename Scalar>
  static void compute(size_t N, const Scalar* t, const Scalar* h, size_t M, const Scalar df,
                      Scalar* sinwt, Scalar* coswt) {
    Scalar df0 = 2 * M_PI * df;
    for (size_t m = 0; m < M; ++m) {
      Scalar omega = m * df0;
      Scalar s = Scalar(0), c = Scalar(0);
      for (size_t n = 0; n < N; ++n) {
        Scalar wt = omega * t[n];
        s += h[n] * std::sin(wt);
        c += h[n] * std::cos(wt);
      }
      sinwt[m] = s;
      coswt[m] = c;
    }
  }
};

}  // namespace periodogram
