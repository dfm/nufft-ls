#pragma once

#include <cmath>

#include "trig_sum.hpp"

namespace periodogram {

template <typename T>
inline int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

struct baseline {
  template <typename Scalar>
  static void compute(size_t N, const Scalar* t, const Scalar* y, const Scalar* w, size_t M,
                      const Scalar f0, const Scalar df, Scalar* power) {
    const Scalar sqrt_half = std::sqrt(Scalar(0.5));
    const Scalar domega = 2 * M_PI * df;
    const Scalar omega0 = 2 * M_PI * f0;

    // astropy's "standard" normalization
    const Scalar norm = normalization(N, y, w);

    for (size_t m = 0; m < M; ++m) {
      Scalar omega = m * domega + omega0;
      Scalar Sh = Scalar(0), Ch = Scalar(0);
      Scalar S2 = Scalar(0), C2 = Scalar(0);
      for (size_t n = 0; n < N; ++n) {
        Scalar wn = w[n];
        Scalar hn = wn * y[n];
        Scalar omegat = omega * t[n];
        Scalar sin = std::sin(omegat);
        Scalar cos = std::cos(omegat);

        Sh += hn * sin;
        Ch += hn * cos;

        // sin(2*x) = 2 * sin(x) * cos(x)
        // cos(2*x) = cos(x)*cos(x) - sin(x)*sin(x)
        S2 += 2 * wn * sin * cos;
        C2 += wn * (cos * cos - sin * sin);
      }

      Scalar tan_2omega_tau = S2 / C2;

      Scalar C2w = Scalar(1) / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
      Scalar S2w = tan_2omega_tau * C2w;
      Scalar Cw = sqrt_half * std::sqrt(1 + C2w);
      Scalar Sw = sqrt_half * sgn(S2w) * std::sqrt(1 - C2w);

      Scalar YC = Ch * Cw + Sh * Sw;
      Scalar YS = Sh * Cw - Ch * Sw;
      Scalar CC = 0.5 * (1 + C2 * C2w + S2 * S2w);
      Scalar SS = 0.5 * (1 - C2 * C2w - S2 * S2w);

      power[m] = norm * (YC * YC / CC + YS * YS / SS);
    }
  }

  template <typename Scalar>
  static Scalar normalization(size_t N, const Scalar* y, const Scalar* w){
    Scalar invnorm = Scalar(0);
    for(size_t n = 0; n < N; ++n){
      invnorm += w[n] * y[n] * y[n];
    }
    return Scalar(1)/invnorm;
  }
};

}  // namespace periodogram
