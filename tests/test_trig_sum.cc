#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "periodogram/trig_sum.hpp"

unsigned int Factorial(unsigned int number) {
  return number <= 1 ? number : Factorial(number - 1) * number;
}

TEST_CASE("Factorials are computed", "[factorial]") {
  const size_t N = 100;
  const size_t M = 12;
  double h[N], t[N], sinwt[M], coswt[M];

  for (size_t n = 0; n < N; ++n) {
    h[n] = 1.0;
  }
  periodogram::trig_sum_naive::compute<double>(N, t, h, M, 0.5, sinwt, coswt);

  REQUIRE(sinwt[0] == 0.0);
}
