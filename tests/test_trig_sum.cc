#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "data.hpp"
#include "periodogram/trig_sum.hpp"

TEST_CASE("Test trig_sum_naive", "[trig_sum_naive]") {
  double sinwt[M], coswt[M];
  periodogram::trig_sum_naive::compute<double>(N, input_t, input_y, M, df, sinwt, coswt);
  REQUIRE(std::abs(sinwt[0]) < 1e-8);
  for (size_t m = 0; m < M; ++m) {
    REQUIRE(std::abs(sinwt[m] - trig_sum_sin[m]) < 1e-8);
    REQUIRE(std::abs(coswt[m] - trig_sum_cos[m]) < 1e-8);
  }
}
