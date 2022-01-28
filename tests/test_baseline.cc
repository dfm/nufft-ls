#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "data.hpp"
#include "periodogram/periodogram.hpp"

TEST_CASE("Test baseline periodogram", "[baseline]") {
  double power[M];
  periodogram::baseline::compute<double>(N, input_t, input_y, input_w, M, f0, df, power);
  for (size_t m = 0; m < M; ++m) {
    REQUIRE(std::abs(power[m] - output_power[m]) < 1e-8);
  }
}
