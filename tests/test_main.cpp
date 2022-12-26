#include <catch2/catch_test_macros.hpp>

#include "../src/main.h"

std::vector<double> load_input() {
  std::vector<double> buffer;
  std::ifstream file("resources/input.txt");
  for (double d; file >> d;) buffer.push_back(d);
  return buffer;
}

TEST_CASE("load_input", "[ecg_model]") {
  auto input = load_input();
  REQUIRE(input.size() == 144000);
}
