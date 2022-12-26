#include <catch2/catch_test_macros.hpp>

#include "../src/forward.h"

template <typename T>
std::vector<T> load_txt(const std::string& path) {
  std::vector<T> buffer;
  std::ifstream file(path);
  for (T d; file >> d;) buffer.push_back(d);
  return buffer;
}

TEST_CASE("test", "[ecg_model]") {
  auto input = load_txt<double>("resources/input.txt");
  REQUIRE(input.size() == 144000);

  auto direct_output = forward(input.data(), 144000);
  REQUIRE(direct_output.sizes() == std::vector<int64_t>{4, 144000});

  auto expected_direct_output_raw =
      load_txt<double>("resources/direct_output.txt");
  auto expected_direct_output = torch::from_blob(
      expected_direct_output_raw.data(), {4, 144000}, torch::kDouble);
  REQUIRE(direct_output.allclose(expected_direct_output));
}
