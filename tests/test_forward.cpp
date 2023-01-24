#include <catch2/catch_test_macros.hpp>

#include "../src/forward.cpp"

template <typename T>
auto load_txt(const std::string& path) -> std::vector<T> {
  std::vector<T> buffer;
  std::ifstream file(path);
  for (T d; file >> d;) buffer.push_back(d);
  return buffer;
}

static constexpr int n = 144000;
static constexpr int m = 4;

TEST_CASE("test_forward", "[ecg_model]") {
  auto input = load_txt<double>("resources/input.txt");
  REQUIRE(input.size() == n);

  auto direct_output = forward(input.data(), n);
  REQUIRE(direct_output.sizes() == std::vector<int64_t>{m, n});

  auto expected_direct_output_raw =
      load_txt<double>("resources/direct_output.txt");
  auto expected_direct_output = torch::from_blob(
      expected_direct_output_raw.data(), {m, n}, torch::kDouble);
  REQUIRE(direct_output.allclose(expected_direct_output));

  auto argmax_output = direct_output.argmax(0);
  REQUIRE(argmax_output.sizes() == std::vector<int64_t>{n});

  auto expected_argmax_output_raw =
      load_txt<int64_t>("resources/argmax_output.txt");
  auto expected_argmax_output =
      torch::from_blob(expected_argmax_output_raw.data(), {n}, torch::kInt64);
  REQUIRE(argmax_output.equal(expected_argmax_output));
}
