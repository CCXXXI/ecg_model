#include "get_beats.h"

namespace {
auto output_sliding_voting_v2(const nc::NdArray<int>& ori_output)
    -> nc::NdArray<int> {
  static constexpr auto window = 9;

  auto output = ori_output;
  const auto n = static_cast<int>(output.size());
  const auto half_window = window / 2;
  auto cnt = nc::zeros<int>(4, 1);
  auto l_index = 0;
  auto r_index = -1;
  for (auto i = 0; i < n; ++i) {
    if (r_index - l_index + 1 == window && half_window < i &&
        i < n - half_window) {
      --cnt[ori_output[l_index]];
      ++l_index;
    }
    while (r_index - l_index + 1 < window && r_index + 1 < n) {
      ++r_index;
      ++cnt[ori_output[r_index]];
    }
    output[i] = static_cast<int>(nc::argmax(cnt).front());
  }
  return output;
}

auto u_net_peak(const nc::NdArray<double>& data) -> nc::NdArray<bool> {
  // todo: implement
}

auto u_net_r_peak(const nc::NdArray<bool>& is_qrs) -> std::vector<int> {
  // todo: implement
}
}  // namespace

auto get_beats(const nc::NdArray<double>& data, const int ori_fs)
    -> std::vector<Beat> {
  // todo: implement
}
