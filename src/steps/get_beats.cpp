#include "get_beats.h"

namespace {
auto u_net_peak(const nc::NdArray<double>& data) -> nc::NdArray<bool> {
  // todo: implement
}

auto u_net_r_peak(const nc::NdArray<bool>& is_qrs) -> std::vector<int> {
  // todo: implement
}

auto output_sliding_voting_v2(const nc::NdArray<int>& ori_output)
    -> nc::NdArray<int> {
  // todo: implement
}
}  // namespace

auto get_beats(const nc::NdArray<double>& data, const int ori_fs)
    -> std::vector<Beat> {
  // todo: implement
}
