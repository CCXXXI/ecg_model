#include "get_beats.h"

#include "../scipy/scipy.h"

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
  auto x = bsw(data);

  x = (x - nc::mean(x)) / nc::stdev(x);
  auto x_tensor = torch::from_blob(x.data(), {1, 1, x.size()}, torch::kDouble);

  auto model = load_model("u_net.pt");
  auto pred = model.forward({x_tensor}).toTensor()[0].argmax(0);
  auto pred_array = nc::NdArray<int>(pred.data_ptr<int>(), pred.size(0));
  auto output = output_sliding_voting_v2(pred_array);

  auto is_qrs = output == 1;
  return is_qrs;
}

auto u_net_r_peak(const nc::NdArray<bool>& is_qrs_origin) -> std::vector<int> {
  const auto origin_len = static_cast<int>(is_qrs_origin.size());

  auto is_qrs = is_qrs_origin;
  is_qrs = nc::insert(is_qrs, origin_len, false);
  is_qrs = nc::insert(is_qrs, 0, false);

  auto y = nc::zeros_like<bool>(is_qrs);
  for (auto pre = 0; pre < origin_len; ++pre) {
    const auto cur = pre + 1;
    const auto nxt = cur + 1;
    if (is_qrs[cur] && (is_qrs[pre] || is_qrs[nxt])) {
      y[pre] = !is_qrs[pre] || !is_qrs[nxt];
    }
  }

  auto start = 0;
  auto flag = false;
  auto r_list = std::vector<int>{};
  for (auto i = 0; i < origin_len; ++i) {
    if (!y[i]) {
      continue;
    }
    if (flag) {
      flag = false;
      r_list.push_back(start + static_cast<int>(std::floor((i - start) / 2)));
    } else {
      flag = true;
      start = i;
    }
  }

  return r_list;
}
}  // namespace

auto get_beats(const nc::NdArray<double>& data, const int ori_fs)
    -> std::vector<Beat> {
  auto data_vector = data.toStlVector();
  const auto n = static_cast<int>(data_vector.size());
  auto data_resampled_vector = std::vector<double>{};
  scipy::resample<double>(n * fs / ori_fs, n, data_vector,
                          data_resampled_vector);
  auto data_resampled = nc::NdArray<double>(data_resampled_vector);
  static constexpr auto len_u_net = 10 * 60 * fs;

  const auto len_data = static_cast<int>(data_resampled_vector.size());
  auto beats = std::vector<Beat>{};
  auto cur_s = 0;
  int now_s;
  while (cur_s < len_data) {
    if (cur_s + len_u_net <= len_data) {
      now_s = cur_s + len_u_net;
    } else {
      break;
    }
    const auto is_qrs = u_net_peak(
        data_resampled(data_resampled.rSlice(), nc::Slice(cur_s, now_s)));

    const auto r_list = u_net_r_peak(is_qrs);

    auto append_start = static_cast<int>(0.5 * 60 * fs);
    static constexpr auto append_end = static_cast<int>(9.5 * 60 * fs);
    if (cur_s == 0) {
      append_start = 0;
    }

    for (const auto& beat : r_list) {
      if (append_start < beat && beat <= append_end) {
        beats.push_back(Beat(beat + cur_s, Label::unknown));
      }
    }

    cur_s += 9 * 60 * fs;
  }

  return beats;
}
