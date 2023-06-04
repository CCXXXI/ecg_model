#include "label_beats.h"

#include <torch/all.h>

#include "../scipy/scipy.h"

namespace {
auto transform(const nc::NdArray<double>& sig) -> torch::Tensor {
  auto sig_vector = sig.toStlVector();
  const auto n = static_cast<int>(sig_vector.size());
  auto sig_resampled_vector = std::vector<double>{};
  scipy::resample<double>(360, n, sig_vector, sig_resampled_vector);
  auto sig_resampled = nc::NdArray<double>(sig_resampled_vector);
  auto sig_tensor = torch::from_blob(sig_resampled.data(), {1, 360});
  return sig_tensor;
}
}  // namespace

auto label_beats(const nc::NdArray<double>& data,
                 const std::vector<Beat>& beats, const int ori_fs)
    -> std::vector<Beat> {
  // todo: implement
}
