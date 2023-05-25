#include "label_beats.h"

#include <torch/all.h>

namespace {
auto bsw(const nc::NdArray<double>& data) -> nc::NdArray<double> {
  // todo: implement
}

auto transform(const nc::NdArray<double>& sig) -> torch::Tensor {
  // todo: implement
}
}  // namespace

auto label_beats(const nc::NdArray<double>& data,
                 const std::vector<Beat>& beats, const int ori_fs)
    -> std::vector<Beat> {
  // todo: implement
}
