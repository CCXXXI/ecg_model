#include "main.h"

#include <fstream>
#include <nlohmann/json.hpp>

#include "steps/steps.h"

auto infer(const nc::NdArray<double>& data, const int ori_fs)
    -> std::vector<Beat> {
  auto beats = get_beats(data, ori_fs);
  auto labelled_beats = label_beats(data, beats, ori_fs);
  return labelled_beats;
}

auto get_input() -> nc::NdArray<double> {
  std::ifstream in("./assets/ecg_data/assets/data.json");
  auto points = nlohmann::json::parse(in);

  auto n = static_cast<int>(points.size());
  auto data = nc::NdArray<double>(1, n);
  for (auto i = 0; i < n; ++i) {
    data(0, i) = points[i]["leadII"];
  }

  return data;
}
