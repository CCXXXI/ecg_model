#include "utils.h"

#include <torch/script.h>

namespace {
std::string models_path;
}

auto set_models_path(const std::string& path) -> void { models_path = path; }

auto load_model(const std::string& filename) -> torch::jit::Module {
  auto model = torch::jit::load(models_path + filename);
  model.eval();
  return model;
}
