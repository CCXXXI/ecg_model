#include "utils.h"

#include <torch/script.h>

#include "scipy/scipy.h"

namespace {
std::string models_path;
}

auto set_models_path(const std::string& path) -> void {
    models_path = path;
}

auto load_model(const std::string& filename) -> torch::jit::Module {
    auto model = torch::jit::load(models_path + filename);
    model.eval();
    return model;
}

auto bsw(const nc::NdArray<double>& data) -> nc::NdArray<double> {
    auto data_vector = data.toStlVector();
    static const auto b = std::vector<double> {0.99349748, -0.99349748};
    static const auto a = std::vector<double> {1.0, -0.98699496};
    auto x_vector = std::vector<double> {};
    scipy::filtfilt(b, a, data_vector, x_vector);
    auto x = nc::NdArray<double>(x_vector);
    return x;
}
