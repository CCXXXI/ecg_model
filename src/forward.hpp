#include <torch/script.h>

auto load_model() -> torch::jit::script::Module {
  auto model = torch::jit::load("resources/model.pt");
  model.eval();
  return model;
}

auto forward(double* data, int size) -> at::Tensor {
  static auto model = load_model();

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(torch::from_blob(data, {1, 1, size}, torch::kDouble));

  torch::NoGradGuard no_grad;
  return model.forward(inputs).toTensor()[0];
}

extern "C" auto forward_argmax(double* data, int size, uint8_t* out) -> void {
  auto res = forward(data, size).argmax(0).to(torch::kU8);
  memcpy(out, res.data_ptr(), size);
}
