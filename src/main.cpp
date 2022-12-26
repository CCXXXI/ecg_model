#include <torch/script.h>

#include <iostream>

int main() {
  auto model = torch::jit::load("resources/model.pt");
  std::cout << "Model loaded" << std::endl;
}
