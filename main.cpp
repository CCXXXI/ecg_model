#include <torch/torch.h>

#include <iostream>

namespace nn = torch::nn;

class CbrImpl : public nn::SequentialImpl {
 public:
  CbrImpl(int64_t in_channels, int64_t out_channels, int64_t kernel = 9,
          int64_t stride = 1, int64_t padding = 4)
      : nn::SequentialImpl(
            nn::Conv1d(nn::Conv1dOptions(in_channels, out_channels, kernel)
                           .stride(stride)
                           .padding(padding)
                           .bias(false)),
            nn::BatchNorm1d(nn::BatchNorm1dOptions(out_channels)), nn::ReLU()) {
  }
};

TORCH_MODULE(Cbr);

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
