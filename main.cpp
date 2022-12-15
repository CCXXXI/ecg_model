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

class UNetImpl : public nn::Module {
  Cbr enc1_1, enc1_2, enc1_3, enc2_1, enc2_2, enc3_1, enc3_2, enc4_1, enc4_2,
      dec3_1, dec3_2, dec2_1, dec2_2, dec1_1, dec1_2, dec1_3, dec1_4;
  nn::ConvTranspose1d upsample_3, upsample_2, upsample_1;

 public:
  UNetImpl(int64_t class_n, int64_t layer_n)
      : enc1_1(1, layer_n),
        enc1_2(layer_n, layer_n),
        enc1_3(layer_n, layer_n),
        enc2_1(layer_n, layer_n * 2),
        enc2_2(layer_n * 2, layer_n * 2),
        enc3_1(layer_n * 2, layer_n * 4),
        enc3_2(layer_n * 4, layer_n * 4),
        enc4_1(layer_n * 4, layer_n * 8),
        enc4_2(layer_n * 8, layer_n * 8),
        dec3_1(layer_n * 4 + layer_n * 8, layer_n * 4),
        dec3_2(layer_n * 4, layer_n * 4),
        dec2_1(layer_n * 2 + layer_n * 4, layer_n * 2),
        dec2_2(layer_n * 2, layer_n * 2),
        dec1_1(layer_n * 1 + layer_n * 2, layer_n * 1),
        dec1_2(layer_n * 1, layer_n * 1),
        dec1_3(layer_n * 1, class_n * 2),
        dec1_4(class_n * 2, class_n),
        upsample_3(nn::ConvTranspose1dOptions(layer_n * 8, layer_n * 8, 8)
                       .stride(2)
                       .padding(3)),
        upsample_2(nn::ConvTranspose1dOptions(layer_n * 4, layer_n * 4, 8)
                       .stride(2)
                       .padding(3)),
        upsample_1(nn::ConvTranspose1dOptions(layer_n * 2, layer_n * 2, 8)
                       .stride(2)
                       .padding(3)) {}

  torch::Tensor forward(torch::Tensor x) {
    auto enc1 = enc1_1->forward(x);
    enc1 = enc1_2->forward(enc1);
    enc1 = enc1_3->forward(enc1);

    auto enc2 = torch::max_pool1d(enc1, 2);
    enc2 = enc2_1->forward(enc2);
    enc2 = enc2_2->forward(enc2);

    auto enc3 = torch::max_pool1d(enc2, 2);
    enc3 = enc3_1->forward(enc3);
    enc3 = enc3_2->forward(enc3);

    auto enc4 = torch::max_pool1d(enc3, 2);
    enc4 = enc4_1->forward(enc4);
    enc4 = enc4_2->forward(enc4);

    auto dec3 = upsample_3->forward(enc4);
    dec3 = dec3_1->forward(torch::cat({enc3, dec3}, 1));
    dec3 = dec3_2->forward(dec3);

    auto dec2 = upsample_2->forward(dec3);
    dec2 = dec2_1->forward(torch::cat({enc2, dec2}, 1));
    dec2 = dec2_2->forward(dec2);

    auto dec1 = upsample_1->forward(dec2);
    dec1 = dec1_1->forward(torch::cat({enc1, dec1}, 1));
    dec1 = dec1_2->forward(dec1);
    dec1 = dec1_3->forward(dec1);
    auto out = dec1_4->forward(dec1);

    return out;
  }
};

TORCH_MODULE(UNet);

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
