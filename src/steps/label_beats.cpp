#include "label_beats.h"

#include <torch/all.h>

#include "../scipy/scipy.h"

namespace {
auto transform(const nc::NdArray<double>& sig) -> torch::Tensor {
    auto sig_vector = sig.toStlVector();
    const auto n = static_cast<int>(sig_vector.size());
    auto sig_resampled_vector = std::vector<double> {};
    scipy::resample<double>(360, n, sig_vector, sig_resampled_vector);
    auto sig_resampled = nc::NdArray<double>(sig_resampled_vector);
    auto sig_tensor = torch::from_blob(sig_resampled.data(), {1, 360}, torch::kDouble);
    return sig_tensor;
}
}  // namespace

auto label_beats(const nc::NdArray<double>& data_ori, std::vector<Beat>& beats,
                 const int ori_fs) -> std::vector<Beat> {
    static constexpr auto half_len = static_cast<int>(0.75 * fs);

    auto data_vector = data_ori.toStlVector();
    const auto n = static_cast<int>(data_vector.size());
    auto data_resampled_vector = std::vector<double> {};
    scipy::resample<double>(n * fs / ori_fs, n, data_vector,
                            data_resampled_vector);
    auto data_resampled = nc::NdArray<double>(data_resampled_vector);
    auto data = bsw(data_resampled);

    static constexpr auto batch_size = 64;
    auto input_tensor = std::vector<torch::Tensor> {};
    auto input_beats = std::vector<Beat> {};

    for (auto idx = 0u; idx < beats.size(); ++idx) {
        auto& beat = beats[idx];
        if (beat.position < half_len ||
                beat.position >= static_cast<int>(data.shape().rows) - half_len) {
            beat.label = Label::unknown;
            continue;
        }

        auto x = data(data.rSlice(),
        {beat.position - half_len, beat.position + half_len});
        x = x.reshape({1, half_len * 2});
        x = (x - nc::mean(x)) / nc::stdev(x);
        x = x.transpose();
        auto x_tensor = transform(x).unsqueeze(0);
        input_tensor.push_back(x_tensor);
        input_beats.push_back(beat);

        if (input_tensor.size() % batch_size == 0 || idx == beats.size() - 1) {
            x_tensor = torch::vstack(input_tensor);
            auto model = load_model("res_net.pt");
            auto output =
                torch::softmax(model.forward({x_tensor}).toTensor(), 1).squeeze();

            auto y_pred = torch::argmax(output, 1, false);
            for (auto i = 0u; i < y_pred.size(0); ++i) {
                auto pred = y_pred[i];
                auto pred_i = pred.item<int>();
                beat = input_beats[i];
                beat.label = static_cast<Label>(pred_i);
            }
            input_tensor.clear();
            input_beats.clear();
        }
    }

    return beats;
}
