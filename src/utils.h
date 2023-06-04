#pragma once

#include <torch/torch.h>

#include <NumCpp.hpp>

enum class Label : uint8_t {
    /// 窦性心律 Sinus rhythm
    sinus_rhythm = 0,

    /// 房性早搏 Atrial premature beat
    atrial_premature_beat = 1,

    /// 心房扑动 Atrial flutter
    atrial_flutter = 2,

    /// 心房颤动 Atrial fibrillation
    atrial_fibrillation = 3,

    /// 室性早搏 Ventricular premature beat
    ventricular_premature_beat = 4,

    /// 阵发性室上性心动过速 Paroxysmal supra-ventricular tachycardia
    paroxysmal_supra_ventricular_tachycardia = 5,

    /// 心室预激 Ventricular pre-excitation
    ventricular_pre_excitation = 6,

    /// 室扑室颤 Ventricular flutter and fibrillation
    ventricular_flutter_and_fibrillation = 7,

    /// 房室传导阻滞 Atrioventricular block
    atrioventricular_block = 8,

    /// 噪声 Noise
    noise = 9,

    /// 未知 Unknown
    unknown = 10,
};

struct Beat {
    int position;
    Label label;
};

constexpr auto fs = 240;

auto set_models_path(const std::string& path) -> void;

auto load_model(const std::string& filename) -> torch::jit::Module;

auto bsw(const nc::NdArray<double>& data) -> nc::NdArray<double>;
