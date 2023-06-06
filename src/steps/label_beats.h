#pragma once

#include <NumCpp.hpp>

#include "../utils.h"

auto label_beats(const nc::NdArray<double>& data_ori, std::vector<Beat>& beats,
                 int ori_fs) -> std::vector<Beat>;
