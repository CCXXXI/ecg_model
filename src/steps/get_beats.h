#pragma once

#include <NumCpp.hpp>

#include "../utils.h"

auto get_beats(const nc::NdArray<double>& data, int ori_fs)
    -> std::vector<Beat>;
