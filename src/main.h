#pragma once

#include <NumCpp.hpp>

#include "utils.h"

auto infer(const nc::NdArray<double>& data, int ori_fs) -> std::vector<Beat>;

auto get_input() -> nc::NdArray<double>;
