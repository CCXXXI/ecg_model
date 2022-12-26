#pragma once

#include <torch/script.h>

auto forward(double* data, int size) -> at::Tensor;
