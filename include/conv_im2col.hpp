#pragma once
#include "tensor.hpp"
#include "gemm.hpp"

template<GemmType G = GemmType::CACHE_FRIENDLY>
Tensor conv_im2col(const Tensor& input, const Tensor& kernel);