#include "tensor.hpp"
#include <iostream>
#include <algorithm>

Tensor::Tensor(int n, int c, int h, int w)
    : N(n), C(c), H(h), W(w), 
      data(static_cast<size_t>(n) * c * h * w, 0.0f) {}

float& Tensor::operator()(int n, int c, int h, int w) {
    return data[((static_cast<size_t>(n) * C + c) * H + h) * W + w];
}

const float& Tensor::operator()(int n, int c, int h, int w) const {
    return data[((static_cast<size_t>(n) * C + c) * H + h) * W + w];
}

void Tensor::fill_random(float scale) {
    for (auto& v : data) {
        v = static_cast<float>(std::rand()) / RAND_MAX * scale;
    }
}

bool Tensor::is_close(const Tensor& other, float eps) const {
    if (N != other.N || C != other.C || H != other.H || W != other.W) 
        return false;
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::abs(data[i] - other.data[i]) > eps) 
            return false;
    }
    
    return true;
}

void Tensor::print_slice(int n, int c) const {

    std::cout << "Tensor[" << N << "," << C << "," << H << "," << W << "] ";
    std::cout << "slice n=" << n << ", c=" << c << ":\n";

    int print_h = std::min(H, 5);
    int print_w = std::min(W, 10);

    for (int h = 0; h < print_h; ++h) {
        for (int w = 0; w < print_w; ++w) {
            std::printf("%7.3f ", (*this)(n, c, h, w));
        }
        std::cout << "\n";
    }

    if (H > 5 || W > 10) std::cout << "... (truncated)\n";
}