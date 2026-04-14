#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <cstdlib>

class Tensor {
public:
    int N, C, H, W;
    std::vector<float> data;

    Tensor(int n, int c, int h, int w);
    
    float& operator()(int n, int c, int h, int w);
    const float& operator()(int n, int c, int h, int w) const;
    
    void fill_random(float scale = 1.0f);
    bool is_close(const Tensor& other, float eps = 1e-3f) const;
    void print_slice(int n = 0, int c = 0) const;
};