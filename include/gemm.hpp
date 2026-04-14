#pragma once
#include <vector>
#include <cstddef>

enum class GemmType { 
    NAIVE, 
    CACHE_FRIENDLY, 
    AVX2 
};

template<GemmType G>
struct Gemm {
    static void run(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    size_t M, size_t K, size_t N);
};