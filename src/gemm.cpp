#include "gemm.hpp"
#include <algorithm>
#include <immintrin.h> 

// 1. NAIVE
template<>
void Gemm<GemmType::NAIVE>::run(const std::vector<float>& A,
                                const std::vector<float>& B,
                                std::vector<float>& C,
                                size_t M, size_t K, size_t N) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// 2. CACHE_FRIENDLY
template<>
void Gemm<GemmType::CACHE_FRIENDLY>::run(const std::vector<float>& A,
                                         const std::vector<float>& B,
                                         std::vector<float>& C,
                                         size_t M, size_t K, size_t N) {
    std::fill(C.begin(), C.end(), 0.0f);
    constexpr size_t BS = 32;
    for (size_t i0 = 0; i0 < M; i0 += BS)
        for (size_t k0 = 0; k0 < K; k0 += BS)
            for (size_t j0 = 0; j0 < N; j0 += BS) {
                size_t i_end = std::min(i0 + BS, M);
                size_t k_end = std::min(k0 + BS, K);
                size_t j_end = std::min(j0 + BS, N);
                for (size_t i = i0; i < i_end; ++i)
                    for (size_t k = k0; k < k_end; ++k) {
                        float a = A[i * K + k];
                        for (size_t j = j0; j < j_end; ++j)
                            C[i * N + j] += a * B[k * N + j];
                    }
            }
}

// 3. AVX2 
template<>
void Gemm<GemmType::AVX2>::run(const std::vector<float>& A,
                               const std::vector<float>& B,
                               std::vector<float>& C,
                               size_t M, size_t K, size_t N) {
    // Если -mavx2 не передан, компилятор упадёт на __m256
    std::fill(C.begin(), C.end(), 0.0f);
    constexpr size_t SIMD_W = 8;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m256 a = _mm256_set1_ps(A[i * K + k]);
            size_t j = 0;
            for (; j + SIMD_W <= N; j += SIMD_W) {
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                _mm256_storeu_ps(&C[i * N + j], c);
            }
            for (; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}

template struct Gemm<GemmType::NAIVE>;
template struct Gemm<GemmType::CACHE_FRIENDLY>;
template struct Gemm<GemmType::AVX2>;