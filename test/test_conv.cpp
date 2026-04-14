#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tensor.hpp"
#include "conv_naive.hpp"
#include "conv_im2col.hpp"
#include "gemm.hpp"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>

using Catch::Matchers::WithinRel;

// Вспомогательная функция для поэлементного сравнения тензоров
static bool tensors_close(const Tensor& a, const Tensor& b, float eps = 1e-4f) {
    if (a.N != b.N || a.C != b.C || a.H != b.H || a.W != b.W) return false;
    for (size_t i = 0; i < a.data.size(); ++i) {
        if (std::abs(a.data[i] - b.data[i]) > eps) return false;
    }
    return true;
}

// 1. Проверка conv_naive с фиксированными значениями (2 теста)
TEST_CASE("Naive Fixed 1: 1x1x3x3 input, 1x1x3x3 kernel (all ones)", "[naive][fixed]") {
    // Вход: числа от 1 до 9
    Tensor input(1, 1, 3, 3);
    for (int i = 0; i < 9; ++i) input.data[i] = static_cast<float>(i + 1);
    
    // Ядро: все единицы
    Tensor kernel(1, 1, 3, 3);
    std::fill(kernel.data.begin(), kernel.data.end(), 1.0f);

    auto out = conv_naive(input, kernel);
    
    REQUIRE(out.N == 1); REQUIRE(out.C == 1);
    REQUIRE(out.H == 1); REQUIRE(out.W == 1);
    // Ожидаемая сумма: 45
    REQUIRE_THAT(out(0, 0, 0, 0), WithinRel(45.0f, 1e-5f));
}

TEST_CASE("Naive Fixed 2: Multi-channel 4x4, known pattern", "[naive][fixed]") {
    // Input 1x2x4x4. Ch0=1.0, Ch1=2.0
    Tensor input(1, 2, 4, 4);
    for (int h = 0; h < 4; ++h)
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = 1.0f;
            input(0, 1, h, w) = 2.0f;
        }

    // Kernel 1x2x3x3. Ch0=1.0, Ch1=-1.0
    Tensor kernel(1, 2, 3, 3);
    for (int k = 0; k < 18; ++k)
        kernel.data[k] = (k < 9) ? 1.0f : -1.0f;

    auto out = conv_naive(input, kernel);
    
    // Ожидаемый результат на каждом патче 3x3:
    // 9*(1*1) + 9*(2*-1) = 9 - 18 = -9
    // Output shape: 1x1x2x2
    REQUIRE(out.N == 1); REQUIRE(out.C == 1);
    REQUIRE(out.H == 2); REQUIRE(out.W == 2);
    
    for (int h = 0; h < 2; ++h)
        for (int w = 0; w < 2; ++w)
            REQUIRE_THAT(out(0, 0, h, w), WithinRel(-9.0f, 1e-5f));
}

// 2. Проверка im2col для 3 разных GEMM с фиксированными значениями (2 теста)
TEST_CASE("Im2Col Fixed 1: Verify all 3 GEMM variants on simple input", "[im2col][fixed][gemm]") {
    Tensor input(1, 1, 3, 3);
    for (int i = 0; i < 9; ++i) input.data[i] = static_cast<float>(i + 1);
    Tensor kernel(1, 1, 3, 3);
    std::fill(kernel.data.begin(), kernel.data.end(), 1.0f);

    auto out_naive   = conv_im2col<GemmType::NAIVE>(input, kernel);
    auto out_cached  = conv_im2col<GemmType::CACHE_FRIENDLY>(input, kernel);
    auto out_avx2    = conv_im2col<GemmType::AVX2>(input, kernel);

    // Все три реализации должны дать 45.0
    REQUIRE_THAT(out_naive(0, 0, 0, 0),   WithinRel(45.0f, 1e-5f));
    REQUIRE_THAT(out_cached(0, 0, 0, 0),  WithinRel(45.0f, 1e-5f));
    REQUIRE_THAT(out_avx2(0, 0, 0, 0),    WithinRel(45.0f, 1e-5f));
}

TEST_CASE("Im2Col Fixed 2: Verify all 3 GEMM variants on multi-channel input", "[im2col][fixed][gemm]") {
    Tensor input(1, 2, 4, 4);
    for (int h = 0; h < 4; ++h)
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = 1.0f;
            input(0, 1, h, w) = 2.0f;
        }

    Tensor kernel(1, 2, 3, 3);
    for (int k = 0; k < 18; ++k)
        kernel.data[k] = (k < 9) ? 1.0f : -1.0f;

    auto out_naive   = conv_im2col<GemmType::NAIVE>(input, kernel);
    auto out_cached  = conv_im2col<GemmType::CACHE_FRIENDLY>(input, kernel);
    auto out_avx2    = conv_im2col<GemmType::AVX2>(input, kernel);

    // Все должны дать -9 на каждом элементе выхода 2x2
    for (int h = 0; h < 2; ++h)
        for (int w = 0; w < 2; ++w) {
            REQUIRE_THAT(out_naive(0, 0, h, w),  WithinRel(-9.0f, 1e-5f));
            REQUIRE_THAT(out_cached(0, 0, h, w), WithinRel(-9.0f, 1e-5f));
            REQUIRE_THAT(out_avx2(0, 0, h, w),   WithinRel(-9.0f, 1e-5f));
        }
}

// 3. Проверка совпадения im2col и naive на случайных данных (2 теста)
TEST_CASE("Random Cross-Check 1: im2col vs naive (K=3, medium size)", "[random][cross]") {
    std::srand(42); // Фиксированный seed для воспроизводимости
    Tensor input(2, 8, 16, 16);
    Tensor kernel(4, 8, 3, 3);
    input.fill_random(2.0f);
    kernel.fill_random(1.5f);

    auto out_naive  = conv_naive(input, kernel);
    auto out_im2col = conv_im2col<GemmType::CACHE_FRIENDLY>(input, kernel);

    REQUIRE(tensors_close(out_naive, out_im2col, 1e-4f));
}

TEST_CASE("Random Cross-Check 2: im2col vs naive (K=5, larger size)", "[random][cross]") {
    std::srand(123); // Другой seed
    Tensor input(2, 16, 32, 32);
    Tensor kernel(8, 16, 5, 5);
    input.fill_random(2.0f);
    kernel.fill_random(1.5f);

    auto out_naive  = conv_naive(input, kernel);
    auto out_im2col = conv_im2col<GemmType::AVX2>(input, kernel); 

    REQUIRE(tensors_close(out_naive, out_im2col, 1e-3f));
}