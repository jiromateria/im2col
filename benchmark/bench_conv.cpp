#include <benchmark/benchmark.h>
#include <random>
#include "tensor.hpp"
#include "conv_naive.hpp"
#include "conv_im2col.hpp"
#include "gemm.hpp"

// Фиксированные параметры для стабильного сравнения
constexpr int N      = 2;
constexpr int C_IN   = 32;
constexpr int H      = 32;
constexpr int W      = 32;
constexpr int C_OUT  = 64;

static void fill_random(Tensor& t) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : t.data) v = dist(rng);
}

template<typename ConvFunc>
static void run_bench(benchmark::State& state, ConvFunc func) {
    int K = static_cast<int>(state.range(0));
    Tensor input(N, C_IN, H, W);
    Tensor kernel(C_OUT, C_IN, K, K);
    fill_random(input);
    fill_random(kernel);

    // Аллокация вынесена за цикл замеров
    for (auto _ : state) {
        auto out = func(input, kernel);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(K);
}

// 1. Прямая наивная свёртка
static void BM_ConvNaive(benchmark::State& state) { run_bench(state, conv_naive); }
BENCHMARK(BM_ConvNaive)
    ->Args({3})->Args({5})->Args({7})->Args({9})->Args({11})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 2. im2col + GEMM NAIVE
static void BM_Im2Col_Naive(benchmark::State& state) { run_bench(state, conv_im2col<GemmType::NAIVE>); }
BENCHMARK(BM_Im2Col_Naive)
    ->Args({3})->Args({5})->Args({7})->Args({9})->Args({11})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 3. im2col + GEMM CACHE_FRIENDLY
static void BM_Im2Col_Cache(benchmark::State& state) { run_bench(state, conv_im2col<GemmType::CACHE_FRIENDLY>); }
BENCHMARK(BM_Im2Col_Cache)
    ->Args({3})->Args({5})->Args({7})->Args({9})->Args({11})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 4. im2col + GEMM AVX2
static void BM_Im2Col_AVX2(benchmark::State& state) { run_bench(state, conv_im2col<GemmType::AVX2>); }
BENCHMARK(BM_Im2Col_AVX2)
    ->Args({3})->Args({5})->Args({7})->Args({9})->Args({11})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();