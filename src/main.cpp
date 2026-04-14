#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "tensor.hpp"
#include "conv_naive.hpp"
#include "conv_im2col.hpp"
#include "gemm.hpp"

int main() {
    std::srand(42);
    int N, C_in, H, W, C_out, K;
    
    std::cout << "=== Convolution Comparison (4 methods) ===\n\n";
    std::cout << "Входной тензор [N, C, H, W]:\n";
    std::cout << "  N: "; std::cin >> N;
    std::cout << "  C: "; std::cin >> C_in;
    std::cout << "  H: "; std::cin >> H;
    std::cout << "  W: "; std::cin >> W;
    std::cout << "\nЯдро:\n";
    std::cout << "  C_out: "; std::cin >> C_out;
    std::cout << "  K: "; std::cin >> K;
    
    if (H < K || W < K) { std::cerr << "Error: K > H or W (padding=0)\n"; return 1; }
    
    std::cout << "\nГенерация данных...\n";
    Tensor input(N, C_in, H, W);
    Tensor kernel(C_out, C_in, K, K);
    input.fill_random(1.0f);
    kernel.fill_random(1.0f);
    
    int OH = H - K + 1, OW = W - K + 1;
    std::cout << "Output: [" << N << "," << C_out << "," << OH << "," << OW << "]\n\n";
    
    std::cout << "1. conv_naive... ";
    Tensor out_naive = conv_naive(input, kernel);
    std::cout << "OK\n";
    
    std::cout << "2. im2col + GEMM(NAIVE)... ";
    Tensor out_g1 = conv_im2col<GemmType::NAIVE>(input, kernel);
    std::cout << "OK\n";
    
    std::cout << "3. im2col + GEMM(CACHE_FRIENDLY)... ";
    Tensor out_g2 = conv_im2col<GemmType::CACHE_FRIENDLY>(input, kernel);
    std::cout << "OK\n";
    
    std::cout << "4. im2col + GEMM(AVX2)... ";
    Tensor out_g3 = conv_im2col<GemmType::AVX2>(input, kernel);
    std::cout << "OK\n\n";
    
    std::cout << "=== Проверка ===\n";
    std::cout << "GEMM(NAIVE):         " << (out_naive.is_close(out_g1) ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "GEMM(CACHE_FRIENDLY):" << (out_naive.is_close(out_g2) ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "GEMM(AVX2):          " << (out_naive.is_close(out_g3) ? "✓ PASS" : "✗ FAIL") << "\n\n";
    
    std::cout << "=== Результат (n=0, c=0) ===\n";
    out_naive.print_slice(0, 0);
    return 0;
}