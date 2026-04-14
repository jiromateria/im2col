#include "conv_im2col.hpp"
#include <vector>
#include <cassert>

static std::vector<float> im2col(const Tensor& input, int K, size_t& rows, size_t& cols) {
    int N = input.N, C = input.C, H = input.H, W = input.W;
    int OH = H - K + 1, OW = W - K + 1;
    rows = static_cast<size_t>(N) * OH * OW;
    cols = static_cast<size_t>(C) * K * K;
    std::vector<float> A(rows * cols);
    
    size_t r = 0;
    for (int n = 0; n < N; ++n)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                size_t c_idx = 0;
                for (int c = 0; c < C; ++c)
                    for (int kh = 0; kh < K; ++kh)
                        for (int kw = 0; kw < K; ++kw)
                            A[r * cols + c_idx++] = input(n, c, oh + kh, ow + kw);
                ++r;
            }
    return A;
}

static std::vector<float> kernel2mat(const Tensor& kernel, size_t& rows, size_t& cols) {
    int C_out = kernel.N, C_in = kernel.C, K = kernel.H;
    rows = static_cast<size_t>(C_in) * K * K;
    cols = static_cast<size_t>(C_out);
    std::vector<float> B(rows * cols);
    
    for (int cout = 0; cout < C_out; ++cout) {
        size_t r_idx = 0;
        for (int cin = 0; cin < C_in; ++cin)
            for (int kh = 0; kh < K; ++kh)
                for (int kw = 0; kw < K; ++kw)
                    B[r_idx++ * cols + cout] = kernel(cout, cin, kh, kw);
    }
    return B;
}

template<GemmType G>
Tensor conv_im2col(const Tensor& input, const Tensor& kernel) {
    assert(input.C == kernel.C && "Channel mismatch!");
    assert(kernel.H == kernel.W && "Kernel must be square!");
    
    int K = kernel.H;
    size_t Ar, Ac, Br, Bc;
    auto A = im2col(input, K, Ar, Ac);
    auto B = kernel2mat(kernel, Br, Bc);
    
    std::vector<float> C(Ar * Bc);
    Gemm<G>::run(A, B, C, Ar, Ac, Bc);
    
    int N_out = input.N, OH = input.H - K + 1, OW = input.W - K + 1, OC = kernel.N;
    Tensor output(N_out, OC, OH, OW);
    
    size_t r = 0;
    for (int n = 0; n < N_out; ++n)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                for (int oc = 0; oc < OC; ++oc)
                    output(n, oc, oh, ow) = C[r * OC + oc];
                ++r;
            }
    return output;
}

template Tensor conv_im2col<GemmType::NAIVE>(const Tensor&, const Tensor&);
template Tensor conv_im2col<GemmType::CACHE_FRIENDLY>(const Tensor&, const Tensor&);
template Tensor conv_im2col<GemmType::AVX2>(const Tensor&, const Tensor&);