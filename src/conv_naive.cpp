#include "conv_naive.hpp"

Tensor conv_naive(const Tensor& input, const Tensor& kernel) {
    int N = input.N, C_in = input.C, H = input.H, W = input.W;
    int C_out = kernel.N;  
    int K = kernel.H;
    
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    Tensor output(N, C_out, H_out, W_out);
    
    for (int n = 0; n < N; ++n)
        for (int cout = 0; cout < C_out; ++cout)
            for (int h = 0; h < H_out; ++h)
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                    for (int cin = 0; cin < C_in; ++cin)
                        for (int kh = 0; kh < K; ++kh)
                            for (int kw = 0; kw < K; ++kw)
                                sum += input(n, cin, h + kh, w + kw) * kernel(cout, cin, kh, kw);
                    output(n, cout, h, w) = sum;
                }
    return output;
}