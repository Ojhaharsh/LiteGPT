#include "matmul.h"
#include <omp.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

void matmul_simd(const Tensor& A, const Tensor& B, Tensor& C) {
    // SIMD version for AVX2, assuming contiguous row-major
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("matmul_simd: input tensors must be 2D");
    }
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    int M = A_shape[0];
    int K = A_shape[1];
    int N = B_shape[1];
    
    if (B_shape[0] != K || C_shape[0] != M || C_shape[1] != N) {
        throw std::invalid_argument("matmul_simd: shape mismatch");
    }
    
    float* A_data = A.data();
    float* B_data = B.data();
    float* C_data = C.data();
    
    // Assume contiguous for SIMD
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 a = _mm256_loadu_ps(&A_data[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B_data[k * N + j]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            // Horizontal sum
            float total = 0.0f;
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            for (int t = 0; t < 8; ++t) total += temp[t];
            // Remaining
            for (; k < K; ++k) {
                total += A_data[i * K + k] * B_data[k * N + j];
            }
            C_data[i * N + j] = total;
        }
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    // A shape: (M, K)
    // B shape: (K, N)
    // C shape: (M, N)
    
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("matmul: input tensors must be 2D");
    }
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    int M = A_shape[0];
    int K = A_shape[1];
    int N = B_shape[1];
    
    if (B_shape[0] != K || C_shape[0] != M || C_shape[1] != N) {
        throw std::invalid_argument("matmul: shape mismatch");
    }
    
    float* A_data = A.data();
    float* B_data = B.data();
    float* C_data = C.data();
    
    auto A_strides = A.strides();
    auto B_strides = B.strides();
    auto C_strides = C.strides();
    
    // Naive triple nested loop with OpenMP parallelization
    #ifdef __AVX2__
        if (false) {
            matmul_simd(A, B, C);
        } else {
            // #pragma omp parallel for collapse(2)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        size_t A_idx = i * A_strides[0] + k * A_strides[1];
                        size_t B_idx = k * B_strides[0] + j * B_strides[1];
                        sum += A_data[A_idx] * B_data[B_idx];
                    }
                    size_t C_idx = i * C_strides[0] + j * C_strides[1];
                    C_data[C_idx] = sum;
                }
            }
        }
    #else
        // #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    size_t A_idx = i * A_strides[0] + k * A_strides[1];
                    size_t B_idx = k * B_strides[0] + j * B_strides[1];
                    sum += A_data[A_idx] * B_data[B_idx];
                }
                size_t C_idx = i * C_strides[0] + j * C_strides[1];
                C_data[C_idx] = sum;
            }
        }
    #endif
}

void matmul_bias(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
    // First do matmul
    matmul(A, B, C);
    
    // Then add bias
    if (bias.ndim() != 1) {
        throw std::invalid_argument("matmul_bias: bias must be 1D");
    }
    
    auto C_shape = C.shape();
    int M = C_shape[0];
    int N = C_shape[1];
    
    if (bias.shape()[0] != N) {
        throw std::invalid_argument("matmul_bias: bias shape mismatch");
    }
    
    float* C_data = C.data();
    float* bias_data = bias.data();
    auto C_strides = C.strides();
    
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t C_idx = i * C_strides[0] + j * C_strides[1];
            C_data[C_idx] += bias_data[j];
        }
    }
}

Tensor transpose_2d(const Tensor& A) {
    if (A.ndim() != 2) {
        throw std::invalid_argument("transpose_2d: input must be 2D");
    }
    
    auto shape = A.shape();
    Tensor result({shape[1], shape[0]});
    
    float* A_data = A.data();
    float* result_data = result.data();
    auto A_strides = A.strides();
    auto result_strides = result.strides();
    
    int M = shape[0];
    int N = shape[1];
    
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t A_idx = i * A_strides[0] + j * A_strides[1];
            size_t result_idx = j * result_strides[0] + i * result_strides[1];
            result_data[result_idx] = A_data[A_idx];
        }
    }
    
    return result;
}
