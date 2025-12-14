#ifndef MATMUL_H
#define MATMUL_H

#include "tensor.h"

// Naive matrix multiplication: C = A @ B
// A shape: (M, K)
// B shape: (K, N)
// C shape: (M, N)
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Matrix multiplication with bias: C = (A @ B) + bias
// bias shape: (N,)
void matmul_bias(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);

// Transpose a 2D tensor
Tensor transpose_2d(const Tensor& A);

#endif // MATMUL_H
