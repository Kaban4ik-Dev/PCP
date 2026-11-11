#pragma once
#include <cstddef>

// C = A * B, все матрицы N×N, row-major
void matmul_cpu(const float* A, const float* B, float* C, size_t N)
{
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}
