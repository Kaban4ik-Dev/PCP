#pragma once
#include <sycl/sycl.hpp>
#include <chrono>

// Каждый work-item считает один элемент C[i][j]
void matmul_gpu_naive(sycl::queue& q,
                      const float* h_A, const float* h_B, float* h_C,
                      size_t N, float& ms)
{
    const size_t bytes = N * N * sizeof(float);

    float* d_A = sycl::malloc_device<float>(N * N, q);
    float* d_B = sycl::malloc_device<float>(N * N, q);
    float* d_C = sycl::malloc_device<float>(N * N, q);

    // Копируем данные на устройство (не входит в замер)
    q.memcpy(d_A, h_A, bytes);
    q.memcpy(d_B, h_B, bytes);
    q.wait();

    // Замеряем ТОЛЬКО ядро
    auto t0 = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> idx) {
            const size_t i = idx[0];
            const size_t j = idx[1];
            float sum = 0.0f;
            for (size_t k = 0; k < N; k++)
                sum += d_A[i * N + k] * d_B[k * N + j];
            d_C[i * N + j] = sum;
        });
    }).wait();
    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Забираем результат
    q.memcpy(h_C, d_C, bytes).wait();

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);
}
