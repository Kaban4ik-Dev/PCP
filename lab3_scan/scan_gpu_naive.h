#pragma once
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstddef>

void scan_gpu_naive(sycl::queue& q, const float* h_x, float* h_y, size_t n, float& ms)
{
    const size_t bytes = n * sizeof(float);
    float* d_a = sycl::malloc_device<float>(n, q);
    float* d_b = sycl::malloc_device<float>(n, q);
    q.memcpy(d_a, h_x, bytes).wait();

    auto t0 = std::chrono::steady_clock::now();

    float* in  = d_a;
    float* out = d_b;
    for (size_t d = 1; d < n; d *= 2) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            out[i] = in[i] + (i >= d ? in[i - d] : 0.0f);
        }).wait();
        std::swap(in, out);
    }

    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    q.memcpy(h_y, in, bytes).wait();
    sycl::free(d_a, q);
    sycl::free(d_b, q);
}
