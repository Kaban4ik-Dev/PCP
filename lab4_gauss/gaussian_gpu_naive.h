#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <cstddef>

// Наивный двухпроходный сепарабельный фильтр Гаусса на GPU (без shared/local memory)
inline std::vector<float> gaussian_gpu_naive(
    sycl::queue& q,
    const std::vector<float>& h_in,
    int width,
    int height,
    int channels,
    const std::vector<float>& h_kernel,
    int radius,
    float& kernel_ms)
{
    const size_t plane = (size_t)width * (size_t)height;
    const size_t total = plane * (size_t)channels;
    const size_t ksize = h_kernel.size();

    float* d_in     = sycl::malloc_device<float>(total, q);
    float* d_temp   = sycl::malloc_device<float>(total, q);
    float* d_out    = sycl::malloc_device<float>(total, q);
    float* d_kernel = sycl::malloc_device<float>(ksize, q);

    q.memcpy(d_in, h_in.data(), total * sizeof(float));
    q.memcpy(d_kernel, h_kernel.data(), ksize * sizeof(float));
    q.wait();

    // Проход 1: горизонтальный
    sycl::event e1 = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>((size_t)channels, (size_t)height, (size_t)width),
            [=](sycl::id<3> idx) {
                const int c = (int)idx[0];
                const int y = (int)idx[1];
                const int x = (int)idx[2];

                const size_t base = (size_t)c * plane + (size_t)y * (size_t)width;
                float sum = 0.0f;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = x + dx;
                    if (sx < 0) sx = 0;
                    else if (sx >= width) sx = width - 1;
                    sum += d_in[base + (size_t)sx] * d_kernel[dx + radius];
                }
                d_temp[base + (size_t)x] = sum;
            });
    });

    // Проход 2: вертикальный
    sycl::event e2 = q.submit([&](sycl::handler& h) {
        h.depends_on(e1);
        h.parallel_for(sycl::range<3>((size_t)channels, (size_t)height, (size_t)width),
            [=](sycl::id<3> idx) {
                const int c = (int)idx[0];
                const int y = (int)idx[1];
                const int x = (int)idx[2];

                const size_t chan_off = (size_t)c * plane;
                float sum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = y + dy;
                    if (sy < 0) sy = 0;
                    else if (sy >= height) sy = height - 1;
                    sum += d_temp[chan_off + (size_t)sy * (size_t)width + (size_t)x] * d_kernel[dy + radius];
                }
                d_out[chan_off + (size_t)y * (size_t)width + (size_t)x] = sum;
            });
    });

    e2.wait();

    auto s1 = e1.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f1 = e1.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto s2 = e2.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f2 = e2.get_profiling_info<sycl::info::event_profiling::command_end>();
    kernel_ms = ((f1 - s1) + (f2 - s2)) / 1e6f;

    std::vector<float> h_out(total);
    q.memcpy(h_out.data(), d_out, total * sizeof(float)).wait();

    sycl::free(d_in, q);
    sycl::free(d_temp, q);
    sycl::free(d_out, q);
    sycl::free(d_kernel, q);

    return h_out;
}
