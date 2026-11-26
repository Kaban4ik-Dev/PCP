#pragma once
#include "lanczos_kernel.h"

// Простая GPU-реализация: 2 ядра (гориз. + верт.), без локальной памяти
std::vector<float> lanczos_gpu_simple(sycl::queue& q,
    const std::vector<float>& src,
    const LanczosParams& P,
    float& kernel_ms)
{
    if (src.size() != P.inH * P.inW * P.channels) {
        std::cerr << "[ERR] GPU simple: неверный размер изображения.\n";
        exit(1);
    }

    AxisTables th = build_axis_tables(P.inW, P.outW);
    AxisTables tv = build_axis_tables(P.inH, P.outH);

    const size_t src_bytes = P.inH * P.inW * P.channels * sizeof(float);
    const size_t tmp_bytes = P.inH * P.outW * P.channels * sizeof(float);
    const size_t dst_bytes = P.outH * P.outW * P.channels * sizeof(float);

    float* d_src = sycl::malloc_device<float>(P.inH * P.inW * P.channels, q);
    float* d_tmp = sycl::malloc_device<float>(P.inH * P.outW * P.channels, q);
    float* d_dst = sycl::malloc_device<float>(P.outH * P.outW * P.channels, q);
    int* d_th = sycl::malloc_device<int>(th.taps.size(), q);
    float* d_wh = sycl::malloc_device<float>(th.w.size(), q);
    int* d_tv = sycl::malloc_device<int>(tv.taps.size(), q);
    float* d_wv = sycl::malloc_device<float>(tv.w.size(), q);

    q.memcpy(d_src, src.data(), src_bytes);
    q.memcpy(d_th, th.taps.data(), th.taps.size() * sizeof(int));
    q.memcpy(d_wh, th.w.data(), th.w.size() * sizeof(float));
    q.memcpy(d_tv, tv.taps.data(), tv.taps.size() * sizeof(int));
    q.memcpy(d_wv, tv.w.data(), tv.w.size() * sizeof(float));
    q.wait();

    sycl::event e1 = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(P.inH, P.outW, P.channels), [=](sycl::id<3> id) {
            const int y = id[0], x = id[1], c = id[2];
            float s = 0.0f;
            for (int k = 0; k < LANCZOS_TAPS; ++k) {
                const int sx = d_th[(size_t)x * LANCZOS_TAPS + k];
                s += d_src[((size_t)y * P.inW + sx) * P.channels + c]
                    * d_wh[(size_t)x * LANCZOS_TAPS + k];
            }
            d_tmp[((size_t)y * P.outW + x) * P.channels + c] = s;
            });
        });

    sycl::event e2 = q.submit([&](sycl::handler& h) {
        h.depends_on(e1);
        h.parallel_for(sycl::range<3>(P.outH, P.outW, P.channels), [=](sycl::id<3> id) {
            const int y = id[0], x = id[1], c = id[2];
            float s = 0.0f;
            for (int k = 0; k < LANCZOS_TAPS; ++k) {
                const int sy = d_tv[(size_t)y * LANCZOS_TAPS + k];
                s += d_tmp[((size_t)sy * P.outW + x) * P.channels + c]
                    * d_wv[(size_t)y * LANCZOS_TAPS + k];
            }
            d_dst[((size_t)y * P.outW + x) * P.channels + c] = s;
            });
        });
    e2.wait();

    // Суммарное время работы обоих ядер
    auto s1 = e1.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f2 = e2.get_profiling_info<sycl::info::event_profiling::command_end>();
    kernel_ms = (f2 - s1) / 1e6;

    std::vector<float> result(P.outH * P.outW * P.channels, 0.0f);
    q.memcpy(result.data(), d_dst, dst_bytes).wait();

    sycl::free(d_src, q); sycl::free(d_tmp, q); sycl::free(d_dst, q);
    sycl::free(d_th, q);  sycl::free(d_wh, q);
    sycl::free(d_tv, q);  sycl::free(d_wv, q);

    return result;
}