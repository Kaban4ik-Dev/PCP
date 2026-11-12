#pragma once
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>
#include <cstddef>

#ifndef SCAN_BLOCK
#define SCAN_BLOCK 256
#endif

void scan_gpu_blocked(sycl::queue& q, const float* h_x, float* h_y, size_t n, float& ms)
{
    const size_t B = SCAN_BLOCK;
    const size_t num_blocks = n / B;
    const size_t bytes = n * sizeof(float);

    float* d_x          = sycl::malloc_device<float>(n, q);
    float* d_y          = sycl::malloc_device<float>(n, q);
    float* d_block_sums = sycl::malloc_device<float>(num_blocks, q);
    float* d_offsets    = sycl::malloc_device<float>(num_blocks, q);

    q.memcpy(d_x, h_x, bytes).wait();

    auto t0 = std::chrono::steady_clock::now();

    // Шаг 1: внутриблочный Hillis-Steele (в локальной памяти)
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> loc(B, h);
        h.parallel_for(sycl::nd_range<1>(n, B), [=](sycl::nd_item<1> it) {
            const size_t gid = it.get_global_id(0);
            const size_t lid = it.get_local_id(0);
            const size_t bid = it.get_group(0);

            loc[lid] = d_x[gid];
            it.barrier();

            for (size_t d = 1; d < B; d *= 2) {
                float val = (lid >= d) ? loc[lid - d] : 0.0f;
                it.barrier();
                loc[lid] += val;
                it.barrier();
            }

            d_y[gid] = loc[lid];
            if (lid == B - 1)
                d_block_sums[bid] = loc[lid];
        });
    }).wait();

    // Шаг 2: эксклюзивный скан block_sums на CPU (N/B = 16384 элементов)
    {
        std::vector<float> h_sums(num_blocks), h_off(num_blocks);
        q.memcpy(h_sums.data(), d_block_sums, num_blocks * sizeof(float)).wait();
        h_off[0] = 0.0f;
        for (size_t i = 1; i < num_blocks; i++)
            h_off[i] = h_off[i - 1] + h_sums[i - 1];
        q.memcpy(d_offsets, h_off.data(), num_blocks * sizeof(float)).wait();
    }

    // Шаг 3: прибавить офсеты к результату шага 1
    q.parallel_for(sycl::nd_range<1>(n, B), [=](sycl::nd_item<1> it) {
        const size_t gid = it.get_global_id(0);
        const size_t bid = it.get_group(0);
        d_y[gid] += d_offsets[bid];
    }).wait();

    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    q.memcpy(h_y, d_y, bytes).wait();
    sycl::free(d_x, q);
    sycl::free(d_y, q);
    sycl::free(d_block_sums, q);
    sycl::free(d_offsets, q);
}
