#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <vector>

#ifndef SCAN_BLOCK
#define SCAN_BLOCK 256
#endif

inline void scan_gpu_pipeline(sycl::queue &q, const float *h_x, float *h_y,
                              size_t n, float &ms,
                              size_t chunk_elems = (1ull << 20)) {
  if (n == 0) {
    ms = 0.0f;
    return;
  }

  const size_t B = SCAN_BLOCK;
  const size_t eff_chunk = std::max<size_t>(B, std::min(chunk_elems, n));
  const size_t max_blocks = (eff_chunk + B - 1) / B;

  float *d_in = sycl::malloc_device<float>(eff_chunk, q);
  float *d_out = sycl::malloc_device<float>(eff_chunk, q);
  float *d_block_sums = sycl::malloc_device<float>(max_blocks, q);
  float *d_offsets = sycl::malloc_device<float>(max_blocks, q);

  std::vector<float> h_sums(max_blocks);
  std::vector<float> h_off(max_blocks);

  float running_offset = 0.0f;

  auto t0 = std::chrono::steady_clock::now();

  for (size_t off = 0; off < n; off += eff_chunk) {
    const size_t len = std::min(eff_chunk, n - off);
    const size_t num_blocks = (len + B - 1) / B;
    const size_t global_size = num_blocks * B;

    q.memcpy(d_in, h_x + off, len * sizeof(float)).wait();

    // 1) Внутриблочный scan в локальной памяти
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<float, 1> loc(B, h);
       h.parallel_for(sycl::nd_range<1>(global_size, B),
                      [=](sycl::nd_item<1> it) {
                        const size_t gid = it.get_global_id(0);
                        const size_t lid = it.get_local_id(0);
                        const size_t bid = it.get_group(0);

                        loc[lid] = (gid < len) ? d_in[gid] : 0.0f;
                        sycl::group_barrier(it.get_group());

                        for (size_t d = 1; d < B; d *= 2) {
                          float val = (lid >= d) ? loc[lid - d] : 0.0f;
                          sycl::group_barrier(it.get_group());
                          loc[lid] += val;
                          sycl::group_barrier(it.get_group());
                        }
                        if (gid < len)
                          d_out[gid] = loc[lid];
                        if (lid == B - 1 && bid < num_blocks)
                          d_block_sums[bid] = loc[lid];
                      });
     }).wait();

    // 2) Префиксная сумма по блокам внутри чанка на хосте
    q.memcpy(h_sums.data(), d_block_sums, num_blocks * sizeof(float)).wait();
    h_off[0] = 0.0f;
    for (size_t i = 1; i < num_blocks; i++) {
      h_off[i] = h_off[i - 1] + h_sums[i - 1];
    }
    q.memcpy(d_offsets, h_off.data(), num_blocks * sizeof(float)).wait();

    // 3) Прибавление внутричанкового смещения и накопительного сдвига
    // предыдущих чанков
    const float chunk_carry = running_offset;
    q.parallel_for(sycl::nd_range<1>(global_size, B), [=](sycl::nd_item<1> it) {
       const size_t gid = it.get_global_id(0);
       const size_t bid = it.get_group(0);
       if (gid < len) {
         d_out[gid] += d_offsets[bid] + chunk_carry;
       }
     }).wait();

    q.memcpy(h_y + off, d_out, len * sizeof(float)).wait();
    running_offset = h_y[off + len - 1];
  }

  auto t1 = std::chrono::steady_clock::now();
  ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

  sycl::free(d_in, q);
  sycl::free(d_out, q);
  sycl::free(d_block_sums, q);
  sycl::free(d_offsets, q);
}
