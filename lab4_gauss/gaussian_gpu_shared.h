#pragma once
#include <algorithm>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <vector>

// Оптимизированный двухпроходный фильтр Гаусса на GPU с использованием
// local_accessor
inline std::vector<float>
gaussian_gpu_shared(sycl::queue &q, const std::vector<float> &h_in, int width,
                    int height, int channels,
                    const std::vector<float> &h_kernel, int radius,
                    float &kernel_ms) {
  const size_t plane = (size_t)width * (size_t)height;
  const size_t total = plane * (size_t)channels;
  const size_t ksize = h_kernel.size();

  float *d_in = sycl::malloc_device<float>(total, q);
  float *d_temp = sycl::malloc_device<float>(total, q);
  float *d_out = sycl::malloc_device<float>(total, q);
  float *d_kernel = sycl::malloc_device<float>(ksize, q);

  q.memcpy(d_in, h_in.data(), total * sizeof(float));
  q.memcpy(d_kernel, h_kernel.data(), ksize * sizeof(float));
  q.wait();

  constexpr int TILE_W = 256;
  const size_t groups_x_h = ((size_t)width + TILE_W - 1) / TILE_W;

  // Проход 1: горизонтальный с local memory (строка тайла + левый/правый ореол)
  sycl::event e1 = q.submit([&](sycl::handler &h) {
    const size_t loc_len = (size_t)(TILE_W + 2 * radius);
    sycl::local_accessor<float, 1> s_row(sycl::range<1>(loc_len), h);

    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>((size_t)channels, (size_t)height,
                                         groups_x_h * TILE_W),
                          sycl::range<3>(1, 1, TILE_W)),
        [=](sycl::nd_item<3> it) {
          const int c = (int)it.get_global_id(0);
          const int y = (int)it.get_global_id(1);
          const int group_x = (int)it.get_group(2);
          const int tid = (int)it.get_local_id(2);

          const int tile_start_x = group_x * TILE_W;
          const int x = tile_start_x + tid;
          const size_t base = (size_t)c * plane + (size_t)y * (size_t)width;

          for (int i = tid; i < TILE_W + 2 * radius; i += TILE_W) {
            int sx = tile_start_x - radius + i;
            if (sx < 0)
              sx = 0;
            else if (sx >= width)
              sx = width - 1;
            s_row[(size_t)i] = (y < height) ? d_in[base + (size_t)sx] : 0.0f;
          }

          sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);

          if (x < width && y < height) {
            float sum = 0.0f;
            for (int k = 0; k < 2 * radius + 1; ++k) {
              sum += s_row[(size_t)(tid + k)] * d_kernel[(size_t)k];
            }
            d_temp[base + (size_t)x] = sum;
          }
        });
  });

  constexpr int V_W = 16;
  constexpr int V_H = 16;
  const size_t groups_x_v = ((size_t)width + V_W - 1) / V_W;
  const size_t groups_y_v = ((size_t)height + V_H - 1) / V_H;

  // Проход 2: вертикальный с local memory (тайл V_W x (V_H + 2*radius))
  sycl::event e2 = q.submit([&](sycl::handler &h) {
    h.depends_on(e1);
    const size_t loc_h = (size_t)(V_H + 2 * radius);
    sycl::local_accessor<float, 1> s_col(sycl::range<1>(loc_h * V_W), h);

    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>((size_t)channels, groups_y_v * V_H,
                                         groups_x_v * V_W),
                          sycl::range<3>(1, V_H, V_W)),
        [=](sycl::nd_item<3> it) {
          const int c = (int)it.get_global_id(0);
          const int group_y = (int)it.get_group(1);
          const int group_x = (int)it.get_group(2);
          const int ly = (int)it.get_local_id(1);
          const int lx = (int)it.get_local_id(2);

          const int tile_start_x = group_x * V_W;
          const int tile_start_y = group_y * V_H;
          const int x = tile_start_x + lx;
          const int y = tile_start_y + ly;

          const size_t chan_off = (size_t)c * plane;
          const int total_rows_to_load = V_H + 2 * radius;

          for (int r = ly; r < total_rows_to_load; r += V_H) {
            int sy = tile_start_y - radius + r;
            if (sy < 0)
              sy = 0;
            else if (sy >= height)
              sy = height - 1;

            float val = 0.0f;
            if (x < width) {
              val = d_temp[chan_off + (size_t)sy * (size_t)width + (size_t)x];
            }
            s_col[(size_t)r * V_W + (size_t)lx] = val;
          }

          sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);

          if (x < width && y < height) {
            float sum = 0.0f;
            for (int k = 0; k < 2 * radius + 1; ++k) {
              sum += s_col[(size_t)(ly + k) * V_W + (size_t)lx] *
                     d_kernel[(size_t)k];
            }
            d_out[chan_off + (size_t)y * (size_t)width + (size_t)x] = sum;
          }
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
