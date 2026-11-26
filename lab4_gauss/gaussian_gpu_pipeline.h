#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

// Конвейерная обработка фильтра Гаусса на GPU с разбиением на чанки и
// использованием shared/local memory
inline std::vector<float>
gaussian_gpu_pipeline(sycl::queue &q, const std::vector<float> &h_in, int width,
                      int height, int channels,
                      const std::vector<float> &h_kernel, int radius,
                      float max_chunk_mb, float &kernel_ms) {
  const size_t plane = (size_t)width * (size_t)height;
  const size_t total = plane * (size_t)channels;
  const size_t ksize = h_kernel.size();

  // Расчёт размера чанка
  const size_t budget_bytes =
      std::max<size_t>((size_t)(max_chunk_mb * 1024.0f * 1024.0f),
                       (size_t)width * (size_t)channels * sizeof(float) * 4);
  const size_t row_all_ch_bytes =
      (size_t)width * (size_t)channels * sizeof(float);
  int chunk_rows = (int)std::max<size_t>(1, budget_bytes / row_all_ch_bytes);
  chunk_rows = std::min(chunk_rows, std::max(1, height));
  const int num_chunks = (height + chunk_rows - 1) / chunk_rows;

  const int max_in_rows = chunk_rows + 2 * radius;
  const size_t max_in_elems =
      (size_t)channels * (size_t)max_in_rows * (size_t)width;
  const size_t max_out_elems =
      (size_t)channels * (size_t)chunk_rows * (size_t)width;

  // Двойная буферизация
  float *d_in[2] = {sycl::malloc_device<float>(max_in_elems, q),
                    sycl::malloc_device<float>(max_in_elems, q)};
  float *d_temp[2] = {sycl::malloc_device<float>(max_in_elems, q),
                      sycl::malloc_device<float>(max_in_elems, q)};
  float *d_out[2] = {sycl::malloc_device<float>(max_out_elems, q),
                     sycl::malloc_device<float>(max_out_elems, q)};
  float *d_kernel = sycl::malloc_device<float>(ksize, q);
  q.memcpy(d_kernel, h_kernel.data(), ksize * sizeof(float)).wait();

  // Подготовка чанков на хосте
  std::vector<std::vector<float>> h_chunk_in(
      (size_t)num_chunks, std::vector<float>(max_in_elems, 0.0f));
  std::vector<std::vector<float>> h_chunk_out(
      (size_t)num_chunks, std::vector<float>(max_out_elems, 0.0f));

  for (int k = 0; k < num_chunks; ++k) {
    const int y0 = k * chunk_rows;
    const int y1 = std::min(y0 + chunk_rows, height);
    const int out_rows = y1 - y0;
    const int in_rows = out_rows + 2 * radius;
    for (int c = 0; c < channels; ++c) {
      const float *src_plane = h_in.data() + (size_t)c * plane;
      float *dst_plane = h_chunk_in[(size_t)k].data() +
                         (size_t)c * (size_t)in_rows * (size_t)width;
      for (int r = 0; r < in_rows; ++r) {
        int sy = y0 - radius + r;
        if (sy < 0)
          sy = 0;
        else if (sy >= height)
          sy = height - 1;
        std::memcpy(dst_plane + (size_t)r * (size_t)width,
                    src_plane + (size_t)sy * (size_t)width,
                    (size_t)width * sizeof(float));
      }
    }
  }

  std::vector<sycl::event> h_events((size_t)num_chunks);
  std::vector<sycl::event> v_events((size_t)num_chunks);
  std::vector<sycl::event> d2h_events((size_t)num_chunks);

  // Размеры тайлов для local memory
  constexpr int TILE_W = 256;
  constexpr int V_W = 16;
  constexpr int V_H = 16;

  for (int k = 0; k < num_chunks; ++k) {
    const int slot = k % 2;
    const int y0 = k * chunk_rows;
    const int y1 = std::min(y0 + chunk_rows, height);
    const int out_rows = y1 - y0;
    const int in_rows = out_rows + 2 * radius;
    const size_t cur_in_elems =
        (size_t)channels * (size_t)in_rows * (size_t)width;
    const size_t cur_out_elems =
        (size_t)channels * (size_t)out_rows * (size_t)width;

    // Копирование входных данных чанка на устройство
    sycl::event e_copy_in = q.submit([&](sycl::handler &h) {
      if (k >= 2) {
        h.depends_on(d2h_events[(size_t)(k - 2)]);
      }
      h.memcpy(d_in[slot], h_chunk_in[(size_t)k].data(),
               cur_in_elems * sizeof(float));
    });

    // 1) Горизонтальная свёртка с использованием shared memory
    const size_t groups_x_h = ((size_t)width + TILE_W - 1) / TILE_W;
    h_events[(size_t)k] = q.submit([&](sycl::handler &h) {
      h.depends_on(e_copy_in);
      const size_t loc_len = (size_t)(TILE_W + 2 * radius);
      sycl::local_accessor<float, 1> s_row(sycl::range<1>(loc_len), h);

      h.parallel_for(
          sycl::nd_range<3>(sycl::range<3>((size_t)channels, (size_t)in_rows,
                                           groups_x_h * TILE_W),
                            sycl::range<3>(1, 1, TILE_W)),
          [=](sycl::nd_item<3> it) {
            const int c = (int)it.get_global_id(0);
            const int r = (int)it.get_global_id(1);
            const int group_x = (int)it.get_group(2);
            const int tid = (int)it.get_local_id(2);
            const int tile_start_x = group_x * TILE_W;
            const int x = tile_start_x + tid;
            const size_t base =
                ((size_t)c * (size_t)in_rows + (size_t)r) * (size_t)width;

            // Загрузка строки в local memory с учётом halo
            for (int i = tid; i < TILE_W + 2 * radius; i += TILE_W) {
              int sx = tile_start_x - radius + i;
              if (sx < 0)
                sx = 0;
              else if (sx >= width)
                sx = width - 1;
              s_row[(size_t)i] = d_in[slot][base + (size_t)sx];
            }
            sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);

            // Свёртка
            if (x < width) {
              float sum = 0.0f;
              for (int kk = 0; kk < 2 * radius + 1; ++kk) {
                sum += s_row[(size_t)(tid + kk)] * d_kernel[(size_t)kk];
              }
              d_temp[slot][base + (size_t)x] = sum;
            }
          });
    });

    // 2) Вертикальная свёртка с использованием shared memory
    const size_t groups_x_v = ((size_t)width + V_W - 1) / V_W;
    const size_t groups_y_v = ((size_t)out_rows + V_H - 1) / V_H;
    v_events[(size_t)k] = q.submit([&](sycl::handler &h) {
      h.depends_on(h_events[(size_t)k]);
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
            const int y_out = tile_start_y + ly;

            const size_t chan_tmp_off =
                (size_t)c * (size_t)in_rows * (size_t)width;
            const size_t chan_out_off =
                (size_t)c * (size_t)out_rows * (size_t)width;

            const int total_rows_to_load = V_H + 2 * radius;
            for (int r = ly; r < total_rows_to_load; r += V_H) {
              int sr = tile_start_y + r;
              if (sr >= in_rows)
                sr = in_rows - 1;

              float val = 0.0f;
              if (x < width) {
                val = d_temp[slot][chan_tmp_off + (size_t)sr * (size_t)width +
                                   (size_t)x];
              }
              s_col[(size_t)r * V_W + (size_t)lx] = val;
            }
            sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);

            // Свёртка
            if (x < width && y_out < out_rows) {
              float sum = 0.0f;
              for (int kk = 0; kk < 2 * radius + 1; ++kk) {
                sum += s_col[(size_t)(ly + kk) * V_W + (size_t)lx] *
                       d_kernel[(size_t)kk];
              }
              d_out[slot][chan_out_off + (size_t)y_out * (size_t)width +
                          (size_t)x] = sum;
            }
          });
    });

    // Копирование результата чанка обратно на хост
    d2h_events[(size_t)k] = q.submit([&](sycl::handler &h) {
      h.depends_on(v_events[(size_t)k]);
      h.memcpy(h_chunk_out[(size_t)k].data(), d_out[slot],
               cur_out_elems * sizeof(float));
    });
  }

  for (auto &e : d2h_events)
    e.wait();

  // Замер времени
  uint64_t t_min = UINT64_MAX, t_max = 0;
  for (int k = 0; k < num_chunks; ++k) {
    t_min = std::min(
        t_min,
        h_events[(size_t)k]
            .get_profiling_info<sycl::info::event_profiling::command_start>());
    t_max = std::max(
        t_max,
        v_events[(size_t)k]
            .get_profiling_info<sycl::info::event_profiling::command_end>());
  }
  kernel_ms = (t_max - t_min) / 1e6f;

  // Сборка итогового изображения из чанков
  std::vector<float> h_out(total, 0.0f);
  for (int k = 0; k < num_chunks; ++k) {
    const int y0 = k * chunk_rows;
    const int y1 = std::min(y0 + chunk_rows, height);
    const int out_rows = y1 - y0;
    for (int c = 0; c < channels; ++c) {
      const float *src_out = h_chunk_out[(size_t)k].data() +
                             (size_t)c * (size_t)out_rows * (size_t)width;
      float *dst_out =
          h_out.data() + (size_t)c * plane + (size_t)y0 * (size_t)width;
      std::memcpy(dst_out, src_out,
                  (size_t)out_rows * (size_t)width * sizeof(float));
    }
  }

  for (int s = 0; s < 2; ++s) {
    sycl::free(d_in[s], q);
    sycl::free(d_temp[s], q);
    sycl::free(d_out[s], q);
  }
  sycl::free(d_kernel, q);

  return h_out;
}
