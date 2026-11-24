#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstddef>

// Конвейерный двухпроходный фильтр Гаусса на GPU с разбивкой по строкам на чанки
// Размер буферов каждого чанка ограничен искусственно через max_chunk_mb
inline std::vector<float> gaussian_gpu_pipeline(
    sycl::queue& q,
    const std::vector<float>& h_in,
    int width,
    int height,
    int channels,
    const std::vector<float>& h_kernel,
    int radius,
    float max_chunk_mb,
    float& kernel_ms)
{
    const size_t plane = (size_t)width * (size_t)height;
    const size_t total = plane * (size_t)channels;
    const size_t ksize = h_kernel.size();

    // Искусственный бюджет памяти на один рабочий буфер чанка (в байтах), минимум 1 строка на канал
    const size_t budget_bytes = std::max<size_t>(
        (size_t)(max_chunk_mb * 1024.0f * 1024.0f),
        (size_t)width * (size_t)channels * sizeof(float) * 4);
    const size_t row_all_ch_bytes = (size_t)width * (size_t)channels * sizeof(float);
    int chunk_rows = (int)std::max<size_t>(1, budget_bytes / row_all_ch_bytes);
    chunk_rows = std::min(chunk_rows, std::max(1, height));

    const int num_chunks = (height + chunk_rows - 1) / chunk_rows;
    const int max_in_rows = chunk_rows + 2 * radius;
    const size_t max_in_elems = (size_t)channels * (size_t)max_in_rows * (size_t)width;
    const size_t max_out_elems = (size_t)channels * (size_t)chunk_rows * (size_t)width;

    // Двойная буферизация (2 слота) для перекрытия копирования и вычислений
    float* d_in[2] = {
        sycl::malloc_device<float>(max_in_elems, q),
        sycl::malloc_device<float>(max_in_elems, q)
    };
    float* d_temp[2] = {
        sycl::malloc_device<float>(max_in_elems, q),
        sycl::malloc_device<float>(max_in_elems, q)
    };
    float* d_out[2] = {
        sycl::malloc_device<float>(max_out_elems, q),
        sycl::malloc_device<float>(max_out_elems, q)
    };
    float* d_kernel = sycl::malloc_device<float>(ksize, q);
    q.memcpy(d_kernel, h_kernel.data(), ksize * sizeof(float)).wait();

    // Подготовка хостовых буферов чанков (с учётом вертикального halo и clamping)
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
            const float* src_plane = h_in.data() + (size_t)c * plane;
            float* dst_plane = h_chunk_in[(size_t)k].data() + (size_t)c * (size_t)in_rows * (size_t)width;

            for (int r = 0; r < in_rows; ++r) {
                int sy = y0 - radius + r;
                if (sy < 0) sy = 0;
                else if (sy >= height) sy = height - 1;

                std::memcpy(
                    dst_plane + (size_t)r * (size_t)width,
                    src_plane + (size_t)sy * (size_t)width,
                    (size_t)width * sizeof(float));
            }
        }
    }

    std::vector<sycl::event> h_events((size_t)num_chunks);
    std::vector<sycl::event> v_events((size_t)num_chunks);
    std::vector<sycl::event> d2h_events((size_t)num_chunks);

    for (int k = 0; k < num_chunks; ++k) {
        const int slot = k % 2;
        const int y0 = k * chunk_rows;
        const int y1 = std::min(y0 + chunk_rows, height);
        const int out_rows = y1 - y0;
        const int in_rows = out_rows + 2 * radius;
        const size_t cur_in_elems = (size_t)channels * (size_t)in_rows * (size_t)width;
        const size_t cur_out_elems = (size_t)channels * (size_t)out_rows * (size_t)width;

        // Если предыдущий чанк использовал тот же слот, дождёмся его выгрузки на хост
        sycl::event e_copy_in = q.submit([&](sycl::handler& h) {
            if (k >= 2) {
                h.depends_on(d2h_events[(size_t)(k - 2)]);
            }
            h.memcpy(d_in[slot], h_chunk_in[(size_t)k].data(), cur_in_elems * sizeof(float));
        });

        // 1) Горизонтальный проход по всем in_rows строкам чанка
        h_events[(size_t)k] = q.submit([&](sycl::handler& h) {
            h.depends_on(e_copy_in);
            h.parallel_for(
                sycl::range<3>((size_t)channels, (size_t)in_rows, (size_t)width),
                [=](sycl::id<3> idx) {
                    const int c = (int)idx[0];
                    const int r = (int)idx[1];
                    const int x = (int)idx[2];

                    const size_t base = ((size_t)c * (size_t)in_rows + (size_t)r) * (size_t)width;
                    float sum = 0.0f;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int sx = x + dx;
                        if (sx < 0) sx = 0;
                        else if (sx >= width) sx = width - 1;
                        sum += d_in[slot][base + (size_t)sx] * d_kernel[dx + radius];
                    }
                    d_temp[slot][base + (size_t)x] = sum;
                });
        });

        // 2) Вертикальный проход только для полезных строк [0 .. out_rows) (в temp сдвиг на +radius)
        v_events[(size_t)k] = q.submit([&](sycl::handler& h) {
            h.depends_on(h_events[(size_t)k]);
            h.parallel_for(
                sycl::range<3>((size_t)channels, (size_t)out_rows, (size_t)width),
                [=](sycl::id<3> idx) {
                    const int c = (int)idx[0];
                    const int r_out = (int)idx[1];
                    const int x = (int)idx[2];

                    const int r_center = r_out + radius;
                    const size_t chan_tmp_off = (size_t)c * (size_t)in_rows * (size_t)width;
                    const size_t chan_out_off = (size_t)c * (size_t)out_rows * (size_t)width;

                    float sum = 0.0f;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        int sr = r_center + dy;
                        sum += d_temp[slot][chan_tmp_off + (size_t)sr * (size_t)width + (size_t)x] * d_kernel[dy + radius];
                    }
                    d_out[slot][chan_out_off + (size_t)r_out * (size_t)width + (size_t)x] = sum;
                });
        });

        // Выгрузка результата чанка обратно на хост
        d2h_events[(size_t)k] = q.submit([&](sycl::handler& h) {
            h.depends_on(v_events[(size_t)k]);
            h.memcpy(h_chunk_out[(size_t)k].data(), d_out[slot], cur_out_elems * sizeof(float));
        });
    }

    for (auto& e : d2h_events) e.wait();

    uint64_t t_min = UINT64_MAX, t_max = 0;
    for (int k = 0; k < num_chunks; ++k) {
        t_min = std::min(t_min, h_events[(size_t)k].get_profiling_info<sycl::info::event_profiling::command_start>());
        t_max = std::max(t_max, v_events[(size_t)k].get_profiling_info<sycl::info::event_profiling::command_end>());
    }
    kernel_ms = (t_max - t_min) / 1e6f;

    // Сборка выходного планарного массива из чанков
    std::vector<float> h_out(total, 0.0f);
    for (int k = 0; k < num_chunks; ++k) {
        const int y0 = k * chunk_rows;
        const int y1 = std::min(y0 + chunk_rows, height);
        const int out_rows = y1 - y0;

        for (int c = 0; c < channels; ++c) {
            const float* src_out = h_chunk_out[(size_t)k].data() + (size_t)c * (size_t)out_rows * (size_t)width;
            float* dst_out = h_out.data() + (size_t)c * plane + (size_t)y0 * (size_t)width;

            std::memcpy(dst_out, src_out, (size_t)out_rows * (size_t)width * sizeof(float));
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
