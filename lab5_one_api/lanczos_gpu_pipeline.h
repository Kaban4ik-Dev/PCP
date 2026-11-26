#pragma once
#include "lanczos_kernel.h"

// ============================================================================
//  Конвейерная реализация.
//
//  Схема:
//    1) Горизонтальный проход — ОДНО ядро по всему изображению
//       (он дешёвый и не требует halo; копировать его по полосам
//        смысла мало — накладные расходы съедят выигрыш).
//    2) Вертикальный проход — КОНВЕЙЕР по полосам:
//         для каждой полосы: копирование d_tmp-строк (с halo) -> ядро -> копирование выхода
//       с перекрытием через несколько слотов.
//
//  Halo по вертикали:
//    При ресемплинге H -> outH окно выборки Ланцоша имеет полуширину
//    SUPPORT = ceil(a * max(1, scale)) в координатах исходных строк.
//    Поэтому для полосы выходных строк [oy0, oy1) нужны входные строки
//    [src_lo, src_hi), где src_lo/src_hi вычисляются по таблицам taps.
//    Грузим ровно этот диапазон (вместо rows_in_per_chunk + 2*SUPPORT) —
//    это точная граница, а не оценка сверху.
// ============================================================================

std::vector<float> lanczos_gpu_pipeline(sycl::queue& q,
    const std::vector<float>& src,
    const LanczosParams& P,
    float& kernel_ms,
    size_t NUM_CHUNKS = 8)
{
    if (src.size() != P.inH * P.inW * P.channels) {
        std::cerr << "[ERR] GPU pipeline: неверный размер изображения.\n";
        exit(1);
    }
    if (NUM_CHUNKS == 0) NUM_CHUNKS = 1;
    if (P.outH % NUM_CHUNKS != 0) {
        std::cerr << "[ERR] GPU pipeline: outH должно делиться на NUM_CHUNKS.\n";
        exit(1);
    }
    const int rows_out_per_chunk = P.outH / NUM_CHUNKS;

    AxisTables th = build_axis_tables(P.inW, P.outW);
    AxisTables tv = build_axis_tables(P.inH, P.outH);

    // ------------------------------------------------------------------
    //  Общие буферы устройства
    // ------------------------------------------------------------------
    const size_t src_elems = P.inH * P.inW * P.channels;
    const size_t tmp_elems = P.inH * P.outW * P.channels;
    const size_t dst_elems = P.outH * P.outW * P.channels;

    float* d_src = sycl::malloc_device<float>(src_elems, q);
    float* d_tmp = sycl::malloc_device<float>(tmp_elems, q);

    int* d_th = sycl::malloc_device<int>(th.taps.size(), q);
    float* d_wh = sycl::malloc_device<float>(th.w.size(), q);
    int* d_tv = sycl::malloc_device<int>(tv.taps.size(), q);
    float* d_wv = sycl::malloc_device<float>(tv.w.size(), q);

    q.memcpy(d_src, src.data(), src_elems * sizeof(float));
    q.memcpy(d_th, th.taps.data(), th.taps.size() * sizeof(int));
    q.memcpy(d_wh, th.w.data(), th.w.size() * sizeof(float));
    q.memcpy(d_tv, tv.taps.data(), tv.taps.size() * sizeof(int));
    q.memcpy(d_wv, tv.w.data(), tv.w.size() * sizeof(float));

    // ------------------------------------------------------------------
    //  Шаг 1: горизонтальный проход по всему изображению
    // ------------------------------------------------------------------
    sycl::event e_h = q.submit([&](sycl::handler& h) {
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
    e_h.wait();   // дальше читаем d_tmp полосами — нужен готовый буфер

    // ------------------------------------------------------------------
    //  Шаг 2: конвейер по полосам для вертикального прохода
    // ------------------------------------------------------------------
    const size_t num_slots = 4;
    std::vector<float*> d_dst_slots(num_slots);
    for (size_t s = 0; s < num_slots; ++s) {
        d_dst_slots[s] = sycl::malloc_device<float>(
            (size_t)rows_out_per_chunk * P.outW * P.channels, q);
    }

    std::vector<sycl::event> kernel_events;
    kernel_events.reserve(NUM_CHUNKS);
    std::vector<sycl::event> copy_out_events;
    copy_out_events.reserve(NUM_CHUNKS);

    std::vector<float> result(dst_elems, 0.0f);

    // Для каждой полосы заранее вычислим src_lo/src_hi по таблицам taps
    // (это дешёвая операция на CPU, O(rows_out_per_chunk * TAPS))
    struct ChunkInfo { int out_oy0, out_oy1, src_lo, src_hi; };
    std::vector<ChunkInfo> chunks(NUM_CHUNKS);
    for (size_t c = 0; c < NUM_CHUNKS; ++c) {
        const int oy0 = (int)c * rows_out_per_chunk;
        const int oy1 = oy0 + rows_out_per_chunk;
        int lo = INT_MAX, hi = INT_MIN;
        for (int y = oy0; y < oy1; ++y) {
            for (int k = 0; k < LANCZOS_TAPS; ++k) {
                const int t = tv.taps[(size_t)y * LANCZOS_TAPS + k];
                lo = std::min(lo, t);
                hi = std::max(hi, t);
            }
        }
        chunks[c] = { oy0, oy1, lo, hi + 1 };
    }

    for (size_t c = 0; c < NUM_CHUNKS; ++c)
    {
        const size_t slot = c % num_slots;
        const ChunkInfo& ci = chunks[c];

        // Ждём освобождения слота (если он уже использовался)
        if (c >= num_slots) {
            copy_out_events[c - num_slots].wait();
        }

        float* d_dst_c = d_dst_slots[slot];

        // 1) Вертикальное ядро для полосы (читает d_tmp с halo из src_lo..src_hi)
        sycl::event e_v = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<3>(rows_out_per_chunk, P.outW, P.channels),
                [=](sycl::id<3> id) {
                    const int ly = id[0], x = id[1], ch = id[2];
                    const int oy = ci.out_oy0 + ly;
                    float s = 0.0f;
                    for (int k = 0; k < LANCZOS_TAPS; ++k) {
                        const int sy = d_tv[(size_t)oy * LANCZOS_TAPS + k];
                        // sy гарантированно в [src_lo, src_hi) — читаем напрямую
                        s += d_tmp[((size_t)sy * P.outW + x) * P.channels + ch]
                            * d_wv[(size_t)oy * LANCZOS_TAPS + k];
                    }
                    d_dst_c[((size_t)ly * P.outW + x) * P.channels + ch] = s;
                });
            });
        kernel_events.push_back(e_v);

        // 2) Копирование выхода полосы обратно на CPU
        copy_out_events.push_back(q.memcpy(
            result.data() + (size_t)ci.out_oy0 * P.outW * P.channels,
            d_dst_c,
            (size_t)rows_out_per_chunk * P.outW * P.channels * sizeof(float)));
    }

    // Дождаться ВСЕХ копирований (а не только последних num_slots)
    for (auto& ev : copy_out_events) ev.wait();

    // Профилирование: от старта горизонтального ядра до конца последнего вертикального
    auto s1 = e_h.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f2 = kernel_events.back()
        .get_profiling_info<sycl::info::event_profiling::command_end>();
    kernel_ms = (f2 - s1) / 1e6;

    // Освобождение ресурсов
    for (size_t s = 0; s < num_slots; ++s)
        sycl::free(d_dst_slots[s], q);
    sycl::free(d_src, q); sycl::free(d_tmp, q);
    sycl::free(d_th, q);  sycl::free(d_wh, q);
    sycl::free(d_tv, q);  sycl::free(d_wv, q);

    return result;
}