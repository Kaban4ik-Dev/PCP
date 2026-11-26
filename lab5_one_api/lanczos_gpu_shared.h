#pragma once
#include "lanczos_kernel.h"

// ============================================================================
//  Оптимизированная реализация: тайлинг через local_accessor
//  Горизонтальный и вертикальный проходы — отдельные ядра.
//
//  Идея тайлинга:
//    - Блок считает TY x TX выходных пикселей.
//    - Для горизонтального прохода исходные столбцы, которые нужны блоку,
//      лежат в диапазоне [src_lo, src_hi), где src_lo/src_hi вычисляются
//      по таблицам taps для всех out_x блока.
//    - В local-память грузим тайл размера TY x (src_hi - src_lo) x C.
//    - Верхняя граница TILE_X = TX * LANCZOS_TAPS гарантированно покрывает
//      все taps блока (каждый out_x даёт не более LANCZOS_TAPS столбцов).
//    - Аналогично для вертикального прохода: TILE_Y = TY * LANCZOS_TAPS.
// ============================================================================

template<int TX, int TY>
std::vector<float> lanczos_gpu_shared_impl(sycl::queue& q,
    const std::vector<float>& src,
    const LanczosParams& P,
    float& kernel_ms)
{
    if (src.size() != (size_t)P.inH * P.inW * P.channels) {
        std::cerr << "[ERR] GPU shared: неверный размер изображения.\n";
        exit(1);
    }

    AxisTables th = build_axis_tables(P.inW, P.outW);
    AxisTables tv = build_axis_tables(P.inH, P.outH);

    const size_t src_elems = (size_t)P.inH * P.inW * P.channels;
    const size_t tmp_elems = (size_t)P.inH * P.outW * P.channels;
    const size_t dst_elems = (size_t)P.outH * P.outW * P.channels;

    float* d_src = sycl::malloc_device<float>(src_elems, q);
    float* d_tmp = sycl::malloc_device<float>(tmp_elems, q);
    float* d_dst = sycl::malloc_device<float>(dst_elems, q);

    int* d_th = sycl::malloc_device<int>(th.taps.size(), q);
    float* d_wh = sycl::malloc_device<float>(th.w.size(), q);
    int* d_tv = sycl::malloc_device<int>(tv.taps.size(), q);
    float* d_wv = sycl::malloc_device<float>(tv.w.size(), q);

    q.memcpy(d_src, src.data(), src_elems * sizeof(float));
    q.memcpy(d_th, th.taps.data(), th.taps.size() * sizeof(int));
    q.memcpy(d_wh, th.w.data(), th.w.size() * sizeof(float));
    q.memcpy(d_tv, tv.taps.data(), tv.taps.size() * sizeof(int));
    q.memcpy(d_wv, tv.w.data(), tv.w.size() * sizeof(float));
    q.wait();

    // Верхние границы размеров тайлов (статически известны)
    constexpr int TILE_X = TX * LANCZOS_TAPS;
    constexpr int TILE_Y = TY * LANCZOS_TAPS;

    // ------------------------------------------------------------------
    //                     Горизонтальный проход
    // ------------------------------------------------------------------
    const size_t gx = (P.outW + TX - 1) / TX;
    const size_t gy = (P.inH + TY - 1) / TY;

    sycl::event e1 = q.submit([&](sycl::handler& h) {
        // Тайл: TY строк x (не более TILE_X столбцов) x C каналов
        sycl::local_accessor<float, 3> tile({ TY, TILE_X, P.channels }, h);
        sycl::local_accessor<int, 2> l_taps({ TX, LANCZOS_TAPS }, h);
        sycl::local_accessor<float, 2> l_w({ TX, LANCZOS_TAPS }, h);
        sycl::local_accessor<int, 1> l_lo({ 1 }, h);   // src_lo для блока
        sycl::local_accessor<int, 1> l_hi({ 1 }, h);   // src_hi (эксклюзивно)

        h.parallel_for(sycl::nd_range<2>({ gy * TY, gx * TX }, { TY, TX }),
            [=](sycl::nd_item<2> it) {
                const int ly = it.get_local_id(0);
                const int lx = it.get_local_id(1);
                const int by = it.get_group(0);
                const int bx = it.get_group(1);

                const int out_x0 = bx * TX;
                const int in_y0 = by * TY;

                // 1) Веса/индексы в local — каждый поток грузит все taps своего out_x
                {
                    const int x = out_x0 + lx;
                    if (x < P.outW) {
                        for (int k = 0; k < LANCZOS_TAPS; ++k) {
                            l_taps[lx][k] = d_th[(size_t)x * LANCZOS_TAPS + k];
                            l_w[lx][k] = d_wh[(size_t)x * LANCZOS_TAPS + k];
                        }
                    }
                }

                // 2) src_lo / src_hi по всем out_x блока (делает один поток)
                if (ly == 0 && lx == 0) {
                    int lo = INT_MAX, hi = INT_MIN;
                    for (int xx = 0; xx < TX; ++xx) {
                        const int x = out_x0 + xx;
                        if (x >= P.outW) continue;
                        for (int k = 0; k < LANCZOS_TAPS; ++k) {
                            const int t = d_th[(size_t)x * LANCZOS_TAPS + k];
                            lo = sycl::min(lo, t);
                            hi = sycl::max(hi, t);
                        }
                    }
                    l_lo[0] = lo;
                    l_hi[0] = hi + 1;   // эксклюзивно
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int src_lo = l_lo[0];
                const int tile_x = l_hi[0] - src_lo;   // <= TILE_X

                // 3) Загрузка входного тайла: TY строк x tile_x столбцов x C каналов
                for (int ty = ly; ty < TY; ty += TY) {
                    const int sy = in_y0 + ty;
                    if (sy >= P.inH) continue;
                    for (int tx = lx; tx < tile_x; tx += TX) {
                        const int sx = src_lo + tx;
                        for (int c = 0; c < P.channels; ++c) {
                            tile[ty][tx][c] =
                                d_src[((size_t)sy * P.inW + sx) * P.channels + c];
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // 4) Вычисление для своего (oy, ox)
                const int oy = in_y0 + ly;
                const int ox = out_x0 + lx;
                if (oy < P.inH && ox < P.outW) {
                    for (int c = 0; c < P.channels; ++c) {
                        float s = 0.0f;
                        for (int k = 0; k < LANCZOS_TAPS; ++k) {
                            const int sx_local = l_taps[lx][k] - src_lo;
                            s += tile[ly][sx_local][c] * l_w[lx][k];
                        }
                        d_tmp[((size_t)oy * P.outW + ox) * P.channels + c] = s;
                    }
                }
            });
        });

    // ------------------------------------------------------------------
    //                      Вертикальный проход
    // ------------------------------------------------------------------
    const size_t gx2 = (P.outW + TX - 1) / TX;
    const size_t gy2 = (P.outH + TY - 1) / TY;

    sycl::event e2 = q.submit([&](sycl::handler& h) {
        h.depends_on(e1);
        // Тайл: (не более TILE_Y строк) x TX столбцов x C каналов
        sycl::local_accessor<float, 3> tile({ TILE_Y, TX, P.channels }, h);
        sycl::local_accessor<int, 2> l_taps({ TY, LANCZOS_TAPS }, h);
        sycl::local_accessor<float, 2> l_w({ TY, LANCZOS_TAPS }, h);
        sycl::local_accessor<int, 1> l_lo({ 1 }, h);
        sycl::local_accessor<int, 1> l_hi({ 1 }, h);

        h.parallel_for(sycl::nd_range<2>({ gy2 * TY, gx2 * TX }, { TY, TX }),
            [=](sycl::nd_item<2> it) {
                const int ly = it.get_local_id(0);
                const int lx = it.get_local_id(1);
                const int by = it.get_group(0);
                const int bx = it.get_group(1);

                const int out_y0 = by * TY;
                const int out_x0 = bx * TX;

                // 1) Веса/индексы для своего out_y
                {
                    const int y = out_y0 + ly;
                    if (y < P.outH) {
                        for (int k = 0; k < LANCZOS_TAPS; ++k) {
                            l_taps[ly][k] = d_tv[(size_t)y * LANCZOS_TAPS + k];
                            l_w[ly][k] = d_wv[(size_t)y * LANCZOS_TAPS + k];
                        }
                    }
                }

                // 2) src_lo / src_hi по всем out_y блока
                if (ly == 0 && lx == 0) {
                    int lo = INT_MAX, hi = INT_MIN;
                    for (int yy = 0; yy < TY; ++yy) {
                        const int y = out_y0 + yy;
                        if (y >= P.outH) continue;
                        for (int k = 0; k < LANCZOS_TAPS; ++k) {
                            const int t = d_tv[(size_t)y * LANCZOS_TAPS + k];
                            lo = sycl::min(lo, t);
                            hi = sycl::max(hi, t);
                        }
                    }
                    l_lo[0] = lo;
                    l_hi[0] = hi + 1;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int src_lo = l_lo[0];
                const int tile_y = l_hi[0] - src_lo;   // <= TILE_Y

                // 3) Загрузка тайла из d_tmp: tile_y строк x TX столбцов x C
                for (int ty = ly; ty < tile_y; ty += TY) {
                    const int sy = src_lo + ty;
                    for (int tx = lx; tx < TX; tx += TX) {
                        const int sx = out_x0 + tx;
                        if (sx >= P.outW) continue;
                        for (int c = 0; c < P.channels; ++c) {
                            tile[ty][tx][c] =
                                d_tmp[((size_t)sy * P.outW + sx) * P.channels + c];
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // 4) Вычисление
                const int oy = out_y0 + ly;
                const int ox = out_x0 + lx;
                if (oy < P.outH && ox < P.outW) {
                    for (int c = 0; c < P.channels; ++c) {
                        float s = 0.0f;
                        for (int k = 0; k < LANCZOS_TAPS; ++k) {
                            const int sy_local = l_taps[ly][k] - src_lo;
                            s += tile[sy_local][lx][c] * l_w[ly][k];
                        }
                        d_dst[((size_t)oy * P.outW + ox) * P.channels + c] = s;
                    }
                }
            });
        });
    e2.wait();

    auto s1 = e1.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f2 = e2.get_profiling_info<sycl::info::event_profiling::command_end>();
    kernel_ms = (f2 - s1) / 1e6;

    std::vector<float> result(dst_elems, 0.0f);
    q.memcpy(result.data(), d_dst, dst_elems * sizeof(float)).wait();

    sycl::free(d_src, q); sycl::free(d_tmp, q); sycl::free(d_dst, q);
    sycl::free(d_th, q);  sycl::free(d_wh, q);
    sycl::free(d_tv, q);  sycl::free(d_wv, q);

    return result;
}

std::vector<float> lanczos_gpu_shared(sycl::queue& q,
    const std::vector<float>& src,
    const LanczosParams& P,
    float& kernel_ms)
{
    return lanczos_gpu_shared_impl<16, 16>(q, src, P, kernel_ms);
}