#pragma once
#include "lanczos_kernel.h"
#include <chrono>

// Последовательный ресемплинг (сепарабельно) на CPU
// Возвращает RGBA-изображение outH*outW*C
std::vector<float> lanczos_cpu(const std::vector<float>& src,
    const LanczosParams& P,
    float& kernel_ms)
{
    if (src.size() != P.inH * P.inW * P.channels) {
        std::cerr << "[ERR] CPU: неверный размер исходного изображения.\n";
        exit(1);
    }

    auto t0 = std::chrono::steady_clock::now();

    AxisTables th = build_axis_tables(P.inW, P.outW);
    AxisTables tv = build_axis_tables(P.inH, P.outH);

    // Промежуточный буфер: H x outW x C
    std::vector<float> tmp(P.inH * P.outW * P.channels, 0.0f);
    // Горизонтальный проход
    for (int y = 0; y < P.inH; ++y)
        for (int x = 0; x < P.outW; ++x)
            for (int c = 0; c < P.channels; ++c) {
                float s = 0.0f;
                for (int k = 0; k < LANCZOS_TAPS; ++k) {
                    const int sx = th.taps[(size_t)x * LANCZOS_TAPS + k];
                    s += src[((size_t)y * P.inW + sx) * P.channels + c]
                        * th.w[(size_t)x * LANCZOS_TAPS + k];
                }
                tmp[((size_t)y * P.outW + x) * P.channels + c] = s;
            }

    // Вертикальный проход
    std::vector<float> dst((size_t)P.outH * P.outW * P.channels, 0.0f);
    for (int y = 0; y < P.outH; ++y)
        for (int x = 0; x < P.outW; ++x)
            for (int c = 0; c < P.channels; ++c) {
                float s = 0.0f;
                for (int k = 0; k < LANCZOS_TAPS; ++k) {
                    const int sy = tv.taps[(size_t)y * LANCZOS_TAPS + k];
                    s += tmp[((size_t)sy * P.outW + x) * P.channels + c]
                        * tv.w[(size_t)y * LANCZOS_TAPS + k];
                }
                dst[((size_t)y * P.outW + x) * P.channels + c] = s;
            }

    auto t1 = std::chrono::steady_clock::now();
    kernel_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return dst;
}