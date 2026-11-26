#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

// Радиус ядра Ланцоша (a = 3 по заданию)
constexpr int LANCZOS_A = 3;
// Полуширина окна выборки
constexpr int LANCZOS_TAPS = 2 * LANCZOS_A;

// Ядро Ланцоша L(x) = sinc(x) * sinc(x/a), |x| < a
inline float lanczos_weight(float x)
{
    if (x == 0.0f) return 1.0f;
    const float ax = std::fabs(x);
    if (ax >= (float)LANCZOS_A) return 0.0f;
    const float px = (float)M_PI * x;
    const float pa = px / (float)LANCZOS_A;
    return (std::sin(px) / px) * (std::sin(pa) / pa);
}

// Таблицы весов и индексов для одного прохода.
// Для каждой выходной координаты out (0..out_size-1) храним:
//   taps  — индексы исходных пикселей (clamp к [0, in_size-1])
//   w     — веса, нормированные так, чтобы сумма = 1
struct AxisTables
{
    std::vector<int>   taps;   // [out_size * TAPS]
    std::vector<float> w;      // [out_size * TAPS]
};

// Построение таблиц для ресемплинга in_size -> out_size
inline AxisTables build_axis_tables(int in_size, int out_size)
{
    AxisTables t;
    t.taps.resize((size_t)out_size * LANCZOS_TAPS);
    t.w.resize((size_t)out_size * LANCZOS_TAPS);

    const float scale = (float)in_size / (float)out_size;   // 1 пиксель выхода в пикселях входа
    const float support = (float)LANCZOS_A * std::max(1.0f, scale); // антиалиасинг при уменьшении

    for (int o = 0; o < out_size; ++o)
    {
        // Центр окна в координатах исходного изображения
        const float center = (o + 0.5f) * scale - 0.5f;
        const int left = (int)std::floor(center - support) + 1;

        float wsum = 0.0f;
        for (int k = 0; k < LANCZOS_TAPS; ++k)
        {
            const int src_idx = left + k;
            const float dist = (center - (float)src_idx) / std::max(1.0f, scale);
            const float wv = lanczos_weight(dist);
            const int clamped = std::min(std::max(src_idx, 0), in_size - 1);
            t.taps[(size_t)o * LANCZOS_TAPS + k] = clamped;
            t.w[(size_t)o * LANCZOS_TAPS + k] = wv;
            wsum += wv;
        }
        if (std::fabs(wsum) > 1e-12f)
            for (int k = 0; k < LANCZOS_TAPS; ++k)
                t.w[(size_t)o * LANCZOS_TAPS + k] /= wsum;
    }
    return t;
}

// Параметры задачи
struct LanczosParams
{
    size_t inH, inW;   // Размеры исходного изображения
    size_t outH, outW; // Размеры результирующего изображения
    size_t channels;   // Количество каналов (4 = RGBA)
};