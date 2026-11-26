#pragma once
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstddef>

// Однопоточный двухпроходный сепарабельный фильтр Гаусса на CPU (планарный формат CHW)
inline std::vector<float> gaussian_cpu(
    const std::vector<float>& in,
    int width,
    int height,
    int channels,
    const std::vector<float>& kernel,
    int radius,
    float& kernel_ms)
{
    const size_t plane = (size_t)width * (size_t)height;
    const size_t total = plane * (size_t)channels;
    std::vector<float> temp(total, 0.0f);
    std::vector<float> out(total, 0.0f);

    const float* kptr = kernel.data();

    auto t0 = std::chrono::steady_clock::now();

    // Проход 1: горизонтальная свёртка
    for (int c = 0; c < channels; ++c) {
        const float* in_c = in.data() + (size_t)c * plane;
        float* tmp_c = temp.data() + (size_t)c * plane;

        for (int y = 0; y < height; ++y) {
            const size_t row_off = (size_t)y * (size_t)width;
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = std::clamp(x + dx, 0, width - 1);
                    sum += in_c[row_off + (size_t)sx] * kptr[dx + radius];
                }
                tmp_c[row_off + (size_t)x] = sum;
            }
        }
    }

    // Проход 2: вертикальная свёртка
    for (int c = 0; c < channels; ++c) {
        const float* tmp_c = temp.data() + (size_t)c * plane;
        float* out_c = out.data() + (size_t)c * plane;

        for (int y = 0; y < height; ++y) {
            const size_t row_off = (size_t)y * (size_t)width;
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = std::clamp(y + dy, 0, height - 1);
                    sum += tmp_c[(size_t)sy * (size_t)width + (size_t)x] * kptr[dy + radius];
                }
                out_c[row_off + (size_t)x] = sum;
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    kernel_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return out;
}
