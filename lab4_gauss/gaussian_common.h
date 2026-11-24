#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

// Построение нормализованного 1D-ядра Гаусса длины (2 * radius + 1)
inline std::vector<float> make_gaussian_kernel_1d(int radius, float sigma)
{
    const int ksize = 2 * radius + 1;
    std::vector<float> kernel(ksize, 0.0f);

    if (radius <= 0 || sigma <= 0.0f) {
        kernel[0] = 1.0f;
        return kernel;
    }

    const float two_sigma2 = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float w = std::exp(-(float)(i * i) / two_sigma2);
        kernel[i + radius] = w;
        sum += w;
    }

    for (float& w : kernel) {
        w /= sum;
    }
    return kernel;
}

// Конвертация interleaved RGB (HWC, uint8) -> planar float (CHW, 0..255)
inline std::vector<float> interleaved_u8_to_planar_f32(
    const unsigned char* src, int width, int height, int channels)
{
    const size_t plane = (size_t)width * (size_t)height;
    std::vector<float> dst(plane * (size_t)channels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t pix = (size_t)y * (size_t)width + (size_t)x;
            for (int c = 0; c < channels; ++c) {
                dst[(size_t)c * plane + pix] = (float)src[pix * (size_t)channels + (size_t)c];
            }
        }
    }
    return dst;
}

// Конвертация planar float (CHW, 0..255) -> interleaved RGB (HWC, uint8)
inline std::vector<unsigned char> planar_f32_to_interleaved_u8(
    const std::vector<float>& src, int width, int height, int channels)
{
    const size_t plane = (size_t)width * (size_t)height;
    std::vector<unsigned char> dst(plane * (size_t)channels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t pix = (size_t)y * (size_t)width + (size_t)x;
            for (int c = 0; c < channels; ++c) {
                float v = std::clamp(src[(size_t)c * plane + pix], 0.0f, 255.0f);
                dst[pix * (size_t)channels + (size_t)c] = (unsigned char)(v + 0.5f);
            }
        }
    }
    return dst;
}
