#pragma once
#include <vector>
#include <cstdint>

// Простая последовательная редукция на CPU
// data       — исходный массив на CPU
// kernel_ms  — выход - время выполнения ядра в миллисекундах
// Возвращает сумму всех элементов
inline float reduction_cpu(const std::vector<float>& data, float& kernel_ms)
{
    auto t0 = std::chrono::steady_clock::now();

    float sum = 0;
    for (size_t i = 0; i < data.size(); i++) 
        sum += data[i];

    auto t1 = std::chrono::steady_clock::now();
    kernel_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return sum;
}