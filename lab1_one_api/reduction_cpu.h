#pragma once
#include <vector>
#include <cstdint>

// Простая последовательная редукция на CPU
// h_data     — исходный массив на CPU
// Возвращает сумму всех элементов
inline float reduction_cpu(const std::vector<float>& data)
{
    float sum = 0;
    for (size_t i = 0; i < data.size(); i++) sum += data[i];
    return sum;
}