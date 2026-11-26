#pragma once
#include <cstddef>

void scan_cpu(const float* x, float* y, size_t n)
{
    if (n == 0) return;
    y[0] = x[0];
    for (size_t i = 1; i < n; i++)
        y[i] = y[i - 1] + x[i];
}
