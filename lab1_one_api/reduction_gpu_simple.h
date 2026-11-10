#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <cstdint>

// Простая редукция на GPU: каждый поток суммирует свой элемент в глобальный массив
// q          — очередь SYCL
// data       — исходный массив на CPU
// kernel_ms  — выход - время выполнения ядра в миллисекундах
// Возвращает сумму всех элементов
inline float reduction_gpu_simple(sycl::queue& q, const std::vector<float>& data, float& kernel_ms)
{
    // Размер данных
    const size_t N = data.size();

    // Массив входных данных на GPU
    float* d_in = sycl::malloc_device<float>(N, q);
    // Массив выходных данных на GPU
    float* d_out = sycl::malloc_device<float>(N, q);
    // Копирование входных данных с CPU на GPU
    q.memcpy(d_in, data.data(), N * sizeof(float)).wait();

    // Размер рабочей группы (число потоков)
    const size_t WG = 256;
    // Количество рабочих групп
    const size_t num_blocks = (N + WG - 1) / WG;

    // Запуск работы ядра GPU
    sycl::event e = q.submit([&](sycl::handler& h) {
        // Параллельное выполнение с аргументами: глобальный размер, размер рабочей группы
        h.parallel_for(sycl::nd_range<1>{num_blocks* WG, WG}, [=](sycl::nd_item<1> it) {
            // Каждый поток получает свой глобальный индекс 
            size_t i = it.get_global_id(0);
            // Пишет в глобальную память, начиная со своего индекса, с защитой от выхода за границы
            if (i < N) d_out[i] = d_in[i];
            });
        });
    // Ожидание завершения задачи ядра
    e.wait();
    // Профилирование времени выполнения задачи
    auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    kernel_ms = (f - s) / 1e6;

    // Освобождение памяти GPU
    sycl::free(d_in, q);

    // Финальная редукция на CPU
    std::vector<float> tmp(N);
    q.memcpy(tmp.data(), d_out, N * sizeof(float)).wait();
    float sum = 0;
    for (auto v : tmp) sum += v;

    // Освобождение памяти GPU
    sycl::free(d_out, q);

    return sum;
}