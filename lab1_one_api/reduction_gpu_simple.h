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
    float* d_sum = sycl::malloc_device<float>(1, q);

    // Копирование входных данных с CPU на GPU
    q.memcpy(d_in, data.data(), N * sizeof(float));
    // Обнуление результата
    q.memset(d_sum, 0, sizeof(float)).wait();

    // Чисто последовательная реализация
    //sycl::event e = q.submit([&](sycl::handler& h) {
    //    // Запуск num_blocks * WG глобальных потоков, сгруппированных в группы по WG
    //    h.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
    //        sycl::atomic_ref<float, sycl::memory_order::relaxed,
    //        sycl::memory_scope::device,
    //        sycl::access::address_space::global_space> ref(*d_sum);
    //    ref.fetch_add(d_in[i]);
    //        });
    //    });
    //// Ожидание завершения задачи ядра
    //e.wait();
    
    // Число потоков
    const size_t num_threads = std::min<size_t>(N, 4096);
    // Размер рабочей группы
    const size_t WG = std::min<size_t>(num_threads, 256);
    // Количество задач (большее кратное размеру рабочей группы)
    const size_t num_items = ((num_threads + WG - 1) / WG) * WG;

    // Последовательная реализация с оптимизацией числа задач
    sycl::event e = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>{num_items, WG}, [=](sycl::nd_item<1> it) {
                const size_t gid = it.get_global_id(0);
                const size_t stride = it.get_global_range(0);

                // Локальный аккумулятор в задаче
                float local = 0.0f;
                for (size_t j = gid; j < N; j += stride)
                    local += d_in[j];

                // Один глобальный atomic на задачу
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> ref(*d_sum);
                ref.fetch_add(local);
            });
        });
    e.wait();

    // Получение результата
    float result;
    q.memcpy(&result, d_sum, sizeof(float)).wait();

    // Профилирование времени выполнения задачи
    auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto f = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    kernel_ms = (f - s) / 1e6;

    // Освобождение памяти GPU
    sycl::free(d_in, q);
    sycl::free(d_sum, q);

    return result;
}