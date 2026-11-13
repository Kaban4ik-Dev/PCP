#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <cstdint>

// Редукция на GPU: каждый поток суммирует свой элемент в локальную память, которая затем объединяется с остальными частями
// q          — очередь SYCL
// h_data     — исходный массив на CPU
// kernel_ms  — выход - время выполнения ядра в миллисекундах
// Возвращает сумму всех элементов
inline float reduction_gpu_shared(sycl::queue& q, const std::vector<float>& h_data, float& kernel_ms)
{
    // Размер данных
    const size_t N = h_data.size();
    // Размер рабочей группы (число потоков внутри)
    const size_t WG = 256;
    // Общее число потоков
    const size_t num_blocks = (N + WG - 1) / WG;

    // Массив входных данных на GPU
    float* d_in = sycl::malloc_device<float>(N, q);
    // Массив выходных данных на GPU
    float* d_out = sycl::malloc_device<float>(num_blocks, q);
    // Копирование входных данных с CPU на GPU
    q.memcpy(d_in, h_data.data(), N * sizeof(float)).wait();
    // Инициализация массива результата нулями
    q.fill(d_out, float(0), num_blocks).wait();

    // Запуск работы ядра GPU
    sycl::event e = q.submit([&](sycl::handler& h) {
        // Выделение общей локальной памяти (на каждую группу своя)
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(WG), h);

        // Запуск num_blocks * WG глобальных потоков, сгруппированных в группы по WG
        h.parallel_for(sycl::nd_range<1>{num_blocks* WG, WG}, [=](sycl::nd_item<1> it) {
            // Локальный индекс внутри группы
            size_t tid = it.get_local_id(0);
            // Глобальный индекс
            size_t gid = it.get_global_id(0);

            // Загрузка в локальную память с защитой от выхода за границы
            // Тернарный оператор обеспечивает компиляцию предикатной загрузки вместо ветвления
            sdata[tid] = (gid < N) ? d_in[gid] : float(0);
            // Барьер гарантирует, что все потоки записали данные до начала редукции
            it.barrier(sycl::access::fence_space::local_space);

            // Древовидная редукция
            for (size_t s = WG / 2; s > 0; s /= 2) 
            {
                // Активны только потоки первой половины
                if (tid < s) sdata[tid] += sdata[tid + s];
                // Барьер гарантирует, что все потоки записали данные до начала следующей итерации
                it.barrier(sycl::access::fence_space::local_space);
            }

            // Поток 0 пишет результат в глобальную память
            if (tid == 0) d_out[it.get_group(0)] = sdata[0];
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

    // Возврат результатов групп на CPU
    std::vector<float> partial(num_blocks);
    q.memcpy(partial.data(), d_out, num_blocks * sizeof(float)).wait();

    // Финальная редукция на CPU
    float sum = 0;
    for (auto v : partial) sum += v;

    // Освобождение памяти GPU
    sycl::free(d_out, q);

    return sum;
}