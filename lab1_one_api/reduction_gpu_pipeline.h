#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <cstdint>
#include <cstdio>

// Конвейерная редукция на GPU
// q          — очередь SYCL
// h_data     — исходный массив на CPU
// kernel_ms  — выход - время выполнения ядра в миллисекундах
// NUM_CHUNKS - количество частей, на которые разбиваются данные
// Возвращает сумму всех элементов
inline float reduction_gpu_pipeline(sycl::queue& q, const std::vector<float>& h_data, float& kernel_ms, size_t NUM_CHUNKS = 8)
{
    // Размер данных
    const size_t N = h_data.size();
    // Размер рабочей группы (число потоков внутри)
    const size_t WG = 256;

    // Размер данных, обрабатываемых за 1 итерацию конвейера
    size_t chunk = (N / NUM_CHUNKS + WG - 1) / WG * WG;
    NUM_CHUNKS = (N + chunk - 1) / chunk;

    // Количество групп для редукции одного чанка
    const size_t blocks_per_chunk = chunk / WG;
    // Общее число рабочих групп на весь массив
    const size_t total_blocks = (N + WG - 1) / WG;

    // Буфер всех данных
    float* d_in = sycl::malloc_device<float>(N, q);
    // Буфер частичных сумм
    float* d_partial = sycl::malloc_device<float>(total_blocks, q);

    // Список событий копирования и вычисления для замеров и ожидания
    std::vector<sycl::event> copy_events(NUM_CHUNKS);
    std::vector<sycl::event> kernel_events(NUM_CHUNKS);

    // Основной конвейерный цикл
    // Поскольку e_copy для k+1 не зависит от ядра k, SYCL-планировщик может выполнить их одновременно
    for (size_t k = 0; k < NUM_CHUNKS; ++k) 
    {
        const size_t off = k * chunk;
        const size_t len = std::min(chunk, N - off);

        // 1) Копирование чанка k
        copy_events[k] = q.memcpy(d_in + off,
            h_data.data() + off,
            len * sizeof(float));

        // 2) Ядро редукции чанка k
        kernel_events[k] = q.submit([&](sycl::handler& h) {
            // Каждое событие ожидает только своего копирования
            h.depends_on(copy_events[k]);
            // Выделение общей локальной памяти (на каждую группу своя)
            sycl::local_accessor<float, 1> sdata(sycl::range<1>(WG), h);

            // Запуск blocks_per_chunk * WG глобальных потоков, сгруппированных в группы по WG
            h.parallel_for(sycl::nd_range<1>{blocks_per_chunk* WG, WG}, [=](sycl::nd_item<1> it) {
                size_t tid = it.get_local_id(0);
                size_t gid = it.get_global_id(0);
                size_t off_ = k * chunk;        // локальная копия off

                // Загрузка в локальную память с защитой от выхода за границы
                size_t src = off_ + gid;
                sdata[tid] = (src < N) ? d_in[src] : float(0);
                it.barrier(sycl::access::fence_space::local_space);

                // Древовидная редукция внутри группы
                for (size_t s = WG / 2; s > 0; s /= 2) 
                {
                    if (tid < s) sdata[tid] += sdata[tid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // Запись частичной суммы группы в глобальную память
                if (tid == 0) 
                {
                    size_t global_group_id = (off_ / WG) + it.get_group(0);
                    d_partial[global_group_id] = sdata[0];
                }
                });
            });
    }

    // Замер времени (от начала первого и до окончания последнего действия из списка)
    for (auto& e : kernel_events) e.wait();
    for (auto& e : copy_events)   e.wait();

    // Учитываются минимальный старт и максимальный конец из всех событий
    uint64_t t_min = UINT64_MAX, t_max = 0;
    for (auto& e : kernel_events) 
    {
        t_min = std::min(t_min, e.get_profiling_info<sycl::info::event_profiling::command_start>());
        t_max = std::max(t_max, e.get_profiling_info<sycl::info::event_profiling::command_end>());
    }
    kernel_ms = (t_max - t_min) / 1e6;

    // Финальная редукция на хосте
    std::vector<float> partial(total_blocks);
    q.memcpy(partial.data(), d_partial, total_blocks * sizeof(float)).wait();

    float sum = 0;
    for (auto v : partial) sum += v;

    sycl::free(d_in, q);
    sycl::free(d_partial, q);

    return sum;
}