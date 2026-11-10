#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <clocale>
#include <iostream>

#include "reduction_cpu.h"
#include "reduction_gpu_simple.h"
#include "reduction_gpu_shared.h"
#include "reduction_gpu_pipeline.h"
#include "config_loader.h"
#include "metrics.h"

// Количество разных seed
const size_t SEED_CNT = 5;
// Количество прогонов на одном и том же seed
const size_t RUN_CNT = 10;

// Функция выбора подходящего графического устройства
sycl::device pick_device()
{
    // Проверка существования GPU
    try {
        auto gpu = sycl::device{ sycl::gpu_selector_v };
        //if (!gpu.has(sycl::aspect::fp64)) throw std::runtime_error("Device has no float support\n");
        std::cout << "[INFO] Используется GPU: " << gpu.get_info<sycl::info::device::name>() << "\n";
        return gpu;
    }
    // Проверка существования CPU
    catch (const sycl::exception& e) {
        try {
            std::cerr << "[WARN] GPU не найден (" << e.what() << "), поиск CPU.\n";
            auto cpu = sycl::device{ sycl::cpu_selector_v };
            std::cout << "[INFO] Используется CPU: " << cpu.get_info<sycl::info::device::name>() << "\n";
            return cpu;
        }
        // Если совместимых устройств нет - выход
        catch (const sycl::exception& e) {
            std::cerr << "[ERR] GPU и CPU не найдены (" << e.what() << "), завершение работы.\n";
            exit(1);
            return;
        }
    }
}

// Функция расчёта медианы в векторе
float median(std::vector<float> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size() / 2;
    return (v.size() % 2)
        ? v[n]
        : 0.5 * (v[n - 1] + v[n]);
}

// Функция проверки корректности суммы
bool approx_equal(double a, double b, double rel_tol = 1e-4) 
{
    double diff = std::fabs(a - b);
    double scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= rel_tol * scale;
};


// ============================================
// ===      Основная функция программы      ===
// ============================================
int main() 
{
    setlocale(LC_ALL, "RU");
    const size_t N = 1ull << 25;               // Количество элементов данных
    const size_t elem_size = sizeof(float);    // Размер одного элемента данных

    // Параметры устройств
    DeviceSpecs specs;
    cfg::load_config("config.ini", specs);

    // Создание ГПСЧ
    std::printf("Размер входных данных: N = %zu\n", N);
    std::vector<float> h_data(N);

    // ---------- Очередь SYCL ----------
    sycl::queue q{ pick_device(), sycl::property::queue::enable_profiling() };

    // Структура результатов
    Results res;
    res.N = N;
    res.elem_size = elem_size;

    // Промежуточные массивы результатов
    const size_t s = SEED_CNT * RUN_CNT;
    std::vector<float> cpu_ms(s), memcpy_ms(s), gpu_simple_ms(s), gpu_shared_ms(s), gpu_pipeline_ms(s);
    // Буфер для замера времени копирования
    float* d_buf = sycl::malloc_device<float>(N, q);

    // Цикл по разным значениям seed
    for (size_t i = 0; i < SEED_CNT; i++)
    {
        // Генерация данных по seed текущей итерации
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + i;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
        for (auto& v : h_data) v = dist(rng);

        // Цикл одинаковых замеров для получения средних данных
        for (size_t j = 0; j < RUN_CNT; j++)
        {
            const size_t idx = i * RUN_CNT + j;

            // ---------- CPU ----------
            reduction_cpu(h_data); // Прогрев
            auto t0 = std::chrono::steady_clock::now();
            float tmp_cpu_sum = reduction_cpu(h_data);
            auto t1 = std::chrono::steady_clock::now();
            cpu_ms[idx] = std::chrono::duration<float, std::milli>(t1 - t0).count();

            // ---------- Замер времени передачи данных ----------
            q.memcpy(d_buf, h_data.data(), N * elem_size).wait(); // Прогрев
            auto m0 = std::chrono::steady_clock::now();
            sycl::event e = q.memcpy(d_buf, h_data.data(), N * elem_size);
            e.wait();
            auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto f = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            auto m1 = std::chrono::steady_clock::now();
            memcpy_ms[idx] = std::chrono::duration<float, std::milli>(m1 - m0).count();

            // ---------- Простая редукция на GPU ----------
            float tmp_gpu_simple_sum = reduction_gpu_simple(q, h_data, gpu_simple_ms[idx]);

            // ---------- Оптимизированная редукция на GPU ----------
            float tmp_gpu_shared_sum = reduction_gpu_shared(q, h_data, gpu_shared_ms[idx]);

            // ---------- Конвейерная редукция на GPU ----------
            float tmp_gpu_pipeline_sum = reduction_gpu_pipeline(q, h_data, gpu_pipeline_ms[idx], 8);

            // ---------- Проверка корректности ----------
            if (!approx_equal(tmp_cpu_sum, tmp_gpu_simple_sum, 1e-2) ||
                !approx_equal(tmp_cpu_sum, tmp_gpu_shared_sum, 1e-2) ||
                !approx_equal(tmp_cpu_sum, tmp_gpu_pipeline_sum, 1e-2)) {
                std::printf("[ERR] Результаты seed %3zu итерации %3zu сложения расходятся.\n", i + 1, j + 1);
            }
        }
    }

    // Освобождение ресурсов
    sycl::free(d_buf, q);

    // Вычисление результатов по медиане
    res.cpu_ms = median(cpu_ms);
    res.memcpy_ms = median(memcpy_ms);
    res.gpu_simple_ms = median(gpu_simple_ms);
    res.gpu_shared_ms = median(gpu_shared_ms);
    res.gpu_pipeline_ms = median(gpu_pipeline_ms);
    
    std::printf("\n===================================================\n");
    std::printf("CPU время = %.2f мс\n", res.cpu_ms);
    std::printf("GPU simple время = %.3f мс\n", res.gpu_simple_ms);
    std::printf("GPU shared время = %.3f мс\n", res.gpu_shared_ms);
    std::printf("GPU pipeline время = %.3f мс\n", res.gpu_pipeline_ms);

    std::printf("\n Эффективность   | cpu_ms | gpu_simple_ms\n");
    std::printf("  gpu_shared_ms  |  %.3f |    %.3f\n", res.cpu_ms / res.gpu_shared_ms, res.gpu_simple_ms / res.gpu_shared_ms);
    std::printf(" gpu_pipeline_ms |  %.3f |    %.3f\n", res.cpu_ms / res.gpu_pipeline_ms, res.gpu_simple_ms / res.gpu_pipeline_ms);

    // ---------- Метрики ----------
    print_reduction_metrics(res, specs, q.get_device());

    return 0;
}