#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <clocale>
#include <iostream>

#include "lanczos_kernel.h"
#include "lanczos_cpu.h"
#include "lanczos_gpu_simple.h"
#include "lanczos_gpu_shared.h"
#include "lanczos_gpu_pipeline.h"
#include "config_loader.h"
#include "metrics.h"

// Количество разных seed
const size_t SEED_CNT = 3;
// Количество прогонов на одном и том же seed
const size_t RUN_CNT = 3;

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

// Сравнение изображений (по среднему абсолютному отклонению)
bool approx_equal_img(const std::vector<float>& A,
    const std::vector<float>& B,
    double tol = 1e-3,
    double* mae_out = nullptr)
{
    if (A.size() != B.size()) {
        std::cerr << "[ERR] approx_equal_img: размеры не совпадают.\n";
        return false;
    }
    double mae = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        mae += std::fabs((double)A[i] - (double)B[i]);
    mae /= (double)A.size();
    if (mae_out) *mae_out = mae;
    return mae <= tol;
}


// ============================================
// ===      Основная функция программы      ===
// ============================================
int main()
{
    setlocale(LC_ALL, "RU");

    // ---------- Параметры задачи ----------
    LanczosParams P;
    P.inH = 512;  P.inW = 512;      // Размеры входного изображения
    P.channels = 4;                 // Число каналов (RGBA)
    P.outH = 1024; P.outW = 1024;   // Размеры выходного изображения

    const size_t src_elems = P.inH * P.inW * P.channels;
    const size_t dst_elems = P.outH * P.outW * P.channels;

    std::printf("Изображение: %dx%d -> %dx%d, каналов = %d\n",
        P.inH, P.inW, P.outH, P.outW, P.channels);
    std::printf("Радиус ядра Ланцоша a = %d (taps = %d)\n",
        LANCZOS_A, LANCZOS_TAPS);
    std::printf("Исходный объём: %.2f МБ, результат: %.2f МБ\n",
        (double)src_elems * sizeof(float) / (1024.0 * 1024.0),
        (double)dst_elems * sizeof(float) / (1024.0 * 1024.0));

    DeviceSpecs specs;
    cfg::load_config("config.ini", specs);

    sycl::queue q{ pick_device(), sycl::property::queue::enable_profiling() };

    // Результаты работы
    Results res;
    res.p = P;

    // Результаты итераций
    const size_t s = SEED_CNT * RUN_CNT;
    std::vector<float> cpu_ms(s), memcpy_ms(s),
        gpu_simple_ms(s), gpu_shared_ms(s),
        gpu_pipeline_ms(s);

    // Буфер для замера копирования
    float* d_buf = sycl::malloc_device<float>(src_elems, q);

    for (size_t i = 0; i < SEED_CNT; ++i)
    {
        // Генерация «изображения» по seed
        unsigned seed = (unsigned)std::chrono::high_resolution_clock::now()
            .time_since_epoch().count() + (unsigned)i;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);
        std::vector<float> src(src_elems);
        for (auto& v : src) v = dist(rng);

        for (size_t j = 0; j < RUN_CNT; ++j)
        {
            const size_t idx = i * RUN_CNT + j;

            // ---------- CPU ----------
            lanczos_cpu(src, P, cpu_ms[idx]);                 // прогрев
            auto tmp_cpu = lanczos_cpu(src, P, cpu_ms[idx]);

            // ---------- Замер memcpy ----------
            const size_t src_bytes = src_elems * sizeof(float);
            q.memcpy(d_buf, src.data(), src_bytes).wait();    // прогрев
            auto t0 = std::chrono::steady_clock::now();
            q.memcpy(d_buf, src.data(), src_bytes).wait();
            auto t1 = std::chrono::steady_clock::now();
            memcpy_ms[idx] = std::chrono::duration<float, std::milli>(t1 - t0).count();

            // ---------- GPU simple ----------
            lanczos_gpu_simple(q, src, P, gpu_simple_ms[idx]);
            auto tmp_simple = lanczos_gpu_simple(q, src, P, gpu_simple_ms[idx]);

            // ---------- GPU shared ----------
            lanczos_gpu_shared(q, src, P, gpu_shared_ms[idx]);
            auto tmp_shared = lanczos_gpu_shared(q, src, P, gpu_shared_ms[idx]);

            // ---------- GPU pipeline ----------
            lanczos_gpu_pipeline(q, src, P, gpu_pipeline_ms[idx], 8);
            auto tmp_pipeline = lanczos_gpu_pipeline(q, src, P, gpu_pipeline_ms[idx], 8);

            // ---------- Проверка корректности ----------
            double mae_simple = 0, mae_shared = 0, mae_pipeline = 0;
            bool ok_s = approx_equal_img(tmp_cpu, tmp_simple, specs.tol, &mae_simple);
            bool ok_h = approx_equal_img(tmp_cpu, tmp_shared, specs.tol, &mae_shared);
            bool ok_p = approx_equal_img(tmp_cpu, tmp_pipeline, specs.tol, &mae_pipeline);
            bool ok = ok_s && ok_h && ok_p;

            if (!ok) {
                std::printf("[ERR] seed %3zu итерация %3zu: MAE simple=%.3e shared=%.3e pipeline=%.3e\n",
                    i + 1, j + 1, mae_simple, mae_shared, mae_pipeline);
            }
        }
    }

    sycl::free(d_buf, q);

    // ---------- Медианы ----------
    res.cpu_ms = median(cpu_ms);
    res.memcpy_ms = median(memcpy_ms);
    res.gpu_simple_ms = median(gpu_simple_ms);
    res.gpu_shared_ms = median(gpu_shared_ms);
    res.gpu_pipeline_ms = median(gpu_pipeline_ms);

    // ---------- Метрики ----------
    sycl::device cpu{ sycl::cpu_selector_v };
    MetricCalc metrics(res, specs, cpu, q.get_device());
    metrics.calculate();

    return 0;
}