#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <clocale>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "matmul_cpu.h"
#include "matmul_gpu_naive.h"
#include "matmul_gpu_tiled.h"
#include "config_loader.h"
#include "metrics_matmul.h"

const size_t SEED_CNT = 3;   // меньше, чем в лабе 1: matmul дороже
const size_t RUN_CNT  = 3;

// Выбор устройства (тот же, что в лабе 1)
sycl::device pick_device()
{
    try {
        auto gpu = sycl::device{ sycl::gpu_selector_v };
        std::cout << "[INFO] Используется GPU: " << gpu.get_info<sycl::info::device::name>() << "\n";
        return gpu;
    } catch (const sycl::exception& e) {
        try {
            std::cerr << "[WARN] GPU не найден (" << e.what() << "), поиск CPU.\n";
            auto cpu = sycl::device{ sycl::cpu_selector_v };
            std::cout << "[INFO] Используется CPU: " << cpu.get_info<sycl::info::device::name>() << "\n";
            return cpu;
        } catch (const sycl::exception& e) {
            std::cerr << "[ERR] GPU и CPU не найдены (" << e.what() << "), завершение.\n";
            std::exit(1);
        }
    }
}

float median(std::vector<float> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size() / 2;
    return (v.size() % 2) ? v[n] : 0.5 * (v[n - 1] + v[n]);
}

bool approx_equal(double a, double b, double rel_tol = 1e-2)
{
    double diff = std::fabs(a - b);
    double scale = std::max({ std::fabs(a), std::fabs(b), 1e-6 });
    return diff <= rel_tol * scale;
}

bool matrices_close(const float* a, const float* b, size_t count, double tol = 1e-3)
{
    for (size_t i = 0; i < count; i++) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// ============================================
// ===      Основная функция программы      ===
// ============================================
int main()
{
    setlocale(LC_ALL, "RU");
    const size_t N = 512;                         // размер квадратной матрицы
    const size_t elem_size = sizeof(float);
    const size_t mat_bytes = N * N * elem_size;

    // Параметры устройств
    DeviceSpecs specs;
    cfg::load_config("config.ini", specs);

    std::printf("Размер матрицы: N = %zu, объём одной = %.2f МБ\n",
                N, mat_bytes / (1024.0 * 1024.0));

    // Хостовые матрицы (A, B и три C — CPU/naive/tiled)
    std::vector<float> h_A(N * N), h_B(N * N);
    std::vector<float> h_C_cpu(N * N), h_C_naive(N * N), h_C_tiled(N * N);

    // ---------- Очередь SYCL ----------
    sycl::queue q{ pick_device(), sycl::property::queue::enable_profiling() };

    // Результаты
    Results res;
    res.N = N;
    res.elem_size = elem_size;

    const size_t s = SEED_CNT * RUN_CNT;
    std::vector<float> cpu_ms(s), memcpy_ms(s), gpu_naive_ms(s), gpu_tiled_ms(s);

    // Буфер для замера memcpy
    float* d_buf = sycl::malloc_device<float>(N * N, q);

    for (size_t i = 0; i < SEED_CNT; i++) {
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + i;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);

        for (size_t j = 0; j < RUN_CNT; j++) {
            const size_t idx = i * RUN_CNT + j;

            // ---------- CPU ----------
            matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);       // прогрев
            auto t0 = std::chrono::steady_clock::now();
            matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
            auto t1 = std::chrono::steady_clock::now();
            cpu_ms[idx] = std::chrono::duration<float, std::milli>(t1 - t0).count();

            // ---------- Замер memcpy ----------
            q.memcpy(d_buf, h_A.data(), mat_bytes).wait();
            auto m0 = std::chrono::steady_clock::now();
            q.memcpy(d_buf, h_A.data(), mat_bytes).wait();
            auto m1 = std::chrono::steady_clock::now();
            memcpy_ms[idx] = std::chrono::duration<float, std::milli>(m1 - m0).count();

            // ---------- GPU naive ----------
            matmul_gpu_naive(q, h_A.data(), h_B.data(), h_C_naive.data(), N, gpu_naive_ms[idx]);

            // ---------- GPU tiled ----------
            matmul_gpu_tiled(q, h_A.data(), h_B.data(), h_C_tiled.data(), N, gpu_tiled_ms[idx]);

            // ---------- Проверка ----------
            if (!matrices_close(h_C_cpu.data(), h_C_naive.data(), N * N) ||
                !matrices_close(h_C_cpu.data(), h_C_tiled.data(), N * N)) {
                std::printf("[ERR] seed %zu, run %zu: результаты расходятся\n", i + 1, j + 1);
            }
        }
    }

    sycl::free(d_buf, q);

    res.cpu_ms = median(cpu_ms);
    res.memcpy_ms = median(memcpy_ms);
    res.gpu_naive_ms = median(gpu_naive_ms);
    res.gpu_tiled_ms = median(gpu_tiled_ms);

    std::printf("\n===================================================\n");
    std::printf("CPU        = %.2f мс\n", res.cpu_ms);
    std::printf("GPU naive  = %.3f мс\n", res.gpu_naive_ms);
    std::printf("GPU tiled  = %.3f мс\n", res.gpu_tiled_ms);
    std::printf("memcpy A   = %.2f мс\n", res.memcpy_ms);
    std::printf("Ускорение tiled vs naive: %.2fx\n", res.gpu_naive_ms / res.gpu_tiled_ms);
    std::printf("Ускорение tiled vs CPU:   %.2fx\n", res.cpu_ms / res.gpu_tiled_ms);

    print_matmul_metrics(res, specs, q.get_device());

    return 0;
}
