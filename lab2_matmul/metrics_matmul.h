#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "config_loader.h"

struct DeviceSpecs {
    double cpu_freq_hz         = 1.3e9;
    double cpu_cycles_per_op   = 0.125;
    double ddr_bandwidth       = 42.6;
    double gpu_core_freq_hz    = 1.3e9;
    double vram_freq_ghz       = 28.0;
    double vram_bus_bytes      = 16;
    bool   is_uma              = false;
    double pcie_expected_gbps  = 7.0;
};

struct Results {
    size_t N = 0;
    size_t elem_size = 0;
    float cpu_ms = 0.0f;
    float memcpy_ms = 0.0f;
    float gpu_naive_ms = 0.0f;
    float gpu_tiled_ms = 0.0f;
};

void print_matmul_metrics(const Results& res, const DeviceSpecs& specs, const sycl::device& dev)
{
    const size_t N = res.N;

    // --- Пункт 2: вычислительная интенсивность (CI) ---
    // FLOPs: N^2 элементов, каждый = N умножений + N сложений = 2*N^3
    // Байты: прочитали A, прочитали B, записали C = 3*N^2*4 байт
    const double flops_total = 2.0 * N * N * N;
    const double bytes_total = 3.0 * N * N * 4.0;
    const double CI = flops_total / bytes_total;

    // --- Пункт 3: roofline-модель ---
    const double ops_per_cycle = 1.0 / specs.cpu_cycles_per_op;
    const double peak_flops_cpu = specs.cpu_freq_hz * ops_per_cycle;
    const double peak_flops_gpu = specs.gpu_core_freq_hz * 16.0;

    const double bw_ddr = specs.ddr_bandwidth * 1e9;
    const double bw_vram = specs.is_uma
        ? bw_ddr
        : specs.vram_freq_ghz * 1e9 * specs.vram_bus_bytes / 8.0;

    const bool has_gpu = dev.is_gpu();
    const double peak_flops = has_gpu ? peak_flops_gpu : peak_flops_cpu;
    const double bandwidth  = has_gpu ? bw_vram : bw_ddr;

    const double mem_bound = bandwidth * CI;
    const double roof = std::min(peak_flops, mem_bound);
    const char* bottleneck = (mem_bound < peak_flops) ? "ПАМЯТЬ" : "ВЫЧИСЛЕНИЯ";

    const double measured_flops_naive = flops_total / (res.gpu_naive_ms * 1e-3);
    const double measured_flops_tiled = flops_total / (res.gpu_tiled_ms * 1e-3);
    const double eff_naive = measured_flops_naive / roof;
    const double eff_tiled = measured_flops_tiled / roof;

    // --- Пункт 4: вывод (по образцу print_reduction_metrics) ---
    std::printf("\n================ АНАЛИЗ MATMUL ================\n");
    std::printf("N = %zu, FLOPs = %.3e, байт = %.3e\n", N, flops_total, bytes_total);
    std::printf("CI (compute intensity)  = %.2f FLOP/байт\n", CI);
    std::printf("Пик CPU (GFLOP/s)       = %.2f\n", peak_flops_cpu / 1e9);
    std::printf("Пик GPU (GFLOP/s, оц.)  = %.2f\n", peak_flops_gpu / 1e9);
    std::printf("Пропускная способность  = %.2f ГБ/с\n", bandwidth / 1e9);
    std::printf("Memory-bound предел     = %.2f GFLOP/s\n", mem_bound / 1e9);
    std::printf("Roofline (потолок)      = %.2f GFLOP/s\n", roof / 1e9);
    std::printf("Узкое место             = %s\n", bottleneck);
    std::printf("\n--- Замеры ---\n");
    std::printf("Naive:  %.2f GFLOP/s, эффективность %.1f%%\n",
                measured_flops_naive / 1e9, eff_naive * 100.0);
    std::printf("Tiled:  %.2f GFLOP/s, эффективность %.1f%%\n",
                measured_flops_tiled / 1e9, eff_tiled * 100.0);
    std::printf("Ускорение tiled vs naive: %.2fx\n", res.gpu_naive_ms / res.gpu_tiled_ms);
    std::printf("================================================\n");
}
