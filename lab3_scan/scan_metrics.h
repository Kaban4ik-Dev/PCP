#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "config_loader.h"

struct Results {
    size_t N = 0;
    size_t elem_size = 0;
    float cpu_ms = 0.0f;
    float gpu_naive_ms = 0.0f;
    float gpu_blocked_ms = 0.0f;
    float gpu_pipeline_ms = 0.0f;
    float memcpy_ms = 0.0f;
};

inline float scan_data_size_gb(size_t N, size_t elem_size)
{
    return float(N) * float(elem_size) / (1024.0f * 1024.0f * 1024.0f);
}

inline float scan_traffic_gb(size_t N, size_t elem_size)
{
    // Чтение входного массива и запись выходного массива (2 * N * elem_size)
    return (2.0f * float(N) * float(elem_size)) / (1024.0f * 1024.0f * 1024.0f);
}

inline float memory_read_theoretical_ms(float V_gb, float bandwidth_gbs)
{
    return (V_gb / bandwidth_gbs) * 1000.0f;
}

inline float cpu_compute_theoretical_ms(size_t N, float cycles_per_op, float freq_hz)
{
    return (float(N) * cycles_per_op) / freq_hz * 1000.0f;
}

inline float gpu_effective_bandwidth_gbs(const DeviceSpecs& d)
{
    if (d.is_uma) return d.ddr_bandwidth;
    return d.vram_freq_ghz * float(d.vram_bus_bytes);
}

inline float dram_to_vram_gbs(float V_gb, float measured_ms)
{
    if (measured_ms <= 0.0f) return 0.0f;
    return V_gb / (measured_ms / 1000.0f);
}

inline float gpu_scan_iterations_min_ms(size_t N, float gpu_freq_hz)
{
    if (N <= 1 || gpu_freq_hz <= 0.0f) return 0.0f;
    float iterations = std::log2(float(N));
    return (iterations / gpu_freq_hz) * 1000.0f;
}

void print_scan_metrics(const Results& res, const DeviceSpecs& specs, const sycl::device& dev)
{
    const size_t N = res.N;
    const size_t elem_size = res.elem_size;
    const double flops_total = static_cast<double>(N);                 // одна сумма на элемент
    const double bytes_seq   = 8.0 * N;                                // 1 проход: read + write
    const double bytes_naive = bytes_seq * std::log2(static_cast<double>(N));
    const double bytes_blk   = bytes_seq * 3.0;                        // 3 прохода
    const double CI = flops_total / bytes_seq;
    const float V_in_gb = scan_data_size_gb(N, elem_size);
    const float V_rw_gb = scan_traffic_gb(N, elem_size);
    const float cpu_tread = memory_read_theoretical_ms(V_rw_gb, specs.ddr_bandwidth);
    const float cpu_tcomp = cpu_compute_theoretical_ms(N, specs.cpu_cycles_per_op, specs.cpu_freq_hz);
    const float gpu_bw_gbs = gpu_effective_bandwidth_gbs(specs);
    const float gpu_tread = memory_read_theoretical_ms(V_rw_gb, gpu_bw_gbs);
    const float gpu_iter_ms = gpu_scan_iterations_min_ms(N, specs.gpu_core_freq_hz);
    const float copy_gbs = dram_to_vram_gbs(V_in_gb, res.memcpy_ms);

    const double bw_ddr  = specs.ddr_bandwidth * 1e9;
    const double bw_vram = specs.is_uma
        ? bw_ddr
        : specs.vram_freq_ghz * 1e9 * specs.vram_bus_bytes / 8.0;
    const double bandwidth = dev.is_gpu() ? bw_vram : bw_ddr;

    const double bw_cpu  = bytes_seq   / (res.cpu_ms         * 1e-3) / 1e9;
    const double bw_naive = bytes_naive / (res.gpu_naive_ms  * 1e-3) / 1e9;
    const double bw_blk   = bytes_blk   / (res.gpu_blocked_ms * 1e-3) / 1e9;

    const double eff_cpu   = bw_cpu   / (bandwidth / 1e9);
    const double eff_naive = bw_naive / (bandwidth / 1e9);
    const double eff_blk   = bw_blk   / (bandwidth / 1e9);

    std::printf("\n================ АНАЛИЗ SCAN ================\n");
    std::printf("N = %zu, FLOPs = %.3e\n", N, flops_total);
    std::printf("CI (compute intensity)  = %.4f FLOP/байт\n", CI);
    std::printf("Пропускная способность  = %.2f ГБ/с (теор.)\n", bandwidth / 1e9);
    std::printf("Узкое место             = ПАМЯТЬ (низкое CI)\n");
    std::printf("\n--- Эффективная пропускная способность ---\n");
    std::printf("CPU sequential: %.2f ГБ/с (%.1f%% от теор.)\n", bw_cpu,   eff_cpu   * 100.0);
    std::printf("GPU naive:      %.2f ГБ/с (%.1f%% от теор.)\n", bw_naive, eff_naive * 100.0);
    std::printf("GPU blocked:    %.2f ГБ/с (%.1f%% от теор.)\n", bw_blk,   eff_blk   * 100.0);
    std::printf("\n--- Относительные замеры ---\n");
    std::printf("naive vs CPU:   %.2fx\n", res.cpu_ms / res.gpu_naive_ms);
    std::printf("blocked vs CPU: %.2fx\n", res.cpu_ms / res.gpu_blocked_ms);
    std::printf("blocked vs naive: %.2fx\n", res.gpu_naive_ms / res.gpu_blocked_ms);
    std::printf("==============================================\n");
}
