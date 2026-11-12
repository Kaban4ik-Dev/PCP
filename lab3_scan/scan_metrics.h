#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include "config_loader.h"


struct DeviceSpecs
{
    // CPU
    float cpu_freq_hz = 1.3e9;      // Ãö
    float ddr_bandwidth = 42.6;     // ÃÁ/ñ
    float cpu_cycles_per_op = 1.0;  // Êîëè÷åñòâî îïåðàöèé â òàêò

    // GPU
    float   gpu_core_freq_hz = 3.0e9;   // Ãö, ÷àñòîòà ÿäðà GPU
    float   vram_freq_ghz = 28.0;       // ÃÃö, ÷àñòîòà âèäåîïàìÿòè
    int     vram_bus_bytes = 16;        // Áàéò, øèðèíà øèíû
    bool    is_uma = false;             // Ôëàã èíòåãðèðîâàííîé ãðàôèêè

    // PCIe
    float pcie_expected_gbps = 7.0;   // ÃÁ/ñ, ñêîðîñòü ðàáîòû øèíû
};

struct Results {
    size_t N = 0;
    size_t elem_size = 0;
    float cpu_ms = 0.0f;
    float gpu_naive_ms = 0.0f;
    float gpu_blocked_ms = 0.0f;
};

void print_scan_metrics(const Results& res, const DeviceSpecs& specs, const sycl::device& dev)
{
    const size_t N = res.N;
    const double flops_total = static_cast<double>(N);                 // одна сумма на элемент
    const double bytes_seq   = 8.0 * N;                                // 1 проход: read + write
    const double bytes_naive = bytes_seq * std::log2(static_cast<double>(N));
    const double bytes_blk   = bytes_seq * 3.0;                        // 3 прохода
    const double CI = flops_total / bytes_seq;

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
