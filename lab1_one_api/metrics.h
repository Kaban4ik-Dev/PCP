#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================================
// Параметры устройств (CPU + GPU)
// ============================================================================
struct DeviceSpecs
{
    // CPU
    float cpu_freq_hz = 1.3e9;      // Гц
    float ddr_bandwidth = 42.6;     // ГБ/с
    float cpu_cycles_per_op = 1.0;  // Количество операций в такт

    // GPU
    float   gpu_core_freq_hz = 3.0e9;   // Гц, частота ядра GPU
    float   vram_freq_ghz = 28.0;       // ГГц, частота видеопамяти
    int     vram_bus_bytes = 16;        // Байт, ширина шины
    bool    is_uma = false;             // Флаг интегрированной графики

    // PCIe
    float pcie_expected_gbps = 7.0;   // ГБ/с, скорость работы шины
};

// Структура результатов одной серии замеров
struct Results
{
    // Размер задачи
    size_t N = 0;           // Количество элементов
    size_t elem_size = 0;   // Размер одного элемента, байт

    // Измеренные времена, мс
    float cpu_ms = 0.0;             // Последовательная редукция на CPU
    float gpu_simple_ms = 0.0;      // Простая редукция на GPU (только ядро)
    float gpu_shared_ms = 0.0;      // Оптимизированная редукция на GPU (только ядро)
    float gpu_pipeline_ms = 0.0;    // Конвейерная редукция на GPU (только ядро)
    float memcpy_ms = 0.0;          // Копирование памяти (DRAM -> VRAM/UMA)
};

// ============ МЕТРИКИ РЕАЛИСТИЧНОСТИ ============

// Функция расчёта объёма данных в гигабайтах
inline float data_size_gb(size_t N, size_t elem_size)
{
    return float(N) * float(elem_size) / (1024.0 * 1024.0 * 1024.0);
}

// Функция расчёта теоретического минимального времени чтения V ГБ из памяти, мс
inline float cpu_theoretical_ms(float V_gb, float bandwidth_gbs)
{
    return (V_gb / bandwidth_gbs) * 1000.0;
}

// Функция расчёта теоретического времени N операций сложения на CPU, мс
inline float cpu_compute_ms(size_t N, float cycles_per_op, float freq_hz)
{
    return (float(N) * cycles_per_op) / freq_hz * 1000.0;
}

// Функция расчёта скорости передачи данных между CPU и GPU
inline float gpu_bandwidth_gbs(const DeviceSpecs& d)
{
    // Для UMA — это системная DDR
    if (d.is_uma) return d.ddr_bandwidth;
    // Для дискретной GPU
    return d.vram_freq_ghz * d.vram_bus_bytes;
}

// Функция расчёта минимального времени чтения V ГБ из VRAM, мс
inline float gpu_theoretical_ms(float V_gb, float bandwidth_gbs)
{
    return (V_gb / bandwidth_gbs) * 1000.0;
}

// Функция рассчёта эффективной скорости копирования из DRAM в VRAM, ГБ/с
inline float dram_to_vram_gbs(float V_gb, float measured_ms)
{
    if (measured_ms <= 0.0) return 0.0;
    return V_gb / (measured_ms / 1000.0);
}

// Функция рассчёта максимального GFLOPS для CPU
float estimate_cpu_peak_gflops(const sycl::device& dev)
{
    auto cus = dev.get_info<sycl::info::device::max_compute_units>();
    auto freq = dev.get_info<sycl::info::device::max_clock_frequency>();    // МГц
    auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();         // Размер группы для операций
    int sg = *std::max_element(sgs.begin(), sgs.end());                     // для CPU берётся МАКС
    return 2.0f * cus * sg * freq * 1e6f / 1e9f;
}

// Функция рассчёта максимального GFLOPS для GPU
float estimate_gpu_peak_gflops(const sycl::device& dev) 
{
    auto cus = dev.get_info<sycl::info::device::max_compute_units>();
    auto freq = dev.get_info<sycl::info::device::max_clock_frequency>();
    auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
    int sg = *std::min_element(sgs.begin(), sgs.end());                     // для GPU берётся МИН
    return 2.0f * cus * sg * freq * 1e6f / 1e9f;
}

// Функция рассчёта идеального нижнего предела времени редукции, мс
inline float gpu_reduction_min_ms(size_t N, float gpu_freq_hz)
{
    float iterations = std::log2(float(N));
    return iterations / gpu_freq_hz * 1000.0;
}

// Функция рассчёта вычислительной интенсивности редукции, байт
inline float reduction_ci(size_t N, size_t elem_size)
{
    float ops = float(N - 1);
    float bytes = float(N) * float(elem_size);
    return ops / bytes;
}

// Структура вывода из интенсивности
struct ResVerdict
{
    float CI = 0.0;         // Вычислительная интенсивность
    float mem_ceil = 0.0;   // GFLOPS
    float comp_ceil = 0.0;  // GFLOPS
};

// Функция классификация алгоритма по интенсивности
inline ResVerdict analyze_reduction(size_t N, size_t elem_size, float peak_gflops, float bandwidth_gbs)
{
    ResVerdict r;
    r.CI = reduction_ci(N, elem_size);
    r.mem_ceil = r.CI * bandwidth_gbs;
    r.comp_ceil = peak_gflops;
    return r;
}



// ============ ВЫВОД ОТЧЁТА ============
inline void print_reduction_metrics(const Results& res, const DeviceSpecs& specs, const sycl::device& dev)
{
    const size_t N = res.N;
    const size_t elem_size = res.elem_size;

    // --- Объём данных ---
    float V = data_size_gb(N, elem_size);

    // --- CPU ---
    float cpu_tread = cpu_theoretical_ms(V, specs.ddr_bandwidth);
    float cpu_tcomp = cpu_compute_ms(N, specs.cpu_cycles_per_op, specs.cpu_freq_hz);
    float cpu_ttheor = cpu_tread + cpu_tcomp;

    // --- GPU ---
    float gpu_bw = gpu_bandwidth_gbs(specs);
    float gpu_tread = gpu_theoretical_ms(V, gpu_bw);
    float gpu_min = gpu_reduction_min_ms(N, specs.gpu_core_freq_hz);

    // --- Копирование данных ---
    float copy_speed = dram_to_vram_gbs(V, res.memcpy_ms);

    // --- Подведение итогов ---
    auto rv_cpu = analyze_reduction(N, elem_size, estimate_cpu_peak_gflops(dev), specs.ddr_bandwidth);
    auto rv_gpu = analyze_reduction(N, elem_size, estimate_gpu_peak_gflops(dev), gpu_bw);

    std::printf("\n================ МЕТРИКИ РЕДУКЦИИ ================\n");
    std::printf("N = %zu, размер элемента = %zu байт\n", N, elem_size);
    std::printf("V (объём данных) = %.3f ГБ\n\n", V);

    std::printf("--- CPU ---\n");
    std::printf("T_read(теор)   = %.2f мс\n", cpu_tread);
    std::printf("T_comp(теор)   = %.2f мс\n", cpu_tcomp);
    std::printf("T_theor общ.   = %.2f мс\n", cpu_ttheor);
    std::printf("Измерено       = %.2f мс\n", res.cpu_ms);

    std::printf("--- GPU ---\n");
    std::printf("Band_gpu       = %.1f ГБ/с%s\n",
        gpu_bw,
        specs.is_uma ? " (UMA)" : " (VRAM)");
    std::printf("T_read(теор)   = %.3f мс\n", gpu_tread);
    std::printf("T_reduce(теор) = %.6f мс\n", gpu_min);
    std::printf("Простая        = %.3f мс\n", res.gpu_simple_ms);
    std::printf("Shared-версия  = %.3f мс\n", res.gpu_shared_ms);

    std::printf("--- Копирование данных (%s) ---\n", specs.is_uma ? "UMA" : "PCIe");
    std::printf("Memcpy время = %.3f мс\n", res.memcpy_ms);
    if (specs.is_uma)
        std::printf("Скорость     = %.2f ГБ/с (UMA, ожидается 5–15)\n\n", copy_speed);
    else
        std::printf("Скорость     = %.2f ГБ/с (PCIe, ожидается ~%.1f)\n\n", copy_speed, specs.pcie_expected_gbps);

    std::printf("--- Подведение итогов ---\n");
    std::printf("Потолок памяти CPU     = %.4f FLOP/байт\n", rv_cpu.mem_ceil);
    std::printf("Потолок вычислений CPU = %.4f FLOP/байт\n", rv_cpu.comp_ceil);
    std::printf("CI редукции CPU        = %.4f FLOP/байт\n", rv_cpu.CI);
    std::printf("Потолок памяти GPU     = %.4f FLOP/байт\n", rv_gpu.mem_ceil);
    std::printf("Потолок вычислений GPU = %.4f FLOP/байт\n", rv_gpu.comp_ceil);
    std::printf("CI редукции GPU        = %.4f FLOP/байт\n", rv_gpu.CI);
    std::printf("===================================================\n");
}