#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================================
//                      Параметры устройств CPU и GPU
// ============================================================================
struct DeviceSpecs
{
    // CPU
    float cpu_freq_hz = 1.3e9;      // Частота работы процессора, Гц
    float ram_bandwidth = 42.6;     // Пропускная способность ОЗУ, ГБ/с
    float cpu_cycles_per_add = 1.0; // Количество тактов на операцию сложения
    unsigned cpu_vector_size = 8;   // Размер вектора операции (AVX)

    // GPU
    float   gpu_core_freq_hz = 3.0e9;   // Частота работы ядра GPU, Гц
    float   vram_freq_hz = 28.0e9;      // Частота работы видеопамяти, Гц
    int     vram_bus_bytes = 16;        // Ширина шины GPU, Байт
    bool    is_uma = false;             // Флаг наличия интегрированной графики

    // PCIe
    float pcie_expected_gbps = 7.0;   // Скорость работы шины PCIe, ГБ/с
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


// ============================================================================
//                      Класс рассчёта метрик
// ============================================================================
class MetricCalc
{
private:
    Results _r;
    DeviceSpecs _c;
    sycl::device _cpu;
    sycl::device _gpu;

public:

    // Конструктор и деструктор
    MetricCalc(Results r, DeviceSpecs c, const sycl::device& cpu, const sycl::device& gpu)
        : _r(std::move(r)), _c(std::move(c)), _cpu(cpu), _gpu(gpu) {}

    ~MetricCalc() = default;

    // Функция рассчёта метрик
    void calculate()
    {
        // ------ Метрики по "Оценке реалистичности" ------
        
        // Размер данных в байтах
        int64_t data_size = _r.N * _r.elem_size;
        // Минимальное теоретическое время чтения из ОЗУ, мс
        double T_ram = (double)data_size / std::pow(2, 30) / (double)_c.ram_bandwidth * 1000.0;
        // Минимальное теоретическое время вычислений на CPU, мс
        double T_calc = (double)(_r.N - 1) * _c.cpu_cycles_per_add / _c.cpu_vector_size / _c.cpu_freq_hz * 1000.0;
        // Общее теоретическое время работы процессора
        double T_cpu_theor = T_ram + T_calc;

        // Минимальное теоретическое время копирования в видеопамять
        double T_vram = (_c.is_uma) ? T_ram : data_size / ((double)_c.vram_bus_bytes * _c.vram_freq_hz);
        // Количество итераций на GPU
        int64_t gpu_num_iter = std::ceil(std::log2(_r.N));
        // Минимальное теоретическое время работы GPU, нс
        double T_gpu_theor = (double)gpu_num_iter * 1000.0 * 1000.0 / _c.gpu_core_freq_hz;


        // ------ Метрики по "Вычислительной интенсивности" ------

        // Количество операций
        int64_t flops = _r.N - 1;
        // Количество обращений к памяти, байт
        int64_t mops = _r.N * sizeof(float) * 2;
        // Вычислительная интенсивность, flops/байт
        double CI = (double)flops / mops;

        // Потолок памяти CPU, Gflops
        double cpu_mem_ceil = CI * _c.ram_bandwidth;
        
        auto cus_c = _cpu.get_info<sycl::info::device::max_compute_units>();      // Количество ядер
        auto freq_c = _cpu.get_info<sycl::info::device::max_clock_frequency>();   // Частота работы, МГц
        auto sgs_c = _cpu.get_info<sycl::info::device::sub_group_sizes>();        // Размер группы для операций
        int sg_c = *std::max_element(sgs_c.begin(), sgs_c.end());                 // для CPU берётся МАКС
        // Потолок вычислений CPU, flops
        double cpu_calc_ceil = 2 * cus_c * sg_c * freq_c * 1e6 / std::pow(2, 30);

        // Потолок памяти GPU, Gflops
        double gpu_mem_ceil = (_c.is_uma) ? cpu_mem_ceil : CI * _c.vram_bus_bytes * _c.vram_freq_hz / std::pow(2, 30);
        
        auto cus_g = _gpu.get_info<sycl::info::device::max_compute_units>();      // Количество ядер
        auto freq_g = _gpu.get_info<sycl::info::device::max_clock_frequency>();   // Частота работы, МГц
        auto sgs_g = _gpu.get_info<sycl::info::device::sub_group_sizes>();        // Размер группы для операций
        int sg_g = *std::min_element(sgs_g.begin(), sgs_g.end());                 // для GPU берётся МИН
        // Потолок вычислений GPU, Gflops
        double gpu_calc_ceil = 2 * cus_g * sg_g * freq_g * 1e6 / std::pow(2, 30);


        // ------ Вывод метрик ------
        std::cout << "\n===================================================\n";
        std::cout << "\n    Ускорение    | cpu_ms | gpu_simple_ms\n";
        std::cout << "  gpu_shared_ms  | " << std::fixed << std::setprecision(3)
            << std::setw(6) << (_r.cpu_ms / _r.gpu_shared_ms) << " | "
            << std::setw(13) << (_r.gpu_simple_ms / _r.gpu_shared_ms) << "\n";
        std::cout << " gpu_pipeline_ms | " << std::fixed << std::setprecision(3)
            << std::setw(6) << (_r.cpu_ms / _r.gpu_pipeline_ms) << " | "
            << std::setw(13) << (_r.gpu_simple_ms / _r.gpu_pipeline_ms) << "\n";

        std::cout << "\n===================== МЕТРИКИ =====================\n";
        std::cout << "N = " << _r.N << ", размер элемента = " << sizeof(float) << " байт\n";
        std::cout << "V (объём данных) = " << std::fixed << std::setprecision(3)
            << ((double)data_size / 1024.0 / 1024.0 / 1024.0) << " ГБ\n";

        std::cout << "\n--- CPU ---\n";
        std::cout << std::left << std::setw(12) << "T_read(теор)" << " = " << std::fixed << std::setprecision(4) << std::setw(6) << T_ram << " мс\n";
        std::cout << std::left << std::setw(12) << "T_comp(теор)" << " = " << std::fixed << std::setprecision(4) << std::setw(6) << T_calc << " мс\n";
        std::cout << std::left << std::setw(12) << "T_cpu (теор)" << " = " << std::fixed << std::setprecision(4) << std::setw(6) << T_cpu_theor << " мс\n";
        std::cout << std::left << std::setw(12) << "Измерено" << " = " << std::fixed << std::setprecision(4) << std::setw(6) << _r.cpu_ms << " мс\n";

        std::cout << "\n--- GPU " << (_c.is_uma ? "(UMA)" : "(VRAM)") << " ---\n";
        std::cout << std::left << std::setw(20) << "Скорость шины" << " = " << std::fixed << std::setprecision(1) << std::setw(6)
            << (_c.is_uma ? _c.ram_bandwidth : _c.pcie_expected_gbps) << " ГБ/с\n";
        std::cout << std::left << std::setw(20) << "Скорость копирования" << " = " << std::fixed << std::setprecision(3) << std::setw(6)
            << ((double)data_size / ((double)_r.memcpy_ms / 1000.0) / std::pow(2, 30)) << " ГБ/с\n";
        std::cout << std::left << std::setw(20) << "Количество итераций" << " = " << std::setw(6) << gpu_num_iter << "\n";
        std::cout << std::left << std::setw(20) << "T_gpu (теор)" << " = " << std::fixed << std::setprecision(3) << std::setw(6) << T_gpu_theor << " нс\n";
        std::cout << std::left << std::setw(20) << "T_mem" << " = " << std::fixed << std::setprecision(3) << std::setw(6) << _r.memcpy_ms << " мс\n";
        std::cout << std::left << std::setw(20) << "T_simple" << " = " << std::fixed << std::setprecision(3) << std::setw(6) << _r.gpu_simple_ms << " мс\n";
        std::cout << std::left << std::setw(20) << "T_shared" << " = " << std::fixed << std::setprecision(3) << std::setw(6) << _r.gpu_shared_ms << " мс\n";
        std::cout << std::left << std::setw(20) << "T_pipeline" << " = " << std::fixed << std::setprecision(3) << std::setw(6) << _r.gpu_pipeline_ms << " мс\n";

        std::cout << "\n--- Вычислительная интенсивность ---\n";
        std::cout << std::left << std::setw(23) << "CI редукции" << " = " << std::fixed << std::setprecision(4) << std::setw(10) << CI << " FLOP/байт\n";
        std::cout << std::left << std::setw(23) << "Потолок памяти CPU" << " = " << std::fixed << std::setprecision(4) << std::setw(10) << cpu_mem_ceil << " GFLOP/байт\n";
        std::cout << std::left << std::setw(23) << "Потолок вычислений CPU" << " = " << std::fixed << std::setprecision(4) << std::setw(10) << cpu_calc_ceil << " GFLOP/байт\n";
        std::cout << std::left << std::setw(23) << "Потолок памяти GPU" << " = " << std::fixed << std::setprecision(4) << std::setw(10) << gpu_mem_ceil << " GFLOP/байт\n";
        std::cout << std::left << std::setw(23) << "Потолок вычислений GPU" << " = " << std::fixed << std::setprecision(4) << std::setw(10) << gpu_calc_ceil << " GFLOP/байт\n";
        std::cout << "===================================================\n";
    }
};
