#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "metrics.h"

// ============================================================================
// Поддерживается:
// - строки key = value
// - комментарии # и ;
// - пустые строки
// - регистронезависимые имена ключей (приводятся к нижнему регистру)
//
// Пример config.ini:
//   cpu_freq_hz      = 1.3e9
//   ddr_bandwidth    = 42.6
//   gpu_core_freq_hz = 3.0e9
//   is_uma           = true
// ============================================================================

namespace cfg
{
    // Функция удаления лишних символов
    inline std::string trim(const std::string& s) 
    {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        return s.substr(a, b - a + 1);
    }

    // Функция приведения к нижнему регистру
    inline std::string lower(std::string s) 
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    }

    // Функция загрузки файла конфигурации
    inline bool load_config(const std::string& path, DeviceSpecs& specs) 
    {
        std::ifstream f(path);
        if (!f.is_open()) 
        {
            std::printf("[WARN] config '%s' не найден — используются значения по умолчанию\n", path.c_str());
            return false;
        }

        std::unordered_map<std::string, std::string> kv;
        std::string line;
        while (std::getline(f, line)) 
        {
            // Удаление комментариев и пустых строк
            size_t hash = line.find_first_of("#;");
            if (hash != std::string::npos) line = line.substr(0, hash);

            line = trim(line);
            if (line.empty()) continue;

            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = lower(trim(line.substr(0, eq)));
            std::string val = trim(line.substr(eq + 1));
            kv[key] = val;
        }

        // Лямбда извлечения float
        auto get_f = [&](const char* key, float def) -> float {
            auto it = kv.find(key);
            if (it == kv.end()) return def;
            try { return std::stof(it->second); }
            catch (...) { return def; }
            };
        // Лямбда извлечения int
        auto get_i = [&](const char* key, int def) -> int {
            auto it = kv.find(key);
            if (it == kv.end()) return def;
            try { return std::stoi(it->second); }
            catch (...) { return def; }
            };
        // Лямбда извлечения bool
        auto get_b = [&](const char* key, bool def) -> bool {
            auto it = kv.find(key);
            if (it == kv.end()) return def;
            std::string v = lower(it->second);
            if (v == "true" || v == "1" || v == "yes") return true;
            if (v == "false" || v == "0" || v == "no")  return false;
            return def;
            };

        specs.cpu_freq_hz = get_f("cpu_freq_hz", specs.cpu_freq_hz);
        specs.ram_bandwidth = get_f("ddr_bandwidth", specs.ram_bandwidth);
        specs.cpu_cycles_per_add = get_f("cpu_cycles_per_add", specs.cpu_cycles_per_add);
        specs.cpu_vector_size = get_i("cpu_vector_size", specs.cpu_vector_size);
        
        specs.gpu_core_freq_hz = get_f("gpu_core_freq_hz", specs.gpu_core_freq_hz);
        specs.vram_freq_hz = get_f("vram_freq_hz", specs.vram_freq_hz);
        specs.vram_bus_bytes = get_i("vram_bus_bytes", specs.vram_bus_bytes);
        specs.is_uma = get_b("is_uma", specs.is_uma);

        specs.pcie_expected_gbps = get_f("pcie_expected_gbps", specs.pcie_expected_gbps);

        std::printf("[INFO] Загружен config '%s'\n", path.c_str());
        return true;
    }

} // namespace cfg