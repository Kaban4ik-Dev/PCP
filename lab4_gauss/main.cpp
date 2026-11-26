#include <cstddef>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

#include "gaussian_common.h"
#include "gaussian_cpu.h"
#include "gaussian_gpu_naive.h"
#include "gaussian_gpu_pipeline.h"
#include "gaussian_gpu_shared.h"
#include "metrics.h"

// Количество разных seed
const size_t SEED_CNT = 5;
// Количество прогонов на одном и том же seed
const size_t RUN_CNT = 10;

// Функция выбора подходящего графического устройства
sycl::device pick_device() {
  // Проверка существования GPU
  try {
    auto gpu = sycl::device{sycl::gpu_selector_v};
    // if (!gpu.has(sycl::aspect::fp64)) throw std::runtime_error("Device has no
    // float support\n");
    std::cout << "[INFO] Используется GPU: "
              << gpu.get_info<sycl::info::device::name>() << "\n";
    return gpu;
  }
  // Проверка существования CPU
  catch (const sycl::exception &e) {
    try {
      std::cerr << "[WARN] GPU не найден (" << e.what() << "), поиск CPU.\n";
      auto cpu = sycl::device{sycl::cpu_selector_v};
      std::cout << "[INFO] Используется CPU: "
                << cpu.get_info<sycl::info::device::name>() << "\n";
      return cpu;
    }
    // Если совместимых устройств нет - выход
    catch (const sycl::exception &e) {
      std::cerr << "[ERR] GPU и CPU не найдены (" << e.what()
                << "), завершение работы.\n";
      exit(1);
    }
  }
}

// Функция расчёта медианы в векторе
float median(std::vector<float> v) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size() / 2;
  return (v.size() % 2) ? v[n] : 0.5 * (v[n - 1] + v[n]);
}

// Функция проверки корректности двух массивов (возвращает true, если max_diff
// <= abs_tol)
bool arrays_approx_equal(const std::vector<float> &a,
                         const std::vector<float> &b, float abs_tol = 1.0f,
                         float *out_max_diff = nullptr) {
  if (a.size() != b.size())
    return false;
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > max_diff)
      max_diff = d;
  }
  if (out_max_diff)
    *out_max_diff = max_diff;
  return max_diff <= abs_tol;
}

// Сохранение планарного float-изображения (0..255) в JPEG
void save_planar_as_jpeg(const std::string &path,
                         const std::vector<float> &planar, int width,
                         int height, int channels) {
  if (path.empty())
    return;
  const int save_ch = (channels == 1) ? 1 : 3;
  std::vector<unsigned char> interleaved =
      planar_f32_to_interleaved_u8(planar, width, height, channels);

  if (channels != save_ch) {
    std::vector<unsigned char> adapted(
        (size_t)width * (size_t)height * (size_t)save_ch, 0);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        size_t p = (size_t)y * (size_t)width + (size_t)x;
        for (int c = 0; c < save_ch; ++c) {
          adapted[p * (size_t)save_ch + (size_t)c] =
              (c < channels) ? interleaved[p * (size_t)channels + (size_t)c]
                             : interleaved[p * (size_t)channels];
        }
      }
    }
    stbi_write_jpg(path.c_str(), width, height, save_ch, adapted.data(), 92);
  } else {
    stbi_write_jpg(path.c_str(), width, height, save_ch, interleaved.data(),
                   92);
  }
}

// ============================================
// ===      Основная функция программы      ===
// ============================================
int main() {
  setlocale(LC_ALL, "RU");

  // Параметры устройств и фильтра Гаусса
  DeviceSpecs specs;
  cfg::load_config("config.ini", specs);

  const int radius = std::max(0, specs.gaussian_radius);
  const float sigma = specs.gaussian_sigma;
  std::vector<float> h_kernel = make_gaussian_kernel_1d(radius, sigma);

  int width = specs.img_width;
  int height = specs.img_height;
  int channels = specs.img_channels;

  std::vector<unsigned char> raw_u8;
  const bool from_jpeg = !specs.jpeg_path.empty();

  if (from_jpeg) {
    int w = 0, h = 0, c = 0;
    // Принудительно 3 канала (RGB без альфа-канала)
    unsigned char *loaded = stbi_load(specs.jpeg_path.c_str(), &w, &h, &c, 3);
    if (!loaded) {
      std::printf(
          "[ERR] Не удалось загрузить JPEG '%s', переход к синтетике.\n",
          specs.jpeg_path.c_str());
    } else {
      width = w;
      height = h;
      channels = 3;
      raw_u8.assign(loaded,
                    loaded + (size_t)width * (size_t)height * (size_t)channels);
      stbi_image_free(loaded);
      std::printf("[INFO] Загружен JPEG '%s' (%dx%dx%d)\n",
                  specs.jpeg_path.c_str(), width, height, channels);
    }
  }

  const size_t N = (size_t)width * (size_t)height * (size_t)channels;
  const size_t elem_size = sizeof(float);

  std::printf(
      "Размер изображения: %dx%dx%d, N = %zu, radius = %d, sigma = %.3f\n",
      width, height, channels, N, radius, sigma);

  std::vector<float> h_data(N, 0.0f);

  // ---------- Очередь SYCL ----------
  sycl::queue q{pick_device(), sycl::property::queue::enable_profiling()};

  // Структура результатов
  Results res;
  res.N = N;
  res.elem_size = elem_size;

  // Промежуточные массивы результатов
  const size_t s = SEED_CNT * RUN_CNT;
  std::vector<float> cpu_ms(s), memcpy_ms(s), gpu_naive_ms(s), gpu_shared_ms(s),
      gpu_pipeline_ms(s);
  // Буфер для замера времени копирования
  float *d_buf = sycl::malloc_device<float>(N, q);

  std::vector<float> last_cpu_out, last_naive_out, last_shared_out,
      last_pipeline_out;

  // Цикл по разным значениям seed
  for (size_t i = 0; i < SEED_CNT; i++) {
    if (!raw_u8.empty()) {
      h_data =
          interleaved_u8_to_planar_f32(raw_u8.data(), width, height, channels);
    } else {
      // Генерация синтетического изображения по seed текущей итерации (0..255)
      unsigned seed =
          std::chrono::high_resolution_clock::now().time_since_epoch().count() +
          (unsigned)i;
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<float> dist(0.0f, 255.0f);
      for (auto &v : h_data)
        v = dist(rng);
    }

    // Цикл одинаковых замеров для получения устойчивых данных
    for (size_t j = 0; j < RUN_CNT; j++) {
      const size_t idx = i * RUN_CNT + j;

      // ---------- CPU (1 поток) ----------
      (void)gaussian_cpu(h_data, width, height, channels, h_kernel, radius,
                         cpu_ms[idx]); // Прогрев
      last_cpu_out = gaussian_cpu(h_data, width, height, channels, h_kernel,
                                  radius, cpu_ms[idx]);

      // ---------- Замер времени передачи данных ----------
      q.memcpy(d_buf, h_data.data(), N * elem_size).wait(); // Прогрев
      sycl::event e = q.memcpy(d_buf, h_data.data(), N * elem_size);
      e.wait();
      auto t0 =
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 =
          e.get_profiling_info<sycl::info::event_profiling::command_end>();
      memcpy_ms[idx] = (t1 - t0) / 1e6f;

      // ---------- GPU naive ----------
      (void)gaussian_gpu_naive(q, h_data, width, height, channels, h_kernel,
                               radius, gpu_naive_ms[idx]); // Прогрев
      last_naive_out = gaussian_gpu_naive(q, h_data, width, height, channels,
                                          h_kernel, radius, gpu_naive_ms[idx]);

      // ---------- GPU shared ----------
      (void)gaussian_gpu_shared(q, h_data, width, height, channels, h_kernel,
                                radius, gpu_shared_ms[idx]); // Прогрев
      last_shared_out =
          gaussian_gpu_shared(q, h_data, width, height, channels, h_kernel,
                              radius, gpu_shared_ms[idx]);

      // ---------- GPU конвейер ----------
      (void)gaussian_gpu_pipeline(q, h_data, width, height, channels, h_kernel,
                                  radius, specs.pipeline_chunk_mb,
                                  gpu_pipeline_ms[idx]); // Прогрев
      last_pipeline_out = gaussian_gpu_pipeline(
          q, h_data, width, height, channels, h_kernel, radius,
          specs.pipeline_chunk_mb, gpu_pipeline_ms[idx]);

      // ---------- Проверка корректности ----------
      float d_naive = 0.0f, d_shared = 0.0f, d_pipe = 0.0f;
      bool ok_naive =
          arrays_approx_equal(last_cpu_out, last_naive_out, 1.0f, &d_naive);
      bool ok_shared =
          arrays_approx_equal(last_cpu_out, last_shared_out, 1.0f, &d_shared);
      bool ok_pipe =
          arrays_approx_equal(last_cpu_out, last_pipeline_out, 1.0f, &d_pipe);
      if (!ok_naive || !ok_shared || !ok_pipe) {
        std::printf("[ERR] seed %3zu iter %3zu расхождение (max_diff: "
                    "naive=%.4f, shared=%.4f, pipe=%.4f)\n",
                    i + 1, j + 1, d_naive, d_shared, d_pipe);
      }
    }
  }

  // Сохранение входного и выходного изображений (включая синтетический режим)
  if (!raw_u8.empty()) {
    std::vector<float> planar_in =
        interleaved_u8_to_planar_f32(raw_u8.data(), width, height, channels);
    save_planar_as_jpeg(specs.output_in_path, planar_in, width, height,
                        channels);
  } else {
    save_planar_as_jpeg(specs.output_in_path, h_data, width, height, channels);
  }
  save_planar_as_jpeg(specs.output_out_path, last_shared_out, width, height,
                      channels);

  // Освобождение ресурсов
  sycl::free(d_buf, q);

  // Вычисление результатов по медиане
  res.cpu_ms = median(cpu_ms);
  res.memcpy_ms = median(memcpy_ms);
  res.gpu_naive_ms = median(gpu_naive_ms);
  res.gpu_shared_ms = median(gpu_shared_ms);
  res.gpu_pipeline_ms = median(gpu_pipeline_ms);

  // ---------- Метрики ----------
  sycl::device cpu{sycl::cpu_selector_v};
  MetricCalc metrics(res, specs, cpu, q.get_device());
  metrics.calculate();

  return 0;
}
