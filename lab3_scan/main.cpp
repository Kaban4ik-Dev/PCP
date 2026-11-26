#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

#include "config_loader.h"
#include "scan_cpu.h"
#include "scan_gpu_blocked.h"
#include "scan_gpu_naive.h"
#include "scan_gpu_pipeline.h"
#include "scan_metrics.h"

const size_t SEED_CNT = 3;
const size_t RUN_CNT = 5;

sycl::device pick_device() {
  try {
    auto gpu = sycl::device{sycl::gpu_selector_v};
    std::cout << "[INFO] Используется GPU: "
              << gpu.get_info<sycl::info::device::name>() << "\n";
    return gpu;
  } catch (const sycl::exception &e) {
    try {
      std::cerr << "[WARN] GPU не найден (" << e.what() << "), поиск CPU.\n";
      auto cpu = sycl::device{sycl::cpu_selector_v};
      std::cout << "[INFO] Используется CPU: "
                << cpu.get_info<sycl::info::device::name>() << "\n";
      return cpu;
    } catch (const sycl::exception &e) {
      std::cerr << "[ERR] GPU и CPU не найдены (" << e.what() << "), выход.\n";
      std::exit(1);
    }
  }
}

float median(std::vector<float> v) {
  if (v.empty())
    return 0.0f;
  std::sort(v.begin(), v.end());
  size_t n = v.size() / 2;
  return (v.size() % 2) ? v[n] : 0.5f * (v[n - 1] + v[n]);
}

bool close_to_ref(const float *a, const double *ref, size_t n,
                  double rel_tol = 1e-2) {
  double max_abs = 1.0;
  for (size_t i = 0; i < n; i++)
    max_abs = std::max(max_abs, std::fabs(ref[i]));
  const double threshold = rel_tol * max_abs;
  for (size_t i = 0; i < n; i++)
    if (std::fabs((double)a[i] - ref[i]) > threshold)
      return false;
  return true;
}

bool scans_close(const float *a, const float *b, size_t n,
                 double abs_tol = 1e-2) {
  for (size_t i = 0; i < n; i++)
    if (std::fabs(a[i] - b[i]) > abs_tol)
      return false;
  return true;
}

int main() {
  setlocale(LC_ALL, "RU");
  const size_t N = 1ull << 22;
  const size_t elem_size = sizeof(float);

  DeviceSpecs specs;
  cfg::load_config("config.ini", specs);

  std::printf("Размер массива: N = %zu (%.1f МБ)\n", N,
              N * elem_size / (1024.0 * 1024.0));

  std::vector<float> h_x(N), h_y_cpu(N), h_y_naive(N), h_y_blocked(N),
      h_y_pipeline(N);

  sycl::queue q{pick_device(), sycl::property::queue::enable_profiling()};

  Results res;
  res.N = N;
  res.elem_size = elem_size;

  const size_t s = SEED_CNT * RUN_CNT;
  std::vector<float> cpu_ms(s), memcpy_ms(s), gpu_naive_ms(s),
      gpu_blocked_ms(s), gpu_pipeline_ms(s);
  float *d_copy_buf = sycl::malloc_device<float>(N, q);
  const size_t pipeline_chunk_elems = 1ull << 20;

  for (size_t i = 0; i < SEED_CNT; i++) {
    unsigned seed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count() +
        i;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : h_x)
      v = dist(rng);
    // Эталон в double (считаем один раз на seed, не на run)
    std::vector<double> h_ref(N);
    {
      double acc = 0.0;
      for (size_t k = 0; k < N; k++) {
        acc += (double)h_x[k];
        h_ref[k] = acc;
      }
    }

    for (size_t j = 0; j < RUN_CNT; j++) {
      const size_t idx = i * RUN_CNT + j;

      scan_cpu(h_x.data(), h_y_cpu.data(), N);
      auto t0 = std::chrono::steady_clock::now();
      scan_cpu(h_x.data(), h_y_cpu.data(), N);
      auto t1 = std::chrono::steady_clock::now();
      cpu_ms[idx] = std::chrono::duration<float, std::milli>(t1 - t0).count();

      q.memcpy(d_copy_buf, h_x.data(), N * elem_size).wait();
      auto m0 = std::chrono::steady_clock::now();
      q.memcpy(d_copy_buf, h_x.data(), N * elem_size).wait();
      auto m1 = std::chrono::steady_clock::now();
      memcpy_ms[idx] =
          std::chrono::duration<float, std::milli>(m1 - m0).count();

      scan_gpu_naive(q, h_x.data(), h_y_naive.data(), N, gpu_naive_ms[idx]);
      scan_gpu_blocked(q, h_x.data(), h_y_blocked.data(), N,
                       gpu_blocked_ms[idx]);
      scan_gpu_pipeline(q, h_x.data(), h_y_pipeline.data(), N,
                        gpu_pipeline_ms[idx], pipeline_chunk_elems);

      if (!close_to_ref(h_y_cpu.data(), h_ref.data(), N))
        std::printf("[ERR] CPU: расхождение (seed %zu, run %zu)\n", i + 1,
                    j + 1);
      if (!close_to_ref(h_y_naive.data(), h_ref.data(), N))
        std::printf("[ERR] naive: расхождение (seed %zu, run %zu)\n", i + 1,
                    j + 1);
      if (!close_to_ref(h_y_blocked.data(), h_ref.data(), N))
        std::printf("[ERR] blocked: расхождение (seed %zu, run %zu)\n", i + 1,
                    j + 1);
      if (!close_to_ref(h_y_pipeline.data(), h_ref.data(), N))
        std::printf("[ERR] pipeline: расхождение (seed %zu, run %zu)\n", i + 1,
                    j + 1);
    }
  }

  sycl::free(d_copy_buf, q);

  res.cpu_ms = median(cpu_ms);
  res.memcpy_ms = median(memcpy_ms);
  res.gpu_naive_ms = median(gpu_naive_ms);
  res.gpu_blocked_ms = median(gpu_blocked_ms);
  res.gpu_pipeline_ms = median(gpu_pipeline_ms);

  std::printf("\n===================================================\n");
  std::printf("CPU sequential   = %.3f мс\n", res.cpu_ms);
  std::printf("GPU naive        = %.3f мс\n", res.gpu_naive_ms);
  std::printf("GPU blocked      = %.3f мс\n", res.gpu_blocked_ms);
  std::printf("GPU pipeline     = %.3f мс\n", res.gpu_pipeline_ms);

  print_scan_metrics(res, specs, q.get_device());
  return 0;
}
