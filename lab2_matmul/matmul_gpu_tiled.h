#pragma once
#include <sycl/sycl.hpp>
#include <chrono>

#ifndef TILE_SIZE
#define TILE_SIZE 16      // 16×16 = 256 work-item в группе
#endif

void matmul_gpu_tiled(sycl::queue& q,
                      const float* h_A, const float* h_B, float* h_C,
                      size_t N, float& ms)
{
    const size_t bytes = N * N * sizeof(float);

    float* d_A = sycl::malloc_device<float>(N * N, q);
    float* d_B = sycl::malloc_device<float>(N * N, q);
    float* d_C = sycl::malloc_device<float>(N * N, q);

    q.memcpy(d_A, h_A, bytes);
    q.memcpy(d_B, h_B, bytes);
    q.wait();

    const size_t num_tiles = N / TILE_SIZE;   // N должен делиться на TILE_SIZE!

    auto t0 = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler& h) {
        // Быстрая «разделяемая» память для двух тайлов
        sycl::local_accessor<float, 2> tileA({TILE_SIZE, TILE_SIZE}, h);
        sycl::local_accessor<float, 2> tileB({TILE_SIZE, TILE_SIZE}, h);

        h.parallel_for(
            sycl::nd_range<2>({N, N}, {TILE_SIZE, TILE_SIZE}),
            [=](sycl::nd_item<2> item) {
                const size_t gi = item.get_global_id(0);  // строка C
                const size_t gj = item.get_global_id(1);  // столбец C
                const size_t li = item.get_local_id(0);
                const size_t lj = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t t = 0; t < num_tiles; t++) {
                    // Загрузка тайлов (коалесцированно — по соседним адресам)
                    tileA[li][lj] = d_A[gi * N + (t * TILE_SIZE + lj)];
                    tileB[li][lj] = d_B[(t * TILE_SIZE + li) * N + gj];

                    // Барьер: все потоки загрузили, можно считать
                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < TILE_SIZE; k++)
                        sum += tileA[li][k] * tileB[k][lj];

                    // Барьер: все посчитали, можно грузить следующий тайл
                    item.barrier(sycl::access::fence_space::local_space);
                }
                d_C[gi * N + gj] = sum;
            });
    }).wait();
    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    q.memcpy(h_C, d_C, bytes).wait();

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);
}
