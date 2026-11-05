#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------
// Макрос для проверки ошибок CUDA
// ---------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------
// 1. CPU-версия (эталон)
// ---------------------------------------------------------------
float reduceCPU(const float* data, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += data[i];
    }
    return sum;
}

// ---------------------------------------------------------------
// 2. Наивная GPU-версия (без оптимизации)
//    Каждый поток суммирует часть массива в глобальной памяти.
//    Используется атомарное сложение для получения итоговой суммы.
// ---------------------------------------------------------------
__global__ void reduceNaive(const float* __restrict__ g_data,
    float* __restrict__ g_out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float localSum = 0.0f;
    for (int k = i; k < N; k += stride) {
        localSum += g_data[k];
    }

    // Атомарное сложение результата потока в глобальную переменную
    atomicAdd(g_out, localSum);
}

// ---------------------------------------------------------------
// 3. Оптимизированная GPU-версия с __shared__ и древовидной редукцией
//    Обрабатывает произвольное N (не кратное размеру блока).
//
//    Схема:
//      - Каждый блок обрабатывает свою часть массива.
//      - Данные загружаются в __shared__ память (с инициализацией нулём
//        для потоков, вышедших за пределы массива).
//      - Выполняется древовидная редукция внутри блока.
//      - Поток 0 блока записывает локальную сумму в g_out[blockIdx.x].
//    После этого массив g_out (размер = число блоков) суммируется
//    либо на CPU, либо вторым ядром.
// ---------------------------------------------------------------
__global__ void reduceShared(const float* __restrict__ g_data,
    float* __restrict__ g_out,
    int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Загрузка в разделяемую память с обработкой выхода за пределы N
    sdata[tid] = (i < N) ? g_data[i] : 0.0f;
    __syncthreads();

    // Древовидная редукция внутри блока
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Поток 0 записывает результат блока в глобальную память
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------
// Второе ядро: суммирование результатов блоков (межблочная редукция)
// Один блок суммирует массив из numBlocks элементов.
// ---------------------------------------------------------------
__global__ void reduceBlocks(const float* __restrict__ g_in,
    float* __restrict__ g_out,
    int numBlocks)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = tid;

    sdata[tid] = (i < numBlocks) ? g_in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[0] = sdata[0];
    }
}

// ---------------------------------------------------------------
// Вспомогательная функция: запуск оптимизированной редукции
// ---------------------------------------------------------------
float reduceGPU(const float* d_data, int N, int blockSize, int numBlocks)
{
    // Буфер для результатов блоков
    float* d_blockResults = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockResults, numBlocks * sizeof(float)));

    // --- Этап 1: внутриблочная редукция ---
    size_t sharedBytes = blockSize * sizeof(float);
    reduceShared << <numBlocks, blockSize, sharedBytes >> > (d_data, d_blockResults, N);
    CUDA_CHECK(cudaGetLastError());

    // --- Этап 2: межблочная редукция ---
    // Если блоков больше, чем потоков в одном блоке, запускаем ещё одно ядро
    // с достаточным числом потоков (или несколько итераций).
    float* d_finalResult = nullptr;
    CUDA_CHECK(cudaMalloc(&d_finalResult, sizeof(float)));

    if (numBlocks <= 1024) {
        int finalBlockSize = 1024; // достаточно для numBlocks <= 1024
        reduceBlocks << <1, finalBlockSize, finalBlockSize * sizeof(float) >> > (
            d_blockResults, d_finalResult, numBlocks);
        CUDA_CHECK(cudaGetLastError());
    }
    else {
        // Много блоков: редуцируем итеративно
        float* d_in = d_blockResults;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, numBlocks * sizeof(float)));

        int remaining = numBlocks;
        while (remaining > 1) {
            int curBlockSize = 1024;
            int curNumBlocks = (remaining + curBlockSize - 1) / curBlockSize;
            reduceShared << <curNumBlocks, curBlockSize,
                curBlockSize * sizeof(float) >> > (d_in, d_out, remaining);
            CUDA_CHECK(cudaGetLastError());

            remaining = curNumBlocks;
            float* tmp = d_in;
            d_in = d_out;
            d_out = tmp;
        }
        CUDA_CHECK(cudaMemcpy(d_finalResult, d_in, sizeof(float),
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_out));
    }

    // Копируем результат на хост
    float hostResult = 0.0f;
    CUDA_CHECK(cudaMemcpy(&hostResult, d_finalResult, sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_blockResults));
    CUDA_CHECK(cudaFree(d_finalResult));
    return hostResult;
}

// ---------------------------------------------------------------
// main
// ---------------------------------------------------------------
int main()
{
    // ---- Параметры задачи ----
    const int N = 1000003;             // размер массива (не степень двойки)
    const int blockSize = 256;         // потоков в блоке
    const int numBlocks = 512;         // блоков в сетке

    printf("=== Лабораторная работа №1: Редукция (суммирование) ===\n");
    printf("N = %d, blockSize = %d, numBlocks = %d\n\n", N, blockSize, numBlocks);

    // ---- Выделение и инициализация памяти на хосте ----
    size_t bytes = N * sizeof(float);
    float* h_data = (float*)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Ошибка выделения памяти на хосте\n");
        return EXIT_FAILURE;
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;  // все единицы — эталонная сумма = N
    }

    // ---- CPU-версия ----
    cudaEvent_t cpuStart, cpuStop;
    // для CPU используем std::chrono, но чтобы не подключать chrono —
    // замерим через cudaEvent (они работают и для хоста как таймеры).
    // Проще: используем clock() из <ctime>.
    // Здесь для единообразия — cudaEvent.
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));
    float cpuSum = reduceCPU(h_data, N);
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float cpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cpuMs, evStart, evStop));

    printf("[CPU] Сумма = %.1f, время = %.3f мс\n", cpuSum, cpuMs);

    // ---- Выделение памяти на устройстве ----
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // ---- Наивная GPU-версия ----
    float* d_naiveOut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_naiveOut, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_naiveOut, 0, sizeof(float)));

    CUDA_CHECK(cudaEventRecord(evStart));
    reduceNaive << <numBlocks, blockSize >> > (d_data, d_naiveOut, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float naiveMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naiveMs, evStart, evStop));

    float naiveSum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&naiveSum, d_naiveOut, sizeof(float),
        cudaMemcpyDeviceToHost));
    printf("[GPU naive] Сумма = %.1f, время = %.3f мс\n", naiveSum, naiveMs);

    // ---- Оптимизированная GPU-версия ----
    CUDA_CHECK(cudaEventRecord(evStart));
    float gpuSum = reduceGPU(d_data, N, blockSize, numBlocks);
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float gpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, evStart, evStop));

    printf("[GPU shared] Сумма = %.1f, время = %.3f мс\n", gpuSum, gpuMs);

    // ---- Проверка корректности ----
    printf("\n--- Проверка корректности ---\n");
    printf("Эталон (CPU)          : %.1f\n", cpuSum);
    printf("GPU naive   отклонение: %.6f\n", fabsf(naiveSum - cpuSum));
    printf("GPU shared  отклонение: %.6f\n", fabsf(gpuSum - cpuSum));

    // ---- Ускорение ----
    printf("\n--- Ускорение относительно CPU ---\n");
    printf("GPU naive  : %.2fx\n", cpuMs / naiveMs);
    printf("GPU shared : %.2fx\n", cpuMs / gpuMs);

    // ---- Освобождение ресурсов ----
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_naiveOut));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    free(h_data);
    return EXIT_SUCCESS;
}