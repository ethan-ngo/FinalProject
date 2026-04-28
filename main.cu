// main.cu
// ---------------------------------------------------------------
//  GPU-accelerated portfolio optimisation — driver & benchmarks
//
//  Flow
//  ----
//  1. Load returns CSV on the host
//  2. Compute mean on the host (cheap — O(N·T) single pass)
//  3. Transfer data to GPU
//  4. GPU covariance via custom CUDA kernel (gpu_cov.cuh)
//  5. GPU gradient descent via cuBLAS DGEMV + update kernels
//  6. Transfer results back; print portfolio summary
//  7. Also run the CPU baseline for comparison and print speedup
// ---------------------------------------------------------------

// srun --account=bchn-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=16 --cpus-per-task=1 --mem=20g --pty bash

#include "utils.h"
#include "gpu_cov.cuh"
#include "gpu_gradient.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <string>

// -------------------------------------------------------------------
// Error macros
// -------------------------------------------------------------------
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t st = (call);                                       \
        if (st != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %s:%d  status=%d\n",           \
                    __FILE__, __LINE__, (int)st);                         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// -------------------------------------------------------------------
// Thin CPU covariance + GD re-used from cpu_baseline logic
// (inlined here so main.cu is self-contained for timing comparison)
// -------------------------------------------------------------------
#include "cpu_baseline_inline.h"   // see note below — we #include the impl

// -------------------------------------------------------------------
// main
// -------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // ----- Parse arguments
    std::string csv_path = (argc > 1) ? argv[1] : "data/returns.csv";
    double lambda   = (argc > 2) ? std::stod(argv[2]) : 1.0;
    int    max_iter = (argc > 3) ? std::stoi(argv[3]) : 2000;
    double lr       = (argc > 4) ? std::stod(argv[4]) : 1e-3;
    double tol      = 1e-8;

    std::cout << "============================================================\n";
    std::cout << "  GPU Portfolio Optimisation\n";
    std::cout << "  csv=" << csv_path << "  lambda=" << lambda
              << "  lr=" << lr << "  max_iter=" << max_iter << "\n";
    std::cout << "============================================================\n";

    // ----- Load data
    ReturnMatrix rm = load_csv(csv_path);
    int T = rm.T, N = rm.N;
    std::cout << "Data: T=" << T << " days, N=" << N << " assets\n\n";

    // ----- Host mean
    auto mu = compute_mean(rm);

    // ================================================================
    //  CPU BASELINE (for speedup comparison)
    // ================================================================
    std::cout << "--- CPU Baseline ---\n";
    Timer cpu_total; cpu_total.start();

    Timer cpu_cov_t; cpu_cov_t.start();
    auto sigma_cpu = cpu_compute_covariance(rm, mu);
    double cpu_cov_ms = cpu_cov_t.elapsed_ms();

    Timer cpu_opt_t; cpu_opt_t.start();
    auto w_cpu = cpu_gradient_descent(sigma_cpu, mu, N, lambda, lr, max_iter, tol);
    double cpu_opt_ms = cpu_opt_t.elapsed_ms();
    double cpu_total_ms = cpu_total.elapsed_ms();

    printf("  Covariance : %.2f ms\n",  cpu_cov_ms);
    printf("  GD loop    : %.2f ms\n",  cpu_opt_ms);
    printf("  Total      : %.2f ms\n\n", cpu_total_ms);

    // ================================================================
    //  GPU PIPELINE
    // ================================================================
    std::cout << "--- GPU Accelerated ---\n";

    // Print device info
    {
        int dev = 0;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("  Device: %s  (SM %d.%d)\n", prop.name,
               prop.major, prop.minor);
    }

    // Create cuBLAS handle
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // ----- Allocate device memory
    double *d_R, *d_mu, *d_sigma, *d_Rdm;
    double *d_w, *d_Sw, *d_grad, *d_tmp;

    CUDA_CHECK(cudaMalloc(&d_R,     (size_t)T * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,    (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Rdm,   (size_t)T * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,     (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Sw,    (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad,  (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp,   (size_t)N     * sizeof(double)));

    // ----- H → D transfers
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_R,  rm.data.data(), T * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu.data(),       N     * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms_h2d;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev0, ev1));
    printf("  H→D transfer: %.2f ms\n", ms_h2d);

    // ----- Covariance kernel
    float ms_cov_gpu = 0.f;
    gpu_compute_covariance(d_R, d_mu, d_sigma, d_Rdm, T, N, &ms_cov_gpu);
    printf("  Covariance   : %.2f ms\n", ms_cov_gpu);

    // ----- Initialise weights uniformly on device
    {
        std::vector<double> w0(N, 1.0 / N);
        CUDA_CHECK(cudaMemcpy(d_w, w0.data(), N * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    // ----- Gradient descent loop
    float ms_opt_gpu = 0.f;
    gpu_gradient_descent(cublas,
                         d_sigma, d_mu,
                         d_w, d_Sw, d_grad, d_tmp,
                         N, lambda, lr, max_iter, tol,
                         &ms_opt_gpu);
    printf("  GD loop      : %.2f ms\n", ms_opt_gpu);

    // ----- D → H: weights + covariance (for summary)
    std::vector<double> w_gpu(N), sigma_gpu(N * N);
    CUDA_CHECK(cudaMemcpy(w_gpu.data(),    d_w,     N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sigma_gpu.data(), d_sigma, N * N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double gpu_total_ms = ms_h2d + ms_cov_gpu + ms_opt_gpu;
    printf("  Total (incl H→D): %.2f ms\n\n", gpu_total_ms);

    // ================================================================
    //  Speedup report
    // ================================================================
    printf("============================================================\n");
    printf("  SPEEDUP SUMMARY  (N=%d assets, T=%d days)\n", N, T);
    printf("------------------------------------------------------------\n");
    printf("  %-24s  CPU: %8.2f ms   GPU: %8.2f ms   x%.1f\n",
           "Covariance",
           cpu_cov_ms, (double)ms_cov_gpu,
           cpu_cov_ms / ms_cov_gpu);
    printf("  %-24s  CPU: %8.2f ms   GPU: %8.2f ms   x%.1f\n",
           "Optimisation (GD)",
           cpu_opt_ms, (double)ms_opt_gpu,
           cpu_opt_ms / ms_opt_gpu);
    printf("  %-24s  CPU: %8.2f ms   GPU: %8.2f ms   x%.1f\n",
           "Total",
           cpu_total_ms, gpu_total_ms,
           cpu_total_ms / gpu_total_ms);
    printf("============================================================\n\n");

    // ================================================================
    //  Portfolio summaries
    // ================================================================
    std::cout << "=== CPU Portfolio ===\n";
    print_portfolio_summary(w_cpu, mu, sigma_cpu, rm.tickers);

    std::cout << "\n=== GPU Portfolio ===\n";
    print_portfolio_summary(w_gpu, mu, sigma_gpu, rm.tickers);

    // Machine-readable line for benchmark script
    printf("\nGPU_TIMING N=%d cov=%.4f opt=%.4f total=%.4f\n",
           N, (double)ms_cov_gpu, (double)ms_opt_gpu, gpu_total_ms);
    printf("CPU_TIMING N=%d cov=%.4f opt=%.4f total=%.4f\n",
           N, cpu_cov_ms, cpu_opt_ms, cpu_total_ms);

    // ----- Cleanup
    cudaFree(d_R);  cudaFree(d_mu);  cudaFree(d_sigma);
    cudaFree(d_Rdm); cudaFree(d_w);  cudaFree(d_Sw);
    cudaFree(d_grad); cudaFree(d_tmp);
    cublasDestroy(cublas);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    return 0;
}
