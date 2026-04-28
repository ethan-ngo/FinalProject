// gpu_gradient.cu
// ---------------------------------------------------------------
//  CUDA kernels for the gradient-descent optimisation loop.
//
//  Each iteration:
//    1. cuBLAS DGEMV:  Sw  = Σ · w          (gpu_gradient.cuh)
//    2. grad_kernel:   g[i]= 2*Sw[i] - λ*µ[i]
//    3. update_kernel: w[i]= w[i] - η*g[i]
//    4. project_simplex_gpu (device-side simplex projection)
// ---------------------------------------------------------------

#include "gpu_gradient.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
// Kernel: compute gradient  g[i] = 2·(Σw)[i] - λ·µ[i]
// -------------------------------------------------------------------
__global__ void grad_kernel(const double* __restrict__ Sw,
                             const double* __restrict__ mu,
                             double*       grad,
                             double lambda,
                             int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        grad[i] = 2.0 * Sw[i] - lambda * mu[i];
}

// -------------------------------------------------------------------
// Kernel: gradient step  w[i] = w[i] - η·g[i]
// -------------------------------------------------------------------
__global__ void update_kernel(double*       w,
                               const double* __restrict__ grad,
                               double lr,
                               int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        w[i] -= lr * grad[i];
}

// -------------------------------------------------------------------
// Simplex projection (device-side, single-block reduction)
//
// Algorithm (Duchi 2008):
//   sort u descending, find rho = max{j : u[j] - (sum_{i<=j} u[i] - 1)/j > 0}
//   theta = (sum_{i<=rho} u[i] - 1) / (rho+1)
//   w[i]  = max(w[i] - theta, 0)
//
// We use Thrust sort on the device so we don't need to transfer back.
// -------------------------------------------------------------------
__global__ void apply_threshold_kernel(double*       w,
                                        double        theta,
                                        int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double v = w[i] - theta;
        w[i] = v > 0.0 ? v : 0.0;
    }
}

void gpu_project_simplex(double* d_w, double* d_tmp, int N)
{
    // Copy w → tmp, sort tmp descending
    CUDA_CHECK(cudaMemcpy(d_tmp, d_w, N * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    thrust::device_ptr<double> ptr(d_tmp);
    thrust::sort(thrust::device, ptr, ptr + N, thrust::greater<double>());

    // Copy sorted values to host to find rho and theta
    // (N is at most ~1000 in this project; a small host round-trip is fine)
    std::vector<double> u(N);
    CUDA_CHECK(cudaMemcpy(u.data(), d_tmp, N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double cssv = 0.0;
    int    rho  = 0;
    for (int i = 0; i < N; ++i) {
        cssv += u[i];
        if (u[i] - (cssv - 1.0) / (i + 1) > 0)
            rho = i;
    }
    double tsum = 0.0;
    for (int i = 0; i <= rho; ++i) tsum += u[i];
    double theta = (tsum - 1.0) / (rho + 1);

    // Apply threshold on device
    int blk = 256;
    int grd = (N + blk - 1) / blk;
    apply_threshold_kernel<<<grd, blk>>>(d_w, theta, N);
}

// -------------------------------------------------------------------
// Host-side optimisation loop
// -------------------------------------------------------------------
void gpu_gradient_descent(
        cublasHandle_t  handle,
        const double*   d_sigma,  // device N×N covariance
        const double*   d_mu,     // device N mean vector
        double*         d_w,      // device N weights (initialised on entry)
        double*         d_Sw,     // device N scratch (Σw product)
        double*         d_grad,   // device N scratch (gradient)
        double*         d_tmp,    // device N scratch (simplex sort)
        int             N,
        double          lambda,
        double          lr,
        int             max_iter,
        double          tol,
        float*          ms_out)
{
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    int    blk = 256;
    int    grd = (N + blk - 1) / blk;
    double alpha = 1.0, beta = 0.0;

    // Host-side copy of w for convergence check
    std::vector<double> w_old(N), w_cur(N);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Save old weights for convergence check
        CUDA_CHECK(cudaMemcpy(w_old.data(), d_w, N * sizeof(double),
                              cudaMemcpyDeviceToHost));

        // 1. Sw = Σ · w   (cuBLAS DGEMV, row-major → treat as col-major trans)
        //    cublasDgemv: y = alpha * op(A) * x + beta * y
        //    We stored sigma row-major; cuBLAS assumes col-major.
        //    row-major A ≡ col-major Aᵀ  →  use CUBLAS_OP_T
        CUBLAS_CHECK(cublasDgemv(handle,
                                  CUBLAS_OP_T,    // transpose for row-major
                                  N, N,
                                  &alpha,
                                  d_sigma, N,     // leading dim
                                  d_w,     1,
                                  &beta,
                                  d_Sw,    1));

        // 2. gradient
        grad_kernel<<<grd, blk>>>(d_Sw, d_mu, d_grad, lambda, N);

        // 3. weight update
        update_kernel<<<grd, blk>>>(d_w, d_grad, lr, N);

        // 4. project onto simplex
        gpu_project_simplex(d_w, d_tmp, N);

        // 5. convergence check (host side, every 50 iterations)
        if ((iter + 1) % 50 == 0 || iter == max_iter - 1) {
            CUDA_CHECK(cudaMemcpy(w_cur.data(), d_w, N * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            double delta = 0.0;
            for (int i = 0; i < N; ++i) {
                double d = w_cur[i] - w_old[i];
                delta += d * d;
            }
            if (std::sqrt(delta) < tol) {
                printf("  GPU GD converged at iteration %d\n", iter + 1);
                break;
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    if (ms_out)
        CUDA_CHECK(cudaEventElapsedTime(ms_out, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}
