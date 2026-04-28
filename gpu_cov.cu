// gpu_cov.cu
// ---------------------------------------------------------------
//  Custom CUDA kernel for the covariance matrix computation.
//
//  Kernel: each thread block handles a TILE×TILE sub-block of
//  the symmetric N×N output matrix.
//
//  Optimisations:
//    • Coalesced global memory reads (column-major de-meaned
//      return layout so each warp reads a contiguous burst)
//    • Shared-memory tiling for the time dimension → reduces
//      global bandwidth from O(T·N²) to O(T·N + N²)
//    • Exploit symmetry: only upper triangle is written; a
//      separate kernel mirrors it
// ---------------------------------------------------------------

#include "gpu_cov.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// -------------------------------------------------------------------
// Error checking macro
// -------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------
#define TILE 16   // shared-memory tile size (time dimension)

// -------------------------------------------------------------------
// Kernel 1: demean the return matrix
//   R_dm[t, n]  =  R[t, n]  −  mu[n]
//   Layout: row-major  R[t*N + n]
// -------------------------------------------------------------------
__global__ void demean_kernel(const double* __restrict__ R,
                              const double* __restrict__ mu,
                              double*       Rdm,
                              int T, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // asset index
    int t = blockIdx.y * blockDim.y + threadIdx.y;  // time  index
    if (n < N && t < T)
        Rdm[t * N + n] = R[t * N + n] - mu[n];
}

// -------------------------------------------------------------------
// Kernel 2: covariance via shared-memory tiling
//
//  Each thread (i, j) accumulates  sum_t Rdm[t,i] * Rdm[t,j]
//  using TILE-wide tiles loaded into shared memory.
//
//  Grid:  (ceil(N/TILE), ceil(N/TILE))
//  Block: (TILE, TILE)
//
//  Memory: two TILE×TILE shared memory arrays (one for col i,
//          one for col j of Rdm)
// -------------------------------------------------------------------
__global__ void cov_tiled_kernel(const double* __restrict__ Rdm,
                                 double*       sigma,
                                 int T, int N,
                                 double inv_T)
{
    // Which output element does this thread own?
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    __shared__ double tileI[TILE][TILE];   // TILE rows of time × 1 col
    __shared__ double tileJ[TILE][TILE];

    double acc = 0.0;

    // Slide the time-dimension tile
    int num_tiles = (T + TILE - 1) / TILE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int t_base = tile * TILE;

        // Load TILE time steps for column i
        int t_i = t_base + threadIdx.x;   // threadIdx.x indexes time within tile
        if (t_i < T && i < N)
            tileI[threadIdx.y][threadIdx.x] = Rdm[t_i * N + i];
        else
            tileI[threadIdx.y][threadIdx.x] = 0.0;

        // Load TILE time steps for column j
        int t_j = t_base + threadIdx.y;
        if (t_j < T && j < N)
            tileJ[threadIdx.x][threadIdx.y] = Rdm[t_j * N + j];
        else
            tileJ[threadIdx.x][threadIdx.y] = 0.0;

        __syncthreads();

        // Accumulate dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += tileI[threadIdx.y][k] * tileJ[threadIdx.x][k];

        __syncthreads();
    }

    // Write result (exploit symmetry — write both (i,j) and (j,i))
    if (i < N && j < N) {
        double val = acc * inv_T;
        sigma[i * N + j] = val;
        sigma[j * N + i] = val;   // mirror
    }
}

// -------------------------------------------------------------------
// Host-side wrapper
// -------------------------------------------------------------------
void gpu_compute_covariance(
        const double* d_R,    // device: T×N return matrix (row-major)
        const double* d_mu,   // device: mean vector (N)
        double*       d_sigma,// device: output N×N covariance
        double*       d_Rdm,  // device: scratch T×N demeaned matrix
        int T, int N,
        float* ms_out)         // optional: kernel time in ms
{
    // ---- Step 1: demean
    dim3 blk_dm(32, 8);
    dim3 grd_dm((N + 31) / 32, (T + 7) / 8);
    demean_kernel<<<grd_dm, blk_dm>>>(d_R, d_mu, d_Rdm, T, N);

    // ---- Step 2: tiled covariance
    // We time just these kernels with CUDA events
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    dim3 blk_cov(TILE, TILE);
    dim3 grd_cov((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    double inv_T = 1.0 / (T - 1);

    CUDA_CHECK(cudaEventRecord(ev_start));
    cov_tiled_kernel<<<grd_cov, blk_cov>>>(d_Rdm, d_sigma, T, N, inv_T);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    if (ms_out)
        CUDA_CHECK(cudaEventElapsedTime(ms_out, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}
