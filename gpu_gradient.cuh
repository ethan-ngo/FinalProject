#pragma once
// gpu_gradient.cuh

#include <cublas_v2.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/// Project w onto the probability simplex in-place (device memory).
/// d_tmp must be a device buffer of length N (scratch).
void gpu_project_simplex(double* d_w, double* d_tmp, int N);

#ifdef __cplusplus
}
#endif

/// Full gradient-descent loop on the GPU.
///
/// @param handle    cuBLAS handle (already created by caller)
/// @param d_sigma   device N×N covariance matrix (row-major)
/// @param d_mu      device N mean-return vector
/// @param d_w       device N weights (uniform init on entry, result on exit)
/// @param d_Sw      device N scratch  (Σw)
/// @param d_grad    device N scratch  (gradient)
/// @param d_tmp     device N scratch  (simplex sort)
/// @param N         number of assets
/// @param lambda    risk-aversion parameter
/// @param lr        learning rate
/// @param max_iter  maximum iterations
/// @param tol       convergence tolerance
/// @param ms_out    if non-null, receives GPU wall time in ms
void gpu_gradient_descent(
        cublasHandle_t  handle,
        const double*   d_sigma,
        const double*   d_mu,
        double*         d_w,
        double*         d_Sw,
        double*         d_grad,
        double*         d_tmp,
        int             N,
        double          lambda,
        double          lr,
        int             max_iter,
        double          tol,
        float*          ms_out);
