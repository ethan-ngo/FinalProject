#pragma once
// gpu_cov.cuh  –  declaration of the GPU covariance helper

#ifdef __cplusplus
extern "C" {
#endif

/// Compute the N×N covariance matrix on the GPU.
///
/// @param d_R      device pointer – T×N return matrix (row-major, double)
/// @param d_mu     device pointer – N mean-return vector (double)
/// @param d_sigma  device pointer – N×N output covariance (double)
/// @param d_Rdm    device pointer – T×N scratch for demeaned returns (double)
/// @param T        number of time periods
/// @param N        number of assets
/// @param ms_out   if non-null, receives kernel wall time in milliseconds
void gpu_compute_covariance(const double* d_R,
                             const double* d_mu,
                             double*       d_sigma,
                             double*       d_Rdm,
                             int T, int N,
                             float* ms_out);

#ifdef __cplusplus
}
#endif
