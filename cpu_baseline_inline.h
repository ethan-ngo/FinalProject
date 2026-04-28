#pragma once
// cpu_baseline_inline.h
// ---------------------------------------------------------------
//  Inline (header-only) versions of the CPU covariance and GD
//  functions so main.cu can run both CPU and GPU and compare them
//  without linking a separate .cpp translation unit.
// ---------------------------------------------------------------
 
#include "utils.h"
#include <vector>
#include <cmath>
#include <iostream>
 
// ================================================================
//  Covariance  (mirrors cpu_baseline.cpp::compute_covariance)
// ================================================================
inline std::vector<double> cpu_compute_covariance(
        const ReturnMatrix& rm,
        const std::vector<double>& mu)
{
    int N = rm.N, T = rm.T;
    std::vector<double> Rdm(rm.data);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n)
            Rdm[t * N + n] -= mu[n];
 
    std::vector<double> sigma(N * N, 0.0);
    double inv_T = 1.0 / (T - 1);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            double s = 0.0;
            for (int t = 0; t < T; ++t)
                s += Rdm[t * N + i] * Rdm[t * N + j];
            sigma[i * N + j] = sigma[j * N + i] = s * inv_T;
        }
    }
    return sigma;
}
 
// ================================================================
//  Matrix-vector product
// ================================================================
inline void cpu_matvec(const std::vector<double>& A,
                       const std::vector<double>& x,
                       std::vector<double>& y,
                       int N)
{
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j)
            s += A[i * N + j] * x[j];
        y[i] = s;
    }
}
 
// ================================================================
//  Gradient descent
// ================================================================
inline std::vector<double> cpu_gradient_descent(
        const std::vector<double>& sigma,
        const std::vector<double>& mu,
        int N,
        double lambda   = 1.0,
        double lr       = 1e-3,
        int    max_iter = 2000,
        double tol      = 1e-8)
{
    std::vector<double> w(N, 1.0 / N);
    std::vector<double> Sw(N), grad(N), w_old(N);
 
    for (int iter = 0; iter < max_iter; ++iter) {
        w_old = w;
        cpu_matvec(sigma, w, Sw, N);
        for (int i = 0; i < N; ++i)
            grad[i] = 2.0 * Sw[i] - lambda * mu[i];
        for (int i = 0; i < N; ++i)
            w[i] -= lr * grad[i];
        project_simplex(w);
 
        double delta = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = w[i] - w_old[i];
            delta += d * d;
        }
        if (std::sqrt(delta) < tol) {
            std::cout << "  CPU GD converged at iteration " << iter + 1 << "\n";
            break;
        }
    }
    return w;
}