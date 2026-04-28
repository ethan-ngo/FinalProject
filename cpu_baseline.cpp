// cpu_baseline.cpp
// ---------------------------------------------------------------
//  Pure-CPU Markowitz mean-variance portfolio optimisation
//
//  Steps
//  ------
//  1. Load returns CSV
//  2. Compute mean return vector  µ  (N)
//  3. Compute covariance matrix   Σ  (N×N)
//  4. Gradient descent to minimise  f(w) = wᵀΣw - λ µᵀw
//     Gradient: ∇f = 2Σw - λµ
//  5. Project w onto the simplex after each step
//  6. Report timing and portfolio metrics
// ---------------------------------------------------------------
 
#include "utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <cassert>
 
// ================================================================
//  Covariance matrix  (O(N²·T) — the main bottleneck)
//  sigma[i*N+j] = (1/(T-1)) * sum_t (R[t,i]-µ[i])*(R[t,j]-µ[j])
// ================================================================
std::vector<double> compute_covariance(const ReturnMatrix& rm,
                                       const std::vector<double>& mu)
{
    int N = rm.N, T = rm.T;
    // Demean the return matrix  R_dm[t,n] = R[t,n] - mu[n]
    std::vector<double> Rdm(rm.data);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n)
            Rdm[t * N + n] -= mu[n];
 
    // Σ[i,j] = (1/(T-1)) * (column i) · (column j)
    std::vector<double> sigma(N * N, 0.0);
    double inv_T = 1.0 / (T - 1);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {       // exploit symmetry
            double s = 0.0;
            for (int t = 0; t < T; ++t)
                s += Rdm[t * N + i] * Rdm[t * N + j];
            sigma[i * N + j] = s * inv_T;
            sigma[j * N + i] = s * inv_T;   // mirror
        }
    }
    return sigma;
}
 
// ================================================================
//  Matrix-vector product  y = A x   (N×N times N)
// ================================================================
void matvec(const std::vector<double>& A,
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
//  Objective :  f(w) = wᵀΣw - λ µᵀw
//  Gradient  :  ∇f   = 2Σw  - λ µ
// ================================================================
std::vector<double> gradient_descent(
        const std::vector<double>& sigma,
        const std::vector<double>& mu,
        int N,
        double lambda    = 1.0,   // risk-aversion / return trade-off
        double lr        = 1e-3,  // learning rate (step size)
        int    max_iter  = 2000,
        double tol       = 1e-8)  // convergence tolerance
{
    // Initialise with uniform weights
    std::vector<double> w(N, 1.0 / N);
    std::vector<double> Sw(N), grad(N), w_old(N);
 
    for (int iter = 0; iter < max_iter; ++iter) {
        w_old = w;
 
        // gradient  =  2 Σ w  −  λ µ
        matvec(sigma, w, Sw, N);
        for (int i = 0; i < N; ++i)
            grad[i] = 2.0 * Sw[i] - lambda * mu[i];
 
        // gradient step
        for (int i = 0; i < N; ++i)
            w[i] -= lr * grad[i];
 
        // project onto the simplex (enforces w>=0, sum=1)
        project_simplex(w);
 
        // convergence check (L2 norm of weight change)
        double delta = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = w[i] - w_old[i];
            delta += d * d;
        }
        if (std::sqrt(delta) < tol) {
            std::cout << "  Converged at iteration " << iter + 1 << "\n";
            break;
        }
    }
    return w;
}
 
// ================================================================
//  main
// ================================================================
int main(int argc, char* argv[]) {
    std::string csv_path = (argc > 1) ? argv[1] : "data/returns.csv";
    double lambda   = (argc > 2) ? std::stod(argv[2]) : 1.0;
    int    max_iter = (argc > 3) ? std::stoi(argv[3]) : 2000;
    double lr       = (argc > 4) ? std::stod(argv[4]) : 1e-3;
 
    std::cout << "=== CPU Baseline ===\n";
    std::cout << "Loading: " << csv_path << "\n";
 
    ReturnMatrix rm = load_csv(csv_path);
    std::cout << "  T=" << rm.T << " days,  N=" << rm.N << " assets\n";
 
    Timer t_total;
    t_total.start();
 
    // ---- 1. Mean returns
    Timer t_mu; t_mu.start();
    auto mu = compute_mean(rm);
    double ms_mu = t_mu.elapsed_ms();
    std::cout << "  Mean computation      : " << ms_mu << " ms\n";
 
    // ---- 2. Covariance
    Timer t_cov; t_cov.start();
    auto sigma = compute_covariance(rm, mu);
    double ms_cov = t_cov.elapsed_ms();
    std::cout << "  Covariance computation: " << ms_cov << " ms\n";
 
    // ---- 3. Gradient descent
    Timer t_opt; t_opt.start();
    auto w = gradient_descent(sigma, mu, rm.N, lambda, lr, max_iter);
    double ms_opt = t_opt.elapsed_ms();
    std::cout << "  Optimisation (GD)     : " << ms_opt << " ms\n";
 
    double ms_total = t_total.elapsed_ms();
    std::cout << "  TOTAL                 : " << ms_total << " ms\n";
 
    // ---- 4. Results
    print_portfolio_summary(w, mu, sigma, rm.tickers);
 
    // Machine-readable summary for the benchmark script
    std::cout << "\nCPU_TIMING N=" << rm.N
              << " cov=" << ms_cov
              << " opt=" << ms_opt
              << " total=" << ms_total << "\n";
 
    return 0;
}