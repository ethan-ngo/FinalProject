#pragma once
// utils.h  –  shared helpers for CPU baseline and GPU driver

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>

// ----------------------------------------------------------------
// Timer
// ----------------------------------------------------------------
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> t0;
    void start() { t0 = Clock::now(); }
    double elapsed_ms() const {
        auto dt = Clock::now() - t0;
        return std::chrono::duration<double, std::milli>(dt).count();
    }
};

// ----------------------------------------------------------------
// CSV loader
// Returns: data[t*N + n]  = return of asset n on day t
// ----------------------------------------------------------------
struct ReturnMatrix {
    int T = 0;
    int N = 0;
    std::vector<double> data;
    std::vector<std::string> tickers;
};

inline ReturnMatrix load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);

    ReturnMatrix rm;
    std::string line;

    // Strip \r and surrounding quotes from a token
    auto strip = [](std::string s) -> std::string {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n'))
            s.pop_back();
        if (!s.empty() && s.front() == '"') s = s.substr(1);
        if (!s.empty() && s.back()  == '"') s.pop_back();
        return s;
    };

    // --- Header row: Date, ticker1, ticker2, ...
    if (!std::getline(f, line))
        throw std::runtime_error("Empty file: " + path);
    {
        std::istringstream ss(line);
        std::string tok;
        bool first = true;
        while (std::getline(ss, tok, ',')) {
            tok = strip(tok);
            if (first) { first = false; continue; }  // skip "Date" column
            rm.tickers.push_back(tok);
        }
    }
    rm.N = static_cast<int>(rm.tickers.size());
    if (rm.N == 0)
        throw std::runtime_error("No asset columns found in header");

    // --- Data rows
    while (std::getline(f, line)) {
        std::string clean = strip(line);
        if (clean.empty()) continue;

        std::istringstream ss(clean);
        std::string tok;
        bool first = true;
        int col = 0;

        while (std::getline(ss, tok, ',') && col < rm.N) {
            tok = strip(tok);
            if (first) { first = false; continue; }  // skip date value
            double v = 0.0;
            if (!tok.empty()) {
                try { v = std::stod(tok); } catch(...) { v = 0.0; }
            }
            rm.data.push_back(v);
            ++col;
        }
        // Pad any missing columns
        while (col < rm.N) { rm.data.push_back(0.0); ++col; }
        ++rm.T;
    }

    std::cout << "  Loaded " << rm.T << " rows x " << rm.N << " assets\n";
    return rm;
}

// ----------------------------------------------------------------
// Compute mean return vector  µ[n] = (1/T) * sum_t R[t,n]
// ----------------------------------------------------------------
inline std::vector<double> compute_mean(const ReturnMatrix& rm) {
    std::vector<double> mu(rm.N, 0.0);
    for (int t = 0; t < rm.T; ++t)
        for (int n = 0; n < rm.N; ++n)
            mu[n] += rm.data[t * rm.N + n];
    for (int n = 0; n < rm.N; ++n)
        mu[n] /= rm.T;
    return mu;
}

// ----------------------------------------------------------------
// Project w onto the probability simplex  (w >= 0, sum = 1)
// Algorithm: sort-based O(N log N) projection (Duchi et al. 2008)
// ----------------------------------------------------------------
inline void project_simplex(std::vector<double>& w) {
    int N = static_cast<int>(w.size());
    std::vector<double> u(w);
    std::sort(u.begin(), u.end(), std::greater<double>());

    double cssv = 0.0;
    int rho = 0;
    for (int i = 0; i < N; ++i) {
        cssv += u[i];
        if (u[i] - (cssv - 1.0) / (i + 1) > 0)
            rho = i;
    }
    double theta = 0.0;
    for (int i = 0; i <= rho; ++i) theta += u[i];
    theta = (theta - 1.0) / (rho + 1);

    for (int i = 0; i < N; ++i)
        w[i] = std::max(w[i] - theta, 0.0);
}

// ----------------------------------------------------------------
// Print a short summary of the optimised portfolio
// ----------------------------------------------------------------
inline void print_portfolio_summary(
        const std::vector<double>& w,
        const std::vector<double>& mu,
        const std::vector<double>& sigma,
        const std::vector<std::string>& tickers,
        int top_k = 10)
{
    int N = static_cast<int>(w.size());

    double ret = 0.0;
    for (int i = 0; i < N; ++i) ret += w[i] * mu[i];

    double var = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            var += w[i] * sigma[i * N + j] * w[j];

    double wsum = std::accumulate(w.begin(), w.end(), 0.0);

    std::cout << "\n=== Portfolio Summary ===\n";
    std::cout << "  Weight sum   : " << wsum << "\n";
    std::cout << "  Expected ret : " << ret  * 252 << " (annualised)\n";
    std::cout << "  Std dev (σ)  : " << std::sqrt(std::abs(var)) * std::sqrt(252.0)
              << " (annualised)\n";
    std::cout << "  Sharpe ratio : "
              << (ret * 252) / (std::sqrt(std::abs(var)) * std::sqrt(252.0)) << "\n";

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min(top_k, N), idx.end(),
                      [&](int a, int b){ return w[a] > w[b]; });

    std::cout << "\n  Top-" << top_k << " holdings:\n";
    for (int k = 0; k < std::min(top_k, N); ++k)
        std::cout << "    "
                  << (tickers.empty() ? std::to_string(idx[k]) : tickers[idx[k]])
                  << "  " << w[idx[k]] * 100.0 << " %\n";
}