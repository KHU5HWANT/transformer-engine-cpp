/**
 * @file    benchmarks/bench_matmul.cpp
 * @brief   GFLOPS comparison: naive O(N³) vs optimized cache-blocked matmul.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Why GFLOPS?
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Matrix multiplication C = A × B of shape [N, N] × [N, N] requires:
 *   Multiplications: N³
 *   Additions:       N³ − N²  ≈  N³
 *   Total FLOPs:    2N³
 *
 * GFLOPS = (2 × N³) / (time_seconds × 1e9)
 *
 * A modern CPU core with AVX2 (8-wide double FMA) and 3 GHz clock can
 * theoretically compute 3e9 × 8 × 2 = 48 GFLOPS.  Cache-blocked matmul
 * with OpenMP can approach 20–40 GFLOPS for large N.  Naive O(N³) barely
 * reaches 1–2 GFLOPS due to cache thrashing on the inner K dimension.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Cache-blocking rationale
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The naive triple loop  C[i,j] += A[i,k] * B[k,j]  traverses B column-by-
 * column.  For large N, B does not fit in L1/L2 cache.  Every access to
 * B[k,j] is a cache miss → we stream from DRAM at ~40 GB/s instead of
 * computing from the L1 cache register file at >400 GB/s effective bandwidth.
 *
 * The blocked algorithm divides A, B, C into BLOCK_SIZE × BLOCK_SIZE tiles.
 * Each tile fits in L1/L2 cache.  All three tiles (A_tile, B_tile, C_tile)
 * are reused BLOCK_SIZE times within the tile multiply → ~BLOCK_SIZE× fewer
 * cache misses on B.
 *
 * Build:
 *   g++ -std=c++17 -O2 -fopenmp \
 *       benchmarks/bench_matmul.cpp engine/tensor.cpp engine/node.cpp \
 *       engine/ops.cpp engine/autograd.cpp -o bench_matmul
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "engine/ops.hpp"
#include "engine/node.hpp"

using namespace std::chrono;
using engine::Tensor;
using engine::NodePtr;
namespace ops = engine::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Naive O(N³) matrix multiplication — no tiling, no SIMD, no OpenMP
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Naive triple-loop matmul.  C[M,N] = A[M,K] @ B[K,N].
 * Row-major storage: A[i,k] = A_data[i*K + k]
 */
static std::vector<double> naive_matmul(
    const std::vector<double>& A,
    const std::vector<double>& B,
    size_t M, size_t K, size_t N)
{
    std::vector<double> C(M * N, 0.0);
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < K; ++k) {
            const double a_ik = A[i * K + k];
            for (size_t j = 0; j < N; ++j)
                C[i * N + j] += a_ik * B[k * N + j];
        }
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static NodePtr make_matrix(size_t M, size_t N, double fill_val = 0.5)
{
    Tensor t({M, N});
    t.fill(fill_val);
    // Add slight non-uniformity to prevent compiler optimization via constant-fold
    for (size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = 0.1 + 0.9 * static_cast<double>((i * 1234567891ULL) % 1000) / 1000.0;
    return engine::make_input(std::move(t));
}

static void print_separator()
{
    std::cout << std::string(64, '─') << "\n";
}

static void print_result(const std::string& label, double gflops, double ms)
{
    std::cout << std::left  << std::setw(30) << label
              << std::right << std::setw(10) << std::fixed << std::setprecision(3) << ms    << " ms"
              <<               std::setw(10) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
              << "\n";
}

// ─────────────────────────────────────────────────════════════════════════════
// Benchmark driver
// ─────────────────────────────────────────────────────────────────────────────

static void run_benchmark(size_t N, int warmup_reps = 2, int bench_reps = 5)
{
    const double flops = 2.0 * static_cast<double>(N) * N * N;

    std::cout << "\n▶ Matmul Benchmark  N=" << N << " × " << N
              << "  (2×N³ = " << std::fixed << std::setprecision(2)
              << flops / 1e9 << " GFLOPS theoretical per run)\n";
    print_separator();

    // ── Prepare raw data for the naive benchmark ──────────────────────────────
    NodePtr A_node = make_matrix(N, N);
    NodePtr B_node = make_matrix(N, N);
    const auto& A_data = A_node->data.data();
    const auto& B_data = B_node->data.data();

    // ── 1. Naive matmul benchmark ─────────────────────────────────────────────
    // Warm-up
    for (int r = 0; r < warmup_reps; ++r)
        (void)naive_matmul(A_data, B_data, N, N, N);

    auto t0 = steady_clock::now();
    for (int r = 0; r < bench_reps; ++r)
        (void)naive_matmul(A_data, B_data, N, N, N);
    auto t1 = steady_clock::now();

    const double naive_ms    = duration<double, std::milli>(t1 - t0).count() / bench_reps;
    const double naive_gflops = flops / (naive_ms * 1e6);
    print_result("Naive O(N³)", naive_gflops, naive_ms);

    // ── 2. Optimized engine::ops::matmul benchmark ────────────────────────────
    // Warm-up (also builds the first DAG node — ensures JIT-style effects settle)
    for (int r = 0; r < warmup_reps; ++r)
        (void)ops::matmul(A_node, B_node);

    t0 = steady_clock::now();
    for (int r = 0; r < bench_reps; ++r)
        (void)ops::matmul(A_node, B_node);
    t1 = steady_clock::now();

    const double opt_ms    = duration<double, std::milli>(t1 - t0).count() / bench_reps;
    const double opt_gflops = flops / (opt_ms * 1e6);
    print_result("Cache-blocked + OpenMP", opt_gflops, opt_ms);

    print_separator();
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
              << naive_ms / opt_ms << "×  ("
              << std::setprecision(2) << opt_gflops / naive_gflops
              << "× more GFLOPS)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         MATMUL BENCHMARK — Naive vs Cache-Blocked            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    // Benchmark three sizes:
    //   128×128 — fits easily in L1  (naive and blocked both fast)
    //   512×512 — spills to L2/L3   (cache blocking shows first advantage)
    //   1024×1024 — spills to DRAM  (cache blocking shows maximum advantage)
    run_benchmark(128,  /*warmup=*/3, /*reps=*/10);
    run_benchmark(512,  /*warmup=*/2, /*reps=*/5);
    run_benchmark(1024, /*warmup=*/1, /*reps=*/3);

    std::cout << "\nTarget: opt matmul ≥ 2× naive at N=512.  Both measured on same CPU.\n";
    return 0;
}
