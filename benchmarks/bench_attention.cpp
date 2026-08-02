/**
 * @file    benchmarks/bench_attention.cpp
 * @brief   CausalSelfAttention forward throughput benchmark.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What we measure
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Tokens/second:
 *   processed_tokens = batch_size × seq_len × num_reps
 *   throughput       = processed_tokens / elapsed_seconds
 *
 * This is the most useful unit for a language model because it captures both
 * the sequence length (quadratic attention cost) and the batch dimension.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Configuration matrix
 * ════════════════════════════════════════════════════════════════════════════
 *
 * We sweep (d_model, n_heads, T) to show how attention cost scales with T²:
 *
 *   Tiny   : d=64,  H=4,  T=32   — should be very fast
 *   Default: d=64,  H=4,  T=128  — primary training config
 *   Long   : d=64,  H=4,  T=512  — 4× longer context, ~16× attention cost
 *   Wide   : d=256, H=8,  T=128  — larger d_model, 4× more parameters
 *
 * The T² scaling of attention cost is directly observable as tokens/sec
 * drops faster than linearly with increasing T.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Target (single CPU core, d=64, T=128)
 * ════════════════════════════════════════════════════════════════════════════
 *   > 10,000 tokens/second
 *
 * Build:
 *   g++ -std=c++17 -O2 -fopenmp \
 *       benchmarks/bench_attention.cpp engine/tensor.cpp engine/node.cpp \
 *       engine/ops.cpp nn/linear.cpp nn/attention.cpp nn/softmax.cpp \
 *       nn/layernorm.cpp nn/activation.cpp -o bench_attention
 */

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "engine/ops.hpp"
#include "engine/node.hpp"
#include "nn/attention.hpp"

using namespace std::chrono;
using engine::NodePtr;
using engine::Tensor;
using engine::nn::CausalSelfAttention;
namespace ops = engine::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Helper — create a [B, T, d] input
// ─────────────────────────────────────────────────────────────────────────────

static NodePtr make_input(size_t B, size_t T, size_t d)
{
    Tensor t({B, T, d});
    for (size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = 0.01 * static_cast<double>((i * 6364136223846793005ULL) % 1000);
    return engine::make_input(std::move(t));
}

// ─────────────────────────────────────────────────────────────────────────────
// Single benchmark configuration
// ─────────────────────────────────────────────────────────────────────────────

struct AttentionConfig {
    std::string label;
    size_t d_model;
    size_t n_heads;
    size_t B;        // batch size
    size_t T;        // sequence length (context window)
    int    warmup;
    int    reps;
};

static void run_attention_benchmark(const AttentionConfig& cfg)
{
    CausalSelfAttention attn(cfg.d_model, cfg.n_heads);
    auto x = make_input(cfg.B, cfg.T, cfg.d_model);

    // Warm-up: let the OS settle page faults and JIT effects
    for (int r = 0; r < cfg.warmup; ++r)
        (void)attn.forward(x);

    // Timed run
    auto t0 = steady_clock::now();
    for (int r = 0; r < cfg.reps; ++r)
        (void)attn.forward(x);
    auto t1 = steady_clock::now();

    const double elapsed_s    = duration<double>(t1 - t0).count();
    const double elapsed_ms   = elapsed_s * 1e3 / cfg.reps;
    const double total_tokens = static_cast<double>(cfg.B * cfg.T * cfg.reps);
    const double tok_per_sec  = total_tokens / elapsed_s;

    // FLOPs: attention scores (B*H*T²*d_k) dominate; rough estimate
    const size_t d_k      = cfg.d_model / cfg.n_heads;
    const double attn_flops =
        static_cast<double>(cfg.B * cfg.n_heads) *
        (2.0 * cfg.T * cfg.T * d_k +   // Q@K^T
         2.0 * cfg.T * cfg.T * d_k);   // A@V
    const double gflops = attn_flops / (elapsed_ms * 1e6);

    std::cout << "  " << std::left << std::setw(28) << cfg.label
              << "  d=" << std::setw(4) << cfg.d_model
              << "  H=" << std::setw(2) << cfg.n_heads
              << "  T=" << std::setw(4) << cfg.T
              << "  " << std::right << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed_ms << " ms/fwd"
              << "  " << std::setw(10) << std::fixed << std::setprecision(0)
              << tok_per_sec << " tok/s"
              << "  " << std::setw(6) << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS"
              << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       CausalSelfAttention Forward Throughput Benchmark       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Metric: tokens/second = (B × T × reps) / elapsed_seconds   ║\n";
    std::cout << "║  Target: ≥ 10,000 tok/s  (d=64, H=4, T=128, B=4)            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    const std::vector<AttentionConfig> configs = {
        // label                  d     H   B    T   warm reps
        {"Tiny  (T=32)",          64,   4,  4,   32,  3,   20},
        {"Default (T=128)",       64,   4,  4,  128,  3,   10},
        {"Long context (T=512)",  64,   4,  4,  512,  2,    5},
        {"Wide model (d=256)",   256,   8,  4,  128,  2,    5},
        {"Single sample (B=1)",   64,   4,  1,  128,  3,   10},
    };

    for (const auto& cfg : configs)
        run_attention_benchmark(cfg);

    std::cout << "\n";
    std::cout << "Note: attention cost scales as O(T²·d_k) — observe tok/s\n";
    std::cout << "      dropping faster than linearly as T grows (T² attention).\n";

    return 0;
}
