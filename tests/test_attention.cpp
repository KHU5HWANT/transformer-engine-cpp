/**
 * @file    tests/test_attention.cpp
 * @brief   GTest unit tests for engine::nn::CausalSelfAttention (Step 2.3).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What the causal mask guarantees
 * ════════════════════════════════════════════════════════════════════════════
 *
 * After the scaled dot-product attention softmax, the attention weight matrix
 * for each head has shape [T, T]:
 *
 *   A[i, j] = probability that position i attends to position j
 *
 * For CAUSAL (autoregressive) attention:
 *   - A[i, j]  must be 0 for all j > i  (no future leakage)
 *   - A[i, j] >= 0 for all j <= i      (probabilities, sum to 1 over row)
 *
 * The implementation sets logits[i, j] = −∞ (−1e9) for j > i before
 * the softmax.  exp(−∞) = 0, so those positions receive exactly 0.0 weight.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * White-box causal mask verification strategy
 * ════════════════════════════════════════════════════════════════════════════
 *
 * We cannot directly inspect the internal attention weights because they are
 * fused into a custom file-local op in attention.cpp.  Instead we use a
 * modified input that lets us INFER the mask indirectly:
 *
 *   1. Feed x = zeros([1, T, d_model]) — all-zero input.
 *   2. After Q=0, K=0, the raw scores = Q@K^T/sqrt(d_k) = 0 everywhere.
 *   3. The causal mask sets upper-triangular positions to −1e9.
 *   4. Softmax of (0 with −1e9 upper triangle) = uniform over lower triangle.
 *   5. V = 0 everywhere, so output = A @ V = 0 regardless of A.
 *
 * This doesn't directly test the mask.  For a direct test we use a
 * CUSTOM PROXY: feed a crafted V where each position has a unique marker,
 * and verify the output at position i is a convex combination of positions
 * 0..i only (not i+1..T-1).
 *
 * For simplicity in this test suite we:
 *   a) Verify output shape [B, T, d_model] — always tested.
 *   b) Verify output is finite (no NaN/Inf from mask implementation).
 *   c) Use a zero-Q/K trick: set W_Q = W_K = 0, W_V = I, W_O = I.
 *      Then output[b, i, :] = Σ_j A[i,j] × x[b,j,:].
 *      With uniform lower-triangle attention, the last position i=T-1
 *      receives contributions from ALL T positions equally (1/T each),
 *      while position i=0 receives contribution only from position 0.
 *      We verify output[0, 0, :] == x[0, 0, :] (only self-attention).
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_attention.cpp engine/tensor.cpp engine/node.cpp \
 *       engine/ops.cpp engine/autograd.cpp nn/linear.cpp nn/layernorm.cpp \
 *       nn/attention.cpp nn/softmax.cpp nn/activation.cpp -lgtest -lgtest_main -pthread
 */

#include <gtest/gtest.h>
#include "nn/attention.hpp"
#include "engine/ops.hpp"
#include "engine/autograd.hpp"

#include <cmath>
#include <vector>

using engine::NodePtr;
using engine::Tensor;
using engine::nn::CausalSelfAttention;
namespace ops = engine::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Helper — create an input [B, T, d_model] filled with a deterministic pattern
// ─────────────────────────────────────────────────────────────────────────────

static NodePtr make_seq_input(size_t B, size_t T, size_t d,
                               double start = 0.1, double step = 0.05)
{
    Tensor t({B, T, d});
    for (size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = start + static_cast<double>(i) * step;
    return engine::make_input(std::move(t));
}

// ─────────────────────────────────────────────────────────────────────────────
// Output shape
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, OutputShape_B2_T8_D16_H4)
{
    CausalSelfAttention attn(16, 4);
    auto x   = make_seq_input(2, 8, 16);
    auto out = attn.forward(x);

    ASSERT_EQ(out->data.ndim(), 3u);
    EXPECT_EQ(out->data.shape()[0],  2u);   // B
    EXPECT_EQ(out->data.shape()[1],  8u);   // T
    EXPECT_EQ(out->data.shape()[2], 16u);   // d_model
}

TEST(CausalSelfAttention, OutputShape_B1_T4_D8_H2)
{
    CausalSelfAttention attn(8, 2);
    auto x   = make_seq_input(1, 4, 8);
    auto out = attn.forward(x);

    ASSERT_EQ(out->data.ndim(), 3u);
    EXPECT_EQ(out->data.shape()[0], 1u);
    EXPECT_EQ(out->data.shape()[1], 4u);
    EXPECT_EQ(out->data.shape()[2], 8u);
}

TEST(CausalSelfAttention, OutputShape_B4_T16_D64_H8)
{
    CausalSelfAttention attn(64, 8);
    auto x   = make_seq_input(4, 16, 64);
    auto out = attn.forward(x);

    ASSERT_EQ(out->data.ndim(), 3u);
    EXPECT_EQ(out->data.shape()[0],  4u);
    EXPECT_EQ(out->data.shape()[1], 16u);
    EXPECT_EQ(out->data.shape()[2], 64u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Output is finite — no NaN or Inf from the mask or softmax
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, OutputIsFinite)
{
    CausalSelfAttention attn(16, 4);
    auto x   = make_seq_input(2, 8, 16);
    auto out = attn.forward(x);

    for (double v : out->data.data()) {
        EXPECT_TRUE(std::isfinite(v))
            << "Output contains NaN or Inf — mask or softmax bug";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// d_model % n_heads != 0 must throw
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, InvalidHeadDivisionThrows)
{
    // d_model=10, n_heads=3 → 10 % 3 != 0 → must throw
    EXPECT_THROW(CausalSelfAttention(10, 3), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// parameters() — correct count
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, ParameterCount)
{
    CausalSelfAttention attn(16, 4);
    auto params = attn.parameters();
    // 4 Linear sub-modules (w_q, w_k, w_v, w_o), each with weight + bias = 2
    // Total: 4 × 2 = 8 NodePtrs
    EXPECT_EQ(params.size(), 8u);
}

TEST(CausalSelfAttention, AllParamsRequireGrad)
{
    CausalSelfAttention attn(16, 4);
    for (const auto& p : attn.parameters()) {
        EXPECT_TRUE(p->requires_grad);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal mask — verify position 0 only attends to itself
//
// Strategy: set W_V = identity, W_Q = W_K = zero, W_O = identity.
//   Q = K = 0  →  scores = 0  everywhere
//   After mask: scores[i, j<i] = 0, scores[i, j>i] = -1e9
//   After softmax: A[i, j] = 1/(i+1) for j≤i, 0 for j>i
//   V = x (identity W_V)
//   output[b, i, :] = (1/(i+1)) * Σ_{j≤i} x[b, j, :]
//
//   → output[b, 0, :] = (1/1) * x[b, 0, :]   (ONLY position 0)
//   → output[b, 1, :] = (1/2) * (x[b,0,:] + x[b,1,:])
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, CausalMaskPosition0OnlyAttendsToSelf)
{
    const size_t d = 4;
    const size_t T = 4;
    CausalSelfAttention attn(d, 1);   // 1 head, d_k = d

    // Set all projection weights to zero except W_V = I, W_O = I
    attn.w_q.weight->data.fill(0.0);  attn.w_q.bias->data.fill(0.0);
    attn.w_k.weight->data.fill(0.0);  attn.w_k.bias->data.fill(0.0);
    attn.w_o.weight->data.fill(0.0);  attn.w_o.bias->data.fill(0.0);

    // W_V = identity [d, d], b_v = 0
    attn.w_v.weight->data.fill(0.0);
    attn.w_v.bias->data.fill(0.0);
    for (size_t i = 0; i < d; ++i)
        attn.w_v.weight->data.at({i, i}) = 1.0;

    // W_O = identity [d, d], b_o = 0
    for (size_t i = 0; i < d; ++i)
        attn.w_o.weight->data.at({i, i}) = 1.0;

    // Input: each position has a distinct marker value
    // x[0, t, :] = t+1 everywhere  (so x[0,0,:]=1, x[0,1,:]=2, ...)
    Tensor xt({1, T, d});
    for (size_t t = 0; t < T; ++t)
        for (size_t di = 0; di < d; ++di)
            xt.at({0, t, di}) = static_cast<double>(t + 1);
    auto x = engine::make_input(std::move(xt));

    auto out = attn.forward(x);

    // With W_O = 0 (and b_o = 0), output is all zeros — the causal mask
    // test via W_O=I only works if the output projection preserves the signal.
    // Since W_O was set to identity AFTER fill(0.0), verify output[0,0,:]:
    //
    // output[0, 0, :] = W_O × (A[0,:] @ V[0,:,:]) + b_o
    //                  = W_O × (1.0 × x[0,0,:]) + 0   (only self-attention)
    //                  = x[0, 0, :] = [1, 1, 1, 1]
    //
    // This verifies the causal mask prevents position 0 from attending forward.
    for (size_t di = 0; di < d; ++di) {
        EXPECT_NEAR(out->data.at({0, 0, di}), 1.0, 1e-5)
            << "Position 0 output dim " << di
            << " should equal x[0,0,di]=1.0 (self-attention only)";
    }
}

TEST(CausalSelfAttention, CausalMaskUniformAttentionForLastPosition)
{
    const size_t d = 4;
    const size_t T = 4;
    CausalSelfAttention attn(d, 1);

    // Same setup as above: Q=K=0, V=I, O=I
    attn.w_q.weight->data.fill(0.0);  attn.w_q.bias->data.fill(0.0);
    attn.w_k.weight->data.fill(0.0);  attn.w_k.bias->data.fill(0.0);
    attn.w_v.weight->data.fill(0.0);  attn.w_v.bias->data.fill(0.0);
    attn.w_o.weight->data.fill(0.0);  attn.w_o.bias->data.fill(0.0);
    for (size_t i = 0; i < d; ++i) {
        attn.w_v.weight->data.at({i, i}) = 1.0;
        attn.w_o.weight->data.at({i, i}) = 1.0;
    }

    // x[0, t, :] = t+1
    Tensor xt({1, T, d});
    for (size_t t = 0; t < T; ++t)
        for (size_t di = 0; di < d; ++di)
            xt.at({0, t, di}) = static_cast<double>(t + 1);
    auto x = engine::make_input(std::move(xt));

    auto out = attn.forward(x);

    // Last position i=T-1=3 attends UNIFORMLY to positions 0..3
    // A[3, j] = 1/4 for j in {0,1,2,3}
    // output[0, 3, di] = (1/4)*(1+2+3+4) = 10/4 = 2.5
    const double expected_last = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;  // = 2.5
    for (size_t di = 0; di < d; ++di) {
        EXPECT_NEAR(out->data.at({0, T-1, di}), expected_last, 1e-5)
            << "Last position should attend uniformly to all T positions";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward — gradients flow to all parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, BackwardGradFlowsToAllParams)
{
    CausalSelfAttention attn(8, 2);
    auto x    = make_seq_input(1, 4, 8, 0.1, 0.1);
    auto out  = attn.forward(x);
    auto loss = ops::sum(out);

    engine::backward(loss);

    // Every parameter should have received a gradient (non-zero norm)
    for (const auto& p : attn.parameters()) {
        if (p == attn.w_k.bias) {
            // b_K contributes only a per-row constant to the pre-softmax scores.
            // Due to the shift-invariance of softmax, b_K has exactly zero effect 
            // on the output, and thus its mathematical gradient is exactly zero.
            continue;
        }
        double grad_norm = 0.0;
        for (double v : p->grad.data()) grad_norm += v * v;
        EXPECT_GT(grad_norm, 1e-15)
            << "Parameter has zero-norm gradient after backward — broken chain";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// zero_grad clears all parameter gradients
// ─────────────────────────────────────────────────────────────────────────────

TEST(CausalSelfAttention, ZeroGradClearsAllParams)
{
    CausalSelfAttention attn(8, 2);
    auto x    = make_seq_input(1, 4, 8);
    auto out  = attn.forward(x);
    auto loss = ops::sum(out);
    engine::backward(loss);

    attn.zero_grad();

    for (const auto& p : attn.parameters()) {
        for (double v : p->grad.data()) {
            EXPECT_DOUBLE_EQ(v, 0.0) << "zero_grad() must clear all gradients";
        }
    }
}
