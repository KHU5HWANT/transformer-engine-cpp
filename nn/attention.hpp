/**
 * @file    nn/attention.hpp
 * @brief   Causal Multi-Head Self-Attention.
 *
 * Design (implementation_plan_v2.md, Step 2.3):
 *
 *  Architecture
 *  ────────────
 *   Four explicit Linear sub-modules for the Q, K, V, and output projections,
 *   each of shape [d_model, d_model].  Keeping them separate makes per-head
 *   gradient inspection and weight-tying experiments straightforward.
 *
 *  Multi-head splitting strategy (critical design note)
 *  ─────────────────────────────────────────────────────
 *   engine::ops only supports 2-D and 3-D tensors.  Rather than introducing
 *   4-D tensor support, we fold the batch and head dimensions together:
 *
 *     [B, T, d_model]  →  split_heads  →  [B×H, T, d_k]
 *
 *   This makes every subsequent op (transpose, matmul, softmax) a standard
 *   3-D batched call with batch size B×H.  The approach is equivalent to the
 *   4-D version mathematically and avoids any new op infrastructure.
 *
 *  Forward pass
 *  ────────────
 *   1.  Q = w_q_(x)                 [B, T, d_model]  via Linear
 *   2.  K = w_k_(x)                 [B, T, d_model]
 *   3.  V = w_v_(x)                 [B, T, d_model]
 *   4.  Q_h = split_heads(Q)        [B×H, T, d_k]   (fused permute)
 *   5.  K_h = split_heads(K)        [B×H, T, d_k]
 *   6.  V_h = split_heads(V)        [B×H, T, d_k]
 *   7.  K_hᵀ = ops::transpose(K_h) [B×H, d_k, T]
 *   8.  scores = Q_h @ K_hᵀ        [B×H, T, T]      batched matmul
 *   9.  attn = scaled_causal_softmax(scores, 1/√d_k)
 *              [B×H, T, T]   (fused scale+mask+softmax — single Node)
 *  10.  ctx  = attn @ V_h          [B×H, T, d_k]
 *  11.  ctx_m = merge_heads(ctx)   [B, T, d_model]  (inverse permute)
 *  12.  out  = w_o_(ctx_m)         [B, T, d_model]  via Linear
 *
 *  Causal mask
 *  ────────────
 *   Applied inside scaled_causal_softmax (file-local op in attention.cpp).
 *   Positions j > i in each attention row are held at exactly 0.0 in the
 *   output (exp underflows to 0 for large negative inputs) and receive
 *   zero gradient in the backward pass (explicit check in the backward lambda).
 *
 *  Autograd contract
 *  ─────────────────
 *   The module is purely composite — no manually constructed Node beyond those
 *   built by the three file-local custom ops.  Parameters are collected
 *   recursively from the four Linear sub-modules.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"      // Module base
#include "nn/linear.hpp"      // Linear sub-module
#include "engine/node.hpp"    // NodePtr

#include <cstddef>            // size_t

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// CausalSelfAttention
// ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct a Causal Multi-Head Self-Attention module.
     *
     * @param d_model  Embedding / model dimension.
     * @param n_heads  Number of attention heads.
     *                 Must divide d_model evenly.
     *
     * @throws std::invalid_argument if d_model % n_heads != 0.
     *
     * Each head operates on d_k = d_model / n_heads dimensions.
     * All four projection matrices are Xavier-initialised (via Linear).
     */
    CausalSelfAttention(size_t d_model, size_t n_heads);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Run causal multi-head self-attention.
     *
     * @param x  Input NodePtr of shape [B, T, d_model].
     * @return   Output NodePtr of shape [B, T, d_model].
     *
     * @throws std::invalid_argument if x is not 3-D or last dim ≠ d_model.
     */
    [[nodiscard]] NodePtr forward(const NodePtr& x) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /**
     * @brief Collect parameters from all four projection sub-modules.
     *
     * Returns {w_q.weight, w_q.bias, w_k.weight, w_k.bias,
     *          w_v.weight, w_v.bias, w_o.weight, w_o.bias}
     * (8 NodePtrs total when use_bias=true on all four linears).
     */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

    // ── Public sub-module access (for inspection / weight tying) ──────────────

    Linear w_q;   ///< Query projection  [d_model → d_model]
    Linear w_k;   ///< Key projection    [d_model → d_model]
    Linear w_v;   ///< Value projection  [d_model → d_model]
    Linear w_o;   ///< Output projection [d_model → d_model]

private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_k_;      ///< d_model / n_heads  — dimension per head
};

}  // namespace engine::nn
