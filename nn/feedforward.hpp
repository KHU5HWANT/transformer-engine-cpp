/**
 * @file    nn/feedforward.hpp
 * @brief   Transformer Position-wise Feed-Forward Network (FFN).
 *
 * Design (implementation_plan_v2.md, Step 2.4):
 *
 *  Structure
 *  ─────────
 *   fc1  : Linear(d_model,  d_ff)        [d_ff = 4 × d_model by default]
 *   GELU : engine::nn::gelu              (exact erf formula, Step 2.2)
 *   fc2  : Linear(d_ff,  d_model)
 *
 *   out = fc2(gelu(fc1(x)))
 *
 *  Why 4 × d_model?
 *  ─────────────────
 *   The expansion factor of 4 comes from Vaswani et al. (2017) "Attention Is
 *   All You Need" and is preserved in GPT-2 / GPT-3.  The MLP forms a
 *   "superposition memory": each of the 4d_model neurons can, in theory, store
 *   one key-value fact independently of the others.
 *
 *  Parameters collected by parameters()
 *  ─────────────────────────────────────
 *   {fc1.weight, fc1.bias, fc2.weight, fc2.bias}  — 4 NodePtrs total.
 *
 *  Autograd graph
 *  ───────────────
 *   All three sub-operations are fully differentiable:
 *     fc1 via Linear::forward (matmul + bias_add)
 *     gelu via the fused gelu op (Step 2.2 — exact erf backward)
 *     fc2 via Linear::forward
 *   No custom ops are needed here — the DAG is assembled from existing ops.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"      // Module base
#include "nn/linear.hpp"      // Linear sub-module
#include "nn/activation.hpp"  // GELU
#include "engine/node.hpp"    // NodePtr

#include <cstddef>            // size_t

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward
// ─────────────────────────────────────────────────────────────────────────────

class FeedForward final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct an FFN sub-layer.
     *
     * @param d_model  Input and output embedding dimension.
     * @param d_ff     Hidden dimension.  Pass 0 (default) to use 4 × d_model.
     *
     * Weights are Xavier-initialised by Linear.  Biases are zero-initialised.
     */
    explicit FeedForward(size_t d_model, size_t d_ff = 0);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Apply the FFN:  out = fc2(gelu(fc1(x)))
     *
     * @param x  Input NodePtr, shape [*, d_model].
     * @return   Output NodePtr, same shape [*, d_model].
     */
    [[nodiscard]] NodePtr forward(const NodePtr& x) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /**
     * @brief Returns {fc1.weight, fc1.bias, fc2.weight, fc2.bias}.
     */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

private:
    size_t d_model_;
    size_t d_ff_;

public:
    // ── Public sub-module access ──────────────────────────────────────────────

    Linear fc1;   ///< First projection:  [d_model → d_ff]
    Linear fc2;   ///< Second projection: [d_ff → d_model]
};

}  // namespace engine::nn
