/**
 * @file    nn/transformer_block.hpp
 * @brief   A single Transformer block using the Pre-LayerNorm architecture.
 *
 * Design (implementation_plan_v2.md, Step 2.4):
 *
 *  Architecture: Pre-LayerNorm (Pre-LN)
 *  ─────────────────────────────────────
 *  Pre-LN normalises BEFORE the attention and FFN sub-layers (as opposed to
 *  the original Post-LN from Vaswani et al. 2017).  This was adopted by GPT-2
 *  and most modern LLMs because it:
 *    - Stabilises training without a warm-up learning rate schedule.
 *    - Prevents gradient vanishing at large depth (gradients flow cleanly
 *      through the un-normalised residual branch).
 *
 *  Forward pass (6 NodePtr operations):
 *  ───────────────────────────────────────
 *   r₁ = x                         (save residual 1)
 *   x  = ln1(x)                    (pre-norm 1)
 *   x  = attn(x)                   (causal self-attention)
 *   x  = ops::add(x, r₁)          (residual connection 1)
 *   r₂ = x                         (save residual 2)
 *   x  = ln2(x)                    (pre-norm 2)
 *   x  = ffn(x)                    (feed-forward network)
 *   x  = ops::add(x, r₂)          (residual connection 2)
 *   return x
 *
 *  Why ops::add for the skip connections?
 *  ───────────────────────────────────────
 *  ops::add creates a new output Node whose backward propagates the upstream
 *  gradient identically to BOTH inputs.  This correctly captures the residual
 *  path: gradients flow through the attention/FFN path AND through the direct
 *  skip connection simultaneously.
 *
 *  Sub-modules and parameter count
 *  ─────────────────────────────────
 *   ln1  (LayerNorm):              2 params (γ, β)
 *   attn (CausalSelfAttention):    8 params (W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O)
 *   ln2  (LayerNorm):              2 params (γ, β)
 *   ffn  (FeedForward):            4 params (fc1.W, fc1.b, fc2.W, fc2.b)
 *                                  ─────────────────────────────────
 *   Total per block:               16 NodePtrs
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"          // Module base
#include "nn/layernorm.hpp"       // LayerNorm
#include "nn/attention.hpp"       // CausalSelfAttention
#include "nn/feedforward.hpp"     // FeedForward
#include "engine/node.hpp"        // NodePtr

#include <cstddef>                // size_t

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// TransformerBlock
// ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct a single Transformer block.
     *
     * @param d_model  Embedding / model dimension (must be divisible by n_heads).
     * @param n_heads  Number of attention heads.
     *
     * @throws std::invalid_argument (propagated from CausalSelfAttention)
     *         if d_model % n_heads != 0.
     */
    TransformerBlock(size_t d_model, size_t n_heads);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Apply one Transformer block (Pre-LN).
     *
     * @param x  Input NodePtr, shape [B, T, d_model].
     * @return   Output NodePtr, same shape [B, T, d_model].
     */
    [[nodiscard]] NodePtr forward(const NodePtr& x) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /**
     * @brief Collect all 16 parameters from ln1, attn, ln2, ffn.
     *
     * Order: ln1 → attn → ln2 → ffn (consistent with declaration order).
     */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

    // ── Public sub-module access ──────────────────────────────────────────────

    LayerNorm          ln1;    ///< Pre-norm 1 (before attention)
    CausalSelfAttention attn;  ///< Multi-head causal self-attention
    LayerNorm          ln2;    ///< Pre-norm 2 (before FFN)
    FeedForward        ffn;    ///< Position-wise feed-forward network
};

}  // namespace engine::nn
