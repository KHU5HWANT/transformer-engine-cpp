/**
 * @file    nn/transformer_block.cpp
 * @brief   TransformerBlock — Pre-LayerNorm forward and recursive parameters().
 *
 * Pre-LN vs Post-LN — why residual is saved BEFORE LayerNorm
 * ─────────────────────────────────────────────────────────────
 * In Pre-LN, the residual path bypasses the normalisation entirely:
 *
 *   r = x            ← x flows directly to the skip connection
 *   x = ln(x)        ← only the normalised copy goes into the sub-layer
 *   x = sub_layer(x)
 *   x = x + r        ← clean residual: no LN in the gradient path
 *
 * This is critical for gradient flow.  In Post-LN (original Vaswani), the
 * residual is added BEFORE LayerNorm:
 *   x = ln(x + sub_layer(x))
 *
 * The Post-LN gradient must pass through the LayerNorm Jacobian on every
 * backward step, which can shrink gradients at depth > 12.  Pre-LN avoids
 * this — the skip connection is always a unit-Jacobian identity path.
 *
 * ops::add — why it's correct for the residual
 * ─────────────────────────────────────────────
 * ops::add(a, b)->_backward distributes the upstream gradient identically to
 * both `a` and `b`:
 *   ∂L/∂a[i] = ∂L/∂out[i]
 *   ∂L/∂b[i] = ∂L/∂out[i]
 * So the residual path receives a FULL copy of the upstream gradient.  The
 * attention/FFN path also receives a FULL copy and propagates it through its
 * own sub-graph.  The two streams are then ACCUMULATED (+=) at shared input
 * nodes — exactly how residual networks are trained.
 */

#include "nn/transformer_block.hpp"
#include "engine/ops.hpp"          // ops::add  (for skip connections)

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════
//
// Declaration order in the header drives initialisation order:
//   ln1, attn, ln2, ffn  (top-to-bottom in the class body).
//
TransformerBlock::TransformerBlock(size_t d_model, size_t n_heads)
    : ln1 (d_model)
    , attn(d_model, n_heads)
    , ln2 (d_model)
    , ffn (d_model)          // d_ff = 4 * d_model by default
{
}

// ═════════════════════════════════════════════════════════════════════════════
// forward  —  Pre-LayerNorm Transformer block
// ═════════════════════════════════════════════════════════════════════════════

NodePtr TransformerBlock::forward(const NodePtr& x) const
{
    // ─────────────────────────────────────────────────────────────────────────
    // Attention sub-layer (with residual connection 1)
    // ─────────────────────────────────────────────────────────────────────────

    // Save residual BEFORE normalisation — the skip connection must bypass LN.
    // `r1` is just a shared_ptr alias to the same underlying Node as `x`.
    // Capturing it here creates a second reference: the backward of ops::add
    // will accumulate gradients into the same Node via two distinct paths.
    const NodePtr r1 = x;

    // Pre-norm 1 — normalise along the embedding (last) dimension
    auto x_n1 = ln1.forward(x);                // [B, T, d_model]

    // Causal self-attention (12-step B×H fold, fused causal softmax)
    auto x_a  = attn.forward(x_n1);            // [B, T, d_model]

    // Skip connection 1: add post-attention output + original (pre-norm) input
    auto x1   = ops::add(x_a, r1);             // [B, T, d_model]

    // ─────────────────────────────────────────────────────────────────────────
    // FFN sub-layer (with residual connection 2)
    // ─────────────────────────────────────────────────────────────────────────

    // Save residual after the first skip connection (post-attention stream)
    const NodePtr r2 = x1;

    // Pre-norm 2 — independent gamma/beta from ln1
    auto x_n2 = ln2.forward(x1);               // [B, T, d_model]

    // Position-wise FFN: fc1 → GELU → fc2
    auto x_f  = ffn.forward(x_n2);             // [B, T, d_model]

    // Skip connection 2: add post-FFN output + post-attention stream
    return ops::add(x_f, r2);                  // [B, T, d_model]
}

// ═════════════════════════════════════════════════════════════════════════════
// parameters
// ═════════════════════════════════════════════════════════════════════════════

std::vector<NodePtr> TransformerBlock::parameters() const
{
    // Collect in the same order as the forward pass for debuggability.
    // ln1  → 2 params  {γ₁, β₁}
    // attn → 8 params  {W_Q,b_Q, W_K,b_K, W_V,b_V, W_O,b_O}
    // ln2  → 2 params  {γ₂, β₂}
    // ffn  → 4 params  {W₁,b₁, W₂,b₂}
    // Total: 16 NodePtrs per block
    auto p = ln1.parameters();

    auto a = attn.parameters();
    p.insert(p.end(), a.begin(), a.end());

    auto l = ln2.parameters();
    p.insert(p.end(), l.begin(), l.end());

    auto f = ffn.parameters();
    p.insert(p.end(), f.begin(), f.end());

    return p;
}

}  // namespace engine::nn
