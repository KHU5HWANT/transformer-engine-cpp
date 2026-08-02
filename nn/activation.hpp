/**
 * @file    nn/activation.hpp
 * @brief   Element-wise differentiable activation functions.
 *
 * Design (implementation_plan_v2.md, Step 2.2):
 *
 *  All three activations are implemented as FUSED CUSTOM OPS in activation.cpp,
 *  following the same autograd contract as layernorm_op (Step 2.1).
 *  No intermediate Nodes are created — a single op reads the input tensor,
 *  computes the output, and stores the analytically-derived backward in a
 *  single lambda.
 *
 *  Why fused vs. composed primitives?
 *  ────────────────────────────────────
 *  - GELU via primitives would require: mul → div → erf → add → mul → mul
 *    (6 intermediate nodes, 6 backward lambdas, 6 passes over memory).
 *    The fused op reads x once and writes dX once in a single backward call.
 *  - ReLU's backward is a simple mask (x > 0 ? 1 : 0); composing with mul/exp
 *    is unnecessary overhead.
 *  - Sigmoid's backward uses y·(1−y) — it reads self->data (the FORWARD output),
 *    not the input x.  This saves storing an extra buffer.
 *
 *  Backward summaries
 *  ──────────────────
 *  Let dY = ∂L/∂y (upstream gradient from this op's output node).
 *
 *   GELU (exact):
 *     y     = x · Φ(x)      where Φ(x) = 0.5 · (1 + erf(x / √2))
 *     dX[i] = dY[i] · (Φ(xᵢ) + xᵢ · φ(xᵢ))
 *             where φ(x) = exp(−x²/2) / √(2π)  (standard normal PDF)
 *
 *   ReLU:
 *     y     = max(0, x)
 *     dX[i] = dY[i]  if xᵢ > 0,  else 0
 *             (mask derived from INPUT x, not output y)
 *
 *   Sigmoid:
 *     y     = 1 / (1 + exp(−x))
 *     dX[i] = dY[i] · yᵢ · (1 − yᵢ)
 *             (uses FORWARD OUTPUT y — no need to recompute or cache x)
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "engine/node.hpp"    // NodePtr

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// GELU — Gaussian Error Linear Unit
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Element-wise GELU: y[i] = x[i] · 0.5 · (1 + erf(x[i] / √2))
 *
 * Uses the EXACT formula (not the tanh approximation) via std::erf.
 * The backward is the full analytical derivative: Φ(x) + x·φ(x).
 *
 * Preferred over ReLU for Transformers — smoother gradient flow and better
 * empirical performance on language modelling tasks.
 *
 * @param x  Input NodePtr, any shape.
 * @return   NodePtr of the same shape.
 */
[[nodiscard]] NodePtr gelu(const NodePtr& x);

// ─────────────────────────────────────────────────────────────────────────────
// ReLU — Rectified Linear Unit
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Element-wise ReLU: y[i] = max(0, x[i])
 *
 * Backward: dX[i] = dY[i] if x[i] > 0, else 0.
 * The backward captures the INPUT x (not the output) to apply the mask.
 *
 * @param x  Input NodePtr, any shape.
 * @return   NodePtr of the same shape.
 */
[[nodiscard]] NodePtr relu(const NodePtr& x);

// ─────────────────────────────────────────────────────────────────────────────
// Sigmoid
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Element-wise sigmoid: y[i] = 1 / (1 + exp(−x[i]))
 *
 * Backward: dX[i] = dY[i] · y[i] · (1 − y[i]).
 * The backward reads self->data (the FORWARD output) — numerically identical
 * to recomputing sigmoid(x) but without touching the input tensor at all.
 *
 * @param x  Input NodePtr, any shape.
 * @return   NodePtr of the same shape.
 */
[[nodiscard]] NodePtr sigmoid(const NodePtr& x);

}  // namespace engine::nn
