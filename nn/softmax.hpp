/**
 * @file    nn/softmax.hpp
 * @brief   Numerically stable Softmax over the last dimension.
 *
 * Design (implementation_plan_v2.md, Step 2.2):
 *
 *  Numerical Stability — The Max-Subtraction Trick
 *  ─────────────────────────────────────────────────
 *  Naïve softmax computes exp(x[i]) which overflows to ∞ for x[i] > ~710
 *  (double precision) or x[i] > ~88 (float precision).  In a Transformer,
 *  scaled attention logits can reach values of ±100 or beyond — naïve
 *  softmax would produce NaN gradients immediately.
 *
 *  The max-subtraction trick exploits the property:
 *    softmax(x[i]) = softmax(x[i] − c)   for any constant c
 *
 *  Setting c = max_j(x[j]) keeps every exponent ≤ 0, preventing overflow
 *  while keeping numerical error bounded.
 *
 *  Per-row, per-last-dim formula
 *  ──────────────────────────────
 *  Input x: shape [*, N].  Treat leading dims as M independent rows.
 *
 *   m[i]    = max_j x[i, j]
 *   e[i, j] = exp(x[i, j] − m[i])
 *   s[i]    = Σ_j e[i, j]
 *   y[i, j] = e[i, j] / s[i]
 *
 *  Backward (Jacobian-vector product)
 *  ────────────────────────────────────
 *  Given dY = ∂L/∂Y and Y = softmax(X):
 *
 *   dot[i]   = Σ_j dY[i,j] · Y[i,j]         (per-row dot product)
 *   dX[i,j]  = Y[i,j] · (dY[i,j] − dot[i])
 *
 *  This is the closed-form Jacobian-vector product of softmax.
 *  Derivation:
 *    ∂y[i,j]/∂x[i,k] = y[i,j] · (δ_{jk} − y[i,k])
 *    → dX[i,j] = Σ_k dY[i,k] · y[i,k] · (δ_{jk} − y[i,j])
 *              = y[i,j] · (dY[i,j] − Σ_k dY[i,k] · y[i,k])
 *              = y[i,j] · (dY[i,j] − dot[i])
 *
 *  The saved activation is Y (the forward output), read directly from
 *  self->data in the backward lambda — no extra buffer needed.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "engine/node.hpp"    // NodePtr

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// softmax
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Numerically stable softmax over the last dimension of x.
 *
 * Applies the max-subtraction trick per row to prevent floating-point overflow.
 * The backward computes the exact Jacobian-vector product:
 *     dX[i,j] = Y[i,j] · (dY[i,j] − Σ_k dY[i,k]·Y[i,k])
 *
 * @param x  Input NodePtr, shape [*, N]  (N = last dimension size).
 * @return   NodePtr of the same shape, with output values in (0, 1)
 *           summing to 1.0 along the last dimension.
 *
 * @throws std::invalid_argument if x is a scalar (shape is empty).
 */
[[nodiscard]] NodePtr softmax(const NodePtr& x);

}  // namespace engine::nn
