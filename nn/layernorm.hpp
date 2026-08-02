/**
 * @file    nn/layernorm.hpp
 * @brief   Layer Normalization over the last (embedding) dimension.
 *
 * Design (implementation_plan_v2.md, Step 2.1):
 *
 *  Formula
 *  ───────
 *   Given input  x  of shape  [*, d_model]  (any number of leading dims):
 *
 *     μᵢ    = (1/d_model) · Σⱼ x[i, j]                  (per-row mean)
 *     σᵢ²   = (1/d_model) · Σⱼ (x[i,j] − μᵢ)²           (per-row variance)
 *     x̂[i,j] = (x[i,j] − μᵢ) / √(σᵢ² + ε)              (normalised)
 *     y[i,j] = γ[j] · x̂[i,j] + β[j]                     (scale & shift)
 *
 *  where γ (weight) and β (bias) are learnable parameters of shape [d_model],
 *  initialised to 1.0 and 0.0 respectively.
 *
 *  Implementation strategy
 *  ────────────────────────
 *  The existing ops (sum, mul, exp, log) operate on flat tensors only — they
 *  do NOT support per-dimension reduction.  LayerNorm requires per-row mean
 *  and variance over the last dimension.
 *
 *  We therefore implement LayerNorm as a single fused custom op (layernorm_op)
 *  in the .cpp file, similar in structure to the matmul kernels in ops.cpp.
 *  This gives:
 *    - A clean forward pass in a single loop.
 *    - The exact analytically-derived backward pass (see below), which is more
 *      numerically stable and faster than composing many small ops.
 *    - Saved x̂ and 1/σ buffers captured in the backward lambda (no re-computation).
 *
 *  Backward derivation (for an interview)
 *  ─────────────────────────────────────
 *  Let N = d_model, r = 1/σ (reciprocal std), dY = ∂L/∂y.
 *
 *    ∂L/∂β[j]   = Σᵢ dY[i,j]
 *    ∂L/∂γ[j]   = Σᵢ dY[i,j] · x̂[i,j]
 *
 *  For ∂L/∂x, define per-row scalars:
 *    A[i] = Σₖ γ[k] · dY[i,k]
 *    B[i] = Σₖ γ[k] · dY[i,k] · x̂[i,k]
 *
 *  Then:
 *    ∂L/∂x[i,j] = rᵢ · (γ[j]·dY[i,j] − A[i]/N − x̂[i,j]·B[i]/N)
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"      // Module base
#include "engine/node.hpp"    // NodePtr, make_parameter, Tensor

#include <cstddef>            // size_t

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm
// ─────────────────────────────────────────────────────────────────────────────

class LayerNorm final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct a LayerNorm for tensors with last dimension d_model.
     *
     * @param d_model  Size of the feature/embedding dimension to normalise over.
     * @param eps      Numerical stability constant added to the variance.
     *                 Default: 1e-5 (standard in most implementations).
     */
    explicit LayerNorm(size_t d_model, double eps = 1e-5);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Normalise x over its last dimension, then scale and shift.
     *
     * Accepts any shape [..., d_model].  The leading dimensions are treated as
     * independent rows; normalisation operates only along the last axis.
     *
     * Uses the fused custom op (layernorm_op) defined in layernorm.cpp,
     * which saves x̂ and 1/σ for use in the backward pass.
     *
     * @param x  Input NodePtr, last dimension must equal d_model_.
     * @return   Output NodePtr of the same shape as x.
     *
     * @throws std::invalid_argument if x's last dimension ≠ d_model_.
     */
    [[nodiscard]] NodePtr forward(const NodePtr& x) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /**
     * @brief Returns {weight (γ), bias (β)}.
     */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

    // ── Public parameter access ───────────────────────────────────────────────

    NodePtr weight;   ///< γ (scale)  — shape [d_model], initialised to 1.0
    NodePtr bias;     ///< β (shift)  — shape [d_model], initialised to 0.0

private:
    size_t d_model_;
    double eps_;
};

}  // namespace engine::nn
