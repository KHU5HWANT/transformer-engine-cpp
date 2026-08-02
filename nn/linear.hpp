/**
 * @file    nn/linear.hpp
 * @brief   Fully-connected linear transformation:  y = x Wᵀ + b
 *
 * Design (implementation_plan_v2.md, Step 2.1):
 *
 *  Shapes
 *  ──────
 *   weight  :  [out_features, in_features]   (W)
 *   bias    :  [out_features]                (b)
 *   input x :  [B, in_features]  or  [B, T, in_features]   (2-D or 3-D)
 *   output  :  [B, out_features] or  [B, T, out_features]
 *
 *  Forward Pass
 *  ─────────────
 *   1. Wᵀ = ops::transpose(weight)         [in_features, out_features]
 *   2. xWᵀ = ops::matmul(x, Wᵀ)           [..., out_features]
 *   3. out = bias_add(xWᵀ, bias)           [..., out_features]  (broadcast)
 *
 *   ops::transpose and ops::matmul are fully differentiable (Step 1.3).
 *   bias_add is a file-local differentiable op in linear.cpp that correctly
 *   reduces the bias gradient by summing over all batch/time dimensions.
 *
 *  Initialisation
 *  ───────────────
 *   Weights use Xavier/Glorot uniform:
 *     bound = sqrt(6 / (in_features + out_features))
 *     W ~ Uniform(-bound, +bound)
 *   Bias is initialised to 0.0.
 *   Both are wrapped with make_parameter() so requires_grad = true.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"      // Module base
#include "engine/node.hpp"    // NodePtr, make_parameter, Tensor

#include <cstddef>            // size_t

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// Linear
// ─────────────────────────────────────────────────────────────────────────────

class Linear final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct a Linear layer with Xavier-initialised weights.
     *
     * @param in_features   Width of each input vector.
     * @param out_features  Width of each output vector.
     * @param use_bias      If false, no bias term is applied or tracked.
     */
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Apply the linear transformation:  out = x @ Wᵀ + b
     *
     * Accepts 2-D input [B, in_features] or 3-D input [B, T, in_features].
     * Returns the same rank with the last dimension replaced by out_features.
     *
     * All operations are fully differentiable — ops::transpose, ops::matmul,
     * and the file-local bias_add all register correct backward lambdas.
     *
     * @param x  Input NodePtr.  Last dimension must equal in_features_.
     * @return   Output NodePtr of shape [..., out_features].
     *
     * @throws std::invalid_argument if x's last dimension ≠ in_features_.
     */
    [[nodiscard]] NodePtr forward(const NodePtr& x) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /**
     * @brief Returns {weight} or {weight, bias} depending on use_bias_.
     */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

    // ── Public parameter access (useful for inspection / weight tying) ────────

    NodePtr weight;   ///< [out_features, in_features]  — Xavier uniform init
    NodePtr bias;     ///< [out_features]               — zero init (nullptr if !use_bias_)

private:
    size_t in_features_;
    size_t out_features_;
    bool   use_bias_;
};

}  // namespace engine::nn
