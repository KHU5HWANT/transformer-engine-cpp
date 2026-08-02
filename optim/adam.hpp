/**
 * @file    optim/adam.hpp
 * @brief   AdamW optimizer — Adam with decoupled weight decay.
 *
 * Design (implementation_plan_v2.md, Step 4.1):
 *
 *  AdamW vs Adam+L2
 *  ─────────────────
 *  Standard Adam with L2 regularisation adds weight_decay × param to the
 *  gradient BEFORE the adaptive moment scaling.  This means the weight-decay
 *  step size is modulated by v̂ (the second moment), which shrinks the
 *  effective decay for parameters with large gradient variance.
 *
 *  AdamW (Loshchilov & Hutter, 2018) decouples weight decay from the gradient:
 *
 *    param ← param × (1 − lr × λ)         ← direct weight decay
 *    param ← param − lr × m̂ / (√v̂ + ε)  ← Adam gradient step
 *
 *  The weight decay is applied at the raw learning-rate scale, not at the
 *  adaptive scale — giving a consistent regularisation strength across all
 *  parameters regardless of their gradient history.
 *
 *  Update rule (one step, bias-corrected)
 *  ───────────────────────────────────────
 *   t   ← t + 1
 *   m_i ← β₁ × m_i + (1 − β₁) × g_i
 *   v_i ← β₂ × v_i + (1 − β₂) × g_i²
 *   m̂_i = m_i / (1 − β₁ᵗ)              ← bias correction
 *   v̂_i = v_i / (1 − β₂ᵗ)              ← bias correction
 *   θ_i ← θ_i − lr × λ × θ_i           ← decoupled weight decay
 *   θ_i ← θ_i − lr × m̂_i / (√v̂_i + ε) ← Adam update
 *
 *  Bias correction
 *  ────────────────
 *  At step t=1, m_1 = (1-β₁)×g.  Without bias correction, m̂_1 would still
 *  equal (1-β₁)×g rather than g, so the first step size would be smaller by
 *  a factor of (1-β₁) = 0.1.  Bias correction makes every step size adaptive
 *  from the very first update.
 *
 *  Storage
 *  ────────
 *  Per-parameter std::vector<double> m_ and v_ (not Tensors — no autograd
 *  needed for optimiser state).  Memory: 2 × numel(param) × 8 bytes per param.
 *
 * Target: Linux/WSL2, C++17, pure STL (optional OpenMP for large tensors).
 */

#pragma once

#include "engine/node.hpp"

#include <cstddef>
#include <vector>

namespace optim {

// ─────────────────────────────────────────────────────────────────────────────
// AdamW
// ─────────────────────────────────────────────────────────────────────────────

class AdamW {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct AdamW with a fixed parameter list.
     *
     * Call model.parameters() to obtain the parameter list:
     *   AdamW optim(model.parameters(), lr=3e-4);
     *
     * @param params       All trainable NodePtrs (from Module::parameters()).
     * @param lr           Learning rate η.  Typical range: [1e-4, 1e-3].
     * @param beta1        Exponential decay for the first moment (momentum).
     *                     Default: 0.9 (Adam paper value).
     * @param beta2        Exponential decay for the second moment (RMSprop).
     *                     Default: 0.999 (Adam paper value).
     * @param eps          Denominator numerical stability term.
     *                     Default: 1e-8.
     * @param weight_decay Decoupled weight decay λ.
     *                     Default: 0.01 (Loshchilov & Hutter recommended).
     *
     * @throws std::invalid_argument if params is empty.
     */
    explicit AdamW(std::vector<engine::NodePtr> params,
                   double lr           = 1e-3,
                   double beta1        = 0.9,
                   double beta2        = 0.999,
                   double eps          = 1e-8,
                   double weight_decay = 0.01);

    // ── Optimiser interface ───────────────────────────────────────────────────

    /**
     * @brief Perform one AdamW parameter update.
     *
     * Prerequisites:
     *   1. loss.backward() has been called — all param->grad tensors are
     *      populated.
     *   2. This is called BEFORE zero_grad() (grads are read, then zeroed).
     *
     * Implements:
     *   (1) Decoupled weight decay:   θ ← θ × (1 − lr × λ)
     *   (2) Moment updates:           m ← β₁m + (1−β₁)g
     *                                 v ← β₂v + (1−β₂)g²
     *   (3) Bias-corrected update:    θ ← θ − lr × m̂ / (√v̂ + ε)
     *
     * Parameters with requires_grad = false are silently skipped.
     */
    void step();

    /**
     * @brief Zero all parameter gradients (convenience — delegates to zero_grad()).
     *
     * Equivalent to calling p->zero_grad() for each parameter.
     * Prefer calling Module::zero_grad() directly to avoid redundancy, but
     * this method exists for training loops that hold only the optimizer.
     */
    void zero_grad();

    // ── Accessors ─────────────────────────────────────────────────────────────

    /** Current step count (1-indexed after the first step() call). */
    [[nodiscard]] size_t current_step() const noexcept { return t_; }

    /** Number of tracked parameters. */
    [[nodiscard]] size_t num_params()   const noexcept { return params_.size(); }

    /** Learning rate (mutable — caller may implement lr scheduling). */
    double lr;

private:
    std::vector<engine::NodePtr> params_;

    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;

    size_t t_;   ///< Global step counter — incremented at the START of each step()

    // Per-parameter first and second moment vectors (not Tensors — no autograd needed)
    std::vector<std::vector<double>> m_;   ///< First moments  m_i  (same numel as params_[i])
    std::vector<std::vector<double>> v_;   ///< Second moments v_i  (same numel as params_[i])
};

}  // namespace optim
