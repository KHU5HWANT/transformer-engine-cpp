/**
 * @file    optim/adam.cpp
 * @brief   AdamW optimizer implementation.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Decoupled weight decay — why it matters
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Adam with L2 (naïve "AdamL2"):
 *   g̃ = g + λθ
 *   m ← β₁m + (1−β₁)g̃
 *   v ← β₂v + (1−β₂)g̃²
 *   θ ← θ − η × m̂ / (√v̂ + ε)
 *
 * The weight decay λθ is absorbed into the gradient BEFORE the adaptive
 * scaling.  For a parameter with large g², the second moment v is large,
 * so the effective step size η/(√v̂+ε) is small.  This means the
 * weight-decay force is ALSO small for high-variance parameters — the
 * regularisation strength depends on gradient history, which is unwanted.
 *
 * AdamW (decoupled):
 *   θ ← θ × (1 − η×λ)                   ← applied at full learning rate
 *   m ← β₁m + (1−β₁)g                   ← g, NOT g̃
 *   v ← β₂v + (1−β₂)g²
 *   θ ← θ − η × m̂ / (√v̂ + ε)
 *
 * Weight decay is applied at the raw learning rate, independent of the
 * per-parameter adaptive scale.  Every parameter decays at exactly η×λ
 * per step, giving uniform L2 regularisation regardless of gradient variance.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Bias correction — why the first few steps are stable
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Without bias correction:
 *   m₁ = (1−β₁)×g₁  ≈ 0.1 × g₁  (initialised at 0)
 *   v₁ = (1−β₂)×g₁² ≈ 0.001 × g₁²
 *
 * First step size = η × m₁ / (√v₁ + ε) ≈ η × 0.1×g₁ / (0.032×|g₁| + ε)
 *                ≈ η × 3.16 × sign(g₁)   ← 3.16× amplification!
 *
 * With bias correction:
 *   m̂₁ = m₁/(1−β₁ᵗ) = m₁/(1−0.9) = m₁/0.1 = g₁
 *   v̂₁ = v₁/(1−β₂ᵗ) = v₁/(1−0.999) = v₁/0.001 = g₁²
 *   First step = η × g₁ / (|g₁| + ε)  ≈ η × sign(g₁)  ← clean unit step
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Memory layout
 * ════════════════════════════════════════════════════════════════════════════
 *
 * m_ and v_ are std::vector<std::vector<double>>, indexed by parameter index.
 * m_[i] and v_[i] have the same size as params_[i]->data.numel().
 *
 * We do NOT use Tensor here because:
 *   1. No autograd is needed for optimiser state.
 *   2. std::vector<double> gives direct data_ptr() access without overhead.
 *   3. Avoids shape-validation costs inside the inner loop.
 */

#include "optim/adam.hpp"

#include <cmath>          // std::pow, std::sqrt
#include <stdexcept>
#include <string>

namespace optim {

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════

AdamW::AdamW(std::vector<engine::NodePtr> params,
             double lr_init,
             double beta1,
             double beta2,
             double eps,
             double weight_decay)
    : params_      (std::move(params))
    , lr           (lr_init)
    , beta1_       (beta1)
    , beta2_       (beta2)
    , eps_         (eps)
    , weight_decay_(weight_decay)
    , t_           (0)
{
    if (params_.empty()) {
        throw std::invalid_argument(
            "AdamW: parameter list is empty. "
            "Call model.parameters() and ensure the model has trainable weights.");
    }

    // Pre-allocate moment vectors — initialised to 0 (unbiased start).
    m_.resize(params_.size());
    v_.resize(params_.size());

    for (size_t i = 0; i < params_.size(); ++i) {
        const size_t n = params_[i]->data.numel();
        m_[i].assign(n, 0.0);
        v_[i].assign(n, 0.0);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// step
// ═════════════════════════════════════════════════════════════════════════════

void AdamW::step()
{
    // ── Increment step counter ────────────────────────────────────────────────
    ++t_;

    // ── Bias-correction denominators ──────────────────────────────────────────
    //
    // Precompute once per step — identical for all parameters.
    //
    // bc1 = 1 − β₁ᵗ,  bc2 = 1 − β₂ᵗ
    //
    // As t → ∞, bc1 → 1 and bc2 → 1 (bias correction vanishes).
    //
    const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(t_));
    const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(t_));

    // ── Per-parameter update ──────────────────────────────────────────────────
    for (size_t i = 0; i < params_.size(); ++i) {
        const engine::NodePtr& p = params_[i];

        // Skip non-trainable parameters (e.g. input nodes, frozen layers)
        if (!p->requires_grad) continue;

        const size_t  n      = p->data.numel();
        double*       theta  = p->data.data_ptr();   // parameter values (mutable)
        const double* g_ptr  = p->grad.data_ptr();   // gradient (read-only in this step)
        double*       m      = m_[i].data();          // first moment
        double*       v      = v_[i].data();          // second moment

        // ─────────────────────────────────────────────────────────────────────
        // Inner loop — three fused sub-steps per element
        // ─────────────────────────────────────────────────────────────────────
        //
        // Step (a): decoupled weight decay
        //   θ ← θ × (1 − lr × λ)
        //
        // Step (b): moment updates
        //   m ← β₁ × m + (1 − β₁) × g
        //   v ← β₂ × v + (1 − β₂) × g²
        //
        // Step (c): bias-corrected Adam parameter update
        //   m̂ = m / bc1,  v̂ = v / bc2
        //   θ ← θ − lr × m̂ / (√v̂ + ε)
        //
        // Steps (a) and (c) are separate applications to θ, both within
        // the same iteration for L1 cache friendliness.
        //
        // NOTE: OpenMP is intentionally not added here because the inner loop
        // is per-parameter (each param has its own m/v arrays with no sharing),
        // and the outer loop over params_ is already the natural parallel unit.
        // Applying #pragma omp parallel for on the outer loop over params_ in
        // a future multi-threaded trainer would be the correct granularity.
        //
        const double decay_factor = 1.0 - lr * weight_decay_;

        for (size_t j = 0; j < n; ++j) {
            const double g = g_ptr[j];

            // (a) Decoupled weight decay — applied at raw lr scale
            theta[j] *= decay_factor;

            // (b) Exponential moving averages
            m[j] = beta1_ * m[j] + (1.0 - beta1_) * g;
            v[j] = beta2_ * v[j] + (1.0 - beta2_) * g * g;

            // (c) Bias-corrected Adam step
            const double m_hat = m[j] / bc1;
            const double v_hat = v[j] / bc2;
            theta[j] -= lr * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// zero_grad
// ═════════════════════════════════════════════════════════════════════════════

void AdamW::zero_grad()
{
    for (const engine::NodePtr& p : params_) {
        p->zero_grad();
    }
}

}  // namespace optim
