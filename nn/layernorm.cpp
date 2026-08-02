/**
 * @file    nn/layernorm.cpp
 * @brief   Implementation of Layer Normalization.
 *
 * See nn/layernorm.hpp for the full mathematical derivation, backward formula,
 * and motivation for using a fused custom op.
 *
 * The core of this file is layernorm_op(), a file-local differentiable op that
 * follows the same 4-step autograd contract as the ops in engine/ops.cpp:
 *
 *   Step 1: Forward computation (single pass over rows).
 *   Step 2: Create output NodePtr.
 *   Step 3: Register DAG edges via add_child().
 *   Step 4: Assign _backward lambda with the analytically-derived gradient
 *           and the weak_ptr self-capture cycle-prevention pattern.
 *
 * Memory layout
 * ─────────────
 *   Input x has shape [*, N] where N = d_model.
 *   We flatten the leading dimensions to M rows:
 *     M = x->data.numel() / N
 *   This works for 2-D ([B, N]), 3-D ([B, T, N]), or higher-rank inputs.
 *
 * Saved activations for backward
 * ──────────────────────────────
 *   x_hat  : std::vector<double>(M * N)  — normalised values (x̂[i,j])
 *   inv_std: std::vector<double>(M)      — per-row reciprocal standard dev (rᵢ)
 *
 *   These are captured by MOVE into the backward lambda using C++14/17
 *   generalised (init-capture) syntax:
 *     [x_hat = std::move(x_hat_buf), inv_std = std::move(inv_std_buf), ...]
 *   The lambda owns the storage; the raw pointers used inside are always valid
 *   for the lifetime of the lambda (i.e. the lifetime of the output Node).
 */

#include "nn/layernorm.hpp"
#include "engine/node.hpp"     // Node::make, make_parameter, NodePtr, Tensor

#include <cassert>
#include <cmath>               // std::sqrt
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// File-local helper op: layernorm_op
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// layernorm_op
// ─────────────────────────────────────────────────────────────────────────────
//
// Fused differentiable LayerNorm over the last dimension of x.
//
//   Forward  (per row i):
//     μᵢ     = (1/N) Σⱼ x[i,j]
//     σᵢ²    = (1/N) Σⱼ (x[i,j] − μᵢ)²
//     rᵢ     = 1 / √(σᵢ² + ε)
//     x̂[i,j] = (x[i,j] − μᵢ) · rᵢ
//     y[i,j] = γ[j] · x̂[i,j] + β[j]
//
//   Backward (let dY = self->grad, gd = γ.data):
//     ∂L/∂β[j]   = Σᵢ dY[i,j]
//     ∂L/∂γ[j]   = Σᵢ dY[i,j] · x̂[i,j]
//     A[i]       = Σₖ γ[k]·dY[i,k]
//     B[i]       = Σₖ γ[k]·dY[i,k]·x̂[i,k]
//     ∂L/∂x[i,j] = rᵢ · (γ[j]·dY[i,j] − A[i]/N − x̂[i,j]·B[i]/N)
//
// Parameters:
//   x      — NodePtr, shape [*, N]
//   gamma  — NodePtr, shape [N]  (the learnable scale γ)
//   beta   — NodePtr, shape [N]  (the learnable shift β)
//   eps    — numerical stability constant
//   N      — size of last dimension (d_model)
//
NodePtr layernorm_op(const NodePtr& x,
                     const NodePtr& gamma,
                     const NodePtr& beta,
                     double         eps,
                     size_t         N)
{
    const size_t total = x->data.numel();
    const size_t M     = total / N;            // number of independent rows

    const double* xd  = x->data.data_ptr();
    const double* gd  = gamma->data.data_ptr();
    const double* bd  = beta->data.data_ptr();

    // ── Forward: compute x_hat, inv_std, and output y ────────────────────────

    std::vector<double> x_hat_buf(total);      // saved for backward
    std::vector<double> inv_std_buf(M);        // saved for backward
    std::vector<double> fwd(total);

    const double inv_N = 1.0 / static_cast<double>(N);

    for (size_t i = 0; i < M; ++i) {
        const double* x_row   = xd           + i * N;
        double*       xh_row  = x_hat_buf.data() + i * N;
        double*       y_row   = fwd.data()   + i * N;

        // ── Per-row mean ──────────────────────────────────────────────────────
        double mean = 0.0;
        for (size_t j = 0; j < N; ++j) mean += x_row[j];
        mean *= inv_N;

        // ── Per-row variance ──────────────────────────────────────────────────
        double var = 0.0;
        for (size_t j = 0; j < N; ++j) {
            const double d = x_row[j] - mean;
            var += d * d;
        }
        var *= inv_N;

        // ── Reciprocal std and normalised values ──────────────────────────────
        const double r = 1.0 / std::sqrt(var + eps);
        inv_std_buf[i] = r;

        for (size_t j = 0; j < N; ++j) {
            xh_row[j] = (x_row[j] - mean) * r;
            y_row[j]  = gd[j] * xh_row[j] + bd[j];
        }
    }

    // ── Create output node ────────────────────────────────────────────────────
    auto out = Node::make(
        Tensor(x->data.shape(), std::move(fwd)),
        x->requires_grad || gamma->requires_grad || beta->requires_grad
    );

    out->add_child(x);
    out->add_child(gamma);
    out->add_child(beta);

    // ── Backward lambda ───────────────────────────────────────────────────────
    // Captures:
    //   x, gamma, beta   — shared_ptr (keep inputs alive; provide data ptrs)
    //   x_hat, inv_std   — MOVED in (lambda owns the saved-activation buffers)
    //   M, N, inv_N      — by value (cheap scalars)
    //   wout             — weak_ptr (breaks Node → lambda → Node self-cycle)
    out->_backward = [x, gamma, beta, M, N, inv_N, eps,
                      x_hat    = std::move(x_hat_buf),
                      inv_std  = std::move(inv_std_buf),
                      w_out    = std::weak_ptr<Node>(out)]()
    {
        auto self = w_out.lock();
        if (!self) return;

        const double* dY  = self->grad.data_ptr();
        const double* gd  = gamma->data.data_ptr();
        const double* xh  = x_hat.data();
        const double* rs  = inv_std.data();

        // Zero-initialised gradient tensors
        Tensor dx   (x->data.shape()    );   // [*, N]
        Tensor dgamma(gamma->data.shape());   // [N]
        Tensor dbeta (beta->data.shape() );   // [N]

        double* dxp  = dx.data_ptr();
        double* dgp  = dgamma.data_ptr();
        double* dbp  = dbeta.data_ptr();

        // ── Sweep over rows ───────────────────────────────────────────────────
        for (size_t i = 0; i < M; ++i) {
            const double* dY_row = dY + i * N;
            const double* xh_row = xh + i * N;
            double*       dx_row = dxp + i * N;
            const double  r_i    = rs[i];

            // ── Per-row accumulators ──────────────────────────────────────────
            // A[i] = Σₖ γ[k] · dY[i,k]
            // B[i] = Σₖ γ[k] · dY[i,k] · x̂[i,k]
            double Ai = 0.0, Bi = 0.0;
            for (size_t j = 0; j < N; ++j) {
                const double g_dy = gd[j] * dY_row[j];
                Ai += g_dy;
                Bi += g_dy * xh_row[j];
            }

            // ── ∂L/∂x[i,j] ───────────────────────────────────────────────────
            // = rᵢ · (γ[j]·dY[i,j]  −  A[i]/N  −  x̂[i,j]·B[i]/N)
            for (size_t j = 0; j < N; ++j) {
                dx_row[j] = r_i * (gd[j] * dY_row[j]
                                   - inv_N * Ai
                                   - xh_row[j] * inv_N * Bi);
            }

            // ── ∂L/∂γ and ∂L/∂β (accumulated over all rows i) ────────────────
            for (size_t j = 0; j < N; ++j) {
                dgp[j] += dY_row[j] * xh_row[j];   // ∂L/∂γ[j] += dY[i,j]·x̂[i,j]
                dbp[j] += dY_row[j];               // ∂L/∂β[j] += dY[i,j]
            }
        }

        x->accumulate_grad(dx);
        gamma->accumulate_grad(dgamma);
        beta->accumulate_grad(dbeta);
    };

    return out;
}

}  // anonymous namespace


// ═════════════════════════════════════════════════════════════════════════════
// LayerNorm  —  public implementation
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
//
// γ (weight) = 1.0  → identity scale at init; output == normalised input.
// β (bias)   = 0.0  → no shift at init.
// Both are learnable parameters tracked by the autograd graph.
//
LayerNorm::LayerNorm(size_t d_model, double eps)
    : d_model_(d_model)
    , eps_    (eps)
{
    // γ (scale) — initialised to 1.0
    weight = make_parameter(
        Tensor({d_model}, std::vector<double>(d_model, 1.0)));

    // β (shift) — initialised to 0.0
    bias   = make_parameter(
        Tensor({d_model}, std::vector<double>(d_model, 0.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// forward
// ─────────────────────────────────────────────────────────────────────────────

NodePtr LayerNorm::forward(const NodePtr& x) const
{
    // Validate last dimension
    if (x->data.shape().empty() ||
        x->data.shape().back() != d_model_) {
        throw std::invalid_argument(
            "LayerNorm::forward: input last dimension " +
            std::to_string(x->data.shape().empty()
                           ? 0 : x->data.shape().back()) +
            " != d_model " + std::to_string(d_model_));
    }

    return layernorm_op(x, weight, bias, eps_, d_model_);
}

// ─────────────────────────────────────────────────────────────────────────────
// parameters
// ─────────────────────────────────────────────────────────────────────────────

std::vector<NodePtr> LayerNorm::parameters() const
{
    return {weight, bias};   // {γ, β}
}

}  // namespace engine::nn
