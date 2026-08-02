/**
 * @file    nn/activation.cpp
 * @brief   Fused custom-op implementations of GELU, ReLU, and Sigmoid.
 *
 * All three follow the same 4-step autograd contract used in layernorm.cpp:
 *   1. Forward pass: single element-wise loop over x->data.
 *   2. Create output NodePtr with Node::make().
 *   3. Register DAG edge: out->add_child(x).
 *   4. Assign _backward lambda with analytical derivative and weak_ptr self-capture.
 *
 * Numerical constants used across all activations (file-scope, unnamed namespace):
 *   INV_SQRT2   = 1/√2    ≈ 0.7071067811865476
 *   INV_SQRT2PI = 1/√(2π) ≈ 0.3989422804014327
 *
 * Memory safety pattern (same as layernorm_op):
 *   - GELU / ReLU capture `x` (shared_ptr) in the backward lambda to access
 *     x->data for the local gradient computation.
 *   - Sigmoid captures NOTHING except wout — the backward only reads
 *     self->data (the forward output already stored on the node).
 *   - wout = std::weak_ptr<Node>(out) breaks the reference cycle in all cases.
 *
 * Why GELU's backward reads INPUT x (not output y):
 *   GELU backward = Φ(x) + x·φ(x), which depends on x directly.
 *   Recomputing from y is not algebraically simple (unlike sigmoid), so we
 *   must retain a path to x->data.  Capturing x by shared_ptr achieves this
 *   without saving a separate buffer.
 */

#include "nn/activation.hpp"

#include <cmath>               // std::erf, std::exp
#include <vector>

namespace engine::nn {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Numerical constants
// ─────────────────────────────────────────────────────────────────────────────

// 1/√2 — used to normalise x before erf() in GELU
static constexpr double INV_SQRT2   = 0.7071067811865476;

// 1/√(2π) — standard normal PDF at x=0;  used in GELU backward
static constexpr double INV_SQRT2PI = 0.3989422804014327;

}  // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// GELU
// ═════════════════════════════════════════════════════════════════════════════
//
// Forward:  y[i] = 0.5 · x[i] · (1 + erf(x[i] / √2))
//         = x[i] · Φ(x[i])        (Φ = CDF of standard normal)
//
// Backward: dX[i] = dY[i] · (Φ(x[i]) + x[i] · φ(x[i]))
//   where   Φ(x)  = 0.5 · (1 + erf(x/√2))
//           φ(x)  = exp(−x²/2) / √(2π)   (PDF of standard normal)
//
// We capture `x` (shared_ptr) in the backward lambda to access x->data_ptr().
// The input values are NOT separately cached — we recompute Φ and φ from x
// on demand.  This saves memory at the cost of re-evaluating erf() once per
// backward pass (acceptable: erf is O(1), memory is the bottleneck).
//
NodePtr gelu(const NodePtr& x)
{
    const size_t n  = x->data.numel();
    const double* xd = x->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) {
        const double xi = xd[i];
        fwd[i] = 0.5 * xi * (1.0 + std::erf(xi * INV_SQRT2));
    }

    auto out = Node::make(
        Tensor(x->data.shape(), std::move(fwd)),
        x->requires_grad
    );
    out->add_child(x);

    // ── Backward ─────────────────────────────────────────────────────────────
    // dX[i] = dY[i] * (Phi(xi) + xi * phi(xi))
    // Reads x->data to recompute Phi and phi.
    out->_backward = [x, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const size_t n   = self->grad.numel();
        const double* xd  = x->data.data_ptr();
        const double* dYd = self->grad.data_ptr();

        Tensor dx(x->data.shape());
        double* dxp = dx.data_ptr();

        for (size_t i = 0; i < n; ++i) {
            const double xi  = xd[i];
            const double cdf = 0.5 * (1.0 + std::erf(xi * INV_SQRT2));
            const double pdf = std::exp(-0.5 * xi * xi) * INV_SQRT2PI;
            dxp[i] = dYd[i] * (cdf + xi * pdf);
        }

        x->accumulate_grad(dx);
    };

    return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// ReLU
// ═════════════════════════════════════════════════════════════════════════════
//
// Forward:  y[i] = max(0, x[i])
//
// Backward: dX[i] = dY[i]  if x[i] > 0
//                 = 0       otherwise
//
// The backward uses a mask derived from x->data (the input, NOT the output).
// At x = 0 exactly, the sub-gradient is defined as 0 — consistent with
// PyTorch and all major frameworks.
//
NodePtr relu(const NodePtr& x)
{
    const size_t n   = x->data.numel();
    const double* xd  = x->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) {
        fwd[i] = xd[i] > 0.0 ? xd[i] : 0.0;
    }

    auto out = Node::make(
        Tensor(x->data.shape(), std::move(fwd)),
        x->requires_grad
    );
    out->add_child(x);

    // ── Backward ─────────────────────────────────────────────────────────────
    // dX[i] = dY[i] * (x[i] > 0 ? 1 : 0)
    // Reads x->data for the indicator mask.
    out->_backward = [x, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const size_t n   = self->grad.numel();
        const double* xd  = x->data.data_ptr();       // original input (for mask)
        const double* dYd = self->grad.data_ptr();

        Tensor dx(x->data.shape());
        double* dxp = dx.data_ptr();

        for (size_t i = 0; i < n; ++i) {
            dxp[i] = xd[i] > 0.0 ? dYd[i] : 0.0;
        }

        x->accumulate_grad(dx);
    };

    return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// Sigmoid
// ═════════════════════════════════════════════════════════════════════════════
//
// Forward:  y[i] = 1 / (1 + exp(−x[i]))
//
// Backward: dX[i] = dY[i] · y[i] · (1 − y[i])
//
// The backward reads self->data (the FORWARD OUTPUT — the sigmoid values).
// This avoids capturing x altogether.  The derivative σ'(x) = σ(x)·(1−σ(x))
// is expressed entirely in terms of σ(x), which is already stored on the
// output node.  This is the same trick used in sigmoid gates in LSTMs.
//
NodePtr sigmoid(const NodePtr& x)
{
    const size_t n   = x->data.numel();
    const double* xd  = x->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) {
        fwd[i] = 1.0 / (1.0 + std::exp(-xd[i]));
    }

    auto out = Node::make(
        Tensor(x->data.shape(), std::move(fwd)),
        x->requires_grad
    );
    out->add_child(x);

    // ── Backward ─────────────────────────────────────────────────────────────
    // dX[i] = dY[i] * out[i] * (1 - out[i])
    // Uses self->data — NO capture of x->data needed.
    out->_backward = [x, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const size_t n   = self->grad.numel();
        const double* sig = self->data.data_ptr();    // forward output (sigmoid values)
        const double* dYd = self->grad.data_ptr();

        Tensor dx(x->data.shape());
        double* dxp = dx.data_ptr();

        for (size_t i = 0; i < n; ++i) {
            dxp[i] = dYd[i] * sig[i] * (1.0 - sig[i]);
        }

        x->accumulate_grad(dx);
    };

    return out;
}

}  // namespace engine::nn
