/**
 * @file    nn/softmax.cpp
 * @brief   Numerically stable Softmax — fused forward + exact Jacobian-VJP backward.
 *
 * See nn/softmax.hpp for the full mathematical derivation of the backward formula.
 *
 * Forward pass (per row i):
 *   m[i]    = max_j x[i, j]                     (numeric stability anchor)
 *   e[i, j] = exp(x[i, j] − m[i])               (all exponents ≤ 0 → no overflow)
 *   s[i]    = Σ_j e[i, j]                        (partition function)
 *   y[i, j] = e[i, j] / s[i]                     (normalised probabilities)
 *
 * Backward pass (per row i):
 *   dot[i]   = Σ_j dY[i,j] · Y[i,j]             (expected value of dY under Y)
 *   dX[i, j] = Y[i,j] · (dY[i,j] − dot[i])
 *
 * Saved activation trick:
 *   The backward reads self->data.data_ptr() — the forward output Y — directly
 *   from the output Node.  No separate buffer needs to be saved or captured.
 *   This is safe because `self = wout.lock()` guarantees the Node is alive,
 *   and `self->data` is immutable after Node construction.
 *
 * The `M × N` flattening convention:
 *   Input x has shape [*, N] where N = last dimension.
 *   M = total_elements / N = product of all leading dimensions.
 *   This correctly handles 2-D [B, N] and 3-D [B, T, N] inputs without any
 *   shape-specific branching — the same pattern used in layernorm_op.
 */

#include "nn/softmax.hpp"

#include <algorithm>           // std::max_element
#include <cmath>               // std::exp
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// softmax
// ═════════════════════════════════════════════════════════════════════════════

NodePtr softmax(const NodePtr& x)
{
    // ── Validate ──────────────────────────────────────────────────────────────
    if (x->data.shape().empty()) {
        throw std::invalid_argument(
            "softmax: input is a 0-D scalar — softmax requires at least 1 dimension.");
    }

    const size_t N     = x->data.shape().back();   // last dimension
    const size_t total = x->data.numel();
    const size_t M     = total / N;                // number of independent rows

    const double* xd = x->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(total);

    for (size_t i = 0; i < M; ++i) {
        const double* x_row = xd  + i * N;
        double*       y_row = fwd.data() + i * N;

        // ── Step 1: subtract per-row max (prevents exp() overflow) ────────────
        double max_val = *std::max_element(x_row, x_row + N);

        // ── Step 2: exponentiate shifted values ───────────────────────────────
        // All arguments to exp() are ≤ 0, so output is in (0, 1].
        double sum = 0.0;
        for (size_t j = 0; j < N; ++j) {
            y_row[j] = std::exp(x_row[j] - max_val);
            sum += y_row[j];
        }

        // ── Step 3: normalise ─────────────────────────────────────────────────
        // sum > 0 is guaranteed (at least one exp result equals 1.0).
        for (size_t j = 0; j < N; ++j) {
            y_row[j] /= sum;
        }
    }

    // ── Create output node ────────────────────────────────────────────────────
    auto out = Node::make(
        Tensor(x->data.shape(), std::move(fwd)),
        x->requires_grad
    );
    out->add_child(x);

    // ── Backward — Jacobian-vector product ────────────────────────────────────
    //
    // dX[i, j] = Y[i, j] · (dY[i, j] − dot[i])
    //   where dot[i] = Σ_k dY[i,k] · Y[i,k]
    //
    // The forward output Y is read from self->data — no saved buffer needed.
    // This is safe: self is alive (guaranteed by wout.lock()), and self->data
    // is never mutated after construction.
    //
    out->_backward = [x, M, N,
                      w_out = std::weak_ptr<Node>(out)]()
    {
        auto self = w_out.lock();
        if (!self) return;

        const double* dY = self->grad.data_ptr();   // upstream gradient
        const double* Y  = self->data.data_ptr();   // softmax output (saved automatically)

        Tensor dx(x->data.shape());
        double* dxp = dx.data_ptr();

        for (size_t i = 0; i < M; ++i) {
            const double* dY_row = dY  + i * N;
            const double* Y_row  = Y   + i * N;
            double*       dx_row = dxp + i * N;

            // ── Per-row dot product: dot[i] = Σ_k dY[i,k] · Y[i,k] ──────────
            // This is the expected value of dY under the predicted distribution Y.
            double dot = 0.0;
            for (size_t k = 0; k < N; ++k) {
                dot += dY_row[k] * Y_row[k];
            }

            // ── dX[i,j] = Y[i,j] · (dY[i,j] − dot[i]) ──────────────────────
            for (size_t j = 0; j < N; ++j) {
                dx_row[j] = Y_row[j] * (dY_row[j] - dot);
            }
        }

        x->accumulate_grad(dx);
    };

    return out;
}

}  // namespace engine::nn
