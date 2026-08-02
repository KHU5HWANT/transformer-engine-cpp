/**
 * @file    nn/linear.cpp
 * @brief   Implementation of the Linear module.
 *
 * Key implementation detail — bias_add (file-local differentiable op):
 * ─────────────────────────────────────────────────────────────────────
 *
 * Our engine::ops::add() requires identical shapes — it does NOT broadcast.
 * A Linear bias has shape [out_features] but the matmul output has shape
 * [..., out_features].  We therefore implement a file-local op, bias_add(),
 * that:
 *
 *   Forward : y[..., j] = x[..., j] + b[j]   (b is broadcast over all
 *                                               leading dimensions)
 *
 *   Backward:
 *     ∂L/∂x  = ∂L/∂y                          (pass-through, same shape)
 *     ∂L/∂b[j] = Σᵢ ∂L/∂y[i, j]              (sum over all leading dims)
 *
 * bias_add is an autograd-graph-aware op: it registers children and a
 * backward lambda using the same weak_ptr self-capture pattern as ops.cpp,
 * so it is completely memory-safe and participates in the backward sweep.
 *
 * Build flags:
 *   g++ -std=c++17 -O2 engine/tensor.cpp engine/node.cpp engine/ops.cpp
 *       engine/autograd.cpp nn/linear.cpp nn/layernorm.cpp
 */

#include "nn/linear.hpp"
#include "engine/ops.hpp"      // ops::transpose, ops::matmul
#include "engine/node.hpp"     // make_parameter, Node, Tensor

#include <algorithm>           // std::min
#include <cassert>
#include <cmath>               // std::sqrt
#include <random>              // mt19937, uniform_real_distribution
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// File-local helper op: bias_add
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// bias_add  :  out[..., j] = x[..., j] + b[j]
// ─────────────────────────────────────────────────────────────────────────────
//
// Broadcasts b (shape [N]) over the leading dims of x (shape [..., N]).
// Fully differentiable: registers a backward lambda that passes x's gradient
// straight through and sums b's gradient over the batch/time dimensions.
//
// Parameters:
//   x   —  NodePtr, shape [..., N]  (output of ops::matmul)
//   b   —  NodePtr, shape [N]       (the bias vector)
//
// Returns:
//   NodePtr with the same shape as x.
//
NodePtr bias_add(const NodePtr& x, const NodePtr& b)
{
    const size_t N     = b->data.numel();          // last dimension = out_features
    const size_t total = x->data.numel();
    const size_t rows  = total / N;                // product of leading dims

    if (total % N != 0) {
        throw std::invalid_argument(
            "bias_add: last dimension of x (" + std::to_string(total) +
            " elements total) is not divisible by bias size (" +
            std::to_string(N) + ").");
    }

    const double* xd = x->data.data_ptr();
    const double* bd = b->data.data_ptr();

    // ── Forward: broadcast-add ───────────────────────────────────────────────
    std::vector<double> fwd(total);
    for (size_t i = 0; i < rows; ++i) {
        const size_t row_off = i * N;
        for (size_t j = 0; j < N; ++j) {
            fwd[row_off + j] = xd[row_off + j] + bd[j];
        }
    }

    auto out = Node::make(
        Tensor(x->data.shape(), std::move(fwd)),
        x->requires_grad || b->requires_grad
    );

    out->add_child(x);
    out->add_child(b);

    // ── Backward ─────────────────────────────────────────────────────────────
    // ∂L/∂x[..., j] = ∂L/∂out[..., j]       (identity — add is linear)
    // ∂L/∂b[j]      = Σᵢ ∂L/∂out[i, j]      (reduce over leading dims)
    out->_backward = [x, b, N, rows,
                      w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const double* dout = self->grad.data_ptr();

        // ∂L/∂x — pass-through (same shape as out)
        x->accumulate_grad(self->grad);

        // ∂L/∂b — reduce (sum) over all leading dimensions
        Tensor db(b->data.shape());          // zero-initialised, shape [N]
        double* dbp = db.data_ptr();
        for (size_t i = 0; i < rows; ++i) {
            const size_t row_off = i * N;
            for (size_t j = 0; j < N; ++j) {
                dbp[j] += dout[row_off + j];
            }
        }
        b->accumulate_grad(db);
    };

    return out;
}

}  // anonymous namespace


// ═════════════════════════════════════════════════════════════════════════════
// Linear  —  public implementation
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Constructor — Xavier/Glorot uniform initialisation
// ─────────────────────────────────────────────────────────────────────────────
//
// Xavier uniform:   W ~ Uniform(-bound, +bound)
//                   bound = sqrt(6 / (fan_in + fan_out))
//
// This keeps the variance of activations constant across layers when the
// network uses symmetric activation functions (tanh, linear) and is a
// solid default for linear projections in Transformers.
//
// bias initialised to 0.0 — standard practice; the network learns the offset.
//
Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_ (in_features)
    , out_features_(out_features)
    , use_bias_    (use_bias)
{
    // ── RNG setup ─────────────────────────────────────────────────────────────
    std::mt19937 rng(std::random_device{}());
    const double bound = std::sqrt(
        6.0 / static_cast<double>(in_features + out_features));
    std::uniform_real_distribution<double> dist(-bound, bound);

    // ── Weight: [out_features, in_features] ──────────────────────────────────
    const size_t w_numel = out_features * in_features;
    std::vector<double> w_data(w_numel);
    for (double& v : w_data) v = dist(rng);

    weight = make_parameter(
        Tensor({out_features, in_features}, std::move(w_data)));

    // ── Bias: [out_features], zero-init ──────────────────────────────────────
    if (use_bias_) {
        bias = make_parameter(
            Tensor({out_features}, std::vector<double>(out_features, 0.0)));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// forward  :  out = x @ Wᵀ + b
// ─────────────────────────────────────────────────────────────────────────────
//
// Step-by-step, each creating a new Node in the autograd DAG:
//
//   Wt  = ops::transpose(weight)    [in_features, out_features]
//         Backward: dW = transpose(dWt)  — handled by ops::transpose
//
//   xWt = ops::matmul(x, Wt)       [..., out_features]
//         Backward: dx = dxWt @ Wt^T,  dWt = x^T @ dxWt
//                   — handled by ops::matmul (mm_nt / mm_tn kernels)
//
//   out = bias_add(xWt, bias)      [..., out_features]
//         Backward: d(xWt) = dout (pass-through),
//                   d(bias)[j] = Σᵢ dout[i,j]  — handled by bias_add
//
NodePtr Linear::forward(const NodePtr& x) const
{
    // Shape validation — last dim of x must equal in_features_
    if (x->data.shape().empty() ||
        x->data.shape().back() != in_features_) {
        throw std::invalid_argument(
            "Linear::forward: input last dimension " +
            std::to_string(x->data.shape().empty()
                           ? 0 : x->data.shape().back()) +
            " != in_features " + std::to_string(in_features_));
    }

    // 1. Wᵀ  — creates a transpose node in the DAG
    auto Wt = ops::transpose(weight);          // [in_features, out_features]

    // 2. xWᵀ — creates a matmul node (tiled+OpenMP, handles 2-D and 3-D)
    auto xWt = ops::matmul(x, Wt);            // [..., out_features]

    // 3. + b — creates a broadcast-add node with correct bias gradient
    if (use_bias_) {
        return bias_add(xWt, bias);
    }
    return xWt;
}

// ─────────────────────────────────────────────────────────────────────────────
// parameters
// ─────────────────────────────────────────────────────────────────────────────

std::vector<NodePtr> Linear::parameters() const
{
    if (use_bias_ && bias) {
        return {weight, bias};
    }
    return {weight};
}

}  // namespace engine::nn
