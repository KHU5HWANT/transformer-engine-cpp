/**
 * @file    nn/embedding.cpp
 * @brief   Token Embedding implementation — gather forward, scatter-add backward.
 *
 * Key algorithmic detail — Scatter-Add in the backward pass
 * ──────────────────────────────────────────────────────────
 *
 * The forward pass is a GATHER: copy weight rows indexed by the token IDs.
 * The backward pass is a SCATTER-ADD: for each (b,t) pair, add the upstream
 * gradient slice dout[b,t,:] into dW[ids[b*T+t],:].
 *
 * Why scatter-add and not scatter-assign?
 *   A single token may appear multiple times in the batch (e.g. "the" at
 *   positions 0, 5, 12 of the same sequence, or across multiple sequences).
 *   Each occurrence contributes an independent gradient to the same weight
 *   row.  These must be ACCUMULATED (+=) — overwriting would discard all but
 *   the last gradient, causing incorrect training.
 *
 * Thread safety of the scatter-add loop
 *   The inner scatter loop is SEQUENTIAL by design.  OpenMP parallelism on
 *   the outer (b,t) loop would cause data races on shared weight rows when
 *   the same token ID appears in two threads simultaneously.  For correctness
 *   we do NOT parallelize this loop.  (Alternative: per-thread dW buffers
 *   with a reduce step — reserved for future optimisation.)
 *
 * Captured values in the backward lambda
 *   ids  — captured by value (O(B×T) integers, always small).
 *   BT   — B×T, avoids recomputing batch_size*seq_len inside lambda.
 *   D    — d_model, row stride for both output and weight.
 *   weight — shared_ptr (keeps weight node alive; also provides data_ptr).
 */

#include "nn/embedding.hpp"

#include <algorithm>           // std::copy
#include <random>              // mt19937, normal_distribution
#include <stdexcept>
#include <string>

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// Constructor — N(0, 1) initialisation
// ═════════════════════════════════════════════════════════════════════════════
//
// Standard normal initialisation is common for Transformer embeddings.
// The Transformer class (Step 2.4) will optionally scale by 1/sqrt(d_model)
// when adding the positional encoding.
//
Embedding::Embedding(size_t vocab_size, size_t d_model)
    : vocab_size_(vocab_size)
    , d_model_   (d_model)
{
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);

    const size_t w_numel = vocab_size * d_model;
    std::vector<double> w_data(w_numel);
    for (double& v : w_data) v = dist(rng);

    weight = make_parameter(
        Tensor({vocab_size, d_model}, std::move(w_data)));
}

// ═════════════════════════════════════════════════════════════════════════════
// forward  —  gather rows from the weight table
// ═════════════════════════════════════════════════════════════════════════════

NodePtr Embedding::forward(const std::vector<size_t>& ids,
                            size_t batch_size,
                            size_t seq_len) const
{
    const size_t BT = batch_size * seq_len;

    // ── Validate ──────────────────────────────────────────────────────────────
    if (ids.size() != BT) {
        throw std::invalid_argument(
            "Embedding::forward: ids.size() (" +
            std::to_string(ids.size()) + ") != batch_size × seq_len (" +
            std::to_string(batch_size) + " × " +
            std::to_string(seq_len) + " = " + std::to_string(BT) + ").");
    }

    const double* wp = weight->data.data_ptr();

    // ── Forward: gather (copy weight rows) ────────────────────────────────────
    // out[b, t, d] = W[ids[b*T + t], d]
    std::vector<double> fwd(BT * d_model_);

    for (size_t bt = 0; bt < BT; ++bt) {
        const size_t tok = ids[bt];
        if (tok >= vocab_size_) {
            throw std::out_of_range(
                "Embedding::forward: token id " + std::to_string(tok) +
                " at position " + std::to_string(bt) +
                " is out of range [0, " + std::to_string(vocab_size_) + ").");
        }
        const double* src = wp + tok * d_model_;
        double*       dst = fwd.data() + bt * d_model_;
        std::copy(src, src + d_model_, dst);
    }

    // ── Create output node ────────────────────────────────────────────────────
    // Only weight is a DAG child — token IDs have no gradient.
    auto out = Node::make(
        Tensor({batch_size, seq_len, d_model_}, std::move(fwd)),
        weight->requires_grad
    );

    out->add_child(weight);    // DAG edge: out depends on weight

    // ── Backward: scatter-add ─────────────────────────────────────────────────
    //
    // For each (b,t), the gradient flowing back is:
    //   dW[ids[bt], :] += dout[bt, :]
    //
    // ids is captured by VALUE (cheap: O(B×T) integers).
    // weight is captured by shared_ptr VALUE (keeps the node alive).
    //
    out->_backward = [weight = weight, ids, BT, D = d_model_,
                      w_out = std::weak_ptr<Node>(out)]()
    {
        auto self = w_out.lock();
        if (!self) return;

        const double* dout = self->grad.data_ptr();

        // Accumulate: zero-initialised gradient tensor for W  [vocab, D]
        Tensor dw(weight->data.shape());
        double* dwp = dw.data_ptr();

        // Sequential loop — DO NOT parallelise with OpenMP:
        // the same token ID may appear multiple times (data race on dwp rows).
        for (size_t bt = 0; bt < BT; ++bt) {
            const size_t  tok   = ids[bt];
            const double* drow  = dout + bt * D;   // gradient slice for this (b,t)
            double*       wrow  = dwp  + tok * D;  // target row in dW

            for (size_t d = 0; d < D; ++d) {
                wrow[d] += drow[d];   // scatter-add
            }
        }

        weight->accumulate_grad(dw);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// parameters
// ─────────────────────────────────────────────────────────────────────────────

std::vector<NodePtr> Embedding::parameters() const
{
    return {weight};
}

}  // namespace engine::nn
