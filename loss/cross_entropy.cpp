/**
 * @file    loss/cross_entropy.cpp
 * @brief   Fused forward+backward Cross-Entropy loss implementation.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Why fuse forward and backward into a single custom op?
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Cross-entropy is the composition: softmax → log → negate → mean.
 * If we implement these as four separate autograd nodes, the backward pass
 * must traverse all four lambdas and materialise four intermediate gradient
 * tensors, including the full Jacobian of softmax (V×V per position).
 *
 * The analytically collapsed gradient bypasses all of that:
 *   ∂CE/∂logits[bt, v] = (softmax(logits)[bt, v] − 1{v==target[bt]}) / (B×T)
 *
 * This is a O(B×T×V) computation vs O(B×T×V²) for the naive Jacobian.
 * The softmax probabilities (shape B×T×V) are the only quantity that must be
 * saved from the forward pass — no other intermediate state is needed.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Log-sum-exp trick — derivation
 * ════════════════════════════════════════════════════════════════════════════
 *
 * log softmax(x)[c] = x[c] - log(Σ_v exp(x[v]))
 *
 * Let m = max_v x[v]:
 *   log(Σ_v exp(x[v])) = log(Σ_v exp(x[v] − m + m))
 *                       = log(exp(m) · Σ_v exp(x[v] − m))
 *                       = m + log(Σ_v exp(x[v] − m))   ← safe: x[v]−m ≤ 0
 *
 * Therefore:
 *   log softmax(x)[c] = (x[c] − m) − log(Σ_v exp(x[v] − m))
 *                     = shifted[c] − log_Z
 *
 * No overflow possible: all exponents are ≤ 0 → exp(·) ∈ (0, 1].
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Backward gradient derivation
 * ════════════════════════════════════════════════════════════════════════════
 *
 * For a single position bt with correct class c = targets[bt]:
 *
 *   L_bt = -log p_c  where p_v = softmax(logits)[bt, v]
 *
 * Chain rule through softmax:
 *   ∂L_bt/∂logits[bt, v] = p_v - 1{v==c}
 *
 * After averaging over B×T positions and applying the upstream gradient g:
 *   ∂L/∂logits[bt, v] = g · (p_v - 1{v==c}) / (B×T)
 *
 * where g = out->grad.data()[0] (scalar upstream gradient from backward()).
 * In a standalone training loop g = 1.0 (loss is the root node).
 */

#include "loss/cross_entropy.hpp"
#include "engine/node.hpp"
#include "engine/tensor.hpp"

#include <algorithm>      // std::max_element
#include <cassert>
#include <cmath>          // std::exp, std::log, std::sqrt
#include <memory>         // std::weak_ptr
#include <stdexcept>
#include <string>

namespace loss {

// ═════════════════════════════════════════════════════════════════════════════
// cross_entropy
// ═════════════════════════════════════════════════════════════════════════════

engine::NodePtr cross_entropy(const engine::NodePtr&     logits,
                               const std::vector<size_t>& targets)
{
    using engine::Tensor;
    using engine::NodePtr;

    // ── Input validation ──────────────────────────────────────────────────────
    if (logits->data.ndim() != 3) {
        throw std::invalid_argument(
            "cross_entropy: logits must be 3-D [B, T, V], got ndim=" +
            std::to_string(logits->data.ndim()) + ".");
    }

    const auto& shape = logits->data.shape();
    const size_t B  = shape[0];
    const size_t T  = shape[1];
    const size_t V  = shape[2];
    const size_t BT = B * T;

    if (targets.size() != BT) {
        throw std::invalid_argument(
            "cross_entropy: targets.size()=" + std::to_string(targets.size()) +
            " does not match B×T=" + std::to_string(BT) + ".");
    }

    for (size_t i = 0; i < BT; ++i) {
        if (targets[i] >= V) {
            throw std::out_of_range(
                "cross_entropy: targets[" + std::to_string(i) + "]=" +
                std::to_string(targets[i]) + " >= V=" + std::to_string(V) + ".");
        }
    }

    // ── Forward pass ──────────────────────────────────────────────────────────
    //
    // We interleave loss computation and probability computation into a single
    // pass over the logit matrix to minimise memory bandwidth.
    //
    // Layout: logits->data is [B, T, V] in row-major order.
    //   logit_data[bt * V + v]  where bt = b * T + t.
    //
    const double* logit_data = logits->data.data_ptr();

    // Save softmax probabilities: shape [B*T, V] flat, allocated on heap.
    // This is the ONLY state saved from the forward pass for the backward.
    std::vector<double> probs(BT * V);

    double total_nll = 0.0;

    for (size_t bt = 0; bt < BT; ++bt) {
        const double* row = logit_data + bt * V;

        // Step 1: numerical stability anchor — find row maximum
        const double max_val = *std::max_element(row, row + V);

        // Step 2: compute exp(shifted) and partition function Z = Σ_v exp(x_v - max)
        double sum_exp = 0.0;
        double* p_row  = probs.data() + bt * V;

        for (size_t v = 0; v < V; ++v) {
            p_row[v]  = std::exp(row[v] - max_val);
            sum_exp  += p_row[v];
        }

        // Step 3: normalise → softmax probabilities (saved for backward)
        for (size_t v = 0; v < V; ++v) {
            p_row[v] /= sum_exp;
        }

        // Step 4: log-softmax for the correct class only
        //   log_p[target] = (row[target] - max_val) - log(sum_exp)
        // This is mathematically equivalent to log(p_row[target]) but avoids
        // the additional log(prob) call on the already-normalised value.
        const size_t c = targets[bt];
        total_nll -= (row[c] - max_val) - std::log(sum_exp);
    }

    // Average over all B×T token positions
    const double loss_val    = total_nll / static_cast<double>(BT);

    // ── Build output node ─────────────────────────────────────────────────────
    //
    // Shape {1} — scalar loss.  requires_grad = true so autograd::backward()
    // can set grad.data()[0] = 1.0 and call _backward.
    //
    Tensor loss_tensor({1});
    loss_tensor.data()[0] = loss_val;

    auto out = engine::Node::make(std::move(loss_tensor), /*requires_grad=*/true);
    out->add_child(logits);  // registers logits as a DAG dependency for topo-sort

    // ── Backward closure ──────────────────────────────────────────────────────
    //
    // Captured by the lambda (all copies or moves):
    //   w_logits  — weak_ptr (no cycle) to the logits node
    //   w_out     — weak_ptr to this output node (to read out->grad)
    //   probs     — softmax probabilities saved from forward (moved in)
    //   targets   — target token IDs (copied by value)
    //   BT, V     — loop bounds
    //
    std::weak_ptr<engine::Node> w_logits = logits;
    std::weak_ptr<engine::Node> w_out    = out;

    out->_backward = [w_logits,
                      w_out,
                      probs   = std::move(probs),    // moved — avoids copy
                      targets,                        // copied by value
                      BT, V]()
    {
        auto logits_node = w_logits.lock();
        auto out_node    = w_out.lock();
        if (!logits_node || !out_node) return;

        // Upstream gradient (scalar).  In a standalone training loop this is 1.0.
        const double g     = out_node->grad.data()[0];
        const double scale = g / static_cast<double>(BT);

        // ∂loss/∂logits[bt, v] = (p[bt,v] - 1{v==targets[bt]}) × g / (B×T)
        Tensor d_logits(logits_node->data.shape());  // zero-initialised
        double* d_data = d_logits.data_ptr();

        for (size_t bt = 0; bt < BT; ++bt) {
            const double* p_row  = probs.data() + bt * V;
            double*       dL_row = d_data       + bt * V;
            const size_t  c      = targets[bt];

            for (size_t v = 0; v < V; ++v) {
                dL_row[v] = p_row[v] * scale;
            }
            // Subtract 1/BT for the correct class (the "one-hot" subtraction)
            dL_row[c] -= scale;
        }

        // accumulate_grad uses +=, correctly handling fan-out in the DAG
        logits_node->accumulate_grad(d_logits);
    };

    return out;
}

}  // namespace loss
