/**
 * @file    loss/cross_entropy.hpp
 * @brief   Numerically stable Cross-Entropy loss for next-token prediction.
 *
 * Design (implementation_plan_v2.md, Step 4.1):
 *
 *  Inputs
 *  ──────
 *   logits  : NodePtr, shape [B, T, V]   — raw scores from lm_head (NOT softmax'd)
 *   targets : std::vector<size_t>, size B×T — ground-truth token IDs from DataLoader.Y
 *
 *  Forward pass (log-sum-exp trick)
 *  ──────────────────────────────────
 *   For each of the B×T positions:
 *     1. max_v = max(logits[bt, :])                  ← numerical stability anchor
 *     2. shifted[v] = logits[bt, v] - max_v
 *     3. log_Z = log(Σ_v exp(shifted[v]))             ← partition function
 *     4. log_p[target] = shifted[target] - log_Z     ← log-softmax of correct class
 *     5. nll[bt] = -log_p[target]
 *   loss = mean(nll)  =  (1 / B×T) × Σ_bt nll[bt]
 *
 *  Why max-subtraction?
 *  ─────────────────────
 *  Without it, exp(logit[v]) overflows to +inf for logit > ~710 (double).
 *  Subtracting max_v makes every exponent ≤ 0, so exp ∈ (0, 1].
 *  The identity exp(x - max)/Σ_v exp(x_v - max) = exp(x)/Σ_v exp(x_v)
 *  means the result is mathematically identical.
 *
 *  Backward pass (analytically fused)
 *  ────────────────────────────────────
 *  Let p[bt, v] = softmax(logits[bt, :]) (saved during forward).
 *  The gradient of NLL w.r.t. logits is the well-known softmax Jacobian
 *  reduction:
 *
 *    ∂loss/∂logits[bt, v] = (p[bt, v] − 1{v == targets[bt]}) / (B×T)
 *
 *  This is accumulated via Node::accumulate_grad(), scaled by the upstream
 *  gradient from autograd::backward().
 *
 *  Returns
 *  ────────
 *  NodePtr of shape {1} — a scalar loss with _backward registered.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "engine/node.hpp"

#include <cstddef>
#include <vector>

namespace loss {

/**
 * @brief Compute the mean cross-entropy loss over a batch of logit sequences.
 *
 * @param logits   NodePtr, shape [B, T, V].  Must be 3-D.
 *                 All three dimensions must be ≥ 1.
 * @param targets  Flat token-ID array of length B×T.
 *                 Each value must satisfy targets[i] < V.
 *
 * @return  Scalar NodePtr (shape {1}) with value = mean NLL over B×T tokens.
 *          The _backward lambda is registered and produces correct gradients
 *          for logits when autograd::backward() is called.
 *
 * @throws std::invalid_argument if logits is not 3-D.
 * @throws std::invalid_argument if targets.size() != B×T.
 * @throws std::out_of_range     if any target ID >= V.
 */
[[nodiscard]] engine::NodePtr cross_entropy(const engine::NodePtr&      logits,
                                             const std::vector<size_t>&  targets);

}  // namespace loss
