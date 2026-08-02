/**
 * @file    nn/embedding.hpp
 * @brief   Token Embedding lookup table with scatter-add backward pass.
 *
 * Design (implementation_plan_v2.md, Step 2.2):
 *
 *  The Embedding layer maps discrete token IDs to dense vectors.
 *  Unlike all other layers, the forward input is NOT a NodePtr — token IDs
 *  are integers with no gradient; only the weight matrix is differentiable.
 *
 *  Forward (gather)
 *  ─────────────────
 *   ids : flat std::vector<size_t> of length B × T
 *   W   : weight matrix [vocab_size, d_model], initialised N(0,1)
 *
 *   out[b, t, :] = W[ids[b×T + t], :]      shape → [B, T, d_model]
 *
 *  Backward (scatter-add)
 *  ───────────────────────
 *   ∂W[tok, :] += Σ  ∂out[b, t, :]    for all (b,t) where ids[b×T+t] == tok
 *
 *   This is a scatter-add: the same token may appear multiple times in a
 *   batch, so contributions ACCUMULATE (+=) into the same weight row.
 *
 *  Initialisation
 *  ───────────────
 *   W ~ N(0, 1) — matches the common practice for Transformer embeddings.
 *   A typical follow-up is to scale by 1/sqrt(d_model) at the call site
 *   (done inside the Transformer class, Step 2.4).
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"      // Module base
#include "engine/node.hpp"    // NodePtr, make_parameter, Tensor

#include <cstddef>            // size_t
#include <vector>

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// Embedding
// ─────────────────────────────────────────────────────────────────────────────

class Embedding final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Create an embedding table of shape [vocab_size, d_model].
     *
     * @param vocab_size  Number of distinct tokens.
     * @param d_model     Embedding dimension.
     */
    Embedding(size_t vocab_size, size_t d_model);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Gather embedding vectors for a batch of token sequences.
     *
     * @param ids        Flat token IDs of length batch_size × seq_len.
     *                   All values must satisfy ids[i] < vocab_size.
     * @param batch_size Number of sequences B.
     * @param seq_len    Tokens per sequence T.
     * @return NodePtr of shape [B, T, d_model].
     *
     * @throws std::invalid_argument  if ids.size() != batch_size × seq_len.
     * @throws std::out_of_range      if any id >= vocab_size.
     */
    [[nodiscard]] NodePtr forward(const std::vector<size_t>& ids,
                                  size_t batch_size,
                                  size_t seq_len) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /** Returns {weight}. */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

    // ── Public parameter access ───────────────────────────────────────────────

    NodePtr weight;   ///< [vocab_size, d_model] — N(0,1) initialised

private:
    size_t vocab_size_;
    size_t d_model_;
};

}  // namespace engine::nn
