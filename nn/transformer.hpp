/**
 * @file    nn/transformer.hpp
 * @brief   The full Transformer Language Model.
 *
 * Design (implementation_plan_v2.md, Step 2.4):
 *
 *  Architecture (GPT-style, decoder-only)
 *  ──────────────────────────────────────
 *   tok_emb  : Embedding(vocab_size,  d_model)  — token lookup
 *   pos_emb  : Embedding(context_len, d_model)  — absolute position lookup
 *   blocks   : N × TransformerBlock(d_model, n_heads)  — stacked transformer layers
 *   ln_f     : LayerNorm(d_model)               — final normalisation
 *   lm_head  : Linear(d_model, vocab_size, bias=false) — logit projection
 *
 *  Forward pass
 *  ─────────────
 *   1. tok_emb = tok_emb_.forward(ids,       B, T)   → [B, T, d_model]
 *   2. pos_emb = pos_emb_.forward(pos_ids,   B, T)   → [B, T, d_model]
 *      where pos_ids = {0,1,...,T-1} repeated B times
 *   3. x = ops::add(tok_emb, pos_emb)               → [B, T, d_model]
 *   4. for each block: x = block.forward(x)          → [B, T, d_model]
 *   5. x = ln_f_.forward(x)                          → [B, T, d_model]
 *   6. logits = lm_head_.forward(x)                  → [B, T, vocab_size]
 *
 *  Parameter count (for reference)
 *  ─────────────────────────────────
 *   tok_emb  : 1           (vocab_size × d_model weight matrix)
 *   pos_emb  : 1           (context_len × d_model weight matrix)
 *   N blocks : N × 16      (16 NodePtrs per TransformerBlock — Step 2.4)
 *   ln_f     : 2           (γ, β)
 *   lm_head  : 1           (d_model × vocab_size, NO bias)
 *              ─────────────────────────────────────────────────────
 *   Total    : 4 + 16N   NodePtrs
 *
 *  Weight tying note
 *  ──────────────────
 *  GPT-2 ties lm_head.weight with tok_emb.weight (transposed).  This halves
 *  the embedding parameter count and often improves perplexity.  Weight tying
 *  is NOT implemented here (would require assigning lm_head.weight =
 *  tok_emb.weight after construction), but the architecture supports it trivially
 *  since both weight members are public NodePtrs.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "nn/module.hpp"              // Module base
#include "nn/embedding.hpp"           // Embedding
#include "nn/transformer_block.hpp"   // TransformerBlock
#include "nn/layernorm.hpp"           // LayerNorm
#include "nn/linear.hpp"              // Linear (lm_head)
#include "engine/node.hpp"            // NodePtr

#include <cstddef>                    // size_t
#include <vector>                     // std::vector

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// Transformer
// ─────────────────────────────────────────────────────────────────────────────

class Transformer final : public Module {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct the full Transformer language model.
     *
     * @param vocab_size   Number of distinct token types  (|V|).
     * @param context_len  Maximum sequence length the model can process.
     *                     Positional embedding table has exactly this many rows.
     * @param d_model      Embedding / model dimension.
     *                     Must be divisible by n_heads.
     * @param n_heads      Number of attention heads per block.
     * @param n_layers     Number of stacked TransformerBlock layers.
     *
     * @throws std::invalid_argument (from CausalSelfAttention)
     *         if d_model % n_heads != 0.
     */
    Transformer(size_t vocab_size,
                size_t context_len,
                size_t d_model,
                size_t n_heads,
                size_t n_layers);

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @brief Compute logits for a batch of token sequences.
     *
     * @param ids        Flat token-ID array of length batch_size × seq_len.
     *                   All values must satisfy ids[i] < vocab_size.
     * @param batch_size Number of sequences B.
     * @param seq_len    Tokens per sequence T.  Must satisfy T ≤ context_len.
     * @return           NodePtr of shape [B, T, vocab_size] — raw logits
     *                   (NOT softmax-normalised; softmax is applied inside the
     *                   cross-entropy loss at Step 4.1).
     *
     * @throws std::invalid_argument if seq_len > context_len.
     * @throws std::out_of_range     (from Embedding) if any token id ≥ vocab_size.
     */
    [[nodiscard]] NodePtr forward(const std::vector<size_t>& ids,
                                  size_t batch_size,
                                  size_t seq_len) const;

    // ── Module interface ──────────────────────────────────────────────────────

    /**
     * @brief Recursively collect ALL trainable parameters in the model.
     *
     * Traversal order:
     *   tok_emb → pos_emb → block[0] → block[1] → ... → ln_f → lm_head
     *
     * @return Flat vector of (4 + 16×n_layers) NodePtrs.
     */
    [[nodiscard]] std::vector<NodePtr> parameters() const override;

    // ── Public sub-module access ──────────────────────────────────────────────

    Embedding                  tok_emb;    ///< Token embedding table [vocab_size, d_model]
    Embedding                  pos_emb;    ///< Position embedding table [context_len, d_model]
    std::vector<TransformerBlock> blocks;  ///< Stacked transformer layers
    LayerNorm                  ln_f;       ///< Final layer norm
    Linear                     lm_head;    ///< Logit head [d_model → vocab_size], no bias

private:
    size_t vocab_size_;
    size_t context_len_;
    size_t d_model_;
    size_t n_heads_;
    size_t n_layers_;
};

}  // namespace engine::nn
