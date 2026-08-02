/**
 * @file    nn/transformer.cpp
 * @brief   Full Transformer Language Model — constructor, forward, parameters().
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Constructor design — blocks_.reserve() before emplace_back
 * ════════════════════════════════════════════════════════════════════════════
 *
 * std::vector<TransformerBlock> requires TransformerBlock to be
 * move-constructible (for reallocation).  Module explicitly defaults its move
 * constructor (see nn/module.hpp), so all derived classes inherit movability.
 *
 * We still call blocks_.reserve(n_layers) before emplacing for two reasons:
 *   1. Avoids n_layers-1 intermediate reallocations and moves.
 *   2. Makes the constructor O(n_layers) instead of O(n_layers · log n_layers).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Forward — positional IDs generation
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Positional embeddings are looked up with absolute position IDs:
 *   pos_ids[b · T + t] = t   (for b ∈ [0,B), t ∈ [0,T))
 *
 * This generates {0,1,...,T-1, 0,1,...,T-1, ...} repeated B times.
 * Each sequence sees the same positional encoding, which is the standard
 * GPT-2 approach (absolute learned positions, not sinusoidal).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Forward — embedding sum via ops::add
 * ════════════════════════════════════════════════════════════════════════════
 *
 * ops::add requires both operands to have the same shape.  After:
 *   tok_embed: [B, T, d_model]
 *   pos_embed: [B, T, d_model]
 * Their element-wise sum [B, T, d_model] is valid and fully differentiable.
 *
 * Gradients flow back separately to tok_emb.weight and pos_emb.weight via
 * the scatter-add ops registered by Embedding::forward (Step 2.2).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * lm_head — no bias, no softmax
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The output of lm_head is RAW LOGITS [B, T, vocab_size].
 * Softmax is NOT applied here.  The cross-entropy loss (Step 4.1) will call
 * softmax internally (or use numerically stable log-softmax + NLL).
 * This matches PyTorch's nn.CrossEntropyLoss(logits, targets) convention.
 */

#include "nn/transformer.hpp"
#include "engine/ops.hpp"     // ops::add  (embedding sum)

#include <stdexcept>
#include <string>

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════

Transformer::Transformer(size_t vocab_size,
                         size_t context_len,
                         size_t d_model,
                         size_t n_heads,
                         size_t n_layers)
    : vocab_size_ (vocab_size)
    , context_len_(context_len)
    , d_model_    (d_model)
    , n_heads_    (n_heads)
    , n_layers_   (n_layers)
    //
    // Sub-modules initialised in declaration order (see transformer.hpp).
    //
    , tok_emb (vocab_size,   d_model)
    , pos_emb (context_len,  d_model)
    // blocks: constructed below in the body via emplace_back
    , ln_f    (d_model)
    //
    // lm_head has NO bias: weight tying with tok_emb is easier without a bias
    // term, and GPT-2 style models consistently omit the lm_head bias.
    //
    , lm_head (d_model, vocab_size, /*use_bias=*/false)
{
    // Pre-allocate to avoid O(n_layers · log n_layers) moves on resize.
    blocks.reserve(n_layers);

    for (size_t i = 0; i < n_layers; ++i) {
        // emplace_back constructs TransformerBlock in-place — no copy or move
        // of the partially-constructed vector is triggered because we reserved.
        blocks.emplace_back(d_model, n_heads);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// forward
// ═════════════════════════════════════════════════════════════════════════════

NodePtr Transformer::forward(const std::vector<size_t>& ids,
                              size_t batch_size,
                              size_t seq_len) const
{
    // ── Input validation ──────────────────────────────────────────────────────
    if (seq_len > context_len_) {
        throw std::invalid_argument(
            "Transformer::forward: seq_len (" + std::to_string(seq_len) +
            ") exceeds context_len (" + std::to_string(context_len_) + "). "
            "The positional embedding table only has " +
            std::to_string(context_len_) + " rows.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 1–2: Token and positional embeddings
    // ─────────────────────────────────────────────────────────────────────────

    // Token embeddings: gather rows from tok_emb.weight for each input token.
    // Shape: [B, T, d_model].  Scatter-add backward registered automatically.
    auto tok_embed = tok_emb.forward(ids, batch_size, seq_len);

    // Positional embeddings: position t uses row t of pos_emb.weight.
    // pos_ids[b * T + t] = t  (absolute position, same for every batch item)
    std::vector<size_t> pos_ids;
    pos_ids.reserve(batch_size * seq_len);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            pos_ids.push_back(t);
        }
    }
    auto pos_embed = pos_emb.forward(pos_ids, batch_size, seq_len);

    // ─────────────────────────────────────────────────────────────────────────
    // Step 3: Embedding sum — token + position
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Both tensors are [B, T, d_model]; ops::add is elementwise and exact-shape.
    // The combined embedding is the input to the first TransformerBlock.
    //
    auto x = ops::add(tok_embed, pos_embed);   // [B, T, d_model]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 4: Stacked Transformer blocks (Pre-LN)
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Each block: Pre-LN → Attention → Residual → Pre-LN → FFN → Residual.
    // The output of each block is fed directly into the next.
    //
    for (const TransformerBlock& block : blocks) {
        x = block.forward(x);                  // [B, T, d_model]  (shape invariant)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 5: Final layer norm
    // ─────────────────────────────────────────────────────────────────────────
    x = ln_f.forward(x);                       // [B, T, d_model]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 6: Language model head — project to vocabulary size
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Linear supports 3-D input [B, T, d_model] and produces [B, T, vocab_size].
    // Output values are RAW LOGITS — do NOT softmax here.
    //
    return lm_head.forward(x);                 // [B, T, vocab_size]
}

// ═════════════════════════════════════════════════════════════════════════════
// parameters
// ═════════════════════════════════════════════════════════════════════════════

std::vector<NodePtr> Transformer::parameters() const
{
    // Traversal order: tok_emb → pos_emb → blocks (in layer order) → ln_f → lm_head
    //
    // Total: 1 (tok) + 1 (pos) + n_layers × 16 (blocks) + 2 (ln_f) + 1 (lm_head)
    //      = 4 + 16 × n_layers  NodePtrs
    //
    auto p = tok_emb.parameters();   // 1:  W_token

    auto pos_p = pos_emb.parameters();
    p.insert(p.end(), pos_p.begin(), pos_p.end());  // +1: W_pos

    for (const TransformerBlock& block : blocks) {
        auto bp = block.parameters();               // +16 per block
        p.insert(p.end(), bp.begin(), bp.end());
    }

    auto ln_p = ln_f.parameters();
    p.insert(p.end(), ln_p.begin(), ln_p.end());    // +2: γ_f, β_f

    auto head_p = lm_head.parameters();
    p.insert(p.end(), head_p.begin(), head_p.end());  // +1: W_head

    return p;
}

}  // namespace engine::nn
