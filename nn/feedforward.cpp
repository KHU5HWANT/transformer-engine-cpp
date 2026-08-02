/**
 * @file    nn/feedforward.cpp
 * @brief   FeedForward implementation — fc1 → GELU → fc2.
 *
 * Autograd graph built by forward():
 *
 *   x ──► fc1 ──► h1 ──► gelu ──► h2 ──► fc2 ──► out
 *                              [saved x̂]
 *
 * Every arrow is a fully-differentiable op with its own backward lambda:
 *   • fc1 (matmul + bias_add):  registered by Linear::forward
 *   • gelu:                     registered by engine::nn::gelu (Step 2.2)
 *   • fc2 (matmul + bias_add):  registered by Linear::forward
 *
 * No custom ops are required here — we rely entirely on the composability
 * of our existing autograd primitives.
 *
 * Initialiser list ordering
 * ──────────────────────────
 * d_model_ and d_ff_ are declared before fc1 and fc2 in the header.
 * Therefore d_model_ and d_ff_ are fully initialised before the Linear
 * constructors run, allowing us to use them directly.
 */

#include "nn/feedforward.hpp"
#include "nn/activation.hpp"    // engine::nn::gelu

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════

FeedForward::FeedForward(size_t d_model, size_t d_ff)
    : d_model_(d_model)
    , d_ff_   (d_ff == 0 ? 4 * d_model : d_ff)
    //
    // fc1 and fc2 are declared before d_model_ and d_ff_, so they are initialized FIRST!
    // Therefore we MUST use the constructor arguments here, not the uninitialized member variables.
    //
    , fc1(d_model, d_ff == 0 ? 4 * d_model : d_ff,   /*use_bias=*/true)
    , fc2(d_ff == 0 ? 4 * d_model : d_ff,   d_model, /*use_bias=*/true)
{
}

// ═════════════════════════════════════════════════════════════════════════════
// forward
// ═════════════════════════════════════════════════════════════════════════════

NodePtr FeedForward::forward(const NodePtr& x) const
{
    // Step 1: expand from d_model → d_ff
    auto h1 = fc1.forward(x);     // [*, d_ff]

    // Step 2: non-linearity
    // GELU is chosen over ReLU because it is smooth and has been shown to give
    // better perplexity on language modelling tasks (BERT, GPT-2, etc.).
    auto h2 = gelu(h1);            // [*, d_ff]

    // Step 3: project back to d_model
    return fc2.forward(h2);       // [*, d_model]
}

// ═════════════════════════════════════════════════════════════════════════════
// parameters
// ═════════════════════════════════════════════════════════════════════════════

std::vector<NodePtr> FeedForward::parameters() const
{
    // {fc1.weight, fc1.bias, fc2.weight, fc2.bias}
    auto p = fc1.parameters();
    auto q = fc2.parameters();
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

}  // namespace engine::nn
