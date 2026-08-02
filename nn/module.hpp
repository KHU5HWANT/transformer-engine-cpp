/**
 * @file    nn/module.hpp
 * @brief   Abstract base class for all neural network modules.
 *
 * Design (implementation_plan_v2.md, Step 2.1):
 *
 *  Every trainable layer (Linear, LayerNorm, CausalSelfAttention, ...) inherits
 *  from Module.  The contract is:
 *
 *   parameters()
 *   ─────────────
 *   Returns a flat std::vector<NodePtr> of every NodePtr that is a trainable
 *   parameter (requires_grad = true).  Concrete modules MUST override this and
 *   include parameters from any sub-modules they own — the optimizer and the
 *   AdamW update loop (Step 4.1) will call parameters() once per training step.
 *
 *   zero_grad()
 *   ────────────
 *   Calls Node::zero_grad() on every NodePtr returned by parameters().
 *   Must be called before every new forward+backward step to prevent gradient
 *   accumulation across iterations.
 *
 * Why parameters() is pure virtual:
 *   - Prevents instantiating Module directly.
 *   - Forces each concrete layer to declare its own parameter list explicitly —
 *     no accidental omission of a weight matrix.
 *   - Composite modules (e.g. TransformerBlock) implement parameters() by
 *     calling and concatenating sub-module parameters(), giving recursive
 *     collection without any registration bookkeeping.
 *
 * Header-only: no .cpp needed — Module has no implementation state.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "engine/node.hpp"    // NodePtr, Node — the autograd primitive

#include <vector>

namespace engine::nn {

// ─────────────────────────────────────────────────────────────────────────────
// Module  —  abstract base for all trainable neural network components
// ─────────────────────────────────────────────────────────────────────────────

class Module {
public:
    // ── Destructor ────────────────────────────────────────────────────────────
    virtual ~Module() = default;

    // Modules own sub-modules via shared_ptr or by value; allow move.
    Module()                          = default;
    Module(Module&&)                  = default;
    Module& operator=(Module&&)       = default;

    // Do NOT copy modules — weights are shared_ptr (NodePtr).
    // Copying would silently alias parameters, causing gradient aliasing bugs.
    Module(const Module&)             = delete;
    Module& operator=(const Module&)  = delete;

    // ── Parameters ────────────────────────────────────────────────────────────

    /**
     * @brief Collect all trainable parameters reachable from this module.
     *
     * Concrete modules must:
     *   1. Include their own NodePtr members (weights, biases, scale vectors).
     *   2. Call and append the results of sub-module.parameters() if they own
     *      child modules.
     *
     * Example for a composite module:
     * @code
     *   std::vector<NodePtr> TransformerBlock::parameters() const {
     *       auto p = attention_.parameters();
     *       auto q = ffn_.parameters();
     *       p.insert(p.end(), q.begin(), q.end());
     *       return p;
     *   }
     * @endcode
     *
     * @return Flat vector of NodePtrs, all with requires_grad = true.
     */
    [[nodiscard]] virtual std::vector<NodePtr> parameters() const = 0;

    // ── Gradient Reset ────────────────────────────────────────────────────────

    /**
     * @brief Zero the gradient tensor of every trainable parameter.
     *
     * Call this at the start of each training step (before the forward pass)
     * to clear accumulated gradients from the previous iteration.
     *
     * Implementation delegates to Node::zero_grad() on each parameter.
     * Because parameters() is virtual, this correctly handles composite modules
     * that aggregate sub-module parameters.
     */
    void zero_grad() {
        for (const NodePtr& p : parameters()) {
            p->zero_grad();
        }
    }
};

}  // namespace engine::nn
