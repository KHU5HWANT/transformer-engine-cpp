/**
 * @file    engine/node.hpp
 * @brief   Autograd Node — the atomic unit of the computation graph (DAG).
 *
 * Design decisions (from implementation_plan_v2.md, Step 1.2):
 *
 *  Ownership model
 *  ───────────────
 *  Every Node is heap-allocated and managed exclusively via
 *  std::shared_ptr<Node> (aliased as NodePtr).  This lets multiple consumers
 *  safely co-own a leaf tensor (e.g. a weight matrix used by two layers).
 *
 *  Cycle prevention
 *  ────────────────
 *  The `children` vector stores std::weak_ptr<Node>, NOT shared_ptr.
 *  The backward lambda (_backward) is the true ownership mechanism: it closes
 *  over shared_ptr<Node> to each input, keeping them alive exactly as long as
 *  the lambda itself is alive.  Using weak_ptr in `children` means that once
 *  all user-side references and backward-lambda captures are dropped, every
 *  intermediate node is freed — no reference cycle is possible even when the
 *  same node appears as input to multiple ops.
 *
 *  Gradient accumulation
 *  ─────────────────────
 *  Because one tensor can fan out to N operations (e.g. a weight matrix used
 *  by two linear layers in the same forward pass), its gradient must be
 *  ACCUMULATED (+=) across all upstream contributions, not overwritten.
 *  accumulate_grad() performs this elementwise addition directly on the flat
 *  data buffer — no autograd, no new graph nodes, pure arithmetic.
 *
 *  No-op backward
 *  ──────────────
 *  Leaf nodes (requires_grad=true) and non-differentiable nodes
 *  (requires_grad=false) both default to a no-op _backward.  ops.cpp (Step 1.3)
 *  replaces this with the real gradient lambda when an operation is registered.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "tensor.hpp"

#include <functional>   // std::function
#include <memory>       // std::shared_ptr, std::weak_ptr
#include <string>
#include <vector>

namespace engine {

// ─────────────────────────────────────────────────────────────────────────────
// Forward declaration & convenience alias
// ─────────────────────────────────────────────────────────────────────────────

class Node;

/**
 * @brief Canonical handle type for all Nodes.
 *
 * All user-facing code, ops, and module parameters hold NodePtr values.
 * Never store a raw Node* or a bare Node on the stack after constructing ops.
 */
using NodePtr = std::shared_ptr<Node>;

// ─────────────────────────────────────────────────────────────────────────────
// Node
// ─────────────────────────────────────────────────────────────────────────────

class Node {
public:
    // ── Core Members (intentionally public for ops / autograd access) ─────────

    /**
     * @brief Forward-pass values — set at node creation time and never
     *        overwritten by the backward pass.
     */
    Tensor data;

    /**
     * @brief Gradient accumulator — always the same shape as `data`,
     *        initialised to all zeros.
     *
     * The backward pass accumulates (+=) into this tensor via accumulate_grad().
     * Call zero_grad() before each new backward pass.
     */
    Tensor grad;

    /**
     * @brief The backward closure registered by the op that produced this node.
     *
     * Captures shared_ptr to each input Node so those inputs remain alive
     * while backprop is in progress.  Defaults to a no-op (λ={}) for leaf
     * nodes and non-differentiable intermediate nodes.
     *
     * Called once per backward pass by autograd::backward() (Step 1.4).
     */
    std::function<void()> _backward;

    /**
     * @brief Weak references to the input nodes that produced this node.
     *
     * Used by the topological sort in autograd::backward() (Step 1.4) to
     * traverse the DAG without extending lifetimes.
     *
     * std::weak_ptr is used instead of shared_ptr to prevent reference cycles:
     *   If c = f(a, b), then c.children = {weak(a), weak(b)}.
     *   The backward lambda in c already holds shared_ptr{a} and shared_ptr{b},
     *   so the inputs are kept alive without children needing to own them.
     */
    std::vector<std::weak_ptr<Node>> children;

    /**
     * @brief When true, gradients will be computed and accumulated for this
     *        node during the backward pass.
     *
     * Set to true for trainable parameters (weights, biases).
     * Set to false (default) for non-leaf intermediate nodes whose gradient
     * is only needed transiently during backprop.
     */
    bool requires_grad;

    // ── Constructors ──────────────────────────────────────────────────────────

    /**
     * @brief Construct a Node holding a copy of @p tensor_data.
     *
     * `grad` is initialised to the same shape as `data`, filled with zeros.
     * `_backward` defaults to a no-op.
     * `children` is empty.
     *
     * @param tensor_data   Forward-pass Tensor to store.
     * @param requires_grad True if gradients should be accumulated here.
     */
    explicit Node(Tensor tensor_data, bool requires_grad = false);

    // Nodes must NOT be copied — shared ownership is via shared_ptr only.
    Node(const Node&)            = delete;
    Node& operator=(const Node&) = delete;

    // Move is permitted (rare, but useful in make-factory helpers).
    Node(Node&&)            = default;
    Node& operator=(Node&&) = default;

    ~Node() = default;

    // ── Factory (preferred construction path) ─────────────────────────────────

    /**
     * @brief Create a heap-allocated Node wrapped in a shared_ptr.
     *
     * Prefer this over std::make_shared<Node>(...) at call sites for clarity.
     *
     * Example:
     *   auto w = Node::make(Tensor({4, 8}), true);
     *
     * @param tensor_data   Forward-pass data Tensor.
     * @param requires_grad Whether gradients should accumulate here.
     * @return NodePtr (std::shared_ptr<Node>)
     */
    [[nodiscard]] static NodePtr make(Tensor tensor_data,
                                      bool   requires_grad = false);

    // ── Graph Construction Helpers ────────────────────────────────────────────

    /**
     * @brief Register @p child as an input to this node.
     *
     * Called by ops (Step 1.3) when building the DAG.  Stores a weak_ptr so
     * the topological sort can reach `child` without extending its lifetime.
     *
     * @param child  Shared pointer to an input Node.
     */
    void add_child(const NodePtr& child);

    // ── Gradient Helpers ──────────────────────────────────────────────────────

    /**
     * @brief Accumulate @p incoming_grad into this node's gradient tensor.
     *
     * Implements elementwise  grad[i] += incoming_grad[i]  directly on the
     * flat data buffers — no autograd graph, no new Nodes, pure arithmetic.
     *
     * Called inside _backward lambdas registered by ops (Step 1.3):
     *
     *   input_node->accumulate_grad(d_loss_d_input);
     *
     * @param incoming_grad  Gradient contribution arriving from a downstream op.
     *                       Must have the same shape as this node's `data`.
     * @throws std::invalid_argument if shapes do not match.
     */
    void accumulate_grad(const Tensor& incoming_grad);

    /**
     * @brief Reset `grad` to all zeros.
     *
     * Must be called on all parameters before every new backward pass.
     * Module::zero_grad() (Step 2.1) calls this on each parameter Node.
     */
    void zero_grad() noexcept;

    // ── Introspection & Debug ─────────────────────────────────────────────────

    /**
     * @brief Whether this node has any registered input nodes (children).
     *
     * Leaf nodes (user-created inputs / parameters) have no children and
     * represent the boundary where backprop terminates.
     */
    [[nodiscard]] bool is_leaf() const noexcept { return children.empty(); }

    /**
     * @brief Print node metadata to stdout for debugging.
     *
     * Prints: shape, requires_grad, is_leaf, grad_norm (L2 norm of grad).
     */
    void print_info() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Free-function convenience wrappers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Create a leaf parameter Node (requires_grad = true) from a Tensor.
 *
 * Equivalent to Node::make(t, true).  Mirrors the PyTorch idiom of
 * `torch.tensor(..., requires_grad=True)`.
 */
[[nodiscard]] inline NodePtr make_parameter(Tensor t) {
    return Node::make(std::move(t), /*requires_grad=*/true);
}

/**
 * @brief Create a non-differentiable input Node (requires_grad = false).
 *
 * Used for data tensors that never need gradients (e.g. token ID tensors).
 */
[[nodiscard]] inline NodePtr make_input(Tensor t) {
    return Node::make(std::move(t), /*requires_grad=*/false);
}

}  // namespace engine
