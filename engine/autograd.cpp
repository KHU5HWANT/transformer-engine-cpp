/**
 * @file    engine/autograd.cpp
 * @brief   Implementation of the reverse-mode automatic differentiation engine.
 *
 * See autograd.hpp for the full API documentation and design rationale.
 *
 * The two key algorithms are:
 *
 *  dfs_postorder()  [file-local]
 *  ──────────────────────────────
 *  Recursive DFS that appends nodes to `topo` in post-order:
 *    - All of a node's children (inputs) are appended BEFORE the node itself.
 *    - The `visited` set (keyed by raw Node*) prevents re-processing nodes
 *      that are shared inputs to multiple operations (DAG fan-out).
 *
 *  The resulting `topo` vector is in "leaves-first, root-last" order.
 *  Iterating it in REVERSE gives "root-first, leaves-last" — exactly the
 *  order in which the backward lambdas must fire so that every node receives
 *  its complete upstream gradient before propagating it further.
 *
 *  Why raw Node* in the visited set?
 *  ──────────────────────────────────
 *  We need pointer identity, not value equality.  std::unordered_set<Node*>
 *  uses std::hash<Node*> (the default pointer hash) which is fast and correct.
 *  The NodePtrs in `topo` keep the pointed-to nodes alive, so the raw
 *  pointers in `visited` are never dangling during traversal.
 *
 * Build:
 *   g++ -std=c++17 -O2 -fopenmp \
 *       engine/tensor.cpp engine/node.cpp engine/ops.cpp engine/autograd.cpp
 */

#include "autograd.hpp"

#include <stdexcept>        // std::invalid_argument
#include <string>
#include <unordered_set>    // std::unordered_set (visited set, O(1) lookup)
#include <vector>           // std::vector (topo order)

namespace engine {

// ═════════════════════════════════════════════════════════════════════════════
// File-local helpers
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// dfs_postorder
// ─────────────────────────────────────────────────────────────────────────────
//
// Recursive DFS post-order traversal.
//
// Post-order property (critical for correctness):
//   For any node N, every node that N depends on (its children / inputs)
//   appears in `topo` BEFORE N.  Reversing `topo` therefore guarantees that
//   N._backward() is called BEFORE any of N's inputs' _backward() functions —
//   i.e. each node receives a fully accumulated `grad` before it propagates
//   further.
//
// Parameters:
//   node    — current node being visited (may be null after a failed lock).
//   topo    — output vector; nodes appended in post-order.
//   visited — raw-pointer set for O(1) duplicate detection across DAG branches.
//
// Thread safety: single-threaded (backward is always sequential).
//
void dfs_postorder(const NodePtr&                 node,
                   std::vector<NodePtr>&           topo,
                   std::unordered_set<const Node*>& visited)
{
    // ── Guard: null or already processed ────────────────────────────────────
    if (!node) return;

    const Node* raw = node.get();
    if (visited.count(raw)) return;   // O(1) lookup — handles DAG fan-out
    visited.insert(raw);

    // ── Recurse into children (the inputs that produced this node) ───────────
    //
    // children are stored as std::weak_ptr<Node> to avoid reference cycles.
    // We lock() each one to obtain a temporary shared_ptr for the recursive
    // call.  A null result (node already freed) is silently skipped — this
    // should never occur during a well-formed backward pass since `topo`
    // holds shared_ptr to every discovered node, keeping them alive.
    //
    for (const std::weak_ptr<Node>& weak_child : node->children) {
        if (NodePtr child = weak_child.lock()) {
            dfs_postorder(child, topo, visited);
        }
        // Null lock: child was freed before traversal reached it.
        // Safe to skip — it has no gradient to propagate.
    }

    // ── Append this node AFTER all its children (post-order) ────────────────
    topo.push_back(node);
}

}  // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// engine::backward
// ═════════════════════════════════════════════════════════════════════════════

void backward(const NodePtr& root)
{
    // ── Validate root ────────────────────────────────────────────────────────
    if (!root) {
        throw std::invalid_argument(
            "engine::backward(): root NodePtr is null.");
    }
    if (root->data.numel() != 1) {
        throw std::invalid_argument(
            "engine::backward(): root must be a scalar node (numel == 1). "
            "Got shape " + root->data.shape_str() +
            " (numel = " + std::to_string(root->data.numel()) + "). "
            "Did you forget to call ops::sum() to reduce to a scalar loss?");
    }

    // ── Step 1: Seed the gradient ────────────────────────────────────────────
    //
    // ∂loss/∂loss = 1.0  — the base case of the chain rule.
    // root->grad already has the same shape as root->data ({1}) and is
    // zero-initialised; we simply set the single element to 1.0.
    //
    root->grad.data_ptr()[0] = 1.0;

    // ── Step 2: Topological sort (DFS post-order) ────────────────────────────
    //
    // `topo` ends up as: [ leaf_0, leaf_1, ..., intermediate_nodes..., root ]
    //
    // The `visited` set uses raw Node* (pointer identity) for O(1) detection
    // of already-visited nodes when the DAG has shared inputs (fan-out).
    // Nodes in `topo` are kept alive by their shared_ptr entries, so raw
    // pointers in `visited` are never dangling.
    //
    std::vector<NodePtr>             topo;
    std::unordered_set<const Node*>  visited;

    topo.reserve(64);       // amortise reallocations for typical graph sizes
    visited.reserve(64);

    dfs_postorder(root, topo, visited);

    // ── Step 3: Reverse-order backward sweep ─────────────────────────────────
    //
    // `topo` is leaves-first, root-last.
    // Reversing: root-first, leaves-last.
    //
    // For each node (starting from root):
    //   node->_backward() reads node->grad  (the accumulated upstream gradient)
    //   and calls input->accumulate_grad(local_grad) for each input.
    //
    // By the time we reach any node N in this sweep, all nodes that produce N
    // as their output have already fired their _backward() — so N->grad is
    // fully accumulated before N->_backward() propagates it further.
    //
    // The lambda's weak_ptr self-capture (wout in ops.cpp) is safe here
    // because `topo` holds a shared_ptr to every node in the graph, preventing
    // premature deallocation for the entire duration of this loop.
    //
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const NodePtr& node = *it;

        // _backward is always valid (defaults to no-op lambda, never null).
        // Calling it on leaf nodes with no children is harmless — the no-op
        // lambda simply returns immediately.
        node->_backward();
    }

    // `topo` goes out of scope here.
    // Reference counts of all intermediate nodes drop; any node with no other
    // live NodePtr is freed.  This is the safe cleanup window documented in
    // autograd.hpp.
}

// ═════════════════════════════════════════════════════════════════════════════
// engine::zero_grad_all
// ═════════════════════════════════════════════════════════════════════════════

void zero_grad_all(const NodePtr& root)
{
    if (!root) return;

    // Reuse the same DFS traversal — O(N) in graph size.
    std::vector<NodePtr>             topo;
    std::unordered_set<const Node*>  visited;
    dfs_postorder(root, topo, visited);

    for (const NodePtr& node : topo) {
        node->zero_grad();    // calls Tensor::zero() on node->grad
    }
}

}  // namespace engine
