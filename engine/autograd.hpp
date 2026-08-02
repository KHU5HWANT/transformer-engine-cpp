/**
 * @file    engine/autograd.hpp
 * @brief   Reverse-mode automatic differentiation — backward pass entry point.
 *
 * This header completes the core autograd engine (Step 1.4).
 * It provides two public functions in the engine namespace:
 *
 *  backward(root)
 *  ──────────────
 *  The main entry point for gradient computation.  Given a scalar loss node,
 *  it seeds root->grad = 1.0, runs a DFS post-order topological sort over the
 *  computation DAG, then calls every node's _backward() in reverse topological
 *  order (from root to leaves).  This is the "reverse-mode" in reverse-mode AD.
 *
 *  zero_grad_all(root)
 *  ────────────────────
 *  Traverses the same DAG and calls Node::zero_grad() on every node.
 *  Useful for clearing gradients at the start of a training step before any
 *  Module::zero_grad() calls are available (Step 2.1).
 *
 * Design notes:
 *  - Topological sort uses DFS post-order with an std::unordered_set<Node*>
 *    to handle DAG fan-out (one node used as input to multiple ops).
 *  - children are std::weak_ptr<Node>; they are locked during traversal.
 *    A null lock result (freed node) is silently skipped — safe by design.
 *  - backward() does NOT zero existing gradients before running.
 *    Gradients accumulate (+=) by design — call zero_grad_all() or
 *    Module::zero_grad() before each training step.
 *
 * Target: Linux/WSL2, C++17, pure STL.
 */

#pragma once

#include "node.hpp"   // NodePtr

namespace engine {

// ─────────────────────────────────────────────────────────────────────────────
// backward
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute gradients for the entire computation graph rooted at @p root.
 *
 * Algorithm:
 *  1. Validate that @p root is a scalar node (numel == 1).
 *  2. Seed: set root->grad.data_ptr()[0] = 1.0  (∂loss/∂loss = 1).
 *  3. Topological sort: DFS post-order traversal of the DAG (leaves first,
 *     root last), tracking visited nodes with std::unordered_set<Node*>.
 *  4. Iterate the sorted list in reverse (root first, leaves last) and invoke
 *     node->_backward() on each node.  Each call propagates gradients one
 *     step upstream via accumulate_grad().
 *
 * @param root  Scalar loss node (numel must be 1).
 *
 * @throws std::invalid_argument  if root is null or non-scalar.
 *
 * @note Gradients are ACCUMULATED (+=).  Callers must zero all parameter
 *       gradients before each training step to avoid double-counting.
 *
 * @note The topological sort holds a local std::vector<NodePtr> that keeps
 *       every reachable node alive for the full duration of the backward
 *       pass.  After backward() returns, the vector is freed, releasing all
 *       intermediate nodes simultaneously.  This is the mechanism that makes
 *       the weak_ptr self-capture in ops.cpp safe.
 */
void backward(const NodePtr& root);

// ─────────────────────────────────────────────────────────────────────────────
// zero_grad_all
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Zero the gradient tensor of every node reachable from @p root.
 *
 * Traverses the DAG using the same DFS post-order algorithm as backward()
 * and calls Node::zero_grad() on each visited node.
 *
 * Use this before each new forward+backward training step to clear
 * accumulated gradients from the previous iteration.
 *
 * @param root  Any node in the computation graph (typically the loss node
 *              or any parameter whose graph you want to reset).
 */
void zero_grad_all(const NodePtr& root);

}  // namespace engine
