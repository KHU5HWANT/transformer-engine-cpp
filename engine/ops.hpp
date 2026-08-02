/**
 * @file    engine/ops.hpp
 * @brief   Differentiable tensor operations — the DAG-building layer.
 *
 * Design decisions (implementation_plan_v2.md, Step 1.3):
 *
 *  Every free function in engine::ops follows the autograd contract:
 *   1. Compute the forward result into a new Tensor.
 *   2. Wrap it in a new NodePtr (the output node).
 *   3. Call out->add_child(input) for each input to register DAG edges.
 *   4. Assign out->_backward — a lambda that applies the chain rule and
 *      calls input->accumulate_grad(local_grad) for each differentiable input.
 *
 *  Backward lambda ownership model (cycle safety):
 *   - Input NodePtrs are captured by VALUE (shared_ptr) — keeps inputs alive
 *     for the full duration of the backward pass.
 *   - The output node (out) is captured as std::weak_ptr<Node> to prevent
 *     the self-reference cycle:  Node → _backward lambda → Node.
 *     The topological sort (Step 1.4) holds shared_ptr to all live nodes,
 *     so weak_ptr::lock() is always valid during backward execution.
 *
 *  requires_grad propagation:
 *   - Output node requires_grad = OR of all input requires_grad flags.
 *   - This mirrors PyTorch's gradient tape behaviour.
 *
 *  High-performance matmul:
 *   - Cache-blocked tiling with BLOCK_SIZE = 64 (fits L1 cache).
 *   - #pragma omp parallel for on the outermost tile loop.
 *   - Three helpers: mm_nn (fwd), mm_nt (dA backward), mm_tn (dB backward).
 *
 * Target: Linux/WSL2, C++17, -O2, -fopenmp.
 */

#pragma once

#include "node.hpp"     // NodePtr, Node, Tensor — all pulled in transitively

namespace engine::ops {

// ─────────────────────────────────────────────────────────────────────────────
// Binary Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Elementwise addition:  out[i] = a[i] + b[i]
 *
 * Backward:
 *   ∂L/∂a[i] = ∂L/∂out[i]   (pass-through)
 *   ∂L/∂b[i] = ∂L/∂out[i]   (pass-through)
 *
 * @throws std::invalid_argument if a and b have different shapes.
 */
[[nodiscard]] NodePtr add(const NodePtr& a, const NodePtr& b);

/**
 * @brief Elementwise (Hadamard) multiplication:  out[i] = a[i] * b[i]
 *
 * Backward:
 *   ∂L/∂a[i] = ∂L/∂out[i] · b[i]
 *   ∂L/∂b[i] = ∂L/∂out[i] · a[i]
 *
 * @throws std::invalid_argument if a and b have different shapes.
 */
[[nodiscard]] NodePtr mul(const NodePtr& a, const NodePtr& b);

/**
 * @brief Matrix multiplication (2-D or batched 3-D).
 *
 *   2-D: out[M,N] = a[M,K] @ b[K,N]
 *   3-D: out[B,M,N] = a[B,M,K] @ b[B,K,N]   (batched, loop over B)
 *
 * Forward uses cache-blocked tiling (BLOCK_SIZE=64) with OpenMP.
 *
 * Backward (per batch slice):
 *   ∂L/∂A = ∂L/∂out  ·  Bᵀ   →  mm_nt kernel
 *   ∂L/∂B = Aᵀ       ·  ∂L/∂out  →  mm_tn kernel
 *
 * @throws std::invalid_argument on shape incompatibility.
 */
[[nodiscard]] NodePtr matmul(const NodePtr& a, const NodePtr& b);

// ─────────────────────────────────────────────────────────────────────────────
// Unary Elementwise Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Elementwise natural exponential:  out[i] = exp(a[i])
 *
 * Backward:
 *   ∂L/∂a[i] = ∂L/∂out[i] · exp(a[i]) = ∂L/∂out[i] · out[i]
 *   (out->data is reused — no need to re-compute exp in backward)
 */
[[nodiscard]] NodePtr exp(const NodePtr& a);

/**
 * @brief Elementwise natural logarithm:  out[i] = ln(a[i])
 *
 * Backward:
 *   ∂L/∂a[i] = ∂L/∂out[i] / a[i]
 *
 * @note Undefined behaviour if any a[i] <= 0.  Add a clamp op if needed.
 */
[[nodiscard]] NodePtr log(const NodePtr& a);

// ─────────────────────────────────────────────────────────────────────────────
// Reduction Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Global reduce-sum:  out = scalar(Σ a[i])
 *
 * Output node has shape {1} (a scalar wrapped in a 1-element tensor).
 *
 * Backward (gradient broadcast):
 *   ∂L/∂a[i] = ∂L/∂out   for all i
 *   (the scalar upstream gradient is broadcast to every element of a)
 */
[[nodiscard]] NodePtr sum(const NodePtr& a);

// ─────────────────────────────────────────────────────────────────────────────
// Shape Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Swap the last two dimensions of a tensor.
 *
 *   2-D input  [M, N]    →  output [N, M]
 *   3-D input  [B, M, N] →  output [B, N, M]
 *
 * Backward:
 *   ∂L/∂a = transpose(∂L/∂out)   (same operation applied to the gradient)
 *
 * @throws std::invalid_argument if input has fewer than 2 dimensions.
 */
[[nodiscard]] NodePtr transpose(const NodePtr& a);

}  // namespace engine::ops
