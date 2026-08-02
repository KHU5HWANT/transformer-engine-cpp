/**
 * @file    engine/node.cpp
 * @brief   Implementation of the autograd Node.
 *
 * See node.hpp for the full API documentation, ownership model, and design
 * rationale.
 *
 * Implementation notes:
 *  - accumulate_grad() operates directly on flat data_ptr() buffers with a
 *    hand-written loop.  This avoids any dependency on ops.hpp (Step 1.3)
 *    and keeps Step 1.2 fully self-contained.
 *  - No BLAS, no Eigen.  Pure C++17 STL only.
 */

#include "node.hpp"

#include <cmath>        // std::sqrt (used in grad norm in print_info)
#include <iostream>     // std::cout
#include <numeric>      // std::inner_product (for L2 norm)
#include <stdexcept>    // std::invalid_argument
#include <string>

namespace engine {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (file-local)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/**
 * @brief Verify that two Tensors have identical shapes.
 *
 * Used to guard accumulate_grad() before touching raw pointers.
 *
 * @throws std::invalid_argument on mismatch.
 */
void assert_same_shape(const Tensor& a, const Tensor& b,
                       const char* caller) {
    if (a.shape() != b.shape()) {
        std::string msg = std::string(caller) +
            ": shape mismatch — lhs=" + a.shape_str() +
            "  rhs=" + b.shape_str();
        throw std::invalid_argument(msg);
    }
}

/**
 * @brief Compute the L2 norm of a Tensor's flat data buffer.
 *
 * Used only in print_info() — not part of the differentiable graph.
 */
double l2_norm(const Tensor& t) noexcept {
    const double* p   = t.data_ptr();
    const size_t  n   = t.numel();
    double        sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += p[i] * p[i];
    }
    return std::sqrt(sum);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

Node::Node(Tensor tensor_data, bool req_grad)
    : data(std::move(tensor_data))
    , grad(data.shape())               // same shape as data, zero-initialised
    , _backward([]() noexcept {})      // default: no-op lambda
    , children()                       // empty — leaf node until an op registers children
    , requires_grad(req_grad)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

NodePtr Node::make(Tensor tensor_data, bool requires_grad) {
    // std::make_shared performs a single allocation for both the Node object
    // and the shared_ptr control block — better cache behaviour than
    // new Node(...) + shared_ptr wrapping.
    return std::make_shared<Node>(std::move(tensor_data), requires_grad);
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Construction
// ─────────────────────────────────────────────────────────────────────────────

void Node::add_child(const NodePtr& child) {
    // Store a weak_ptr — the topological sort (Step 1.4) will lock() each
    // weak_ptr when traversing the DAG.  If the child has already been freed
    // (which should not happen during a normal training step), lock() returns
    // nullptr and the traversal simply skips that edge.
    children.emplace_back(child);   // implicit NodePtr → std::weak_ptr<Node>
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient Helpers
// ─────────────────────────────────────────────────────────────────────────────

void Node::accumulate_grad(const Tensor& incoming_grad) {
    // ── Guard ────────────────────────────────────────────────────────────────
    assert_same_shape(grad, incoming_grad, "Node::accumulate_grad");

    // ── Elementwise grad[i] += incoming_grad[i] ───────────────────────────
    //
    // Operates directly on the flat 1-D buffers exposed by data_ptr().
    // This is safe and correct because Tensor guarantees contiguous row-major
    // storage.  No new Nodes, no graph entries — this is pure arithmetic on
    // raw doubles.
    //
    // The loop is trivially auto-vectorisable by GCC/Clang under -O2.
    //
    const size_t  n   = grad.numel();
    double*       dst = grad.data_ptr();
    const double* src = incoming_grad.data_ptr();

    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void Node::zero_grad() noexcept {
    // Delegate to Tensor::zero() which calls fill(0.0) on the flat buffer.
    grad.zero();
}

// ─────────────────────────────────────────────────────────────────────────────
// Introspection & Debug
// ─────────────────────────────────────────────────────────────────────────────

void Node::print_info() const {
    std::cout << "Node {\n"
              << "  data.shape    = " << data.shape_str()           << "\n"
              << "  requires_grad = " << (requires_grad ? "true" : "false") << "\n"
              << "  is_leaf       = " << (is_leaf()     ? "true" : "false") << "\n"
              << "  num_children  = " << children.size()            << "\n"
              << "  grad_l2_norm  = " << l2_norm(grad)              << "\n"
              << "  has_backward  = ";

    // A default no-op std::function is still truthy in C++, so we distinguish
    // by checking if the target is the empty std::function (which would be
    // falsy).  The cleanest check is simply whether _backward is non-null.
    if (_backward) {
        std::cout << "yes\n";
    } else {
        std::cout << "no (null)\n";
    }

    std::cout << "}\n";
}

}  // namespace engine
