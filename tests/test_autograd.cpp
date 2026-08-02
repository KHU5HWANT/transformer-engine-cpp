/**
 * @file    tests/test_autograd.cpp
 * @brief   GTest unit tests for the reverse-mode autograd engine (Steps 1.2–1.4).
 *
 * Tests cover:
 *   - Node construction and DAG edge registration
 *   - Leaf node detection (is_leaf())
 *   - accumulate_grad() (elementwise +=)
 *   - 3-node chain: a → b → c, backward propagates to a
 *   - Branching DAG: one node used as input to two ops — gradient must accumulate
 *   - ops::add, ops::mul, ops::exp, ops::sum backward correctness
 *   - zero_grad_all() resets all gradients in the DAG
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_autograd.cpp engine/tensor.cpp engine/node.cpp \
 *       engine/ops.cpp engine/autograd.cpp -lgtest -lgtest_main -pthread
 */

#include <gtest/gtest.h>
#include "engine/node.hpp"
#include "engine/ops.hpp"
#include "engine/autograd.hpp"

#include <cmath>      // std::exp

using engine::Node;
using engine::NodePtr;
using engine::Tensor;
namespace ops = engine::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Create a scalar parameter NodePtr with a given value.
static NodePtr scalar_param(double val)
{
    Tensor t({1});
    t.data()[0] = val;
    return engine::make_parameter(std::move(t));
}

// Create a scalar input (non-differentiable) NodePtr.
static NodePtr scalar_input(double val)
{
    Tensor t({1});
    t.data()[0] = val;
    return engine::make_input(std::move(t));
}

// ─────────────────────────────────────────────────────────────────────────────
// Node construction & DAG structure
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradNode, LeafNodeHasNoChildren)
{
    auto a = scalar_param(1.0);
    EXPECT_TRUE(a->is_leaf());
    EXPECT_EQ(a->children.size(), 0u);
    EXPECT_TRUE(a->requires_grad);
}

TEST(AutogradNode, InputNodeNotRequiresGrad)
{
    auto x = scalar_input(3.0);
    EXPECT_FALSE(x->requires_grad);
    EXPECT_TRUE(x->is_leaf());
}

TEST(AutogradNode, AddChildRegistersEdge)
{
    auto a = scalar_param(1.0);
    auto b = scalar_param(2.0);
    auto c = Node::make(Tensor({1}));
    c->add_child(a);
    c->add_child(b);

    EXPECT_EQ(c->children.size(), 2u);
    // Children are weak_ptrs — must be lockable
    EXPECT_EQ(c->children[0].lock(), a);
    EXPECT_EQ(c->children[1].lock(), b);
    EXPECT_FALSE(c->is_leaf());  // has children → not a leaf
}

TEST(AutogradNode, AccumulateGradAddsElementwise)
{
    auto a = scalar_param(5.0);
    // grad starts at 0
    EXPECT_DOUBLE_EQ(a->grad.data()[0], 0.0);

    Tensor g1({1}); g1.data()[0] =  3.0;
    Tensor g2({1}); g2.data()[0] = -1.0;

    a->accumulate_grad(g1);
    EXPECT_DOUBLE_EQ(a->grad.data()[0], 3.0);

    a->accumulate_grad(g2);   // second accumulation += -1
    EXPECT_DOUBLE_EQ(a->grad.data()[0], 2.0);
}

TEST(AutogradNode, ZeroGradClearsGradient)
{
    auto a = scalar_param(1.0);
    Tensor g({1}); g.data()[0] = 7.0;
    a->accumulate_grad(g);
    EXPECT_DOUBLE_EQ(a->grad.data()[0], 7.0);

    a->zero_grad();
    EXPECT_DOUBLE_EQ(a->grad.data()[0], 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// ops::add backward — linear chain a + b → sum → backward
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradAdd, GradientPassThrough)
{
    // f(a, b) = a + b   →   ∂f/∂a = 1,  ∂f/∂b = 1
    auto a = scalar_param(3.0);
    auto b = scalar_param(5.0);
    auto c = ops::add(a, b);           // c = 8
    auto loss = ops::sum(c);           // scalar

    engine::backward(loss);

    EXPECT_NEAR(a->grad.data()[0], 1.0, 1e-10);
    EXPECT_NEAR(b->grad.data()[0], 1.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// ops::mul backward — a * b
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradMul, GradientMul)
{
    // f(a, b) = a * b   →   ∂f/∂a = b,  ∂f/∂b = a
    auto a = scalar_param(3.0);
    auto b = scalar_param(4.0);
    auto c = ops::mul(a, b);           // c = 12
    auto loss = ops::sum(c);

    engine::backward(loss);

    EXPECT_NEAR(a->grad.data()[0], 4.0, 1e-10);   // = b
    EXPECT_NEAR(b->grad.data()[0], 3.0, 1e-10);   // = a
}

// ─────────────────────────────────────────────────────────────────────────────
// ops::exp backward — d(exp(a))/da = exp(a)
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradExp, GradientExp)
{
    double val = 1.5;
    auto a    = scalar_param(val);
    auto b    = ops::exp(a);           // b = exp(1.5)
    auto loss = ops::sum(b);

    engine::backward(loss);

    EXPECT_NEAR(a->grad.data()[0], std::exp(val), 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-node chain: loss = sum(exp(a + b))
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradChain, ThreeNodeChain)
{
    // f = exp(a + b),  ∂f/∂a = ∂f/∂b = exp(a+b)
    auto a = scalar_param(1.0);
    auto b = scalar_param(2.0);
    auto s = ops::add(a, b);           // s = 3
    auto e = ops::exp(s);              // e = exp(3)
    auto loss = ops::sum(e);

    engine::backward(loss);

    const double expected = std::exp(3.0);
    EXPECT_NEAR(a->grad.data()[0], expected, 1e-9);
    EXPECT_NEAR(b->grad.data()[0], expected, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Branching DAG — gradient accumulation
// a fans out to two ops: grad must be SUM of both contributions
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradBranch, GradientAccumulation)
{
    // f(a) = a + a = 2a   →   ∂f/∂a = 2
    // (both branches produce grad 1; they must ADD to 2)
    auto a  = scalar_param(5.0);
    auto c1 = ops::add(a, a);          // fan-out: a used twice
    auto loss = ops::sum(c1);

    engine::backward(loss);

    EXPECT_NEAR(a->grad.data()[0], 2.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// zero_grad_all resets every node in the DAG
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradZeroGradAll, ResetsFullDAG)
{
    auto a = scalar_param(1.0);
    auto b = scalar_param(2.0);
    auto c = ops::add(a, b);
    auto loss = ops::sum(c);

    engine::backward(loss);

    // After backward, a and b have non-zero grad
    EXPECT_GT(std::abs(a->grad.data()[0]), 0.0);

    engine::zero_grad_all(loss);

    EXPECT_DOUBLE_EQ(a->grad.data()[0], 0.0);
    EXPECT_DOUBLE_EQ(b->grad.data()[0], 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// backward() on non-scalar must throw
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutogradBackward, NonScalarRootThrows)
{
    auto a = scalar_param(1.0);
    auto b = scalar_param(2.0);
    auto c = ops::add(a, b);           // shape {1} — scalar, OK
    EXPECT_NO_THROW(engine::backward(c));

    // Create a 2-element non-scalar node
    Tensor t2({2}); t2.data()[0] = 1.0; t2.data()[1] = 2.0;
    auto non_scalar = engine::make_input(std::move(t2));
    EXPECT_THROW(engine::backward(non_scalar), std::invalid_argument);
}
