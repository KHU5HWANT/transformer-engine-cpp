/**
 * @file    tests/test_grad_check.cpp
 * @brief   Numerical gradient checker — the gold standard proof of autograd correctness.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The Finite Difference Method
 * ════════════════════════════════════════════════════════════════════════════
 *
 * For a scalar-valued function f(θ), the numerical gradient at θ_i is:
 *
 *   ∂f/∂θ_i ≈ [ f(θ + e·eᵢ) − f(θ − e·eᵢ) ] / (2e)
 *
 * where eᵢ is the i-th standard basis vector and e is a small perturbation.
 *
 * Why the CENTRAL difference (not forward difference)?
 *   Forward difference:  [f(x+e) − f(x)] / e   → O(e)  error
 *   Central difference:  [f(x+e) − f(x−e)] / 2e → O(e²) error
 *
 * Using e = 1e-4 and central differences gives ~1e-8 numerical accuracy
 * for smooth functions, easily within our 1e-6 tolerance.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Tolerance
 * ════════════════════════════════════════════════════════════════════════════
 *
 * We require:  |analytical_grad[i] − numerical_grad[i]| < 1e-6
 *
 * This is strict enough to catch wrong-sign gradients, missing chain-rule
 * factors, and incorrect backward lambda implementations.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Functions tested
 * ════════════════════════════════════════════════════════════════════════════
 *   - f(a) = sum(exp(a))            — tests exp backward
 *   - f(a,b) = sum(a * b)           — tests mul backward (both inputs)
 *   - f(a,b) = sum(a + b)           — tests add backward
 *   - f(A,B) = sum(A @ B)           — tests matmul backward (both inputs)
 *   - f(a) = sum(log(abs(a)+1))     — tests log backward
 *   - f(a) = sum(exp(a*b + c))      — tests composed chain
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_grad_check.cpp engine/tensor.cpp engine/node.cpp \
 *       engine/ops.cpp engine/autograd.cpp -lgtest -lgtest_main -pthread
 */

#include <gtest/gtest.h>
#include "engine/node.hpp"
#include "engine/ops.hpp"
#include "engine/autograd.hpp"

#include <cmath>
#include <functional>
#include <vector>

using engine::Node;
using engine::NodePtr;
using engine::Tensor;
namespace ops = engine::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Central finite difference gradient checker
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double EPS = 1e-4;
static constexpr double TOL = 1e-6;

/**
 * @brief Compute analytical gradient of a scalar function and compare it
 *        to the central-difference numerical gradient for every element.
 *
 * @param params   All differentiable leaf NodePtrs in the computation graph.
 * @param build_graph  Callable () → NodePtr(scalar loss).
 *                     Must rebuild the full computation graph each call.
 */
static void check_gradients(
    const std::vector<NodePtr>& params,
    const std::function<NodePtr()>& build_graph)
{
    // ── 1. Analytical gradient ────────────────────────────────────────────────
    // Zero all grads, run one forward+backward pass.
    for (const auto& p : params) p->zero_grad();

    NodePtr loss = build_graph();
    engine::backward(loss);

    // Save analytical gradients before perturbing data
    std::vector<std::vector<double>> analytical;
    for (const auto& p : params) {
        analytical.push_back(p->grad.data());   // copy
    }

    // ── 2. Numerical gradient via central finite difference ───────────────────
    for (size_t pi = 0; pi < params.size(); ++pi) {
        const NodePtr& p = params[pi];
        double* pdata    = p->data.data_ptr();
        const size_t n   = p->data.numel();

        for (size_t j = 0; j < n; ++j) {
            const double orig = pdata[j];

            // f(θ + e)
            pdata[j] = orig + EPS;
            const double f_plus = build_graph()->data.data_ptr()[0];

            // f(θ - e)
            pdata[j] = orig - EPS;
            const double f_minus = build_graph()->data.data_ptr()[0];

            // Restore original value
            pdata[j] = orig;

            const double numerical  = (f_plus - f_minus) / (2.0 * EPS);
            const double analytical_ij = analytical[pi][j];

            EXPECT_NEAR(analytical_ij, numerical, TOL)
                << "Gradient mismatch for param[" << pi << "][" << j << "]: "
                << "analytical=" << analytical_ij
                << "  numerical=" << numerical;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: create a NodePtr of given shape with random-ish data (deterministic)
// ─────────────────────────────────────────────────────────────────────────────

static NodePtr make_param(std::vector<size_t> shape,
                           double start = 0.5, double step = 0.1)
{
    Tensor t(shape);
    for (size_t i = 0; i < t.numel(); ++i) {
        t.data()[i] = start + static_cast<double>(i) * step;
    }
    return engine::make_parameter(std::move(t));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: f(a) = sum(exp(a))
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, ExpScalar)
{
    auto a = make_param({3}, 0.1, 0.3);

    check_gradients({a}, [&]() -> NodePtr {
        return ops::sum(ops::exp(a));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: f(a) = sum(log(a))  — a must be > 0
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, LogScalar)
{
    auto a = make_param({4}, 1.0, 0.5);  // values: 1.0, 1.5, 2.0, 2.5 — all > 0

    check_gradients({a}, [&]() -> NodePtr {
        return ops::sum(ops::log(a));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: f(a, b) = sum(a + b)
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, Add)
{
    auto a = make_param({4}, 0.5, 0.2);
    auto b = make_param({4}, 1.0, 0.3);

    check_gradients({a, b}, [&]() -> NodePtr {
        return ops::sum(ops::add(a, b));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: f(a, b) = sum(a * b)
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, Mul)
{
    auto a = make_param({4}, 0.5, 0.4);
    auto b = make_param({4}, 1.0, 0.2);

    check_gradients({a, b}, [&]() -> NodePtr {
        return ops::sum(ops::mul(a, b));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: f(A, B) = sum(A @ B)  —  2D matmul
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, Matmul2D)
{
    // A: [2, 3],  B: [3, 2]  →  C: [2, 2]
    auto A = make_param({2, 3}, 0.1, 0.15);
    auto B = make_param({3, 2}, 0.2, 0.10);

    check_gradients({A, B}, [&]() -> NodePtr {
        return ops::sum(ops::matmul(A, B));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: composed chain  f(a, b, c) = sum(exp(a * b + c))
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, ComposedChain)
{
    auto a = make_param({3}, 0.2, 0.1);
    auto b = make_param({3}, 0.5, 0.2);
    auto c = make_param({3}, 0.1, 0.05);

    check_gradients({a, b, c}, [&]() -> NodePtr {
        auto ab   = ops::mul(a, b);
        auto abc  = ops::add(ab, c);
        auto eabc = ops::exp(abc);
        return ops::sum(eabc);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: transpose backward  f(A) = sum(transpose(A))
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, Transpose)
{
    auto A = make_param({3, 4}, 0.1, 0.2);

    check_gradients({A}, [&]() -> NodePtr {
        return ops::sum(ops::transpose(A));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: fan-out DAG  f(a) = sum(a * a)   (a used twice)
// ─────────────────────────────────────────────────────────────────────────────

TEST(GradCheck, FanOutDAG)
{
    // f(a) = a²  →  ∂f/∂a = 2a
    auto a = make_param({3}, 1.0, 0.5);

    check_gradients({a}, [&]() -> NodePtr {
        return ops::sum(ops::mul(a, a));
    });
}
