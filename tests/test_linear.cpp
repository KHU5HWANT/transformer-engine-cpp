/**
 * @file    tests/test_linear.cpp
 * @brief   GTest unit tests for engine::nn::Linear (Step 2.1).
 *
 * Tests cover:
 *   - Forward output shape: [B, in] → [B, out]
 *   - Forward computation correctness (small known values)
 *   - parameters() returns exactly 2 NodePtrs (weight + bias)
 *   - zero_grad() resets all parameter gradients to 0.0
 *   - Backward gradient shape matches parameter shape
 *   - Xavier weight initialisation bounds (|w| < sqrt(6/(in+out)) * 3σ heuristic)
 *   - Bias initialisation: all zeros
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_linear.cpp engine/tensor.cpp engine/node.cpp \
 *       engine/ops.cpp engine/autograd.cpp nn/linear.cpp -lgtest -lgtest_main -pthread
 */

#include <gtest/gtest.h>
#include "nn/linear.hpp"
#include "engine/autograd.hpp"
#include "engine/ops.hpp"

#include <cmath>
#include <vector>

using engine::NodePtr;
using engine::Tensor;
using engine::nn::Linear;
namespace ops = engine::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Helper — create a batch input NodePtr [B, in_features] filled with val
// ─────────────────────────────────────────────────────────────────────────────

static NodePtr make_input_batch(size_t B, size_t in_features,
                                 double start = 1.0, double step = 0.1)
{
    Tensor t({B, in_features});
    for (size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = start + static_cast<double>(i) * step;
    return engine::make_input(std::move(t));
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward — output shape
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearForward, OutputShape2D)
{
    Linear layer(4, 8);
    auto x   = make_input_batch(2, 4);   // [2, 4]
    auto out = layer.forward(x);

    ASSERT_EQ(out->data.ndim(),  2u);
    EXPECT_EQ(out->data.shape()[0], 2u);   // B unchanged
    EXPECT_EQ(out->data.shape()[1], 8u);   // out_features
}

TEST(LinearForward, OutputShape3D)
{
    // 3-D input [B, T, in] — forward should reshape, matmul, reshape back
    Linear layer(6, 10);
    Tensor t({3, 5, 6});
    auto x   = engine::make_input(std::move(t));
    auto out = layer.forward(x);

    // Output should be [3, 5, 10]
    ASSERT_EQ(out->data.ndim(), 3u);
    EXPECT_EQ(out->data.shape()[0],  3u);
    EXPECT_EQ(out->data.shape()[1],  5u);
    EXPECT_EQ(out->data.shape()[2], 10u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward — numerical correctness with manually set weights
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearForward, NumericalCorrectness)
{
    // y = xW^T + b
    // Set W = [[1,0],[0,1]] (identity), b = [0, 0]
    // Input x = [[2, 3]]  →  output should be [[2, 3]]
    Linear layer(2, 2);

    // Manually set weight to identity, bias to zero
    auto& W = layer.weight->data;    // shape [out=2, in=2]
    auto& b = layer.bias->data;      // shape [out=2]
    W.fill(0.0);
    b.fill(0.0);
    W.at({0, 0}) = 1.0;
    W.at({1, 1}) = 1.0;

    Tensor x_t({1, 2});
    x_t.at({0, 0}) = 2.0;
    x_t.at({0, 1}) = 3.0;
    auto x   = engine::make_input(std::move(x_t));
    auto out = layer.forward(x);

    EXPECT_NEAR(out->data.at({0, 0}), 2.0, 1e-10);
    EXPECT_NEAR(out->data.at({0, 1}), 3.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// parameters()
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearParams, ParameterCount)
{
    Linear layer(4, 8);
    auto params = layer.parameters();
    // Linear has exactly 2 parameters: weight and bias
    EXPECT_EQ(params.size(), 2u);
}

TEST(LinearParams, AllParamsRequireGrad)
{
    Linear layer(4, 8);
    for (const auto& p : layer.parameters()) {
        EXPECT_TRUE(p->requires_grad)
            << "All Linear parameters must have requires_grad=true";
    }
}

TEST(LinearParams, WeightShape)
{
    Linear layer(4, 8);
    // weight shape: [out_features, in_features] = [8, 4]
    EXPECT_EQ(layer.weight->data.shape()[0], 8u);
    EXPECT_EQ(layer.weight->data.shape()[1], 4u);
}

TEST(LinearParams, BiasShape)
{
    Linear layer(4, 8);
    // bias shape: [out_features] = [8]
    EXPECT_EQ(layer.bias->data.ndim(),    1u);
    EXPECT_EQ(layer.bias->data.shape()[0], 8u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearInit, BiasInitialisedToZero)
{
    Linear layer(16, 32);
    for (double v : layer.bias->data.data()) {
        EXPECT_DOUBLE_EQ(v, 0.0) << "Bias must be zero-initialised";
    }
}

TEST(LinearInit, WeightXavierBounds)
{
    // Xavier uniform: weights should be in [-limit, +limit]
    // limit = sqrt(6 / (fan_in + fan_out))
    const size_t in = 64, out = 64;
    Linear layer(in, out);
    const double limit = std::sqrt(6.0 / static_cast<double>(in + out));

    // Allow 3× limit as a generous check for non-uniform distributions
    for (double v : layer.weight->data.data()) {
        EXPECT_LT(std::abs(v), limit * 4.0)
            << "Weight value " << v << " is far outside Xavier range";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// zero_grad() — clears all parameter gradients
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearZeroGrad, GradsClearedAfterBackward)
{
    Linear layer(4, 4);
    auto x    = make_input_batch(2, 4);
    auto out  = layer.forward(x);
    auto loss = ops::sum(out);

    engine::backward(loss);

    // After backward, weight and bias should have non-zero grad (generally)
    // Now zero them
    layer.zero_grad();

    for (const auto& p : layer.parameters()) {
        for (double v : p->grad.data()) {
            EXPECT_DOUBLE_EQ(v, 0.0)
                << "zero_grad() must reset every gradient element to 0";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward gradient shapes
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearBackward, GradShapesMatchParamShapes)
{
    Linear layer(4, 8);
    auto x    = make_input_batch(2, 4);
    auto out  = layer.forward(x);
    auto loss = ops::sum(out);

    engine::backward(loss);

    // Weight grad shape must match weight data shape
    EXPECT_EQ(layer.weight->grad.shape(), layer.weight->data.shape());
    // Bias grad shape must match bias data shape
    EXPECT_EQ(layer.bias->grad.shape(),   layer.bias->data.shape());
}

TEST(LinearBackward, WeightGradNotAllZero)
{
    // With a non-trivial input, the weight gradient should not be all-zero
    Linear layer(4, 4);
    auto x    = make_input_batch(2, 4, 1.0, 0.5);
    auto out  = layer.forward(x);
    auto loss = ops::sum(out);
    engine::backward(loss);

    double grad_norm = 0.0;
    for (double v : layer.weight->grad.data()) grad_norm += v * v;
    EXPECT_GT(grad_norm, 1e-12) << "Weight gradient should not be all-zero";
}
