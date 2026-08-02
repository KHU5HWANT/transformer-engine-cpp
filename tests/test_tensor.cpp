/**
 * @file    tests/test_tensor.cpp
 * @brief   GTest unit tests for engine::Tensor (Step 1.1).
 *
 * Tests cover:
 *   - Construction: zero-init, shape/data constructor, brace-list constructor
 *   - Shape & stride metadata
 *   - Bounds-checked at() access (mutable and const)
 *   - Out-of-bounds and rank-mismatch exceptions
 *   - reshape() correctness and error handling
 *   - fill() / zero() helpers
 *   - numel() and ndim() consistency
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_tensor.cpp engine/tensor.cpp -lgtest -lgtest_main -pthread
 */

#include <gtest/gtest.h>
#include "engine/tensor.hpp"

using engine::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorConstruction, ZeroInitShape)
{
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.ndim(),  3u);
    EXPECT_EQ(t.numel(), 24u);
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 3u);
    EXPECT_EQ(t.shape()[2], 4u);

    // All elements must be 0.0 after zero-init construction
    for (double v : t.data()) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(TensorConstruction, ShapeDataConstructor)
{
    std::vector<double> vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor t({2, 3}, vals);

    EXPECT_EQ(t.ndim(),  2u);
    EXPECT_EQ(t.numel(), 6u);
    EXPECT_DOUBLE_EQ(t.data()[0], 1.0);
    EXPECT_DOUBLE_EQ(t.data()[5], 6.0);
}

TEST(TensorConstruction, ShapeDataSizeMismatch)
{
    // data vector size != product of shape → must throw
    EXPECT_THROW(
        (Tensor({2, 3}, {1.0, 2.0, 3.0})),   // size 3 != 6
        std::invalid_argument
    );
}

TEST(TensorConstruction, InitializerListShape)
{
    Tensor t({5});
    EXPECT_EQ(t.ndim(),  1u);
    EXPECT_EQ(t.numel(), 5u);
    EXPECT_EQ(t.shape()[0], 5u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stride Calculation (row-major C-order)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorStrides, RowMajorOrder)
{
    // shape [2, 3, 4] → strides [12, 4, 1]
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.strides()[0], 12u);
    EXPECT_EQ(t.strides()[1],  4u);
    EXPECT_EQ(t.strides()[2],  1u);
}

TEST(TensorStrides, Matrix2D)
{
    // shape [5, 7] → strides [7, 1]
    Tensor t({5, 7});
    EXPECT_EQ(t.strides()[0], 7u);
    EXPECT_EQ(t.strides()[1], 1u);
}

TEST(TensorStrides, Vector1D)
{
    // shape [N] → strides [1]
    Tensor t({10});
    EXPECT_EQ(t.strides()[0], 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Element Access — at()
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorAccess, ReadWriteMultiDim)
{
    Tensor t({3, 4});
    t.at({1, 2}) = 42.0;
    EXPECT_DOUBLE_EQ(t.at({1, 2}), 42.0);

    // Flat index for row-major: 1*4 + 2 = 6
    EXPECT_DOUBLE_EQ(t.data()[6], 42.0);
}

TEST(TensorAccess, ConstAccess)
{
    std::vector<double> vals(6, 0.0);
    vals[4] = 99.0;
    const Tensor t({2, 3}, vals);
    EXPECT_DOUBLE_EQ(t.at({1, 1}), 99.0);   // flat: 1*3+1 = 4
}

TEST(TensorAccess, OutOfBoundsThrows)
{
    Tensor t({3, 4});
    // Index 3 is out of range for dimension 0 (size=3, valid: 0,1,2)
    EXPECT_THROW(t.at({3, 0}), std::out_of_range);
    // Index 4 is out of range for dimension 1 (size=4, valid: 0..3)
    EXPECT_THROW(t.at({0, 4}), std::out_of_range);
}

TEST(TensorAccess, WrongRankThrows)
{
    Tensor t({3, 4});
    // Providing 3 indices for a 2-D tensor
    EXPECT_THROW(t.at({0, 0, 0}), std::invalid_argument);
    // Providing 1 index for a 2-D tensor
    EXPECT_THROW(t.at({0}), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// reshape()
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorReshape, ValidReshape)
{
    std::vector<double> vals = {1,2,3,4,5,6};
    Tensor t({2, 3}, vals);

    t.reshape({3, 2});
    EXPECT_EQ(t.ndim(),     2u);
    EXPECT_EQ(t.shape()[0], 3u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(t.numel(),    6u);

    // Data buffer must be unchanged — reshape is a zero-copy metadata op
    EXPECT_DOUBLE_EQ(t.data()[0], 1.0);
    EXPECT_DOUBLE_EQ(t.data()[5], 6.0);

    // Strides recomputed: shape [3,2] → strides [2,1]
    EXPECT_EQ(t.strides()[0], 2u);
    EXPECT_EQ(t.strides()[1], 1u);
}

TEST(TensorReshape, FlattenTo1D)
{
    Tensor t({2, 3, 4});    // 24 elements
    t.reshape({24});
    EXPECT_EQ(t.ndim(), 1u);
    EXPECT_EQ(t.numel(), 24u);
    EXPECT_EQ(t.strides()[0], 1u);
}

TEST(TensorReshape, NuMelMismatchThrows)
{
    Tensor t({2, 3});    // 6 elements
    EXPECT_THROW(t.reshape({4, 2}), std::invalid_argument);  // 8 ≠ 6
}

// ─────────────────────────────────────────────────────────────────────────────
// fill() / zero()
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorFill, FillAndZero)
{
    Tensor t({3, 3});
    t.fill(7.5);
    for (double v : t.data()) EXPECT_DOUBLE_EQ(v, 7.5);

    t.zero();
    for (double v : t.data()) EXPECT_DOUBLE_EQ(v, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// numel / ndim consistency
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorMeta, NumelNdimConsistency)
{
    Tensor t({2, 3, 5, 7});
    EXPECT_EQ(t.ndim(),  4u);
    EXPECT_EQ(t.numel(), 2u * 3u * 5u * 7u);
    EXPECT_EQ(t.numel(), t.data().size());
}

TEST(TensorMeta, DataPtrPointsToBuffer)
{
    Tensor t({4});
    t.fill(3.14);
    // data_ptr() and data().data() must refer to the same memory
    EXPECT_EQ(t.data_ptr(), t.data().data());
    EXPECT_DOUBLE_EQ(*t.data_ptr(), 3.14);
}
