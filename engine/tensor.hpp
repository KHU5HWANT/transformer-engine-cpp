/**
 * @file    engine/tensor.hpp
 * @brief   N-dimensional Tensor with flat contiguous storage and row-major strides.
 *
 * Design decisions (from implementation_plan_v2.md, Step 1.1):
 *  - Storage  : flat std::vector<double> for L1/L2 cache locality.
 *  - Metadata : shape + pre-computed row-major strides.
 *  - Indexing : bounds-checked at() converts N-d indices → 1-D flat index.
 *  - No Eigen, no LibTorch, no Boost.  Pure C++17 STL only.
 *
 * Target: Linux/WSL2, C++17, -O2 / -O3.
 */

#pragma once

#include <cstddef>      // size_t
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

// ─────────────────────────────────────────────────────────────────────────────
// Tensor
// ─────────────────────────────────────────────────────────────────────────────

class Tensor {
public:
    // ── Constructors ──────────────────────────────────────────────────────────

    /**
     * @brief Construct a zero-initialised Tensor with the given shape.
     *
     * Example:
     *   Tensor t({2, 3, 4});   // shape [2,3,4], 24 doubles, all 0.0
     *
     * @throws std::invalid_argument  if shape is empty.
     */
    explicit Tensor(std::vector<size_t> shape);

    /**
     * @brief Construct a Tensor from an existing flat data buffer.
     *
     * The size of @p data must exactly equal the product of @p shape dimensions.
     *
     * @throws std::invalid_argument  on shape/data size mismatch.
     */
    Tensor(std::vector<size_t> shape, std::vector<double> data);

    /**
     * @brief Convenience constructor accepting a brace-list shape, e.g.
     *        Tensor t({3, 4});
     */
    Tensor(std::initializer_list<size_t> shape);

    // ── Shape & Stride Accessors ──────────────────────────────────────────────

    /** Number of dimensions (rank). */
    [[nodiscard]] size_t ndim()  const noexcept { return shape_.size(); }

    /** Total number of elements (product of all dimensions). */
    [[nodiscard]] size_t numel() const noexcept { return data_.size();  }

    /** Const reference to the shape vector. */
    [[nodiscard]] const std::vector<size_t>& shape()   const noexcept { return shape_;   }

    /** Const reference to the strides vector (in elements, not bytes). */
    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return strides_; }

    // ── Element Access ────────────────────────────────────────────────────────

    /**
     * @brief Bounds-checked N-dimensional element access (mutable).
     *
     * Converts the N-dimensional index vector into a 1-D flat index using the
     * pre-computed row-major strides:
     *
     *   flat_index = sum_i ( indices[i] * strides_[i] )
     *
     * @param indices  Multi-dimensional index; length must equal ndim().
     * @throws std::out_of_range      if any index exceeds its dimension bound.
     * @throws std::invalid_argument  if indices.size() != ndim().
     */
    [[nodiscard]] double& at(const std::vector<size_t>& indices);

    /** Bounds-checked N-dimensional element access (const). */
    [[nodiscard]] const double& at(const std::vector<size_t>& indices) const;

    // ── Raw Data Access ───────────────────────────────────────────────────────

    /** Direct access to the flat storage buffer (mutable). */
    [[nodiscard]] std::vector<double>&       data()       noexcept { return data_; }

    /** Direct access to the flat storage buffer (const). */
    [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }

    /** Pointer to the first element (compatible with BLAS/OpenMP). */
    [[nodiscard]] double*       data_ptr()       noexcept { return data_.data(); }
    [[nodiscard]] const double* data_ptr() const noexcept { return data_.data(); }

    // ── Shape Manipulation ────────────────────────────────────────────────────

    /**
     * @brief Reshape the tensor in-place.
     *
     * The new shape must encode the same total element count.
     * Strides are recomputed from the new shape in row-major order.
     *
     * @param new_shape  Target shape.
     * @throws std::invalid_argument  if numel() does not match new_shape product.
     */
    void reshape(std::vector<size_t> new_shape);

    // ── Utility ───────────────────────────────────────────────────────────────

    /**
     * @brief Fill every element with @p value.
     */
    void fill(double value) noexcept;

    /**
     * @brief Zero every element (convenience wrapper around fill(0.0)).
     */
    void zero() noexcept { fill(0.0); }

    /**
     * @brief Print a human-readable representation to @p os.
     *
     * Format (example for shape [2,3]):
     *
     *   Tensor shape=[2, 3]  strides=[3, 1]  numel=6
     *   [ 1.000  2.000  3.000 ]
     *   [ 4.000  5.000  6.000 ]
     *
     * For tensors with ndim > 2, slices along the last two dims are printed
     * with a header identifying the outer indices.
     *
     * @param os       Output stream (defaults to std::cout in the .cpp helper).
     * @param precision  Number of decimal places (default 3).
     */
    void print(std::ostream& os, int precision = 3) const;

    /** Convenience overload — prints to std::cout. */
    void print(int precision = 3) const;

    /**
     * @brief Return a compact string describing shape, e.g. "[2, 3, 4]".
     */
    [[nodiscard]] std::string shape_str() const;

private:
    // ── Internal Helpers ──────────────────────────────────────────────────────

    /**
     * @brief Compute row-major strides from the current shape_ and store
     *        them in strides_.
     *
     * Row-major (C-order) stride formula:
     *   strides_[ndim-1] = 1
     *   strides_[i]      = strides_[i+1] * shape_[i+1]   for i < ndim-1
     */
    void compute_strides();

    /**
     * @brief Compute the flat 1-D index from an N-d index vector after
     *        validating that indices are in range.
     *
     * Separated from at() so both const and non-const overloads share logic.
     *
     * @throws std::invalid_argument  if indices.size() != ndim().
     * @throws std::out_of_range      if indices[i] >= shape_[i] for any i.
     */
    [[nodiscard]] size_t flat_index(const std::vector<size_t>& indices) const;

    // ── Data Members ──────────────────────────────────────────────────────────

    std::vector<size_t> shape_;    ///< Dimension sizes, e.g. {2, 3, 4}
    std::vector<size_t> strides_;  ///< Row-major strides in element units
    std::vector<double> data_;     ///< Flat contiguous storage (row-major)
};

}  // namespace engine
