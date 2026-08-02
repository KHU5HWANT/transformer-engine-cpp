/**
 * @file    engine/tensor.cpp
 * @brief   Implementation of the N-dimensional Tensor class.
 *
 * See tensor.hpp for the full API documentation and design rationale.
 */

#include "tensor.hpp"

#include <algorithm>    // std::accumulate, std::all_of
#include <cassert>
#include <cmath>        // std::abs (used in future grad-check; included early)
#include <iomanip>      // std::setw, std::setprecision, std::fixed
#include <iostream>     // std::cout
#include <numeric>      // std::accumulate
#include <sstream>      // std::ostringstream
#include <stdexcept>    // std::invalid_argument, std::out_of_range

namespace engine {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (file-local)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/**
 * @brief Compute the product of all elements in a size_t vector.
 *        Returns 1 for an empty vector (scalar case).
 */
[[nodiscard]] size_t product(const std::vector<size_t>& v) noexcept {
    return std::accumulate(v.begin(), v.end(),
                           size_t{1},
                           std::multiplies<size_t>{});
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape))
{
    if (shape_.empty()) {
        throw std::invalid_argument(
            "Tensor: shape must have at least one dimension.");
    }
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] == 0) {
            std::cout << "[DEBUG Tensor] dimension " << i << " is 0. Shape: [";
            for(auto s : shape_) std::cout << s << ", ";
            std::cout << "]" << std::endl;
            throw std::invalid_argument(
                "Tensor: dimension " + std::to_string(i) +
                " has size 0; zero-size dimensions are not permitted.");
        }
    }
    compute_strides();
    data_.assign(product(shape_), 0.0);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data)
    : shape_(std::move(shape))
    , data_ (std::move(data))
{
    if (shape_.empty()) {
        throw std::invalid_argument(
            "Tensor: shape must have at least one dimension.");
    }
    const size_t expected = product(shape_);
    if (data_.size() != expected) {
        throw std::invalid_argument(
            "Tensor: data size " + std::to_string(data_.size()) +
            " does not match shape product " + std::to_string(expected) + ".");
    }
    compute_strides();
}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : Tensor(std::vector<size_t>(shape))
{}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

void Tensor::compute_strides() {
    const size_t ndim = shape_.size();
    strides_.resize(ndim);

    // Row-major (C-order): last dimension has stride 1.
    //   strides_[ndim-1] = 1
    //   strides_[i]      = strides_[i+1] * shape_[i+1]
    if (ndim > 0) {
        strides_[ndim - 1] = 1;
        for (size_t i = ndim - 1; i-- > 0; ) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
}

size_t Tensor::flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Tensor::at(): indices rank " + std::to_string(indices.size()) +
            " does not match tensor rank " + std::to_string(shape_.size()) + ".");
    }

    size_t flat = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Tensor::at(): index " + std::to_string(indices[i]) +
                " out of range for dimension " + std::to_string(i) +
                " with size "               + std::to_string(shape_[i]) + ".");
        }
        flat += indices[i] * strides_[i];
    }
    return flat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Element Access
// ─────────────────────────────────────────────────────────────────────────────

double& Tensor::at(const std::vector<size_t>& indices) {
    return data_[flat_index(indices)];
}

const double& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[flat_index(indices)];
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape Manipulation
// ─────────────────────────────────────────────────────────────────────────────

void Tensor::reshape(std::vector<size_t> new_shape) {
    if (new_shape.empty()) {
        throw std::invalid_argument(
            "Tensor::reshape(): new shape must have at least one dimension.");
    }
    const size_t new_numel = product(new_shape);
    if (new_numel != numel()) {
        throw std::invalid_argument(
            "Tensor::reshape(): cannot reshape tensor of size " +
            std::to_string(numel()) + " into shape with size " +
            std::to_string(new_numel) + ".");
    }
    shape_ = std::move(new_shape);
    compute_strides();                 // Recompute strides for the new shape.
    // data_ is unchanged — same flat storage, different view.
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

void Tensor::fill(double value) noexcept {
    std::fill(data_.begin(), data_.end(), value);
}

std::string Tensor::shape_str() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "]";
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// print() — human-readable tensor display
// ─────────────────────────────────────────────────────────────────────────────

void Tensor::print(std::ostream& os, int precision) const {
    // ── Header ────────────────────────────────────────────────────────────────
    os << "Tensor  shape=" << shape_str()
       << "  strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) os << ", ";
        os << strides_[i];
    }
    os << "]  numel=" << numel() << "\n";

    // ── Scalar edge case ─────────────────────────────────────────────────────
    if (ndim() == 1) {
        // Print as a single row.
        os << "[ ";
        for (size_t j = 0; j < shape_[0]; ++j) {
            os << std::fixed << std::setprecision(precision)
               << std::setw(precision + 4) << data_[j];
            if (j + 1 < shape_[0]) os << "  ";
        }
        os << " ]\n";
        return;
    }

    // ── N-dimensional display ─────────────────────────────────────────────────
    // Flatten all outer dimensions, then print the innermost 2-D slice.
    //
    // For a shape [d0, d1, ..., d_{n-2}, d_{n-1}]:
    //   - rows   = shape_[ndim-2]  (or 1 for 1-D, handled above)
    //   - cols   = shape_[ndim-1]
    //   - outer  = numel / (rows * cols)
    //
    // We iterate over "outer" and for each outer index print the 2-D sub-matrix.

    const size_t cols  = shape_[ndim() - 1];
    const size_t rows  = shape_[ndim() - 2];
    const size_t slice = rows * cols;
    const size_t outer = numel() / slice;

    // Compute outer-dimension indices for labelling the slices.
    // We abuse the flat_index machinery by treating the last two dims as one.
    std::vector<size_t> outer_shape(shape_.begin(), shape_.end() - 2);

    for (size_t s = 0; s < outer; ++s) {
        // Print a header for slices only when ndim > 2.
        if (ndim() > 2) {
            // Decode the outer flat index s into outer-dim indices.
            os << "[:, ";
            size_t tmp = s;
            std::vector<size_t> outer_idx(outer_shape.size());
            // Reuse strides of the outer shape (computed inline).
            // We only need this for labelling; compute right-to-left.
            for (size_t d = outer_shape.size(); d-- > 0; ) {
                // Accumulate the product of dimensions to the right of d.
                size_t span = 1;
                for (size_t k = d + 1; k < outer_shape.size(); ++k)
                    span *= outer_shape[k];
                outer_idx[d] = tmp / span;
                tmp %= span;
            }
            for (size_t d = 0; d < outer_idx.size(); ++d) {
                if (d > 0) os << ", ";
                os << outer_idx[d];
            }
            os << ", :, :]\n";
        }

        const size_t base = s * slice;
        for (size_t r = 0; r < rows; ++r) {
            os << "[ ";
            for (size_t c = 0; c < cols; ++c) {
                os << std::fixed << std::setprecision(precision)
                   << std::setw(precision + 4)
                   << data_[base + r * cols + c];
                if (c + 1 < cols) os << "  ";
            }
            os << " ]\n";
        }
        if (outer > 1 && s + 1 < outer) os << "\n";
    }
}

void Tensor::print(int precision) const {
    print(std::cout, precision);
}

}  // namespace engine
