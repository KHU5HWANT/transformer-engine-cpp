/**
 * @file    engine/ops.cpp
 * @brief   Implementation of all differentiable tensor operations.
 *
 * See ops.hpp for the full API documentation, backward derivations, and the
 * autograd contract every op must satisfy.
 *
 * CRITICAL — backward lambda cycle safety
 * ──────────────────────────────────────
 * Every backward lambda captures the output node as std::weak_ptr<Node>
 * (named `wout`), NOT as shared_ptr.  This breaks the self-reference cycle:
 *
 *   Node C owns _backward lambda  →  lambda captures shared_ptr{C}  →  CYCLE
 *
 * With weak_ptr the lambda does NOT keep Node C alive.  When the topological
 * sort (Step 1.4) runs backward(), it holds shared_ptr to all nodes in a
 * local vector, so wout.lock() is guaranteed to succeed during that window.
 * After backward(), the vector is released and all intermediate nodes free
 * automatically — zero leaks.
 *
 * Build flags required:
 *   g++ -std=c++17 -O2 -fopenmp engine/tensor.cpp engine/node.cpp engine/ops.cpp
 */

#include "ops.hpp"

#include <algorithm>    // std::min
#include <cmath>        // std::exp, std::log
#include <numeric>      // std::accumulate
#include <stdexcept>    // std::invalid_argument
#include <string>
#include <vector>

namespace engine::ops {

// ═════════════════════════════════════════════════════════════════════════════
// File-local helpers  (anonymous namespace — not visible outside this TU)
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// ── Tiling constant ───────────────────────────────────────────────────────────
// BLOCK_SIZE = 64 doubles × 8 bytes = 512 bytes per row of a tile.
// A single tile (64×64) = 32 KB — fits comfortably inside a 32–64 KB L1 cache.
static constexpr size_t BLOCK_SIZE = 64;

// ─────────────────────────────────────────────────────────────────────────────
// mm_nn  :  C[M,N] += A[M,K] @ B[K,N]           (forward matmul)
// ─────────────────────────────────────────────────────────────────────────────
// Algorithm: cache-oblivious tiled GEMM.
//  Outer tile loops enumerate (ii, jj, kk) output / contraction blocks.
//  Inner triple loop processes each (BLOCK_SIZE³) tile with:
//    - register reuse: `a_ik` hoisted out of the j-loop.
//    - sequential j-access of B: maximises hardware prefetch hits.
//
// Parallelism: `omp parallel for` on the outermost tile loop (ii).
//  Each thread owns complete rows of output tiles → no write conflicts on C.
//
void mm_nn(const double* A, const double* B, double* C,
           size_t M, size_t K, size_t N)
{
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                const size_t imax = std::min(ii + BLOCK_SIZE, M);
                const size_t jmax = std::min(jj + BLOCK_SIZE, N);
                const size_t kmax = std::min(kk + BLOCK_SIZE, K);

                for (size_t i = ii; i < imax; ++i) {
                    for (size_t k = kk; k < kmax; ++k) {
                        const double a_ik = A[i * K + k];
                        for (size_t j = jj; j < jmax; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// mm_nt  :  C[M,K] += A[M,N] @ Bᵀ   (B stored as [K,N])
// ─────────────────────────────────────────────────────────────────────────────
// Used in matmul backward for ∂L/∂A:
//
//   ∂L/∂A[M,K] = ∂L/∂out[M,N] @ B[K,N]ᵀ
//   C[i,j]     += Σₙ  A[i,n] · B[j,n]
//
// i ∈ [0,M)  j ∈ [0,K)  contraction over n ∈ [0,N)
// Thread safety: parallelism on ii → each thread owns distinct C rows.
//
void mm_nt(const double* A, const double* B, double* C,
           size_t M, size_t N, size_t K)
{
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < K; jj += BLOCK_SIZE) {
            for (size_t nn = 0; nn < N; nn += BLOCK_SIZE) {
                const size_t imax = std::min(ii + BLOCK_SIZE, M);
                const size_t jmax = std::min(jj + BLOCK_SIZE, K);
                const size_t nmax = std::min(nn + BLOCK_SIZE, N);

                for (size_t i = ii; i < imax; ++i) {
                    for (size_t n = nn; n < nmax; ++n) {
                        const double a_in = A[i * N + n];
                        for (size_t j = jj; j < jmax; ++j) {
                            // B[j,n] is the (j,n) element of B[K,N]
                            C[i * K + j] += a_in * B[j * N + n];
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// mm_tn  :  C[K,N] += Aᵀ @ B   (A stored as [M,K], B as [M,N])
// ─────────────────────────────────────────────────────────────────────────────
// Used in matmul backward for ∂L/∂B:
//
//   ∂L/∂B[K,N] = A[M,K]ᵀ @ ∂L/∂out[M,N]
//   C[j,l]     += Σᵢ  A[i,j] · B[i,l]
//
// j ∈ [0,K)  l ∈ [0,N)  contraction over i ∈ [0,M)
// Thread safety: parallelism on jj → each thread owns distinct C row-blocks.
//
void mm_tn(const double* A, const double* B, double* C,
           size_t M, size_t K, size_t N)
{
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t jj = 0; jj < K; jj += BLOCK_SIZE) {
        for (size_t ll = 0; ll < N; ll += BLOCK_SIZE) {
            for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
                const size_t jmax = std::min(jj + BLOCK_SIZE, K);
                const size_t lmax = std::min(ll + BLOCK_SIZE, N);
                const size_t imax = std::min(ii + BLOCK_SIZE, M);

                for (size_t i = ii; i < imax; ++i) {
                    for (size_t j = jj; j < jmax; ++j) {
                        // A^T[j,i] = A[i,j] stored as A[i*K + j]
                        const double a_ij = A[i * K + j];
                        for (size_t l = ll; l < lmax; ++l) {
                            C[j * N + l] += a_ij * B[i * N + l];
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// raw_transpose_2d  :  dst[N,M] = src[M,N]
// ─────────────────────────────────────────────────────────────────────────────
// Pure data rearrangement — no graph nodes, no gradients, no allocation.
// Used in the transpose op's forward pass and its backward lambda.
//
void raw_transpose_2d(const double* src, double* dst, size_t M, size_t N)
{
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: product of a shape vector
// ─────────────────────────────────────────────────────────────────────────────
size_t shape_product(const std::vector<size_t>& shape) {
    size_t p = 1;
    for (size_t d : shape) p *= d;
    return p;
}

}  // anonymous namespace


// ═════════════════════════════════════════════════════════════════════════════
// engine::ops  —  public API implementations
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// add
// ─────────────────────────────────────────────────────────────────────────────

NodePtr add(const NodePtr& a, const NodePtr& b)
{
    if (a->data.shape() != b->data.shape()) {
        throw std::invalid_argument(
            "ops::add: shape mismatch — a=" + a->data.shape_str() +
            "  b=" + b->data.shape_str());
    }

    const size_t n = a->data.numel();
    const double* ad = a->data.data_ptr();
    const double* bd = b->data.data_ptr();

    // ── Forward ──────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) fwd[i] = ad[i] + bd[i];

    auto out = Node::make(
        Tensor(a->data.shape(), std::move(fwd)),
        a->requires_grad || b->requires_grad
    );

    // ── DAG edges ────────────────────────────────────────────────────────────
    out->add_child(a);
    out->add_child(b);

    // ── Backward  ∂L/∂a = ∂L/∂out,  ∂L/∂b = ∂L/∂out  (both are identity) ──
    // `a` and `b` captured by value (shared_ptr) — keeps inputs alive.
    // `wout` (weak_ptr) breaks the Node → lambda → Node self-cycle.
    out->_backward = [a, b, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;                  // safety: should never be null during backward
        // add is a linear op: local gradient w.r.t. each input is 1.0
        // ∴ upstream gradient passes through unchanged.
        a->accumulate_grad(self->grad);
        b->accumulate_grad(self->grad);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// mul  (Hadamard / elementwise product)
// ─────────────────────────────────────────────────────────────────────────────

NodePtr mul(const NodePtr& a, const NodePtr& b)
{
    if (a->data.shape() != b->data.shape()) {
        throw std::invalid_argument(
            "ops::mul: shape mismatch — a=" + a->data.shape_str() +
            "  b=" + b->data.shape_str());
    }

    const size_t n = a->data.numel();
    const double* ad = a->data.data_ptr();
    const double* bd = b->data.data_ptr();

    // ── Forward ──────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) fwd[i] = ad[i] * bd[i];

    auto out = Node::make(
        Tensor(a->data.shape(), std::move(fwd)),
        a->requires_grad || b->requires_grad
    );

    out->add_child(a);
    out->add_child(b);

    // ── Backward ─────────────────────────────────────────────────────────────
    // ∂L/∂a[i] = ∂L/∂out[i] · b[i]      (b is the "other factor")
    // ∂L/∂b[i] = ∂L/∂out[i] · a[i]
    out->_backward = [a, b, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const size_t  n        = self->grad.numel();
        const double* dout     = self->grad.data_ptr();
        const double* a_data   = a->data.data_ptr();
        const double* b_data   = b->data.data_ptr();

        // ∂L/∂a
        {
            Tensor da(a->data.shape());
            double* dp = da.data_ptr();
            for (size_t i = 0; i < n; ++i) dp[i] = dout[i] * b_data[i];
            a->accumulate_grad(da);
        }
        // ∂L/∂b
        {
            Tensor db(b->data.shape());
            double* dp = db.data_ptr();
            for (size_t i = 0; i < n; ++i) dp[i] = dout[i] * a_data[i];
            b->accumulate_grad(db);
        }
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// matmul  (2-D or batched 3-D, cache-blocked + OpenMP)
// ─────────────────────────────────────────────────────────────────────────────

NodePtr matmul(const NodePtr& a, const NodePtr& b)
{
    const size_t ndim_a = a->data.ndim();
    const size_t ndim_b = b->data.ndim();

    // ── Shape validation ─────────────────────────────────────────────────────
    if (!((ndim_a == 2 && ndim_b == 2) ||
          (ndim_a == 3 && ndim_b == 3) ||
          (ndim_a == 3 && ndim_b == 2))) {
        throw std::invalid_argument(
            "ops::matmul: unsupported ranks (requires 2Dx2D, 3Dx3D, or 3Dx2D); got " +
            a->data.shape_str() + " and " + b->data.shape_str());
    }

    const size_t M = a->data.shape()[ndim_a - 2];
    const size_t K = a->data.shape()[ndim_a - 1];
    const size_t K2 = b->data.shape()[ndim_b - 2];
    const size_t N  = b->data.shape()[ndim_b - 1];

    if (K != K2) {
        throw std::invalid_argument(
            "ops::matmul: inner dimensions mismatch — a=" +
            a->data.shape_str() + "  b=" + b->data.shape_str());
    }

    const size_t batch = (ndim_a == 3) ? a->data.shape()[0] : 1;
    if (ndim_a == 3 && ndim_b == 3 && a->data.shape()[0] != b->data.shape()[0]) {
        throw std::invalid_argument(
            "ops::matmul: batch dimensions mismatch — a=" +
            a->data.shape_str() + "  b=" + b->data.shape_str());
    }

    // ── Output shape ─────────────────────────────────────────────────────────
    std::vector<size_t> out_shape;
    if (ndim_a == 3) out_shape = {batch, M, N};
    else             out_shape = {M, N};

    // ── Forward: C += A @ B  (one slice per batch element) ───────────────────
    Tensor out_data(out_shape);          // zero-initialised; mm_nn uses +=
    const double* ap = a->data.data_ptr();
    const double* bp = b->data.data_ptr();
    double*       cp = out_data.data_ptr();

    for (size_t batch_i = 0; batch_i < batch; ++batch_i) {
        const double* bp_slice = (ndim_b == 3) ? (bp + batch_i * K * N) : bp;
        mm_nn(ap + batch_i * M * K,
              bp_slice,
              cp + batch_i * M * N,
              M, K, N);
    }

    auto out = Node::make(std::move(out_data),
                          a->requires_grad || b->requires_grad);

    out->add_child(a);
    out->add_child(b);

    // ── Backward ─────────────────────────────────────────────────────────────
    // Per batch slice:
    //   ∂L/∂A[b] = ∂L/∂out[b] @ B[b]ᵀ     →  mm_nt
    //   ∂L/∂B[b] = A[b]ᵀ       @ ∂L/∂out[b] →  mm_tn
    out->_backward = [a, b, M, K, N, batch, ndim_b,
                      w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const double* dout = self->grad.data_ptr();
        const double* ap   = a->data.data_ptr();
        const double* bp   = b->data.data_ptr();

        Tensor da(a->data.shape());          // zero-init gradient buffers
        Tensor db(b->data.shape());
        double* dap = da.data_ptr();
        double* dbp = db.data_ptr();

        for (size_t batch_i = 0; batch_i < batch; ++batch_i) {
            const double* dout_b = dout + batch_i * M * N;
            const double* a_b    = ap   + batch_i * M * K;
            const double* b_b    = (ndim_b == 3) ? (bp + batch_i * K * N) : bp;
            double*       da_b   = dap  + batch_i * M * K;
            double*       db_b   = (ndim_b == 3) ? (dbp + batch_i * K * N) : dbp;

            // ∂L/∂A[M,K] += ∂L/∂out[M,N] @ B[K,N]ᵀ
            mm_nt(dout_b, b_b, da_b, M, N, K);

            // ∂L/∂B[K,N] += A[M,K]ᵀ @ ∂L/∂out[M,N]
            mm_tn(a_b, dout_b, db_b, M, K, N);
        }

        a->accumulate_grad(da);
        b->accumulate_grad(db);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// exp  (elementwise natural exponential)
// ─────────────────────────────────────────────────────────────────────────────

NodePtr exp(const NodePtr& a)
{
    const size_t n = a->data.numel();
    const double* ad = a->data.data_ptr();

    // ── Forward ──────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) fwd[i] = std::exp(ad[i]);

    auto out = Node::make(Tensor(a->data.shape(), std::move(fwd)),
                          a->requires_grad);
    out->add_child(a);

    // ── Backward ─────────────────────────────────────────────────────────────
    // d/dx exp(x) = exp(x)  →  local gradient = out->data  (already computed)
    // ∂L/∂a[i] = ∂L/∂out[i] · out[i]
    out->_backward = [a, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const size_t  n     = self->grad.numel();
        const double* dout  = self->grad.data_ptr();
        const double* out_d = self->data.data_ptr();   // out[i] = exp(a[i])

        Tensor da(a->data.shape());
        double* dp = da.data_ptr();
        for (size_t i = 0; i < n; ++i) dp[i] = dout[i] * out_d[i];

        a->accumulate_grad(da);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// log  (elementwise natural logarithm)
// ─────────────────────────────────────────────────────────────────────────────

NodePtr log(const NodePtr& a)
{
    const size_t n = a->data.numel();
    const double* ad = a->data.data_ptr();

    // ── Forward ──────────────────────────────────────────────────────────────
    std::vector<double> fwd(n);
    for (size_t i = 0; i < n; ++i) fwd[i] = std::log(ad[i]);

    auto out = Node::make(Tensor(a->data.shape(), std::move(fwd)),
                          a->requires_grad);
    out->add_child(a);

    // ── Backward ─────────────────────────────────────────────────────────────
    // d/dx ln(x) = 1/x
    // ∂L/∂a[i] = ∂L/∂out[i] / a[i]
    // We capture `a` (not `out`) because we need the *input* value a[i].
    out->_backward = [a, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const size_t  n    = self->grad.numel();
        const double* dout = self->grad.data_ptr();
        const double* ad   = a->data.data_ptr();    // original input values

        Tensor da(a->data.shape());
        double* dp = da.data_ptr();
        for (size_t i = 0; i < n; ++i) dp[i] = dout[i] / ad[i];

        a->accumulate_grad(da);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// sum  (reduce all elements to a scalar)
// ─────────────────────────────────────────────────────────────────────────────

NodePtr sum(const NodePtr& a)
{
    const size_t n = a->data.numel();
    const double* ad = a->data.data_ptr();

    // ── Forward: scalar = Σ a[i] ─────────────────────────────────────────────
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) total += ad[i];

    // Output is a 1-element tensor — the canonical scalar representation.
    auto out = Node::make(Tensor({1}, {total}), a->requires_grad);
    out->add_child(a);

    // ── Backward: broadcast scalar gradient to every element of a ────────────
    // ∂L/∂a[i] = ∂L/∂out   for all i  (since ∂out/∂a[i] = 1 for all i)
    out->_backward = [a, w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        const double g = self->grad.data_ptr()[0];   // scalar upstream gradient

        Tensor da(a->data.shape());
        double* dp = da.data_ptr();
        const size_t n = da.numel();
        for (size_t i = 0; i < n; ++i) dp[i] = g;  // broadcast

        a->accumulate_grad(da);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// transpose  (swap last two dimensions)
// ─────────────────────────────────────────────────────────────────────────────

NodePtr transpose(const NodePtr& a)
{
    const size_t ndim = a->data.ndim();
    if (ndim < 2) {
        throw std::invalid_argument(
            "ops::transpose: input must have >= 2 dimensions; got " +
            a->data.shape_str());
    }

    const size_t M = a->data.shape()[ndim - 2];   // rows before transpose
    const size_t N = a->data.shape()[ndim - 1];   // cols before transpose

    // ── Output shape: swap last two dims ─────────────────────────────────────
    std::vector<size_t> out_shape = a->data.shape();
    out_shape[ndim - 2] = N;
    out_shape[ndim - 1] = M;

    const size_t batch = shape_product(a->data.shape()) / (M * N);

    // ── Forward: rearrange every 2-D slice ───────────────────────────────────
    Tensor out_data(out_shape);
    const double* src = a->data.data_ptr();
    double*       dst = out_data.data_ptr();

    for (size_t b = 0; b < batch; ++b) {
        raw_transpose_2d(src + b * M * N,
                         dst + b * N * M,
                         M, N);
    }

    auto out = Node::make(std::move(out_data), a->requires_grad);
    out->add_child(a);

    // ── Backward ─────────────────────────────────────────────────────────────
    // Transposing is its own inverse:
    //   if out[..., i, j] = a[..., j, i]
    //   then ∂L/∂a[..., j, i] = ∂L/∂out[..., i, j]
    //   → da = transpose(out->grad)   using raw_transpose_2d per slice
    out->_backward = [a, M, N, batch,
                      w_out = std::weak_ptr<Node>(out)]() {
        auto self = w_out.lock();
        if (!self) return;

        // out->grad has shape [..., N, M]; da has shape [..., M, N]
        Tensor da(a->data.shape());
        const double* grad_src = self->grad.data_ptr();
        double*       da_dst   = da.data_ptr();

        for (size_t b = 0; b < batch; ++b) {
            // Transpose each [N, M] gradient slice back to [M, N]
            raw_transpose_2d(grad_src + b * N * M,
                             da_dst   + b * M * N,
                             N, M);
        }

        a->accumulate_grad(da);
    };

    return out;
}

}  // namespace engine::ops
