/**
 * @file    nn/attention.cpp
 * @brief   Causal Multi-Head Self-Attention — three file-local ops + forward pass.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Design rationale — why the B×H fold-into-batch trick works
 * ════════════════════════════════════════════════════════════════════════════
 *
 * engine::ops::matmul is defined for 2-D and 3-D tensors:
 *   3-D: out[b, i, j] = Σ_k  a[b, i, k] · b[b, k, j]    (batched)
 *
 * After split_heads_op, Q, K, V all have shape [B×H, T, d_k].  The batch
 * axis is now B×H, so every matmul dispatches correctly to mm_nn across
 * B×H independent [T, d_k] slices.  No 4-D infrastructure is required.
 *
 * The same logic applies to ops::transpose — it swaps the last two dims
 * across all leading batch dimensions, so [B×H, T, d_k] → [B×H, d_k, T].
 *
 * ════════════════════════════════════════════════════════════════════════════
 * File-local custom op summary
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  ┌─────────────────────────────────┬────────────────────────────────────┐
 *  │  Op                             │  Forward / Backward                 │
 *  ├─────────────────────────────────┼────────────────────────────────────┤
 *  │  split_heads_op                 │  Permute [B,T,H·dk] → [B·H,T,dk]  │
 *  │  ([B,T,D] → [B·H,T,dk])        │  ∂: inverse permute (= merge_heads)│
 *  ├─────────────────────────────────┼────────────────────────────────────┤
 *  │  merge_heads_op                 │  Inverse permute [B·H,T,dk]→[B,T,D]│
 *  │  ([B·H,T,dk] → [B,T,D])        │  ∂: = split_heads permutation      │
 *  ├─────────────────────────────────┼────────────────────────────────────┤
 *  │  scaled_causal_softmax_op       │  scale → causal mask → softmax     │
 *  │  ([BH,T,T] → [BH,T,T])         │  ∂: JVP * scale; masked pos = 0    │
 *  └─────────────────────────────────┴────────────────────────────────────┘
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Causal mask implementation
 * ════════════════════════════════════════════════════════════════════════════
 *
 * In scaled_causal_softmax_op, for each row i of the attention matrix:
 *   - Only positions j ≤ i are considered for max computation (avoiding
 *     -∞ from polluting the max-subtract trick)
 *   - Positions j > i are directly written as 0.0 in the output (the
 *     IEEE 754 double exp(-∞) underflows to exactly 0)
 *   - Their backward gradient is explicitly set to 0.0 in the lambda:
 *       if (j > i) ds_row[j] = 0.0;
 *   This makes the gradient semantics precise rather than relying on
 *   ~0 from Y[masked] * (anything).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Softmax backward inside scaled_causal_softmax_op
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Let Y = output of this op (attention weights), scale = 1/√d_k.
 *
 * For unmasked position (j ≤ i):
 *   dX_raw[i,j] = scale · Y[i,j] · (dY[i,j] − dot[i])
 *   where  dot[i] = Σ_{k≤i} dY[i,k] · Y[i,k]
 *
 * Chain rule: the scale factor enters because:
 *   score_scaled = score_raw × scale
 *   softmax input = score_scaled  (+ mask constant for j>i)
 *   ∂L/∂score_raw = ∂L/∂score_scaled × scale
 */

#include "nn/attention.hpp"
#include "engine/ops.hpp"    // ops::matmul, ops::transpose

#include <algorithm>         // std::max_element
#include <cmath>             // std::exp, std::sqrt
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::nn {

// ═════════════════════════════════════════════════════════════════════════════
// File-local helper ops  (anonymous namespace — not visible outside this TU)
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// split_heads_op
// ─────────────────────────────────────────────────────────────────────────────
//
// Forward permutation
//   Input x: [B, T, D]   where D = H × dk
//   Output:  [B·H, T, dk]
//
// Index mapping (forward):
//   out[(b·H + h)·T·dk + t·dk + d]  =  x[b·T·D + t·D + h·dk + d]
//
// Backward is the inverse permutation (= merge_heads layout):
//   dx[b·T·D + t·D + h·dk + d]  =  dout[(b·H + h)·T·dk + t·dk + d]
//
static NodePtr split_heads_op(const NodePtr& x,
                               size_t B, size_t T, size_t H, size_t dk)
{
    const size_t D = H * dk;   // d_model

    // ── Validate ──────────────────────────────────────────────────────────────
    if (x->data.numel() != B * T * D) {
        throw std::invalid_argument(
            "split_heads_op: expected " + std::to_string(B*T*D) +
            " elements, got " + std::to_string(x->data.numel()));
    }

    const double* xd = x->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(B * H * T * dk);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t out_batch = b * H + h;         // combined batch index
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < dk; ++d) {
                    fwd[out_batch * T * dk + t * dk + d] =
                        xd[b * T * D + t * D + h * dk + d];
                }
            }
        }
    }

    auto out = Node::make(Tensor({B * H, T, dk}, std::move(fwd)),
                          x->requires_grad);
    out->add_child(x);

    // ── Backward: inverse permute ─────────────────────────────────────────────
    out->_backward = [x, B, T, H, dk, D,
                      w_out = std::weak_ptr<Node>(out)]()
    {
        auto self = w_out.lock();
        if (!self) return;

        const double* dout = self->grad.data_ptr();
        Tensor dx(x->data.shape());                // zero-initialised
        double* dxp = dx.data_ptr();

        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H; ++h) {
                const size_t out_batch = b * H + h;
                for (size_t t = 0; t < T; ++t) {
                    for (size_t d = 0; d < dk; ++d) {
                        dxp[b * T * D + t * D + h * dk + d] =
                            dout[out_batch * T * dk + t * dk + d];
                    }
                }
            }
        }

        x->accumulate_grad(dx);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// merge_heads_op
// ─────────────────────────────────────────────────────────────────────────────
//
// Forward permutation (inverse of split_heads_op)
//   Input x: [B·H, T, dk]
//   Output:  [B, T, D]    where D = H × dk
//
// Index mapping (forward):
//   out[b·T·D + t·D + h·dk + d]  =  x[(b·H + h)·T·dk + t·dk + d]
//
// Backward is split_heads permutation:
//   dx[(b·H + h)·T·dk + t·dk + d]  =  dout[b·T·D + t·D + h·dk + d]
//
static NodePtr merge_heads_op(const NodePtr& x,
                               size_t B, size_t T, size_t H, size_t dk)
{
    const size_t D = H * dk;   // d_model

    // ── Validate ──────────────────────────────────────────────────────────────
    if (x->data.numel() != B * H * T * dk) {
        throw std::invalid_argument(
            "merge_heads_op: expected " + std::to_string(B*H*T*dk) +
            " elements, got " + std::to_string(x->data.numel()));
    }

    const double* xd = x->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(B * T * D);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t in_batch = b * H + h;
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < dk; ++d) {
                    fwd[b * T * D + t * D + h * dk + d] =
                        xd[in_batch * T * dk + t * dk + d];
                }
            }
        }
    }

    auto out = Node::make(Tensor({B, T, D}, std::move(fwd)),
                          x->requires_grad);
    out->add_child(x);

    // ── Backward: split_heads permutation ─────────────────────────────────────
    out->_backward = [x, B, T, H, dk, D,
                      w_out = std::weak_ptr<Node>(out)]()
    {
        auto self = w_out.lock();
        if (!self) return;

        const double* dout = self->grad.data_ptr();
        Tensor dx(x->data.shape());                // zero-initialised
        double* dxp = dx.data_ptr();

        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H; ++h) {
                const size_t in_batch = b * H + h;
                for (size_t t = 0; t < T; ++t) {
                    for (size_t d = 0; d < dk; ++d) {
                        dxp[in_batch * T * dk + t * dk + d] =
                            dout[b * T * D + t * D + h * dk + d];
                    }
                }
            }
        }

        x->accumulate_grad(dx);
    };

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// scaled_causal_softmax_op
// ─────────────────────────────────────────────────────────────────────────────
//
// Fuses three operations into a single Node for efficiency:
//   1. Scale raw scores by `scale`  (= 1/√d_k in the caller)
//   2. Apply causal mask:  positions j > i are held at exactly 0.0
//   3. Apply row-wise numerically-stable softmax (max-subtract trick)
//
// Input:  scores [BH, T, T]   (raw attention logits)
// Output: attn   [BH, T, T]   (attention weights, sum to 1 per row)
//
// Backward
// ─────────
// The backward is the standard softmax Jacobian-vector product, with the
// scale factor chained in, and masked positions receiving zero gradient:
//
//   dot[i]     = Σ_j dY[i,j] · Y[i,j]            (expectation of dY under Y)
//
//   dX_raw[i,j] = scale · Y[i,j] · (dY[i,j] − dot[i])   if j ≤ i  (unmasked)
//   dX_raw[i,j] = 0                                        if j > i  (masked)
//
// The forward output Y is read from self->data — no additional saved buffer.
//
static NodePtr scaled_causal_softmax_op(const NodePtr& scores,
                                         double         scale,
                                         size_t         BH,
                                         size_t         T)
{
    if (scores->data.numel() != BH * T * T) {
        throw std::invalid_argument(
            "scaled_causal_softmax_op: expected shape [" +
            std::to_string(BH) + ", " + std::to_string(T) + ", " +
            std::to_string(T) + "] but got " + scores->data.shape_str());
    }

    const double* sd = scores->data.data_ptr();

    // ── Forward ───────────────────────────────────────────────────────────────
    std::vector<double> fwd(BH * T * T, 0.0);  // masked positions stay 0.0

    for (size_t bh = 0; bh < BH; ++bh) {
        const double* bh_src = sd   + bh * T * T;
        double*       bh_dst = fwd.data() + bh * T * T;

        for (size_t i = 0; i < T; ++i) {
            const double* row_src = bh_src + i * T;
            double*       row_dst = bh_dst + i * T;

            // ── Step 1: max over UNMASKED positions only (j ≤ i) ─────────────
            // Positions j > i are masked to -inf; including them in the max
            // computation would give max_val = 0 (if all scores are negative)
            // and would be subtracted away anyway.  Scanning only j≤i keeps
            // the max numerically meaningful and avoids unnecessary comparisons.
            double max_val = -1e18;
            for (size_t j = 0; j <= i; ++j) {
                double v = row_src[j] * scale;
                if (v > max_val) max_val = v;
            }

            // ── Step 2: exp of unmasked positions ─────────────────────────────
            double sum = 0.0;
            for (size_t j = 0; j <= i; ++j) {
                double e = std::exp(row_src[j] * scale - max_val);
                row_dst[j] = e;
                sum += e;
            }
            // Positions j > i remain 0.0 (masked) — no exp needed.

            // ── Step 3: normalise unmasked positions ──────────────────────────
            for (size_t j = 0; j <= i; ++j) {
                row_dst[j] /= sum;
            }
        }
    }

    auto out = Node::make(Tensor({BH, T, T}, std::move(fwd)),
                          scores->requires_grad);
    out->add_child(scores);

    // ── Backward ──────────────────────────────────────────────────────────────
    //
    // Reads self->data (Y) directly — no separate buffer captured.
    // self->data is immutable after Node construction and self is kept alive
    // by the topological-sort vector during backward().
    //
    out->_backward = [scores, scale, BH, T,
                      w_out = std::weak_ptr<Node>(out)]()
    {
        auto self = w_out.lock();
        if (!self) return;

        const double* dY    = self->grad.data_ptr();
        const double* Y     = self->data.data_ptr();   // forward output (attn weights)

        Tensor dscores(scores->data.shape());          // zero-initialised
        double* dsp = dscores.data_ptr();

        for (size_t bh = 0; bh < BH; ++bh) {
            for (size_t i = 0; i < T; ++i) {
                const double* dY_row = dY  + bh * T * T + i * T;
                const double* Y_row  = Y   + bh * T * T + i * T;
                double*       ds_row = dsp + bh * T * T + i * T;

                // ── dot[i] = Σ_j dY[i,j] · Y[i,j] ──────────────────────────
                // Y[i, j>i] = 0 so only j≤i contributes, but we loop all T for
                // clarity (multiplying by zero is safe and branch-prediction
                // friendly on unmasked-heavy attention maps).
                double dot = 0.0;
                for (size_t j = 0; j < T; ++j) {
                    dot += dY_row[j] * Y_row[j];
                }

                // ── Gradient ─────────────────────────────────────────────────
                for (size_t j = 0; j < T; ++j) {
                    if (j > i) {
                        // Masked position: ∂L/∂score_raw[i,j] = 0.
                        // The mask is a constant, not a function of the input.
                        ds_row[j] = 0.0;
                    } else {
                        // Unmasked: chain through softmax JVP × scale factor.
                        //   ∂L/∂score_raw[i,j]
                        //   = ∂L/∂score_scaled[i,j] × scale
                        //   = Y[i,j] · (dY[i,j] − dot[i]) × scale
                        ds_row[j] = scale * Y_row[j] * (dY_row[j] - dot);
                    }
                }
            }
        }

        scores->accumulate_grad(dscores);
    };

    return out;
}

}  // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════

CausalSelfAttention::CausalSelfAttention(size_t d_model, size_t n_heads)
    : d_model_(d_model)
    , n_heads_(n_heads)
    , d_k_   (d_model / n_heads)
    //
    // Four Linear projections — all [d_model → d_model], Xavier-initialised.
    //
    // Standard practice (GPT-2 config) uses bias=true on all four.
    // The Q and K projections produce identical shapes; keeping them as
    // separate modules avoids accidental weight sharing and simplifies
    // per-head gradient inspection.
    //
    , w_q    (d_model, d_model, /*use_bias=*/true)
    , w_k    (d_model, d_model, /*use_bias=*/true)
    , w_v    (d_model, d_model, /*use_bias=*/true)
    , w_o    (d_model, d_model, /*use_bias=*/true)
{
    // ── Validation ────────────────────────────────────────────────────────────
    // Checked AFTER initialiser list to avoid guard-helper boilerplate.
    // If n_heads==0 the integer division above is UB; catch it explicitly.
    if (n_heads == 0) {
        throw std::invalid_argument(
            "CausalSelfAttention: n_heads must be > 0.");
    }
    if (d_model % n_heads != 0) {
        throw std::invalid_argument(
            "CausalSelfAttention: d_model (" + std::to_string(d_model) +
            ") must be divisible by n_heads (" + std::to_string(n_heads) + "). "
            "Got remainder " + std::to_string(d_model % n_heads) + ".");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// forward
// ═════════════════════════════════════════════════════════════════════════════

NodePtr CausalSelfAttention::forward(const NodePtr& x) const
{
    // ── Input validation ──────────────────────────────────────────────────────
    if (x->data.ndim() != 3) {
        throw std::invalid_argument(
            "CausalSelfAttention::forward: expected 3-D input [B, T, d_model], "
            "got shape " + x->data.shape_str());
    }
    if (x->data.shape()[2] != d_model_) {
        throw std::invalid_argument(
            "CausalSelfAttention::forward: last dim (" +
            std::to_string(x->data.shape()[2]) +
            ") != d_model (" + std::to_string(d_model_) + ").");
    }

    const size_t B  = x->data.shape()[0];   // batch size
    const size_t T  = x->data.shape()[1];   // sequence length
    const size_t BH = B * n_heads_;         // combined batch+head axis

    // ─────────────────────────────────────────────────────────────────────────
    // Step 1–3: Q, K, V projections — [B, T, d_model]
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Linear::forward runs y = x @ Wᵀ + b and registers x → W in the DAG.
    // All three share the same input node x, so x will have three downstream
    // consumers — the autograd engine handles fan-out correctly via
    // accumulate_grad().
    //
    auto Q = w_q.forward(x);    // [B, T, d_model]
    auto K = w_k.forward(x);    // [B, T, d_model]
    auto V = w_v.forward(x);    // [B, T, d_model]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 4–6: Split into heads — [B, T, d_model] → [B×H, T, d_k]
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Permutation: x[b, t, h·dk + d] → out[b·H + h, t, d]
    // After this, B×H acts as the "batch" dimension for all downstream ops,
    // allowing ops::matmul and ops::transpose to operate correctly.
    //
    auto Qh  = split_heads_op(Q, B, T, n_heads_, d_k_);   // [B·H, T, d_k]
    auto Kh  = split_heads_op(K, B, T, n_heads_, d_k_);   // [B·H, T, d_k]
    auto Vh  = split_heads_op(V, B, T, n_heads_, d_k_);   // [B·H, T, d_k]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 7: Transpose K — [B×H, T, d_k] → [B×H, d_k, T]
    // ─────────────────────────────────────────────────────────────────────────
    //
    // ops::transpose swaps the last two dimensions across every leading batch
    // element.  With shape [B·H, T, d_k] this gives [B·H, d_k, T].
    //
    auto KhT = ops::transpose(Kh);                         // [B·H, d_k, T]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 8: Attention scores — [B×H, T, d_k] @ [B×H, d_k, T] = [B×H, T, T]
    // ─────────────────────────────────────────────────────────────────────────
    //
    // ops::matmul detects 3-D inputs and loops over the batch axis (B×H).
    // Each slice is a [T, d_k] @ [d_k, T] = [T, T] matrix multiply.
    //
    auto scores = ops::matmul(Qh, KhT);                   // [B·H, T, T]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 9: Scale + causal mask + softmax — fused single Node
    // ─────────────────────────────────────────────────────────────────────────
    //
    // The scale 1/√d_k is folded into scaled_causal_softmax_op.
    // The causal mask zeroes positions j > i in each row (upper triangle).
    // Numerically stable softmax uses the max-subtract trick per row.
    // The backward correctly propagates zero gradient to masked positions.
    //
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_k_));
    auto attn = scaled_causal_softmax_op(scores, scale, BH, T);   // [B·H, T, T]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 10: Context vectors — [B×H, T, T] @ [B×H, T, d_k] = [B×H, T, d_k]
    // ─────────────────────────────────────────────────────────────────────────
    auto ctx = ops::matmul(attn, Vh);                      // [B·H, T, d_k]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 11: Merge heads — [B×H, T, d_k] → [B, T, d_model]
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Inverse permutation of split_heads_op:
    //   x[b·H + h, t, d] → out[b, t, h·dk + d]
    //
    auto ctx_m = merge_heads_op(ctx, B, T, n_heads_, d_k_);  // [B, T, d_model]

    // ─────────────────────────────────────────────────────────────────────────
    // Step 12: Output projection — [B, T, d_model]
    // ─────────────────────────────────────────────────────────────────────────
    return w_o.forward(ctx_m);                             // [B, T, d_model]
}

// ═════════════════════════════════════════════════════════════════════════════
// parameters
// ═════════════════════════════════════════════════════════════════════════════

std::vector<NodePtr> CausalSelfAttention::parameters() const
{
    // Collect from all four sub-modules in a defined order.
    // With use_bias=true each Linear contributes {weight, bias} → 8 total.
    auto p = w_q.parameters();
    auto k = w_k.parameters();
    auto v = w_v.parameters();
    auto o = w_o.parameters();

    p.reserve(p.size() + k.size() + v.size() + o.size());
    p.insert(p.end(), k.begin(), k.end());
    p.insert(p.end(), v.begin(), v.end());
    p.insert(p.end(), o.begin(), o.end());

    return p;
}

}  // namespace engine::nn
