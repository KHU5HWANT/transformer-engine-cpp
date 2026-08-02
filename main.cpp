/**
 * @file    main.cpp
 * @brief   C++ Deep Learning Engine — Master Curriculum Training Loop.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * PROJECT OVERVIEW  (for recruiters reading this file)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * This file is the entry point of a from-scratch Deep Learning Engine
 * written in pure C++17.  Every component used here was implemented without
 * any machine-learning library (no PyTorch, no Eigen, no Boost):
 *
 *   engine/   — N-dimensional Tensor, reverse-mode autograd DAG
 *   nn/       — Transformer, CausalSelfAttention, LayerNorm, GELU, ...
 *   loss/     — Numerically stable Cross-Entropy (log-sum-exp trick)
 *   optim/    — AdamW with decoupled weight decay & bias correction
 *   data_loader/ — POSIX mmap DataLoader, CurriculumScheduler
 *
 * ════════════════════════════════════════════════════════════════════════════
 * CURRICULUM LEARNING PIPELINE
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Training proceeds across three progressively harder datasets:
 *
 *   Phase 0 — MATH    (data/math.bin)    : Synthetic addition equations
 *   Phase 1 — STORIES (data/stories.bin) : TinyStories natural language
 *   Phase 2 — WIKI    (data/wiki.bin)    : Simple Wikipedia facts
 *
 * On each phase boundary the DataLoader hot-swaps its mmap() pointer from
 * the old .bin file to the new one — model weights are NEVER reset between
 * phases.  A checkpoint is saved just before each transition so training can
 * be resumed from any phase.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * TRAINING LOOP (one step)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   1.  batch = loader.next_batch(B, T)         — mmap random-offset sampling
 *   2.  model.zero_grad()                        — clear all parameter grads
 *   3.  logits = model.forward(batch.X, B, T)   — forward pass, builds DAG
 *   4.  loss   = cross_entropy(logits, batch.Y) — fused NLL + log-sum-exp
 *   5.  engine::backward(loss)                  — reverse DAG traversal
 *   6.  optimizer.step()                         — AdamW parameter update
 *
 * ════════════════════════════════════════════════════════════════════════════
 * CLI USAGE
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   ./transformer \
 *     --d_model     64           \  # embedding dimension
 *     --n_heads     4            \  # attention heads (must divide d_model)
 *     --n_layers    2            \  # stacked TransformerBlocks
 *     --lr          3e-4         \  # initial learning rate
 *     --batch_size  32           \  # sequences per mini-batch
 *     --seq_len     128          \  # tokens per sequence
 *     --phase_steps 5000,10000,20000  # step counts for MATH,STORIES,WIKI phases
 *
 * Quick smoke test (CI-friendly, ~minutes on CPU):
 *   ./transformer --d_model 64 --n_heads 4 --n_layers 2 --phase_steps 100,200,400
 *
 * Build:
 *   g++ -std=c++17 -O2 -fopenmp \
 *       main.cpp engine/*.cpp nn/*.cpp loss/*.cpp optim/*.cpp data_loader/*.cpp \
 *       -o transformer
 *
 * Target: Linux/WSL2, C++17, pure POSIX (no Windows-specific APIs).
 */

// ─── Standard library ────────────────────────────────────────────────────────
#include <algorithm>          // std::min
#include <cassert>
#include <chrono>             // wall-clock timing
#include <cmath>              // std::isfinite
#include <cstddef>            // size_t
#include <cstdlib>            // std::exit
#include <fstream>            // std::ofstream (checkpointing)
#include <iomanip>            // std::setw, std::fixed, std::setprecision
#include <iostream>
#include <sstream>            // std::istringstream (CLI parsing)
#include <stdexcept>
#include <string>
#include <vector>

// ─── Project headers ─────────────────────────────────────────────────────────
#include "engine/autograd.hpp"           // engine::backward()
#include "engine/node.hpp"               // NodePtr
#include "loss/cross_entropy.hpp"        // loss::cross_entropy()
#include "nn/transformer.hpp"            // engine::nn::Transformer
#include "optim/adam.hpp"                // optim::AdamW
#include "data_loader/curriculum.hpp"    // data_loader::CurriculumScheduler
#include "data_loader/dataloader.hpp"    // data_loader::DataLoader

// ═════════════════════════════════════════════════════════════════════════════
// ── Compile-time defaults (all overridable via CLI) ───────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

static constexpr size_t DEFAULT_D_MODEL     = 64;
static constexpr size_t DEFAULT_N_HEADS     = 4;
static constexpr size_t DEFAULT_N_LAYERS    = 2;
static constexpr size_t DEFAULT_BATCH_SIZE  = 32;
static constexpr size_t DEFAULT_SEQ_LEN     = 128;
static constexpr double DEFAULT_LR          = 1e-3;
static constexpr size_t DEFAULT_VOCAB_SIZE  = 256;   // char-level tokeniser

// Default curriculum step thresholds — overridden by --phase_steps
static constexpr size_t DEFAULT_PHASE0_STEPS =  5'000;
static constexpr size_t DEFAULT_PHASE1_STEPS = 15'000;
static constexpr size_t DEFAULT_PHASE2_STEPS = 35'000;

static constexpr size_t LOG_EVERY_N_STEPS   = 100;   // loss print frequency


// ═════════════════════════════════════════════════════════════════════════════
// ── Training Config ───────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief All hyperparameters parsed from the command line.
 *
 * Fields have sensible defaults; unknown flags are ignored with a warning.
 */
struct TrainConfig {
    size_t d_model     = DEFAULT_D_MODEL;
    size_t n_heads     = DEFAULT_N_HEADS;
    size_t n_layers    = DEFAULT_N_LAYERS;
    size_t batch_size  = DEFAULT_BATCH_SIZE;
    size_t seq_len     = DEFAULT_SEQ_LEN;
    double lr          = DEFAULT_LR;

    // Curriculum phase thresholds (cumulative step counts)
    size_t phase0_end  = DEFAULT_PHASE0_STEPS;   // end of MATH phase
    size_t phase1_end  = DEFAULT_PHASE1_STEPS;   // end of STORIES phase
    size_t phase2_end  = DEFAULT_PHASE2_STEPS;   // end of WIKI phase
};


// ═════════════════════════════════════════════════════════════════════════════
// ── CLI Parser ────────────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Parse comma-separated phase step counts from "--phase_steps A,B,C".
 *
 * Expected format: "A,B,C" where A < B < C are positive integers.
 * Example: "5000,15000,35000"
 *
 * @throws std::invalid_argument if fewer than 3 values are provided.
 */
static void parse_phase_steps(const std::string& spec, TrainConfig& cfg)
{
    std::istringstream ss(spec);
    std::string token;
    std::vector<size_t> vals;

    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            vals.push_back(static_cast<size_t>(std::stoul(token)));
        }
    }

    if (vals.size() < 3) {
        throw std::invalid_argument(
            "--phase_steps requires exactly 3 comma-separated values, e.g. "
            "\"5000,15000,35000\".  Got: \"" + spec + "\".");
    }

    cfg.phase0_end = vals[0];
    cfg.phase1_end = vals[1];
    cfg.phase2_end = vals[2];
}

/**
 * @brief Parse argc/argv into a TrainConfig struct.
 *
 * Recognized flags:
 *   --d_model    INT      Embedding / model dimension
 *   --n_heads    INT      Attention heads per block
 *   --n_layers   INT      Number of TransformerBlocks
 *   --batch_size INT      Sequences per mini-batch
 *   --seq_len    INT      Tokens per sequence
 *   --lr         FLOAT    Initial learning rate
 *   --phase_steps A,B,C  Curriculum threshold step counts
 *
 * Unknown flags are printed as warnings and ignored (for forward-compat).
 */
static TrainConfig parse_args(int argc, char** argv)
{
    TrainConfig cfg;

    for (int i = 1; i < argc - 1; ++i) {
        std::string flag(argv[i]);

        if (flag == "--d_model")     { cfg.d_model    = std::stoul(argv[++i]); }
        else if (flag == "--n_heads")     { cfg.n_heads    = std::stoul(argv[++i]); }
        else if (flag == "--n_layers")    { cfg.n_layers   = std::stoul(argv[++i]); }
        else if (flag == "--batch_size")  { cfg.batch_size = std::stoul(argv[++i]); }
        else if (flag == "--seq_len")     { cfg.seq_len    = std::stoul(argv[++i]); }
        else if (flag == "--lr")          { cfg.lr         = std::stod(argv[++i]);  }
        else if (flag == "--phase_steps") { parse_phase_steps(argv[++i], cfg);      }
        else {
            std::cerr << "[WARN] Unknown flag ignored: " << flag << "\n";
        }
    }

    return cfg;
}


// ═════════════════════════════════════════════════════════════════════════════
// ── Config printer ────────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

static void print_config(const TrainConfig& cfg)
{
    const size_t total_params =
        /* tok_emb */  DEFAULT_VOCAB_SIZE  * cfg.d_model  +
        /* pos_emb */  cfg.seq_len         * cfg.d_model  +
        /* N blocks */ cfg.n_layers * (
            /* 4 Linear (Q,K,V,O): 2×W each */ 4 * 2 * cfg.d_model * cfg.d_model +
            /* FFN: W1+b1+W2+b2 */             2 * cfg.d_model * (4 * cfg.d_model) +
            2 * (4 * cfg.d_model) +
            /* 2 LayerNorms: γ+β */            4 * cfg.d_model
        ) +
        /* ln_f */     2 * cfg.d_model +
        /* lm_head */  cfg.d_model * DEFAULT_VOCAB_SIZE;

    std::cout
        << "╔══════════════════════════════════════════════╗\n"
        << "║   C++ Transformer — Curriculum Training      ║\n"
        << "╠══════════════════════════════════════════════╣\n"
        << "║  d_model     = " << std::setw(8) << cfg.d_model
        << "    n_heads  = " << std::setw(3) << cfg.n_heads      << " ║\n"
        << "║  n_layers    = " << std::setw(8) << cfg.n_layers
        << "    seq_len  = " << std::setw(3) << cfg.seq_len       << " ║\n"
        << "║  batch_size  = " << std::setw(8) << cfg.batch_size
        << "    lr       = " << cfg.lr                             << " ║\n"
        << "║  ~params     ≈ " << std::setw(8) << total_params                 << "                 ║\n"
        << "╠══════════════════════════════════════════════╣\n"
        << "║  Phase 0 MATH    steps [0,     " << std::setw(6) << cfg.phase0_end << ")      ║\n"
        << "║  Phase 1 STORIES steps [" << std::setw(6) << cfg.phase0_end
        << ", " << std::setw(6) << cfg.phase1_end << ")      ║\n"
        << "║  Phase 2 WIKI    steps [" << std::setw(6) << cfg.phase1_end
        << ", " << std::setw(6) << cfg.phase2_end << ")      ║\n"
        << "╚══════════════════════════════════════════════╝\n\n";
}


// ═════════════════════════════════════════════════════════════════════════════
// ── Checkpoint helper ─────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Serialize all model parameters to a flat binary file.
 *
 * Format: raw little-endian double values concatenated in the order returned
 * by model.parameters().  No header — weights are uniquely identified by
 * the model architecture, so any mismatch will be obvious when loading.
 *
 * The checkpoint can be loaded back with:
 *   std::ifstream fin(path, std::ios::binary);
 *   for (auto& p : model.parameters())
 *       fin.read(reinterpret_cast<char*>(p->data.data_ptr()),
 *                p->data.numel() * sizeof(double));
 *
 * @param model  The Transformer model whose parameters are written.
 * @param path   File path for the checkpoint (e.g. "ckpt_phase0.bin").
 *
 * @throws std::runtime_error if the file cannot be opened for writing.
 */
static void save_checkpoint(const engine::nn::Transformer& model,
                             const std::string& path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error(
            "save_checkpoint: cannot open '" + path + "' for writing.");
    }

    size_t bytes_written = 0;
    for (const engine::NodePtr& p : model.parameters()) {
        const size_t n     = p->data.numel();
        const char*  begin = reinterpret_cast<const char*>(p->data.data_ptr());
        out.write(begin, static_cast<std::streamsize>(n * sizeof(double)));
        bytes_written += n * sizeof(double);
    }

    out.close();

    std::cout << "  [CKPT] Saved " << bytes_written / 1024
              << " KB → " << path << "\n";
}


// ═════════════════════════════════════════════════════════════════════════════
// ── Phase name helper ─────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

static const char* phase_name(data_loader::DatasetPhase p)
{
    switch (p) {
        case data_loader::DatasetPhase::MATH:    return "MATH";
        case data_loader::DatasetPhase::STORIES: return "STORIES";
        case data_loader::DatasetPhase::WIKI:    return "WIKI";
        default:                                  return "UNKNOWN";
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// ── main ──────────────────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    // ── 0. Parse CLI args and print configuration ─────────────────────────────
    const TrainConfig cfg = parse_args(argc, argv);
    print_config(cfg);

    // ── 1. Curriculum Scheduler ───────────────────────────────────────────────
    //
    // CurriculumScheduler stores the step thresholds and knows which .bin file
    // corresponds to each curriculum phase.
    //
    data_loader::CurriculumScheduler scheduler(
        {cfg.phase0_end, cfg.phase1_end, cfg.phase2_end},
        {"data/math.bin", "data/stories.bin", "data/wiki.bin"}
    );

    std::cout << "[INFO] Curriculum initialized.\n"
              << "       Total training steps: " << scheduler.total_steps() << "\n\n";

    // ── 2. DataLoader (POSIX mmap) ────────────────────────────────────────────
    //
    // Opens the Phase 0 (MATH) binary file and maps it into the process address
    // space.  Batch sampling is a series of pointer dereferences — no syscalls,
    // no copies after the initial mmap().
    //
    data_loader::DataLoader loader(
        scheduler.dataset_path(data_loader::DatasetPhase::MATH),
        /*seed=*/42
    );

    std::cout << "[INFO] DataLoader opened: " << scheduler.dataset_path(data_loader::DatasetPhase::MATH)
              << "\n       Tokens mapped: " << loader.num_tokens() << "\n\n";

    // ── 3. Transformer Model ──────────────────────────────────────────────────
    //
    // GPT-style decoder-only Transformer.  seq_len acts as the context window
    // (positional embedding table has exactly seq_len rows).
    //
    // Architecture:
    //   tok_emb(256, d_model) → pos_emb(seq_len, d_model)
    //   → N × TransformerBlock(d_model, n_heads)  [Pre-LN, Causal Attention, FFN]
    //   → LayerNorm(d_model) → Linear(d_model, 256)  → logits [B, T, 256]
    //
    engine::nn::Transformer model(
        DEFAULT_VOCAB_SIZE,   // vocab_size = 256 (char-level)
        cfg.seq_len,          // context_len = seq_len (positional table size)
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers
    );

    std::cout << "[INFO] Transformer initialized.\n"
              << "       Parameters: " << model.parameters().size() << " tensors\n\n";

    // ── 4. AdamW Optimizer ────────────────────────────────────────────────────
    //
    // AdamW separates weight decay from the adaptive gradient step:
    //   θ ← θ(1 − lr·λ)   (decoupled L2)
    //   θ ← θ − lr · m̂/(√v̂ + ε)
    //
    // This gives uniform regularisation strength across all parameters,
    // unlike vanilla Adam+L2 where high-variance parameters are under-regularised.
    //
    optim::AdamW optimizer(
        model.parameters(),
        cfg.lr,
        /*beta1=*/0.9,
        /*beta2=*/0.999,
        /*eps=*/1e-8,
        /*weight_decay=*/0.01
    );

    std::cout << "[INFO] AdamW optimizer initialized.\n"
              << "       lr=" << cfg.lr
              << "  β₁=0.9  β₂=0.999  ε=1e-8  λ=0.01\n\n";

    // ── 5. Training state ─────────────────────────────────────────────────────

    const size_t total_steps = scheduler.total_steps();
    data_loader::DatasetPhase current_phase = data_loader::DatasetPhase::MATH;

    // Timing for throughput reporting
    auto wall_t0 = std::chrono::steady_clock::now();

    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  TRAINING START\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // ██  MAIN TRAINING LOOP  ██
    // ─────────────────────────────────────────────────────────────────────────
    for (size_t step = 0; step < total_steps; ++step) {

        // ── 5a. Curriculum phase transition ───────────────────────────────────
        //
        // Check if we're crossing a curriculum boundary this step.
        // phase_changed() compares advance(step-1) vs advance(step) — it fires
        // exactly once at the boundary step.
        //
        // On a transition:
        //   1. Save a checkpoint (model weights at end of completed phase)
        //   2. Hot-swap the mmap pointer to the new .bin file
        //   3. The model weights are NOT touched — knowledge transfers forward
        //
        if (step > 0 && scheduler.phase_changed(step - 1, step)) {
            const data_loader::DatasetPhase new_phase = scheduler.advance(step);

            std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
            std::cout << "  CURRICULUM TRANSITION: "
                      << phase_name(current_phase) << " → "
                      << phase_name(new_phase)     << "  (step " << step << ")\n";
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

            // Save a checkpoint for the completed phase
            const std::string ckpt_name =
                "checkpoint_phase" +
                std::to_string(static_cast<int>(current_phase)) + ".bin";
            save_checkpoint(model, ckpt_name);

            // Hot-swap: munmap old .bin, mmap new .bin.  Zero-downtime.
            // The optimizer's moment vectors are preserved — no warm-up needed.
            loader.switch_dataset(scheduler.dataset_path(new_phase));
            current_phase = new_phase;

            std::cout << "  [INFO] Dataset swapped to "
                      << scheduler.dataset_path(new_phase) << "\n"
                      << "         Tokens available: " << loader.num_tokens() << "\n\n";
        }

        // ── 5b. Sample a random mini-batch from the mmap'd dataset ────────────
        //
        // next_batch() picks batch_size random start positions uniformly from
        // [0, num_tokens − seq_len − 1] and reads X/Y windows in O(1).
        // Y is X shifted right by 1: the standard next-token prediction target.
        //
        data_loader::Batch batch = loader.next_batch(cfg.batch_size, cfg.seq_len);

        // ── 5c. Zero all parameter gradients ──────────────────────────────────
        //
        // Gradients ACCUMULATE (+=) during backward() by design, so we must
        // zero them before each new forward pass.  zero_grad() traverses the
        // Module tree and calls Node::zero_grad() on every leaf parameter.
        //
        model.zero_grad();

        // ── 5d. Forward pass — builds the autograd computation DAG ────────────
        //
        // model.forward() returns a NodePtr [B, T, vocab_size] whose _backward
        // lambdas close over all intermediate computation nodes, keeping them
        // alive until backward() completes and the loss NodePtr goes out of scope.
        //
        engine::NodePtr logits = model.forward(batch.X, cfg.batch_size, cfg.seq_len);

        // ── 5e. Loss — fused log-sum-exp Cross-Entropy ────────────────────────
        //
        // cross_entropy() computes:
        //   loss = mean(-log softmax(logits)[bt, targets[bt]])  over all B×T
        //
        // The backward lambda is analytically derived:
        //   ∂loss/∂logits[bt,v] = (softmax[bt,v] − 1{v==target[bt]}) / (B×T)
        //
        // This avoids an explicit V×V Jacobian — saving O(B·T·V²) work.
        //
        engine::NodePtr loss = loss::cross_entropy(logits, batch.Y);

        // Sanity check — NaN/Inf loss indicates a bug (e.g. bad data, lr too high)
        const double loss_val = loss->data.data_ptr()[0];
        if (!std::isfinite(loss_val)) {
            std::cerr << "[ERROR] Loss is " << loss_val
                      << " at step " << step
                      << ". Check lr, data, and weight init. Aborting.\n";
            std::exit(1);
        }

        // ── 5f. Backward pass — reverse DAG traversal ─────────────────────────
        //
        // engine::backward() performs:
        //   1. Seeds loss->grad = 1.0 (∂L/∂L = 1)
        //   2. Topological sort of the computation DAG (DFS post-order)
        //   3. Calls every node's _backward() lambda in reverse topo order
        //      → gradients propagate from loss back to every leaf parameter
        //
        engine::backward(loss);

        // ── 5g. AdamW parameter update ────────────────────────────────────────
        //
        // optimizer.step() reads param->grad for each parameter and performs:
        //   θ ← θ(1 − lr·λ)                     (decoupled weight decay)
        //   m ← β₁m + (1−β₁)g  ;  v ← β₂v + (1−β₂)g²
        //   θ ← θ − lr · m̂/(√v̂ + ε)             (bias-corrected Adam step)
        //
        optimizer.step();

        // ── 5h. Logging ───────────────────────────────────────────────────────
        if (step % LOG_EVERY_N_STEPS == 0) {
            auto now = std::chrono::steady_clock::now();
            const double elapsed_s =
                std::chrono::duration<double>(now - wall_t0).count();
            const double steps_per_sec =
                (step == 0) ? 0.0 : static_cast<double>(step) / elapsed_s;

            std::cout << std::fixed << std::setprecision(4)
                      << "  step=" << std::setw(6) << step
                      << "  phase=" << std::setw(7) << phase_name(current_phase)
                      << "  loss=" << std::setw(8) << loss_val
                      << "  lr=" << std::scientific << std::setprecision(2)
                      << optimizer.lr
                      << std::fixed    << std::setprecision(1)
                      << "  speed=" << std::setw(7) << steps_per_sec << " step/s"
                      << "\n";
        }

    }  // ── end training loop ──────────────────────────────────────────────────

    // ── 6. Final checkpoint ───────────────────────────────────────────────────
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  TRAINING COMPLETE  (" << total_steps << " steps)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    save_checkpoint(model, "checkpoint_final.bin");

    // ── 7. Wall-clock summary ─────────────────────────────────────────────────
    auto wall_t1 = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();

    std::cout << "\n[SUMMARY]\n"
              << "  Total steps  : " << total_steps << "\n"
              << "  Total time   : " << std::fixed << std::setprecision(1)
              << total_seconds << " s\n"
              << "  Avg speed    : " << std::fixed << std::setprecision(2)
              << static_cast<double>(total_steps) / total_seconds << " step/s\n"
              << "  Final ckpt   : checkpoint_final.bin\n\n";

    return 0;
}
