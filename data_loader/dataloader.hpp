/**
 * @file    data_loader/dataloader.hpp
 * @brief   POSIX mmap-based DataLoader for binary token datasets.
 *
 * Design (implementation_plan_v2.md, Step 3.2):
 *
 *  Memory mapping strategy
 *  ─────────────────────────
 *  The .bin file is mapped into the process address space with mmap(MAP_SHARED).
 *  After mapping, random-access to any token is a single pointer dereference —
 *  no fread(), no buffering, no syscall overhead per token.
 *
 *  Page faults are the only cost for cold reads.  When MAP_POPULATE is
 *  available (Linux), the kernel pre-faults all pages at mmap() time,
 *  eliminating page faults entirely for small-to-medium datasets (< RAM).
 *
 *  Hot-swapping (zero-downtime curriculum transitions)
 *  ────────────────────────────────────────────────────
 *  switch_dataset() atomically (from the DataLoader's perspective):
 *    1. munmap() the old file
 *    2. close() the old fd
 *    3. mmap() the new file
 *  The training loop continues without a restart.  The RNG state is preserved,
 *  so the batch-sampling sequence does not reset across transitions.
 *
 *  Batch sampling (causal language modelling)
 *  ───────────────────────────────────────────
 *  Each sample is a (X, Y) pair of length seq_len where:
 *    X = tokens[start   : start + seq_len]
 *    Y = tokens[start+1 : start + seq_len + 1]
 *  Y is the one-step-right-shifted target for next-token prediction.
 *
 *  Start positions are drawn uniformly at random from
 *    [0, num_tokens − seq_len − 1]
 *  to ensure both X and Y stay within bounds.
 *
 *  Thread safety
 *  ─────────────
 *  DataLoader is NOT thread-safe.  The RNG (rng_) is mutable shared state.
 *  If multi-threaded batch prefetching is needed, guard next_batch() with a
 *  std::mutex or create one DataLoader per thread.
 *
 *  Ownership and RAII
 *  ───────────────────
 *  The destructor calls munmap() and close() unconditionally.
 *  DataLoader is move-only (deleted copy constructor/assignment).
 *
 * Build:
 *   g++ -std=c++17 -O2 data_loader/dataloader.cpp
 * Target: Linux/WSL2, POSIX, C++17.
 */

#pragma once

#include <cstddef>     // size_t
#include <cstdint>     // uint16_t, uint32_t, uint64_t
#include <random>      // mt19937, uniform_int_distribution
#include <string>
#include <vector>

namespace data_loader {

// ─────────────────────────────────────────────────────────────────────────────
// Batch
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief A mini-batch for causal language modelling.
 *
 * Both X and Y are flat vectors of length batch_size × seq_len.
 * Index (b, t) maps to flat index b * seq_len + t.
 *
 * Y[b, t] = token immediately following X[b, t] in the source text.
 * The Transformer learns to predict Y[b, t] given X[b, 0..t].
 */
struct Batch {
    std::vector<size_t> X;      ///< Input tokens    — flat [B × T]
    std::vector<size_t> Y;      ///< Target tokens   — flat [B × T],  Y = X shifted by 1
    size_t              batch_size = 0;
    size_t              seq_len    = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// DataLoader
// ─────────────────────────────────────────────────────────────────────────────

class DataLoader {
public:
    // ── Construction / Destruction ────────────────────────────────────────────

    /**
     * @brief Open a .bin dataset file and memory-map its token payload.
     *
     * Validates the 16-byte header (magic = 0xDEADBEEF, vocab_size = 256).
     * On Linux with MAP_POPULATE defined, all pages are pre-faulted so
     * subsequent batch sampling incurs zero page faults.
     *
     * @param path  Path to the .bin file produced by scripts/preprocess.py.
     * @param seed  Seed for the internal Mersenne-Twister RNG.
     *
     * @throws std::runtime_error on any OS error or header validation failure.
     */
    explicit DataLoader(const std::string& path, uint32_t seed = 42);

    /**
     * @brief Destructor — guaranteed munmap() + close(), no leaks.
     */
    ~DataLoader();

    // Move-only: copying a DataLoader would alias fd, mapping, and RNG state.
    DataLoader(DataLoader&&)            = default;
    DataLoader& operator=(DataLoader&&) = default;
    DataLoader(const DataLoader&)       = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    // ── Batch extraction ──────────────────────────────────────────────────────

    /**
     * @brief Sample a mini-batch of (X, Y) pairs at random offsets.
     *
     * Each of the batch_size samples starts at a uniformly random position
     * in [0, num_tokens − seq_len − 1].
     *
     * @param batch_size  Number of sequences per batch (B).
     * @param seq_len     Tokens per sequence (T).  Must satisfy
     *                    T + 1 ≤ num_tokens.
     *
     * @return  Batch with X.size() == Y.size() == batch_size × seq_len.
     *
     * @throws std::runtime_error if the dataset is smaller than seq_len + 1.
     */
    [[nodiscard]] Batch next_batch(size_t batch_size, size_t seq_len);

    // ── Hot-swap ──────────────────────────────────────────────────────────────

    /**
     * @brief Hot-swap to a new .bin dataset file.
     *
     * Called by the training loop when CurriculumScheduler::phase_changed()
     * returns true.  The old mapping is safely released before the new one
     * is established.  The RNG state is NOT reset — batch sampling continues
     * its pseudo-random sequence across the dataset boundary.
     *
     * @param path  Path to the new .bin file.
     *
     * @throws std::runtime_error on any OS error or validation failure.
     */
    void switch_dataset(const std::string& path);

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * @brief Number of uint16 tokens in the currently mapped dataset.
     */
    [[nodiscard]] size_t   num_tokens()  const noexcept { return num_tokens_;  }

    /**
     * @brief Vocabulary size read from the file header (always 256).
     */
    [[nodiscard]] uint32_t vocab_size()  const noexcept { return vocab_size_;  }

    /**
     * @brief Total file size in bytes (header + payload).
     */
    [[nodiscard]] size_t   map_size()    const noexcept { return map_size_;    }

private:
    // ── OS-level mmap state ───────────────────────────────────────────────────
    int            fd_         = -1;             ///< File descriptor (O_RDONLY)
    void*          mapped_     = nullptr;        ///< Base of mapped region (or MAP_FAILED)
    size_t         map_size_   = 0;              ///< Mapped region size in bytes

    const uint16_t* tokens_    = nullptr;        ///< Pointer past the 16-byte header
    size_t          num_tokens_ = 0;             ///< Number of uint16 tokens
    uint32_t        vocab_size_ = 0;             ///< vocab_size from header

    // ── RNG ───────────────────────────────────────────────────────────────────
    std::mt19937   rng_;                         ///< Seeded once; preserved across switch_dataset

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * @brief Open a .bin file, validate header, and mmap the full payload.
     * Called by the constructor and switch_dataset().
     */
    void open_and_mmap(const std::string& path);

    /**
     * @brief Safely unmap and close the current file descriptor.
     * Idempotent: safe to call when fd_ == -1 or mapped_ == MAP_FAILED.
     */
    void close_mmap() noexcept;
};

}  // namespace data_loader
