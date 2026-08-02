/**
 * @file    data_loader/dataloader.cpp
 * @brief   POSIX mmap DataLoader — OS-level implementation.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * mmap vs read() — why we use memory mapping
 * ════════════════════════════════════════════════════════════════════════════
 *
 * With read() or fread(), every batch requires:
 *   1. A syscall (read) into kernel space
 *   2. A kernel→userspace copy into a malloc'd buffer
 *   3. Manual buffer management and seeking
 *
 * With mmap():
 *   1. The OS maps the file directly into our virtual address space
 *   2. Token access is a single pointer dereference (memory load)
 *   3. The page cache IS our "buffer" — zero copies, no syscall per token
 *
 * For a 500 K-token math dataset:
 *   Payload: 500,000 × 2 bytes = 1 MB.
 *   On a system with 8 GB RAM, the entire file fits in the page cache
 *   after the first epoch.  Subsequent epochs run entirely from cache —
 *   training throughput becomes CPU-bound, not I/O bound.
 *
 * MAP_POPULATE (Linux extension):
 *   Instructs the kernel to fault in all pages immediately at mmap() time.
 *   Trades a one-time startup cost for zero page faults during training.
 *   Wrapped in #ifdef so the code compiles on non-Linux POSIX systems.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * File header layout  (binary, little-endian, 16 bytes total)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   Offset  Size  Type     Value
 *   0       4     uint32   0xDEADBEEF  (magic — corrupt file detection)
 *   4       4     uint32   256         (vocab_size — format version check)
 *   8       8     uint64   N           (num_tokens — payload length)
 *
 * After the 16-byte header: N × uint16_t token IDs (little-endian).
 *
 * The header is validated BEFORE any data is accessed.  A wrong magic number
 * throws immediately rather than silently training on garbage data.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * switch_dataset — hot-swap without restart
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Order of operations in switch_dataset():
 *   1. close_mmap() — munmap() old region, close() old fd
 *   2. open_and_mmap() — open new file, mmap, validate header
 *   3. RNG state is NOT touched
 *
 * If open_and_mmap() throws in step 2 (e.g., file missing), the DataLoader
 * is left with fd_ == -1 and mapped_ == MAP_FAILED (a "null" state).
 * The caller should catch the exception and handle the error (e.g., by
 * retrying or falling back to the previous dataset).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Batch layout  (causal language modelling)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   For batch item b, token offset s:
 *     X[b * T + t] = tokens[s + t]       for t ∈ [0, T)
 *     Y[b * T + t] = tokens[s + t + 1]   for t ∈ [0, T)
 *
 *   s is drawn uniformly from [0, num_tokens − T − 1] so both X and Y
 *   access at most tokens[s + T] — the last valid index in a T+1-wide window.
 *
 * Build:
 *   g++ -std=c++17 -O2 data_loader/dataloader.cpp
 */

#include "data_loader/dataloader.hpp"

// POSIX headers
#include <fcntl.h>          // open, O_RDONLY
#include <sys/mman.h>       // mmap, munmap, PROT_READ, MAP_SHARED, MAP_FAILED
#include <sys/stat.h>       // fstat, struct stat
#include <unistd.h>         // close

#include <cerrno>
#include <cstring>          // strerror
#include <cstdint>
#include <stdexcept>
#include <string>

namespace data_loader {

// ── File-local constants ──────────────────────────────────────────────────────

namespace {

static constexpr uint32_t MAGIC_NUMBER       = 0xDEAD'BEEF;
static constexpr uint32_t EXPECTED_VOCAB_SIZE = 256;

// ─────────────────────────────────────────────────────────────────────────────
// FileHeader
// ─────────────────────────────────────────────────────────────────────────────
// Maps directly onto the first 16 bytes of the .bin file (little-endian, x86).
// Natural alignment guarantees: uint32@0 (OK), uint32@4 (OK), uint64@8 (OK).
// static_assert below confirms the layout is exactly 16 bytes.
//
#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic;       // 0xDEADBEEF
    uint32_t vocab_size;  // 256
    uint64_t num_tokens;  // N (payload length)
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 16,
              "FileHeader must be exactly 16 bytes to match the Python serialiser.");

// Helper: build a std::runtime_error with the system error string appended.
std::runtime_error make_os_error(const std::string& prefix)
{
    return std::runtime_error(prefix + ": " + std::strerror(errno));
}

}  // anonymous namespace


// ═════════════════════════════════════════════════════════════════════════════
// Private helpers
// ═════════════════════════════════════════════════════════════════════════════

void DataLoader::open_and_mmap(const std::string& path)
{
    // ── 1. Open file ──────────────────────────────────────────────────────────
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw make_os_error("DataLoader: cannot open '" + path + "'");
    }

    // ── 2. Stat for file size ─────────────────────────────────────────────────
    struct stat st{};
    if (::fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw make_os_error("DataLoader: fstat failed on '" + path + "'");
    }

    if (st.st_size < static_cast<off_t>(sizeof(FileHeader))) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error(
            "DataLoader: file '" + path + "' is too small to hold a valid header "
            "(expected ≥ 16 bytes, got " + std::to_string(st.st_size) + ").");
    }

    map_size_ = static_cast<size_t>(st.st_size);

    // ── 3. mmap the entire file ───────────────────────────────────────────────
    //
    // MAP_SHARED: changes to the mapping are written through to the file.
    //   (We only PROT_READ so no writes will occur, but MAP_SHARED is correct
    //   for read-only data files and allows multiple processes to share pages.)
    //
    // MAP_POPULATE (Linux): pre-faults every page of the mapping immediately.
    //   Training won't incur page faults for datasets that fit in RAM.
    //   Silently omitted on non-Linux POSIX (macOS, BSD) via #ifdef guard.
    //
    int mmap_flags = MAP_SHARED;
#ifdef MAP_POPULATE
    mmap_flags |= MAP_POPULATE;
#endif

    mapped_ = ::mmap(nullptr, map_size_, PROT_READ, mmap_flags, fd_, /*offset=*/0);

    if (mapped_ == MAP_FAILED) {
        ::close(fd_);
        fd_     = -1;
        mapped_ = nullptr;
        throw make_os_error("DataLoader: mmap failed on '" + path + "'");
    }

    // ── 4. Validate header ─────────────────────────────────────────────────────
    //
    // We cast the start of the mapping directly to FileHeader* — no read() call
    // needed.  The OS has already mapped the first page (containing the header)
    // into our address space.
    //
    const auto* hdr = static_cast<const FileHeader*>(mapped_);

    if (hdr->magic != MAGIC_NUMBER) {
        uint32_t got_magic = hdr->magic;
        ::munmap(mapped_, map_size_);
        ::close(fd_);
        mapped_ = nullptr;
        fd_     = -1;
        throw std::runtime_error(
            "DataLoader: magic number mismatch in '" + path + "'. "
            "Expected 0xDEADBEEF, got 0x" +
            [&]{ char buf[16]; std::snprintf(buf,sizeof(buf),"%08X",got_magic); return std::string(buf); }());
    }

    if (hdr->vocab_size != EXPECTED_VOCAB_SIZE) {
        uint32_t got_vocab = hdr->vocab_size;
        ::munmap(mapped_, map_size_);
        ::close(fd_);
        mapped_ = nullptr;
        fd_     = -1;
        throw std::runtime_error(
            "DataLoader: vocab_size mismatch in '" + path + "'. "
            "Expected " + std::to_string(EXPECTED_VOCAB_SIZE) +
            ", got "    + std::to_string(got_vocab) + ".");
    }



    num_tokens_ = (map_size_ - sizeof(FileHeader)) / sizeof(uint16_t);
    vocab_size_ = hdr->vocab_size;

    // ── 5. Validate payload size against st_size ───────────────────────────────
    // Since we computed num_tokens_ from map_size_, we just check it's not empty
    if (num_tokens_ == 0) {
        ::munmap(mapped_, map_size_);
        ::close(fd_);
        mapped_ = nullptr;
        fd_     = -1;
        throw std::runtime_error(
            "DataLoader: file '" + path + "' is truncated or empty. "
            "Header expects payload but file is only " + std::to_string(map_size_) + " bytes.");
    }

    // ── 6. Set token pointer ───────────────────────────────────────────────────
    //
    // tokens_ points to the first uint16 token (immediately after the header).
    // All token accesses go through this pointer — the OS page cache handles
    // the rest.  No copies ever occur.
    //
    tokens_ = reinterpret_cast<const uint16_t*>(
                  static_cast<const char*>(mapped_) + sizeof(FileHeader));
}


void DataLoader::close_mmap() noexcept
{
    if (mapped_ != nullptr && mapped_ != MAP_FAILED) {
        ::munmap(mapped_, map_size_);
        mapped_     = nullptr;
        map_size_   = 0;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    tokens_     = nullptr;
    num_tokens_ = 0;
    vocab_size_ = 0;
}


// ═════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ═════════════════════════════════════════════════════════════════════════════

DataLoader::DataLoader(const std::string& path, uint32_t seed)
    : rng_(seed)
{
    open_and_mmap(path);
}

DataLoader::~DataLoader()
{
    close_mmap();
}


// ═════════════════════════════════════════════════════════════════════════════
// next_batch
// ═════════════════════════════════════════════════════════════════════════════

Batch DataLoader::next_batch(size_t batch_size, size_t seq_len)
{
    // ── Validate ──────────────────────────────────────────────────────────────
    if (tokens_ == nullptr) {
        throw std::runtime_error(
            "DataLoader::next_batch: no dataset loaded (call switch_dataset first).");
    }
    if (seq_len == 0) {
        throw std::invalid_argument("DataLoader::next_batch: seq_len must be > 0.");
    }
    if (num_tokens_ < seq_len + 1) {
        throw std::runtime_error(
            "DataLoader::next_batch: dataset has " + std::to_string(num_tokens_) +
            " tokens — too small for seq_len=" + std::to_string(seq_len) +
            " (need at least seq_len + 1 = " + std::to_string(seq_len + 1) + ").");
    }

    // ── Sample ────────────────────────────────────────────────────────────────
    //
    // Uniform distribution over valid start positions.
    // A valid start s satisfies:  s + seq_len < num_tokens
    // (X needs [s, s+seq_len) and Y needs [s+1, s+seq_len+1) ⊆ [0, num_tokens))
    //
    //   → s ≤ num_tokens − seq_len − 1
    //
    // Note: uniform_int_distribution is INCLUSIVE on both ends.
    //
    const size_t max_start = num_tokens_ - seq_len - 1;
    std::uniform_int_distribution<size_t> dist(0, max_start);

    Batch batch;
    batch.batch_size = batch_size;
    batch.seq_len    = seq_len;
    batch.X.resize(batch_size * seq_len);
    batch.Y.resize(batch_size * seq_len);

    for (size_t b = 0; b < batch_size; ++b) {
        const size_t start  = dist(rng_);
        const size_t offset = b * seq_len;

        for (size_t t = 0; t < seq_len; ++t) {
            //
            // tokens_ is a direct pointer into the mmap'd region.
            // This loop is a series of L1-cache-resident memory reads when
            // MAP_POPULATE has pre-faulted the pages.
            //
            batch.X[offset + t] = static_cast<size_t>(tokens_[start + t]);
            batch.Y[offset + t] = static_cast<size_t>(tokens_[start + t + 1]);
        }
    }

    return batch;
}


// ═════════════════════════════════════════════════════════════════════════════
// switch_dataset
// ═════════════════════════════════════════════════════════════════════════════

void DataLoader::switch_dataset(const std::string& path)
{
    //
    // Sequence: release old → acquire new.
    //
    // If open_and_mmap() throws (e.g., file not found for the new phase),
    // the DataLoader is left in a "null" state: fd_=-1, mapped_=nullptr,
    // tokens_=nullptr.  The training loop must catch the exception.
    //
    // The RNG state (rng_) is deliberately NOT touched — the random-batch
    // sequence continues seamlessly across the curriculum transition.
    //
    close_mmap();          // munmap + close old file (noexcept)
    open_and_mmap(path);   // open + mmap new file (may throw)
}

}  // namespace data_loader
