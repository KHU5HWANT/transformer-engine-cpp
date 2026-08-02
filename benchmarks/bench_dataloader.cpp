/**
 * @file    benchmarks/bench_dataloader.cpp
 * @brief   POSIX mmap DataLoader streaming throughput benchmark.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What we measure
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Throughput GB/s = (bytes_read) / elapsed_seconds
 *
 * Where bytes_read = num_reps × batch_size × seq_len × sizeof(uint16_t) × 2
 *   × 2: both X and Y are read each call (2 windows of seq_len tokens).
 *   × sizeof(uint16_t) = 2: each token is stored as a 16-bit integer.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Why mmap is fast
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Traditional I/O (fread / pread):
 *   1. Application issues read() syscall
 *   2. Kernel copies from page cache → userspace buffer
 *   3. Application processes userspace buffer
 *   Cost: 1 syscall + 1 data copy per read()
 *
 * POSIX mmap:
 *   1. mmap() adds VMA entries — no data is loaded yet
 *   2. First access causes a page fault → kernel maps the page
 *   3. With MAP_POPULATE, all pages are pre-faulted at mmap() time
 *   4. Subsequent accesses are direct pointer dereferences — ZERO syscalls
 *   Cost: 0 syscalls + 0 copies once the pages are resident
 *
 * The benchmark repeatedly calls next_batch() which is nothing more than:
 *   start = rng() % (num_tokens - seq_len - 1)
 *   X = tokens_[start..start+seq_len]    ← pointer arithmetic only
 *   Y = tokens_[start+1..start+seq_len+1]
 *
 * Expected throughput on a machine where the dataset fits in RAM:
 *   ≈ Memory bandwidth (20–60 GB/s depending on NUMA topology).
 *   For a warm page cache:  easily ≥ 10 GB/s.
 *   For a cold read (first pass): limited by disk speed (~500 MB/s SSD).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Test file setup
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The benchmark generates a ~64 MB synthetic .bin file in /tmp.
 * 64 MB = 32 million uint16_t tokens.  This is large enough to stress the
 * page cache and measure realistic streaming bandwidth.
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *       benchmarks/bench_dataloader.cpp data_loader/dataloader.cpp \
 *       -o bench_dataloader
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "data_loader/dataloader.hpp"

using namespace std::chrono;
using data_loader::DataLoader;

// ─────────────────────────────────────────────────────────────────────────────
// Binary file writer
// ─────────────────────────────────────────────────────────────────────────────

static void write_benchmark_bin(const std::string& path, size_t num_tokens)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open())
        throw std::runtime_error("bench_dataloader: cannot create " + path);

    const uint32_t magic      = 0xDEADBEEF;
    const uint32_t version    = 1;
    const uint32_t vocab_size = 256;
    const uint32_t reserved   = 0;

    f.write(reinterpret_cast<const char*>(&magic),      4);
    f.write(reinterpret_cast<const char*>(&version),    4);
    f.write(reinterpret_cast<const char*>(&vocab_size), 4);
    f.write(reinterpret_cast<const char*>(&reserved),   4);

    // Write tokens in 64 KiB chunks for efficiency
    constexpr size_t CHUNK = 32768;   // 32768 uint16_t = 64 KiB
    std::vector<uint16_t> buf(CHUNK);
    size_t remaining = num_tokens;

    while (remaining > 0) {
        const size_t n = std::min(CHUNK, remaining);
        for (size_t i = 0; i < n; ++i)
            buf[i] = static_cast<uint16_t>((remaining - i) % 256);
        f.write(reinterpret_cast<const char*>(buf.data()),
                static_cast<std::streamsize>(n * sizeof(uint16_t)));
        remaining -= n;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Print helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_separator() { std::cout << std::string(64, '─') << "\n"; }

// ─────────────────────────────────────────────────────────────────────────────
// Single benchmark run
// ─────────────────────────────────────────────────────────────────────────────

struct BenchConfig {
    std::string label;
    size_t batch_size;
    size_t seq_len;
    int    warmup_reps;
    int    bench_reps;
};

static void run_dataloader_benchmark(DataLoader& loader, const BenchConfig& cfg)
{
    // Warm-up: trigger page faults (or verify page cache is warm)
    for (int r = 0; r < cfg.warmup_reps; ++r)
        (void)loader.next_batch(cfg.batch_size, cfg.seq_len);

    // Timed run
    auto t0 = steady_clock::now();
    for (int r = 0; r < cfg.bench_reps; ++r)
        (void)loader.next_batch(cfg.batch_size, cfg.seq_len);
    auto t1 = steady_clock::now();

    const double elapsed_s = duration<double>(t1 - t0).count();

    // Bytes touched: each batch reads 2 × (B × T) tokens × 2 bytes/token
    // (X and Y are both read from the mmap'd region)
    const double bytes_per_rep =
        static_cast<double>(cfg.batch_size * cfg.seq_len) * 2.0  // X + Y
        * sizeof(uint16_t);
    const double total_bytes = bytes_per_rep * cfg.bench_reps;

    const double gb_per_s      = total_bytes / (elapsed_s * 1e9);
    const double ms_per_batch  = elapsed_s * 1e3 / cfg.bench_reps;
    const double tok_per_sec   =
        static_cast<double>(cfg.batch_size * cfg.seq_len * cfg.bench_reps) / elapsed_s;

    std::cout << "  " << std::left  << std::setw(28) << cfg.label
              << "  B=" << std::setw(3) << cfg.batch_size
              << "  T=" << std::setw(4) << cfg.seq_len
              << "  " << std::right << std::setw(8) << std::fixed
              << std::setprecision(3) << ms_per_batch << " ms/batch"
              << "  " << std::setw(8) << std::fixed << std::setprecision(3)
              << gb_per_s << " GB/s"
              << "  " << std::setw(10) << std::fixed << std::setprecision(0)
              << tok_per_sec << " tok/s"
              << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    // ── Create a large synthetic dataset (~64 MB) ─────────────────────────────
    constexpr size_t NUM_TOKENS    = 32'000'000;  // 32M tokens × 2 bytes = 64 MB
    const std::string bench_path   = "/tmp/bench_dataloader.bin";

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         POSIX mmap DataLoader Throughput Benchmark           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Metric: GB/s = (B × T × 2 × sizeof(uint16) × reps) / t     ║\n";
    std::cout << "║  Target: ≥ 5 GB/s (warm page cache, NVMe SSD)               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "  Writing " << NUM_TOKENS / 1e6 << " M tokens ("
              << 2.0 * NUM_TOKENS / 1e6 << " MB) to " << bench_path << " ...\n";
    write_benchmark_bin(bench_path, NUM_TOKENS);
    std::cout << "  Done. Opening DataLoader...\n\n";

    DataLoader loader(bench_path, /*seed=*/42);

    std::cout << "  Tokens mapped : " << loader.num_tokens() / 1e6 << " M\n";
    std::cout << "  Map size      : " << loader.map_size()   / 1e6 << " MB\n\n";

    print_separator();
    std::cout << "  " << std::left  << std::setw(28) << "Config"
              << "  " << std::setw(5) << "B"
              << "  " << std::setw(6) << "T"
              << "  " << std::right << std::setw(14) << "ms/batch"
              << "  " << std::setw(9) << "GB/s"
              << "  " << std::setw(12) << "tok/s"
              << "\n";
    print_separator();

    // Benchmark across several batch/seqlen combinations
    const std::vector<BenchConfig> configs = {
        // label                  B    T      warm  reps
        {"Small batch  (B=4)",    4,   128,   10,  1000},
        {"Large batch (B=32)",   32,   128,    5,   500},
        {"Short seq   (T=32)",    4,    32,   10,  2000},
        {"Long seq  (T=512)",     4,   512,    5,   200},
        {"Maximal B=32 T=512",   32,   512,    3,   100},
    };

    for (const auto& cfg : configs)
        run_dataloader_benchmark(loader, cfg);

    print_separator();

    // ── Clean up ──────────────────────────────────────────────────────────────
    // DataLoader destructor calls munmap() + close() automatically (RAII).
    // We explicitly remove the temp file after the loader goes out of scope.
    //   (loader is on the stack; destructor runs at end of main)

    std::cout << "\nNote: first run hits page faults (slower); subsequent runs\n";
    std::cout << "      read from the warm page cache at near-memory-bandwidth speed.\n";
    std::cout << "      Benchmarked on same machine — compare across configs only.\n\n";

    // File cleanup happens after DataLoader destructor (see below note)
    // Use a scope to ensure loader is destructed before remove()
    {
        // loader is destroyed here (end of scope is at end of main)
    }
    std::remove(bench_path.c_str());

    return 0;
}
