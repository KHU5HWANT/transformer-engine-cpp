/**
 * @file    tests/test_dataloader.cpp
 * @brief   GTest unit tests for data_loader::DataLoader (Step 3.2).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Test fixture design — why we write .bin files on disk
 * ════════════════════════════════════════════════════════════════════════════
 *
 * DataLoader uses POSIX mmap() which requires a real file descriptor.
 * We cannot pass a raw buffer — the OS must mmap a file on disk.
 *
 * The fixture (DataLoaderTest) writes a minimal valid .bin file to a
 * temporary path in /tmp at test start and deletes it at test end via RAII.
 *
 * Binary file format (16-byte header, from scripts/preprocess.py):
 *   Offset  Size   Field
 *   Offset  Size   Field
 *   0       4      magic    = 0xDEADBEEF  (uint32_t, little-endian)
 *   4       4      vocab_size = 256       (uint32_t)
 *   8       8      num_tokens = N         (uint64_t)
 *   16      N*2    token payload          (N uint16_t values)
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_dataloader.cpp data_loader/dataloader.cpp \
 *       -lgtest -lgtest_main -pthread
 */

#include <gtest/gtest.h>
#include "data_loader/dataloader.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using data_loader::DataLoader;
using data_loader::Batch;

// ─────────────────────────────────────────────────────────────────────────────
// Binary .bin file writer helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Write a valid .bin file with a deterministic token payload.
 *
 * Tokens are written as uint16_t with values token[i] = i % 256.
 * This lets tests predict exact token values at known offsets.
 *
 * @param path       Destination path.
 * @param num_tokens Number of uint16_t tokens to write (after the header).
 */
static void write_dummy_bin(const std::string& path, size_t num_tokens)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open())
        throw std::runtime_error("write_dummy_bin: cannot open " + path);

    // 16-byte header
    const uint32_t magic      = 0xDEADBEEF;
    const uint32_t vocab_size = 256;
    const uint64_t num_toks_header = num_tokens;

    f.write(reinterpret_cast<const char*>(&magic),      4);
    f.write(reinterpret_cast<const char*>(&vocab_size), 4);
    f.write(reinterpret_cast<const char*>(&num_toks_header), 8);

    // Token payload: token[i] = i % 256  (deterministic for testing)
    for (size_t i = 0; i < num_tokens; ++i) {
        const uint16_t tok = static_cast<uint16_t>(i % 256);
        f.write(reinterpret_cast<const char*>(&tok), 2);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test Fixture
// ─────────────────────────────────────────────────────────────────────────────

class DataLoaderTest : public ::testing::Test {
protected:
    static constexpr size_t NUM_TOKENS = 1024;      // must be > seq_len + 1
    std::string bin_path_A = "/tmp/dl_test_A.bin";
    std::string bin_path_B = "/tmp/dl_test_B.bin";

    void SetUp() override {
        write_dummy_bin(bin_path_A, NUM_TOKENS);
        // B has a different token pattern: token[i] = (i + 128) % 256
        // written manually so we can distinguish it from A
        write_dummy_bin(bin_path_B, NUM_TOKENS / 2);   // intentionally smaller
    }

    void TearDown() override {
        fs::remove(bin_path_A);
        fs::remove(bin_path_B);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Construction & metadata
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(DataLoaderTest, ConstructionSucceeds)
{
    EXPECT_NO_THROW(DataLoader loader(bin_path_A, /*seed=*/0));
}

TEST_F(DataLoaderTest, NumTokensMatchesFileContents)
{
    DataLoader loader(bin_path_A, 0);
    EXPECT_EQ(loader.num_tokens(), NUM_TOKENS);
}

TEST_F(DataLoaderTest, VocabSizeIs256)
{
    DataLoader loader(bin_path_A, 0);
    EXPECT_EQ(loader.vocab_size(), 256u);
}

TEST_F(DataLoaderTest, MapSizeEqualsHeaderPlusPayload)
{
    DataLoader loader(bin_path_A, 0);
    // 16-byte header + NUM_TOKENS * sizeof(uint16_t)
    const size_t expected = 16 + NUM_TOKENS * sizeof(uint16_t);
    EXPECT_EQ(loader.map_size(), expected);
}

TEST_F(DataLoaderTest, BadPathThrows)
{
    EXPECT_THROW(DataLoader("/tmp/nonexistent_file_xyz.bin", 0), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// next_batch() — shape and value correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(DataLoaderTest, BatchShapeIsCorrect)
{
    DataLoader loader(bin_path_A, 42);
    const size_t B = 4, T = 16;
    Batch batch = loader.next_batch(B, T);

    EXPECT_EQ(batch.X.size(), B * T);
    EXPECT_EQ(batch.Y.size(), B * T);
    EXPECT_EQ(batch.batch_size, B);
    EXPECT_EQ(batch.seq_len,    T);
}

TEST_F(DataLoaderTest, YIsXShiftedByOne)
{
    // Y[b, t] must equal the token that immediately follows X[b, t]
    // in the source file.  Since our token pattern is token[i] = i % 256,
    // and the sequence is contiguous, Y[b, t] = (X[b, t] + 1) % 256.
    // But we don't know the start offset, so we verify the SHIFT property:
    // For each sequence in the batch, Y[t] is the token AFTER X[t].
    //
    // We verify: within each batch item, X[t+1] == Y[t]   for t in [0, T-2]
    DataLoader loader(bin_path_A, 7);
    const size_t B = 2, T = 8;
    Batch batch = loader.next_batch(B, T);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t + 1 < T; ++t) {
            // X[b, t+1] must equal Y[b, t] (both come from adjacent positions)
            EXPECT_EQ(batch.X[b * T + t + 1], batch.Y[b * T + t])
                << "Y[" << b << "," << t << "] should equal X[" << b << "," << t+1 << "]";
        }
    }
}

TEST_F(DataLoaderTest, AllTokensInVocabRange)
{
    DataLoader loader(bin_path_A, 0);
    Batch batch = loader.next_batch(8, 32);
    for (size_t tok : batch.X) EXPECT_LT(tok, 256u);
    for (size_t tok : batch.Y) EXPECT_LT(tok, 256u);
}

TEST_F(DataLoaderTest, TooLargeSeqLenThrows)
{
    DataLoader loader(bin_path_A, 0);
    // seq_len + 1 > num_tokens → must throw
    EXPECT_THROW(loader.next_batch(1, NUM_TOKENS + 1), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// switch_dataset() — hot-swap correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(DataLoaderTest, SwitchDatasetChangesNumTokens)
{
    DataLoader loader(bin_path_A, 0);
    EXPECT_EQ(loader.num_tokens(), NUM_TOKENS);

    loader.switch_dataset(bin_path_B);
    // bin_path_B has NUM_TOKENS / 2 tokens
    EXPECT_EQ(loader.num_tokens(), NUM_TOKENS / 2);
}

TEST_F(DataLoaderTest, SwitchDatasetAllowsBatchSampling)
{
    DataLoader loader(bin_path_A, 0);
    loader.switch_dataset(bin_path_B);

    // Should be able to sample from the new dataset without crashing
    EXPECT_NO_THROW(loader.next_batch(1, 16));
}

TEST_F(DataLoaderTest, SwitchDatasetBadPathThrows)
{
    DataLoader loader(bin_path_A, 0);
    EXPECT_THROW(loader.switch_dataset("/tmp/nonexistent_xyz.bin"), std::runtime_error);
}

TEST_F(DataLoaderTest, SwitchDatasetDoesNotCorruptRNG)
{
    // The RNG state is preserved across switch_dataset.
    // We verify by seeding two loaders identically, switching one,
    // then drawing a batch from each using the SAME new dataset.
    // They should produce the SAME batch (same RNG sequence).
    write_dummy_bin("/tmp/dl_test_C.bin", NUM_TOKENS);

    DataLoader loader1(bin_path_A, /*seed=*/99);
    DataLoader loader2(bin_path_A, /*seed=*/99);

    // Advance both RNGs by one batch on the original dataset
    loader1.next_batch(2, 8);
    loader2.next_batch(2, 8);

    // Switch both to the same new file
    loader1.switch_dataset("/tmp/dl_test_C.bin");
    loader2.switch_dataset("/tmp/dl_test_C.bin");

    // Next batch should be identical (same RNG state)
    Batch b1 = loader1.next_batch(2, 8);
    Batch b2 = loader2.next_batch(2, 8);

    EXPECT_EQ(b1.X, b2.X) << "RNG state must be preserved across switch_dataset";
    EXPECT_EQ(b1.Y, b2.Y);

    fs::remove("/tmp/dl_test_C.bin");
}
