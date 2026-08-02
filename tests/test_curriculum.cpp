/**
 * @file    tests/test_curriculum.cpp
 * @brief   GTest unit tests for data_loader::CurriculumScheduler (Step 3.3).
 *
 * Tests cover:
 *   - advance() phase mapping at all boundary conditions (step < T0, T0, T1, ∞)
 *   - phase_changed() fires EXACTLY once at each boundary step
 *   - phase_changed() does NOT fire inside a phase
 *   - dataset_path() returns correct paths for each phase
 *   - total_steps(), math_steps(), stories_steps() accessors
 *   - Invalid threshold order throws std::invalid_argument
 *
 * Build:
 *   g++ -std=c++17 -O2 tests/test_curriculum.cpp -lgtest -lgtest_main -pthread
 *   (curriculum.hpp is header-only — no .cpp needed)
 */

#include <gtest/gtest.h>
#include "data_loader/curriculum.hpp"

using data_loader::CurriculumScheduler;
using data_loader::DatasetPhase;

// ─────────────────────────────────────────────────────────────────────────────
// Test fixture with a small, controlled scheduler
// ─────────────────────────────────────────────────────────────────────────────

class CurriculumTest : public ::testing::Test {
protected:
    // Thresholds: MATH ends at step 100, STORIES ends at step 300, total 500
    CurriculumScheduler sched{{100, 300, 500},
                              {"data/math.bin", "data/stories.bin", "data/wiki.bin"}};
};

// ─────────────────────────────────────────────────────────────────────────────
// advance() — phase mapping
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(CurriculumTest, AdvanceStep0IsMAth)
{
    EXPECT_EQ(sched.advance(0), DatasetPhase::MATH);
}

TEST_F(CurriculumTest, AdvanceBeforeFirstThresholdIsMath)
{
    EXPECT_EQ(sched.advance(99), DatasetPhase::MATH);
}

TEST_F(CurriculumTest, AdvanceAtFirstThresholdIsStories)
{
    // Step 100 crosses into STORIES (thresholds[0] = 100)
    EXPECT_EQ(sched.advance(100), DatasetPhase::STORIES);
}

TEST_F(CurriculumTest, AdvanceMidStoriesIsStories)
{
    EXPECT_EQ(sched.advance(200), DatasetPhase::STORIES);
}

TEST_F(CurriculumTest, AdvanceJustBeforeSecondThresholdIsStories)
{
    EXPECT_EQ(sched.advance(299), DatasetPhase::STORIES);
}

TEST_F(CurriculumTest, AdvanceAtSecondThresholdIsWiki)
{
    // Step 300 crosses into WIKI (thresholds[1] = 300)
    EXPECT_EQ(sched.advance(300), DatasetPhase::WIKI);
}

TEST_F(CurriculumTest, AdvanceBeyondTotalIsWiki)
{
    // No upper bound on WIKI phase
    EXPECT_EQ(sched.advance(1'000'000), DatasetPhase::WIKI);
}

// ─────────────────────────────────────────────────────────────────────────────
// phase_changed() — boundary detection
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(CurriculumTest, PhaseChangedAtMathToStoriesBoundary)
{
    // Transition fires exactly at the boundary step (prev=99, curr=100)
    EXPECT_TRUE(sched.phase_changed(99, 100));
}

TEST_F(CurriculumTest, PhaseChangedAtStoriesToWikiBoundary)
{
    EXPECT_TRUE(sched.phase_changed(299, 300));
}

TEST_F(CurriculumTest, PhaseNotChangedInsideMathPhase)
{
    EXPECT_FALSE(sched.phase_changed(0,  1));
    EXPECT_FALSE(sched.phase_changed(50, 51));
    EXPECT_FALSE(sched.phase_changed(0,  99));   // large jump, still in MATH
}

TEST_F(CurriculumTest, PhaseNotChangedInsideStoriesPhase)
{
    EXPECT_FALSE(sched.phase_changed(100, 101));
    EXPECT_FALSE(sched.phase_changed(150, 200));
}

TEST_F(CurriculumTest, PhaseNotChangedInsideWikiPhase)
{
    EXPECT_FALSE(sched.phase_changed(300, 301));
    EXPECT_FALSE(sched.phase_changed(300, 500));
}

TEST_F(CurriculumTest, PhaseChangedIsFalseForSameStep)
{
    // prev == current → same phase → no change
    EXPECT_FALSE(sched.phase_changed(100, 100));
    EXPECT_FALSE(sched.phase_changed(300, 300));
}

TEST_F(CurriculumTest, PhaseChangedDetectsDoubleSkip)
{
    // If the training loop somehow skips both boundaries in one step,
    // phase_changed still fires (MATH → WIKI detected as MATH != WIKI)
    EXPECT_TRUE(sched.phase_changed(0, 400));
}

// ─────────────────────────────────────────────────────────────────────────────
// dataset_path()
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(CurriculumTest, DatasetPathMath)
{
    EXPECT_EQ(sched.dataset_path(DatasetPhase::MATH), "data/math.bin");
}

TEST_F(CurriculumTest, DatasetPathStories)
{
    EXPECT_EQ(sched.dataset_path(DatasetPhase::STORIES), "data/stories.bin");
}

TEST_F(CurriculumTest, DatasetPathWiki)
{
    EXPECT_EQ(sched.dataset_path(DatasetPhase::WIKI), "data/wiki.bin");
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress accessors
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(CurriculumTest, TotalSteps)
{
    EXPECT_EQ(sched.total_steps(),   500u);
}

TEST_F(CurriculumTest, MathSteps)
{
    EXPECT_EQ(sched.math_steps(),    100u);
}

TEST_F(CurriculumTest, StoriesSteps)
{
    EXPECT_EQ(sched.stories_steps(), 300u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom paths
// ─────────────────────────────────────────────────────────────────────────────

TEST(CurriculumCustom, CustomPathsPreserved)
{
    CurriculumScheduler s{{10, 20, 30},
                          {"custom/a.bin", "custom/b.bin", "custom/c.bin"}};

    EXPECT_EQ(s.dataset_path(DatasetPhase::MATH),    "custom/a.bin");
    EXPECT_EQ(s.dataset_path(DatasetPhase::STORIES), "custom/b.bin");
    EXPECT_EQ(s.dataset_path(DatasetPhase::WIKI),    "custom/c.bin");
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid threshold order throws
// ─────────────────────────────────────────────────────────────────────────────

TEST(CurriculumValidation, InvalidThresholdOrderThrows)
{
    // thresholds[0] == thresholds[1] violates the strict ordering invariant
    EXPECT_THROW(
        (CurriculumScheduler{{200, 100, 300}}),
        std::invalid_argument
    );
}

TEST(CurriculumValidation, EqualThresholdsThrow)
{
    // thresholds[0] == thresholds[1] — degenerate (zero-length STORIES phase)
    EXPECT_THROW(
        (CurriculumScheduler{{100, 100, 300}}),
        std::invalid_argument
    );
}
