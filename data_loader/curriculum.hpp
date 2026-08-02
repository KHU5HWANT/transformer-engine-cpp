/**
 * @file    data_loader/curriculum.hpp
 * @brief   Curriculum Learning Scheduler — header-only.
 *
 * Design (implementation_plan_v2.md, Step 3.3):
 *
 *  Curriculum learning trains progressively harder datasets to improve sample
 *  efficiency and final model quality.  This scheduler implements a three-phase
 *  curriculum:
 *
 *    Phase 0 — MATH    : Synthetic addition equations (simple grammar, exact answers)
 *    Phase 1 — STORIES : TinyStories (natural language, short-range dependency)
 *    Phase 2 — WIKI    : Wikipedia  (factual, long-range dependency, open domain)
 *
 *  The three phases correspond to increasing linguistic complexity.  Training on
 *  MATH first forces the model to learn reliable next-token prediction before
 *  encountering the ambiguity of natural language.
 *
 *  step_thresholds layout
 *  ──────────────────────
 *   thresholds[0] : step count at which the model graduates from MATH → STORIES
 *   thresholds[1] : step count at which the model graduates from STORIES → WIKI
 *   thresholds[2] : total planned training steps  (informational / for progress bars)
 *
 *  advance() maps a step counter to the correct phase:
 *   [0,           thresholds[0]) → DatasetPhase::MATH
 *   [thresholds[0], thresholds[1]) → DatasetPhase::STORIES
 *   [thresholds[1],          ∞)  → DatasetPhase::WIKI
 *
 *  phase_changed() lets the training loop detect a transition and call
 *  DataLoader::switch_dataset() at the right moment — triggering a zero-
 *  downtime hot-swap of the memory-mapped file.
 *
 * Target: Linux/WSL2, C++17, pure STL (no external dependencies).
 */

#pragma once

#include <array>
#include <cstddef>     // size_t
#include <limits>
#include <stdexcept>
#include <string>

namespace data_loader {

// ─────────────────────────────────────────────────────────────────────────────
// DatasetPhase
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Identifies which dataset the model is currently training on.
 *
 * The integer values are used as indices into the paths_ and
 * step_thresholds arrays — do NOT reorder or add values without updating
 * those arrays.
 */
enum class DatasetPhase : size_t {
    MATH    = 0,
    STORIES = 1,
    WIKI    = 2,
};

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumScheduler
// ─────────────────────────────────────────────────────────────────────────────

class CurriculumScheduler {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct a CurriculumScheduler.
     *
     * @param step_thresholds
     *   Three monotonically-increasing step counts:
     *     [0] — end of MATH phase  (switch MATH → STORIES at this step)
     *     [1] — end of STORIES phase (switch STORIES → WIKI at this step)
     *     [2] — total planned training steps (used for progress reporting)
     *
     *   Invariant: thresholds[0] < thresholds[1] <= thresholds[2].
     *   Default: {5'000, 15'000, 30'000}
     *
     * @param paths
     *   File paths for the three .bin datasets in phase order:
     *   {MATH path, STORIES path, WIKI path}.
     *   Defaults: {"data/math.bin", "data/stories.bin", "data/wiki.bin"}
     */
    explicit CurriculumScheduler(
        std::array<size_t, 3> step_thresholds = {5'000, 15'000, 30'000},
        std::array<std::string, 3> paths       = {"data/math.bin",
                                                   "data/stories.bin",
                                                   "data/wiki.bin"})
        : step_thresholds_(step_thresholds)
        , paths_(std::move(paths))
    {
        if (step_thresholds_[0] >= step_thresholds_[1]) {
            throw std::invalid_argument(
                "CurriculumScheduler: thresholds[0] must be < thresholds[1].");
        }
    }

    // ── Phase query ───────────────────────────────────────────────────────────

    /**
     * @brief Map a global training step to the correct DatasetPhase.
     *
     * @param current_step  Global optimizer step (0-indexed).
     * @return              DatasetPhase for this step.
     *
     * Transition table:
     *   step < thresholds[0]                      → MATH
     *   thresholds[0] <= step < thresholds[1]     → STORIES
     *   step >= thresholds[1]                     → WIKI
     */
    [[nodiscard]] DatasetPhase advance(size_t current_step) const noexcept
    {
        if (current_step < step_thresholds_[0]) return DatasetPhase::MATH;
        if (current_step < step_thresholds_[1]) return DatasetPhase::STORIES;
        return DatasetPhase::WIKI;
    }

    /**
     * @brief Detect whether a phase transition occurred between two steps.
     *
     * Returns true if advance(prev_step) != advance(current_step).
     * The training loop should call DataLoader::switch_dataset() when this
     * returns true to hot-swap the memory-mapped file.
     *
     * @param prev_step    Step count at the end of the previous iteration.
     * @param current_step Step count at the start of the current iteration.
     */
    [[nodiscard]] bool phase_changed(size_t prev_step,
                                      size_t current_step) const noexcept
    {
        return advance(prev_step) != advance(current_step);
    }

    // ── Path resolution ───────────────────────────────────────────────────────

    /**
     * @brief Return the .bin file path for a given DatasetPhase.
     *
     * @param phase  One of {MATH, STORIES, WIKI}.
     * @return       Reference to the stored path string.
     *
     * @throws std::out_of_range if phase is out of range (defensive check).
     */
    [[nodiscard]] const std::string& dataset_path(DatasetPhase phase) const
    {
        const size_t idx = static_cast<size_t>(phase);
        if (idx >= paths_.size()) {
            throw std::out_of_range(
                "CurriculumScheduler::dataset_path: invalid DatasetPhase.");
        }
        return paths_[idx];
    }

    // ── Progress helpers ──────────────────────────────────────────────────────

    /**
     * @brief Total planned training steps (thresholds[2]).
     *
     * Used by the training loop to display percentage progress.
     */
    [[nodiscard]] size_t total_steps()   const noexcept { return step_thresholds_[2]; }

    /**
     * @brief Step at which the MATH → STORIES transition fires.
     */
    [[nodiscard]] size_t math_steps()    const noexcept { return step_thresholds_[0]; }

    /**
     * @brief Step at which the STORIES → WIKI transition fires.
     */
    [[nodiscard]] size_t stories_steps() const noexcept { return step_thresholds_[1]; }

    // ── Public for inspection ─────────────────────────────────────────────────

    std::array<size_t,      3> step_thresholds;  ///< Phase transition step counts
    std::array<std::string, 3> paths;            ///< .bin file paths per phase

private:
    std::array<size_t,      3> step_thresholds_;
    std::array<std::string, 3> paths_;
};

}  // namespace data_loader
