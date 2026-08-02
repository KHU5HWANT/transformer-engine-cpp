##
## CMakeLists.txt — Root build configuration for the C++ Deep Learning Engine.
##
## ════════════════════════════════════════════════════════════════════════════
## Project layout
## ════════════════════════════════════════════════════════════════════════════
##
##   engine/       — Tensor, Node, differentiable ops, autograd backward()
##   nn/           — Linear, LayerNorm, Embedding, Attention, Transformer, ...
##   data_loader/  — POSIX mmap DataLoader, CurriculumScheduler (header-only)
##   optim/        — AdamW optimizer
##   loss/         — Fused cross-entropy
##   server/       — Raw POSIX HTTP/1.1 server + /predict inference handler
##   tests/        — GTest unit tests (7 files)
##   benchmarks/   — GFLOPS / tokens-per-sec / GB-per-sec benchmarks (3 files)
##   main.cpp      — Curriculum training loop
##
## ════════════════════════════════════════════════════════════════════════════
## Quick-start
## ════════════════════════════════════════════════════════════════════════════
##
##   mkdir build && cd build
##   cmake .. -DCMAKE_BUILD_TYPE=Release
##   make -j$(nproc)
##   ctest --output-on-failure          # run all GTests
##   ./transformer --phase_steps 100,200,400   # quick smoke test
##   ./transformer_server               # start inference server on :8080
##
## AddressSanitizer build (CI / valgrind alternative):
##   cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
##   make -j$(nproc) && ctest --output-on-failure
##
## Target: Linux / WSL2, GCC ≥ 9.3 or Clang ≥ 10 with C++17 and OpenMP.
##

cmake_minimum_required(VERSION 3.16)

project(CppTransformer
    VERSION     1.0.0
    DESCRIPTION "From-scratch C++17 Transformer with autograd, curriculum training, REST inference"
    LANGUAGES   CXX
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. C++ standard
# ─────────────────────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)   # no GNU extensions (-std=c++17, not -std=gnu++17)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build type defaults
# ─────────────────────────────────────────────────────────────────────────────

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compiler flags
# ─────────────────────────────────────────────────────────────────────────────

# -O3 and -fopenmp on all targets via interface propagation through transformer_core.
# AddressSanitizer is opt-in via -DENABLE_ASAN=ON for debugging.

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wno-unused-parameter   # suppress harmless parameter warnings in templates
)

option(ENABLE_ASAN "Build with AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer: ENABLED")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# 4. OpenMP
# ─────────────────────────────────────────────────────────────────────────────

find_package(OpenMP REQUIRED)
message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Google Test — fetched at configure time
# ─────────────────────────────────────────────────────────────────────────────

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0          # pinned for reproducibility
)

# Disable installing GTest into the system (we only need it for tests)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Core static library — transformer_core
#
#    All engine/nn/data_loader/optim/loss translation units compiled once and
#    linked into every executable.  Avoids 7× recompilation of the 2000-line
#    autograd engine for each test binary.
# ─────────────────────────────────────────────────────────────────────────────

add_library(transformer_core STATIC
    # ── Math Engine ──────────────────────────────────────────────────────────
    engine/tensor.cpp
    engine/node.cpp
    engine/ops.cpp
    engine/autograd.cpp

    # ── Neural Network Modules ────────────────────────────────────────────────
    nn/linear.cpp
    nn/layernorm.cpp
    nn/embedding.cpp
    nn/activation.cpp
    nn/softmax.cpp
    nn/attention.cpp
    nn/feedforward.cpp
    nn/transformer_block.cpp
    nn/transformer.cpp

    # ── Data Pipeline ─────────────────────────────────────────────────────────
    data_loader/dataloader.cpp
    # data_loader/curriculum.hpp is header-only — no .cpp

    # ── Optimizer & Loss ──────────────────────────────────────────────────────
    optim/adam.cpp
    loss/cross_entropy.cpp
)

# PUBLIC so that all targets linking transformer_core inherit these settings
target_include_directories(transformer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}   # enables #include "engine/tensor.hpp" etc.
)

target_compile_options(transformer_core PUBLIC
    -O3           # maximum optimisation (overrides -O0 from Debug builds for the core)
)

target_link_libraries(transformer_core PUBLIC
    OpenMP::OpenMP_CXX
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Training executable — transformer
# ─────────────────────────────────────────────────────────────────────────────

add_executable(transformer main.cpp)
target_link_libraries(transformer PRIVATE transformer_core)

message(STATUS "Target: transformer  (curriculum training loop)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Inference server executable — transformer_server
#
#    Depends on server/ sources which use raw POSIX sockets
#    (<sys/socket.h>, <netinet/in.h>, <unistd.h>) — Linux/WSL2 only.
# ─────────────────────────────────────────────────────────────────────────────

add_executable(transformer_server
    server/http_server.cpp
    server/inference_handler.cpp
    server/server_main.cpp          # entry point (load model, register route, serve)
)
target_link_libraries(transformer_server PRIVATE transformer_core)

message(STATUS "Target: transformer_server  (REST /predict endpoint on :8080)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Unit tests — CTest + GTest
# ─────────────────────────────────────────────────────────────────────────────

enable_testing()

# List every test file explicitly (GLOB intentionally avoided — CMake won't
# detect newly added test files without re-running cmake when using GLOB).
set(GTEST_SOURCES
    tests/test_tensor.cpp
    tests/test_autograd.cpp
    tests/test_grad_check.cpp
    tests/test_linear.cpp
    tests/test_attention.cpp
    tests/test_dataloader.cpp
    tests/test_curriculum.cpp
)

foreach(TEST_SRC IN LISTS GTEST_SOURCES)
    # Derive executable name from filename: tests/test_tensor.cpp → test_tensor
    get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)

    add_executable("${TEST_NAME}" "${TEST_SRC}")

    target_link_libraries("${TEST_NAME}" PRIVATE
        transformer_core
        GTest::gtest_main   # provides main() via gtest_main
    )

    # Register with CTest so 'ctest' / 'make test' discovers all tests
    add_test(NAME "${TEST_NAME}" COMMAND "${TEST_NAME}")

    # Emit failure detail on ctest -V
    set_tests_properties("${TEST_NAME}" PROPERTIES
        FAIL_REGULAR_EXPRESSION "FAILED"
    )
endforeach()

message(STATUS "Registered ${CMAKE_CURRENT_LIST_LINE} GTest targets")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Benchmarks
#
#    NOT registered with CTest (they don't pass/fail — they report throughput).
#    Run manually: ./build/benchmarks/bench_matmul
# ─────────────────────────────────────────────────────────────────────────────

set(BENCH_SOURCES
    benchmarks/bench_matmul.cpp
    benchmarks/bench_attention.cpp
    benchmarks/bench_dataloader.cpp
)

foreach(BENCH_SRC IN LISTS BENCH_SOURCES)
    get_filename_component(BENCH_NAME "${BENCH_SRC}" NAME_WE)
    add_executable("${BENCH_NAME}" "${BENCH_SRC}")
    target_link_libraries("${BENCH_NAME}" PRIVATE transformer_core)
endforeach()

# ─────────────────────────────────────────────────────────────────────────────
# 11. Install targets (optional — for packaging)
# ─────────────────────────────────────────────────────────────────────────────

install(TARGETS transformer transformer_server
    RUNTIME DESTINATION bin
)

# ─────────────────────────────────────────────────────────────────────────────
# 12. Summary
# ─────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "  CppTransformer ${PROJECT_VERSION} — CMake configured")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP       : ${OpenMP_CXX_VERSION}")
message(STATUS "  ASAN         : ${ENABLE_ASAN}")
message(STATUS "  Build dir    : ${CMAKE_BINARY_DIR}")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "")
