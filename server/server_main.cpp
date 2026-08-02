/**
 * @file    server/server_main.cpp
 * @brief   Entry point for the transformer_server inference service.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Usage
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   ./transformer_server [options]
 *
 *   Options:
 *     --port PORT          TCP port to bind (default: 8080)
 *     --checkpoint PATH    .bin checkpoint file to load (default: checkpoint_final.bin)
 *     --d_model N          Model width (default: 64)
 *     --n_heads N          Number of attention heads (default: 4)
 *     --n_layers N         Number of transformer blocks (default: 2)
 *     --context_len N      Maximum sequence length (default: 128)
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Quick-start curl test
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   # Terminal 1 — start the server
 *   ./build/transformer_server --port 8080 --checkpoint checkpoint_final.bin
 *
 *   # Terminal 2 — send a prediction request
 *   curl -s -X POST http://localhost:8080/predict \
 *        -H "Content-Type: application/json" \
 *        -d '{"prompt": "1+1="}' | python3 -m json.tool
 *
 *   Expected output: {"completion": "2"}
 *
 *   # Health-check (GET /health)
 *   curl -s http://localhost:8080/health
 *   {"status": "ok", "vocab_size": 256}
 *
 * Target: Linux/WSL2, C++17.
 */

#include "server/http_server.hpp"
#include "server/inference_handler.hpp"
#include "nn/transformer.hpp"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal CLI argument parser
// ─────────────────────────────────────────────────────────────────────────────

struct ServerConfig {
    uint16_t    port         = 8080;
    std::string checkpoint   = "checkpoint_final.bin";
    size_t      d_model      = 64;
    size_t      n_heads      = 4;
    size_t      n_layers     = 2;
    size_t      context_len  = 128;
};

static ServerConfig parse_args(int argc, char** argv)
{
    ServerConfig cfg;
    for (int i = 1; i < argc - 1; ++i) {
        const std::string key = argv[i];
        const std::string val = argv[i + 1];
        if (key == "--port")        { cfg.port        = static_cast<uint16_t>(std::stoul(val)); ++i; }
        else if (key == "--checkpoint")  { cfg.checkpoint   = val;                ++i; }
        else if (key == "--d_model")     { cfg.d_model      = std::stoul(val);    ++i; }
        else if (key == "--n_heads")     { cfg.n_heads      = std::stoul(val);    ++i; }
        else if (key == "--n_layers")    { cfg.n_layers     = std::stoul(val);    ++i; }
        else if (key == "--context_len") { cfg.context_len  = std::stoul(val);    ++i; }
        else {
            std::cerr << "[server_main] Unknown argument: " << key << "\n";
        }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint loading (reads raw parameter doubles from the .bin file)
// ─────────────────────────────────────────────────────────────────────────────

static bool load_checkpoint(engine::nn::Transformer& model,
                             const std::string&       path)
{
    if (!fs::exists(path)) {
        std::cout << "[server_main] No checkpoint found at '" << path
                  << "' — running with random weights.\n";
        return false;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[server_main] Could not open checkpoint: " << path << "\n";
        return false;
    }

    size_t loaded = 0;
    for (const auto& param : model.parameters()) {
        double* data = param->data.data_ptr();
        const size_t n = param->data.numel();
        f.read(reinterpret_cast<char*>(data),
               static_cast<std::streamsize>(n * sizeof(double)));
        if (f.fail()) {
            std::cerr << "[server_main] Checkpoint truncated at param " << loaded << "\n";
            return false;
        }
        ++loaded;
    }

    std::cout << "[server_main] Loaded " << loaded << " parameter tensors from '"
              << path << "'\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Global server pointer for signal handler
// ─────────────────────────────────────────────────────────────────────────────

static server::HttpServer* g_server = nullptr;

static void signal_handler(int /*sig*/)
{
    std::cout << "\n[server_main] Shutdown signal received.\n";
    if (g_server) g_server->stop();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const ServerConfig cfg = parse_args(argc, argv);

    // ── Banner ────────────────────────────────────────────────────────────────
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             C++ Transformer Inference Server                 ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  d_model=" << cfg.d_model
              << "  n_heads="  << cfg.n_heads
              << "  n_layers=" << cfg.n_layers
              << "  T="        << cfg.context_len << "\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // ── Instantiate model ─────────────────────────────────────────────────────
    std::cout << "[server_main] Instantiating Transformer...\n";
    engine::nn::Transformer model(
        /*vocab_size=*/  256,
        /*context_len=*/ cfg.context_len,
        /*d_model=*/     cfg.d_model,
        /*n_heads=*/     cfg.n_heads,
        /*n_layers=*/    cfg.n_layers
    );

    const auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& p : params) total_params += p->data.numel();
    std::cout << "[server_main] Parameters: " << params.size()
              << " tensors, " << total_params << " doubles ("
              << total_params * 8 / 1024 << " KB)\n\n";

    // ── Load checkpoint ───────────────────────────────────────────────────────
    load_checkpoint(model, cfg.checkpoint);
    std::cout << "\n";

    // ── Build HTTP server ─────────────────────────────────────────────────────
    server::HttpServer srv(cfg.port);

    // /predict — POST  — the inference endpoint
    srv.add_route("POST", "/predict",
                  server::make_predict_handler(model, cfg.context_len));

    // /health  — GET   — returns a simple status JSON for load-balancer probes
    srv.add_route("GET", "/health",
                  [](const server::HttpRequest&) -> server::HttpResponse {
                      return {200, "OK", "application/json",
                              R"({"status": "ok", "vocab_size": 256})"};
                  });

    // ── Signal handling ───────────────────────────────────────────────────────
    g_server = &srv;
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "\n[server_main] Routes registered:\n";
    std::cout << "  POST /predict  — character-level next-token prediction\n";
    std::cout << "  GET  /health   — liveness probe\n\n";
    std::cout << "Example:\n";
    std::cout << R"(  curl -s -X POST http://localhost:)" << cfg.port
              << R"(/predict -H "Content-Type: application/json")" << "\n";
    std::cout << R"(       -d '{"prompt": "1+1="}' )" << "\n\n";

    // ── Start serving ─────────────────────────────────────────────────────────
    srv.serve();   // blocks until stop() or SIGINT

    std::cout << "[server_main] Exited cleanly.\n";
    return EXIT_SUCCESS;
}
