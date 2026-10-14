# Transformer Engine C++

> A **zero-dependency, pure C++17** autoregressive language model engine — covering every layer from raw POSIX syscalls to a live React chat interface. Trained a **4.37M-parameter GPT-style decoder** across a **2.17B-token, 3-phase curriculum** on Google Cloud Platform, driving cross-entropy loss from **6.15 → 1.07** over 50,000 steps.

No PyTorch. No Eigen. No Boost. Every tensor operation, gradient computation, memory mapping, and HTTP byte was written by hand.

---

## Live Demo

| Link | Description |
|------|-------------|
| [**Demo →**](https://transformer-engine-ui.vercel.app/) | React chat UI talking to the trained C++ inference server |
| [**GitHub →**](https://github.com/KHU5HWANT/transformer-engine-cpp) | Full source code repository |
| [**Build Log (PDF) →**](./Transformer_internals.pdf) | Full 200-page engineering log documenting every implementation decision |

---

## Key Features

### 🔁 Reverse-Mode Autograd Engine (DAG-based)
- Every operation (`matmul`, `add`, `softmax`, `layernorm`, ...) produces a `Node` that carries its own `_backward` closure.
- The entire DAG is topologically sorted at `backward()` time and traversed in reverse, accumulating gradients via `+=` to support fan-out (weight reuse).
- **Cycle prevention**: `Node::children` stores `std::vector<std::weak_ptr<Node>>`. Every backward lambda captures the output node as `std::weak_ptr<Node>(out)`, breaking the `Node → lambda → Node` reference cycle. RAII guarantees the entire graph is freed after each step with zero memory leaks.

### ⚡ OpenMP Cache-Blocked GEMM Kernel
- The naive triple-loop matrix multiply is replaced by a **64×64 L1-tiling** kernel that keeps all three tile buffers in L1/L2 cache across the inner dimension.
- Parallelised with `#pragma omp parallel for collapse(2)` across the output tile grid.
- **Benchmark result**: `7.25 GFLOPS` at N=1024, a **3.24× speedup** over the naive implementation (run `./bench_matmul` to reproduce).

### 👁️ 3-D Folded Multi-Head Attention
- Avoids 4-D tensor infrastructure entirely by **folding `B×H` heads into the batch dimension** before each `matmul`.
- After `split_heads_op`, Q/K/V all have shape `[B·H, T, d_k]`. The existing 3-D batched `matmul` dispatches correctly across all `B·H` independent `[T, d_k]` slices.
- Scaling, causal masking and softmax are **fused into a single custom op** (`scaled_causal_softmax_op`), computing the causal lower-triangular mask inline with zero extra allocations.

### 📦 POSIX mmap DataLoader with Zero-Downtime Hot-Swap
- Binary `.bin` token files are mapped with `mmap(MAP_SHARED | MAP_POPULATE)`, making every token access a single pointer dereference with **O(1) RAM footprint**.
- Curriculum phase transitions call `switch_dataset()`, which atomically `munmap()`s the old file and `mmap()`s the new one while the training loop runs uninterrupted — no restart required.
- **Throughput**: bounded by L3/DRAM bandwidth; warm page-cache reads achieve `>20 GB/s` on modern hardware.

### 🌐 Raw POSIX TCP Inference Server
- A hand-rolled HTTP/1.1 server using `socket()` → `setsockopt()` → `bind()` → `listen()` → `accept()` → `recv()`/`send()` — zero external networking libraries.
- Exposes `POST /predict` for next-token generation and `GET /health` for liveness probes.
- CORS preflight handled via `OPTIONS /predict`.

### 📐 AdamW Optimizer
- Full bias-corrected AdamW (Loshchilov & Hutter, 2018) with decoupled weight decay.
- Weight decay applied at the raw learning-rate scale (`param ← param × (1 − lr × λ)`) before the adaptive gradient step, giving consistent regularisation regardless of gradient history.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Loop (main.cpp)                   │
│  batch ← DataLoader.next_batch()                             │
│  logits ← Transformer.forward(X, B, T)   ← builds DAG       │
│  loss   ← cross_entropy(logits, Y)       ← fused log-sum-exp │
│  engine::backward(loss)                  ← reverse DAG       │
│  AdamW.step()                            ← update weights    │
└──────────────────────────────────────────────────────────────┘
         │                              ▲
         │ writes checkpoint_*.bin      │ loads checkpoint_final.bin
         ▼                              │
┌──────────────────┐          ┌────────────────────┐
│  DataLoader      │          │  Inference Server   │
│  mmap(.bin file) │          │  POSIX TCP :8080    │
│  hot-swap phases │          │  POST /predict      │
└──────────────────┘          └────────────────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │  React Frontend     │
                              │  Vite + Tailwind    │
                              │  async pre-fetch    │
                              └────────────────────┘
```

### Component Internals

#### `engine/` — Autograd Foundation
| File | Responsibility |
|------|----------------|
| `tensor.hpp/.cpp` | N-dimensional flat `double` buffer; shape/stride arithmetic |
| `node.hpp/.cpp` | `Node` class: `data` tensor, `grad` tensor, `_backward` closure, `weak_ptr` children |
| `ops.hpp/.cpp` | `matmul`, `add`, `transpose`, `reshape`, `gelu`, `embedding_lookup` — each registers a backward lambda capturing `std::weak_ptr<Node>(out)` |
| `autograd.hpp/.cpp` | `engine::backward(loss)` — Kahn topological sort over `Node::children` weak references, then reverse traversal calling each `_backward` |

**Data flow**: `ops::matmul(a, b)` allocates a new `Node c`, computes `c.data = a.data × b.data`, and stores `c._backward = [a, b, w_out=weak_ptr(c)]{ accumulate grads }`. `c.children = {weak(a), weak(b)}` for DAG traversal.

#### `nn/` — Neural Network Modules
| Module | Details |
|--------|---------|
| `Embedding` | `vocab_size × d_model` weight matrix; `forward()` performs a row gather (embedding lookup) |
| `Linear` | `W·x + b`; `bias=false` variant used for `lm_head` |
| `LayerNorm` | Per-sequence mean/variance normalisation with learnable `γ`, `β` |
| `CausalSelfAttention` | Projects Q/K/V → folds to `[B·H, T, d_k]` → batched matmul → `scaled_causal_softmax_op` → output projection |
| `FeedForward` | Two-layer MLP with GELU activation: `Linear(d→4d) → GELU → Linear(4d→d)` |
| `TransformerBlock` | Pre-LN: `LayerNorm → Attention → residual` then `LayerNorm → FFN → residual` |
| `Transformer` | `tok_emb + pos_emb → N×TransformerBlock → LayerNorm → lm_head` |

**Forward pass shape trace** (B=32, T=128, d=256, H=4, V=256):
```
Input ids         [B, T]         = [32, 128]
tok_emb + pos_emb [B, T, d]      = [32, 128, 256]
× N blocks        [B, T, d]      = [32, 128, 256]
lm_head           [B, T, V]      = [32, 128, 256]   ← logits
cross_entropy     [1]             = scalar loss
```

#### `data_loader/` — POSIX I/O
- **`.bin` format**: 4-byte header (`vocab_size: uint32`), followed by `N × uint16` token IDs.
- `DataLoader::open_and_mmap()` validates the header then calls `::mmap(MAP_SHARED | MAP_POPULATE)`.
- `next_batch(B, T)` draws `B` random start offsets, reads `X = tokens[s:s+T]` and `Y = tokens[s+1:s+T+1]` via pointer arithmetic (zero copies).
- `switch_dataset(path)` runs the atomic 3-step hot-swap (munmap → close → mmap).

#### `loss/` — Numerically Stable Cross-Entropy
- Implements the **log-sum-exp trick**: `loss = -logit[y] + log(Σ exp(logit[i] − max_logit))`.
- Prevents `exp()` overflow for large logit values.
- Gradient w.r.t. logits: `softmax(logits) − one_hot(y)`, accumulated via the autograd DAG.

#### `optim/` — AdamW
- Per-parameter `m_` (first moment) and `v_` (second moment) stored as `std::vector<double>` — not Tensor nodes, so they carry zero autograd overhead.
- Bias-correction factors applied at each step: `m̂ = m / (1 − β₁ᵗ)`, `v̂ = v / (1 − β₂ᵗ)`.

#### `server/` — POSIX HTTP/1.1
- `HttpServer` owns the TCP `fd` and routes table (`std::map<{method, path}, handler>`).
- `parse_request()` reads the request-line, header block, and body (up to `Content-Length` bytes) from the raw socket.
- `make_predict_handler(model, context_len)` runs autoregressive generation: repeatedly calls `model.forward()`, takes `argmax` of the last token's logits, appends to the sequence until EOS or `max_tokens`.

---

## Training Run

| Hyperparameter | Value |
|---------------|-------|
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `seq_len` | 128 |
| `batch_size` | 32 |
| `lr` | 0.001 (AdamW, β₁=0.9, β₂=0.999, ε=1e-8, λ=0.01) |
| **Total parameters** | **4,370,944 (~4.37M)** |

| Curriculum Phase | Steps | Dataset | Tokens |
|-----------------|-------|---------|--------|
| MATH | 0 – 4,999 | Synthetic arithmetic | 4.9M |
| STORIES | 5,000 – 24,999 | TinyStories | 1.90B |
| WIKI | 25,000 – 49,999 | Simple Wikipedia | 273M |
| **Total** | **50,000** | | **~2.17B** |

**Loss trajectory**: `6.15 (step 0) → 1.07 (best, step 4,900) → 1.13 (final, step 49,900)`
**Total wall time**: 1,320,808 s (~15.3 days) on a 4-core GCP Compute Engine (Intel Cascade Lake).

---

## Codebase Structure

```
.
├── engine/                  # Autograd core
│   ├── tensor.hpp/.cpp      #   N-D array with shape/stride
│   ├── node.hpp/.cpp        #   DAG node: data, grad, _backward, weak children
│   ├── ops.hpp/.cpp         #   Differentiable ops (matmul, add, gelu, ...)
│   └── autograd.hpp/.cpp    #   backward() — topo sort + reverse traversal
│
├── nn/                      # Neural network layers
│   ├── module.hpp           #   Abstract base: parameters(), zero_grad()
│   ├── embedding.hpp/.cpp
│   ├── linear.hpp/.cpp
│   ├── layernorm.hpp/.cpp
│   ├── activation.hpp/.cpp  #   GELU
│   ├── softmax.hpp/.cpp
│   ├── attention.hpp/.cpp   #   CausalSelfAttention (B×H fold + fused softmax)
│   ├── feedforward.hpp/.cpp
│   ├── transformer_block.hpp/.cpp
│   └── transformer.hpp/.cpp #   Full GPT-style decoder
│
├── data_loader/             # I/O
│   ├── dataloader.hpp/.cpp  #   POSIX mmap loader + hot-swap
│   └── curriculum.hpp       #   Phase schedule (header-only)
│
├── loss/                    # Loss functions
│   └── cross_entropy.hpp/.cpp  # Fused NLL + log-sum-exp
│
├── optim/                   # Optimizers
│   └── adam.hpp/.cpp        #   AdamW with bias correction
│
├── server/                  # Inference server
│   ├── http_server.hpp/.cpp #   Raw POSIX TCP HTTP/1.1
│   ├── inference_handler.hpp/.cpp  # Autoregressive /predict
│   └── server_main.cpp      #   Entry point, CLI arg parser
│
├── tests/                   # GTest unit tests (7 files)
│   ├── test_tensor.cpp
│   ├── test_autograd.cpp
│   ├── test_grad_check.cpp  #   Numerical gradient verification
│   ├── test_linear.cpp
│   ├── test_attention.cpp
│   ├── test_dataloader.cpp
│   └── test_curriculum.cpp
│
├── benchmarks/              # Performance benchmarks
│   ├── bench_matmul.cpp     #   Naive vs cache-blocked GEMM (GFLOPS)
│   ├── bench_attention.cpp  #   Attention throughput (tok/s)
│   └── bench_dataloader.cpp #   mmap I/O throughput (GB/s)
│
├── frontend/                # React + Vite chat interface
│   └── src/
│       ├── App.jsx          #   Main chat UI
│       ├── api/             #   Fetch wrappers for /predict
│       ├── components/      #   Message bubbles, input box
│       └── hooks/           #   Async token streaming hooks
│
├── scripts/
│   ├── preprocess.py        #   Tokenise raw text → .bin
│   └── plot_loss.py         #   Plot training.log loss curve
│
├── main.cpp                 # Curriculum training entry point
├── CMakeLists.txt           # CMake build (C++17, OpenMP, GTest)
├── Dockerfile               # Containerised server build
└── training.log             # Raw nohup log from 13-day GCP run
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Engine language | C++17 (GCC ≥ 9.3 / Clang ≥ 10) |
| Parallelism | OpenMP (`-fopenmp`) |
| Memory I/O | POSIX `mmap` (`sys/mman.h`) |
| Networking | POSIX TCP sockets (`sys/socket.h`, `netinet/in.h`) |
| Build system | CMake ≥ 3.16 |
| Testing | Google Test (auto-fetched by CMake) |
| Frontend | React 18 + Vite + Tailwind CSS |
| Cloud training | GCP Compute Engine (Intel Cascade Lake, 4-core) |
| Containerisation | Docker |

---

## Getting Started

### Prerequisites

- **Linux or WSL2** (required for POSIX APIs)
- GCC ≥ 9.3 or Clang ≥ 10
- CMake ≥ 3.16
- OpenMP (usually ships with GCC)
- Node.js ≥ 18 (for the React frontend only)

### 1. Clone & Build

```bash
git clone https://github.com/KHU5HWANT/transformer-engine-cpp.git
cd transformer-engine-cpp

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. Run the Test Suite

```bash
# From the build/ directory
ctest --output-on-failure
# Expected: 100% tests passed (7/7)
```

### 3. Quick Smoke Training Run (~minutes on CPU)

```bash
# From the build/ directory
./transformer \
  --d_model 64 --n_heads 4 --n_layers 2 \
  --batch_size 32 --seq_len 128 \
  --phase_steps 100,200,400
```

### 4. Full Curriculum Training (reproducing the GCP run)

```bash
# 1. Preprocess datasets (requires raw text files in data/)
python3 scripts/preprocess.py

# 2. Launch training
./transformer \
  --d_model 256 --n_heads 4 --n_layers 4 \
  --batch_size 32 --seq_len 128 \
  --lr 0.001 \
  --phase_steps 5000,25000,50000
```

### 5. Start the Inference Server

```bash
# From the build/ directory
./transformer_server \
  --d_model 256 --n_heads 4 --n_layers 4 --context_len 128 \
  --checkpoint ../checkpoint_final.bin \
  --port 8080

# Test with curl
curl -s -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": "1+1="}' | python3 -m json.tool
# → {"completion": "2"}

# Health check
curl -s http://localhost:8080/health
# → {"status": "ok", "vocab_size": 256}
```

### 6. Start the React Frontend

```bash
cd frontend
cp .env.example .env.local
# Edit VITE_API_BASE_URL=http://localhost:8080

npm install
npm run dev
# → http://localhost:5173
```

### 7. Performance Benchmarks

```bash
# From the build/ directory
./bench_matmul       # Naive vs cache-blocked GEMM (GFLOPS)
./bench_attention     # Attention forward throughput (tok/s)
./bench_dataloader    # mmap I/O throughput (GB/s)
```

### AddressSanitizer Build (Memory Leak CI)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
make -j$(nproc)
ctest --output-on-failure
# ASAN will report any heap errors; expected: zero violations
```

---

## Checkpoints

Pre-trained checkpoints from the GCP run are included in the repository:

| File | Training step | Notes |
|------|-------------|-------|
| `checkpoint_phase0.bin` | Step 5,000 | End of MATH phase |
| `checkpoint_phase1.bin` | Step 25,000 | End of STORIES phase |
| `checkpoint_final.bin` | Step 50,000 | Final weights (loss ≈ 1.07) |

Each checkpoint is a flat binary file of raw `double` values in parameter order, ~25 MB.

---

## License

MIT
