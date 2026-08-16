# Build stage
FROM ubuntu:22.04 AS builder

# Prevent tzdata prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y cmake g++ make git

# Set working directory
WORKDIR /app

# Copy source code and CMake configuration
COPY CMakeLists.txt .
COPY main.cpp .
COPY engine/ engine/
COPY nn/ nn/
COPY optim/ optim/
COPY loss/ loss/
COPY data_loader/ data_loader/
COPY server/ server/
COPY tests/ tests/
COPY benchmarks/ benchmarks/

# Build the project
RUN mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make

# Final stage (Runtime)
FROM ubuntu:22.04

# Install OpenMP runtime library needed by our C++ binary
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built binary from the builder stage
COPY --from=builder /app/build/transformer_server .

# Copy the model checkpoints
COPY checkpoint_phase0.bin .
COPY checkpoint_phase1.bin .

# Expose the port that the cloud provider will route traffic to
EXPOSE 8080

# Limit OpenMP to 1 thread to avoid catastrophic context-switching overhead on Render Free Tier (0.1 vCPU)
ENV OMP_NUM_THREADS=1

# Default command to run the server (Starts the Alice/Story model by default)
CMD ["./transformer_server", "--port", "8080", "--checkpoint", "checkpoint_phase1.bin", "--d_model", "256", "--n_heads", "4", "--n_layers", "4"]
