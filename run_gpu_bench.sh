# Run GPU benchmark on WSL2 with RTX 5070
# The WSL2 CUDA driver is in /usr/lib/wsl/lib, needs explicit LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./target/release/gpu_bench
