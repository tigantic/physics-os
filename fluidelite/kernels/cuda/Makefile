# FluidElite CUDA Kernels Makefile
# Constitutional Compliance: Article I (Build Integrity)
#
# Section 1.1: Deterministic builds - same source → same artifacts
# Section 1.2: All warnings treated as errors
# Section 1.3: Build completes in under 60 seconds

# Toolchain (pinned versions per Article I.4)
NVCC := /usr/local/cuda-12.8/bin/nvcc
PYTHON := python3
PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT := .cpython-312-x86_64-linux-gnu.so

# Architecture (RTX 5070 = sm_120)
CUDA_ARCH := sm_120

# Compiler flags
# -Werror equivalent for nvcc: --compiler-options -Werror
# -O3 for production, deterministic floating point
NVCC_FLAGS := -shared -Xcompiler -fPIC -arch=$(CUDA_ARCH) -O3 \
              -Xcompiler -Werror \
              --use_fast_math \
              -lineinfo

# Include paths
INCLUDES := $(PYBIND_INCLUDES)

# Libraries
LIBS := -lcudart -lcublas -lcusolver

# Source files
SOURCES := fused_mps_ops.cu
TARGETS := fused_mps_ops$(PYTHON_EXT)

# Build directory
BUILD_DIR := .

.PHONY: all clean test benchmark verify

all: $(TARGETS)
	@echo "✅ Build complete"
	@echo "   Target: $(TARGETS)"
	@ls -lh $(TARGETS)

# Main kernel compilation
fused_mps_ops$(PYTHON_EXT): fused_mps_ops.cu
	@echo "Building fused_mps_ops..."
	@time $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "✅ fused_mps_ops built"

# Clean build artifacts
clean:
	rm -f *.so *.o
	@echo "✅ Cleaned"

# Run tests (Article II compliance)
test: $(TARGETS)
	@echo "Running verification tests..."
	@$(PYTHON) -c "\
import torch; \
import fused_mps_ops; \
batch, m, n, k = 15, 128, 128, 128; \
A = torch.randn(batch, m, k, device='cuda').contiguous(); \
B = torch.randn(batch, k, n, device='cuda').contiguous(); \
C = torch.zeros(batch, m, n, device='cuda').contiguous(); \
fused_mps_ops.batched_matmul(A.data_ptr(), B.data_ptr(), C.data_ptr(), batch, m, n, k); \
torch.cuda.synchronize(); \
C_ref = torch.bmm(A, B); \
err = torch.norm(C - C_ref) / torch.norm(C_ref); \
print(f'cuBLAS GEMM error: {err.item():.2e}'); \
assert err < 1e-5, 'GEMM verification failed'; \
print('✅ All tests passed')"

# Run benchmarks
benchmark: $(TARGETS)
	@echo "Running benchmarks..."
	@$(PYTHON) -c "\
import fused_mps_ops; \
configs = [(15, 128, 128, 128), (15, 256, 256, 256), (8, 128, 128, 128)]; \
print('cuBLAS Batched GEMM:'); \
for b, m, n, k in configs: \
    t = fused_mps_ops.benchmark_gemm(b, m, n, k, 100); \
    print(f'  {b}× ({m}×{k})@({k}×{n}): {t:.3f}ms')"

# Verify build integrity (Article I.1)
verify: $(TARGETS)
	@echo "Verifying build integrity..."
	@sha256sum $(TARGETS) | tee build.sha256
	@echo "✅ Build hash recorded"

# Environment info
info:
	@echo "Build Environment:"
	@echo "  NVCC: $(NVCC)"
	@$(NVCC) --version | head -1
	@echo "  Python: $(PYTHON)"
	@$(PYTHON) --version
	@echo "  pybind11: $(shell $(PYTHON) -c 'import pybind11; print(pybind11.__version__)')"
	@echo "  CUDA Arch: $(CUDA_ARCH)"
	@echo "  Python ext: $(PYTHON_EXT)"
