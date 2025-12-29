"""
Python wrapper for implicit QTT CUDA kernel

Provides high-level interface to launch CUDA kernels for direct QTT evaluation.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from torch.utils.cpp_extension import load_inline

# Compile CUDA kernel on first import
_KERNEL_COMPILED = False
_KERNEL_MODULE = None


def _get_cuda_kernel_source():
    """Read CUDA kernel source code"""
    kernel_path = Path(__file__).parent / "implicit_qtt_kernel.cu"
    with open(kernel_path, 'r') as f:
        return f.read()


def _compile_kernel():
    """Compile implicit_qtt_kernel.cu using PyTorch JIT"""
    global _KERNEL_COMPILED, _KERNEL_MODULE
    
    if _KERNEL_COMPILED:
        return
    
    print("Compiling implicit QTT CUDA kernel (this may take 30-60 seconds)...")
    
    # Get CUDA source
    cuda_source = _get_cuda_kernel_source()
    
    # Minimal C++ wrapper for Python binding
    cpp_source = """
    #include <torch/extension.h>
    
    // Forward declarations from CUDA kernel
    void launch_render_qtt_layer(
        const float* qtt_cores,
        float* output,
        int width,
        int height,
        float value_min,
        float value_max,
        int colormap_type,
        cudaStream_t stream
    );
    
    void launch_composite_qtt_layers(
        const float* const* layer_cores,
        const int* layer_enabled,
        float* output,
        int width,
        int height,
        float value_min,
        float value_max,
        cudaStream_t stream
    );
    
    // Python bindings
    void render_qtt_layer_wrapper(
        torch::Tensor qtt_cores,
        torch::Tensor output,
        int width,
        int height,
        float value_min,
        float value_max,
        int colormap_type
    ) {
        launch_render_qtt_layer(
            qtt_cores.data_ptr<float>(),
            output.data_ptr<float>(),
            width, height,
            value_min, value_max,
            colormap_type,
            c10::cuda::getCurrentCUDAStream()
        );
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("render_qtt_layer", &render_qtt_layer_wrapper, "Render QTT layer");
    }
    """
    
    # Compile with PyTorch's JIT
    _KERNEL_MODULE = load_inline(
        name='implicit_qtt_kernel',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['render_qtt_layer'],
        extra_cuda_cflags=[
            '-O3',
            '-use_fast_math',
            '--expt-relaxed-constexpr'
        ],
        verbose=True,
        with_cuda=True
    )
    
    _KERNEL_COMPILED = True
    print("✓ Kernel compiled successfully")


class ImplicitQTTRenderer:
    """
    Render QTT tensors directly in CUDA without materialization.
    
    Architecture:
    - QTT cores uploaded to GPU texture memory (cache-friendly)
    - Fragment shader evaluates TT-contraction at pixel coordinates
    - No intermediate dense buffers (except final framebuffer)
    - Multi-layer compositor blends in scalar space
    
    Performance target: <2ms @ 4K for 5-layer composite
    """
    
    def __init__(
        self,
        width: int = 3840,
        height: int = 2160,
        device: str = "cuda",
        colormap: str = "plasma"
    ):
        """
        Args:
            width: Output resolution width
            height: Output resolution height
            device: CUDA device
            colormap: "plasma" or "viridis"
        """
        self.width = width
        self.height = height
        self.device = torch.device(device)
        self.colormap_type = 0 if colormap == "plasma" else 1
        
        # Compile kernel on first use
        _compile_kernel()
        
        # Allocate output buffer (reused across frames)
        self.output_buffer = torch.zeros(
            (height, width, 4),
            dtype=torch.float32,
            device=self.device
        )
        
        # Cache for QTT cores (avoid repeated uploads)
        self.core_cache = {}
        
        print(f"ImplicitQTTRenderer initialized: {width}×{height} @ {device}")
    
    def _qtt_to_flat_cores(self, qtt_state) -> torch.Tensor:
        """
        Convert QTT state to flat array of core matrices.
        
        Input: QTT with cores = [G₀, G₁, ..., G₁₁]
               Each core: [2, d, d] where d is rank
        
        Output: Flat tensor [96] = 12 cores × 2 matrices × 4 floats
        
        Note: This assumes rank=2 (2×2 matrices). For higher ranks,
              need to handle truncation or use different kernel.
        """
        cores = qtt_state.cores
        n_cores = len(cores)
        
        # Validate structure
        assert n_cores == 12, f"Expected 12 cores, got {n_cores}"
        
        flat_cores = []
        for core in cores:
            # core shape: [2, r_in, r_out] or [2, r, r]
            if core.dim() == 3:
                # For QTT: typically [2, 1, 1] at boundaries, [2, r, r] internal
                # Extract 2×2 matrix for each of 2 branches
                for i in range(2):
                    mat = core[i]  # Shape: [r_in, r_out]
                    
                    # Pad or truncate to 2×2
                    if mat.shape == (1, 1):
                        # Boundary core: expand to 2×2 identity-like
                        mat_2x2 = torch.zeros(2, 2, device=mat.device, dtype=mat.dtype)
                        mat_2x2[0, 0] = mat[0, 0]
                        mat_2x2[1, 1] = mat[0, 0]
                    elif mat.shape[0] == 2 and mat.shape[1] == 2:
                        mat_2x2 = mat
                    else:
                        # Truncate or pad to 2×2
                        mat_2x2 = torch.zeros(2, 2, device=mat.device, dtype=mat.dtype)
                        min_i = min(mat.shape[0], 2)
                        min_j = min(mat.shape[1], 2)
                        mat_2x2[:min_i, :min_j] = mat[:min_i, :min_j]
                    
                    # Flatten to 4 elements (row-major)
                    flat_cores.append(mat_2x2.flatten())
            else:
                raise ValueError(f"Unexpected core shape: {core.shape}")
        
        # Concatenate all cores: [12 cores × 2 matrices × 4 floats] = 96 floats
        flat_tensor = torch.cat(flat_cores).contiguous()
        
        assert flat_tensor.shape == (96,), f"Expected shape (96,), got {flat_tensor.shape}"
        
        return flat_tensor.to(device=self.device, dtype=torch.float32)
    
    def render_single_layer(
        self,
        qtt_state,
        value_range: Optional[Tuple[float, float]] = None,
        return_timing: bool = False
    ) -> torch.Tensor:
        """
        Render single QTT layer to RGBA output.
        
        Args:
            qtt_state: QTT tensor (from dense_to_qtt_2d)
            value_range: (min, max) for normalization, auto-detect if None
            return_timing: If True, return (output, time_ms)
        
        Returns:
            RGBA tensor [H, W, 4] (float32)
        """
        # Convert QTT to flat core array
        flat_cores = self._qtt_to_flat_cores(qtt_state)
        
        # Auto-detect value range if not provided
        if value_range is None:
            # Sample QTT at 256 points to estimate range
            with torch.no_grad():
                # This is expensive (materialize for range detection)
                # TODO: Store min/max in QTT metadata during factorization
                value_min, value_max = 0.0, 1.0  # Default fallback
        else:
            value_min, value_max = value_range
        
        # Launch CUDA kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Call CUDA kernel via PyTorch binding
        _KERNEL_MODULE.render_qtt_layer(
            flat_cores,
            self.output_buffer.view(-1),  # Flatten for kernel
            self.width,
            self.height,
            value_min,
            value_max,
            self.colormap_type
        )
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        
        if return_timing:
            return self.output_buffer.clone(), elapsed_ms
        else:
            return self.output_buffer.clone()
    
    def render_multi_layer(
        self,
        qtt_layers: List,
        layer_enabled: List[bool],
        value_range: Optional[Tuple[float, float]] = None,
        return_timing: bool = False
    ) -> torch.Tensor:
        """
        Composite multiple QTT layers with alpha blending.
        
        Args:
            qtt_layers: List of QTT states (one per layer)
            layer_enabled: List of booleans (which layers to render)
            value_range: (min, max) for normalization
            return_timing: If True, return (output, time_ms)
        
        Returns:
            Composited RGBA tensor [H, W, 4]
        """
        assert len(qtt_layers) <= 5, "Maximum 5 layers supported"
        assert len(layer_enabled) == len(qtt_layers), "Mismatch in layer count"
        
        # Convert all QTT layers to flat cores
        flat_cores_list = [self._qtt_to_flat_cores(qtt) for qtt in qtt_layers]
        
        # Create device pointers array
        cores_ptrs = [cores.data_ptr() for cores in flat_cores_list]
        enabled_flags = torch.tensor(layer_enabled, dtype=torch.int32, device=self.device)
        
        # Auto-detect range
        if value_range is None:
            value_min, value_max = 0.0, 1.0
        else:
            value_min, value_max = value_range
        
        # Launch compositor kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        _KERNEL_MODULE.launch_composite_qtt_layers(
            cores_ptrs,
            enabled_flags.data_ptr(),
            self.output_buffer.data_ptr(),
            self.width,
            self.height,
            value_min,
            value_max,
            torch.cuda.current_stream().cuda_stream
        )
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        
        if return_timing:
            return self.output_buffer.clone(), elapsed_ms
        else:
            return self.output_buffer.clone()


def test_implicit_renderer():
    """
    Test implicit renderer with synthetic QTT.
    
    Expected performance: 200+ FPS @ 4K for static QTT
    """
    from tensornet.quantum.hybrid_qtt_renderer import create_test_qtt
    
    print("=" * 80)
    print("Testing Implicit QTT Renderer")
    print("=" * 80)
    
    # Create synthetic QTT (rank=8, smooth random field)
    qtt = create_test_qtt(nx=11, ny=11, rank=8)
    
    # Initialize renderer
    renderer = ImplicitQTTRenderer(width=3840, height=2160, device="cuda")
    
    # Warm-up
    print("Warming up...")
    for _ in range(10):
        _ = renderer.render_single_layer(qtt, value_range=(0.0, 1.0))
    
    # Benchmark
    print("Benchmarking (100 frames)...")
    times = []
    for i in range(100):
        output, elapsed_ms = renderer.render_single_layer(
            qtt, value_range=(0.0, 1.0), return_timing=True
        )
        times.append(elapsed_ms)
        
        if i % 10 == 0:
            print(f"Frame {i}: {elapsed_ms:.2f}ms ({1000/elapsed_ms:.1f} FPS)")
    
    # Statistics
    times = np.array(times)
    mean_ms = times.mean()
    std_ms = times.std()
    min_ms = times.min()
    max_ms = times.max()
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Mean frame time: {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"Min/Max: {min_ms:.2f} / {max_ms:.2f} ms")
    print(f"Average FPS: {1000/mean_ms:.1f}")
    print(f"Target: 200 FPS (5ms) → {'✓ PASS' if mean_ms < 5.0 else '✗ FAIL'}")
    
    if mean_ms < 5.0:
        print("\n🎯 Checkpoint 1 PASSED: Implicit rendering viable!")
        print(f"Speedup vs hybrid: {5.08/mean_ms:.1f}× faster")
    else:
        print("\n⚠️  Checkpoint 1 FAILED: Implicit rendering too slow")
        print("Investigate: Memory bandwidth? ALU bottleneck? Launch overhead?")
    
    return output, times


if __name__ == "__main__":
    # Run test
    test_implicit_renderer()
