"""
OPERATION VALHALLA - Phase 2.3 Optimization
Build Script for cuSparse Pressure Solver Extension

Compiles CUDA kernel with PyTorch C++ extension interface.
Links against cuSparse and cuBLAS for production HPC performance.

Author: The Architect
Date: 2025-12-28
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA paths (adjust if needed)
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
cusparse_lib = os.path.join(cuda_home, "lib64")

setup(
    name="pressure_solver_cuda",
    ext_modules=[
        CUDAExtension(
            name="pressure_solver_cuda",
            sources=["ontic/gpu/csrc/pressure_solver.cu"],
            include_dirs=[
                os.path.join(cuda_home, "include"),
            ],
            library_dirs=[cusparse_lib],
            libraries=["cusparse", "cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_89",  # RTX 5070 (Ada Lovelace)
                    "--ptxas-options=-v",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
