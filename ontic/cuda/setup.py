"""
Phase 2B-4: CUDA Extension Build Script
========================================

Compiles the CUDA advection kernels into a Python-importable module.

Usage:
    cd ontic/cuda
    pip install .

    # Or for development:
    pip install -e .

    # Or direct build:
    python setup.py build_ext --inplace

After installation:
    import tensornet_cuda
    result = tensornet_cuda.advect_2d(density, velocity, dt)
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
here = os.path.dirname(os.path.abspath(__file__))


def get_extensions():
    """Build list of extension modules."""
    extensions = []

    # Main CUDA extension (advection)
    cuda_sources = [
        os.path.join(here, "bindings.cpp"),
        os.path.join(here, "advection_kernel.cu"),
    ]

    # Include the existing QTT kernel if present
    qtt_kernel = os.path.join(here, "qtt_eval_kernel.cu")
    if os.path.exists(qtt_kernel):
        # We'll compile it separately or include in main extension
        pass

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_86,code=sm_86",  # RTX 30 series (Ampere)
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-lineinfo",  # For profiling
    ]
    
    # Add Blackwell/Ada support if CUDA version supports it
    # These require CUDA 12.x+
    try:
        import torch
        cuda_version = torch.version.cuda
        if cuda_version and int(cuda_version.split('.')[0]) >= 12:
            nvcc_flags.extend([
                "-gencode=arch=compute_89,code=sm_89",  # RTX 40 series (Ada)
                "-gencode=arch=compute_120,code=sm_120",  # RTX 50 series (Blackwell)
            ])
    except:
        pass

    extensions.append(
        CUDAExtension(
            name="tensornet_cuda",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": ["-O3", "-Wall"],
                "nvcc": nvcc_flags,
            },
        )
    )
    
    # QTT Native CUDA extension (new - for CFD solver)
    qtt_native_kernel = os.path.join(here, "qtt_native_kernels.cu")
    if os.path.exists(qtt_native_kernel):
        extensions.append(
            CUDAExtension(
                name="qtt_native_cuda",
                sources=[qtt_native_kernel],
                extra_compile_args={
                    "cxx": ["-O3", "-Wall"],
                    "nvcc": nvcc_flags,
                },
            )
        )

    return extensions


setup(
    name="tensornet_cuda",
    version="0.2.0",
    description="CUDA kernels for TensorNet CFD acceleration",
    author="TiganticLabz",
    author_email="dev@tigantic.com",
    url="https://github.com/tigantic/physics-os",
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
