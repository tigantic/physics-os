# HyperTensor Enterprise SDK

**Version**: 1.0.0  
**License**: Apache 2.0 / Enterprise

## Overview

The HyperTensor Enterprise SDK provides production-ready packaging, deployment tools, and enterprise features for integrating HyperTensor into commercial applications.

## Distribution Formats

| Format | Use Case | Location |
|--------|----------|----------|
| PyPI Wheel | Standard Python installation | `pip install hypertensor` |
| Conda Package | Scientific computing environments | `conda install -c tigantic hypertensor` |
| Docker Image | Containerized deployment | `docker pull tigantic/hypertensor` |
| C++ SDK | Native integration | `sdk/cpp/` |
| WebAssembly | Browser deployment | `sdk/wasm/` |

## Quick Start

### Python (PyPI)

```bash
pip install hypertensor

# With GPU support
pip install hypertensor[cuda]

# With all optional dependencies
pip install hypertensor[all]
```

### Conda

```bash
conda install -c tigantic hypertensor

# With CUDA
conda install -c tigantic hypertensor-cuda
```

### Docker

```bash
# CPU only
docker run -it tigantic/hypertensor

# With GPU
docker run --gpus all -it tigantic/hypertensor:cuda

# Jupyter notebook
docker run -p 8888:8888 tigantic/hypertensor:jupyter
```

## SDK Components

### Python SDK

Full Python package with:
- Core tensor network operations
- Field substrate and operators
- Simulation engines
- Visualization tools
- Python bridge servers

### C++ SDK

Native C++ library for:
- High-performance integration
- Real-time applications
- Game engine plugins
- Embedded systems

### WebAssembly SDK

Browser-compatible build for:
- Web applications
- Interactive demos
- Client-side visualization

## Enterprise Features

### License Management

```python
from hypertensor.enterprise import LicenseManager

# Activate enterprise license
LicenseManager.activate("ENTERPRISE-LICENSE-KEY")

# Check license status
status = LicenseManager.status()
print(f"License: {status.tier}, Expires: {status.expires}")
```

### Telemetry & Monitoring

```python
from hypertensor.enterprise import Telemetry

# Enable telemetry (opt-in)
Telemetry.enable(
    endpoint="https://metrics.yourcompany.com",
    api_key="YOUR_API_KEY"
)

# Custom metrics
Telemetry.record("simulation_step_time", 0.0045)
```

### Priority Support

Enterprise licenses include:
- 24/7 priority support
- Dedicated Slack channel
- Custom feature development
- Training and onboarding

## Building from Source

### Requirements

- Python 3.9+
- CMake 3.20+
- CUDA Toolkit 11.8+ (optional)
- Docker (for container builds)

### Build Steps

```bash
# Clone repository
git clone https://github.com/tigantic/HyperTensor.git
cd HyperTensor

# Build Python wheel
python -m build

# Build Docker image
docker build -t hypertensor -f sdk/docker/Dockerfile .

# Build C++ SDK
cd sdk/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## API Documentation

See [API Reference](https://hypertensor.readthedocs.io/) for complete documentation.

## Support

- **Community**: [GitHub Discussions](https://github.com/tigantic/HyperTensor/discussions)
- **Issues**: [GitHub Issues](https://github.com/tigantic/HyperTensor/issues)
- **Enterprise**: enterprise@tigantic.com
