# The Physics OS — Enterprise SDK

**Version**: 1.0.0  
**License**: Apache 2.0 / Enterprise

## Overview

Ontic Enterprise SDK provides production-ready packaging, deployment tools, and enterprise features for integrating The Physics OS into commercial applications.

## Distribution Formats

| Format | Use Case | Location |
|--------|----------|----------|
| PyPI Wheel | Standard Python installation | `pip install physics-os` |
| Conda Package | Scientific computing environments | `conda install -c tigantic ontic` |
| Docker Image | Containerized deployment | `docker pull tigantic/ontic` |
| C++ SDK | Native integration | `sdk/cpp/` |
| WebAssembly | Browser deployment | `sdk/wasm/` |

## Quick Start

### Python (PyPI)

```bash
pip install physics-os

# With GPU support
pip install physics-os[cuda]

# With all optional dependencies
pip install physics-os[all]
```

### Conda

```bash
conda install -c tigantic ontic

# With CUDA
conda install -c tigantic ontic-cuda
```

### Docker

```bash
# CPU only
docker run tigantic/ontic

# With GPU
docker run tigantic/ontic:cuda

# Jupyter notebook
docker run tigantic/ontic:jupyter
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
from physics_os.enterprise import LicenseManager

# Activate enterprise license
LicenseManager.activate("ENTERPRISE-LICENSE-KEY")

# Check license status
status = LicenseManager.status()
print(f"License: {status.tier}, Expires: {status.expires}")
```

### Telemetry & Monitoring

```python
from physics_os.enterprise import Telemetry

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
git clone https://github.com/tigantic/The Physics OS.git
cd The Ontic Engine

# Build Python wheel
python -m build

# Build Docker image
docker build -t ontic -f sdk/docker/Dockerfile .

# Build C++ SDK
cd sdk/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## API Documentation

See [API Reference](https://physics-os.readthedocs.io/) for complete documentation.

## Support

- **Community**: [GitHub Discussions](https://github.com/tigantic/physics-os/discussions)
- **Issues**: [GitHub Issues](https://github.com/tigantic/physics-os/issues)
- **Enterprise**: enterprise@tigantic.com
