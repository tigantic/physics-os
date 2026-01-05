# Module `tensornet.certification`

Certification Module for HyperTensor ====================================

Provides DO-178C compliance infrastructure and hardware deployment
capabilities for safety-critical tensor network applications.

Modules:
    - do178c: DO-178C certification framework
    - hardware: Hardware deployment and optimization

Usage:
    from tensornet.certification import (
        DAL, VerificationMethod, RequirementsDatabase,
        HardwareSpec, deploy_to_hardware
    )

**Contents:**

- [Submodules](#submodules)

## Submodules

- [`certification.do178c`](#certification-do178c)
- [`certification.hardware`](#certification-hardware)
