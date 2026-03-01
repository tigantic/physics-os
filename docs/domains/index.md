# Domain Packs

The Ontic Engine organizes its 168 physics taxonomy nodes into installable
domain packs. Each pack can be installed independently via pip extras.

## Available Packs

| Pack | Install | Modules |
|------|---------|---------|
| CFD | `pip install ontic-engine[cfd]` | euler_1d, euler_2d, navier_stokes, godunov, LES |
| Quantum | `pip install ontic-engine[quantum]` | qm, qft, condensed_matter, statmech |
| Fluids | `pip install ontic-engine[fluids]` | multiphase, free_surface, fsi, heat_transfer |
| Materials | `pip install ontic-engine[materials]` | mechanics, phase_field |
| Aerospace | `pip install ontic-engine[aerospace]` | flight_validation, guidance, racing |
| Plasma/Nuclear | `pip install ontic-engine[plasma]` | plasma, nuclear, fusion, radiation |
| Life Sciences | (core) | biology, biomedical, biophysics, medical |
| Energy/Environment | (core) | energy, environmental, fuel |
| EM | `pip install ontic-engine[em]` | electromagnetics, optics, semiconductor, acoustics |
| ML | `pip install ontic-engine[ml]` | neural, discovery, surrogates, ml_physics |

## Install All

```bash
pip install ontic-engine[physics-all]
```
