# Domain Packs

The Ontic Engine organizes its 168 physics taxonomy nodes into installable
domain packs. Each pack can be installed independently via pip extras.

## Available Packs

| Pack | Install | Modules |
|------|---------|---------|
| CFD | `pip install tensornet[cfd]` | euler_1d, euler_2d, navier_stokes, godunov, LES |
| Quantum | `pip install tensornet[quantum]` | qm, qft, condensed_matter, statmech |
| Fluids | `pip install tensornet[fluids]` | multiphase, free_surface, fsi, heat_transfer |
| Materials | `pip install tensornet[materials]` | mechanics, phase_field |
| Aerospace | `pip install tensornet[aerospace]` | flight_validation, guidance, racing |
| Plasma/Nuclear | `pip install tensornet[plasma]` | plasma, nuclear, fusion, radiation |
| Life Sciences | (core) | biology, biomedical, biophysics, medical |
| Energy/Environment | (core) | energy, environmental, fuel |
| EM | `pip install tensornet[em]` | electromagnetics, optics, semiconductor, acoustics |
| ML | `pip install tensornet[ml]` | neural, discovery, surrogates, ml_physics |

## Install All

```bash
pip install tensornet[physics-all]
```
