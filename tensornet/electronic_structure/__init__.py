"""
Electronic Structure & Quantum Chemistry Package.

Domains VIII.1-VIII.7:
  - DFT (Kohn-Sham, LDA/PBE, pseudopotentials)
  - Beyond-DFT (HF, MP2, CCSD, CASSCF)
  - Semi-Empirical / Tight-Binding (DFTB, Extended Hückel)
  - Excited States (TDDFT, GW, BSE)
  - Response Properties (DFPT, polarisability, dielectric)
  - Relativistic (ZORA, SOC, Douglas-Kroll-Hess, Dirac)
  - Quantum Embedding (QM/MM, ONIOM, DFT+DMFT, projection-based)
"""

from tensornet.electronic_structure.dft import (
    LDAExchangeCorrelation,
    PBEExchangeCorrelation,
    KohnShamDFT1D,
    AndersonMixer,
    NormConservingPseudopotential,
)
from tensornet.electronic_structure.beyond_dft import (
    RestrictedHartreeFock,
    MP2Correlation,
    CCSDSolver,
    CASSCFSolver,
)
from tensornet.electronic_structure.tight_binding import (
    SlaterKosterTB,
    SCCDFTB,
    ExtendedHuckel,
)
from tensornet.electronic_structure.excited_states import (
    CasidaTDDFT,
    RealTimeTDDFT,
    GWApproximation,
    BetheSalpeterEquation,
)
from tensornet.electronic_structure.response import (
    DFPTSolver,
    Polarisability,
    DielectricFunction,
    BornEffectiveCharge,
)
from tensornet.electronic_structure.relativistic import (
    ZORAHamiltonian,
    SpinOrbitCoupling,
    DouglasKrollHess,
    Dirac4Component,
)
from tensornet.electronic_structure.embedding import (
    QMMMEmbedding,
    ONIOMEmbedding,
    DFTPlusDMFT,
    ProjectionEmbedding,
)

__all__ = [
    # DFT
    'LDAExchangeCorrelation', 'PBEExchangeCorrelation',
    'KohnShamDFT1D', 'AndersonMixer', 'NormConservingPseudopotential',
    # Beyond-DFT
    'RestrictedHartreeFock', 'MP2Correlation', 'CCSDSolver', 'CASSCFSolver',
    # Tight-Binding
    'SlaterKosterTB', 'SCCDFTB', 'ExtendedHuckel',
    # Excited States
    'CasidaTDDFT', 'RealTimeTDDFT', 'GWApproximation', 'BetheSalpeterEquation',
    # Response
    'DFPTSolver', 'Polarisability', 'DielectricFunction', 'BornEffectiveCharge',
    # Relativistic
    'ZORAHamiltonian', 'SpinOrbitCoupling', 'DouglasKrollHess', 'Dirac4Component',
    # Embedding
    'QMMMEmbedding', 'ONIOMEmbedding', 'DFTPlusDMFT', 'ProjectionEmbedding',
]
