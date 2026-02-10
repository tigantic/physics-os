"""
Tests for §1.2 — Domain Expansions
====================================

Covers: PEPS, MERA, NEGF, RelHydro, DEM, Caloric, Tribology, Fracture,
CombustionDNS, Magnetotellurics, LatticeQFT, ABM, BSSN, DynamicalHMC,
IMSRG/CC/NCSM, PartonShower, RadiationMHD, NWayCoupler.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── PEPS ────────────────────────────────────────────────────────────
class TestPEPS:
    def test_import(self):
        from tensornet.algorithms.peps import PEPSState, random_peps, simple_update_step

    def test_random_peps(self):
        from tensornet.algorithms.peps import random_peps
        peps = random_peps(Lx=3, Ly=3, d=2, D=4)
        assert peps is not None


# ── MERA ────────────────────────────────────────────────────────────
class TestMERA:
    def test_import(self):
        from tensornet.algorithms.mera import (
            MERAType,
            MERALayer,
            MERAState,
            random_binary_mera,
        )


# ── NEGF ────────────────────────────────────────────────────────────
class TestNEGF:
    def test_import(self):
        from tensornet.condensed_matter.negf import NEGFDevice, NEGFSolver, NEGFResult


# ── Relativistic Hydrodynamics ──────────────────────────────────────
class TestRelHydro:
    def test_import(self):
        from tensornet.relativity.rel_hydro import (
            EOSType,
            SRHDState,
            SRHDSolver,
            hll_flux,
        )


# ── DEM ─────────────────────────────────────────────────────────────
class TestDEM:
    def test_import(self):
        from tensornet.mechanics.dem import (
            ContactModel,
            MaterialProperties,
            DEMState,
            DEMSolver,
        )


# ── Caloric ─────────────────────────────────────────────────────────
class TestCaloric:
    def test_import(self):
        from tensornet.condensed_matter.caloric import (
            WeissFerromagnet,
            brillouin,
            BeanRodbell,
            LandauDevonshire,
        )

    def test_brillouin(self):
        from tensornet.condensed_matter.caloric import brillouin
        # For large x, B_J(x) → 1 (full saturation)
        val = brillouin(np.array([100.0]), J=3.5)
        assert abs(val[0] - 1.0) < 1e-6


# ── Tribology ───────────────────────────────────────────────────────
class TestTribology:
    def test_import(self):
        from tensornet.mechanics.tribology import (
            ArchardWear,
            GWContact,
            ReynoldsLubrication1D,
            RateStateFriction,
        )

    def test_archard_wear(self):
        from tensornet.mechanics.tribology import ArchardWear
        model = ArchardWear(k=1e-5, H=1e9)
        vol = model.wear_volume(F=100.0, L=0.01)
        assert vol > 0


# ── Fracture ────────────────────────────────────────────────────────
class TestFracture:
    def test_import(self):
        from tensornet.mechanics.fracture import (
            sif_edge_crack,
            sif_center_crack,
            ParisFatigue,
            j_integral_contour,
        )

    def test_sif_edge_crack(self):
        from tensornet.mechanics.fracture import sif_edge_crack
        K_I = sif_edge_crack(sigma=100.0, a=0.01, W=0.1)
        assert K_I > 0


# ── Combustion DNS ──────────────────────────────────────────────────
class TestCombustionDNS:
    def test_import(self):
        from tensornet.cfd.combustion_dns import (
            CombustionDNSSolver,
            CombustionState,
            hydrogen_air_9species,
            methane_skeletal,
        )


# ── Magnetotellurics ────────────────────────────────────────────────
class TestMagnetotellurics:
    def test_import(self):
        from tensornet.geophysics.magnetotellurics import (
            mt_forward_1d,
            OccamInversion1D,
            LayeredEarth,
        )

    def test_1d_forward(self):
        from tensornet.geophysics.magnetotellurics import mt_forward_1d, LayeredEarth
        # LayeredEarth uses sigma (conductivity) and h (thicknesses)
        earth = LayeredEarth(
            sigma=np.array([0.01, 0.1, 0.001]),
            h=np.array([500.0, 1000.0]),
        )
        freqs = np.logspace(-3, 1, 20)
        rho_a, phase = mt_forward_1d(earth, freqs)
        assert len(rho_a) == 20
        assert np.all(rho_a > 0)
        assert np.all(np.isfinite(phase))


# ── Lattice QFT ─────────────────────────────────────────────────────
class TestLatticeQFT:
    def test_import(self):
        from tensornet.qft.lattice_qft import (
            LatticeConfig,
            GaugeField,
            wilson_gauge_action,
        )

    def test_cold_start_action(self):
        from tensornet.qft.lattice_qft import (
            LatticeConfig,
            GaugeField,
            wilson_gauge_action,
        )
        cfg = LatticeConfig(dims=(4, 4), N_c=2, beta=2.0)
        gf = GaugeField.cold_start(cfg)
        S = wilson_gauge_action(gf)
        # Cold start: all plaquettes = identity → action near 0
        assert abs(S) < 1e-10


# ── ABM ─────────────────────────────────────────────────────────────
class TestABM:
    def test_import(self):
        from tensornet.biology.abm import SocialForceModel, BoidsSimulation, Agent

    def test_boids_step(self):
        from tensornet.biology.abm import BoidsSimulation, BoidsParams
        params = BoidsParams()
        sim = BoidsSimulation(n_agents=20, domain_size=10.0, params=params)
        sim.step(dt=0.1)
        assert sim.positions.shape == (20, 2)


# ── BSSN (NR extension) ────────────────────────────────────────────
class TestBSSN:
    def test_import(self):
        from tensornet.relativity.numerical_gr import BSSNEvolution, BSSNEvolver


# ── Dynamical HMC ──────────────────────────────────────────────────
class TestDynamicalHMC:
    def test_import(self):
        from tensornet.qft.lattice_qcd import DynamicalHMC


# ── Nuclear structure extensions ────────────────────────────────────
class TestNuclearExtensions:
    def test_imsrg_import(self):
        from tensornet.nuclear.structure import IMSRG

    def test_ccsd_import(self):
        from tensornet.nuclear.structure import CoupledClusterSD

    def test_ncsm_import(self):
        from tensornet.nuclear.structure import NCSM


# ── Parton shower ──────────────────────────────────────────────────
class TestPartonShower:
    def test_import(self):
        from tensornet.qft.perturbative import SplittingFunctions, PartonShower

    def test_splitting_pqq(self):
        from tensornet.qft.perturbative import SplittingFunctions
        sf = SplittingFunctions()
        p = sf.P_qq(0.5)
        assert np.isfinite(p)


# ── Radiation MHD ──────────────────────────────────────────────────
class TestRadiationMHD:
    def test_import(self):
        from tensornet.plasma.extended_mhd import RadiationMHD, RadiationTransport


# ── N-Way Coupler ──────────────────────────────────────────────────
class TestNWayCoupler:
    def test_import(self):
        from tensornet.platform.coupled import NWayCoupler, CouplingEdge
