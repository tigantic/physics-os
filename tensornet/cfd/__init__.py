"""
CFD Module
==========

Tensor network methods for computational fluid dynamics.

Phase 2: 1D Euler equations as proof of concept.
Phase 3: 2D Euler equations with dimensional splitting for hypersonic flows.
Phase 5: QTT compression for tensor network CFD coupling.
Phase 6: Navier-Stokes viscous terms for boundary layers and heat transfer.
Phase 7: Coupled NS, 3D Euler, and real-gas thermodynamics.
Phase 8: SBLI benchmarks, multi-species chemistry, reactive NS.
Phase 9: RANS turbulence, adjoint solver, shape optimization.
Phase 10: LES turbulence, hybrid RANS-LES, multi-objective optimization.
"""

from tensornet.cfd.euler_1d import (
    BCType1D,
    Euler1D,
    euler_to_mps,
    mps_to_euler,
    sod_shock_tube_ic,
    lax_shock_tube_ic,
    shu_osher_ic,
)

from tensornet.cfd.euler_2d import (
    Euler2D,
    Euler2DState,
    supersonic_wedge_ic,
    oblique_shock_exact,
    double_mach_reflection_ic,
)

from tensornet.cfd.godunov import (
    roe_flux,
    hll_flux,
    hllc_flux,
    exact_riemann,
)

from tensornet.cfd.limiters import (
    minmod,
    superbee,
    van_leer,
    mc_limiter,
)

from tensornet.cfd.boundaries import (
    BCType,
    FlowState,
    BoundaryManager,
    apply_reflective_bc,
    apply_supersonic_inflow_bc,
)

from tensornet.cfd.geometry import (
    WedgeGeometry,
    ImmersedBoundary,
    compute_pressure_coefficient,
    compute_drag_coefficient,
)

from tensornet.cfd.qtt import (
    field_to_qtt,
    qtt_to_field,
    euler_to_qtt,
    qtt_to_euler,
    compression_analysis,
    estimate_area_law_exponent,
    QTTCompressionResult,
    tt_svd,
)

# Phase 21+: QTT-CFD (Logarithmic Complexity - The Holy Grail)
from tensornet.cfd.qtt_cfd import (
    QTTCFDConfig,
    QTTEulerState,
    QTT_Euler1D,
    complexity_comparison,
)

from tensornet.cfd.viscous import (
    sutherland_viscosity,
    thermal_conductivity,
    compute_transport_properties,
    velocity_gradients_2d,
    temperature_gradient_2d,
    stress_tensor_2d,
    heat_flux_2d,
    viscous_flux_x_2d,
    viscous_flux_y_2d,
    viscous_flux_divergence_2d,
    compute_viscous_rhs_2d,
    reynolds_number,
    viscous_timestep_limit,
    recovery_temperature,
    stagnation_temperature,
    stanton_number,
    TransportProperties,
)

from tensornet.cfd.navier_stokes import (
    NavierStokes2D,
    NavierStokes2DConfig,
    NavierStokes2DResult,
    flat_plate_ic,
    compression_corner_ic,
)

from tensornet.cfd.euler_3d import (
    Euler3D,
    Euler3DState,
    hllc_flux_3d,
    uniform_flow_3d,
    sod_3d_ic,
)

from tensornet.cfd.real_gas import (
    gamma_variable,
    gamma_ideal,
    equilibrium_gamma_air,
    cp_polynomial,
    enthalpy_sensible,
    vibrational_energy,
    compute_real_gas_properties,
    specific_gas_constant,
    temperature_from_enthalpy,
    speed_of_sound_real,
    post_shock_equilibrium,
    GasProperties,
    MW,
    THETA_V,
    T_DISSOC,
)

from tensornet.cfd.chemistry import (
    Species,
    ArrheniusCoeffs,
    Reaction,
    REACTIONS,
    equilibrium_constant,
    compute_reaction_rates,
    ChemistryState,
    air_5species_ic,
    chemistry_timestep,
    advance_chemistry_explicit,
    post_shock_composition,
)

from tensornet.cfd.implicit import (
    ImplicitConfig,
    SolverStatus,
    NewtonResult,
    newton_solve,
    numerical_jacobian,
    ChemistryIntegrator,
    backward_euler_scalar,
    bdf2_scalar,
    AdaptiveImplicit,
)

from tensornet.cfd.reactive_ns import (
    ReactiveState,
    ReactiveConfig,
    ReactiveNS,
    reactive_flat_plate_ic,
)

from tensornet.cfd.turbulence import (
    TurbulenceModel,
    TurbulentState,
    k_epsilon_eddy_viscosity,
    k_epsilon_production,
    k_epsilon_source,
    k_omega_sst_eddy_viscosity,
    k_omega_sst_source,
    sst_blending_functions,
    k_omega_blending,
    spalart_allmaras_eddy_viscosity,
    spalart_allmaras_source,
    log_law_velocity,
    wall_function_tau,
    sarkar_correction,
    wilcox_compressibility,
    initialize_turbulence,
)

from tensornet.cfd.adjoint import (
    AdjointMethod,
    ObjectiveType,
    AdjointState,
    SensitivityResult,
    AdjointConfig,
    ObjectiveFunction,
    DragObjective,
    HeatFluxObjective,
    AdjointEuler2D,
    compute_shape_sensitivity,
)

from tensornet.cfd.optimization import (
    OptimizerType,
    ConstraintType,
    OptimizationConfig,
    ConstraintSpec,
    OptimizationResult,
    GeometryParameterization,
    BSplineParameterization,
    FFDParameterization,
    ShapeOptimizer,
    create_wedge_design_problem,
)

from tensornet.cfd.les import (
    LESModel,
    LESState,
    filter_width,
    strain_rate_magnitude,
    vorticity_magnitude,
    smagorinsky_viscosity,
    van_driest_damping,
    dynamic_smagorinsky_coefficient,
    wale_viscosity,
    vreman_viscosity,
    sigma_viscosity,
    sgs_heat_flux,
    compute_sgs_viscosity,
)

from tensornet.cfd.hybrid_les import (
    HybridModel,
    HybridLESState,
    compute_grid_scale,
    compute_wall_distance_scale,
    des_length_scale,
    compute_r_d,
    ddes_delay_function,
    ddes_length_scale,
    iddes_blending_function,
    iddes_length_scale,
    sas_length_scale,
    compute_hybrid_viscosity,
    run_hybrid_les,
    estimate_rans_les_ratio,
)

from tensornet.cfd.multi_objective import (
    MOOAlgorithm,
    ObjectiveSpec,
    ParetoSolution,
    MOOResult,
    MOOConfig,
    dominates,
    fast_non_dominated_sort,
    crowding_distance,
    hypervolume_2d,
    MultiObjectiveOptimizer,
    create_drag_heating_problem,
)

# Phase 21: WENO/TENO Shock Capturing
from tensornet.cfd.weno import (
    smoothness_indicators,
    optimal_weights_left,
    optimal_weights_right,
    nonlinear_weights_js,
    nonlinear_weights_z,
    candidate_stencils_left,
    candidate_stencils_right,
    weno5_js,
    weno5_z,
    teno5,
    weno_reconstruct_euler,
)

# Phase 21: WENO-TT Tensorized Reconstruction
from tensornet.cfd.weno_tt import (
    tensorize_smoothness_indicators,
    tensorize_weights,
    weno_tt_reconstruct,
    apply_weno_tt_flux,
    euler_weno_tt_flux,
    WENOTTConfig,
)

# Phase 21: TT-CFD Core (TDVP-CFD Integration)
from tensornet.cfd.tt_cfd import (
    MPSState,
    EulerMPO,
    tdvp_euler_step,
    TT_Euler1D,
    TT_Euler2D,
    check_conservation,
)

# Phase 21: TT-AMR Adaptive Bonds
from tensornet.cfd.adaptive_tt import (
    ShockDetector,
    BondAdapter,
    AdaptiveTTEuler,
    EntanglementMonitor,
)

__all__ = [
    # 1D Euler equations
    'BCType1D',
    'Euler1D',
    'euler_to_mps',
    'mps_to_euler',
    'sod_shock_tube_ic',
    'lax_shock_tube_ic',
    'shu_osher_ic',
    # 2D Euler equations
    'Euler2D',
    'Euler2DState',
    'supersonic_wedge_ic',
    'oblique_shock_exact',
    'double_mach_reflection_ic',
    # Riemann solvers
    'roe_flux',
    'hll_flux',
    'hllc_flux',
    'exact_riemann',
    # Limiters
    'minmod',
    'superbee',
    'van_leer',
    'mc_limiter',
    # Boundary conditions
    'BCType',
    'FlowState',
    'BoundaryManager',
    'apply_reflective_bc',
    'apply_supersonic_inflow_bc',
    # Geometry
    'WedgeGeometry',
    'ImmersedBoundary',
    'compute_pressure_coefficient',
    'compute_drag_coefficient',
    # QTT Compression (Phase 5)
    'field_to_qtt',
    'qtt_to_field',
    'euler_to_qtt',
    'qtt_to_euler',
    'compression_analysis',
    'estimate_area_law_exponent',
    'QTTCompressionResult',
    'tt_svd',
    # Navier-Stokes Viscous (Phase 6)
    'sutherland_viscosity',
    'thermal_conductivity',
    'compute_transport_properties',
    'velocity_gradients_2d',
    'temperature_gradient_2d',
    'stress_tensor_2d',
    'heat_flux_2d',
    'viscous_flux_x_2d',
    'viscous_flux_y_2d',
    'viscous_flux_divergence_2d',
    'compute_viscous_rhs_2d',
    'reynolds_number',
    'viscous_timestep_limit',
    'recovery_temperature',
    'stagnation_temperature',
    'stanton_number',
    'TransportProperties',
    # Coupled Navier-Stokes (Phase 7)
    'NavierStokes2D',
    'NavierStokes2DConfig',
    'NavierStokes2DResult',
    'flat_plate_ic',
    'compression_corner_ic',
    # 3D Euler (Phase 7)
    'Euler3D',
    'Euler3DState',
    'hllc_flux_3d',
    'uniform_flow_3d',
    'sod_3d_ic',
    # Real-Gas Thermodynamics (Phase 7)
    'gamma_variable',
    'gamma_ideal',
    'equilibrium_gamma_air',
    'cp_polynomial',
    'enthalpy_sensible',
    'vibrational_energy',
    'compute_real_gas_properties',
    'specific_gas_constant',
    'temperature_from_enthalpy',
    'speed_of_sound_real',
    'post_shock_equilibrium',
    'GasProperties',
    'MW',
    'THETA_V',
    'T_DISSOC',
    # Multi-Species Chemistry (Phase 8)
    'Species',
    'ArrheniusCoeffs',
    'Reaction',
    'REACTIONS',
    'equilibrium_constant',
    'compute_reaction_rates',
    'ChemistryState',
    'air_5species_ic',
    'chemistry_timestep',
    'advance_chemistry_explicit',
    'post_shock_composition',
    # Implicit Integration (Phase 8)
    'ImplicitConfig',
    'SolverStatus',
    'NewtonResult',
    'newton_solve',
    'numerical_jacobian',
    'ChemistryIntegrator',
    'backward_euler_scalar',
    'bdf2_scalar',
    'AdaptiveImplicit',
    # Reactive Navier-Stokes (Phase 8)
    'ReactiveState',
    'ReactiveConfig',
    'ReactiveNS',
    'reactive_flat_plate_ic',
    # RANS Turbulence (Phase 9)
    'TurbulenceModel',
    'TurbulentState',
    'k_epsilon_eddy_viscosity',
    'k_epsilon_production',
    'k_epsilon_source',
    'k_omega_sst_eddy_viscosity',
    'k_omega_sst_source',
    'sst_blending_functions',
    'k_omega_blending',
    'spalart_allmaras_eddy_viscosity',
    'spalart_allmaras_source',
    'log_law_velocity',
    'wall_function_tau',
    'sarkar_correction',
    'wilcox_compressibility',
    'initialize_turbulence',
    # Adjoint Solver (Phase 9)
    'AdjointMethod',
    'ObjectiveType',
    'AdjointState',
    'SensitivityResult',
    'AdjointConfig',
    'ObjectiveFunction',
    'DragObjective',
    'HeatFluxObjective',
    'AdjointEuler2D',
    'compute_shape_sensitivity',
    # Shape Optimization (Phase 9)
    'OptimizerType',
    'ConstraintType',
    'OptimizationConfig',
    'ConstraintSpec',
    'OptimizationResult',
    'GeometryParameterization',
    'BSplineParameterization',
    'FFDParameterization',
    'ShapeOptimizer',
    'create_wedge_design_problem',
    # LES Turbulence (Phase 10)
    'LESModel',
    'LESState',
    'filter_width',
    'strain_rate_magnitude',
    'vorticity_magnitude',
    'smagorinsky_viscosity',
    'van_driest_damping',
    'dynamic_smagorinsky_coefficient',
    'wale_viscosity',
    'vreman_viscosity',
    'sigma_viscosity',
    'sgs_heat_flux',
    'compute_sgs_viscosity',
    # Hybrid RANS-LES (Phase 10)
    'HybridModel',
    'HybridLESState',
    'compute_grid_scale',
    'compute_wall_distance_scale',
    'des_length_scale',
    'compute_r_d',
    'ddes_delay_function',
    'ddes_length_scale',
    'iddes_blending_function',
    'iddes_length_scale',
    'sas_length_scale',
    'compute_hybrid_viscosity',
    'run_hybrid_les',
    'estimate_rans_les_ratio',
    # Multi-Objective Optimization (Phase 10)
    'MOOAlgorithm',
    'ObjectiveSpec',
    'ParetoSolution',
    'MOOResult',
    'MOOConfig',
    'dominates',
    'fast_non_dominated_sort',
    'crowding_distance',
    'hypervolume_2d',
    'MultiObjectiveOptimizer',
    'create_drag_heating_problem',
    # WENO/TENO Shock Capturing (Phase 21)
    'smoothness_indicators',
    'optimal_weights_left',
    'optimal_weights_right',
    'nonlinear_weights_js',
    'nonlinear_weights_z',
    'candidate_stencils_left',
    'candidate_stencils_right',
    'weno5_js',
    'weno5_z',
    'teno5',
    'weno_reconstruct_euler',
    # WENO-TT Tensorized Reconstruction (Phase 21)
    'tensorize_smoothness_indicators',
    'tensorize_weights',
    'weno_tt_reconstruct',
    'apply_weno_tt_flux',
    'euler_weno_tt_flux',
    'WENOTTConfig',
    # TT-CFD Core (Phase 21)
    'MPSState',
    'EulerMPO',
    'tdvp_euler_step',
    'TT_Euler1D',
    'TT_Euler2D',
    'check_conservation',
    # TT-AMR Adaptive Bonds (Phase 21)
    'ShockDetector',
    'BondAdapter',
    'AdaptiveTTEuler',
    'EntanglementMonitor',
]
