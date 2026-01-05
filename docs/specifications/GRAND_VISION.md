Project HyperTensor: Real-Time Hypersonic Computational Fluid Dynamics on the Edge
Executive Summary
The strategic supremacy in the twenty-first-century aerospace domain is increasingly defined by the mastery of the hypersonic regime—flight velocities exceeding Mach 5. At these speeds, the conventional assumptions of aerodynamics dissolve into a complex interplay of non-equilibrium thermodynamics, chemical dissociation, and magnetohydrodynamics. The current generation of hypersonic glide vehicles (HGVs) and cruise missiles operates on a paradigm of pre-computation: flight control laws are derived from static wind tunnel databases and interpolated lookup tables. This approach, while sufficient for ballistic trajectories, is dangerously brittle when engaged in aggressive maneuvering or when encountering unforeseen atmospheric anomalies. The vehicle is effectively blind to the immediate, evolving physics of its own environment.
Project HyperTensor proposes a radical architectural shift: the embedding of a high-fidelity, real-time Computational Fluid Dynamics (CFD) simulator directly into the guidance and control loop of the missile. This system, colloquially described as "putting a wind tunnel inside the missile," aims to solve the full three-dimensional Navier-Stokes equations in milliseconds, providing the autopilot with a predictive, physics-based model of the flow field as it evolves.
The realization of this "impossible" goal rests on two convergent technological breakthroughs. First, the application of Quantum-Inspired Tensor Networks—specifically Matrix Product States (MPS) and Tensor Trains (TT)—to classical fluid dynamics. This mathematical framework allows for the compression of the exponentially vast state space of turbulent fluids into low-rank manifolds, reducing computational complexity from exponential to polynomial. Second, the maturation of edge-compute hardware, exemplified by the NVIDIA Jetson AGX Orin Industrial, which brings data-center-class tensor acceleration to a form factor compatible with the Size, Weight, and Power (SWaP) constraints of a tactical missile.
This report provides an exhaustive technical analysis of the HyperTensor architecture. It details the aerothermodynamic challenges of Mach 10 flight, the mathematical machinery of the Tensor Train Navier-Stokes solver, the hardware specification for edge implementation, and the operational revolutions enabled by this technology—specifically in solving the plasma blackout communications crisis and achieving zero-drift inertial navigation through aerodynamic terrain matching.
1. The Hypersonic Aerothermodynamic Environment
To understand the necessity of real-time CFD, one must first appreciate the extreme hostility and complexity of the hypersonic environment. Flight at Mach 10 (approximately 3.4 km/s at altitude) is not merely "faster" supersonic flight; it is a distinct physical regime where the kinetic energy of the flow is sufficient to tear apart the molecular structure of the air itself.
1.1 The Failure of the Ideal Gas Law
In subsonic and low-supersonic design, air is treated as a calorically perfect gas—a continuous fluid where pressure, density, and temperature are related by simple constants. At Mach 10, the shock wave formed ahead of the vehicle's nose compresses the air so violently that the post-shock temperature can exceed 4,000 Kelvin.1 At these temperatures, the internal energy modes of the gas molecules (vibration and rotation) become excited.
As the temperature climbs further, nitrogen ($N_2$) and oxygen ($O_2$) molecules begin to dissociate into atomic species ($N$ and $O$). This dissociation is an endothermic process, absorbing massive amounts of heat from the flow and drastically altering the ratio of specific heats ($\gamma$). A control system relying on a lookup table derived from "perfect gas" assumptions will miscalculate the pressure distribution, potentially leading to a loss of lift or control reversal during a critical maneuver.1 The vehicle essentially flies through a chemically reacting soup of its own making, where the fluid properties are changing on microsecond timescales.
1.2 The Plasma Sheath and Communications Blackout
Perhaps the most critical operational constraint in hypersonics is the "plasma blackout." As the shock layer temperature ionizes the air, a sheath of free electrons forms around the fuselage. This plasma has a characteristic frequency, the plasma frequency ($\omega_{pe}$), which is proportional to the square root of the electron density ($n_e$):

$$\omega_{pe} \approx \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}$$
When the frequency of an incoming or outgoing radio signal (such as GPS L-band at 1.5 GHz or telemetry S-band at 2.2 GHz) is lower than $\omega_{pe}$, the signal is reflected by the sheath as if it were hitting a mirror.3
At Mach 10, the electron density in the stagnation region can exceed $10^{13} \text{ cm}^{-3}$, resulting in a plasma frequency well above 10 GHz (X-band). This effectively blinds the missile. It cannot receive target updates, it cannot receive GPS corrections to its inertial navigation system, and it cannot transmit kill assessment data. Current mitigation strategies are passive and primitive: adding heavy magnetic coils to manipulate the sheath or simply accepting the blackout and flying autonomously with degrading accuracy.5 There is currently no active system capable of predicting the dynamic morphology of this sheath in real-time to exploit momentary "windows" of lower ionization that occur during maneuvers.
1.3 The Viscous Interaction and Boundary Layer Transition
In hypersonics, the boundary layer—the thin layer of viscous air clinging to the vehicle surface—grows rapidly. The "displacement thickness" of this layer becomes so large that it acts like a physical wedge, pushing the shock wave further out and altering the pressure distribution over the entire vehicle. This "viscous interaction" couples the aerodynamics to the thermodynamics; a hot vehicle surface creates a different pressure profile than a cool one.2
Furthermore, the transition from laminar to turbulent flow in the boundary layer creates massive spikes in heat transfer—up to ten times the laminar rate. Predicting exactly where this transition occurs is the "Holy Grail" of hypersonic design. A pre-computed database assumes a fixed transition point. If the transition happens earlier (due to surface roughness from ablation or atmospheric dust), the increased drag and heating can destroy the vehicle or cause it to fall short of its target.1 Only a real-time solver, continuously updating based on sensor feedback, can manage this uncertainty.
2. The Computational Standoff: Why Classical Methods Fail
The proposal to run CFD on a missile is met with skepticism because solving the Navier-Stokes equations is computationally exorbitant. This section analyzes why traditional approaches fail in the edge environment and establishes the need for a paradigm shift.
2.1 The Curse of Dimensionality
The complexity of a Direct Numerical Simulation (DNS) of turbulence scales with the Reynolds number ($Re$). The number of grid points ($N$) required to resolve all turbulent scales grows as $N \sim Re^{9/4}$. For a typical hypersonic vehicle ($L \approx 3$m) at Mach 10 and 30km altitude, $Re$ is on the order of $10^7$.7
This implies a computational grid of trillions of cells. Storing the state vector (density, momentum, energy) for such a grid requires petabytes of memory (RAM). Advancing the solution in time requires exaflops of processing power—capabilities found only in the world's largest supercomputers, which consume megawatts of power and occupy warehouse-sized facilities. A tactical missile has a power budget of roughly 50-100 Watts and a volume constraint of a few cubic liters.
2.2 The Latency of Reduced Order Models (ROMs)
To bypass the cost of DNS, engineers use Reduced Order Models. The most common is the aerodynamic lookup table (LUT). Thousands of wind tunnel hours are distilled into a static database of coefficients ($C_L, C_D, C_m$) indexed by Mach, Angle of Attack, and Sideslip.9
However, LUTs are inherently "sparse" and "static." They cannot capture:
Hysteresis: The aerodynamic forces often depend on the history of the motion (e.g., dynamic stall), not just the instantaneous state.
Damage: If a control fin is damaged by kinetic impact, the LUT is instantly invalid.
Unsteady Interactions: Shock-shock interactions (Type III or Type IV interference) can occur during maneuvers, creating localized pressure spikes that the LUT smooths over.11
2.3 The False Promise of Deep Learning (PINNs)
In recent years, Physics-Informed Neural Networks (PINNs) have been touted as a solution. A neural network is trained to approximate the solution of the PDEs. While fast at inference, PINNs suffer from critical defects for safety-critical flight control:
Spectral Bias: Neural networks struggle to learn high-frequency features. In hypersonics, the shock wave is a high-frequency discontinuity. PINNs often "blur" the shock, leading to inaccurate drag predictions.12
Lack of Conservation: A neural network minimizes a "loss function." It does not mathematically guarantee the conservation of mass, momentum, and energy. A 1% error in energy conservation in a hypersonic flow can translate to a 1000K error in temperature prediction.
Generalization Failure: A PINN is only as good as its training data. If the missile encounters a flow regime outside the training set (e.g., a specific combination of yaw and roll at high altitude), the network's output is unpredictable and often physically nonsensical.14
The HyperTensor project rejects these approximations in favor of a solver that is both physically rigorous and computationally efficient.
3. The HyperTensor Mathematical Engine: Tensor Networks
The core innovation of HyperTensor is the application of Tensor Network (TN) theory to the compressible Navier-Stokes equations. Originally developed for quantum many-body physics to describe the entanglement of quantum states, TNs provide a mathematical framework for breaking the "curse of dimensionality" in classical fluid dynamics.
3.1 From Quantum Entanglement to Fluid Turbulence
In quantum physics, the state of a many-particle system is described by a wave function that lives in a Hilbert space of exponential dimension. However, physical states of interest (ground states) are not random; they exhibit limited "entanglement." The "Area Law" states that the entanglement entropy of a region scales with its boundary area, not its volume.16
HyperTensor leverages the insight that turbulent fluids satisfy a similar Area Law. The correlations in a fluid flow—the structures like eddies, vortices, and shocks—are not random noise. They possess a hierarchical structure. This means the information content of the flow is much lower than the raw grid dimension suggests. By representing the fluid state vector as a Matrix Product State (MPS) or Tensor Train (TT), we can compress the data by orders of magnitude while preserving the essential physics.7
3.2 The Tensor Train (TT) Decomposition
In the standard Finite Volume Method, the fluid state $\mathcal{U}$ is a tensor of order $d$ (where $d$ is the number of spatial dimensions, usually 3). The total elements are $N^d$.
In the Tensor Train format, this massive tensor is decomposed into a chain of $d$ smaller tensors connected by "bonds":

$$\mathcal{U}(i_1, i_2, \dots, i_d) \approx \sum_{\alpha_1, \dots, \alpha_{d-1}} G_1(i_1, \alpha_1) G_2(\alpha_1, i_2, \alpha_2) \cdots G_d(\alpha_{d-1}, i_d)$$
Here, the indices $i_k$ represent the physical grid coordinates, and the indices $\alpha_k$ are the auxiliary "bond indices." The size of these bonds, the bond dimension ($D$), determines the accuracy of the approximation.
Compression: Storage scales as $d \cdot N \cdot D^2$ instead of $N^d$. For a typical grid ($N=512$, $D=100$), this reduces memory usage from gigabytes to megabytes.18
Operations: The Navier-Stokes equations involve linear and non-linear operators (gradients, advection). In the TT format, these operators are also decomposed into Matrix Product Operators (MPO). Applying an operator to a state (e.g., calculating fluxes) becomes a series of small matrix multiplications, the cost of which scales linearly with grid size.19
3.3 The Time-Dependent Variational Principle (TDVP)
To simulate the flight in real-time, the system must advance the state $\mathcal{U}$ in time. Standard explicit time-stepping (like Runge-Kutta) is inefficient for Tensor Networks because adding two tensors increases the bond dimension, requiring a costly "re-compression" step (SVD) that introduces errors.
HyperTensor utilizes the Time-Dependent Variational Principle (TDVP). Instead of leaving the tensor manifold, TDVP projects the evolution equation directly onto the tangent space of the tensor manifold of fixed rank.20

$$\frac{d}{dt}|\Psi(t)\rangle = -i P_{T_{|\Psi(t)\rangle}} \hat{L} |\Psi(t)\rangle$$
Where $\hat{L}$ is the Liouvillian operator representing the discretized Navier-Stokes equations.
Optimality: TDVP finds the "best possible" evolution that fits within the memory constraints (bond dimension).
Stability: It preserves the geometric properties of the flow (symplectic integration) and is exceptionally stable for "stiff" equations like chemical kinetics, where reaction rates are orders of magnitude faster than fluid motion.21
Adaptivity: The algorithm can dynamically adjust the bond dimension. In smooth laminar regions, $D$ is kept low (saving compute). Near a shock wave or in a turbulent wake, $D$ is increased locally to capture the complexity. This is the tensor equivalent of Adaptive Mesh Refinement (AMR).22
3.4 Handling Shocks: The "Gibbs" Phenomenon Resolution
A major challenge in compressing functions with discontinuities (like shock waves) is the Gibbs phenomenon—spurious oscillations near the jump. In a guidance system, these oscillations could be misinterpreted as turbulence, causing the fins to flutter.
HyperTensor integrates WENO-TT (Weighted Essentially Non-Oscillatory Tensor Train) schemes. The high-order reconstruction weights of the WENO scheme are themselves tensorized. This allows the solver to capture shocks with 5th-order accuracy without oscillation, even in the highly compressed TT format. The tensor network naturally "entangles" the state across the shock, effectively encoding the Rankine-Hugoniot jump conditions into the bond structure.24
4. Hardware Architecture: The Edge of the Edge
The mathematical elegance of Tensor Networks must be matched by a hardware platform capable of executing these algorithms within the harsh constraints of a missile airframe. The NVIDIA Jetson AGX Orin Industrial is identified as the enabling hardware.
4.1 The Compute Platform Specification
The Jetson AGX Orin is a System-on-Module (SoM) that brings server-class AI performance to the edge.

Component
Specification
Operational Relevance
GPU
NVIDIA Ampere, 2048 CUDA Cores
Massive parallel throughput for tensor element updates.
Tensor Cores
64 (3rd Gen)
Hardware acceleration for dense matrix multiplications ($D \times D$), the bottleneck of TT algorithms.
AI Performance
248 TOPS (INT8) / 1.26 TFLOPS (FP16)
Sufficient for high-speed, mixed-precision fluid simulation.
Memory
64 GB LPDDR5 (204 GB/s)
High bandwidth is critical for feeding the Tensor Cores. ECC protects against soft errors.
CPU
12-core ARM Cortex-A78AE
Handles the GNC loop, telemetry, and OS tasks.
Power
15W - 75W Configurable
Can be throttled based on mission phase (e.g., low power during cruise, max power during terminal homing).
Ruggedization
50G Shock / 5G Vibration
Rated for launch loads and boost-phase vibration.27

4.2 Tensor Core Acceleration of Fluid Dynamics
The primary operation in the TDVP algorithm is the contraction of core tensors. This involves multiplying matrices of size $D \times D$.
Mapping: The HyperTensor software stack (built on CUDA and cuTensor) maps the tensor blocks directly to the Orin's Tensor Cores.
Efficiency: While standard CUDA cores compute one floating-point operation per cycle, Tensor Cores compute a full $4 \times 4$ matrix multiply-accumulate in one cycle. This provides a theoretical speedup of over $10\times$ for the specific linear algebra dominating the CFD solver.28
Mixed Precision Strategy: To maximize speed, the solver uses Mixed Precision. The bulk of the flow evolution is computed in FP16 (Half Precision). Since fluid dynamics is often limited by discretization error rather than floating-point error, FP16 is sufficient for the "updates." Critical conservation sums are accumulated in FP32. This doubles the effective memory bandwidth and computational throughput.19
4.3 Radiation Hardening Strategy
Operating at 30-50km altitude places the electronics in a region of elevated cosmic ray flux (the Pfotzer maximum). Single Event Upsets (SEUs)—bit flips caused by high-energy particles—are a major risk. Standard "Rad-Hard" chips (like BAE Systems' RAD750) are generations behind in performance and cannot support HyperTensor.
HyperTensor employs a Software-Defined Radiation Hardening approach on the Commercial Off-The-Shelf (COTS) Orin module 30:
Triple Modular Redundancy (TMR) on GPU: The critical tensor update kernel is launched as three independent streams on physically separate partitions of the GPU (using NVIDIA's Multi-Instance GPU or spatial partitioning). A voter kernel compares the result of the three streams at each time step. If one differs, it is overwritten by the consensus of the other two.
Algorithmic Sanity Checks: The physics itself acts as a check. The solver monitors the total energy and mass of the system. A bit flip in the exponent of a float will create a massive, non-physical spike in energy. The TDVP algorithm detects this violation of the "manifold" constraints and reverts to the last valid checkpoint (saved every 10 steps).30
ECC Memory: The Orin's LPDDR5 supports inline ECC, correcting single-bit errors in memory before they reach the processor.27
5. Validation: The Sod Shock Tube Benchmark
Before deployment on a missile, the Tensor Network solver must be validated against canonical fluid dynamics problems. The Sod Shock Tube is the standard litmus test for compressible flow solvers.
5.1 Problem Setup
The Sod problem consists of a 1D tube of gas initially separated by a diaphragm at $x=0.5$.
Left State (High Pressure): $\rho_L = 1.0, P_L = 1.0, u_L = 0.0$
Right State (Low Pressure): $\rho_R = 0.125, P_R = 0.1, u_R = 0.0$
At $t=0$, the diaphragm bursts. The exact solution contains three distinct features: a Rarefaction Wave moving left, a Contact Discontinuity moving right, and a Shock Wave moving right.31
5.2 Tensor Network Performance
When solved using the HyperTensor MPS-WENO algorithm:
Shock Resolution: The solver accurately captures the shock front within 2-3 grid points, matching the performance of high-order Finite Volume methods. Crucially, there are no spurious oscillations (Gibbs ringing) behind the shock.24
Rank Adaptivity: Analysis of the bond dimensions during the simulation reveals the "intelligence" of the algorithm. In the smooth regions (left and right of the waves), the bond dimension remains close to 1 (indicating no correlation/entanglement). At the shock and contact discontinuity, the bond dimension automatically spikes (e.g., to $D=10-20$). This confirms that the algorithm dynamically allocates computational resources exactly where the physics demands it, achieving massive compression ratios compared to dense grids.25
Comparison to Standard Methods: For a fixed accuracy, the MPS solver uses significantly fewer parameters than a standard mesh, and the scaling benefits become exponential as the problem is extended to 2D and 3D.7
This benchmark confirms that the Tensor Network approach is not just a theoretical curiosity but a robust numerical method capable of handling the discontinuities inherent in hypersonic flight.
6. Operational Application I: Solving Plasma Blackout
The primary operational breakthrough of HyperTensor is the mitigation of plasma blackout. This is not solved by new antennas, but by computational foresight.
6.1 The Physics of Signal Loss
As discussed in Section 1.2, the plasma sheath blocks RF communications when $f_{signal} < f_{plasma}$. The sheath is not uniform; it is thickest at the nose (stagnation point) and thinner in the wake. However, the wake structure is highly dynamic. During a high-G turn, the "clean" window shifts rapidly. A static antenna selection logic cannot keep up.3
6.2 Real-Time Sheath Mapping
The HyperTensor system continuously simulates the electron density field $n_e(\mathbf{x}, t)$ around the vehicle.
Chemistry Model: The solver includes a 5-species chemical kinetics model ($N_2, O_2, NO, N, O$) coupled to the fluid flow. The "stiff" source terms of the chemical reactions are handled efficiently by the TDVP time-stepper.21
Mapping: The 3D electron density field is projected onto the vehicle surface to create a "Blackout Map." This map shows the instantaneous attenuation (in dB) for each installed antenna array.
6.3 Dynamic Commutation Strategy
The guidance computer uses this map to execute a Cognitive Communications strategy:
Smart Switching: The system continuously switches the transmitter to the antenna with the lowest predicted attenuation. This switching happens on the millisecond timescale, far faster than a "search and lock" hardware loop could achieve.
Attitude Modulation: If the mission requires a critical "check-in" (e.g., receiving a final target update or abort code) and all antennas are blocked, the HyperTensor system calculates a specific maneuver (e.g., a sideslip or "wiffle") that will momentarily shed the plasma from a specific sector. It "flies the vehicle to communicate," creating a synthetic window.5
Frequency Hopping: The simulation predicts the exact plasma frequency. If the system has a variable-frequency transmitter, it can tune the frequency to just above the local $\omega_{pe}$ threshold to punch through the sheath with minimum power.3
7. Operational Application II: Aero-TRN Navigation
The second breakthrough is Aerodynamic Terrain Relative Navigation (Aero-TRN). In a GPS-denied environment, inertial drift is the enemy. Over a 30-minute flight, a tactical grade IMU can drift by kilometers.
7.1 The Atmosphere as a Map
Terrain Relative Navigation (TRN) uses radar to match the ground below to a map. This is impossible at Mach 10 and 40km altitude (too high, too much plasma noise). However, the "terrain" the missile is interacting with is the atmosphere itself.
The pressure distribution over the vehicle is a unique fingerprint of its state vector (Velocity, Angle of Attack, Sideslip, Altitude/Density).
Sensors: The missile is equipped with a Flush Air Data System (FADS)—a matrix of pressure ports on the nose and chines.9
The Loop: The HyperTensor CFD simulates the expected pressure distribution for the IMU's estimated position and velocity.
7.2 The Inverse Problem
There will be a discrepancy between the measured pressure and the simulated pressure. This discrepancy is the error signal.
Because the Tensor Network solver is essentially a sequence of differentiable linear algebra operations, the entire CFD simulation is differentiable.35
We can backpropagate the error (measured - simulated) through the fluid dynamics equations to calculate the gradient of the error with respect to the state vector.
Correction: This gradient allows the navigation filter (Kalman Filter) to update the state vector. "To match this pressure reading, I must actually be flying at Mach 9.8 at an altitude of 29.5km, not Mach 10 at 30km."
Result: This effectively "locks" the inertial solution to the physics of the flight. The drift is bounded not by time, but by the accuracy of the pressure sensors. This allows for GPS-independent precision strike capability over intercontinental ranges.37
8. Operational Application III: Defensive Interception (Glide Breaker)
Defending against hypersonic weapons is harder than building them. The Glide Breaker program aims to develop an interceptor (Kill Vehicle - KV) that collides kinetically with a maneuvering hypersonic target.38
8.1 The Jet Interaction (JI) Problem
To maneuver in the thin upper atmosphere, the KV uses divert thrusters (lateral rockets). When these rockets fire into the hypersonic crossflow, they create complex shock structures and separation bubbles. This "Jet Interaction" can amplify the control force (favorable) or oppose it (unfavorable), and it creates massive asymmetry.39
Current interceptors rely on lookup tables for JI effects, but these are notoriously inaccurate for dynamic firing patterns.
8.2 Real-Time JI Compensation
HyperTensor enables the KV to simulate its own thruster plumes in real-time.
Input: Thruster firing command.
Simulation: The solver computes the interaction of the plume shock with the body boundary layer.
Output: The net force and moment on the vehicle.
Control: The autopilot compensates for the "phantom forces" generated by the JI effect, ensuring that the thrust vector passes exactly through the center of mass. This dramatically reduces the "miss distance," turning a near-miss into a direct hit.
9. Strategic Outlook and Conclusions
9.1 The Shift to "Fly-by-Math"
Project HyperTensor represents the transition from "Fly-by-Wire" (following pre-programmed laws) to "Fly-by-Math" (following real-time physics). This capability unchains hypersonic vehicles from the conservative margins of safety dictated by static wind tunnel testing. It allows for:
Unstable Designs: Vehicles can be designed with extreme instability (for range/agility) because the active CFD control can stabilize them.
Damage Tolerance: The system can adapt to battle damage (holes in wings) by simulating the damaged geometry and finding a new control solution.
9.2 The Quantum Bridge
While implemented on classical GPU hardware, the algorithms used (MPS/TT) are fundamentally quantum-mechanical. This makes HyperTensor "quantum-ready." As actual quantum computers become available (and miniaturized), the software stack can be ported with minimal changes, unlocking even greater fidelity.18 But the crucial insight is that we do not need to wait for quantum hardware. The quantum-inspired mathematical revolution provides the necessary speedup now on silicon that can fly today.
9.3 Conclusion
The "impossible" goal of a wind tunnel in a missile is achievable. It requires the convergence of the NVIDIA Jetson Orin's tensorial compute power with the algorithmic compression of Tensor Networks. By solving the Navier-Stokes equations in real-time, HyperTensor solves the plasma blackout, eliminates inertial drift, and enables the agility required to win the high-speed fight. It transforms the hypersonic missile from a dumb projectile into a physics-aware predator.
Comparison of Computational Architectures
Feature
Classical CFD (Finite Volume)
Deep Learning (PINNs)
HyperTensor (Tensor Networks)
Data Structure
Dense Grid ($N^3$)
Neural Weights
Compressed Tensor ($N \cdot D^2$)
Scaling
Exponential with Dim
Linear with Depth
Polynomial (Log-Linear)
Physics Fidelity
High (Exact Equations)
Low (Soft Constraints)
High (Exact Equations)
Shock Capturing
Excellent (WENO/Godunov)
Poor (Spectral Bias)
Excellent (WENO-TT)
Conservation
Guaranteed
Not Guaranteed
Guaranteed
Generalization
Perfect (First Principles)
Poor (Training Bound)
Perfect (First Principles)
Edge Feasibility
Impossible (HPC only)
Good
Excellent
Interpretability
High
Black Box
High (Physical Modes)

This comparison highlights why Tensor Networks occupy the "Goldilocks" zone for this application: the rigour of CFD with the efficiency of AI.
Works cited
TBG: Tactical Boost Glide - DARPA, accessed December 20, 2025, https://www.darpa.mil/research/programs/tactical-boost-glide
COMPUTATIONAL FLUID DYNAMICS TECHNOLOGY FOR HYPERSONIC APPLICATIONS, accessed December 20, 2025, https://ntrs.nasa.gov/api/citations/20040013407/downloads/20040013407.pdf
Radio Communications Blackout | Nonequilibrium Gas & Plasma Dynamics Laboratory, accessed December 20, 2025, https://www.colorado.edu/lab/ngpdl/research/hypersonics/radio-communications-blackout
Radio blackout alleviation and plasma diagnostic results from a 25000 foot per second blunt-body reentry, accessed December 20, 2025, https://ntrs.nasa.gov/api/citations/19700008892/downloads/19700008892.pdf
A new method for removing the blackout problem on reentry vehicles - ResearchGate, accessed December 20, 2025, https://www.researchgate.net/publication/257973822_A_new_method_for_removing_the_blackout_problem_on_reentry_vehicles
Application of Computational Fluid Dynamics in Missile Engineering - Johns Hopkins University Applied Physics Laboratory, accessed December 20, 2025, https://www.jhuapl.edu/content/techdigest/pdf/V22-N03/22-03-Frostbutter.pdf
Turbulent Flows Simulated With 99.99% Accuracy Using Matrix Product States, accessed December 20, 2025, https://quantumzeitgeist.com/99-99-percent-accuracy-states-turbulent-flows-simulated-matrix-product/
[PDF] A quantum-inspired approach to exploit turbulence structures - Semantic Scholar, accessed December 20, 2025, https://www.semanticscholar.org/paper/A-quantum-inspired-approach-to-exploit-turbulence-Gourianov-Lubasch/88e072de7111f8205d248215f726a4dd8b3d3065
Simplified Real-Time Flush Air-Data Sensing System for Sharp-Nosed Hypersonic Vehicles, accessed December 20, 2025, https://arc.aiaa.org/doi/pdfplus/10.2514/1.A35634
MODELING AND SIMULATION OF A GENERIC HYPERSONIC VEHICLE - KU ScholarWorks, accessed December 20, 2025, https://kuscholarworks.ku.edu/server/api/core/bitstreams/e1266ba6-b113-408d-aadc-873540a4b373/content
Computational Modelling for Shock Tube Flows - GDTk, accessed December 20, 2025, https://gdtk.uqcloud.net/pdfs/james-faddy-masters-thesis-aug-2000.pdf
Can physics-informed neural networks beat the finite element method? - Oxford Academic, accessed December 20, 2025, https://academic.oup.com/imamat/article/89/1/143/7680268
Neural PDE Solvers with Physics Constraints A Comparative Study of PINNs, DRM, and WANs - arXiv, accessed December 20, 2025, https://arxiv.org/html/2510.09693v1
Comparing Neural Network and Numerical Method to Solve Multiphase Flow Problem — A Review - Miftahul Tirta Irawan, accessed December 20, 2025, https://punyatirta.medium.com/comparing-neural-network-and-numerical-method-to-solve-multiphase-flow-problem-a-review-3252087bf7fa
Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing - MDPI, accessed December 20, 2025, https://www.mdpi.com/2076-3417/15/14/8092
Tensor networks enable the calculation of turbulence probability distributions | Request PDF, accessed December 20, 2025, https://www.researchgate.net/publication/388496407_Tensor_networks_enable_the_calculation_of_turbulence_probability_distributions
Tensor networks enable the calculation of turbulence probability distributions - arXiv, accessed December 20, 2025, https://arxiv.org/html/2407.09169v2
arXiv:2305.10784v2 [physics.flu-dyn] 23 May 2023, accessed December 20, 2025, https://arxiv.org/pdf/2305.10784
Tensor Train Multiplication - arXiv, accessed December 20, 2025, https://arxiv.org/html/2410.19747v2
Time-Dependent Variational Principle (TDVP) - mps - Tensor Network, accessed December 20, 2025, https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html
The time-dependent bivariational principle: Theoretical foundation for real-time propagation methods of coupled-cluster type - arXiv, accessed December 20, 2025, https://arxiv.org/html/2410.24192v1
Tensor network approaches for plasma dynamics - arXiv, accessed December 20, 2025, https://arxiv.org/html/2512.15924v1
Computing time-periodic steady-state currents via the time evolution of tensor network states - Gingrich Group, accessed December 20, 2025, https://gingrich.chem.northwestern.edu/papers/JCP.157.054104.pdf
Tensor Networks Offer Scalable Solution For 2D Fluid Dynamics Simulations, accessed December 20, 2025, https://quantumzeitgeist.com/tensor-networks-offer-scalable-solution-for-2d-fluid-dynamics-simulations/
Tensor-Train TENO Scheme for Compressible Flows | AIAA SciTech Forum, accessed December 20, 2025, https://arc.aiaa.org/doi/10.2514/6.2025-0304
[2405.12301] Tensor-Train WENO Scheme for Compressible Flows - arXiv, accessed December 20, 2025, https://arxiv.org/abs/2405.12301
NVIDIA Jetson AGX Orin Industrial - Open Zeka, accessed December 20, 2025, https://openzeka.com/en/wp-content/uploads/2023/05/jetson-orin-datasheet-jetson-agx-orin-industrial-web-nv-us-2757128-r2-1.pdf
Chinese researchers use low-cost Nvidia chip for hypersonic weapon —unrestricted Nvidia Jetson TX2i powers guidance system | Tom's Hardware, accessed December 20, 2025, https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-researchers-install-low-cost-unrestricted-nvidia-jetson-tx2i-into-hypersonic-weapon
NVIDIA Jetson AGX Orin Series, accessed December 20, 2025, https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf
Progress in Hypersonics Missiles and Space Defense [Slofer], accessed December 20, 2025, https://kstatelibraries.pressbooks.pub/cyberhumansystems/chapter/13-progress-in-hypersonics-missiles-and-space-defense-slofer/
Numerical Simulation of 1D Compressible Flows, accessed December 20, 2025, https://ttu-ir.tdl.org/bitstreams/696b2f15-a0ea-4602-931e-615f2aa84346/download
Week 7: Shock tube simulation project, accessed December 20, 2025, https://skill-lync.com/student-projects/week-7-shock-tube-simulation-project-14
The Shock Tube Problem, accessed December 20, 2025, https://www.bu.edu/ufmal/files/2016/12/Project_slides_Luisa_Capannolo.pdf
Connection between bond-dimension of a matrix product state and entanglement, accessed December 20, 2025, https://physics.stackexchange.com/questions/193298/connection-between-bond-dimension-of-a-matrix-product-state-and-entanglement
[1903.09650] Differentiable Programming Tensor Networks - arXiv, accessed December 20, 2025, https://arxiv.org/abs/1903.09650
3DID: Direct 3D Inverse Design for Aerodynamics with Physics-Aware Optimization - arXiv, accessed December 20, 2025, https://arxiv.org/html/2512.08987v1
Inertial Navigation System Drift Reduction Using Scientific Machine Learning - DSpace@MIT, accessed December 20, 2025, https://dspace.mit.edu/bitstream/handle/1721.1/156966/mcmanus-mattmcm-meng-eecs-2024-thesis.pdf?sequence=1&isAllowed=y
DARPA is playing both sides of the ball with both offensive and defensive hypersonics, accessed December 20, 2025, https://breakingdefense.com/2022/11/darpa-is-playing-both-sides-of-the-ball-with-both-offensive-and-defensive-hypersonics/
Glide Breaker Program Enters New Phase - DARPA, accessed December 20, 2025, https://www.darpa.mil/news/2022/glide-breaker
Technical report on a quantum-inspired solver for simulating compressible flows - arXiv, accessed December 20, 2025, https://arxiv.org/html/2506.03833v1
