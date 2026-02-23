# Industrial-grade CPS job submission, outputs, and cost model

## Executive summary

- Three industrial CPS campaigns are specified for 2026-02-17 (America/New_York) on entity["company","Amazon Web Services","cloud provider"] GPU instances: (I1) 1,000 independent simulations, 2 hours each on `p4d.24xlarge` (8×A100), (I2) a persistent digital twin on 4×`p5.48xlarge` for 720 hours, and (I3) a burst allocation of 100×`p5.48xlarge` for 24 hours. citeturn24search8turn29view0  
- Instance envelopes used in the manifests are taken from AWS primary docs: `p5.48xlarge` is 192 vCPUs, 2 TiB host memory, 8×H100 with 640 GB HBM3, 3,200 Gbps EFA networking, and 8×3.84 TB local NVMe; `p4d.24xlarge` is described as 96 vCPUs, about 1.1 TB RAM, 8×40 GB A100, about 8 TB local NVMe, and 400 Gbps EFA networking. citeturn29view0turn24search8  
- Public list compute proxies (used for all cost math) are anchored to AWS Capacity Blocks pricing: `p4d.24xlarge` effective hourly rate $11.8 per instance-hour ($1.475 per accelerator-hour) and `p5.48xlarge` $31.464 per instance-hour ($3.933 per accelerator-hour), Linux OS fee $0.00 in Capacity Blocks. citeturn27search0turn28view0  
- Capacity Blocks are operationally constrained: reserve up to 8 weeks ahead, duration 1–14 days, and up to 64 instances per block (you can hold up to 256 instances across blocks). This affects I2 (30-day persistence) and I3 (100 instances). citeturn18view0turn18view1  
- Scientific workload family and artifacts are concretized with a production-grade MD campaign template using GROMACS 2024.1 (GPU-accelerated MD), because it provides well-defined output products, strong GPU scaling knobs, and modern, citable best-practices guidance for multi-GPU settings. citeturn10search6turn13view0turn12view2  
- For sizing and per-run I/O, a representative “large system” reference is System 6 (1,066,628 atoms) from the NHR@FAU benchmark set; throughput context is also available (for a single A100, System 6 is reported at ~23.2 ns/day in one benchmark report). citeturn13view0turn10search0  
- Total compute costs (compute only, excluding storage/people) under AWS Capacity Blocks price proxies are: I1 ≈ $23,600 for 2,000 instance-hours, I2 ≈ $90,616 for 2,880 instance-hours, I3 ≈ $75,514 for 2,400 instance-hours (all “derived estimate” from AWS list rates, because campaign-level totals depend on reserved duration utilization and orchestration overhead). citeturn27search0  
- The biggest operational risks are (a) capacity assurance and scheduling constraints (64-instance block limit and 14-day max duration), (b) data-volume explosion driven by trajectory write frequency, and (c) restart correctness and file appending behavior across failures; mitigations are built into the runbooks via Capacity Block partitioning, strict output policies, checkpointing cadence, and deterministic restart handling. citeturn18view0turn20search5turn13view0  

## Scope, assumptions, and inputs

This report produces production-ready manifests and submission examples for the three user-defined industrial campaigns (I1/I2/I3) using an MD-based CPS reference stack. The MD choice is deliberate: “job submission and outputs” can be made concrete with strongly standardized artifacts (trajectory, energy, logs, checkpoints) and known GPU run controls, while still matching the user’s CPS workload-family list (MD and UQ ensembles). citeturn20search3turn20search1turn14search1  

Geography/pricing assumption for all cost calculations is US East (N. Virginia) (`us-east-1`) unless a source is explicitly region-agnostic; when sources provide multi-region tables, the manifests still pin to a single region to avoid mixing rates. Compute is priced using AWS EC2 Capacity Blocks public tables (and, where relevant, AWS published on-demand references for p5). citeturn27search0turn17view0  

Software stack assumptions (explicit, replaceable): Linux/UNIX platform, containerized solver execution, and an HPC scheduler (Slurm via AWS ParallelCluster) for reproducibility and job-array semantics. AWS explicitly documents ParallelCluster as a way to deploy and manage HPC clusters with Slurm, and documents separate mechanisms to target Capacity Blocks from ParallelCluster queues. citeturn9view5turn19search21turn18view1  

Licensing assumptions: the reference solver stack is open-source GROMACS and associated analysis tooling, so line-item solver licensing cost is $0.00; if you swap to commercial CAE (for example, entity["company","Ansys","engineering simulation vendor"]), the licensing is frequently quote-based or pay-per-use (Elastic model is described publicly, but prices are not public list), so you would substitute a vendor quote or elastic-credit consumption model. citeturn14search2  

## Industrial campaign manifests

### Ensemble campaign

This section corresponds to I1 (1,000 simulations × 2 hours on `p4d.24xlarge`, 8×A100). The manifest is designed for high-throughput ensemble execution, strict restartability, and minimized per-run I/O to keep aggregate storage manageable. `p4d.24xlarge` is described by AWS as having dual-socket Intel Cascade Lake CPUs totaling 96 vCPUs, about 1.1 TB RAM, 8×40 GB A100 GPUs with NVSwitch, about 8 TB NVMe instance storage, and 400 Gbps EFA networking. citeturn24search8  

**Cost proxy basis (public list):** AWS Capacity Blocks list `p4d.24xlarge` at $11.8 per instance-hour ($1.475 per accelerator-hour) with Linux OS fee $0.00 in Capacity Blocks pricing tables. citeturn27search0turn28view0  

**Production-ready campaign manifest (YAML)**

```yaml
campaign_id: I1
campaign_type: "ensemble_uq_md"
date_of_submission: "2026-02-17"
region: "us-east-1"
availability_zone_strategy: "single-AZ (Capacity Block constraint)"
pricing_mode:
  primary: "capacity_blocks"
  label: "public list price"
compute_environment:
  scheduler: "slurm (AWS ParallelCluster v3)"
  instance_type: "p4d.24xlarge"
  instance_count_target_concurrency: 42            # derived to fit 1000x2h into a 48h window
  per_instance_resources:
    vcpu: 96
    host_memory: "1.1 TiB (approx)"
    gpu:
      count: 8
      model: "NVIDIA A100 40GB"
      gpu_memory_total: "320 GB"
    local_nvme: "8 TB (approx; 8x NVMe devices)"
    network:
      efa: true
      bandwidth: "400 Gbps"
  instance_mapping:
    job_unit: "1 simulation per 1 p4d.24xlarge node"
    slurm_array: "0-999%42"
container_and_runtime:
  container_runtime: "Apptainer/Singularity (on Slurm) or Docker (on worker AMI)"
  image:
    name: "ecr.us-east-1.amazonaws.com/cps/gromacs:2024.1-cuda12"
    digest_pin: "sha256:<pin-in-ci>"
  mounts:
    - host: "/fsx"          # optional shared FSx for inputs/log aggregation
      container: "/fsx"
    - host: "/local_nvme"
      container: "/scratch"
software_stack:
  solver: "GROMACS 2024.1"
  mpi: "OpenMPI 4.x + EFA plugin (cluster image)"
  cuda: "CUDA 12.x (in container)"
  orchestration:
    workflow_engine: "Slurm job arrays + post-run reducer job"
    retries: "2 automatic retries on transient failures"
reference_workload_parameters:
  workload_family: "MD"
  system_reference: "NHR@FAU benchmark System 6"
  particle_count_atoms: 1066628
  integrator: "md (leap-frog)"
  time_step: "2 fs"
  constraint_algorithm: "LINCS (assumed); lincs_iter/lincs_order per forcefield defaults"
  electrostatics: "PME"
  cutoffs: "1.0 nm (assumed)"
  thermostat: "v-rescale (assumed)"
  barostat: "Parrinello-Rahman (assumed)"
  solver_tolerances:
    constraint_tolerance: "1e-4 (assumed)"
gromacs_gpu_configuration:
  mdrun_flags:
    - "-nb gpu"
    - "-pme gpu"
    - "-bonded gpu"
    - "-update gpu"
    - "-pin on"
    - "-pinstride 1"
    - "-ntmpi 8"          # thread-MPI threads = number of GPUs (best-practice reference)
    - "-ntomp 12"         # 8*12=96 CPU threads, matches p4d vCPU count
  environment_variables:
    - "GMX_GPU_DD_COMMS=true"
    - "GMX_GPU_PME_PP_COMMS=true"
checkpointing_and_restart:
  checkpoint_interval_minutes: 10
  files:
    checkpoint_out: "state.cpt"
    restart_in: "state.cpt"
  restart_policy:
    mode: "restart from last checkpoint"
    append_behavior: "noappend (safer for retries), then consolidate"
io_profile:
  read_patterns:
    - "Read .tpr (run input) + .mdp provenance"
  write_patterns:
    - "Write .xtc compressed trajectory only (no full .trr) (assumed policy)"
    - "Write .edr energy"
    - "Write .log"
    - "Write .cpt checkpoint"
  io_locality:
    scratch: "/scratch on local NVMe"
    stage_out: "S3 (or FSx -> S3)"
expected_runtime:
  per_simulation_walltime: "2h target (hard constraint)"
  overhead_minutes: 5
  campaign_makespan_hours: 48
monitoring_and_audit:
  metrics:
    - "GPU utilization, memory, power (nvidia-smi)"
    - "mdrun performance (ns/day) extracted from log"
    - "I/O throughput and failure counts"
  logs:
    - "Slurm stdout/stderr"
    - "CloudTrail for capacity block + instance launches"
```

The GPU run-control knobs and multi-GPU environment variables in this manifest are aligned with published best-practice guidance (thread-MPI `-ntmpi` equal to number of GPUs, and environment variables enabling GPU domain-decomposition and PME/PP comms). citeturn12view1turn12view2  

### Persistent digital twin campaign

This section corresponds to I2 (persistent digital twin: 4×`p5.48xlarge` reserved for 720 hours). `p5.48xlarge` is specified by AWS as 192 vCPUs, 2 TiB host memory, 8×H100 with total 640 GB HBM3, 3,200 Gbps EFA, and 8×3.84 TB NVMe local storage; AWS also frames P5 as suitable for HPC and large-scale clusters on UltraClusters. citeturn29view0turn9view3  

**Capacity assurance interpretation:** a 720-hour always-on twin is longer than a single Capacity Block’s maximum duration (1–14 days), and also requires careful handoff because Capacity Blocks end at a fixed daily UTC time and have operational constraints (for example, start-time limited to 8 weeks ahead and specific targeting by reservation ID). Two production-safe patterns are therefore offered: (a) use an On-Demand Capacity Reservation (ODCR) for “any duration” capacity assurance, or (b) chain Capacity Blocks (14d + 14d + 2d) if you want the Capacity Blocks price proxy and can tolerate reservation management overhead. citeturn19search4turn18view0turn18view1  

**Cost proxy basis:** Capacity Blocks list `p5.48xlarge` at $31.464 per instance-hour ($3.933 per accelerator-hour) in `us-east-1`; for an alternate comparison baseline, AWS also publishes $98.32/hr on-demand and $43.18/hr “3Yr RI rate” in a public AWS benchmarking/cost table. citeturn27search0turn17view0  

**Production-ready digital-twin manifest (YAML)**

```yaml
campaign_id: I2
campaign_type: "persistent_digital_twin_hybrid_physics_ai"
date_of_submission: "2026-02-17"
region: "us-east-1"
pricing_mode:
  primary_option_A: "on-demand capacity reservation (ODCR)"
  primary_option_B: "chained capacity blocks (14d + 14d + 2d)"
compute_environment:
  always_on_cluster:
    scheduler: "kubernetes (EKS) for services + slurm (optional) for batch physics jobs"
    gpu_instances:
      instance_type: "p5.48xlarge"
      instance_count: 4
      per_instance_resources:
        vcpu: 192
        host_memory: "2 TiB"
        gpu:
          count: 8
          model: "NVIDIA H100"
          gpu_memory_total: "640 GB HBM3"
        local_nvme: "8 x 3.84 TB"
        network:
          efa: true
          bandwidth: "3200 Gbps"
  role_mapping:
    - name: "twin-core-services"
      placement: "one p5 node (CPU-heavy pods pinned; minimal GPU use)"
    - name: "inference-and-assimilation"
      placement: "remaining p5 nodes (GPU pods; MIG off; full-GPU allocation)"
    - name: "recalibration-physics"
      placement: "scheduled jobs across all 4 nodes when needed"
container_and_runtime:
  images:
    physics_solver:
      name: "ecr.us-east-1.amazonaws.com/cps/gromacs:2024.1-cuda12"
      digest_pin: "sha256:<pin-in-ci>"
    twin_services:
      name: "ecr.us-east-1.amazonaws.com/cps/twin-services:2026.02"
      digest_pin: "sha256:<pin-in-ci>"
software_stack:
  physics: "GROMACS 2024.1"
  ai_stack: "PyTorch 2.x (assumed), CUDA 12.x, NCCL (assumed)"
  data_plane: "Kafka/MSK or Kinesis (assumed), object store in S3"
digital_twin_workload_model:
  loop_cadence:
    assimilation_step: "every 5 minutes (assumed)"
    physics_recalibration: "daily or event-triggered (assumed)"
  physics_recalibration_job:
    particle_count_atoms: 1066628
    time_step: "2 fs"
    run_walltime: "2h (assumed per recalibration window)"
    concurrency: "32 GPUs across 4 nodes (assumed)"
checkpointing_and_state:
  physics_checkpoint_minutes: 15
  twin_state_store: "versioned state snapshots every 5 minutes to S3 (assumed)"
io_profile:
  hot_path:
    - "write minimal telemetry + derived metrics"
  cold_path:
    - "store trajectories only for audit windows; otherwise store reduced features"
monitoring_and_slo:
  slo_targets:
    - "p95 assimilation latency < 2 minutes (assumed)"
    - "daily recalibration completes in < 4 hours (assumed)"
  monitoring:
    - "GPU/CPU utilization, EFA errors, pod restarts"
    - "physics drift checks (energy, temperature stability)"
```

The feasibility and rationale for `p5.48xlarge` as a digital-twin backbone is derived from AWS’s published P5 feature set (very high EFA bandwidth, multi-GPU NVSwitch intra-node, and large local NVMe for hot datasets). citeturn29view0turn9view3  

### Burst campaign

This section corresponds to I3 (100×`p5.48xlarge` for 24 hours). Capacity Blocks support up to 64 instances per block, so 100 instances must be split across (at least) two Capacity Blocks (for example, 64 + 36), and the reservation must be sized to the 24-hour duration (minimum 1 day). citeturn18view0turn18view1  

**Cost proxy basis:** Capacity Blocks list `p5.48xlarge` at $31.464 per instance-hour ($3.933 per accelerator-hour), which is used as the “public list price proxy” for burst compute math. citeturn27search0  

**Production-ready burst manifest (YAML)**

```yaml
campaign_id: I3
campaign_type: "burst_screening_or_design_optimization"
date_of_submission: "2026-02-17"
region: "us-east-1"
capacity_design:
  capacity_blocks:
    - name: "cb-a"
      instance_type: "p5.48xlarge"
      instance_count: 64
      duration_hours: 24
    - name: "cb-b"
      instance_type: "p5.48xlarge"
      instance_count: 36
      duration_hours: 24
compute_environment:
  scheduler: "slurm on AWS ParallelCluster v3 (EFA-enabled)"
  instance_type: "p5.48xlarge"
  total_instances: 100
  per_instance_resources:
    vcpu: 192
    host_memory: "2 TiB"
    gpu: "8 x H100 (640 GB HBM3 total)"
    local_nvme: "8 x 3.84 TB"
    network: "3200 Gbps EFA"
workload_layout_assumption:
  pattern: "1 physics job per node (for strong isolation) OR 2 jobs per node (4 GPUs each) for higher throughput"
  mapping_control: "Slurm --gpus-per-task and cgroups"
container_and_runtime:
  image: "ecr.us-east-1.amazonaws.com/cps/gromacs:2024.1-cuda12"
  mounts:
    - "/fsx:/fsx"
    - "/local_nvme:/scratch"
orchestration:
  primary: "Slurm job arrays with per-node exclusivity"
  secondary: "workflow reducer job to aggregate UQ statistics"
checkpointing:
  interval_minutes: 20
  safe_shutdown: "stop new work 60 minutes before block end; drain + stage-out"
io_profile:
  shared_fs: "FSx for Lustre for fan-in/fan-out metadata and aggregation"
  bulk_archive: "S3 for long-term storage"
```

The selection of FSx for Lustre as shared HPC storage is consistent with AWS guidance that P5 supports FSx for Lustre for high-throughput DL/HPC workloads, and AWS FSx for Lustre pricing examples show persistent storage costs on the order of $0.145/GB-month (region-dependent). citeturn29view0turn9view0  

## I1 submission examples

This section provides concrete “submit today” artifacts for I1: an Slurm array script and a cloud CLI path that targets Capacity Blocks and uses AWS ParallelCluster.

### Slurm submission script for I1

The script uses (a) a job array of 1,000 tasks with maximum concurrency 42, (b) a conservative checkpoint interval, and (c) explicitly recommended multi-GPU controls (`-ntmpi` equals GPU count, GPU offload flags, and multi-GPU env vars). citeturn12view1turn12view2turn20search5  

```bash
#!/bin/bash
#SBATCH --job-name=i1_ensemble
#SBATCH --partition=p4d
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus=8
#SBATCH --time=02:15:00
#SBATCH --array=0-999%42
#SBATCH --output=/fsx/logs/%x_%A_%a.out
#SBATCH --error=/fsx/logs/%x_%A_%a.err
#SBATCH --exclusive

set -euo pipefail

# -----------------------------
# Campaign parameters
# -----------------------------
export AWS_REGION="us-east-1"
export CAMPAIGN_ID="I1"
export RUN_INDEX="${SLURM_ARRAY_TASK_ID}"

# Object storage layout (assumed bucket names)
export S3_INPUT_PREFIX="s3://cps-prod-input/i1"
export S3_OUTPUT_PREFIX="s3://cps-prod-output/i1"

# Local scratch on instance NVMe
export SCRATCH_BASE="/scratch/${CAMPAIGN_ID}/${SLURM_JOB_ID}/${RUN_INDEX}"
mkdir -p "${SCRATCH_BASE}"
cd "${SCRATCH_BASE}"

# -----------------------------
# Container + solver stack
# -----------------------------
export CONTAINER_IMAGE="ecr.us-east-1.amazonaws.com/cps/gromacs:2024.1-cuda12"
# If using Apptainer: assume image has been pulled and cached as a SIF.
export SIF_PATH="/fsx/containers/gromacs_2024.1_cuda12.sif"

# GROMACS multi-GPU comm knobs (per best practice guidance)
export GMX_GPU_DD_COMMS=true
export GMX_GPU_PME_PP_COMMS=true

# Optional knobs often used to improve GPU comm and reduce timing overhead
export GMX_DISABLE_GPU_TIMING=1
export GMX_ENABLE_DIRECT_GPU_COMM=1

# -----------------------------
# Pre-hook: stage inputs
# -----------------------------
aws s3 cp "${S3_INPUT_PREFIX}/topol.tpr" "./topol.tpr"
aws s3 cp "${S3_INPUT_PREFIX}/md.mdp" "./md.mdp"
aws s3 cp "${S3_INPUT_PREFIX}/index.ndx" "./index.ndx" || true

# Per-run parameterization (assumed pattern)
aws s3 cp "${S3_INPUT_PREFIX}/params/param_${RUN_INDEX}.json" "./param.json"

# -----------------------------
# Run execution
# -----------------------------
# Checkpoint cadence: 10 minutes
# Use -noappend for safer retries, then consolidate in postprocessing
CMD="gmx_mpi mdrun \
  -s topol.tpr \
  -deffnm run_${RUN_INDEX} \
  -cpt 10 \
  -cpo state.cpt \
  -maxh 1.95 \
  -noappend \
  -pin on -pinstride 1 \
  -ntmpi 8 -ntomp 12 \
  -nb gpu -pme gpu -bonded gpu -update gpu"

# Execute inside container with GPU passthrough
apptainer exec --nv \
  -B /fsx:/fsx \
  -B /scratch:/scratch \
  "${SIF_PATH}" \
  bash -lc "${CMD}"

# -----------------------------
# Post-hook: extract metrics + stage out
# -----------------------------
# Example: extract ns/day from the log (assumes standard GROMACS log formatting)
NS_PER_DAY="$(grep -E 'Performance:' -A2 run_${RUN_INDEX}.log | grep -Eo '[0-9]+\.[0-9]+' | tail -n1 || true)"

# Upload primary artifacts (avoid full-precision trr by policy)
aws s3 cp "./run_${RUN_INDEX}.log" "${S3_OUTPUT_PREFIX}/runs/${RUN_INDEX}/"
aws s3 cp "./run_${RUN_INDEX}.edr" "${S3_OUTPUT_PREFIX}/runs/${RUN_INDEX}/"
aws s3 cp "./state.cpt" "${S3_OUTPUT_PREFIX}/runs/${RUN_INDEX}/"
aws s3 cp "./run_${RUN_INDEX}.xtc" "${S3_OUTPUT_PREFIX}/runs/${RUN_INDEX}/" || true
aws s3 cp "./run_${RUN_INDEX}.gro" "${S3_OUTPUT_PREFIX}/runs/${RUN_INDEX}/" || true

# Monitoring hook: push a custom metric (optional)
aws cloudwatch put-metric-data \
  --namespace "CPS/I1" \
  --metric-data "MetricName=ns_per_day,Value=${NS_PER_DAY:-0},Unit=None,Dimensions=[{Name=run_index,Value=${RUN_INDEX}}]"
```

### Cloud CLI path for I1

Capacity Blocks require you to specify the Capacity Block reservation ID when you launch instances; AWS documents a concrete `run-instances` example using `--instance-market-options MarketType='capacity-block'` and `--capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=...}`. citeturn19search1turn18view1  

If deploying Slurm via AWS ParallelCluster, AWS documents both (a) the general workflow (`pcluster configure`, then `pcluster create-cluster`) and (b) Capacity Block support in the cluster configuration by setting `CapacityType = CAPACITY_BLOCK` and providing `CapacityReservationId` for the queue. citeturn9view5turn9view4turn19search21  

```bash
# (1) Create a ParallelCluster config for a Slurm cluster that targets Capacity Blocks
# File: i1-p4d-capacity-block-cluster.yaml
# (content shown below)

# (2) Create the cluster
pcluster create-cluster \
  --cluster-name cps-i1-p4d \
  --cluster-configuration i1-p4d-capacity-block-cluster.yaml

# (3) Connect to head node and submit
pcluster ssh --cluster-name cps-i1-p4d

# On the head node:
sbatch /fsx/jobs/i1_ensemble.sbatch
squeue
```

```yaml
# i1-p4d-capacity-block-cluster.yaml (illustrative, production skeleton)
Region: us-east-1
Image:
  Os: alinux2023
HeadNode:
  InstanceType: c7i.4xlarge
  Networking:
    SubnetId: subnet-REPLACE_ME
  LocalStorage:
    RootVolume:
      Size: 200
      VolumeType: gp3
Scheduling:
  Scheduler: slurm
  SlurmSettings:
    QueueUpdateStrategy: DRAIN
  SlurmQueues:
    - Name: p4d
      CapacityType: CAPACITY_BLOCK
      CapacityReservationId: cr-REPLACE_WITH_CAPACITY_BLOCK_ID
      ComputeResources:
        - Name: p4d24
          InstanceType: p4d.24xlarge
          MinCount: 0
          MaxCount: 42
      Networking:
        SubnetIds:
          - subnet-REPLACE_ME
        PlacementGroup:
          Enabled: true
SharedStorage:
  - Name: fsx
    StorageType: FsxLustre
    MountDir: /fsx
    FsxLustreSettings:
      StorageCapacity: 1200
      DeploymentType: PERSISTENT_2
      PerUnitStorageThroughput: 125
```

## Outputs, diagnostics, and data management

### Scientific outputs and verification artifacts

GROMACS `mdrun` output expectations are highly standardized: a run takes a `.tpr` run input and typically produces a trajectory (`.trr`, `.tng`, or `.xtc`), a log file, and optionally checkpoints; the energy file (`.edr`) is the portable energy output, and the checkpoint file (`.cpt`) stores the complete simulation state to support restarts. citeturn20search3turn20search1turn20search4  

For all campaigns, primary scientific outputs (per simulation execution) are defined at three tiers, chosen to control data volume and enable auditability:

- **Tier A (minimal, validation-first):** `.log`, `.edr`, `.cpt`, final structure `.gro`. This supports verification (thermostat stability, pressure/temperature time series, constraint health) and enables restart. citeturn20search3turn20search5  
- **Tier B (standard analysis):** Tier A plus compressed coordinates trajectory `.xtc` (reduced precision, typically 3 decimal places by default in common tooling), written at a controlled cadence. This supports post-hoc RMSD/RMSF, diffusion proxies, and feature extraction while controlling storage. citeturn21search1turn21search16  
- **Tier C (heavy, forensic):** add full-precision `.trr` for short windows only (for example, last 5–10% of a run), because `.trr` contains velocities and optionally forces and scales poorly in size. This tier is recommended only for short diagnostic windows, not for 1,000-run ensembles. citeturn20search4turn20search3  

Derived metrics and diagnostics (campaign-level) are produced by postprocessing reducers:

- **Objectives and sensitivities:** objective functions from `.edr` (for example, mean potential energy, pressure stability, or domain-specific CV values), and sensitivity proxies via finite differences across parameterized ensemble members (assumed methodology).  
- **UQ statistics:** per-parameter distributions (mean/variance/quantiles), failure-rate by parameter region, and convergence diagnostics across replicas (assumed methodology).  
- **Verification artifacts:** extracted “performance” lines (ns/day), constraint warnings, PME/neighbor-search diagnostics, and restart/appending consistency checks; `gmx check` is used to sanity-check trajectories and energy files. citeturn20search8turn20search5  

### Data volume model per run and per campaign

A practical upper-bound model for coordinate payload size comes from uncompressed single-precision coordinates: `bytes_per_frame ≈ atoms × 3 × 4`. For the System 6 reference size (1,066,628 atoms), that is about 12.2 MiB per frame before metadata and any compression. (This is a modeling assumption for sizing; XTC frame size is variable and typically smaller because XTC reduces precision and uses compression.) citeturn13view0turn21search16  

Empirically, the XTC ecosystem is positioned as “storage efficient”: (a) XTC applies reduced-precision coordinate storage, and (b) in one peer-reviewed analysis, XTC required about 1.3× less disk space than TNG in a fast-reading mode for the evaluated systems; this supports treating XTC as a meaningful storage reducer, but not as a fixed ratio for any single run. citeturn21search16turn22view0  

**Sizing table (illustrative, replace cadence with your science needs)**

| Output policy (per run) | Example write cadence (assumed) | Coordinate frames (2 ns example) | Upper-bound coordinate payload | Typical additional files | Notes |
|---|---:|---:|---:|---|---|
| Tier A | no trajectory | 0 | ~0 GB | `.log`, `.edr`, `.cpt`, `.gro` | Smallest footprint, highest throughput |
| Tier B | XTC every 100 ps | 20 | ~0.24 GB | + `.xtc` | Good for ensembles, limited time resolution |
| Tier B (denser) | XTC every 10 ps | 200 | ~2.4 GB | + `.xtc` | Higher storage, better spectral analyses |
| Tier C | TRR + XTC (short window) | varies | 5–50+ GB windowed | `.trr` is heavy | Use only for debugging windows |

The aggregate data volume for I1 depends primarily on the trajectory cadence. Under Tier B every 100 ps and a 2 ns illustrative run, raw coordinate payload upper bound is ~0.24 GB/run, which would be ~240 GB for 1,000 runs plus logs/checkpoints. For I3, even Tier B can become multi-terabyte if you retain dense trajectories from 100 nodes; therefore, the storage plan explicitly treats trajectory retention and downsampling as a first-class production control. citeturn13view0turn22view0  

### Postprocessing, storage tiers, and transfer costs

**Intermediate storage.** For I1, per-run scratch should live on instance NVMe (to decouple simulation I/O from shared filesystem contention) and stage out only the curated artifacts (Tier A or Tier B) to object storage. For I3, a shared filesystem is justified to coordinate postprocessing fan-in; AWS documents that P5 supports FSx for Lustre, and FSx pricing examples indicate persistent tiers around $0.145/GB-month (billed proportionally). citeturn29view0turn9view0  

**Long-term retention.** For standard retention, S3 Standard is typically modeled at $0.023/GB-month for the first 50 TB (region-dependent), and AWS cost examples and calculator assumptions commonly use that value. citeturn8search4turn8search14turn8search22  

**Transfer and egress.** AWS EC2 data transfer out to the internet is tiered (first 10 TB/month commonly listed at $0.09/GB; next tiers decline), and AWS S3 pricing examples also demonstrate $0.09/GB for data transfer out to the internet in at least one region example. This report therefore uses $0.09/GB as a conservative proxy for “first-tier internet egress,” with the explicit caveat that region and tier matter. citeturn7search0turn9view2  

**Recommended lifecycle policy (production).** Keep Tier A artifacts “hot” (S3 Standard) for 30–90 days for audit and rapid debugging, transition dense trajectories (Tier B) to colder storage for projects requiring longer retention, and keep Tier C forensic windows in hot storage only during active incident response (assumed policy). The key lever is not storage price, it is output-cadence governance, because trajectory cadence dominates volume. citeturn13view0turn22view0  

## Operations, cost breakdown, and risk mitigation

### Operational runbook

**Pre-run checks (Day −1 to Hour 0).** Confirm (a) Capacity Blocks start/end windows, instance counts, and AZ constraints (Capacity Block requiring same AZ and reservation-ID targeting), (b) EFA-enabled image requirements for P5 custom AMIs, (c) container image digest pinned and pulled to nodes, (d) object store permissions via IAM roles, and (e) smoke-test a single run with checkpoint and restart enabled. citeturn19search1turn18view1turn27search0  

**Failure modes and retry policy (production).**
- **Transient infrastructure failure (node reboot, EFA hiccup, filesystem stall):** automatic retry up to 2 times, restart from last `.cpt`. GROMACS restart behavior and append semantics are documented; for production retries, prefer `-noappend` and explicit consolidation to avoid checksum/appending mismatch across partial failures. citeturn20search5turn20search1  
- **Capacity Block end-of-window termination risk:** stop scheduling new work at least 60 minutes before reservation end, drain queues, stage-out, and verify completion; AWS documents Capacity Block operational timing constraints and that instances must target the reservation ID. citeturn18view1turn19search1  
- **Data-volume runaway (trajectory too frequent):** enforce output budgets (frames per ns, max bytes per run) as a CI gate on `.mdp` templates before production. NHR@FAU best-practice slides explicitly warn that “too much output decreases performance significantly,” making output governance both a cost and performance control. citeturn12view0turn13view0  

**Checkpoint/restart procedure (operator playbook).** On job restart, stage in the last checkpoint and restart with `-cpi`, direct output checkpoint with `-cpo`, and avoid appending unless you guarantee output file integrity; this aligns with GROMACS guidance on continuation and checksum-based validation of output files. citeturn20search5turn20search1  

**Personnel roles and labor hours (assumed, replace with org norms).**  
- HPC/cloud engineer: 24–40 hours (cluster config, IAM, network/EFA validation, monitoring hooks).  
- Simulation engineer: 40–80 hours (baseline physics setup, validation suite, output policy, postprocessing).  
- Data engineer/MLOps (I2 heavy): 40–120 hours (streaming, feature store, model governance, dashboards).  
These hour ranges are “assumed” because labor is organization- and compliance-dependent.

### Cost breakdown

**Compute (public list proxy, primary sources).** Capacity Blocks list effective hourly rates per instance and per accelerator for both P4d and P5. citeturn27search0turn28view0  

**Storage (public list proxy, primary sources).** S3 Standard is commonly modeled at $0.023/GB-month for first 50 TB in AWS examples, and gp3 EBS storage price is published as $0.08/GB-month with baseline 3,000 IOPS and 125 MB/s included (beyond-baseline performance priced separately). citeturn8search4turn30search5turn30search2  

**Cost summary (compute + minimal storage proxies).** Numeric totals below are labeled:

- **Public list price** = directly stated in cited vendor tables/pages.  
- **Derived estimate** = arithmetic from public list prices + campaign specs.  
- **Assumed** = explicitly stated placeholder when no public list exists.

| Campaign | Compute basis | Compute math | Compute cost | Storage basis (illustrative) | Storage cost (1 month) |
|---|---|---:|---:|---|---:|
| I1 | Capacity Blocks (public list) | 1,000 × 2h × $11.8 | **$23,600** (derived) | 300 GB @ $0.023/GB-month | **$6.90** (derived, assumes 300 GB retained) |
| I2 | Capacity Blocks proxy (public list) | 4 × 720h × $31.464 | **$90,616** (derived) | 5 TB @ $0.023/GB-month | **$117.76** (derived, assumes 5 TB retained) |
| I3 | Capacity Blocks (public list) | 100 × 24h × $31.464 | **$75,514** (derived) | 10 TB @ $0.023/GB-month | **$235.52** (derived, assumes 10 TB retained) |

The Capacity Blocks prices used are $11.8/hr for `p4d.24xlarge` and $31.464/hr for `p5.48xlarge` (Linux OS fee $0). citeturn27search0  
The S3 Standard storage proxy ($0.023/GB-month) is taken from AWS published assumptions/examples, and is intended for order-of-magnitude planning rather than exact billing. citeturn8search4turn8search14  

**Software licensing.** For the open-source reference stack, solver licensing is $0.00 (assumed). If you substitute commercial solvers, a practical proxy is “not publicly available; quote required,” consistent with public descriptions of pay-per-use/elastic licensing models that do not publish list prices. citeturn14search2  

**Egress.** If you must download data to on-prem over the internet, use tiered egress pricing (first-tier often $0.09/GB for the first 10 TB/month) as the default proxy, then re-estimate with exact region/tier at execution time. citeturn7search0turn9view2  

### Risks and mitigations

**Capacity risk (high impact, medium likelihood).** Capacity Blocks are limited to 64 instances per block and 1–14 day durations, requiring splits for I3 (100 instances) and chaining or alternative reservations for I2 (30 days). Mitigation: pre-book multiple blocks (64+36) for I3, and use ODCR or chained blocks (14d+14d+2d) for I2 with explicit handoff runbooks. citeturn18view0turn18view1  

**Data management risk (high impact, high likelihood).** Trajectory cadence can dominate both runtime (I/O contention) and storage; published best practices explicitly warn that excessive output can significantly degrade performance. Mitigation: enforce output budgets, keep Tier A as default, and require justification for Tier B/C per campaign. citeturn12view0turn22view0  

**Restart correctness risk (medium impact, medium likelihood).** Appending/restart semantics depend on file integrity and checkpoint checksums; failures mid-run can lead to inconsistent outputs if appending is misused. Mitigation: use `-noappend` for retries, centralized metadata for run parts, and consolidation jobs that verify logs and trajectory consistency with `gmx check`. citeturn20search5turn20search8  

**Pricing drift risk (medium impact, medium likelihood).** AWS notes Capacity Block reservation prices are updated regularly based on supply/demand, with scheduled updates (for example, “scheduled to be updated in April, 2026”). Mitigation: treat reservation rates as time-sensitive and pin a dated cost “snapshot” in the campaign’s approval packet. citeturn27search0  

## Comparison table

| Attribute | I1 ensemble | I2 persistent twin | I3 burst |
|---|---:|---:|---:|
| Instance type | p4d.24xlarge | p5.48xlarge | p5.48xlarge |
| Allocation | 42-node concurrency target (for 48h makespan) | 4 nodes always-on | 100 nodes for 24h (split across ≥2 Capacity Blocks) |
| GPUs total | 42 × 8 = 336 A100 | 4 × 8 = 32 H100 | 100 × 8 = 800 H100 |
| Wall-clock window | 48h campaign makespan (derived) | 720h | 24h |
| Compute proxy | $11.8/instance-hr | $31.464/instance-hr | $31.464/instance-hr |
| Compute cost | ~$23.6k (derived) | ~$90.6k (derived) | ~$75.5k (derived) |
| Expected primary outputs | Tier A/B artifacts per simulation, UQ reducer outputs | streaming state + periodic recalibration artifacts | bulk simulation outputs + global reducer |
| Data-volume driver | XTC cadence across 1,000 runs | retention policy for continuous operation | retention policy across 100 nodes |

Instance envelopes and EFA bandwidth cited from AWS P5 product details and P4d deep-dive. citeturn29view0turn24search8  
Capacity Block rates and constraints cited from AWS Capacity Blocks pricing and documentation. citeturn27search0turn18view1  

## Appendix dataset and bibliography

### Minimal dataset of sources used

CSV (fields required, no extras):

```csv
source_title,source_url,source_type,publish_date,event_date,geography/region,workload_family,technology_advancement_tag,metric_name,metric_value,metric_units,notes/assumptions,confidence
Amazon EC2 Capacity Blocks for ML Pricing,https://aws.amazon.com/ec2/capacityblocks/pricing/,vendor_pricing,2026-02-17,,us-east-1,MD,capacity planning,$/instance-hour (p4d.24xlarge capacity block),11.8,USD per instance-hour,Linux OS fee listed as $0.00 in table,high
Amazon EC2 Capacity Blocks for ML Pricing,https://aws.amazon.com/ec2/capacityblocks/pricing/,vendor_pricing,2026-02-17,,us-east-1,MD,capacity planning,$/accelerator-hour (A100 in p4d CB),1.475,USD per accelerator-hour,Derived from pricing table,high
Amazon EC2 Capacity Blocks for ML Pricing,https://aws.amazon.com/ec2/capacityblocks/pricing/,vendor_pricing,2026-02-17,,us-east-1,MD,capacity planning,$/instance-hour (p5.48xlarge capacity block),31.464,USD per instance-hour,From pricing table,high
Amazon EC2 Capacity Blocks for ML Pricing,https://aws.amazon.com/ec2/capacityblocks/pricing/,vendor_pricing,2026-02-17,,us-east-1,MD,capacity planning,$/accelerator-hour (H100 in p5 CB),3.933,USD per accelerator-hour,From pricing table,high
Amazon EC2 Capacity Blocks for ML expands to P4d instances,https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-ec2-capacity-blocks-ml-p4d-instances/,news,2024-02-01,2024-02-01,,MD,capacity planning,max capacity block duration,14,days,Durations one to 14 days,high
Amazon EC2 Capacity Blocks for ML expands to P4d instances,https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-ec2-capacity-blocks-ml-p4d-instances/,news,2024-02-01,2024-02-01,,MD,capacity planning,max instances per capacity block,64,instances,Cluster sizes 1 to 64 instances,high
Capacity Blocks for ML (EC2 User Guide),https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html,benchmark,2026-02-17,,,"MD",capacity planning,max instances per capacity block (doc),64,instances,Also documents max 256 instances across blocks,high
Launch instances using Capacity Blocks,https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-launch.html,vendor_pricing,2026-02-17,,,"MD",capacity planning,aws cli example uses capacity-block market type,,n/a,Used as submission command reference,high
Amazon EC2 P5 Instances product details,https://aws.amazon.com/ec2/instance-types/p5/,case_study,2026-02-17,,,"MD",gpu acceleration,vCPUs (p5.48xlarge),192,vCPU,From product details table,high
Amazon EC2 P5 Instances product details,https://aws.amazon.com/ec2/instance-types/p5/,case_study,2026-02-17,,,"MD",gpu acceleration,network bandwidth (p5.48xlarge),3200,Gbps,EFA bandwidth in product details,high
Amazon EC2 P4d instances deep dive,https://aws.amazon.com/blogs/compute/amazon-ec2-p4d-instances-deep-dive/,news,2020-11-02,2020-11-02,,MD,gpu acceleration,network bandwidth (p4d),400,Gbps,EFA-enabled networking described,high
Amazon EC2 P4d instances deep dive,https://aws.amazon.com/blogs/compute/amazon-ec2-p4d-instances-deep-dive/,news,2020-11-02,2020-11-02,,MD,gpu acceleration,vCPUs (p4d),96,vCPU,From blog overview section,high
GROMACS 2024.1 release notes,https://manual.gromacs.org/2024.6/release-notes/2024/2024.1.html,paper,2024-02-28,2024-02-28,,MD,solver advances,release date (GROMACS 2024.1),20240228,YYYYMMDD,Used to pin solver version,high
Gromacs — Best Practices (HPC Café PDF),https://hpc.fau.de/files/2022/03/20220308-hpc-cafe-gromacs-bestpractices.pdf,paper,2022-03-08,2022-03-08,,MD,workflow best practice,system 6 atoms,1066628,atoms,Used as representative industrial-scale particle count,high
Gromacs — Best Practices (HPC Café PDF),https://hpc.fau.de/files/2022/03/20220308-hpc-cafe-gromacs-bestpractices.pdf,paper,2022-03-08,2022-03-08,,MD,workflow best practice,env var GMX_GPU_DD_COMMS recommended,,n/a,Used for multi-GPU manifest,high
GROMACS 2024.1 on brand-new GPGPUs,https://hpc.fau.de/2024/08/13/gromacs-2024-1-on-brand-new-gpgpus/,benchmark,2024-08-13,2024-08-13,,MD,gpu acceleration,throughput system 6 on A100,23.2,ns/day,Single-GPU benchmark context,medium
File formats (GROMACS docs),https://manual.gromacs.org/2024.1/reference-manual/file-formats.html,paper,2024-02-28,,,"MD",data management,checkpoint file definition (cpt),,n/a,Defines cpt and edr semantics,high
Managing long simulations (GROMACS docs),https://manual.gromacs.org/2024.1/user-guide/managing-simulations.html,paper,2024-02-28,,,"MD",reproducibility,restart uses -cpi checkpoint,,n/a,Used for restart/runbook logic,high
libxtc paper (PMC),https://pmc.ncbi.nlm.nih.gov/articles/PMC8017739/,paper,2021-03-19,2021-03-19,,MD,data compression,xtc vs tng storage factor,1.3,ratio,XTC requires ~1.3x less disk space than tng in cited mode,medium
Amazon S3 pricing assumptions (AWS calculator assumptions),https://calculator.aws/static/cost_assumption/AmazonS3.html,vendor_pricing,2026-02-17,,us-east-1,MD,object storage,$/GB-month (S3 Standard first 50 TB),0.023,USD per GB-month,Used for storage cost proxy,medium
Amazon EBS volume types (gp3),https://aws.amazon.com/ebs/volume-types/,vendor_pricing,2026-02-17,,us-east-1,MD,storage,$/GB-month (gp3),0.08,USD per GB-month,Baseline perf includes 3000 IOPS and 125 MB/s,high
EC2 On-Demand data transfer pricing (tiered),https://aws.amazon.com/ec2/pricing/on-demand/,vendor_pricing,2026-02-17,,us-east-1,MD,data transfer,Data transfer out first-tier price,0.09,USD per GB,Used as proxy for first 10 TB; check region/tier,medium
What counts as 1 core hour? (SimScale),https://www.simscale.com/product/pricing/,vendor_pricing,2026-02-17,,,"CFD/FEA",pricing normalization,core-hour definition,,n/a,Used only for normalization concept,high
Licensing Resources (Ansys),https://www.ansys.com/it-solutions/licensing,vendor_pricing,2021-01-01,,,"CFD/FEA",licensing model,Elastic licensing described as pay-per-use,,n/a,Pricing not publicly listed,medium
Amazon FSx for Lustre Pricing,https://aws.amazon.com/fsx/lustre/pricing/,vendor_pricing,2026-02-17,,us-east-1,MD,storage,$/GB-month (example persistent),0.145,USD per GB-month,From pricing example; region-dependent,medium
```

JSON (same fields):

```json
[
  {
    "source_title": "Amazon EC2 Capacity Blocks for ML Pricing",
    "source_url": "https://aws.amazon.com/ec2/capacityblocks/pricing/",
    "source_type": "vendor_pricing",
    "publish_date": "2026-02-17",
    "event_date": null,
    "geography/region": "us-east-1",
    "workload_family": "MD",
    "technology_advancement_tag": "capacity planning",
    "metric_name": "$/instance-hour (p4d.24xlarge capacity block)",
    "metric_value": 11.8,
    "metric_units": "USD per instance-hour",
    "notes/assumptions": "Linux OS fee listed as $0.00 in table",
    "confidence": "high"
  },
  {
    "source_title": "Amazon EC2 Capacity Blocks for ML Pricing",
    "source_url": "https://aws.amazon.com/ec2/capacityblocks/pricing/",
    "source_type": "vendor_pricing",
    "publish_date": "2026-02-17",
    "event_date": null,
    "geography/region": "us-east-1",
    "workload_family": "MD",
    "technology_advancement_tag": "capacity planning",
    "metric_name": "$/instance-hour (p5.48xlarge capacity block)",
    "metric_value": 31.464,
    "metric_units": "USD per instance-hour",
    "notes/assumptions": "From pricing table",
    "confidence": "high"
  },
  {
    "source_title": "Amazon EC2 Capacity Blocks for ML expands to P4d instances",
    "source_url": "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-ec2-capacity-blocks-ml-p4d-instances/",
    "source_type": "news",
    "publish_date": "2024-02-01",
    "event_date": "2024-02-01",
    "geography/region": null,
    "workload_family": "MD",
    "technology_advancement_tag": "capacity planning",
    "metric_name": "max capacity block duration",
    "metric_value": 14,
    "metric_units": "days",
    "notes/assumptions": "Durations one to 14 days",
    "confidence": "high"
  },
  {
    "source_title": "Amazon EC2 P5 Instances product details",
    "source_url": "https://aws.amazon.com/ec2/instance-types/p5/",
    "source_type": "case_study",
    "publish_date": "2026-02-17",
    "event_date": null,
    "geography/region": null,
    "workload_family": "MD",
    "technology_advancement_tag": "gpu acceleration",
    "metric_name": "network bandwidth (p5.48xlarge)",
    "metric_value": 3200,
    "metric_units": "Gbps",
    "notes/assumptions": "EFA bandwidth in product details",
    "confidence": "high"
  },
  {
    "source_title": "Gromacs — Best Practices (HPC Café PDF)",
    "source_url": "https://hpc.fau.de/files/2022/03/20220308-hpc-cafe-gromacs-bestpractices.pdf",
    "source_type": "paper",
    "publish_date": "2022-03-08",
    "event_date": "2022-03-08",
    "geography/region": null,
    "workload_family": "MD",
    "technology_advancement_tag": "workflow best practice",
    "metric_name": "system 6 atoms",
    "metric_value": 1066628,
    "metric_units": "atoms",
    "notes/assumptions": "Used as representative industrial-scale particle count",
    "confidence": "high"
  }
]
```

### Bibliography

- AWS Capacity Blocks pricing tables and OS charges. citeturn27search0turn28view0  
- Capacity Blocks constraints (durations, max instances) and operational limitations. citeturn18view0turn18view1turn19search1  
- P5 instance product details and P4d deep-dive specs. citeturn29view0turn24search8  
- ParallelCluster creation and Capacity Block queue configuration concepts. citeturn9view4turn9view5turn19search21  
- GROMACS 2024.1 release pinning, output files, and restart semantics. citeturn10search6turn20search3turn20search5turn20search1  
- NHR@FAU benchmark set size and throughput context; multi-GPU best practices. citeturn13view0turn10search0turn12view2  
- Storage and volume pricing references (S3, EBS gp3, FSx for Lustre) and transfer pricing proxies. citeturn8search4turn30search5turn9view0turn7search0turn9view2