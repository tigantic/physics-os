# ═══════════════════════════════════════════════════════════════════════════════
# FluidElite Incident Runbooks
# ═══════════════════════════════════════════════════════════════════════════════
#
# On-call reference for production alert response. Each runbook corresponds
# to a Prometheus alert rule and provides step-by-step resolution.
#
# On-Call Rotation: PagerDuty schedule "fluidelite-gpu-oncall"
#   - Primary: @gpu-team (L1, 15-min ACK SLA)
#   - Secondary: @infra-team (L2, 30-min ACK SLA)
#   - Escalation: @engineering-lead (L3, 1-hour ACK SLA)
# ═══════════════════════════════════════════════════════════════════════════════

## SLA Targets

| Metric               | Target       | Measurement Window |
|----------------------|-------------|--------------------|
| Uptime               | 99.9%       | Rolling 30 days    |
| Aggregate TPS        | ≥ 88        | Rolling 1 hour     |
| Proof p99 latency    | < 5s        | Rolling 5 minutes  |
| Verification failure | < 0.01%     | Rolling 5 minutes  |
| Alert ACK time       | < 15 min    | Per incident       |
| Resolution time      | < 4 hours   | CRITICAL severity  |
| Resolution time      | < 24 hours  | WARNING severity   |

---

## Runbook: ProofThroughputLow / ProofThroughputCritical

**Alert:** TPS below 10 (warning) or below 1 (critical)

### Diagnosis

```bash
# 1. Check prover pod status
kubectl -n fluidelite get pods -l app=gpu-prover -o wide

# 2. Check GPU health
kubectl -n fluidelite exec -it <prover-pod> -- nvidia-smi

# 3. Check proof queue
kubectl -n fluidelite exec -it <prover-pod> -- curl -s localhost:9090/metrics | grep proof

# 4. Check for OOM kills
kubectl -n fluidelite describe pod <prover-pod> | grep -A5 "Last State"

# 5. Check ICICLE GPU initialization
kubectl -n fluidelite logs <prover-pod> | grep -i "icicle\|gpu\|cuda"
```

### Resolution

1. **GPU not detected:** Restart pod to re-initialize CUDA context
   ```bash
   kubectl -n fluidelite rollout restart deployment/gpu-prover
   ```

2. **OOM kill:** VRAM exhaustion — check `CudaMemoryPool` configuration
   ```bash
   # Verify pool size vs available VRAM
   kubectl -n fluidelite logs <prover-pod> | grep "pool_capacity"
   # Reduce batch size if needed
   kubectl -n fluidelite set env deployment/gpu-prover PROVER_MAX_BATCH=4
   ```

3. **Params not loaded:** SRS file missing or corrupt
   ```bash
   # Check params cache PVC
   kubectl -n fluidelite exec -it <prover-pod> -- ls -la /data/params/
   # Re-download if needed
   kubectl -n fluidelite exec -it <prover-pod> -- /app/download-params.sh
   ```

4. **Network partition:** Check inter-pod connectivity
   ```bash
   kubectl -n fluidelite exec -it <prover-pod> -- curl -s http://certificate-authority:8080/health
   ```

### Escalation

If throughput does not recover within 30 minutes, escalate to L2. If critical
alert persists > 1 hour, page engineering lead.

---

## Runbook: VerificationFailureRate

**Alert:** > 1% of proofs failing verification

**Severity:** CRITICAL — potential soundness issue

### Diagnosis

```bash
# 1. Identify failing proof domains
kubectl -n fluidelite logs <prover-pod> --since=10m | grep "VERIFICATION_FAILED"

# 2. Check if VK matches deployed circuits
kubectl -n fluidelite exec -it <prover-pod> -- /app/check-vk-hash.sh

# 3. Capture a failed proof for analysis
kubectl -n fluidelite exec -it <prover-pod> -- \
  curl -s localhost:9090/debug/last-failed-proof > /tmp/failed_proof.bin

# 4. Check for non-determinism (run same input twice)
kubectl -n fluidelite exec -it <prover-pod> -- /app/determinism-check.sh
```

### Resolution

1. **VK mismatch:** Circuit was updated but VK not redeployed
   ```bash
   # Rebuild and redeploy with correct VK
   cargo build --release --features production
   # Extract new VK hash
   ./target/release/vk-extractor > vk_hash.txt
   # Update on-chain VK
   ./deploy/config/scripts/deploy_mainnet.sh --update-vk
   ```

2. **Non-deterministic witness:** Race condition or uninitialized memory
   - Immediately halt all proof generation
   - File CRITICAL security incident
   - Engage audit firm for emergency review

3. **GPU computation error:** ECC or thermal corruption
   - Drain affected GPU node
   - Run CUDA memtest: `cuda-memcheck --tool racecheck ./prover`
   - Replace GPU if ECC errors detected

### Escalation

**IMMEDIATE ESCALATION** to engineering lead and security team. Verification
failures may indicate a soundness bug. Halt certificate issuance until
root cause confirmed.

---

## Runbook: GpuMemoryHigh

**Alert:** GPU VRAM > 90% used

### Diagnosis

```bash
# 1. Check VRAM breakdown
kubectl -n fluidelite exec -it <prover-pod> -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 2. Check memory pool stats
kubectl -n fluidelite exec -it <prover-pod> -- \
  curl -s localhost:9090/metrics | grep cuda_memory_pool

# 3. Check for memory leaks (growing allocation count)
kubectl -n fluidelite exec -it <prover-pod> -- \
  curl -s localhost:9090/debug/pool-stats
```

### Resolution

1. **Expected high usage:** Large batch in progress — monitor, no action needed
2. **Memory leak:** Pool not resetting between batches
   ```bash
   # Force pool reset
   kubectl -n fluidelite exec -it <prover-pod> -- curl -X POST localhost:9090/debug/pool-reset
   ```
3. **Fragmentation:** Pool too small for current workload
   ```bash
   # Increase pool allocation percentage (default: 60%)
   kubectl -n fluidelite set env deployment/gpu-prover POOL_VRAM_PERCENT=70
   kubectl -n fluidelite rollout restart deployment/gpu-prover
   ```

---

## Runbook: GpuTemperatureCritical

**Alert:** GPU temperature > 90°C

### Diagnosis

```bash
# 1. Check fan speed and power draw
kubectl -n fluidelite exec -it <prover-pod> -- nvidia-smi -q -d POWER,TEMPERATURE

# 2. Check ambient temperature (if IPMI available)
ipmitool sensor | grep -i temp

# 3. Check if other GPUs on same node are also hot
kubectl -n fluidelite exec -it <prover-pod> -- nvidia-smi --query-gpu=index,temperature.gpu --format=csv
```

### Resolution

1. **Throttle workload:** Reduce batch size to lower GPU load
   ```bash
   kubectl -n fluidelite set env deployment/gpu-prover PROVER_MAX_BATCH=2
   ```
2. **Check cooling:** Verify datacenter HVAC, fan health
3. **Drain node:** If persistent, cordon and drain the node
   ```bash
   kubectl cordon <node-name>
   kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
   ```

---

## Runbook: CertificateAuthorityDown

**Alert:** CA service not responding

### Diagnosis

```bash
# 1. Check pod status
kubectl -n fluidelite get pods -l app=certificate-authority

# 2. Check logs
kubectl -n fluidelite logs -l app=certificate-authority --tail=50

# 3. Check signing key availability
kubectl -n fluidelite exec -it <ca-pod> -- ls -la /secrets/signing-key
```

### Resolution

1. **Pod crash:** Check OOM or panic in logs, restart
   ```bash
   kubectl -n fluidelite rollout restart deployment/certificate-authority
   ```
2. **Signing key missing:** Secret not mounted
   ```bash
   kubectl -n fluidelite get secret ca-signing-key -o yaml
   kubectl -n fluidelite rollout restart deployment/certificate-authority
   ```
3. **Database connectivity:** Check connection to certificate store

---

## Runbook: GpuEccErrors

**Alert:** Double-bit ECC errors on GPU

**Severity:** CRITICAL — hardware failure

### Resolution

1. **IMMEDIATELY** drain the node:
   ```bash
   kubectl cordon <node-name>
   kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
   ```
2. Run extended GPU diagnostics:
   ```bash
   nvidia-smi -q -d ECC
   cuda-memcheck --tool memcheck ./prover --self-test
   ```
3. If errors persist: RMA the GPU. Do NOT return to service.
4. Any proofs generated on this GPU since last known-good time must be
   re-verified on a healthy GPU.

---

## On-Call Handoff Checklist

At shift change, the outgoing on-call engineer must:

- [ ] Document any open incidents in PagerDuty with current status
- [ ] Update the #fluidelite-ops Slack channel with shift summary
- [ ] Confirm all CRITICAL alerts are either resolved or have active mitigation
- [ ] Verify Grafana dashboard shows green status for all panels
- [ ] Confirm proof TPS is within SLA (≥ 88 aggregate)
- [ ] Note any upcoming maintenance windows or deployments

---

## Incident Severity Classification

| Severity | Response Time | Resolution SLA | Examples |
|----------|-------------|----------------|----------|
| CRITICAL | 15 min ACK  | 4 hours        | Verification failures, CA down, ECC errors, TPS < 1 |
| WARNING  | 30 min ACK  | 24 hours       | Low throughput, high latency, VRAM pressure |
| INFO     | Next shift  | Best effort    | Low GPU util, minor config drift |

## Post-Incident Review

All CRITICAL incidents require a post-incident review within 5 business days:

1. **Timeline:** Chronological event log from detection to resolution
2. **Root Cause:** Technical analysis of the failure
3. **Impact:** Affected customers, certificates delayed, SLA impact
4. **Action Items:** Preventive measures with owners and deadlines
5. **Detection:** Were alerts timely? Any monitoring gaps?

Document in the incident tracker and link from PagerDuty.
