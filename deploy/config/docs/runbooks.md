# Operational Runbooks — Trustless Physics Prover

> **Audience:** On-call engineers and SREs operating the FluidElite
> Trustless Physics certificate generation system.
>
> **Last updated:** 2026-02-14
>
> **Alert → Runbook mapping:**
>
> | Alert | Runbook |
> |-------|---------|
> | `ProofFailureRateHigh` | [Prover Restart](#1-prover-restart) |
> | `ProverStalled` | [Prover Restart](#1-prover-restart) |
> | `PodRestartLooping` | [Prover Restart](#1-prover-restart) |
> | `NoReadyPods` | [Prover Restart](#1-prover-restart) |
> | `VRAMUsageHigh` / `VRAMUsageCritical` | [GPU Failure Recovery](#3-gpu-failure-recovery) |
> | `CertificateVerificationFailure` | [Signing Key Rotation](#4-certificate-signing-key-rotation) |
> | `API5xxRateHigh` / `API5xxRateCritical` | [Incident Response](#5-incident-response-escalation) |
> | `DiskUsageHigh` / `CertificateStorageFull` | [Certificate Storage Cleanup](#6-certificate-storage-cleanup) |

---

## 1. Prover Restart

**When to use:** Prover pod is crash-looping, stalled (no proofs generated
despite incoming requests), or exhibiting unexpected proof failures.

### Diagnosis

```bash
# 1. Check pod status
kubectl get pods -n fluidelite -l app.kubernetes.io/name=fluidelite

# 2. Check recent events
kubectl describe pod <POD_NAME> -n fluidelite | tail -30

# 3. Check prover logs for errors
kubectl logs <POD_NAME> -n fluidelite -c prover --tail=200 | \
  jq 'select(.level == "error" or .level == "ERROR")'

# 4. Check resource usage
kubectl top pod <POD_NAME> -n fluidelite

# 5. Check if Halo2 key generation completed (startup takes 60-120s)
kubectl logs <POD_NAME> -n fluidelite -c prover | grep "Key generation complete"
```

### Resolution

**Soft restart (preferred — zero-downtime if replicas > 1):**

```bash
# Delete the specific unhealthy pod; Deployment controller recreates it.
kubectl delete pod <POD_NAME> -n fluidelite

# Verify replacement pod starts and becomes ready.
kubectl get pods -n fluidelite -l app.kubernetes.io/name=fluidelite -w
```

**Full rollout restart (all pods, rolling):**

```bash
kubectl rollout restart deployment/fluidelite -n fluidelite
kubectl rollout status deployment/fluidelite -n fluidelite --timeout=300s
```

**If using Argo Rollouts (blue-green):**

```bash
# Abort any in-progress rollout.
kubectl argo rollouts abort fluidelite-prover -n fluidelite

# Retry from current stable version.
kubectl argo rollouts retry rollout fluidelite-prover -n fluidelite
```

### Root Cause Investigation

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| OOMKilled | Memory limit too low for circuit size | Increase `resources.limits.memory` |
| CrashLoopBackOff on startup | Missing config or weights volume | Check PVC mounts, ConfigMap |
| Startup probe failing | Halo2 key gen taking >5 min | Increase `startupProbe.failureThreshold` |
| Proof failures after restart | Stale proving key cache | Delete pod, let key regen |

### Verification

```bash
# Confirm prover is healthy.
curl -sf http://$(kubectl get svc fluidelite -n fluidelite -o jsonpath='{.spec.clusterIP}'):8443/health

# Confirm proofs can be generated.
curl -sf http://$(kubectl get svc fluidelite -n fluidelite -o jsonpath='{.spec.clusterIP}'):8443/stats | jq .
```

---

## 2. KZG Parameter Rotation

**When to use:** Upgrading circuit parameters, moving to a new trusted setup
ceremony, or rotating KZG parameters as part of a security policy.

### Pre-Rotation Checklist

- [ ] New KZG parameters have been generated and verified (see [Trusted Setup Ceremony](../docs/trusted-setup-ceremony.md))
- [ ] New parameters are available in the PVC or S3 bucket
- [ ] New prover binary is built against the new parameters
- [ ] Verification key (VK) update has been queued on the timelock contract

### Procedure

```bash
# 1. Upload new KZG parameters to PVC.
kubectl cp ./new-kzg-params.bin fluidelite/<POD_NAME>:/opt/trustless/data/kzg-params/params.bin

# 2. Build new container image with updated parameters baked in.
podman build -f deploy/Containerfile -t ghcr.io/tiganticlabz/trustless-physics:v1.x.0 .
podman push ghcr.io/tiganticlabz/trustless-physics:v1.x.0

# 3. Update Helm release with new image tag.
helm upgrade fluidelite ./deploy/config/k8s/chart \
  --set image.tag=v1.x.0 \
  -n fluidelite

# 4. Queue VK update on governance timelock (48h delay).
# Use the Foundry deployment script:
cd fluidelite-zk/foundry
forge script script/UpdateVK.s.sol:UpdateVK \
  --rpc-url $RPC_URL \
  --broadcast \
  --private-key $DEPLOYER_KEY

# 5. After 48h timelock, execute the VK update.
forge script script/ExecuteVK.s.sol:ExecuteVK \
  --rpc-url $RPC_URL \
  --broadcast \
  --private-key $EXECUTOR_KEY

# 6. Verify new VK is active on-chain.
cast call $VERIFIER_ADDRESS "verificationKeyHash()(bytes32)" --rpc-url $RPC_URL
```

### Rollback

```bash
# If the new parameters cause proof failures, roll back to the previous image.
helm rollback fluidelite -n fluidelite

# Cancel the pending VK update (if still within timelock).
forge script script/CancelVK.s.sol:CancelVK \
  --rpc-url $RPC_URL \
  --broadcast \
  --private-key $CANCELLER_KEY
```

### Verification

```bash
# Generate a test proof with the new parameters.
curl -X POST http://<PROVER_IP>:8443/prove \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"token_id": 0, "domain": "thermal"}'

# Verify the proof with the new VK.
curl -X POST http://<PROVER_IP>:8443/verify \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @proof.json
```

---

## 3. GPU Failure Recovery

**When to use:** GPU VRAM utilization >90%, CUDA errors in logs, GPU
temperature exceeding thermal limits, or nvidia-smi showing ECC errors.

### Diagnosis

```bash
# 1. Check GPU status on the node.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- nvidia-smi

# 2. Check for CUDA errors in prover logs.
kubectl logs <POD_NAME> -n fluidelite -c prover | grep -i "cuda\|gpu\|vram\|oom"

# 3. Check the NVIDIA device plugin health.
kubectl get pods -n kube-system -l app=nvidia-device-plugin-daemonset
kubectl logs -n kube-system -l app=nvidia-device-plugin-daemonset --tail=50

# 4. Check node-level GPU metrics.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,ecc.errors.corrected.total,ecc.errors.uncorrected.total --format=csv
```

### Resolution

**VRAM exhaustion (soft):**

```bash
# Scale down to reduce concurrent proofs.
kubectl scale deployment/fluidelite -n fluidelite --replicas=1

# Wait for VRAM to clear.
sleep 30

# Scale back up.
kubectl scale deployment/fluidelite -n fluidelite --replicas=3
```

**GPU hardware failure:**

```bash
# 1. Cordon the node to prevent new pods from scheduling.
kubectl cordon <NODE_NAME>

# 2. Drain the node (graceful eviction respects PDB).
kubectl drain <NODE_NAME> --ignore-daemonsets --delete-emptydir-data

# 3. Resolve the GPU issue on the node (reboot, replace card, etc.).
ssh <NODE_NAME> "sudo reboot"

# 4. Once the node is healthy, uncordon.
kubectl uncordon <NODE_NAME>
```

**CUDA driver mismatch:**

```bash
# Verify driver version matches the container's CUDA toolkit.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- nvidia-smi | head -3
# Expected: Driver 550.x for CUDA 12.x

# If mismatched, update the node's driver or rebuild the container.
```

### Prevention

- Set `prover.maxConcurrentProofs` conservatively (2 for 16GB VRAM, 4 for 40GB)
- Enable VRAM alerting at 90% (already configured in PrometheusRules)
- Use `nvidia.com/gpu` resource limits in pod spec

---

## 4. Certificate Signing Key Rotation

**When to use:** Scheduled key rotation policy, suspected key compromise,
or certificate verification failures with valid proofs.

### Pre-Rotation Checklist

- [ ] New Ed25519 keypair generated securely (air-gapped machine or HSM)
- [ ] New public key registered with certificate consumers
- [ ] Transition period planned (both old and new keys accepted)

### Procedure

```bash
# 1. Generate new Ed25519 keypair.
# Use a secure machine — ideally air-gapped.
openssl genpkey -algorithm ed25519 -out signing-key-new.pem
openssl pkey -in signing-key-new.pem -pubout -out signing-pubkey-new.pem

# 2. Extract the raw 64-byte key for the prover (base64).
# The prover expects the full 64-byte Ed25519 keypair.
openssl pkey -in signing-key-new.pem -outform DER | \
  tail -c 64 | base64 > signing-key-new.b64

# 3. Update the Kubernetes Secret.
kubectl create secret generic fluidelite-secrets \
  --from-literal=api-key="$(kubectl get secret fluidelite-secrets -n fluidelite -o jsonpath='{.data.api-key}' | base64 -d)" \
  --from-file=signing-key=signing-key-new.b64 \
  --dry-run=client -o yaml | kubectl apply -n fluidelite -f -

# 4. Rolling restart to pick up the new key.
kubectl rollout restart deployment/fluidelite -n fluidelite
kubectl rollout status deployment/fluidelite -n fluidelite --timeout=300s

# 5. Publish the new public key to the certificate registry.
# (Application-specific — update the PQCCommitmentRegistry or DNS TXT record.)

# 6. Securely destroy the old private key material after the transition period.
shred -u signing-key-old.pem
```

### Verification

```bash
# Generate a certificate with the new key.
curl -X POST http://<PROVER_IP>:8443/prove \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"token_id": 42, "domain": "thermal"}' \
  -o cert.tpc

# Verify with the standalone verifier (uses new public key).
trustless-verify cert.tpc --pubkey signing-pubkey-new.pem
```

### Rollback

If the new key causes widespread verification failures:

```bash
# Revert the Secret to the old key.
kubectl create secret generic fluidelite-secrets \
  --from-file=signing-key=signing-key-old.b64 \
  --dry-run=client -o yaml | kubectl apply -n fluidelite -f -
kubectl rollout restart deployment/fluidelite -n fluidelite
```

---

## 5. Incident Response Escalation

**When to use:** Critical alerts firing, service degraded, customer-impacting
outage, or suspected security incident.

### Severity Levels

| Level | Criteria | Response Time | Escalation |
|-------|----------|---------------|------------|
| **SEV-1** | Service unavailable (0 ready pods), data integrity breach | 5 min | Page on-call + engineering lead |
| **SEV-2** | Degraded (>5% error rate, p99 >10s) | 15 min | Page on-call |
| **SEV-3** | Warning (>1% failure rate, VRAM >90%) | 30 min | Slack notification |

### Incident Procedure

```
1. ACKNOWLEDGE the alert in PagerDuty / Alertmanager.
2. ASSESS the impact:
   - How many customers affected?
   - Are proofs still being generated?
   - Is data integrity at risk?
3. MITIGATE (stop the bleeding):
   - If service is down: kubectl rollout restart
   - If under attack: enable air-gap mode, rotate API keys
   - If disk full: purge old certificates
4. COMMUNICATE:
   - Post in #fluidelite-incidents with: severity, impact, ETA
   - Update status page if customer-facing
5. INVESTIGATE root cause (after mitigation):
   - Check logs: kubectl logs -l app.kubernetes.io/name=fluidelite -c prover
   - Check metrics: Grafana dashboards
   - Check events: kubectl get events -n fluidelite --sort-by='.lastTimestamp'
6. RESOLVE and verify:
   - Confirm health check passes
   - Confirm proof generation succeeds
   - Confirm verification succeeds
7. POST-MORTEM (within 48h for SEV-1/SEV-2):
   - Timeline of events
   - Root cause analysis
   - Action items with owners and due dates
```

### Emergency Contacts

| Role | Contact | Escalation Path |
|------|---------|-----------------|
| On-call engineer | PagerDuty rotation | Slack #fluidelite-oncall |
| Engineering lead | (configured in PagerDuty) | Phone call after 10 min |
| Security lead | (configured in PagerDuty) | Immediate for data integrity |

---

## 6. Certificate Storage Cleanup

**When to use:** `DiskUsageHigh` or `CertificateStorageFull` alert fires.

### Diagnosis

```bash
# Check storage usage.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- \
  du -sh /opt/trustless/data/certificates/

# Count certificates by age.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- \
  find /opt/trustless/data/certificates/ -name "*.tpc" -mtime +30 | wc -l
```

### Resolution

```bash
# Option 1: Purge certificates older than 90 days.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- \
  find /opt/trustless/data/certificates/ -name "*.tpc" -mtime +90 -delete

# Option 2: Archive to object storage first, then purge.
kubectl exec -it <POD_NAME> -n fluidelite -c prover -- \
  tar czf /tmp/certs-archive.tar.gz /opt/trustless/data/certificates/
kubectl cp <POD_NAME>:/tmp/certs-archive.tar.gz -n fluidelite ./certs-archive.tar.gz
# Upload to S3/GCS, then purge.

# Option 3: Expand the PVC (if storage class supports it).
kubectl patch pvc fluidelite-certificates -n fluidelite \
  -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

### Prevention

- Set `retention_days` in `deployment.toml` (e.g., `retention_days = 90`)
- Configure S3 lifecycle policies for long-term certificate archival
- Monitor the `CertificateStorageFull` alert threshold

---

## Quick Reference

```bash
# Health check
curl -sf http://<IP>:8443/health | jq .

# Prover stats
curl -sf http://<IP>:8443/stats | jq .

# Prometheus metrics
curl -sf http://<IP>:9090/metrics | head -30

# Verify a certificate file
kubectl exec -it <POD_NAME> -n fluidelite -- \
  trustless-verify /opt/trustless/data/certificates/<ID>.tpc

# Force HPA scale-out
kubectl scale deployment/fluidelite -n fluidelite --replicas=5

# View Argo Rollout status
kubectl argo rollouts get rollout fluidelite-prover -n fluidelite -w
```
