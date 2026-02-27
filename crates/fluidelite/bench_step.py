#!/usr/bin/env python3
"""Quick benchmark: how long is one training step?"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import math
from datasets import load_dataset

device = torch.device('cuda')
N_FEATURES = 16384
N_VOCAB = 256
CTX_LEN = 16
N_QUBITS_FEAT = 14
TOTAL_QUBITS = 22
MAX_RANK = 24

@triton.jit
def extract_features_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i).to(tl.int32)
        idx = (i * 256 + byte_val) % 1024
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 4096)
        tl.atomic_add(out_base + idx, 1.0)

def extract_features(data, positions):
    B = positions.shape[0]
    features = torch.zeros((B, N_FEATURES), dtype=torch.float32, device=device)
    extract_features_kernel[(B,)](data, positions, features, N_FEATURES)
    return features

class QTTMatrix:
    def __init__(self, cores):
        self.cores = cores
    @classmethod
    def random_init(cls, n_qubits, max_rank):
        cores = []
        r_left = 1
        for i in range(n_qubits):
            r_right = min(max_rank, 2 ** min(i + 1, n_qubits - i - 1)) if i < n_qubits - 1 else 1
            core = torch.randn(r_left, 2, r_right, device=device) * math.sqrt(2.0 / (r_left + r_right))
            cores.append(core)
            r_left = r_right
        return cls(cores)
    def to_dense(self, n_row_qubits):
        result = self.cores[0]
        for core in self.cores[1:]:
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(result.shape[0], -1, result.shape[-1])
        return result.squeeze(0).squeeze(-1).reshape(2**n_row_qubits, -1)

print('Loading data...')
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
train_text = '\n'.join(dataset['train']['text']).encode('utf-8')
train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
print(f'Data loaded: {len(train_data)/1e6:.1f}M bytes')

# Setup
qtt = QTTMatrix.random_init(TOTAL_QUBITS, MAX_RANK)
for c in qtt.cores: c.requires_grad_(True)
optimizer = torch.optim.Adam(qtt.cores, lr=0.01)
batch_size = 4000

# Warm up
positions = torch.randint(0, len(train_data) - CTX_LEN - 1, (batch_size,), device=device)
targets = train_data[positions + CTX_LEN]
x = extract_features(train_data, positions)
logits = x @ qtt.to_dense(N_QUBITS_FEAT)
loss = F.cross_entropy(logits, targets.long())
loss.backward()
optimizer.step()
torch.cuda.synchronize()

# Benchmark 20 steps
print('\nBenchmarking 20 steps...')
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    positions = torch.randint(0, len(train_data) - CTX_LEN - 1, (batch_size,), device=device)
    targets = train_data[positions + CTX_LEN]
    x = extract_features(train_data, positions)
    optimizer.zero_grad()
    logits = x @ qtt.to_dense(N_QUBITS_FEAT)
    loss = F.cross_entropy(logits, targets.long())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qtt.cores, max_norm=1.0)
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.time() - start

print(f'\n{"="*50}')
print(f'TIMING RESULTS')
print(f'{"="*50}')
print(f'20 steps: {elapsed:.2f}s')
print(f'Per step: {elapsed/20*1000:.1f}ms')
print(f'Steps/sec: {20/elapsed:.1f}')
print(f'\n{"="*50}')
print(f'FULL CORPUS ESTIMATE')
print(f'{"="*50}')
n_samples = len(train_data) - CTX_LEN - 1
batches_per_epoch = n_samples // batch_size
print(f'Batches/epoch: {batches_per_epoch:,}')
print(f'Time/epoch: {batches_per_epoch * elapsed/20 / 60:.1f} minutes')
print(f'10 epochs: {10 * batches_per_epoch * elapsed/20 / 3600:.1f} hours')
