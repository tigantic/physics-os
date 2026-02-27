"""Quick tests: rank=32 with 65K features, rank=16 for extreme compression."""
import torch
import triton
import triton.language as tl
import math

device = torch.device('cuda')
print(f'GPU: {torch.cuda.get_device_name(0)}')

@triton.jit
def extract_64k_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    for i in tl.static_range(16):
        byte_val = tl.load(data_ptr + pos + i)
        idx = (i * 256 + byte_val) % 4096
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 4096 + ((i * 65537 + b1 * 257 + b2) % 16384)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 20480 + ((b1 * 65537 + b2 * 257 + b3) % 16384)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        b4 = tl.load(data_ptr + pos + i + 3)
        idx = 36864 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 16384)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 53248 + ((b1 * 257 + b3) % 12288)
        tl.atomic_add(out_base + idx, 1.0)

@triton.jit  
def extract_16k_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i)
        idx = (i * 256 + byte_val) % 1024
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 5120 + ((b1 * 65537 + b2 * 257 + b3) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        b4 = tl.load(data_ptr + pos + i + 3)
        idx = 9216 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 13312 + ((b1 * 257 + b3) % 3072)
        tl.atomic_add(out_base + idx, 1.0)

def extract_64k(data, positions):
    B = positions.shape[0]
    features = torch.zeros((B, 65536), dtype=torch.float32, device=device)
    extract_64k_kernel[(B,)](data, positions, features, 65536)
    return features

def extract_16k(data, positions):
    B = positions.shape[0]
    features = torch.zeros((B, 16384), dtype=torch.float32, device=device)
    extract_16k_kernel[(B,)](data, positions, features, 16384)
    return features

def qtt_to_dense(cores, n_row_qubits):
    result = cores[0]
    for core in cores[1:]:
        r_left, size, _ = result.shape
        result = torch.einsum('ijk,klm->ijlm', result, core)
        result = result.reshape(r_left, size * 2, -1)
    result = result.squeeze(0).squeeze(-1)
    n_col_qubits = len(cores) - n_row_qubits
    return result.reshape(2**n_row_qubits, 2**n_col_qubits)

def init_qtt(n_qubits, max_rank):
    cores = []
    r_left = 1
    for i in range(n_qubits):
        r_right = min(max_rank, 2 ** min(i + 1, n_qubits - i - 1)) if i < n_qubits - 1 else 1
        std = math.sqrt(2.0 / (r_left + r_right))
        core = torch.randn(r_left, 2, r_right, device=device) * std
        core.requires_grad = True
        cores.append(core)
        r_left = r_right
    return cores

def quick_train(train_data, test_data, n_feat_qubits, max_rank, extract_fn, n_features):
    n_qubits = n_feat_qubits + 8
    cores = init_qtt(n_qubits, max_rank)
    n_params = sum(c.numel() for c in cores)
    compression = (n_features * 256) / n_params
    print(f'  {n_features} feat, rank={max_rank}: {n_params:,} params, {compression:.0f}x compression')
    
    n_train = len(train_data) - 17
    positions = torch.arange(0, n_train, n_train//200000, dtype=torch.int64, device=device)[:200000]
    targets = train_data[positions + 16]
    
    optimizer = torch.optim.Adam(cores, lr=0.03)
    
    for epoch in range(8):
        perm = torch.randperm(len(positions), device=device)
        for b in range(0, len(positions), 4096):
            idx = perm[b:b+4096]
            X = extract_fn(train_data, positions[idx])
            W = qtt_to_dense(cores, n_feat_qubits)
            logits = X @ W
            loss = torch.nn.functional.cross_entropy(logits, targets[idx].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    cores = [c.detach() for c in cores]
    n_test = len(test_data) - 17
    test_pos = torch.arange(0, n_test, n_test//20000, dtype=torch.int64, device=device)[:20000]
    X_test = extract_fn(test_data, test_pos)
    W = qtt_to_dense(cores, n_feat_qubits)
    logits = X_test @ W
    acc = (logits.argmax(1) == test_data[test_pos + 16]).float().mean().item()
    ppl = math.exp(torch.nn.functional.cross_entropy(logits, test_data[test_pos + 16].long()).item())
    print(f'    -> acc={acc*100:.1f}%, ppl={ppl:.2f}')
    return acc, ppl, n_params, compression

if __name__ == "__main__":
    from datasets import load_dataset
    print('Loading data...')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    train_text = '\n'.join(dataset['train']['text']).encode('utf-8')
    test_text = '\n'.join(dataset['test']['text']).encode('utf-8')
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)

    print('\n=== Quick Tests ===')
    print('\nTest 1: 65K features + rank=32 (scale up features)')
    quick_train(train_data, test_data, 16, 32, extract_64k, 65536)

    print('\nTest 2: 16K features + rank=16 (extreme compression)')
    quick_train(train_data, test_data, 14, 16, extract_16k, 16384)

    print('\nDone!')
