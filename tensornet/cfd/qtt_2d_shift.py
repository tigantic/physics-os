"""
Native 2D Shift MPO for Morton-Ordered QTT

The key challenge: In Morton order, X and Y bits are interleaved:
  Core 0: x₀ (X bit)
  Core 1: y₀ (Y bit)  
  Core 2: x₁ (X bit)
  Core 3: y₁ (Y bit)
  ...

For Shift-X (+1 to X index):
- Even cores (X bits): Participate in ripple-carry addition
- Odd cores (Y bits): Pass through as identity, but MUST propagate carry state

The carry state must "skip over" Y-bit cores to reach the next X-bit core.
"""

import torch
from typing import List, Tuple
from dataclasses import dataclass

from tensornet.cfd.qtt_2d import (
    dense_to_qtt_2d, qtt_2d_to_dense, QTT2DState,
    morton_encode, morton_decode
)
from tensornet.cfd.pure_qtt_ops import QTTState


def build_shift_x_mpo_native(n_qubits: int, device: torch.device = None, 
                              dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    """
    Build native MPO for +1 shift in X direction (even bits) with interleaved layout.
    
    The MPO performs: |x, y⟩ → |x+1, y⟩ (modular arithmetic)
    
    MPO bond dimension tracks carry state: 
      - State 0: No carry pending
      - State 1: Carry pending for next X-bit
    
    For Y-bit cores (odd), we need identity that preserves carry state across.
    
    Returns:
        List of n_qubits MPO cores, each shape (r_left, 2, 2, r_right)
    """
    if device is None:
        device = torch.device('cpu')
    
    mpo = []
    
    # Count X-bits (even cores)
    n_x_bits = (n_qubits + 1) // 2
    x_bit_idx = 0  # Which X-bit are we at?
    
    for k in range(n_qubits):
        is_x_bit = (k % 2 == 0)
        is_first = (k == 0)
        is_last = (k == n_qubits - 1)
        
        if is_x_bit:
            # X-bit core: Participate in ripple-carry addition
            
            if x_bit_idx == 0:
                # First X-bit: Initiate +1 (always add 1)
                # Input carry state: 1 (always)
                # Output carry state: depends on bit value
                #
                # r_left = 1 (no incoming carry dimension needed - we always add)
                # r_right = 2 (carry/no-carry to next X-bit)
                
                core = torch.zeros(1, 2, 2, 2, dtype=dtype, device=device)
                
                # Input 0, add 1: output = 1, carry = 0
                core[0, 1, 0, 0] = 1.0  # |0⟩_in → |1⟩_out, state 0 (no carry)
                
                # Input 1, add 1: output = 0, carry = 1
                core[0, 0, 1, 1] = 1.0  # |1⟩_in → |0⟩_out, state 1 (carry)
                
                mpo.append(core)
                x_bit_idx += 1
                
            elif x_bit_idx == n_x_bits - 1:
                # Last X-bit: Accept carry, no output carry needed
                # r_left = 2 (carry state from previous)
                # r_right = 1 (no more X-bits)
                
                core = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
                
                # State 0 (no carry): pass through
                core[0, 0, 0, 0] = 1.0  # |0⟩ → |0⟩
                core[0, 1, 1, 0] = 1.0  # |1⟩ → |1⟩
                
                # State 1 (carry): add 1
                core[1, 1, 0, 0] = 1.0  # |0⟩ + 1 → |1⟩
                core[1, 0, 1, 0] = 1.0  # |1⟩ + 1 → |0⟩ (overflow wraps)
                
                mpo.append(core)
                x_bit_idx += 1
                
            else:
                # Middle X-bit: Accept carry, propagate carry
                # r_left = 2, r_right = 2
                
                core = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                
                # State 0 (no carry): pass through
                core[0, 0, 0, 0] = 1.0  # |0⟩ → |0⟩, stay state 0
                core[0, 1, 1, 0] = 1.0  # |1⟩ → |1⟩, stay state 0
                
                # State 1 (carry): add 1
                core[1, 1, 0, 0] = 1.0  # |0⟩ + 1 → |1⟩, no new carry (state 0)
                core[1, 0, 1, 1] = 1.0  # |1⟩ + 1 → |0⟩, generate carry (state 1)
                
                mpo.append(core)
                x_bit_idx += 1
        else:
            # Y-bit core: Pure identity, but propagate carry state through
            # 
            # The carry state from the previous X-bit must pass through
            # to the next X-bit without modification.
            
            # Get bond dimensions from neighboring cores
            prev_core = mpo[-1] if len(mpo) > 0 else None
            r_left = prev_core.shape[3] if prev_core is not None else 1
            
            # Check if there's another X-bit after this
            remaining_x_bits = n_x_bits - x_bit_idx
            r_right = 2 if remaining_x_bits > 0 else 1
            
            # Build identity that preserves bond dimension
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            
            # For each carry state, apply identity to physical dims
            for r in range(min(r_left, r_right)):
                core[r, 0, 0, r] = 1.0  # |0⟩ → |0⟩, preserve state r
                core[r, 1, 1, r] = 1.0  # |1⟩ → |1⟩, preserve state r
            
            mpo.append(core)
    
    return mpo


def build_shift_y_mpo_native(n_qubits: int, device: torch.device = None,
                              dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    """
    Build native MPO for +1 shift in Y direction (odd bits) with interleaved layout.
    
    The MPO performs: |x, y⟩ → |x, y+1⟩ (modular arithmetic)
    
    Symmetric to shift_x, but operates on odd cores instead of even.
    """
    if device is None:
        device = torch.device('cpu')
    
    mpo = []
    
    # Count Y-bits (odd cores)
    n_y_bits = n_qubits // 2
    y_bit_idx = 0
    
    for k in range(n_qubits):
        is_y_bit = (k % 2 == 1)
        is_first = (k == 0)
        is_last = (k == n_qubits - 1)
        
        if is_y_bit:
            # Y-bit core: Participate in ripple-carry for Y index
            
            if y_bit_idx == 0:
                # First Y-bit: Initiate +1
                
                # Need to check previous core for r_left
                prev_core = mpo[-1] if len(mpo) > 0 else None
                r_left = prev_core.shape[3] if prev_core is not None else 1
                
                core = torch.zeros(r_left, 2, 2, 2, dtype=dtype, device=device)
                
                # For each incoming state, apply +1
                for r in range(r_left):
                    # Input 0: output 1, no carry
                    core[r, 1, 0, 0] = 1.0
                    # Input 1: output 0, carry
                    core[r, 0, 1, 1] = 1.0
                
                mpo.append(core)
                y_bit_idx += 1
                
            elif y_bit_idx == n_y_bits - 1:
                # Last Y-bit
                
                core = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
                
                # State 0: pass through
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                
                # State 1: add 1
                core[1, 1, 0, 0] = 1.0
                core[1, 0, 1, 0] = 1.0
                
                mpo.append(core)
                y_bit_idx += 1
                
            else:
                # Middle Y-bit
                
                core = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                
                # State 0: pass through
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                
                # State 1: add 1
                core[1, 1, 0, 0] = 1.0
                core[1, 0, 1, 1] = 1.0
                
                mpo.append(core)
                y_bit_idx += 1
        else:
            # X-bit core: Pure identity, propagate Y carry state
            
            prev_core = mpo[-1] if len(mpo) > 0 else None
            r_left = prev_core.shape[3] if prev_core is not None else 1
            
            remaining_y_bits = n_y_bits - y_bit_idx
            r_right = 2 if remaining_y_bits > 0 else 1
            
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            
            for r in range(min(r_left, r_right)):
                core[r, 0, 0, r] = 1.0
                core[r, 1, 1, r] = 1.0
            
            mpo.append(core)
    
    return mpo


def apply_mpo_to_qtt2d(state: QTT2DState, mpo: List[torch.Tensor], 
                        max_rank: int = 64) -> QTT2DState:
    """
    Apply MPO to QTT2D state and truncate result.
    
    |ψ'⟩ = MPO |ψ⟩
    
    Core contraction:
        new[ml*sl, d_out, mr*sr] = sum_{d_in} mpo[ml, d_out, d_in, mr] * state[sl, d_in, sr]
    """
    new_cores = []
    
    for k in range(len(state.cores)):
        s_core = state.cores[k]  # (sl, d_in, sr)
        m_core = mpo[k]          # (ml, d_out, d_in, mr)
        
        sl, d_in, sr = s_core.shape
        ml, d_out, d_in_m, mr = m_core.shape
        
        assert d_in == d_in_m, f"Physical dimension mismatch at core {k}: {d_in} vs {d_in_m}"
        
        # Contract over d_in
        # result[ml, sl, d_out, mr, sr] = sum_{d_in} m[ml, d_out, d_in, mr] * s[sl, d_in, sr]
        result = torch.einsum('aoim,lid->alomid', m_core, s_core)
        
        # Wait, the einsum indices are wrong. Let me fix:
        # m_core: (ml, d_out, d_in, mr) indexed as 'aoid'
        # s_core: (sl, d_in, sr) indexed as 'ldr'
        # We want: result[a,l,o,d,r] -> but we contract over d_in
        
        # Correct einsum:
        result = torch.einsum('aobm,lbr->alomr', m_core, s_core)
        
        # Reshape: (ml*sl, d_out, mr*sr)
        result = result.reshape(ml * sl, d_out, mr * sr)
        
        new_cores.append(result)
    
    # Build result state
    result = QTT2DState(cores=new_cores, nx=state.nx, ny=state.ny)
    
    # Truncate via SVD sweep
    result = truncate_qtt2d_svd(result, max_rank)
    
    return result


def truncate_qtt2d_svd(state: QTT2DState, max_rank: int) -> QTT2DState:
    """Left-to-right SVD truncation sweep."""
    cores = [c.clone() for c in state.cores]
    n = len(cores)
    
    for k in range(n - 1):
        core = cores[k]
        r_left, d, r_right = core.shape
        
        # Reshape to matrix
        mat = core.reshape(r_left * d, r_right)
        
        # Randomized SVD (4× faster)
        try:
            q = min(max_rank, min(mat.shape))
            U, S, Vh = torch.svd_lowrank(mat, q=q, niter=1)
        except (RuntimeError, torch.linalg.LinAlgError):
            # Fallback for numerical issues (ill-conditioned matrix)
            continue
        
        # Determine rank
        rank = min(len(S), max_rank)
        
        # Threshold small singular values
        if len(S) > 0:
            threshold = 1e-12 * S[0].item()
            valid = (S > threshold).sum().item()
            rank = min(rank, max(valid, 1))
        
        # Truncate
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Update current core
        cores[k] = U.reshape(r_left, d, rank)
        
        # Absorb S @ Vh into next core
        SV = torch.diag(S) @ Vh  # (rank, r_right)
        
        next_core = cores[k + 1]  # (r_right, d_next, r_right_next)
        r_right_old, d_next, r_right_next = next_core.shape
        
        # Contract: SV @ next_core reshaped
        next_flat = next_core.reshape(r_right_old, d_next * r_right_next)
        new_next = SV @ next_flat  # (rank, d_next * r_right_next)
        cores[k + 1] = new_next.reshape(rank, d_next, r_right_next)
    
    return QTT2DState(cores=cores, nx=state.nx, ny=state.ny)


# =============================================================================
# Validation Tests
# =============================================================================

def test_native_shift_x():
    """Test native shift-X MPO on a simple pattern."""
    print("=" * 60)
    print("NATIVE SHIFT-X TEST")
    print("=" * 60)
    
    nx, ny = 4, 4  # 16×16 grid
    Nx, Ny = 2**nx, 2**ny
    n_qubits = 2 * max(nx, ny)
    
    # Create pattern: square at (4:8, 4:8)
    field = torch.zeros(Nx, Ny, dtype=torch.float32)
    field[4:8, 4:8] = 1.0
    
    print(f"Grid: {Nx}×{Ny}, Qubits: {n_qubits}")
    print(f"Original square at x=[4,8), y=[4,8)")
    
    # Compress
    qtt = dense_to_qtt_2d(field, max_bond=16)
    print(f"Max rank before: {qtt.max_rank}")
    
    # Build native shift-X MPO
    shift_x = build_shift_x_mpo_native(n_qubits)
    
    print(f"\nShift-X MPO cores:")
    for i, core in enumerate(shift_x):
        bit_type = "X" if i % 2 == 0 else "Y"
        print(f"  Core {i} ({bit_type}): {core.shape}")
    
    # Apply shift
    qtt_shifted = apply_mpo_to_qtt2d(qtt, shift_x, max_rank=32)
    print(f"\nMax rank after shift: {qtt_shifted.max_rank}")
    
    # Decompress
    shifted = qtt_2d_to_dense(qtt_shifted)
    
    # Expected: square at x=[5,9), y=[4,8)
    expected = torch.roll(field, shifts=1, dims=0)
    
    # Check error
    err = (shifted - expected).abs().max().item()
    print(f"Error vs expected: {err:.2e}")
    
    # Find center of mass
    weights = shifted / (shifted.sum() + 1e-10)
    X, Y = torch.meshgrid(torch.arange(Nx).float(), torch.arange(Ny).float(), indexing='ij')
    cx = (weights * X).sum().item()
    cy = (weights * Y).sum().item()
    
    print(f"Center of mass: ({cx:.1f}, {cy:.1f})")
    print(f"Expected: (6.5, 5.5) → (7.5, 5.5) after +X shift")
    
    return err < 0.1


def test_native_shift_y():
    """Test native shift-Y MPO."""
    print("\n" + "=" * 60)
    print("NATIVE SHIFT-Y TEST")
    print("=" * 60)
    
    nx, ny = 4, 4
    Nx, Ny = 2**nx, 2**ny
    n_qubits = 2 * max(nx, ny)
    
    field = torch.zeros(Nx, Ny, dtype=torch.float32)
    field[4:8, 4:8] = 1.0
    
    print(f"Grid: {Nx}×{Ny}")
    
    qtt = dense_to_qtt_2d(field, max_bond=16)
    
    shift_y = build_shift_y_mpo_native(n_qubits)
    
    print(f"\nShift-Y MPO cores:")
    for i, core in enumerate(shift_y):
        bit_type = "X" if i % 2 == 0 else "Y"
        print(f"  Core {i} ({bit_type}): {core.shape}")
    
    qtt_shifted = apply_mpo_to_qtt2d(qtt, shift_y, max_rank=32)
    shifted = qtt_2d_to_dense(qtt_shifted)
    
    expected = torch.roll(field, shifts=1, dims=1)
    err = (shifted - expected).abs().max().item()
    print(f"\nError vs expected: {err:.2e}")
    
    return err < 0.1


def test_combined_shift():
    """Test X+Y shift sequence."""
    print("\n" + "=" * 60)
    print("COMBINED X+Y SHIFT TEST")
    print("=" * 60)
    
    nx, ny = 5, 5  # 32×32
    Nx, Ny = 2**nx, 2**ny
    n_qubits = 2 * max(nx, ny)
    
    # Gaussian blob
    x = torch.linspace(0, 1, Nx, dtype=torch.float32)
    y = torch.linspace(0, 1, Ny, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    gaussian = torch.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / (2 * 0.08**2))
    
    qtt = dense_to_qtt_2d(gaussian, max_bond=32)
    print(f"Grid: {Nx}×{Ny}")
    print(f"Initial rank: {qtt.max_rank}")
    
    # Build MPOs
    shift_x = build_shift_x_mpo_native(n_qubits)
    shift_y = build_shift_y_mpo_native(n_qubits)
    
    # Apply 5 steps of (+X, +Y)
    n_steps = 5
    for step in range(n_steps):
        qtt = apply_mpo_to_qtt2d(qtt, shift_x, max_rank=48)
        qtt = apply_mpo_to_qtt2d(qtt, shift_y, max_rank=48)
    
    print(f"After {n_steps} steps, rank: {qtt.max_rank}")
    
    # Check center moved
    result = qtt_2d_to_dense(qtt)
    weights = result / (result.sum() + 1e-10)
    cx = (weights * X).sum().item()
    cy = (weights * Y).sum().item()
    
    dx = n_steps / Nx
    expected_cx = 0.3 + dx
    expected_cy = 0.3 + dx
    
    print(f"Center: ({cx:.3f}, {cy:.3f})")
    print(f"Expected: ({expected_cx:.3f}, {expected_cy:.3f})")
    
    err = abs(cx - expected_cx) + abs(cy - expected_cy)
    print(f"Position error: {err:.4f}")
    
    return err < 0.1


if __name__ == "__main__":
    t1 = test_native_shift_x()
    t2 = test_native_shift_y()
    t3 = test_combined_shift()
    
    print("\n" + "=" * 60)
    print("NATIVE 2D SHIFT SUMMARY")
    print("=" * 60)
    print(f"Shift-X: {'✅ PASS' if t1 else '❌ FAIL'}")
    print(f"Shift-Y: {'✅ PASS' if t2 else '❌ FAIL'}")
    print(f"Combined: {'✅ PASS' if t3 else '❌ FAIL'}")
    
    if t1 and t2 and t3:
        print("\n🎯 Native 2D Shift MPOs validated!")
        print("   Ready for Strang splitting with no dense round-trip.")
