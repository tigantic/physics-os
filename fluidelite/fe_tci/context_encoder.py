"""
Context Encoder for FluidElite-TCI

Encodes variable-length token sequences into integer indices for TCI sampling.
This extends Phase 6's 4-byte context to arbitrary length contexts.

The key insight: TCI samples a function f: Z → Z (integer to integer).
We need to map token sequences to integers in a collision-free way.
"""

import numpy as np
from typing import List, Tuple, Union


class ContextEncoder:
    """
    Encode token sequences as integers for TCI sampling.
    
    For a sequence of n tokens where each token is in [0, vocab_size),
    we encode as:
        index = t_0 + t_1 * V + t_2 * V^2 + ... + t_{n-1} * V^{n-1}
    
    This is a bijection between sequences and integers.
    
    For TCI, we need the number of "qubits" (binary digits) to represent
    all possible contexts:
        n_qubits = ceil(log2(vocab_size^context_length))
                 = context_length * ceil(log2(vocab_size))
    """
    
    def __init__(self, vocab_size: int = 256, context_length: int = 8):
        """
        Args:
            vocab_size: Size of vocabulary (default 256 for bytes)
            context_length: Number of tokens in context window
        """
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        # Number of binary digits needed
        self.bits_per_token = int(np.ceil(np.log2(vocab_size)))
        self.n_qubits = context_length * self.bits_per_token
        
        # Total number of possible contexts
        self.n_contexts = vocab_size ** context_length
        
        # Precompute powers for encoding
        self._powers = np.array([vocab_size ** i for i in range(context_length)], 
                                dtype=np.uint64)
    
    def encode(self, tokens: Union[List[int], np.ndarray]) -> int:
        """
        Encode a token sequence as an integer.
        
        Args:
            tokens: Sequence of token IDs (length must equal context_length)
            
        Returns:
            Integer index in [0, vocab_size^context_length)
        """
        tokens = np.asarray(tokens, dtype=np.uint64)
        if len(tokens) != self.context_length:
            raise ValueError(f"Expected {self.context_length} tokens, got {len(tokens)}")
        
        return int(np.sum(tokens * self._powers))
    
    def decode(self, index: int) -> np.ndarray:
        """
        Decode an integer back to token sequence.
        
        Args:
            index: Integer in [0, vocab_size^context_length)
            
        Returns:
            Array of token IDs
        """
        tokens = np.zeros(self.context_length, dtype=np.int64)
        remaining = index
        
        for i in range(self.context_length):
            tokens[i] = remaining % self.vocab_size
            remaining //= self.vocab_size
        
        return tokens
    
    def encode_batch(self, token_sequences: np.ndarray) -> np.ndarray:
        """
        Encode multiple token sequences.
        
        Args:
            token_sequences: Array of shape (batch, context_length)
            
        Returns:
            Array of integer indices of shape (batch,)
        """
        return np.sum(token_sequences.astype(np.uint64) * self._powers, axis=1)
    
    def to_binary(self, index: int) -> np.ndarray:
        """
        Convert index to binary representation for TCI.
        
        Args:
            index: Integer context index
            
        Returns:
            Binary array of shape (n_qubits,)
        """
        binary = np.zeros(self.n_qubits, dtype=np.int32)
        remaining = index
        
        for i in range(self.n_qubits):
            binary[i] = remaining % 2
            remaining //= 2
        
        return binary
    
    def from_binary(self, binary: np.ndarray) -> int:
        """
        Convert binary representation back to index.
        
        Args:
            binary: Binary array of shape (n_qubits,)
            
        Returns:
            Integer context index
        """
        powers = 2 ** np.arange(self.n_qubits, dtype=np.uint64)
        return int(np.sum(binary.astype(np.uint64) * powers))
    
    @property
    def shape_info(self) -> dict:
        """Return info about the encoding space."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "bits_per_token": self.bits_per_token,
            "n_qubits": self.n_qubits,
            "n_contexts": self.n_contexts,
            "max_index": self.n_contexts - 1,
        }


def build_context_oracle(corpus: bytes, context_length: int) -> dict:
    """
    Build a lookup table from contexts to next tokens.
    
    This is the oracle function that TCI will sample:
        f(context_index) → next_token
    
    Args:
        corpus: Byte sequence (training data)
        context_length: Length of context window
        
    Returns:
        Dictionary mapping context_index → next_token
    """
    encoder = ContextEncoder(vocab_size=256, context_length=context_length)
    oracle = {}
    
    for i in range(len(corpus) - context_length):
        context = list(corpus[i:i + context_length])
        next_token = corpus[i + context_length]
        
        ctx_idx = encoder.encode(context)
        oracle[ctx_idx] = next_token
    
    return oracle


def oracle_to_function(oracle: dict, default: int = 0):
    """
    Convert oracle dict to a callable function for TCI.
    
    Args:
        oracle: Dictionary from context_index → next_token
        default: Default value for unseen contexts
        
    Returns:
        Function f(index) → token
    """
    def f(index: int) -> int:
        return oracle.get(index, default)
    
    return f


if __name__ == "__main__":
    # Quick test
    enc = ContextEncoder(vocab_size=256, context_length=8)
    print(f"Context encoder info: {enc.shape_info}")
    
    # Test encode/decode roundtrip
    tokens = [72, 101, 108, 108, 111, 32, 87, 111]  # "Hello Wo"
    idx = enc.encode(tokens)
    decoded = enc.decode(idx)
    print(f"Tokens: {tokens}")
    print(f"Encoded: {idx}")
    print(f"Decoded: {list(decoded)}")
    print(f"Roundtrip OK: {list(decoded) == tokens}")
    
    # Test binary conversion
    binary = enc.to_binary(idx)
    recovered = enc.from_binary(binary)
    print(f"Binary roundtrip OK: {recovered == idx}")
