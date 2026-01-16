"""
Text Data Loader for FluidElite
===============================

Provides dataset and dataloader utilities for training on text data.

Constitutional Compliance:
    - Article V.5.1: All public classes/functions documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class TextStreamDataset(Dataset):
    """
    Dataset that ingests raw text and chunks it into training sequences.
    
    Uses character-level tokenization where each character maps to its
    ASCII value (or custom vocabulary if text contains non-ASCII).
    
    Args:
        text_data: Raw text string
        vocab_size: Maximum vocabulary size (default 256 for ASCII)
        seq_len: Length of each training sequence
        
    Attributes:
        stoi: String-to-index mapping
        itos: Index-to-string mapping
        data: Tokenized text as tensor
        
    Example:
        >>> text = "Hello, world!"
        >>> dataset = TextStreamDataset(text, vocab_size=256, seq_len=32)
        >>> x, y = dataset[0]  # x: input chunk, y: target chunk (shifted by 1)
    """
    
    def __init__(self, text_data: str, vocab_size: int = 256, seq_len: int = 64):
        self.text = text_data
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Build vocabulary from unique characters in text
        chars = sorted(list(set(text_data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Tokenize entire text
        self.data = torch.tensor(
            [self.stoi[c] for c in text_data], 
            dtype=torch.long
        )
        
        print(f"Dataset Loaded: {len(self.data)} tokens, Vocab: {len(chars)} unique chars")

    def __len__(self) -> int:
        """Number of training examples (overlapping windows)."""
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Index into the text
            
        Returns:
            (input_chunk, target_chunk) where target is shifted by 1
        """
        chunk = self.data[idx : idx + self.seq_len]
        target = self.data[idx + 1 : idx + self.seq_len + 1]
        return chunk, target
    
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Convert token tensor back to string.
        
        Args:
            tokens: Tensor of token indices
            
        Returns:
            Decoded string
        """
        return ''.join([self.itos.get(t.item(), '?') for t in tokens])
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Convert string to token tensor.
        
        Args:
            text: String to encode
            
        Returns:
            Tensor of token indices
        """
        return torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)


def create_loader(
    path: str | Path, 
    batch_size: int, 
    seq_len: int,
    max_chars: Optional[int] = 1_000_000
) -> tuple[DataLoader, TextStreamDataset]:
    """
    Create a DataLoader from a text file.
    
    Args:
        path: Path to text file
        batch_size: Batch size for training
        seq_len: Sequence length for each training example
        max_chars: Maximum characters to load (default 1M for dev)
        
    Returns:
        (DataLoader, TextStreamDataset) tuple
        
    Example:
        >>> loader, dataset = create_loader("input.txt", batch_size=32, seq_len=64)
        >>> for x, y in loader:
        ...     # x.shape = (batch_size, seq_len)
        ...     # y.shape = (batch_size, seq_len)
    """
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Limit size for development
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        print(f"Truncated to {max_chars} characters for development")
    
    dataset = TextStreamDataset(text, vocab_size=256, seq_len=seq_len)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True  # Ensure consistent batch size
    )
    
    return loader, dataset


def create_synthetic_dataset(
    num_samples: int = 10000,
    seq_len: int = 64,
    vocab_size: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data for testing.
    
    Generates random sequences where each sample is (input, target)
    with target being the same as input shifted by 1.
    
    Args:
        num_samples: Number of training samples
        seq_len: Length of each sequence
        vocab_size: Size of vocabulary
        
    Returns:
        (inputs, targets) tensors of shape (num_samples, seq_len)
    """
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # For a more interesting task: copy first token to end
    targets[:, -1] = inputs[:, 0]
    
    return inputs, targets
