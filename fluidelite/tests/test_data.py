"""
Tests for fluidelite.llm.data

Constitutional Compliance:
    - Article II.2.2: 80% test coverage required
"""

import pytest
import torch
import tempfile
import os
from fluidelite.llm.data import TextStreamDataset, create_loader, create_synthetic_dataset


class TestTextStreamDataset:
    """Test TextStreamDataset functionality."""
    
    def test_creation_from_string(self):
        """Test dataset creation from string."""
        text = "Hello world! This is a test string for the dataset." * 10
        dataset = TextStreamDataset(text, seq_len=16)
        assert len(dataset) > 0
    
    def test_vocab_building(self):
        """Test vocabulary is built correctly."""
        text = "aabbcc" * 100
        dataset = TextStreamDataset(text, seq_len=8)
        
        assert 'a' in dataset.stoi
        assert 'b' in dataset.stoi
        assert 'c' in dataset.stoi
    
    def test_getitem(self):
        """Test __getitem__ returns correct structure."""
        text = "abcdefghijklmnopqrstuvwxyz " * 100
        dataset = TextStreamDataset(text, seq_len=8)
        
        if len(dataset) > 0:
            x, y = dataset[0]
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert x.shape == (8,)
            assert y.shape == (8,)
    
    def test_len(self):
        """Test __len__ returns sensible value."""
        text = "word " * 100
        dataset = TextStreamDataset(text, seq_len=10)
        length = len(dataset)
        assert isinstance(length, int)
        assert length >= 0
    
    def test_decode(self):
        """Test decode method."""
        text = "hello world" * 100
        dataset = TextStreamDataset(text, seq_len=16)
        
        x, _ = dataset[0]
        decoded = dataset.decode(x)
        assert isinstance(decoded, str)
        assert len(decoded) == 16
    
    def test_encode(self):
        """Test encode method."""
        text = "hello world" * 100
        dataset = TextStreamDataset(text, seq_len=16)
        
        encoded = dataset.encode("hello")
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (5,)


class TestCreateLoader:
    """Test create_loader function."""
    
    def test_create_loader_basic(self):
        """Test basic dataloader creation."""
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("word " * 10000)
            temp_path = f.name
        
        try:
            loader, dataset = create_loader(temp_path, batch_size=4, seq_len=16)
            assert loader is not None
            assert dataset is not None
        finally:
            os.unlink(temp_path)
    
    def test_create_loader_iteration(self):
        """Test iterating over dataloader."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("token " * 10000)
            temp_path = f.name
        
        try:
            loader, dataset = create_loader(temp_path, batch_size=8, seq_len=16)
            
            # Get first batch
            x, y = next(iter(loader))
            assert x.shape == (8, 16)
            assert y.shape == (8, 16)
        finally:
            os.unlink(temp_path)


class TestCreateSyntheticDataset:
    """Test create_synthetic_dataset function."""
    
    def test_synthetic_shapes(self):
        """Test synthetic dataset has correct shapes."""
        inputs, targets = create_synthetic_dataset(num_samples=100, seq_len=32, vocab_size=50)
        
        assert inputs.shape == (100, 32)
        assert targets.shape == (100, 32)
    
    def test_synthetic_values(self):
        """Test synthetic values are in correct range."""
        inputs, targets = create_synthetic_dataset(num_samples=100, seq_len=32, vocab_size=50)
        
        assert inputs.min() >= 0
        assert inputs.max() < 50
        assert targets.min() >= 0
        assert targets.max() < 50
    
    def test_synthetic_copy_task(self):
        """Test copy task property (first token copied to last)."""
        inputs, targets = create_synthetic_dataset(num_samples=100, seq_len=32, vocab_size=50)
        
        # Last target token should equal first input token
        assert torch.all(targets[:, -1] == inputs[:, 0])

