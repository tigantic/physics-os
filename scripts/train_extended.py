#!/usr/bin/env python3
"""
Extended Training Script for FluidEliteZK
Trains on TinyStories for higher quality text generation.

Run with:
    python scripts/train_extended.py --epochs 10 --dataset tinystories

Estimated time: ~4-8 hours on CPU, ~30min on GPU
Target: >40% accuracy (from current 18%)
"""

import os
import sys
import argparse
import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "fluidelite"))

try:
    from fluidelite.llm.fluid_elite_zk import FluidEliteZK
except ImportError:
    from llm.fluid_elite_zk import FluidEliteZK


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for character-level training."""
    
    def __init__(self, split: str = "train", max_chars: int = 10_000_000, window: int = 32):
        self.window = window
        
        # Download TinyStories if needed
        cache_dir = Path.home() / ".cache" / "fluidelite"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"tinystories_{split}.txt"
        
        if not cache_file.exists():
            print(f"Downloading TinyStories {split} split...")
            self._download_tinystories(split, cache_file)
        
        # Load text
        print(f"Loading {cache_file}...")
        with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()[:max_chars]
        
        # Build character vocabulary
        chars = sorted(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        
        # Encode text
        self.data = torch.tensor([self.char_to_idx[c] for c in self.text], dtype=torch.long)
        
        print(f"Loaded {len(self.text):,} characters, vocab size: {self.vocab_size}")
    
    def _download_tinystories(self, split: str, cache_file: Path):
        """Download TinyStories from HuggingFace."""
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split=split)
            
            # Combine all stories
            texts = []
            for item in ds:
                texts.append(item['text'])
                if len(''.join(texts)) > 50_000_000:  # Cap at 50M chars
                    break
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(texts))
            
        except ImportError:
            print("Please install datasets: pip install datasets")
            print("Falling back to WikiText-2...")
            self._download_wikitext(cache_file)
    
    def _download_wikitext(self, cache_file: Path):
        """Fallback to WikiText-2."""
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        urllib.request.urlretrieve(url, cache_file)
    
    def __len__(self):
        return len(self.data) - self.window - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window]
        y = self.data[idx + 1:idx + self.window + 1]
        return x, y


class WikiTextDataset(Dataset):
    """WikiText-2 dataset (smaller, faster)."""
    
    def __init__(self, split: str = "train", max_chars: int = 1_000_000, window: int = 32):
        self.window = window
        
        # Download
        import urllib.request
        cache_dir = Path.home() / ".cache" / "fluidelite"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"wikitext2_{split}.txt"
        
        if not cache_file.exists():
            url = f"https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/{split}.txt"
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, cache_file)
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            self.text = f.read()[:max_chars]
        
        chars = sorted(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.char_to_idx[c] for c in self.text], dtype=torch.long)
        
        print(f"Loaded {len(self.text):,} characters, vocab size: {self.vocab_size}")
    
    def __len__(self):
        return len(self.data) - self.window - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window]
        y = self.data[idx + 1:idx + self.window + 1]
        return x, y


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, log_interval=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - process each position
        batch_loss = 0
        for t in range(x.size(1)):
            token = x[:, t]
            target = y[:, t]
            
            logits = model(token)
            loss = criterion(logits, target)
            batch_loss += loss
            
            # Accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        batch_loss = batch_loss / x.size(1)
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += batch_loss.item()
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx * x.size(0)) / elapsed
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {batch_loss.item():.4f} | Acc: {100*correct/total:.1f}% | "
                  f"{samples_per_sec:.0f} samples/sec")
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            for t in range(x.size(1)):
                token = x[:, t]
                target = y[:, t]
                
                logits = model(token)
                loss = criterion(logits, target)
                total_loss += loss.item()
                
                pred = logits.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.size(0)
    
    avg_loss = total_loss / (len(dataloader) * x.size(1))
    accuracy = correct / total
    perplexity = math.exp(min(avg_loss, 10))  # Cap to avoid overflow
    
    return avg_loss, accuracy, perplexity


def main():
    parser = argparse.ArgumentParser(description="Extended FluidEliteZK Training")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "tinystories"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-chars", type=int, default=5_000_000)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FluidEliteZK Extended Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Max chars: {args.max_chars:,}")
    print("=" * 60)
    
    # Load dataset
    if args.dataset == "tinystories":
        train_ds = TinyStoriesDataset("train", max_chars=args.max_chars, window=args.window)
        val_ds = TinyStoriesDataset("validation", max_chars=args.max_chars // 10, window=args.window)
    else:
        train_ds = WikiTextDataset("train", max_chars=args.max_chars, window=args.window)
        val_ds = WikiTextDataset("valid", max_chars=args.max_chars // 10, window=args.window)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = FluidEliteZK(
        num_sites=12,
        bond_dim=64,
        phys_dim=2
    ).to(args.device)
    
    # Add a readout head for the vocabulary
    model.readout = nn.Linear(64, train_ds.vocab_size).to(args.device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Vocab size: {train_ds.vocab_size}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch
        )
        
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, args.device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.1f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.1f}% | Val PPL: {val_ppl:.1f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_ppl': val_ppl,
                'vocab_size': train_ds.vocab_size,
                'char_to_idx': train_ds.char_to_idx,
                'idx_to_char': train_ds.idx_to_char,
            }
            torch.save(checkpoint, save_dir / "fluidelite_zk_extended_best.pt")
            print(f"  ✓ Saved best model (val_acc: {100*val_acc:.1f}%)")
    
    # Final save
    print("\n" + "=" * 60)
    print("Exporting ZK-ready weights...")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(save_dir / "fluidelite_zk_extended_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export to ZK format
    zk_weights = model.export_zk_weights()
    zk_weights['char_to_idx'] = checkpoint['char_to_idx']
    zk_weights['idx_to_char'] = {str(k): v for k, v in checkpoint['idx_to_char'].items()}
    zk_weights['training'] = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'max_chars': args.max_chars,
        'final_val_acc': checkpoint['val_acc'],
        'final_val_ppl': checkpoint['val_ppl'],
    }
    
    zk_path = save_dir / "fluidelite_zk_extended_zk_weights.json"
    with open(zk_path, 'w') as f:
        json.dump(zk_weights, f)
    
    print(f"✓ Saved ZK weights: {zk_path}")
    print(f"  Size: {os.path.getsize(zk_path) / 1024:.1f} KB")
    print(f"  Final Accuracy: {100*checkpoint['val_acc']:.1f}%")
    print(f"  Final Perplexity: {checkpoint['val_ppl']:.1f}")
    
    print("\n🎉 Training complete!")
    print(f"Copy weights to prover: cp {zk_path} fluidelite-zk/data/")


if __name__ == "__main__":
    main()
