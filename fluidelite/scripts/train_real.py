#!/usr/bin/env python3
"""
Real Text Training Script
=========================

Trains FluidElite on actual text data for language modeling.

Constitutional Compliance:
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
    - Article VII.7.4: Demonstration requirement with terminal output
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fluidelite.llm.fluid_elite import FluidElite
from fluidelite.optim.riemannian import RiemannianAdam
from fluidelite.llm.data import create_loader
from fluidelite.core.mps import MPS


# Training configuration
CONFIG = {
    "L": 8,               # 8 bits = 256 char vocab (ASCII)
    "RANK": 64,           # Higher rank for real language
    "BATCH": 32,          # Batch size
    "LR": 0.002,          # Learning rate
    "FILE": "input.txt",  # Training data file
    "SEQ_LEN": 64,        # Training context window (BPTT limit)
    "EPOCHS": 5,          # Number of epochs
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


def create_sample_data(path: Path):
    """
    Create sample training data if file doesn't exist.
    
    Args:
        path: Path to create sample data at
    """
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """ * 100  # Repeat for more training data
    
    with open(path, "w") as f:
        f.write(sample_text)
    print(f"Created sample training data at {path}")


def train():
    """
    Main training loop for real text language modeling.
    """
    print("=" * 60)
    print("FluidElite: Real Text Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    device = torch.device(CONFIG["DEVICE"])
    print(f"\n🚀 Initializing FluidElite on {device}...")
    
    # Create sample data if needed
    data_path = Path(CONFIG["FILE"])
    if not data_path.exists():
        create_sample_data(data_path)
    
    # Load data
    loader, dataset = create_loader(
        CONFIG["FILE"], 
        CONFIG["BATCH"], 
        CONFIG["SEQ_LEN"]
    )
    print(f"   Data batches per epoch: {len(loader)}")
    
    # Create model
    model = FluidElite(
        num_sites=CONFIG["L"], 
        rank=CONFIG["RANK"], 
        vocab_size=256  # ASCII
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Optimizer and loss
    optimizer = RiemannianAdam(model.parameters(), lr=CONFIG["LR"])
    criterion = nn.CrossEntropyLoss()
    
    print("\n⚔️  Training Started...")
    print("-" * 60)
    
    model.train()
    
    for epoch in range(CONFIG["EPOCHS"]):
        total_loss = 0.0
        num_batches = 0
        start = time.time()
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Start with empty context
            ctx = MPS.random(CONFIG["L"], 2, 1, device=device, dtype=torch.float64)
            
            loss_seq = 0.0
            
            # Process sequence (using first item in batch for simplicity)
            # Note: Full batch processing requires BatchMPS which is future work
            for t in range(x.shape[1]):
                token_in = x[0, t].item()
                target = y[0:1, t]
                
                ctx = model.step(ctx, token_in)
                logits = model.predict(ctx)
                
                loss = criterion(logits.unsqueeze(0), target)
                loss.backward(retain_graph=(t < x.shape[1] - 1))
                loss_seq += loss.item()
            
            optimizer.step()
            
            total_loss += loss_seq / x.shape[1]
            num_batches += 1
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch} | Batch {batch_idx:4d} | Avg Loss: {avg_loss:.4f}")
                
            # Limit batches per epoch for development
            if batch_idx >= 50:
                break
                
        epoch_time = time.time() - start
        epoch_loss = total_loss / num_batches
        
        print("-" * 60)
        print(f"Epoch {epoch} Complete | Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        print("-" * 60)
        
        # Save checkpoint
        checkpoint_path = f"fluidelite_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


def generate_text(model: FluidElite, prompt: str, length: int = 100, device=None) -> str:
    """
    Generate text from a trained model.
    
    Args:
        model: Trained FluidElite model
        prompt: Starting text
        length: Number of characters to generate
        device: Device to use
        
    Returns:
        Generated text string
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # Initialize context
    ctx = MPS.random(model.L, 2, 1, device=device, dtype=torch.float64)
    
    # Process prompt
    for char in prompt:
        token_id = ord(char) % 256
        ctx = model.step(ctx, token_id)
    
    # Generate
    generated = prompt
    with torch.no_grad():
        for _ in range(length):
            logits = model.predict(ctx)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Clamp to printable ASCII
            next_char = chr(max(32, min(126, next_token)))
            generated += next_char
            
            ctx = model.step(ctx, next_token)
    
    return generated


if __name__ == "__main__":
    train()
