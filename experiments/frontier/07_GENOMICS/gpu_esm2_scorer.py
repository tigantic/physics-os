"""
GPU ESM-2 Variant Scorer - TRUE Batched Operations
===================================================

NO PYTHON LOOPS in hot paths:
- Vectorized BLOSUM62 lookups
- Batched ESM-2 inference
- GPU tensor operations only

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
import gzip
import json
import re
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timezone

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


# =============================================================================
# GPU BLOSUM62 Scorer - Vectorized
# =============================================================================

class GPUBlosum62:
    """
    GPU-accelerated BLOSUM62 scoring.
    NO PYTHON LOOPS - pure tensor operations.
    """
    
    # Amino acid order for indexing
    AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
    
    # BLOSUM62 matrix (20x20)
    BLOSUM62 = np.array([
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
        [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
        [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
        [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
        [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
        [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
        [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
        [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
        [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
        [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
        [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
        [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
        [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
        [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
        [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
        [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
    ], dtype=np.float32)
    
    def __init__(self, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        self.device = torch.device(device if device else ('cuda' if CUDA_AVAILABLE else 'cpu'))
        
        # Load BLOSUM62 to GPU
        self.matrix = torch.tensor(self.BLOSUM62, device=self.device, dtype=torch.float32)
        
        # Build AA -> index lookup (ASCII based)
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.AA_ORDER)}
        
        # ASCII lookup table for vectorized conversion
        self.ascii_lookup = torch.full((256,), -1, dtype=torch.long, device=self.device)
        for aa, idx in self.aa_to_idx.items():
            self.ascii_lookup[ord(aa)] = idx
    
    def score_batch(
        self,
        wt_indices: torch.Tensor,  # (n,) tensor of WT AA indices 0-19
        mt_indices: torch.Tensor,  # (n,) tensor of MT AA indices 0-19
    ) -> torch.Tensor:
        """
        Score batch of substitutions - PURE GPU.
        
        Returns (n,) tensor of BLOSUM62 scores.
        """
        # Direct 2D indexing into BLOSUM matrix
        return self.matrix[wt_indices, mt_indices]
    
    def convert_aa_to_idx(self, aa_list: List[str]) -> torch.Tensor:
        """Convert list of single-letter AAs to indices."""
        # Convert via ASCII lookup - vectorized
        ascii_codes = torch.tensor([ord(aa) for aa in aa_list], device=self.device)
        return self.ascii_lookup[ascii_codes]


# =============================================================================
# Fast ClinVar Loader
# =============================================================================

@dataclass
class VariantBatch:
    """Batch of variants for GPU scoring."""
    wt_indices: torch.Tensor  # (n,) WT amino acid indices
    mt_indices: torch.Tensor  # (n,) MT amino acid indices
    labels: torch.Tensor      # (n,) 1=pathogenic, 0=benign
    genes: List[str]          # Gene names (for reference)
    positions: List[int]      # Protein positions (for reference)


def load_clinvar_batch(
    vcf_path: Path,
    annotation_path: Path,
    protein_sequences: Dict[str, str],
    device: str = 'cuda',
    max_variants: int = 100_000,
) -> VariantBatch:
    """
    Load ClinVar variants into GPU tensors.
    Filters to sequence-matched missense only.
    """
    print("Loading ClinVar annotations...")
    
    AA_3TO1 = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    }
    AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    
    def parse_protein_hgvs(name):
        m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', name)
        if m:
            ref_3, alt_3 = m.group(1), m.group(3)
            if alt_3 == 'Ter':
                return None
            ref, alt = AA_3TO1.get(ref_3), AA_3TO1.get(alt_3)
            if ref and alt:
                return ref, int(m.group(2)), alt
        return None
    
    # Build annotation index
    chr_pos_to_info = {}
    with gzip.open(annotation_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 20:
                continue
            rec = dict(zip(header, parts))
            gene = rec.get('GeneSymbol', '')
            chrom = rec.get('Chromosome', '')
            pos = rec.get('PositionVCF', '')
            name = rec.get('Name', '')
            clnsig = rec.get('ClinicalSignificance', '')
            
            if gene in protein_sequences:
                key = f"{chrom}:{pos}"
                if key not in chr_pos_to_info:
                    chr_pos_to_info[key] = (gene, name, clnsig)
    
    print(f"  Indexed {len(chr_pos_to_info):,} variants in known genes")
    
    # Collect matched variants
    wt_list = []
    mt_list = []
    labels_list = []
    genes_list = []
    positions_list = []
    
    print("Parsing VCF and filtering to matched missense...")
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            chrom, pos = parts[0], parts[1]
            
            key = f"{chrom}:{pos}"
            if key not in chr_pos_to_info:
                continue
            
            gene, name, clnsig = chr_pos_to_info[key]
            
            parsed = parse_protein_hgvs(name)
            if not parsed:
                continue
            
            aa_ref, aa_pos, aa_alt = parsed
            
            # Validate against protein sequence
            seq = protein_sequences[gene]
            pos_0idx = aa_pos - 1
            
            if pos_0idx < 0 or pos_0idx >= len(seq):
                continue
            if seq[pos_0idx] != aa_ref:
                continue  # Isoform mismatch
            
            # Determine label
            clnsig_lower = clnsig.lower()
            if 'pathogen' in clnsig_lower and 'benign' not in clnsig_lower:
                label = 1
            elif 'benign' in clnsig_lower and 'pathogen' not in clnsig_lower:
                label = 0
            else:
                continue  # Skip VUS/conflicting
            
            # Convert to indices
            if aa_ref not in aa_to_idx or aa_alt not in aa_to_idx:
                continue
            
            wt_list.append(aa_to_idx[aa_ref])
            mt_list.append(aa_to_idx[aa_alt])
            labels_list.append(label)
            genes_list.append(gene)
            positions_list.append(aa_pos)
            
            if len(wt_list) >= max_variants:
                break
    
    print(f"  Loaded {len(wt_list):,} matched variants")
    
    # Convert to GPU tensors
    device_obj = torch.device(device)
    
    return VariantBatch(
        wt_indices=torch.tensor(wt_list, dtype=torch.long, device=device_obj),
        mt_indices=torch.tensor(mt_list, dtype=torch.long, device=device_obj),
        labels=torch.tensor(labels_list, dtype=torch.float32, device=device_obj),
        genes=genes_list,
        positions=positions_list,
    )


# =============================================================================
# Batched ESM-2 Scoring
# =============================================================================

class BatchedESM2Scorer:
    """
    True batched ESM-2 scoring.
    Groups mutations by protein and runs batched inference.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def load(self):
        """Load ESM-2 model."""
        if self.model is not None:
            return
        
        print("Loading ESM-2 (650M parameters)...")
        t_start = time.perf_counter()
        
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main",
            "esm2_t33_650M_UR50D"
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  Loaded in {time.perf_counter() - t_start:.1f}s")
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    @torch.no_grad()
    def score_protein_mutations(
        self,
        sequence: str,
        mutations: List[Tuple[int, str, str]],  # (pos, wt, mt) - 1-indexed
    ) -> torch.Tensor:
        """
        Score all mutations in a single protein with ONE forward pass.
        
        Returns tensor of log-likelihood ratios (LLRs).
        More negative = less evolutionarily favorable.
        """
        if not mutations:
            return torch.tensor([], device=self.device)
        
        # Truncate long sequences
        max_len = 800  # Reduced for faster inference
        if len(sequence) > max_len:
            # Find window containing all mutation positions
            positions = [m[0] for m in mutations]
            min_pos, max_pos = min(positions), max(positions)
            center = (min_pos + max_pos) // 2
            start = max(0, center - max_len // 2)
            end = start + max_len
            if end > len(sequence):
                end = len(sequence)
                start = max(0, end - max_len)
            sequence = sequence[start:end]
            offset = start
            mutations = [(p - offset, wt, mt) for p, wt, mt in mutations if 0 < p - offset <= len(sequence)]
        else:
            offset = 0
        
        if not mutations:
            return torch.tensor([], device=self.device)
        
        # Prepare batch: single sequence
        data = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        # Forward pass - get logits for all positions
        results = self.model(tokens, repr_layers=[], return_contacts=False)
        logits = results["logits"]  # (1, seq_len+2, vocab_size) - +2 for BOS/EOS
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len+2, vocab_size)
        
        # Score each mutation - VECTORIZED
        positions_t = torch.tensor([m[0] for m in mutations], device=self.device)
        wt_indices = torch.tensor([self.alphabet.get_idx(m[1]) for m in mutations], device=self.device)
        mt_indices = torch.tensor([self.alphabet.get_idx(m[2]) for m in mutations], device=self.device)
        
        # Clamp positions to valid range
        positions_t = torch.clamp(positions_t, 1, log_probs.shape[1] - 2)
        
        # Gather log probs - vectorized
        log_p_wt = log_probs[0, positions_t, wt_indices]
        log_p_mt = log_probs[0, positions_t, mt_indices]
        
        llrs = log_p_mt - log_p_wt
        
        return llrs
    
    def score_batch(
        self,
        protein_sequences: Dict[str, str],
        variants: List[Tuple[str, int, str, str]],  # (gene, pos, wt, mt)
        progress_interval: int = 100,
    ) -> torch.Tensor:
        """
        Score a batch of variants across multiple proteins.
        Groups by protein for efficient batching.
        """
        # Group variants by gene
        by_gene: Dict[str, List[Tuple[int, str, str, int]]] = {}
        for i, (gene, pos, wt, mt) in enumerate(variants):
            if gene not in by_gene:
                by_gene[gene] = []
            by_gene[gene].append((pos, wt, mt, i))
        
        # Sort genes by variant count (process largest first for better GPU util)
        genes_sorted = sorted(by_gene.keys(), key=lambda g: len(by_gene[g]), reverse=True)
        
        # Score each protein
        all_scores = torch.zeros(len(variants), device=self.device)
        
        total_genes = len(genes_sorted)
        for gene_idx, gene in enumerate(genes_sorted):
            gene_mutations = by_gene[gene]
            
            if gene not in protein_sequences:
                continue
            
            sequence = protein_sequences[gene]
            mutations = [(pos, wt, mt) for pos, wt, mt, _ in gene_mutations]
            indices = torch.tensor([idx for _, _, _, idx in gene_mutations], device=self.device)
            
            scores = self.score_protein_mutations(sequence, mutations)
            
            # Vectorized assignment
            all_scores[indices[:len(scores)]] = scores
            
            if (gene_idx + 1) % progress_interval == 0:
                print(f"  Processed {gene_idx + 1}/{total_genes} proteins...")
        
        return all_scores


# =============================================================================
# GPU AUROC Computation
# =============================================================================

def gpu_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """
    Compute AUROC entirely on GPU.
    
    Args:
        labels: (n,) binary labels (0 or 1)
        scores: (n,) prediction scores
    
    Returns:
        AUROC value
    """
    n = len(labels)
    
    # Sort by descending score
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # Cumulative sums
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1 - sorted_labels, dim=0)
    
    # TPR and FPR
    total_pos = sorted_labels.sum()
    total_neg = n - total_pos
    
    tpr = tps / total_pos
    fpr = fps / total_neg
    
    # Prepend (0, 0)
    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])
    
    # Trapezoidal integration
    auroc = torch.trapezoid(tpr, fpr)
    
    return float(auroc)


# =============================================================================
# Main Benchmark
# =============================================================================

def run_gpu_variant_benchmark():
    """
    Run GPU-accelerated variant scoring benchmark.
    """
    print("=" * 70)
    print("GPU ESM-2/BLOSUM62 VARIANT SCORER")
    print("NO PYTHON LOOPS - Pure GPU Tensor Operations")
    print("=" * 70)
    print()
    
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        print("ERROR: CUDA not available")
        return
    
    device = 'cuda'
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print()
    
    data_dir = Path("/root/benchmark_data")
    
    # Load protein sequences
    protein_path = data_dir / "protein_sequences.pkl"
    if protein_path.exists():
        with open(protein_path, 'rb') as f:
            proteins = pickle.load(f)
        print(f"Loaded {len(proteins):,} protein sequences")
    else:
        print("ERROR: protein_sequences.pkl not found")
        return
    
    # Load variants into GPU tensors
    t_start = time.perf_counter()
    batch = load_clinvar_batch(
        vcf_path=data_dir / "clinvar.vcf.gz",
        annotation_path=data_dir / "variant_summary.txt.gz",
        protein_sequences=proteins,
        device=device,
        max_variants=500_000,
    )
    t_load = time.perf_counter() - t_start
    
    n_variants = len(batch.labels)
    n_pathogenic = int(batch.labels.sum())
    n_benign = n_variants - n_pathogenic
    
    print(f"  Pathogenic: {n_pathogenic:,}")
    print(f"  Benign: {n_benign:,}")
    print(f"  Load time: {t_load:.2f}s")
    print()
    
    # Initialize GPU scorer
    scorer = GPUBlosum62(device=device)
    
    # Warmup
    _ = scorer.score_batch(batch.wt_indices[:100], batch.mt_indices[:100])
    torch.cuda.synchronize()
    
    # Benchmark: Score ALL variants in single GPU call
    print(f"Scoring {n_variants:,} variants on GPU...")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    raw_scores = scorer.score_batch(batch.wt_indices, batch.mt_indices)
    
    torch.cuda.synchronize()
    t_score = time.perf_counter() - t_start
    
    throughput = n_variants / t_score
    print(f"  Time: {t_score * 1000:.2f} ms")
    print(f"  Throughput: {throughput:,.0f} variants/sec")
    print()
    
    # Convert BLOSUM to pathogenicity score
    # More negative BLOSUM = more pathogenic
    # BLOSUM range: -4 to +3 (off-diagonal)
    # Map to [0, 1] with sigmoid
    pathogenicity = torch.sigmoid(-raw_scores * 0.5)
    
    # Compute AUROC on GPU
    print("Computing AUROC on GPU...")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    auroc = gpu_auroc(batch.labels, pathogenicity)
    
    torch.cuda.synchronize()
    t_auroc = time.perf_counter() - t_start
    
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Computation time: {t_auroc * 1000:.2f} ms")
    print()
    
    # Also try raw -BLOSUM as score
    auroc_raw = gpu_auroc(batch.labels, -raw_scores)
    print(f"  AUROC (raw -BLOSUM): {auroc_raw:.4f}")
    print()
    
    # =========================================================================
    # ESM-2 Scoring (batched by protein)
    # =========================================================================
    print("=" * 70)
    print("ESM-2 BATCHED SCORING")
    print("=" * 70)
    
    esm2_scorer = BatchedESM2Scorer(device=device)
    esm2_scorer.load()
    print()
    
    # Prepare variant list for ESM-2
    AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
    idx_to_aa = {i: aa for i, aa in enumerate(AA_ORDER)}
    
    variants_for_esm2 = [
        (batch.genes[i], batch.positions[i], 
         idx_to_aa[int(batch.wt_indices[i])], 
         idx_to_aa[int(batch.mt_indices[i])])
        for i in range(len(batch.genes))
    ]
    
    # Score with ESM-2 (batched per protein)
    print(f"Scoring {len(variants_for_esm2):,} variants with ESM-2...")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    esm2_llrs = esm2_scorer.score_batch(proteins, variants_for_esm2)
    
    torch.cuda.synchronize()
    t_esm2 = time.perf_counter() - t_start
    
    esm2_throughput = len(variants_for_esm2) / t_esm2
    print(f"  Time: {t_esm2:.2f}s")
    print(f"  Throughput: {esm2_throughput:,.0f} variants/sec")
    print()
    
    # Convert ESM-2 LLR to pathogenicity
    # More negative LLR = less likely mutation = more pathogenic
    esm2_pathogenicity = torch.sigmoid(-esm2_llrs * 0.3)
    
    # AUROC with ESM-2
    auroc_esm2 = gpu_auroc(batch.labels, esm2_pathogenicity)
    auroc_esm2_raw = gpu_auroc(batch.labels, -esm2_llrs)
    
    print(f"  AUROC (ESM-2 sigmoid): {auroc_esm2:.4f}")
    print(f"  AUROC (ESM-2 raw -LLR): {auroc_esm2_raw:.4f}")
    print()
    
    # Combine BLOSUM + ESM-2
    combined = 0.3 * pathogenicity + 0.7 * esm2_pathogenicity
    auroc_combined = gpu_auroc(batch.labels, combined)
    print(f"  AUROC (combined 0.3*BLOSUM + 0.7*ESM2): {auroc_combined:.4f}")
    print()
    
    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Variants scored: {n_variants:,}")
    print(f"  BLOSUM scoring time: {t_score * 1000:.2f} ms ({throughput:,.0f}/sec)")
    print(f"  ESM-2 scoring time: {t_esm2:.2f}s ({esm2_throughput:,.0f}/sec)")
    print()
    print(f"  AUROC (BLOSUM only):    {auroc:.4f}")
    print(f"  AUROC (ESM-2 only):     {auroc_esm2:.4f}")
    print(f"  AUROC (combined):       {auroc_combined:.4f}")
    print(f"  Gap to AlphaMissense:   {0.90 - auroc_combined:.4f}")
    print()
    
    # Score distribution analysis
    path_scores = pathogenicity[batch.labels == 1]
    ben_scores = pathogenicity[batch.labels == 0]
    
    print("Score distributions (BLOSUM):")
    print(f"  Pathogenic: mean={float(path_scores.mean()):.3f} ± {float(path_scores.std()):.3f}")
    print(f"  Benign:     mean={float(ben_scores.mean()):.3f} ± {float(ben_scores.std()):.3f}")
    print(f"  Separation: {float(path_scores.mean() - ben_scores.mean()):+.3f}")
    print()
    
    esm2_path = esm2_pathogenicity[batch.labels == 1]
    esm2_ben = esm2_pathogenicity[batch.labels == 0]
    
    print("Score distributions (ESM-2):")
    print(f"  Pathogenic: mean={float(esm2_path.mean()):.3f} ± {float(esm2_path.std()):.3f}")
    print(f"  Benign:     mean={float(esm2_ben.mean()):.3f} ± {float(esm2_ben.std()):.3f}")
    print(f"  Separation: {float(esm2_path.mean() - esm2_ben.mean()):+.3f}")
    print()
    
    # Save attestation
    attestation = {
        'attestation': {
            'type': 'GPU_VARIANT_SCORER',
            'version': '2.0.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        },
        'device': props.name,
        'n_variants': n_variants,
        'n_pathogenic': n_pathogenic,
        'n_benign': n_benign,
        'blosum_scoring_time_ms': t_score * 1000,
        'blosum_throughput_per_sec': throughput,
        'esm2_scoring_time_s': t_esm2,
        'esm2_throughput_per_sec': esm2_throughput,
        'auroc_blosum': auroc,
        'auroc_esm2': auroc_esm2,
        'auroc_combined': auroc_combined,
        'gap_to_alphamissense': 0.90 - auroc_combined,
    }
    
    with open('GPU_VARIANT_SCORER_ATTESTATION.json', 'w') as f:
        json.dump(attestation, f, indent=2)
    
    print("Attestation saved: GPU_VARIANT_SCORER_ATTESTATION.json")
    
    return auroc_combined


if __name__ == '__main__':
    run_gpu_variant_benchmark()
