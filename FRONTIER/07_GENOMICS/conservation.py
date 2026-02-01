"""
FRONTIER 07: Conservation Analysis
===================================

Analyze sequence conservation using tensor network rank structure.

Key insight: Conservation correlates with low effective rank.
- Perfectly conserved positions: rank 1 (single state dominates)
- Variable positions: higher rank (multiple states have weight)
- Constrained covariation: intermediate rank (coupled positions)

Applications:
    - Identify functionally important regions
    - Detect selection pressure
    - Prioritize variants by evolutionary conservation

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from dna_tensor import DNATensorTrain, encode_sequence, decode_sequence

BASES = ['A', 'C', 'G', 'T']


@dataclass
class ConservationProfile:
    """Conservation analysis results for a sequence."""
    positions: np.ndarray              # Position indices
    conservation_scores: np.ndarray    # Per-position conservation [0,1]
    entropy_scores: np.ndarray         # Per-position entropy
    marginal_distributions: np.ndarray  # (N, 4) marginal probabilities
    
    # Aggregate statistics
    mean_conservation: float
    highly_conserved_count: int  # Score > 0.8
    variable_count: int          # Score < 0.3
    
    @property
    def sequence_length(self) -> int:
        return len(self.positions)
    
    def get_conserved_regions(
        self,
        min_length: int = 10,
        min_conservation: float = 0.7,
    ) -> list[tuple[int, int, float]]:
        """
        Find contiguous conserved regions.
        
        Returns:
            List of (start, end, mean_conservation) tuples.
        """
        regions = []
        in_region = False
        start = 0
        
        for i, score in enumerate(self.conservation_scores):
            if score >= min_conservation:
                if not in_region:
                    start = i
                    in_region = True
            else:
                if in_region:
                    if i - start >= min_length:
                        mean_cons = float(np.mean(self.conservation_scores[start:i]))
                        regions.append((start, i, mean_cons))
                    in_region = False
        
        # Handle region at end
        if in_region and len(self.positions) - start >= min_length:
            mean_cons = float(np.mean(self.conservation_scores[start:]))
            regions.append((start, len(self.positions), mean_cons))
        
        return regions
    
    def get_variable_positions(
        self,
        max_conservation: float = 0.3,
    ) -> list[int]:
        """Get positions with low conservation (highly variable)."""
        return [
            int(i) for i, score in enumerate(self.conservation_scores)
            if score <= max_conservation
        ]


class ConservationAnalyzer:
    """
    Analyze sequence conservation using tensor network decomposition.
    
    Example:
        >>> analyzer = ConservationAnalyzer()
        >>> msa = ["ACGTACGT...", "ACGTAGGT...", "ACGTACGT..."]
        >>> profile = analyzer.analyze_msa(msa)
        >>> print(f"Mean conservation: {profile.mean_conservation:.2%}")
    """
    
    def __init__(self, max_rank: int = 16):
        self.max_rank = max_rank
    
    def analyze_sequence(
        self,
        sequence: str,
        tensor_train: Optional[DNATensorTrain] = None,
    ) -> ConservationProfile:
        """
        Analyze conservation of a single sequence.
        
        For single sequences, conservation is based on the tensor network's
        learned probability distribution (biased toward observed sequence).
        
        Args:
            sequence: DNA sequence.
            tensor_train: Optional pre-fitted tensor train.
            
        Returns:
            ConservationProfile with per-position scores.
        """
        if tensor_train is None:
            tensor_train = DNATensorTrain.from_sequence(
                sequence, 
                max_rank=self.max_rank,
            )
        
        n = len(sequence)
        positions = np.arange(n)
        conservation_scores = np.zeros(n)
        entropy_scores = np.zeros(n)
        marginals = np.zeros((n, 4))
        
        for pos in range(n):
            marg = tensor_train.marginal_at_position(pos)
            marginals[pos] = marg
            
            # Shannon entropy
            entropy = -np.sum(marg * np.log(marg + 1e-10))
            max_entropy = np.log(4)  # Uniform distribution
            
            entropy_scores[pos] = entropy
            conservation_scores[pos] = 1.0 - (entropy / max_entropy)
        
        mean_cons = float(np.mean(conservation_scores))
        highly_conserved = int(np.sum(conservation_scores > 0.8))
        variable = int(np.sum(conservation_scores < 0.3))
        
        return ConservationProfile(
            positions=positions,
            conservation_scores=conservation_scores,
            entropy_scores=entropy_scores,
            marginal_distributions=marginals,
            mean_conservation=mean_cons,
            highly_conserved_count=highly_conserved,
            variable_count=variable,
        )
    
    def analyze_msa(
        self,
        sequences: list[str],
    ) -> ConservationProfile:
        """
        Analyze conservation from multiple sequence alignment.
        
        This is the primary use case - learning conservation from
        evolutionary comparisons across species.
        
        Args:
            sequences: List of aligned sequences (same length).
            
        Returns:
            ConservationProfile with true evolutionary conservation.
        """
        if not sequences:
            raise ValueError("Need at least one sequence")
        
        # Build tensor train from MSA
        tensor_train = DNATensorTrain.from_msa(
            sequences,
            max_rank=self.max_rank,
        )
        
        return self.analyze_sequence(sequences[0], tensor_train)
    
    def compare_sequences(
        self,
        seq1: str,
        seq2: str,
    ) -> dict:
        """
        Compare conservation between two sequences.
        
        Useful for comparing orthologs or paralogs.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence (same length).
            
        Returns:
            Comparison statistics.
        """
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must have same length")
        
        profile1 = self.analyze_sequence(seq1)
        profile2 = self.analyze_sequence(seq2)
        
        # Correlation of conservation profiles
        corr = np.corrcoef(
            profile1.conservation_scores,
            profile2.conservation_scores,
        )[0, 1]
        
        # Shared highly conserved positions
        conserved1 = set(np.where(profile1.conservation_scores > 0.8)[0])
        conserved2 = set(np.where(profile2.conservation_scores > 0.8)[0])
        shared_conserved = conserved1 & conserved2
        
        return {
            'correlation': float(corr),
            'mean_conservation_1': profile1.mean_conservation,
            'mean_conservation_2': profile2.mean_conservation,
            'shared_conserved_positions': len(shared_conserved),
            'unique_to_1': len(conserved1 - conserved2),
            'unique_to_2': len(conserved2 - conserved1),
        }


def identify_functional_regions(
    sequence: str,
    known_motifs: Optional[dict[str, str]] = None,
) -> dict:
    """
    Identify potential functional regions using conservation.
    
    Combines tensor network conservation with known motif detection.
    
    Args:
        sequence: DNA sequence.
        known_motifs: Optional dict of {name: motif_pattern}.
        
    Returns:
        Dictionary with functional region predictions.
    """
    if known_motifs is None:
        known_motifs = {
            'TATA_box': 'TATAAA',
            'GC_box': 'GGGCGG',
            'CAAT_box': 'CCAAT',
            'splice_donor': 'GT',
            'splice_acceptor': 'AG',
            'Kozak': 'ACCATGG',  # Translation start
        }
    
    analyzer = ConservationAnalyzer()
    profile = analyzer.analyze_sequence(sequence)
    
    # Find conserved regions
    conserved_regions = profile.get_conserved_regions(min_length=5)
    
    # Find motifs
    motif_hits = {}
    for name, motif in known_motifs.items():
        positions = []
        start = 0
        while True:
            pos = sequence.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        if positions:
            motif_hits[name] = positions
    
    # Find motifs in conserved regions
    functional_predictions = []
    for start, end, mean_cons in conserved_regions:
        region_seq = sequence[start:end]
        region_motifs = []
        for name, motif in known_motifs.items():
            if motif in region_seq:
                region_motifs.append(name)
        
        functional_predictions.append({
            'start': start,
            'end': end,
            'length': end - start,
            'mean_conservation': mean_cons,
            'motifs': region_motifs,
        })
    
    return {
        'conserved_regions': conserved_regions,
        'motif_hits': motif_hits,
        'functional_predictions': functional_predictions,
        'profile': profile,
    }


def run_validation() -> dict:
    """
    Validate conservation analysis.
    
    Returns:
        Validation results.
    """
    print("=" * 70)
    print("FRONTIER 07: Conservation Analysis")
    print("=" * 70)
    print()
    
    results = {
        'tests': {},
        'all_pass': True,
    }
    
    # Test 1: Single sequence conservation
    print("Test 1: Single Sequence Conservation Profile")
    print("-" * 70)
    
    sequence = "ACGTACGTACGTACGT" * 10  # 160 bases
    analyzer = ConservationAnalyzer(max_rank=8)
    
    t_start = time.perf_counter()
    profile = analyzer.analyze_sequence(sequence)
    t_end = time.perf_counter()
    
    test1_pass = (
        profile.sequence_length == len(sequence) and
        0 <= profile.mean_conservation <= 1 and
        len(profile.conservation_scores) == len(sequence)
    )
    
    results['tests']['single_sequence'] = {
        'sequence_length': len(sequence),
        'mean_conservation': profile.mean_conservation,
        'highly_conserved': profile.highly_conserved_count,
        'variable': profile.variable_count,
        'analysis_time_ms': (t_end - t_start) * 1000,
        'pass': test1_pass,
    }
    results['all_pass'] &= test1_pass
    
    print(f"  Sequence length: {len(sequence)}")
    print(f"  Mean conservation: {profile.mean_conservation:.3f}")
    print(f"  Highly conserved positions: {profile.highly_conserved_count}")
    print(f"  Variable positions: {profile.variable_count}")
    print(f"  Analysis time: {(t_end - t_start)*1000:.1f} ms")
    print(f"  Status: {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print()
    
    # Test 2: MSA conservation (simulated)
    print("Test 2: Multiple Sequence Alignment Conservation")
    print("-" * 70)
    
    # Create simulated MSA with conserved and variable regions
    base_seq = "ACGTACGTACGTACGT" * 10  # 160 bases
    
    # Create variants: first 40 bases conserved, rest variable
    msa = [base_seq]
    np.random.seed(42)
    for _ in range(19):
        variant = list(base_seq)
        # Mutate variable region (positions 40+)
        for pos in range(40, len(variant)):
            if np.random.random() < 0.3:  # 30% mutation rate
                variant[pos] = np.random.choice(['A', 'C', 'G', 'T'])
        msa.append(''.join(variant))
    
    t_start = time.perf_counter()
    msa_profile = analyzer.analyze_msa(msa)
    t_end = time.perf_counter()
    
    # Check that conserved region has higher conservation
    conserved_region_mean = float(np.mean(msa_profile.conservation_scores[:40]))
    variable_region_mean = float(np.mean(msa_profile.conservation_scores[40:]))
    
    test2_pass = conserved_region_mean > variable_region_mean
    
    results['tests']['msa_conservation'] = {
        'n_sequences': len(msa),
        'sequence_length': len(base_seq),
        'conserved_region_mean': conserved_region_mean,
        'variable_region_mean': variable_region_mean,
        'analysis_time_ms': (t_end - t_start) * 1000,
        'pass': test2_pass,
    }
    results['all_pass'] &= test2_pass
    
    print(f"  MSA size: {len(msa)} sequences × {len(base_seq)} bases")
    print(f"  Conserved region (0-40) mean: {conserved_region_mean:.3f}")
    print(f"  Variable region (40+) mean: {variable_region_mean:.3f}")
    print(f"  Conserved > Variable: {test2_pass}")
    print(f"  Analysis time: {(t_end - t_start)*1000:.1f} ms")
    print(f"  Status: {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print()
    
    # Test 3: Functional region identification
    print("Test 3: Functional Region Identification")
    print("-" * 70)
    
    # Create sequence with known functional elements
    test_seq = (
        "NNNNNTATAAAAANNN" +  # TATA box
        "GGGCGGNNNNCCAAT" +    # GC box + CAAT box
        "ACGTACGTACGTACGT" * 5 +  # Filler
        "GTNNNNNNNNNNNNAG" +    # Splice sites
        "ACGTACGTACGTACGT" * 3
    )
    
    result = identify_functional_regions(test_seq)
    
    # Check that motifs were found
    found_tata = 'TATA_box' in result['motif_hits']
    found_gc = 'GC_box' in result['motif_hits']
    found_splice = 'splice_donor' in result['motif_hits'] and 'splice_acceptor' in result['motif_hits']
    
    test3_pass = found_tata and found_gc and found_splice
    
    results['tests']['functional_regions'] = {
        'motifs_found': list(result['motif_hits'].keys()),
        'n_conserved_regions': len(result['conserved_regions']),
        'n_functional_predictions': len(result['functional_predictions']),
        'pass': test3_pass,
    }
    results['all_pass'] &= test3_pass
    
    print(f"  Sequence length: {len(test_seq)}")
    print(f"  Motifs found: {list(result['motif_hits'].keys())}")
    print(f"  Conserved regions: {len(result['conserved_regions'])}")
    print(f"  Functional predictions: {len(result['functional_predictions'])}")
    print(f"  Status: {'✓ PASS' if test3_pass else '✗ FAIL'}")
    print()
    
    # Test 4: Performance scaling
    print("Test 4: Performance Scaling")
    print("-" * 70)
    
    sizes = [100, 500, 1000, 2000]
    times = []
    
    for size in sizes:
        test_seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size))
        
        t_start = time.perf_counter()
        _ = analyzer.analyze_sequence(test_seq)
        t_end = time.perf_counter()
        
        times.append((t_end - t_start) * 1000)
    
    # Check roughly linear scaling (time/size ratio should be similar)
    rate_100 = times[0] / sizes[0]
    rate_2000 = times[-1] / sizes[-1]
    
    # Allow 3x variance in scaling
    test4_pass = rate_2000 / rate_100 < 3.0
    
    results['tests']['scaling'] = {
        'sizes': sizes,
        'times_ms': times,
        'rate_100': rate_100,
        'rate_2000': rate_2000,
        'scaling_ratio': rate_2000 / rate_100,
        'pass': test4_pass,
    }
    results['all_pass'] &= test4_pass
    
    print(f"  Sizes tested: {sizes}")
    print(f"  Times (ms): {[f'{t:.1f}' for t in times]}")
    print(f"  Time/base at 100: {rate_100*1000:.2f} µs/base")
    print(f"  Time/base at 2000: {rate_2000*1000:.2f} µs/base")
    print(f"  Scaling ratio: {rate_2000/rate_100:.2f}x")
    print(f"  Status: {'✓ PASS' if test4_pass else '✗ FAIL'}")
    print()
    
    print("=" * 70)
    if results['all_pass']:
        print("VALIDATION RESULT: ✓ ALL TESTS PASSED")
    else:
        print("VALIDATION RESULT: ✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = run_validation()
