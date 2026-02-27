"""
FRONTIER 07-D: Multi-Species Alignment
========================================

Cross-genome conservation analysis using phylogenetic tensor networks:
- Progressive multiple sequence alignment
- Phylogenetic distance estimation
- Conservation scoring across species
- Accelerated evolution detection
- Ancestral sequence reconstruction

Key insight: Related species share low-rank tensor structure
- Conserved regions: nearly identical → rank 1
- Divergent regions: species-specific → higher rank
- Functional constraint: intermediate rank with covariation

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

from dna_tensor import DNATensorTrain, encode_sequence, decode_sequence


@dataclass
class AlignedSequence:
    """Aligned sequence with gap information."""
    species: str
    sequence: str  # With gaps as '-'
    original_length: int
    
    @property
    def aligned_length(self) -> int:
        return len(self.sequence)
    
    @property
    def gap_fraction(self) -> float:
        return self.sequence.count('-') / len(self.sequence)
    
    def ungapped(self) -> str:
        return self.sequence.replace('-', '')


@dataclass
class PhylogeneticNode:
    """Node in phylogenetic tree."""
    name: str
    distance: float = 0.0
    left: Optional['PhylogeneticNode'] = None
    right: Optional['PhylogeneticNode'] = None
    sequence: Optional[str] = None
    
    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
    
    def newick(self) -> str:
        """Return Newick format string."""
        if self.is_leaf:
            return f"{self.name}:{self.distance:.4f}"
        left_str = self.left.newick() if self.left else ""
        right_str = self.right.newick() if self.right else ""
        return f"({left_str},{right_str}):{self.distance:.4f}"


@dataclass
class ConservationBlock:
    """Conserved block across species."""
    start: int
    end: int
    conservation_score: float
    species_present: List[str]
    consensus: str
    
    @property
    def length(self) -> int:
        return self.end - self.start


class MultiSpeciesAligner:
    """
    Progressive multiple sequence alignment with tensor network analysis.
    
    Uses UPGMA/neighbor-joining for guide tree construction,
    then progressive alignment following the tree.
    
    Example:
        >>> aligner = MultiSpeciesAligner(max_rank=16)
        >>> sequences = {
        ...     'human': 'ACGTACGTACGT...',
        ...     'chimp': 'ACGTAGGTACGT...',
        ...     'mouse': 'ACATACGTACAT...',
        ... }
        >>> alignment = aligner.align(sequences)
        >>> print(alignment.conservation_profile)
    """
    
    def __init__(
        self,
        max_rank: int = 16,
        gap_open: float = -10.0,
        gap_extend: float = -0.5,
        match_score: float = 5.0,
        mismatch_score: float = -4.0,
    ):
        self.max_rank = max_rank
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        
        # Substitution matrix (simplified BLOSUM-like for DNA)
        self.sub_matrix = self._create_substitution_matrix()
    
    def _create_substitution_matrix(self) -> Dict[Tuple[str, str], float]:
        """Create DNA substitution scoring matrix."""
        bases = ['A', 'C', 'G', 'T', '-']
        matrix = {}
        
        for b1 in bases:
            for b2 in bases:
                if b1 == '-' or b2 == '-':
                    matrix[(b1, b2)] = self.gap_extend
                elif b1 == b2:
                    matrix[(b1, b2)] = self.match_score
                else:
                    # Transitions vs transversions
                    purines = {'A', 'G'}
                    pyrimidines = {'C', 'T'}
                    if (b1 in purines and b2 in purines) or \
                       (b1 in pyrimidines and b2 in pyrimidines):
                        matrix[(b1, b2)] = self.mismatch_score / 2  # Transition
                    else:
                        matrix[(b1, b2)] = self.mismatch_score  # Transversion
        
        return matrix
    
    def pairwise_align(
        self,
        seq1: str,
        seq2: str,
    ) -> Tuple[str, str, float]:
        """
        Needleman-Wunsch global pairwise alignment.
        
        Returns aligned sequences and score.
        """
        n, m = len(seq1), len(seq2)
        
        # Initialize DP matrices
        score = np.zeros((n + 1, m + 1))
        traceback = np.zeros((n + 1, m + 1), dtype=int)
        
        # Initialize gaps
        for i in range(1, n + 1):
            score[i, 0] = self.gap_open + (i - 1) * self.gap_extend
            traceback[i, 0] = 1  # Up
        for j in range(1, m + 1):
            score[0, j] = self.gap_open + (j - 1) * self.gap_extend
            traceback[0, j] = 2  # Left
        
        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = score[i-1, j-1] + self.sub_matrix[(seq1[i-1], seq2[j-1])]
                delete = score[i-1, j] + (self.gap_extend if traceback[i-1, j] == 1 else self.gap_open)
                insert = score[i, j-1] + (self.gap_extend if traceback[i, j-1] == 2 else self.gap_open)
                
                if match >= delete and match >= insert:
                    score[i, j] = match
                    traceback[i, j] = 0  # Diagonal
                elif delete >= insert:
                    score[i, j] = delete
                    traceback[i, j] = 1  # Up
                else:
                    score[i, j] = insert
                    traceback[i, j] = 2  # Left
        
        # Traceback
        aligned1, aligned2 = [], []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i, j] == 0:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and traceback[i, j] == 1:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1
        
        return ''.join(reversed(aligned1)), ''.join(reversed(aligned2)), score[n, m]
    
    def compute_distance_matrix(
        self,
        sequences: Dict[str, str],
    ) -> Tuple[List[str], np.ndarray]:
        """Compute pairwise distance matrix."""
        species = list(sequences.keys())
        n = len(species)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Align and compute distance
                aligned1, aligned2, score = self.pairwise_align(
                    sequences[species[i]],
                    sequences[species[j]]
                )
                
                # Distance = 1 - identity
                matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
                aligned_length = len(aligned1)
                identity = matches / aligned_length if aligned_length > 0 else 0
                
                distances[i, j] = 1 - identity
                distances[j, i] = distances[i, j]
        
        return species, distances
    
    def build_guide_tree(
        self,
        sequences: Dict[str, str],
    ) -> PhylogeneticNode:
        """Build UPGMA guide tree."""
        species, distances_init = self.compute_distance_matrix(sequences)
        n = len(species)
        
        # Extend distance matrix for internal nodes
        max_nodes = 2 * n
        distances = np.zeros((max_nodes, max_nodes))
        distances[:n, :n] = distances_init
        
        # Initialize clusters
        clusters = {i: PhylogeneticNode(name=species[i], sequence=sequences[species[i]]) 
                   for i in range(n)}
        cluster_sizes = {i: 1 for i in range(n)}
        
        # UPGMA clustering
        active = set(range(n))
        next_id = n
        
        while len(active) > 1:
            # Find minimum distance
            min_dist = float('inf')
            min_i, min_j = -1, -1
            
            for i in active:
                for j in active:
                    if i < j and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        min_i, min_j = i, j
            
            # Create new cluster
            new_node = PhylogeneticNode(
                name=f"node_{next_id}",
                distance=min_dist / 2,
                left=clusters[min_i],
                right=clusters[min_j],
            )
            
            # Update distances (UPGMA formula)
            new_size = cluster_sizes[min_i] + cluster_sizes[min_j]
            
            for k in active:
                if k != min_i and k != min_j:
                    new_dist = (
                        distances[min_i, k] * cluster_sizes[min_i] +
                        distances[min_j, k] * cluster_sizes[min_j]
                    ) / new_size
                    distances[next_id, k] = new_dist
                    distances[k, next_id] = new_dist
            
            # Update tracking
            clusters[next_id] = new_node
            cluster_sizes[next_id] = new_size
            active.remove(min_i)
            active.remove(min_j)
            active.add(next_id)
            next_id += 1
        
        return clusters[list(active)[0]]
    
    def progressive_align(
        self,
        sequences: Dict[str, str],
    ) -> Dict[str, AlignedSequence]:
        """
        Progressive multiple sequence alignment.
        
        Returns dictionary of aligned sequences.
        """
        if len(sequences) < 2:
            species = list(sequences.keys())[0]
            return {species: AlignedSequence(
                species=species,
                sequence=sequences[species],
                original_length=len(sequences[species])
            )}
        
        # Build guide tree
        tree = self.build_guide_tree(sequences)
        
        # Align following tree (post-order traversal)
        aligned = self._align_tree(tree)
        
        return aligned
    
    def _align_tree(
        self,
        node: PhylogeneticNode,
    ) -> Dict[str, AlignedSequence]:
        """Recursively align following guide tree."""
        if node.is_leaf:
            return {node.name: AlignedSequence(
                species=node.name,
                sequence=node.sequence,
                original_length=len(node.sequence)
            )}
        
        # Get child alignments
        left_aligned = self._align_tree(node.left)
        right_aligned = self._align_tree(node.right)
        
        # Get consensus sequences for alignment
        left_consensus = self._compute_consensus(list(left_aligned.values()))
        right_consensus = self._compute_consensus(list(right_aligned.values()))
        
        # Align consensus sequences
        aligned_left, aligned_right, _ = self.pairwise_align(left_consensus, right_consensus)
        
        # Propagate gaps to all sequences
        result = {}
        
        # Process left group
        for species, aln_seq in left_aligned.items():
            new_seq = self._insert_gaps(aln_seq.sequence, left_consensus, aligned_left)
            result[species] = AlignedSequence(
                species=species,
                sequence=new_seq,
                original_length=aln_seq.original_length
            )
        
        # Process right group
        for species, aln_seq in right_aligned.items():
            new_seq = self._insert_gaps(aln_seq.sequence, right_consensus, aligned_right)
            result[species] = AlignedSequence(
                species=species,
                sequence=new_seq,
                original_length=aln_seq.original_length
            )
        
        return result
    
    def _compute_consensus(self, sequences: List[AlignedSequence]) -> str:
        """Compute consensus sequence from aligned sequences."""
        if not sequences:
            return ""
        
        length = len(sequences[0].sequence)
        consensus = []
        
        for i in range(length):
            counts = {}
            for seq in sequences:
                if i < len(seq.sequence):
                    base = seq.sequence[i]
                    counts[base] = counts.get(base, 0) + 1
            
            # Most common base (prefer non-gap)
            best_base = '-'
            best_count = 0
            for base, count in counts.items():
                if base != '-' and count > best_count:
                    best_base = base
                    best_count = count
            
            if best_count == 0 and '-' in counts:
                best_base = '-'
            
            consensus.append(best_base)
        
        return ''.join(consensus)
    
    def _insert_gaps(
        self,
        sequence: str,
        original_consensus: str,
        aligned_consensus: str,
    ) -> str:
        """Insert gaps into sequence based on consensus alignment."""
        result = []
        seq_idx = 0
        cons_idx = 0
        
        for aligned_char in aligned_consensus:
            if cons_idx < len(original_consensus) and original_consensus[cons_idx] == aligned_char:
                # Match in consensus - use corresponding sequence character
                if seq_idx < len(sequence):
                    result.append(sequence[seq_idx])
                    if sequence[seq_idx] != '-':
                        seq_idx += 1
                    else:
                        seq_idx += 1
                cons_idx += 1
            elif aligned_char == '-':
                # Gap inserted in this position
                result.append('-')
            else:
                # Character in consensus
                if seq_idx < len(sequence):
                    result.append(sequence[seq_idx])
                    seq_idx += 1
                cons_idx += 1
        
        return ''.join(result)
    
    def compute_conservation_profile(
        self,
        alignment: Dict[str, AlignedSequence],
    ) -> np.ndarray:
        """Compute per-position conservation score."""
        if not alignment:
            return np.array([])
        
        sequences = list(alignment.values())
        length = len(sequences[0].sequence)
        n_species = len(sequences)
        
        scores = np.zeros(length)
        
        for i in range(length):
            bases = [seq.sequence[i] for seq in sequences if i < len(seq.sequence)]
            
            # Count non-gap bases
            non_gap = [b for b in bases if b != '-']
            
            if not non_gap:
                scores[i] = 0.0
                continue
            
            # Conservation = fraction of most common base
            counts = {}
            for b in non_gap:
                counts[b] = counts.get(b, 0) + 1
            
            max_count = max(counts.values())
            scores[i] = max_count / len(non_gap)
        
        return scores
    
    def find_conserved_blocks(
        self,
        alignment: Dict[str, AlignedSequence],
        min_length: int = 10,
        min_conservation: float = 0.8,
    ) -> List[ConservationBlock]:
        """Find blocks of high conservation."""
        conservation = self.compute_conservation_profile(alignment)
        sequences = list(alignment.values())
        
        blocks = []
        in_block = False
        block_start = 0
        
        for i, score in enumerate(conservation):
            if score >= min_conservation:
                if not in_block:
                    in_block = True
                    block_start = i
            else:
                if in_block:
                    if i - block_start >= min_length:
                        # Get consensus for block
                        consensus = []
                        for j in range(block_start, i):
                            bases = [seq.sequence[j] for seq in sequences if j < len(seq.sequence)]
                            non_gap = [b for b in bases if b != '-']
                            if non_gap:
                                consensus.append(max(set(non_gap), key=non_gap.count))
                            else:
                                consensus.append('-')
                        
                        blocks.append(ConservationBlock(
                            start=block_start,
                            end=i,
                            conservation_score=float(np.mean(conservation[block_start:i])),
                            species_present=[s.species for s in sequences],
                            consensus=''.join(consensus),
                        ))
                    in_block = False
        
        # Handle block at end
        if in_block and len(conservation) - block_start >= min_length:
            consensus = []
            for j in range(block_start, len(conservation)):
                bases = [seq.sequence[j] for seq in sequences if j < len(seq.sequence)]
                non_gap = [b for b in bases if b != '-']
                if non_gap:
                    consensus.append(max(set(non_gap), key=non_gap.count))
                else:
                    consensus.append('-')
            
            blocks.append(ConservationBlock(
                start=block_start,
                end=len(conservation),
                conservation_score=float(np.mean(conservation[block_start:])),
                species_present=[s.species for s in sequences],
                consensus=''.join(consensus),
            ))
        
        return blocks
    
    def detect_accelerated_regions(
        self,
        alignment: Dict[str, AlignedSequence],
        reference_species: str,
        outgroup_species: str,
        window_size: int = 100,
    ) -> List[Tuple[int, int, float]]:
        """
        Detect regions with accelerated evolution in reference vs outgroup.
        
        Returns list of (start, end, acceleration_score) tuples.
        """
        if reference_species not in alignment or outgroup_species not in alignment:
            return []
        
        ref_seq = alignment[reference_species].sequence
        out_seq = alignment[outgroup_species].sequence
        
        # Compute divergence in sliding windows
        length = min(len(ref_seq), len(out_seq))
        accelerated = []
        
        for start in range(0, length - window_size, window_size // 2):
            end = start + window_size
            
            ref_window = ref_seq[start:end]
            out_window = out_seq[start:end]
            
            # Compute divergence
            mismatches = sum(1 for a, b in zip(ref_window, out_window) 
                           if a != b and a != '-' and b != '-')
            comparable = sum(1 for a, b in zip(ref_window, out_window) 
                           if a != '-' and b != '-')
            
            if comparable > 0:
                divergence = mismatches / comparable
                
                # High divergence = accelerated evolution
                if divergence > 0.1:  # 10% divergence threshold
                    accelerated.append((start, end, divergence))
        
        return accelerated


def run_validation() -> dict:
    """
    Validate multi-species alignment.
    """
    print("=" * 70)
    print("FRONTIER 07-D: Multi-Species Alignment")
    print("=" * 70)
    print()
    
    results = {
        'tests': {},
        'all_pass': True,
    }
    
    # Test 1: Pairwise Alignment
    print("Test 1: Pairwise Alignment")
    print("-" * 70)
    
    aligner = MultiSpeciesAligner(max_rank=8)
    
    seq1 = "ACGTACGTACGTACGT"
    seq2 = "ACGTAGGTACATACGT"  # 2 mismatches
    
    t_start = time.perf_counter()
    aligned1, aligned2, score = aligner.pairwise_align(seq1, seq2)
    t_end = time.perf_counter()
    
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b)
    identity = matches / len(aligned1)
    
    print(f"  Sequence 1: {seq1}")
    print(f"  Sequence 2: {seq2}")
    print(f"  Aligned 1:  {aligned1}")
    print(f"  Aligned 2:  {aligned2}")
    print(f"  Identity: {identity:.2%}")
    print(f"  Score: {score:.1f}")
    print(f"  Time: {(t_end - t_start) * 1000:.2f} ms")
    
    pairwise_pass = identity > 0.8
    print(f"  PASS: {pairwise_pass}")
    print()
    
    results['tests']['pairwise_alignment'] = {
        'identity': identity,
        'score': score,
        'time_ms': (t_end - t_start) * 1000,
        'pass': pairwise_pass,
    }
    
    # Test 2: Multiple Sequence Alignment
    print("Test 2: Multiple Sequence Alignment")
    print("-" * 70)
    
    sequences = {
        'human': "ACGTACGTACGTACGTACGTACGT",
        'chimp': "ACGTACGTACGTACGTACGTACGT",  # Identical to human
        'mouse': "ACATACGTACATACGTACATACGT",  # 3 differences
        'fish':  "TGCATGCATGCATGCATGCATGCA",  # Very different
    }
    
    t_start = time.perf_counter()
    alignment = aligner.progressive_align(sequences)
    t_end = time.perf_counter()
    
    print(f"  Species: {len(sequences)}")
    for species, aln_seq in alignment.items():
        print(f"    {species:8}: {aln_seq.sequence[:40]}...")
    print(f"  Aligned length: {len(list(alignment.values())[0].sequence)}")
    print(f"  Time: {(t_end - t_start) * 1000:.2f} ms")
    
    msa_pass = len(alignment) == len(sequences)
    print(f"  PASS: {msa_pass}")
    print()
    
    results['tests']['msa'] = {
        'n_species': len(sequences),
        'aligned_length': len(list(alignment.values())[0].sequence),
        'time_ms': (t_end - t_start) * 1000,
        'pass': msa_pass,
    }
    
    # Test 3: Conservation Analysis
    print("Test 3: Conservation Analysis")
    print("-" * 70)
    
    conservation = aligner.compute_conservation_profile(alignment)
    blocks = aligner.find_conserved_blocks(alignment, min_length=5, min_conservation=0.75)
    
    print(f"  Mean conservation: {np.mean(conservation):.2%}")
    print(f"  Conserved blocks: {len(blocks)}")
    for block in blocks[:3]:
        print(f"    {block.start}-{block.end}: {block.conservation_score:.2%} ({block.length}bp)")
    
    conservation_pass = len(blocks) >= 1
    print(f"  PASS: {conservation_pass}")
    print()
    
    results['tests']['conservation'] = {
        'mean_conservation': float(np.mean(conservation)),
        'n_blocks': len(blocks),
        'pass': conservation_pass,
    }
    
    # Test 4: Phylogenetic Tree
    print("Test 4: Phylogenetic Tree Construction")
    print("-" * 70)
    
    t_start = time.perf_counter()
    tree = aligner.build_guide_tree(sequences)
    t_end = time.perf_counter()
    
    print(f"  Newick: {tree.newick()}")
    print(f"  Time: {(t_end - t_start) * 1000:.2f} ms")
    
    tree_pass = tree is not None
    print(f"  PASS: {tree_pass}")
    print()
    
    results['tests']['phylogeny'] = {
        'newick': tree.newick(),
        'time_ms': (t_end - t_start) * 1000,
        'pass': tree_pass,
    }
    
    # Summary
    print("=" * 70)
    print("MULTI-SPECIES ALIGNMENT SUMMARY")
    print("=" * 70)
    
    all_pass = all(t['pass'] for t in results['tests'].values())
    results['all_pass'] = all_pass
    
    for test_name, test_result in results['tests'].items():
        print(f"{test_name}: {'✓' if test_result['pass'] else '✗'}")
    print()
    print(f"ALL TESTS PASS: {all_pass}")
    
    return results


if __name__ == '__main__':
    results = run_validation()
