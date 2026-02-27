"""
FRONTIER 07: Real ClinVar Validation
=====================================

Validates genomics pipeline against actual ClinVar data:
- Parses real VCF from NCBI (~2M variants)
- Extracts clinical significance
- Tests variant classification
- Computes statistics on pathogenic/benign variants

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter
import re


@dataclass
class ClinVarVariant:
    """Parsed ClinVar variant."""
    chrom: str
    pos: int
    id: str
    ref: str
    alt: str
    clnsig: str
    clndn: Optional[str] = None
    gene: Optional[str] = None
    mc: Optional[str] = None  # molecular consequence
    
    @property
    def is_pathogenic(self) -> bool:
        return 'pathogenic' in self.clnsig.lower() and 'benign' not in self.clnsig.lower()
    
    @property
    def is_benign(self) -> bool:
        return 'benign' in self.clnsig.lower() and 'pathogenic' not in self.clnsig.lower()
    
    @property
    def is_vus(self) -> bool:
        return 'uncertain' in self.clnsig.lower()
    
    @property
    def variant_type(self) -> str:
        if len(self.ref) == len(self.alt) == 1:
            return 'SNV'
        elif len(self.ref) > len(self.alt):
            return 'deletion'
        elif len(self.ref) < len(self.alt):
            return 'insertion'
        else:
            return 'complex'


@dataclass
class ClinVarStats:
    """Statistics from ClinVar analysis."""
    total_variants: int = 0
    pathogenic: int = 0
    likely_pathogenic: int = 0
    benign: int = 0
    likely_benign: int = 0
    vus: int = 0
    conflicting: int = 0
    other: int = 0
    snv_count: int = 0
    indel_count: int = 0
    genes_affected: int = 0
    chromosomes: Dict[str, int] = field(default_factory=dict)
    top_genes: List[Tuple[str, int]] = field(default_factory=list)
    parse_time_sec: float = 0.0


def parse_clinvar_vcf(vcf_path: str, max_variants: Optional[int] = None) -> Tuple[List[ClinVarVariant], ClinVarStats]:
    """
    Parse ClinVar VCF file.
    
    Args:
        vcf_path: Path to clinvar.vcf.gz
        max_variants: Optional limit for testing
    
    Returns:
        List of variants and statistics
    """
    variants = []
    stats = ClinVarStats()
    gene_counts: Counter = Counter()
    
    t_start = time.perf_counter()
    
    opener = gzip.open if vcf_path.endswith('.gz') else open
    
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            
            chrom, pos, var_id, ref, alt, qual, filt, info = parts[:8]
            
            # Parse INFO field
            info_dict = {}
            for item in info.split(';'):
                if '=' in item:
                    k, v = item.split('=', 1)
                    info_dict[k] = v
            
            clnsig = info_dict.get('CLNSIG', 'not_provided')
            clndn = info_dict.get('CLNDN', None)
            gene = info_dict.get('GENEINFO', '').split(':')[0] if 'GENEINFO' in info_dict else None
            mc = info_dict.get('MC', None)
            
            variant = ClinVarVariant(
                chrom=chrom,
                pos=int(pos),
                id=var_id,
                ref=ref,
                alt=alt,
                clnsig=clnsig,
                clndn=clndn,
                gene=gene,
                mc=mc,
            )
            
            variants.append(variant)
            stats.total_variants += 1
            
            # Update stats
            if gene:
                gene_counts[gene] += 1
            
            clnsig_lower = clnsig.lower()
            if 'pathogenic' in clnsig_lower and 'likely' not in clnsig_lower and 'benign' not in clnsig_lower:
                stats.pathogenic += 1
            elif 'likely_pathogenic' in clnsig_lower:
                stats.likely_pathogenic += 1
            elif 'benign' in clnsig_lower and 'likely' not in clnsig_lower and 'pathogenic' not in clnsig_lower:
                stats.benign += 1
            elif 'likely_benign' in clnsig_lower:
                stats.likely_benign += 1
            elif 'uncertain' in clnsig_lower:
                stats.vus += 1
            elif 'conflicting' in clnsig_lower:
                stats.conflicting += 1
            else:
                stats.other += 1
            
            if variant.variant_type == 'SNV':
                stats.snv_count += 1
            else:
                stats.indel_count += 1
            
            chrom_key = chrom.replace('chr', '')
            stats.chromosomes[chrom_key] = stats.chromosomes.get(chrom_key, 0) + 1
            
            if max_variants and len(variants) >= max_variants:
                break
    
    stats.genes_affected = len(gene_counts)
    stats.top_genes = gene_counts.most_common(20)
    stats.parse_time_sec = time.perf_counter() - t_start
    
    return variants, stats


def analyze_transitions_transversions(variants: List[ClinVarVariant]) -> Dict:
    """Analyze SNV mutation types."""
    transitions = 0
    transversions = 0
    
    transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    
    for v in variants:
        if v.variant_type != 'SNV':
            continue
        
        ref = v.ref.upper()
        alt = v.alt.upper()
        
        if (ref, alt) in transition_pairs:
            transitions += 1
        else:
            transversions += 1
    
    return {
        'transitions': transitions,
        'transversions': transversions,
        'ti_tv_ratio': transitions / transversions if transversions > 0 else 0,
    }


def analyze_by_chromosome(stats: ClinVarStats) -> Dict:
    """Analyze variant distribution by chromosome."""
    # Sort chromosomes numerically/alphabetically
    def chrom_sort_key(c):
        if c.isdigit():
            return (0, int(c))
        elif c == 'X':
            return (1, 0)
        elif c == 'Y':
            return (1, 1)
        elif c == 'MT' or c == 'M':
            return (1, 2)
        else:
            return (2, c)
    
    sorted_chroms = sorted(stats.chromosomes.items(), key=lambda x: chrom_sort_key(x[0]))
    
    return {
        'by_chromosome': dict(sorted_chroms),
        'max_variants_chrom': max(stats.chromosomes.items(), key=lambda x: x[1]),
        'min_variants_chrom': min(stats.chromosomes.items(), key=lambda x: x[1]),
    }


def run_clinvar_validation(vcf_path: str) -> Dict:
    """
    Run full ClinVar validation.
    """
    print("=" * 70)
    print("FRONTIER 07: Real ClinVar Validation")
    print("=" * 70)
    print()
    
    results = {
        'validation': 'CLINVAR_REAL_DATA',
        'all_pass': True,
        'tests': [],
    }
    
    # Parse ClinVar
    print(f"Parsing: {vcf_path}")
    variants, stats = parse_clinvar_vcf(vcf_path)
    
    print(f"  Total variants: {stats.total_variants:,}")
    print(f"  Parse time: {stats.parse_time_sec:.2f}s")
    print(f"  Rate: {stats.total_variants / stats.parse_time_sec:,.0f} variants/sec")
    print()
    
    # Test 1: Variant counts
    print("Test 1: Clinical Significance Distribution")
    print("-" * 70)
    print(f"  Pathogenic:        {stats.pathogenic:>10,}")
    print(f"  Likely pathogenic: {stats.likely_pathogenic:>10,}")
    print(f"  Benign:            {stats.benign:>10,}")
    print(f"  Likely benign:     {stats.likely_benign:>10,}")
    print(f"  VUS:               {stats.vus:>10,}")
    print(f"  Conflicting:       {stats.conflicting:>10,}")
    print(f"  Other:             {stats.other:>10,}")
    
    total_classified = stats.pathogenic + stats.likely_pathogenic + stats.benign + stats.likely_benign
    path_rate = (stats.pathogenic + stats.likely_pathogenic) / total_classified if total_classified > 0 else 0
    
    print(f"\n  Pathogenic rate: {path_rate:.1%} of classified variants")
    
    test1_pass = stats.total_variants > 100_000
    results['tests'].append({
        'name': 'variant_count',
        'pass': test1_pass,
        'value': stats.total_variants,
        'expected': '>100,000',
    })
    print(f"\n  PASS: {test1_pass}")
    print()
    
    # Test 2: Variant types
    print("Test 2: Variant Types")
    print("-" * 70)
    print(f"  SNVs:   {stats.snv_count:>10,} ({stats.snv_count/stats.total_variants:.1%})")
    print(f"  Indels: {stats.indel_count:>10,} ({stats.indel_count/stats.total_variants:.1%})")
    
    titv = analyze_transitions_transversions(variants)
    print(f"\n  Transitions:    {titv['transitions']:>10,}")
    print(f"  Transversions:  {titv['transversions']:>10,}")
    print(f"  Ti/Tv ratio:    {titv['ti_tv_ratio']:>10.2f}")
    
    # Ti/Tv ratio should be ~2.0-2.1 for human exomes
    test2_pass = 1.5 < titv['ti_tv_ratio'] < 3.0
    results['tests'].append({
        'name': 'ti_tv_ratio',
        'pass': test2_pass,
        'value': titv['ti_tv_ratio'],
        'expected': '1.5-3.0',
    })
    print(f"\n  PASS: {test2_pass}")
    print()
    
    # Test 3: Gene coverage
    print("Test 3: Gene Coverage")
    print("-" * 70)
    print(f"  Unique genes: {stats.genes_affected:,}")
    print("\n  Top 10 genes by variant count:")
    for gene, count in stats.top_genes[:10]:
        print(f"    {gene:>12}: {count:>6,}")
    
    test3_pass = stats.genes_affected > 5_000
    results['tests'].append({
        'name': 'gene_coverage',
        'pass': test3_pass,
        'value': stats.genes_affected,
        'expected': '>5,000',
    })
    print(f"\n  PASS: {test3_pass}")
    print()
    
    # Test 4: Chromosome distribution
    print("Test 4: Chromosome Distribution")
    print("-" * 70)
    chrom_analysis = analyze_by_chromosome(stats)
    
    print("  Variants per chromosome:")
    for chrom, count in list(chrom_analysis['by_chromosome'].items())[:10]:
        bar = '█' * int(count / max(stats.chromosomes.values()) * 30)
        print(f"    chr{chrom:>2}: {count:>8,} {bar}")
    print(f"    ... and {len(stats.chromosomes) - 10} more")
    
    test4_pass = len(stats.chromosomes) >= 22
    results['tests'].append({
        'name': 'chromosome_coverage',
        'pass': test4_pass,
        'value': len(stats.chromosomes),
        'expected': '>=22',
    })
    print(f"\n  PASS: {test4_pass}")
    print()
    
    # Test 5: Sample pathogenic variants
    print("Test 5: Sample Pathogenic Variants")
    print("-" * 70)
    
    pathogenic_samples = [v for v in variants if v.is_pathogenic][:5]
    for v in pathogenic_samples:
        print(f"  {v.id}: {v.chrom}:{v.pos} {v.ref}>{v.alt}")
        if v.gene:
            print(f"    Gene: {v.gene}")
        if v.clndn:
            print(f"    Disease: {v.clndn[:60]}...")
    
    test5_pass = len(pathogenic_samples) > 0
    results['tests'].append({
        'name': 'pathogenic_examples',
        'pass': test5_pass,
        'value': len(pathogenic_samples),
    })
    print(f"\n  PASS: {test5_pass}")
    print()
    
    # Summary
    print("=" * 70)
    print("CLINVAR VALIDATION SUMMARY")
    print("=" * 70)
    
    all_pass = all(t['pass'] for t in results['tests'])
    results['all_pass'] = all_pass
    results['stats'] = {
        'total_variants': stats.total_variants,
        'pathogenic': stats.pathogenic,
        'likely_pathogenic': stats.likely_pathogenic,
        'benign': stats.benign,
        'likely_benign': stats.likely_benign,
        'vus': stats.vus,
        'snv_count': stats.snv_count,
        'indel_count': stats.indel_count,
        'genes': stats.genes_affected,
        'ti_tv_ratio': titv['ti_tv_ratio'],
        'parse_time_sec': stats.parse_time_sec,
    }
    
    print(f"  Tests passed: {sum(1 for t in results['tests'] if t['pass'])}/{len(results['tests'])}")
    print(f"  Status: {'VALIDATED' if all_pass else 'FAILED'}")
    print()
    
    return results


def generate_attestation(results: Dict) -> Dict:
    """Generate attestation for ClinVar validation."""
    return {
        'attestation': {
            'type': 'FRONTIER_07_CLINVAR_VALIDATION',
            'version': '1.0.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'VALIDATED' if results.get('all_pass') else 'FAILED',
        },
        'data_source': {
            'name': 'ClinVar',
            'url': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/',
            'build': 'GRCh38',
        },
        'statistics': results.get('stats', {}),
        'tests': results.get('tests', []),
    }


if __name__ == '__main__':
    import sys
    
    vcf_path = sys.argv[1] if len(sys.argv) > 1 else 'clinvar.vcf.gz'
    
    results = run_clinvar_validation(vcf_path)
    
    attestation = generate_attestation(results)
    
    with open('CLINVAR_VALIDATION_ATTESTATION.json', 'w') as f:
        json.dump(attestation, f, indent=2, default=str)
    
    print(f"Attestation saved: CLINVAR_VALIDATION_ATTESTATION.json")
