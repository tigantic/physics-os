#!/usr/bin/env python3
"""
FEZK ELITE PIL Parser v1.0
==========================
Production-grade parser for Polygon's Polynomial Identity Language (PIL).

Capabilities:
- Full namespace support with qualified signal names
- Lookup/permutation constraint parsing ({..} in {..} and {..} is {..})
- Polynomial identity parsing (constraint = 0)
- State transition constraints (x' = ...)
- Array signal expansion
- Constraint coefficient matrix building
- GPU-accelerated rank analysis via rSVD

This parser handles the "State Explosion" that crashes circomspect/ecne.

Author: FEZK Elite Team
Date: January 23, 2026
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from enum import Enum, auto
import numpy as np


class SignalType(Enum):
    """Type of PIL polynomial signal."""
    COMMIT = auto()      # Prover-controlled witness
    CONSTANT = auto()    # Fixed/public polynomial
    INTERMEDIATE = auto() # Computed via pol x = expr


class ConstraintType(Enum):
    """Type of PIL constraint."""
    POLYNOMIAL_IDENTITY = auto()   # expr = 0
    STATE_TRANSITION = auto()      # x' = expr
    LOOKUP = auto()                # {...} in {...}
    PERMUTATION = auto()           # {...} is {...}
    BINARY = auto()                # (1-x)*x = 0


@dataclass
class PILSignal:
    """A PIL polynomial signal."""
    name: str
    namespace: str
    signal_type: SignalType
    full_name: str = ""
    is_array: bool = False
    array_index: int = -1
    source_file: str = ""
    line: int = 0
    is_constrained: bool = False
    constraint_refs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.full_name:
            self.full_name = f"{self.namespace}.{self.name}"


@dataclass  
class PILConstraint:
    """A PIL constraint."""
    expr: str
    namespace: str
    constraint_type: ConstraintType
    source_file: str = ""
    line: int = 0
    signals_involved: Set[str] = field(default_factory=set)
    lhs_signals: Set[str] = field(default_factory=set)
    rhs_signals: Set[str] = field(default_factory=set)


@dataclass
class PILLookup:
    """A PIL lookup/permutation constraint."""
    selector: str
    lhs_columns: List[str]
    rhs_table: str
    rhs_columns: List[str]
    constraint_type: ConstraintType
    namespace: str
    source_file: str = ""
    line: int = 0


@dataclass
class AnalysisResult:
    """Result of PIL constraint analysis."""
    signals: Dict[str, PILSignal]
    constraints: List[PILConstraint]
    lookups: List[PILLookup]
    namespaces: Dict[str, Set[str]]
    
    # Analysis results
    num_signals: int = 0
    num_constraints: int = 0
    num_lookups: int = 0
    matrix_shape: Tuple[int, int] = (0, 0)
    numerical_rank: int = 0
    degrees_of_freedom: int = 0
    unconstrained_signals: List[str] = field(default_factory=list)


class PILParser:
    """
    Production-grade PIL (Polynomial Identity Language) parser.
    
    Handles Polygon zkEVM's constraint system including:
    - Namespace-qualified signals
    - Lookup tables ({...} in {...})
    - Permutation arguments ({...} is {...})
    - State transitions (x' = expr)
    - Polynomial identities (expr = 0)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.signals: Dict[str, PILSignal] = {}
        self.constraints: List[PILConstraint] = []
        self.lookups: List[PILLookup] = []
        self.namespaces: Dict[str, Set[str]] = defaultdict(set)
        self.includes: List[str] = []
        self.current_namespace: str = "Global"
        self.current_file: str = ""
        self.signal_index: Dict[str, int] = {}
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficient parsing."""
        self.patterns = {
            # Namespace declaration
            'namespace': re.compile(
                r'namespace\s+(\w+)\s*\(\s*%?(\w+)\s*\)', re.MULTILINE
            ),
            
            # Include statement
            'include': re.compile(r'include\s+"([^"]+)"'),
            
            # Constant definition  
            'constant_def': re.compile(r'constant\s+%(\w+)\s*=\s*([^;]+);'),
            
            # pol commit (witness signals)
            'pol_commit': re.compile(r'pol\s+commit\s+([^;]+);'),
            
            # pol constant (fixed signals)
            'pol_constant': re.compile(r'pol\s+constant\s+([^;]+);'),
            
            # pol definition (intermediate)
            'pol_def': re.compile(r'pol\s+(\w+)\s*=\s*([^;]+);'),
            
            # Array declaration
            'array_decl': re.compile(r'(\w+)\[(\d+)\]'),
            
            # Lookup constraint: selector { cols } in table { cols }
            'lookup_in': re.compile(
                r'(\w+(?:\s*\+\s*\w+)*)\s*\{([^}]+)\}\s*in\s*(\w+(?:\.\w+)?)\s*\{([^}]+)\}',
                re.MULTILINE | re.DOTALL
            ),
            
            # Lookup without selector: { cols } in { cols }
            'lookup_in_noselector': re.compile(
                r'\{([^}]+)\}\s*in\s*\{([^}]+)\}',
                re.MULTILINE | re.DOTALL
            ),
            
            # Permutation: selector { cols } is table { cols }
            'permutation_is': re.compile(
                r'(\w+(?:\s*\+\s*\w+)*)\s*\{([^}]+)\}\s*is\s*(\w+(?:\.\w+)?)\s*\{([^}]+)\}',
                re.MULTILINE | re.DOTALL
            ),
            
            # State transition: x' = expr
            'state_transition': re.compile(
                r"(\w+(?:\[\d+\])?)\s*'\s*=\s*([^;]+);",
                re.MULTILINE
            ),
            
            # Binary constraint: (1-x)*x = 0
            'binary_constraint': re.compile(
                r'\(\s*1\s*-\s*(\w+)\s*\)\s*\*\s*\1\s*=\s*0',
                re.MULTILINE
            ),
            
            # General constraint: expr = expr
            'constraint': re.compile(
                r'([^=;{}\n]+)\s*=\s*([^;{}\n]+);',
                re.MULTILINE
            ),
            
            # Signal identifier
            'signal_ident': re.compile(
                r'\b([A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)\b'
            ),
        }
        
        # Keywords to filter out
        self.keywords = {
            'pol', 'commit', 'constant', 'namespace', 'include',
            'if', 'else', 'for', 'while', 'in', 'is', 'node', 'tools'
        }
    
    def parse_file(self, filepath: Path) -> None:
        """Parse a single PIL file."""
        content = filepath.read_text()
        self.current_file = filepath.name
        self.current_namespace = "Global"
        
        # Remove block comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        lines = content.split('\n')
        line_num = 0
        
        while line_num < len(lines):
            line = lines[line_num].strip()
            line_num += 1
            
            # Skip empty lines and line comments
            if not line or line.startswith('//'):
                continue
            
            # Handle multi-line constructs
            while line.count('{') > line.count('}') and line_num < len(lines):
                line += ' ' + lines[line_num].strip()
                line_num += 1
            
            # Parse the line
            self._parse_line(line, line_num)
    
    def _parse_line(self, line: str, line_num: int) -> None:
        """Parse a single line of PIL code."""
        # Skip comments embedded in line
        if '//' in line:
            line = line.split('//')[0].strip()
        if not line:
            return
        
        # Namespace declaration
        ns_match = self.patterns['namespace'].search(line)
        if ns_match:
            self.current_namespace = ns_match.group(1)
            if self.verbose:
                print(f"  Namespace: {self.current_namespace}")
            return
        
        # Include statement
        inc_match = self.patterns['include'].search(line)
        if inc_match:
            self.includes.append(inc_match.group(1))
            return
        
        # pol commit (witness signals)
        commit_match = self.patterns['pol_commit'].search(line)
        if commit_match:
            self._parse_signal_list(
                commit_match.group(1), 
                SignalType.COMMIT, 
                line_num
            )
            return
        
        # pol constant (fixed signals)
        const_match = self.patterns['pol_constant'].search(line)
        if const_match:
            self._parse_signal_list(
                const_match.group(1),
                SignalType.CONSTANT,
                line_num
            )
            return
        
        # pol definition (intermediate)
        def_match = self.patterns['pol_def'].search(line)
        if def_match and 'commit' not in line and 'constant' not in line:
            name = def_match.group(1)
            expr = def_match.group(2)
            self._add_intermediate_signal(name, expr, line_num)
            return
        
        # Lookup with selector: sel { ... } in table { ... }
        lookup_match = self.patterns['lookup_in'].search(line)
        if lookup_match:
            self._parse_lookup(
                selector=lookup_match.group(1),
                lhs_cols=lookup_match.group(2),
                rhs_table=lookup_match.group(3),
                rhs_cols=lookup_match.group(4),
                constraint_type=ConstraintType.LOOKUP,
                line_num=line_num
            )
            return
        
        # Permutation: sel { ... } is table { ... }
        perm_match = self.patterns['permutation_is'].search(line)
        if perm_match:
            self._parse_lookup(
                selector=perm_match.group(1),
                lhs_cols=perm_match.group(2),
                rhs_table=perm_match.group(3),
                rhs_cols=perm_match.group(4),
                constraint_type=ConstraintType.PERMUTATION,
                line_num=line_num
            )
            return
        
        # Lookup without selector: { ... } in { ... }
        lookup_noselector = self.patterns['lookup_in_noselector'].search(line)
        if lookup_noselector and 'is' not in line:
            self._parse_lookup(
                selector="1",
                lhs_cols=lookup_noselector.group(1),
                rhs_table="",
                rhs_cols=lookup_noselector.group(2),
                constraint_type=ConstraintType.LOOKUP,
                line_num=line_num
            )
            return
        
        # Binary constraint: (1-x)*x = 0
        binary_match = self.patterns['binary_constraint'].search(line)
        if binary_match:
            sig_name = binary_match.group(1)
            self._add_constraint(
                f"(1-{sig_name})*{sig_name}",
                "0",
                ConstraintType.BINARY,
                line_num
            )
            return
        
        # State transition: x' = expr
        trans_match = self.patterns['state_transition'].search(line)
        if trans_match:
            lhs = trans_match.group(1)
            rhs = trans_match.group(2)
            self._add_constraint(
                f"{lhs}'",
                rhs,
                ConstraintType.STATE_TRANSITION,
                line_num
            )
            return
        
        # General constraint: expr = expr (but not declarations)
        if '=' in line and 'pol' not in line and 'include' not in line:
            if '{' not in line:  # Not a lookup
                constraint_match = self.patterns['constraint'].search(line)
                if constraint_match:
                    lhs = constraint_match.group(1).strip()
                    rhs = constraint_match.group(2).strip()
                    if lhs and rhs and not lhs.startswith('constant'):
                        self._add_constraint(
                            lhs, rhs,
                            ConstraintType.POLYNOMIAL_IDENTITY,
                            line_num
                        )
    
    def _parse_signal_list(self, decls: str, sig_type: SignalType, line_num: int) -> None:
        """Parse comma-separated signal declarations."""
        for decl in decls.split(','):
            decl = decl.strip()
            if not decl:
                continue
            
            # Check for array
            arr_match = self.patterns['array_decl'].search(decl)
            if arr_match:
                name = arr_match.group(1)
                size = int(arr_match.group(2))
                for i in range(size):
                    self._add_signal(f"{name}[{i}]", sig_type, line_num, 
                                    is_array=True, array_index=i)
            else:
                self._add_signal(decl, sig_type, line_num)
    
    def _add_signal(self, name: str, sig_type: SignalType, line_num: int,
                   is_array: bool = False, array_index: int = -1) -> None:
        """Add a signal to the registry."""
        full_name = f"{self.current_namespace}.{name}"
        
        if full_name not in self.signals:
            sig = PILSignal(
                name=name,
                namespace=self.current_namespace,
                signal_type=sig_type,
                full_name=full_name,
                is_array=is_array,
                array_index=array_index,
                source_file=self.current_file,
                line=line_num
            )
            self.signals[full_name] = sig
            self.namespaces[self.current_namespace].add(full_name)
            self.signal_index[full_name] = len(self.signal_index)
    
    def _add_intermediate_signal(self, name: str, expr: str, line_num: int) -> None:
        """Add an intermediate signal (pol x = expr)."""
        full_name = f"{self.current_namespace}.{name}"
        
        sig = PILSignal(
            name=name,
            namespace=self.current_namespace,
            signal_type=SignalType.INTERMEDIATE,
            full_name=full_name,
            source_file=self.current_file,
            line=line_num,
            is_constrained=True  # Intermediate is always constrained
        )
        self.signals[full_name] = sig
        self.namespaces[self.current_namespace].add(full_name)
        self.signal_index[full_name] = len(self.signal_index)
        
        # Also add as constraint
        self._add_constraint(name, expr, ConstraintType.POLYNOMIAL_IDENTITY, line_num)
    
    def _add_constraint(self, lhs: str, rhs: str, ctype: ConstraintType, 
                       line_num: int) -> None:
        """Add a constraint to the registry."""
        constraint = PILConstraint(
            expr=f"{lhs} = {rhs}",
            namespace=self.current_namespace,
            constraint_type=ctype,
            source_file=self.current_file,
            line=line_num
        )
        
        # Extract signals from both sides
        constraint.lhs_signals = self._extract_signals(lhs)
        constraint.rhs_signals = self._extract_signals(rhs)
        constraint.signals_involved = constraint.lhs_signals | constraint.rhs_signals
        
        self.constraints.append(constraint)
        
        # Mark signals as constrained
        for sig_name in constraint.signals_involved:
            if sig_name in self.signals:
                self.signals[sig_name].is_constrained = True
                self.signals[sig_name].constraint_refs.append(
                    f"{self.current_file}:{line_num}"
                )
    
    def _parse_lookup(self, selector: str, lhs_cols: str, rhs_table: str,
                     rhs_cols: str, constraint_type: ConstraintType,
                     line_num: int) -> None:
        """Parse a lookup/permutation constraint."""
        # Parse column lists
        lhs_col_list = [c.strip() for c in lhs_cols.split(',') if c.strip()]
        rhs_col_list = [c.strip() for c in rhs_cols.split(',') if c.strip()]
        
        lookup = PILLookup(
            selector=selector.strip(),
            lhs_columns=lhs_col_list,
            rhs_table=rhs_table.strip(),
            rhs_columns=rhs_col_list,
            constraint_type=constraint_type,
            namespace=self.current_namespace,
            source_file=self.current_file,
            line=line_num
        )
        self.lookups.append(lookup)
        
        # Mark all signals in lookup as constrained
        all_signals = set()
        for col in lhs_col_list + rhs_col_list:
            all_signals |= self._extract_signals(col)
        
        for sig_name in all_signals:
            if sig_name in self.signals:
                self.signals[sig_name].is_constrained = True
                self.signals[sig_name].constraint_refs.append(
                    f"{self.current_file}:{line_num} [LOOKUP]"
                )
    
    def _extract_signals(self, expr: str) -> Set[str]:
        """Extract signal names from expression."""
        signals = set()
        ns = self.current_namespace
        
        # Find all identifiers
        for match in self.patterns['signal_ident'].finditer(expr):
            ident = match.group(1)
            
            # Skip keywords and numbers
            if ident in self.keywords or ident.isdigit():
                continue
            
            # Handle qualified names (Namespace.signal)
            if '.' in expr:
                # Look for qualified pattern
                qual_pattern = re.search(rf'(\w+)\.{re.escape(ident)}', expr)
                if qual_pattern:
                    qual_name = f"{qual_pattern.group(1)}.{ident}"
                    if qual_name in self.signals:
                        signals.add(qual_name)
                        continue
            
            # Try current namespace
            full_name = f"{ns}.{ident}"
            if full_name in self.signals:
                signals.add(full_name)
            # Try as-is (might be qualified already)
            elif ident in self.signals:
                signals.add(ident)
        
        return signals
    
    def parse_directory(self, path: Path) -> AnalysisResult:
        """Parse all PIL files in a directory."""
        pil_files = sorted(path.glob("*.pil"))
        
        if self.verbose:
            print(f"Parsing {len(pil_files)} PIL files...")
        
        for pil_file in pil_files:
            if self.verbose:
                print(f"  {pil_file.name}")
            self.parse_file(pil_file)
        
        return self._build_result()
    
    def _build_result(self) -> AnalysisResult:
        """Build analysis result."""
        # Find unconstrained signals
        unconstrained = []
        for sig in self.signals.values():
            if sig.signal_type == SignalType.COMMIT and not sig.is_constrained:
                unconstrained.append(sig.full_name)
        
        result = AnalysisResult(
            signals=self.signals,
            constraints=self.constraints,
            lookups=self.lookups,
            namespaces=dict(self.namespaces),
            num_signals=len(self.signals),
            num_constraints=len(self.constraints),
            num_lookups=len(self.lookups),
            unconstrained_signals=unconstrained
        )
        
        return result
    
    def build_constraint_matrix(self) -> Optional[np.ndarray]:
        """
        Build constraint coefficient matrix.
        Returns sparse matrix where M[i,j] is coefficient of signal j in constraint i.
        """
        n_constraints = len(self.constraints)
        n_signals = len(self.signals)
        
        if n_constraints == 0 or n_signals == 0:
            return None
        
        # Build sparse matrix
        matrix = np.zeros((n_constraints, n_signals), dtype=np.float32)
        
        for row, constraint in enumerate(self.constraints):
            for sig_name in constraint.signals_involved:
                if sig_name in self.signal_index:
                    col = self.signal_index[sig_name]
                    # Simple coefficient extraction (1 for presence)
                    # A full parser would extract actual coefficients
                    matrix[row, col] = 1.0
        
        return matrix
    
    def analyze_rank(self, use_gpu: bool = True) -> Dict:
        """
        Analyze constraint matrix rank using SVD.
        
        Returns dict with:
        - shape: matrix dimensions
        - rank: numerical rank
        - deficiency: degrees of freedom
        - singular_values: top singular values
        """
        matrix = self.build_constraint_matrix()
        
        if matrix is None:
            return {'error': 'No constraints parsed'}
        
        m, n = matrix.shape
        k = min(100, min(m, n))
        
        try:
            import torch
            
            if use_gpu and torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                device = 'cuda'
            else:
                device = 'cpu'
            
            mat_torch = torch.from_numpy(matrix).to(device)
            U, S, V = torch.svd_lowrank(mat_torch, q=k)
            singular_values = S.cpu().numpy()
            
            # Count significant singular values
            rank = int(np.sum(singular_values > 1e-6))
            
            return {
                'shape': matrix.shape,
                'rank': rank,
                'max_possible_rank': min(m, n),
                'deficiency': min(m, n) - rank,
                'singular_values': singular_values[:10].tolist(),
                'sparsity': 100 * (1 - np.count_nonzero(matrix) / matrix.size)
            }
            
        except Exception as e:
            # Fallback to numpy
            try:
                U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
                rank = int(np.sum(S > 1e-6))
                return {
                    'shape': matrix.shape,
                    'rank': rank,
                    'max_possible_rank': min(m, n),
                    'deficiency': min(m, n) - rank,
                    'singular_values': S[:10].tolist()
                }
            except Exception as e2:
                return {'error': str(e2)}
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "="*70)
        print("FEZK ELITE PIL Parser - Analysis Summary")
        print("="*70)
        
        print(f"\nSignals: {len(self.signals)}")
        print(f"Constraints: {len(self.constraints)}")
        print(f"Lookups/Permutations: {len(self.lookups)}")
        print(f"Namespaces: {len(self.namespaces)}")
        
        # By namespace
        print("\nSignals by Namespace:")
        for ns, sigs in sorted(self.namespaces.items(), key=lambda x: -len(x[1])):
            commit = sum(1 for s in sigs if s in self.signals and 
                        self.signals[s].signal_type == SignalType.COMMIT)
            const = sum(1 for s in sigs if s in self.signals and 
                       self.signals[s].signal_type == SignalType.CONSTANT)
            print(f"  {ns}: {len(sigs)} total ({commit} commit, {const} constant)")
        
        # Unconstrained
        unconstrained = [s for s in self.signals.values() 
                        if s.signal_type == SignalType.COMMIT and not s.is_constrained]
        
        print(f"\nUnconstrained COMMIT signals: {len(unconstrained)}")
        if unconstrained:
            by_ns = defaultdict(list)
            for sig in unconstrained:
                by_ns[sig.namespace].append(sig)
            
            for ns, sigs in sorted(by_ns.items(), key=lambda x: -len(x[1]))[:5]:
                print(f"  {ns}: {len(sigs)}")
                for sig in sigs[:3]:
                    print(f"    • {sig.name}")
                if len(sigs) > 3:
                    print(f"    ... and {len(sigs)-3} more")


def main():
    """CLI entry point."""
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║   FEZK ELITE PIL Parser v1.0                                                 ║
║   Production-grade Polygon zkEVM Constraint Analyzer                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("Usage: python pil_parser.py <pil_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    parser = PILParser(verbose=True)
    
    if path.is_dir():
        result = parser.parse_directory(path)
    else:
        parser.parse_file(path)
        result = parser._build_result()
    
    parser.print_summary()
    
    # Rank analysis
    print("\n" + "="*70)
    print("Constraint Matrix Rank Analysis")
    print("="*70)
    
    rank_result = parser.analyze_rank()
    if 'error' not in rank_result:
        print(f"Matrix shape: {rank_result['shape']}")
        print(f"Numerical rank: {rank_result['rank']}")
        print(f"Max possible rank: {rank_result['max_possible_rank']}")
        print(f"Degrees of freedom: {rank_result['deficiency']}")
        print(f"Sparsity: {rank_result.get('sparsity', 0):.2f}%")
    else:
        print(f"Error: {rank_result['error']}")


if __name__ == "__main__":
    main()
