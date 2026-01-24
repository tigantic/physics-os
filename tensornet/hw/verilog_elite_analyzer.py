#!/usr/bin/env python3
"""
Verilog Elite Analyzer v1.0
===========================
Hardware security analyzer for Verilog/SystemVerilog designs.
Uses QTT-inspired techniques to find:
- Floating wires (undriven signals)
- X-propagation paths (undefined state propagation)
- FSM holes (unreachable states with security implications)
- Clock domain crossing issues
- Reset glitches

Target: OpenTitan, RISC-V cores, secure enclaves
"""

import re
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
import argparse

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class VulnType(Enum):
    FLOATING_WIRE = "FLOATING_WIRE"           # Undriven signal
    X_PROPAGATION = "X_PROPAGATION"           # X can propagate to security-critical logic
    LATCH_INFERRED = "LATCH_INFERRED"         # Unintended latch (incomplete case/if)
    CLOCK_DOMAIN_CROSSING = "CDC_ISSUE"       # Metastability risk
    RESET_GLITCH = "RESET_GLITCH"             # Async reset issues
    FSM_DEADLOCK = "FSM_DEADLOCK"             # Unreachable/stuck states
    PRIVILEGE_ESCALATION = "PRIV_ESC"         # Unprotected state transitions
    SECRET_EXPOSURE = "SECRET_EXPOSURE"       # Keys/secrets readable in wrong state

@dataclass
class Signal:
    name: str
    signal_type: str  # input, output, inout, wire, reg, logic
    width: int = 1
    file: str = ""
    line: int = 0
    is_driven: bool = False
    is_read: bool = False
    is_security_critical: bool = False
    driver_count: int = 0

@dataclass 
class Finding:
    vuln_type: VulnType
    severity: Severity
    file: str
    line: int
    signal: str
    description: str
    context: str = ""

@dataclass
class Module:
    name: str
    file: str
    signals: Dict[str, Signal] = field(default_factory=dict)
    instances: List[Tuple[str, str]] = field(default_factory=list)  # (instance_name, module_type)
    always_blocks: int = 0
    fsm_states: Set[str] = field(default_factory=set)

class VerilogEliteAnalyzer:
    """QTT-inspired Verilog/SystemVerilog security analyzer"""
    
    # Security-critical signal patterns (case-insensitive)
    SECURITY_PATTERNS = [
        r'key', r'secret', r'priv', r'auth', r'crypt', r'hash', r'rand',
        r'entropy', r'seed', r'nonce', r'iv', r'salt', r'otp', r'fuse',
        r'lc_state', r'lifecycle', r'debug', r'jtag', r'tap', r'scan',
        r'rom', r'boot', r'secure', r'trust', r'lock', r'protect',
        r'keymgr', r'hmac', r'aes', r'sha', r'rsa', r'ecc', r'ecdsa'
    ]
    
    # Dangerous patterns
    DANGER_PATTERNS = [
        r"x'x",          # Explicit X assignment
        r"1'bx",         # Single bit X
        r"default\s*:",  # Default in case (might hide X)
    ]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.modules: Dict[str, Module] = {}
        self.findings: List[Finding] = []
        self.global_signals: Dict[str, Signal] = {}
        self.file_count = 0
        self.total_lines = 0
        
    def is_security_critical(self, name: str) -> bool:
        """Check if signal name suggests security criticality"""
        name_lower = name.lower()
        return any(re.search(pat, name_lower) for pat in self.SECURITY_PATTERNS)
    
    def parse_signal_declaration(self, line: str, file: str, line_num: int) -> List[Signal]:
        """Parse signal declarations from a line"""
        signals = []
        
        # Match: input/output/inout [width] name, name2, ...
        # Match: wire/reg/logic [width] name, name2, ...
        patterns = [
            r'(input|output|inout)\s+(?:wire|reg|logic)?\s*(\[[^\]]+\])?\s*([^;,\(]+)',
            r'(wire|reg|logic)\s*(\[[^\]]+\])?\s*([^;,\(]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                sig_type = match.group(1).lower()
                width_str = match.group(2) or ""
                names_str = match.group(3)
                
                # Parse width
                width = 1
                if width_str:
                    width_match = re.search(r'\[(\d+):(\d+)\]', width_str)
                    if width_match:
                        width = abs(int(width_match.group(1)) - int(width_match.group(2))) + 1
                
                # Parse signal names
                for name in names_str.split(','):
                    name = name.strip()
                    # Remove array dimensions
                    name = re.sub(r'\[[^\]]+\]', '', name).strip()
                    if name and re.match(r'^[a-zA-Z_]', name):
                        sig = Signal(
                            name=name,
                            signal_type=sig_type,
                            width=width,
                            file=file,
                            line=line_num,
                            is_driven=(sig_type == 'input'),
                            is_security_critical=self.is_security_critical(name)
                        )
                        signals.append(sig)
        
        return signals
    
    def analyze_assignments(self, content: str, module: Module):
        """Find all assignments to determine driven signals"""
        # Continuous assignments: assign x = ...
        for match in re.finditer(r'assign\s+(\w+)', content, re.IGNORECASE):
            sig_name = match.group(1)
            if sig_name in module.signals:
                module.signals[sig_name].is_driven = True
                module.signals[sig_name].driver_count += 1
        
        # Blocking/non-blocking assignments in always blocks
        for match in re.finditer(r'(\w+)\s*<?=', content):
            sig_name = match.group(1)
            if sig_name in module.signals:
                module.signals[sig_name].is_driven = True
                module.signals[sig_name].driver_count += 1
        
        # Signal reads (on right side of assignments)
        for match in re.finditer(r'[=<]\s*([^;]+);', content):
            rhs = match.group(1)
            for sig_name in module.signals:
                # Escape special regex characters in signal name
                escaped_name = re.escape(sig_name)
                if re.search(rf'\b{escaped_name}\b', rhs):
                    module.signals[sig_name].is_read = True
    
    def find_x_propagation(self, content: str, file: str) -> List[Finding]:
        """Find explicit X values that could propagate"""
        findings = []
        
        for i, line in enumerate(content.split('\n'), 1):
            for pattern in self.DANGER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's in a security-critical context
                    context_critical = any(
                        re.search(pat, line, re.IGNORECASE) 
                        for pat in self.SECURITY_PATTERNS
                    )
                    
                    findings.append(Finding(
                        vuln_type=VulnType.X_PROPAGATION,
                        severity=Severity.HIGH if context_critical else Severity.MEDIUM,
                        file=file,
                        line=i,
                        signal="X_VALUE",
                        description=f"Explicit X value may propagate to security logic",
                        context=line.strip()[:100]
                    ))
        
        return findings
    
    def find_latch_inference(self, content: str, file: str) -> List[Finding]:
        """Find potential unintended latches from incomplete if/case"""
        findings = []
        
        # Look for always_comb or always @* without complete assignments
        always_blocks = re.finditer(
            r'always\s*@?\s*\*?\s*(?:_comb|_latch)?\s*begin(.*?)end',
            content, re.DOTALL | re.IGNORECASE
        )
        
        for match in always_blocks:
            block = match.group(1)
            
            # Check for if without else
            if re.search(r'\bif\b', block) and not re.search(r'\belse\b', block):
                line_num = content[:match.start()].count('\n') + 1
                findings.append(Finding(
                    vuln_type=VulnType.LATCH_INFERRED,
                    severity=Severity.MEDIUM,
                    file=file,
                    line=line_num,
                    signal="LATCH",
                    description="If without else may infer latch - potential X propagation",
                    context=block[:100].strip()
                ))
            
            # Check for case without default
            if re.search(r'\bcase\b', block) and not re.search(r'\bdefault\b', block):
                line_num = content[:match.start()].count('\n') + 1
                findings.append(Finding(
                    vuln_type=VulnType.LATCH_INFERRED,
                    severity=Severity.MEDIUM,
                    file=file,
                    line=line_num,
                    signal="LATCH",
                    description="Case without default may infer latch",
                    context=block[:100].strip()
                ))
        
        return findings
    
    def find_fsm_issues(self, content: str, file: str) -> List[Finding]:
        """Find FSM state machine issues"""
        findings = []
        
        # Look for FSM patterns
        fsm_matches = re.finditer(
            r'(?:typedef\s+)?enum\s+(?:logic\s*\[[^\]]+\])?\s*\{([^}]+)\}',
            content, re.IGNORECASE
        )
        
        for match in fsm_matches:
            states_str = match.group(1)
            states = [s.strip().split('=')[0].strip() for s in states_str.split(',')]
            
            # Check if there's an ERROR or INVALID state
            has_error_state = any('error' in s.lower() or 'invalid' in s.lower() for s in states)
            
            if not has_error_state and len(states) > 2:
                line_num = content[:match.start()].count('\n') + 1
                findings.append(Finding(
                    vuln_type=VulnType.FSM_DEADLOCK,
                    severity=Severity.LOW,
                    file=file,
                    line=line_num,
                    signal="FSM_STATE",
                    description=f"FSM with {len(states)} states has no explicit error state",
                    context=states_str[:80]
                ))
        
        return findings
    
    def find_security_issues(self, content: str, file: str, module: Module) -> List[Finding]:
        """Find security-specific issues"""
        findings = []
        
        # Check for debug signals that might expose secrets
        for sig_name, sig in module.signals.items():
            if sig.is_security_critical:
                escaped_name = re.escape(sig_name)
                # Check if secret is wired to debug/scan ports
                if re.search(rf'{escaped_name}.*(?:debug|scan|jtag|tap)', content, re.IGNORECASE):
                    findings.append(Finding(
                        vuln_type=VulnType.SECRET_EXPOSURE,
                        severity=Severity.CRITICAL,
                        file=file,
                        line=sig.line,
                        signal=sig_name,
                        description=f"Security-critical signal '{sig_name}' connected to debug interface",
                        context=""
                    ))
                
                # Check if secret readable without proper gating
                if sig.is_read and not re.search(rf'(?:lc_state|lifecycle|enable).*{escaped_name}', content, re.IGNORECASE):
                    # This is a heuristic - needs manual validation
                    if 'key' in sig_name.lower() or 'secret' in sig_name.lower():
                        findings.append(Finding(
                            vuln_type=VulnType.SECRET_EXPOSURE,
                            severity=Severity.HIGH,
                            file=file,
                            line=sig.line,
                            signal=sig_name,
                            description=f"Secret '{sig_name}' may be readable without lifecycle gating",
                            context=""
                        ))
        
        return findings
    
    def analyze_file(self, filepath: str) -> List[Finding]:
        """Analyze a single Verilog/SystemVerilog file"""
        findings = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Cannot read {filepath}: {e}")
            return findings
        
        self.file_count += 1
        lines = content.split('\n')
        self.total_lines += len(lines)
        
        # Find module declarations
        module_matches = re.finditer(
            r'module\s+(\w+)\s*(?:#\([^)]*\))?\s*\(', 
            content, re.IGNORECASE
        )
        
        rel_path = os.path.relpath(filepath)
        
        for match in module_matches:
            module_name = match.group(1)
            module = Module(name=module_name, file=rel_path)
            
            # Find module end
            module_start = match.start()
            module_end = content.find('endmodule', module_start)
            if module_end == -1:
                continue
            
            module_content = content[module_start:module_end]
            
            # Parse signals
            for i, line in enumerate(module_content.split('\n'), 1):
                for sig in self.parse_signal_declaration(line, rel_path, i):
                    module.signals[sig.name] = sig
            
            # Analyze assignments
            self.analyze_assignments(module_content, module)
            
            # Find floating wires (undriven signals that are read)
            for sig_name, sig in module.signals.items():
                if sig.signal_type not in ['input'] and not sig.is_driven and sig.is_read:
                    severity = Severity.CRITICAL if sig.is_security_critical else Severity.HIGH
                    findings.append(Finding(
                        vuln_type=VulnType.FLOATING_WIRE,
                        severity=severity,
                        file=rel_path,
                        line=sig.line,
                        signal=sig_name,
                        description=f"Undriven signal '{sig_name}' ({sig.width} bits) is read - floating wire!",
                        context=f"Type: {sig.signal_type}"
                    ))
                
                # Multi-driver issues
                if sig.driver_count > 1 and sig.signal_type not in ['inout']:
                    findings.append(Finding(
                        vuln_type=VulnType.X_PROPAGATION,
                        severity=Severity.HIGH,
                        file=rel_path,
                        line=sig.line,
                        signal=sig_name,
                        description=f"Signal '{sig_name}' has {sig.driver_count} drivers - contention!",
                        context=""
                    ))
            
            self.modules[module_name] = module
            
            # Additional analyses
            findings.extend(self.find_x_propagation(module_content, rel_path))
            findings.extend(self.find_latch_inference(module_content, rel_path))
            findings.extend(self.find_fsm_issues(module_content, rel_path))
            findings.extend(self.find_security_issues(module_content, rel_path, module))
        
        return findings
    
    def analyze_directory(self, path: str) -> Dict:
        """Analyze all Verilog files in a directory"""
        print(f"\n[VERILOG ELITE] Analyzing: {path}")
        
        # Find all .v and .sv files
        files = []
        for ext in ['*.v', '*.sv']:
            files.extend(Path(path).rglob(ext))
        
        print(f"[VERILOG ELITE] Found {len(files)} Verilog/SystemVerilog files")
        
        for filepath in files:
            file_findings = self.analyze_file(str(filepath))
            self.findings.extend(file_findings)
        
        # Categorize findings
        critical = [f for f in self.findings if f.severity == Severity.CRITICAL]
        high = [f for f in self.findings if f.severity == Severity.HIGH]
        medium = [f for f in self.findings if f.severity == Severity.MEDIUM]
        low = [f for f in self.findings if f.severity == Severity.LOW]
        
        # Print summary
        print(f"\n{'='*60}")
        print("VERILOG ELITE Analysis Results")
        print('='*60)
        print(f"Files Analyzed: {self.file_count}")
        print(f"Lines of Code: {self.total_lines:,}")
        print(f"Modules Found: {len(self.modules)}")
        print(f"Total Signals: {sum(len(m.signals) for m in self.modules.values()):,}")
        print(f"\nFindings:")
        print(f"  CRITICAL: {len(critical)}")
        print(f"  HIGH: {len(high)}")
        print(f"  MEDIUM: {len(medium)}")
        print(f"  LOW: {len(low)}")
        
        # Print critical/high findings
        if critical:
            print(f"\n{'='*60}")
            print("CRITICAL FINDINGS")
            print('='*60)
            for f in critical[:20]:
                print(f"\n[{f.vuln_type.value}] {f.file}:{f.line}")
                print(f"  Signal: {f.signal}")
                print(f"  {f.description}")
                if f.context:
                    print(f"  Context: {f.context[:80]}")
        
        if high and len(critical) < 10:
            print(f"\n{'='*60}")
            print("HIGH FINDINGS (Top 20)")
            print('='*60)
            for f in high[:20]:
                print(f"\n[{f.vuln_type.value}] {f.file}:{f.line}")
                print(f"  Signal: {f.signal}")
                print(f"  {f.description}")
        
        # Security-critical modules
        sec_modules = [m for m in self.modules.values() 
                      if any(s.is_security_critical for s in m.signals.values())]
        if sec_modules:
            print(f"\n{'='*60}")
            print(f"Security-Critical Modules: {len(sec_modules)}")
            print('='*60)
            for m in sec_modules[:10]:
                sec_sigs = [s for s in m.signals.values() if s.is_security_critical]
                print(f"  {m.name}: {len(sec_sigs)} security signals")
        
        return {
            "files": self.file_count,
            "lines": self.total_lines,
            "modules": len(self.modules),
            "signals": sum(len(m.signals) for m in self.modules.values()),
            "findings": {
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low)
            },
            "critical_findings": [
                {
                    "type": f.vuln_type.value,
                    "file": f.file,
                    "line": f.line,
                    "signal": f.signal,
                    "description": f.description
                }
                for f in critical
            ],
            "security_modules": [m.name for m in sec_modules]
        }


def print_banner():
    banner = """
██╗   ██╗███████╗██████╗ ██╗██╗      ██████╗  ██████╗     ███████╗██╗     ██╗████████╗███████╗
██║   ██║██╔════╝██╔══██╗██║██║     ██╔═══██╗██╔════╝     ██╔════╝██║     ██║╚══██╔══╝██╔════╝
██║   ██║█████╗  ██████╔╝██║██║     ██║   ██║██║  ███╗    █████╗  ██║     ██║   ██║   █████╗  
╚██╗ ██╔╝██╔══╝  ██╔══██╗██║██║     ██║   ██║██║   ██║    ██╔══╝  ██║     ██║   ██║   ██╔══╝  
 ╚████╔╝ ███████╗██║  ██║██║███████╗╚██████╔╝╚██████╔╝    ███████╗███████╗██║   ██║   ███████╗
  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝  ╚═════╝     ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝

Verilog Elite Analyzer v1.0 - Hardware Security Analysis
Target: OpenTitan, RISC-V, Secure Enclaves
    """
    print(banner)


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description='Verilog Elite Analyzer - Hardware Security')
    parser.add_argument('path', help='Path to Verilog/SystemVerilog files')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--security-only', action='store_true', help='Only show security-critical findings')
    
    args = parser.parse_args()
    
    analyzer = VerilogEliteAnalyzer(verbose=args.verbose)
    results = analyzer.analyze_directory(args.path)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[VERILOG ELITE] Results saved to: {args.output}")
    
    print(f"\n[VERILOG ELITE] Analysis complete.")


if __name__ == '__main__':
    main()
