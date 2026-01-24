#!/usr/bin/env python3
"""
Yosys Netlist Analyzer v1.0
===========================
Uses Yosys to elaborate Verilog/SystemVerilog designs and find TRUE floating wires.

Key Advantage: Resolves hierarchy, traces through prim_subreg and other wrappers
to identify signals that are ACTUALLY undriven at the netlist level.

Target: OpenTitan, RISC-V, Secure Enclaves
"""

import os
import re
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import argparse


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class VulnType(Enum):
    TRUE_FLOATING = "TRUE_FLOATING"           # Confirmed undriven after elaboration
    UNCONNECTED_PORT = "UNCONNECTED_PORT"     # Module port not connected
    MULTI_DRIVER = "MULTI_DRIVER"             # Multiple drivers on same net
    CONST_X = "CONST_X"                       # Constant X in netlist
    LATCH_CELL = "LATCH_CELL"                 # Inferred latch in synthesis
    COMBINATIONAL_LOOP = "COMB_LOOP"          # Feedback without register


@dataclass
class NetlistFinding:
    vuln_type: VulnType
    severity: Severity
    net_name: str
    description: str
    module: str = ""
    cell: str = ""
    driver_count: int = 0


@dataclass
class NetlistStats:
    modules: int = 0
    cells: int = 0
    wires: int = 0
    ports: int = 0
    undriven_nets: int = 0
    multi_driven_nets: int = 0


class YosysNetlistAnalyzer:
    """Yosys-based netlist analyzer for true floating wire detection"""
    
    # Security-critical signal patterns
    SECURITY_PATTERNS = [
        r'key', r'secret', r'priv', r'auth', r'crypt', r'hash', r'rand',
        r'entropy', r'seed', r'nonce', r'iv', r'salt', r'otp', r'fuse',
        r'lc_state', r'lifecycle', r'debug', r'jtag', r'tap', r'scan',
        r'rom', r'boot', r'secure', r'trust', r'lock', r'protect'
    ]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[NetlistFinding] = []
        self.stats = NetlistStats()
        self.temp_dir = None
        
    def is_security_critical(self, name: str) -> bool:
        """Check if signal name suggests security criticality"""
        name_lower = name.lower()
        return any(re.search(pat, name_lower) for pat in self.SECURITY_PATTERNS)
    
    def find_sv_files(self, path: str) -> List[str]:
        """Find all SystemVerilog/Verilog files"""
        files = []
        abs_path = Path(path).resolve()
        for ext in ['*.sv', '*.v']:
            files.extend(abs_path.rglob(ext))
        # Return absolute paths for Yosys
        return [str(f.resolve()) for f in files]
    
    def create_yosys_script(self, files: List[str], top_module: str = "", work_dir: str = "") -> str:
        """Create Yosys TCL script for elaboration"""
        script = []
        
        # Read all files with absolute paths
        for f in files:
            abs_f = str(Path(f).resolve())
            # Use read_verilog with -sv for SystemVerilog
            if f.endswith('.sv'):
                script.append(f"read_verilog -sv {abs_f}")
            else:
                script.append(f"read_verilog {abs_f}")
        
        # Elaborate hierarchy
        if top_module:
            script.append(f"hierarchy -top {top_module}")
        else:
            script.append("hierarchy -auto-top")
        
        # Flatten for complete signal tracing
        script.append("flatten")
        
        # Run synthesis passes to expose issues
        script.append("proc")  # Convert processes to netlists
        script.append("opt_clean")  # Clean up
        
        # Check for issues
        script.append("check -assert")
        
        # Write JSON netlist for analysis
        script.append("write_json netlist.json")
        
        # Write statistics
        script.append("stat")
        
        return "\n".join(script)
    
    def run_yosys(self, script: str, work_dir: str) -> Tuple[bool, str, str]:
        """Run Yosys with the given script"""
        script_path = os.path.join(work_dir, "analyze.ys")
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                ["yosys", "-s", script_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout: Yosys took too long"
        except Exception as e:
            return False, "", str(e)
    
    def parse_yosys_json(self, json_path: str) -> Dict:
        """Parse Yosys JSON netlist output"""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Cannot parse JSON: {e}")
            return {}
    
    def analyze_netlist(self, netlist: Dict) -> List[NetlistFinding]:
        """Analyze Yosys JSON netlist for security issues"""
        findings = []
        
        if not netlist or 'modules' not in netlist:
            return findings
        
        for module_name, module_data in netlist.get('modules', {}).items():
            self.stats.modules += 1
            
            # Track net drivers
            net_drivers: Dict[str, List[str]] = {}  # net_name -> list of drivers
            net_readers: Dict[str, List[str]] = {}  # net_name -> list of readers
            
            # Analyze ports
            ports = module_data.get('ports', {})
            for port_name, port_data in ports.items():
                self.stats.ports += 1
                direction = port_data.get('direction', '')
                bits = port_data.get('bits', [])
                
                # Input ports are drivers
                if direction == 'input':
                    for bit in bits:
                        if isinstance(bit, int):
                            net_drivers.setdefault(str(bit), []).append(f"port:{port_name}")
                
                # Output ports are readers
                elif direction == 'output':
                    for bit in bits:
                        if isinstance(bit, int):
                            net_readers.setdefault(str(bit), []).append(f"port:{port_name}")
            
            # Analyze cells (instances)
            cells = module_data.get('cells', {})
            for cell_name, cell_data in cells.items():
                self.stats.cells += 1
                cell_type = cell_data.get('type', '')
                connections = cell_data.get('connections', {})
                port_directions = cell_data.get('port_directions', {})
                
                for conn_name, bits in connections.items():
                    direction = port_directions.get(conn_name, 'unknown')
                    
                    for bit in bits:
                        if isinstance(bit, int):
                            net_id = str(bit)
                            if direction == 'output':
                                net_drivers.setdefault(net_id, []).append(f"cell:{cell_name}:{conn_name}")
                            elif direction == 'input':
                                net_readers.setdefault(net_id, []).append(f"cell:{cell_name}:{conn_name}")
                        elif bit == "x":
                            # Constant X - security issue
                            findings.append(NetlistFinding(
                                vuln_type=VulnType.CONST_X,
                                severity=Severity.HIGH,
                                net_name=conn_name,
                                description=f"Constant X value connected to {cell_name}.{conn_name}",
                                module=module_name,
                                cell=cell_name
                            ))
            
            # Analyze netnames for undriven signals
            netnames = module_data.get('netnames', {})
            for net_name, net_data in netnames.items():
                self.stats.wires += 1
                bits = net_data.get('bits', [])
                
                for bit in bits:
                    if isinstance(bit, int):
                        net_id = str(bit)
                        drivers = net_drivers.get(net_id, [])
                        readers = net_readers.get(net_id, [])
                        
                        # TRUE FLOATING: has readers but no drivers
                        if readers and not drivers:
                            self.stats.undriven_nets += 1
                            severity = Severity.CRITICAL if self.is_security_critical(net_name) else Severity.HIGH
                            findings.append(NetlistFinding(
                                vuln_type=VulnType.TRUE_FLOATING,
                                severity=severity,
                                net_name=net_name,
                                description=f"CONFIRMED FLOATING: Net '{net_name}' has {len(readers)} reader(s) but NO driver",
                                module=module_name,
                                driver_count=0
                            ))
                        
                        # Multi-driver
                        if len(drivers) > 1:
                            self.stats.multi_driven_nets += 1
                            findings.append(NetlistFinding(
                                vuln_type=VulnType.MULTI_DRIVER,
                                severity=Severity.HIGH,
                                net_name=net_name,
                                description=f"Multi-driver: Net '{net_name}' driven by {len(drivers)} sources",
                                module=module_name,
                                driver_count=len(drivers)
                            ))
        
        return findings
    
    def parse_yosys_output(self, stdout: str) -> List[NetlistFinding]:
        """Parse Yosys stdout for warnings and errors"""
        findings = []
        
        # Look for warning patterns
        warning_patterns = [
            (r"Warning: Wire ([\w\.\[\]]+) is used but has no driver", VulnType.TRUE_FLOATING),
            (r"Warning: multiple drivers for ([\w\.\[\]]+)", VulnType.MULTI_DRIVER),
            (r"Warning:.*latch.*for signal ([\w\.\[\]]+)", VulnType.LATCH_CELL),
            (r"Warning: Found combinational loop", VulnType.COMBINATIONAL_LOOP),
        ]
        
        for line in stdout.split('\n'):
            for pattern, vuln_type in warning_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    net_name = match.group(1) if match.lastindex else "unknown"
                    severity = Severity.CRITICAL if self.is_security_critical(net_name) else Severity.HIGH
                    findings.append(NetlistFinding(
                        vuln_type=vuln_type,
                        severity=severity,
                        net_name=net_name,
                        description=line.strip()
                    ))
        
        return findings
    
    def analyze_module(self, path: str, top_module: str = "", 
                      include_dirs: List[str] = None) -> Dict:
        """Analyze a Verilog module/directory with Yosys"""
        
        print(f"\n[YOSYS ELITE] Analyzing: {path}")
        
        # Find all source files
        if os.path.isfile(path):
            files = [path]
        else:
            files = self.find_sv_files(path)
        
        print(f"[YOSYS ELITE] Found {len(files)} source files")
        
        if not files:
            return {"error": "No Verilog/SystemVerilog files found"}
        
        # Create temp directory for Yosys output
        import tempfile
        with tempfile.TemporaryDirectory() as work_dir:
            # Create Yosys script
            script = self.create_yosys_script(files, top_module)
            
            if self.verbose:
                print(f"[YOSYS ELITE] Script:\n{script}")
            
            # Run Yosys
            print("[YOSYS ELITE] Running Yosys elaboration...")
            success, stdout, stderr = self.run_yosys(script, work_dir)
            
            if not success:
                print(f"[YOSYS ELITE] Yosys failed: {stderr[:500]}")
                # Still try to extract warnings from partial output
                self.findings.extend(self.parse_yosys_output(stdout))
                self.findings.extend(self.parse_yosys_output(stderr))
            else:
                print("[YOSYS ELITE] Yosys elaboration complete")
                
                # Parse stdout for warnings
                self.findings.extend(self.parse_yosys_output(stdout))
                
                # Parse JSON netlist if generated
                json_path = os.path.join(work_dir, "netlist.json")
                if os.path.exists(json_path):
                    print("[YOSYS ELITE] Parsing JSON netlist...")
                    netlist = self.parse_yosys_json(json_path)
                    self.findings.extend(self.analyze_netlist(netlist))
        
        # Categorize findings
        critical = [f for f in self.findings if f.severity == Severity.CRITICAL]
        high = [f for f in self.findings if f.severity == Severity.HIGH]
        medium = [f for f in self.findings if f.severity == Severity.MEDIUM]
        
        # Print results
        self.print_results(critical, high, medium)
        
        return {
            "stats": {
                "modules": self.stats.modules,
                "cells": self.stats.cells,
                "wires": self.stats.wires,
                "ports": self.stats.ports,
                "undriven_nets": self.stats.undriven_nets,
                "multi_driven_nets": self.stats.multi_driven_nets
            },
            "findings": {
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium)
            },
            "critical_findings": [
                {
                    "type": f.vuln_type.value,
                    "net": f.net_name,
                    "module": f.module,
                    "description": f.description
                }
                for f in critical
            ]
        }
    
    def print_results(self, critical: List, high: List, medium: List):
        """Print analysis results"""
        print(f"\n{'='*70}")
        print("YOSYS NETLIST ANALYSIS RESULTS")
        print('='*70)
        print(f"Modules: {self.stats.modules}")
        print(f"Cells: {self.stats.cells}")
        print(f"Wires: {self.stats.wires}")
        print(f"Ports: {self.stats.ports}")
        print(f"\nUndriven Nets: {self.stats.undriven_nets}")
        print(f"Multi-Driven Nets: {self.stats.multi_driven_nets}")
        print(f"\nFindings:")
        print(f"  CRITICAL: {len(critical)}")
        print(f"  HIGH: {len(high)}")
        print(f"  MEDIUM: {len(medium)}")
        
        if critical:
            print(f"\n{'='*70}")
            print("CRITICAL FINDINGS (Confirmed Floating Wires)")
            print('='*70)
            for f in critical[:30]:
                print(f"\n[{f.vuln_type.value}] {f.net_name}")
                print(f"  Module: {f.module}")
                print(f"  {f.description}")
        
        if high and len(critical) < 15:
            print(f"\n{'='*70}")
            print("HIGH FINDINGS (Top 20)")
            print('='*70)
            for f in high[:20]:
                print(f"\n[{f.vuln_type.value}] {f.net_name}")
                print(f"  {f.description}")


def print_banner():
    banner = """
██╗   ██╗ ██████╗ ███████╗██╗   ██╗███████╗    ███████╗██╗     ██╗████████╗███████╗
╚██╗ ██╔╝██╔═══██╗██╔════╝╚██╗ ██╔╝██╔════╝    ██╔════╝██║     ██║╚══██╔══╝██╔════╝
 ╚████╔╝ ██║   ██║███████╗ ╚████╔╝ ███████╗    █████╗  ██║     ██║   ██║   █████╗  
  ╚██╔╝  ██║   ██║╚════██║  ╚██╔╝  ╚════██║    ██╔══╝  ██║     ██║   ██║   ██╔══╝  
   ██║   ╚██████╔╝███████║   ██║   ███████║    ███████╗███████╗██║   ██║   ███████╗
   ╚═╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝    ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝

Yosys Netlist Analyzer v1.0 - TRUE Floating Wire Detection
Resolves hierarchy through prim_subreg and other wrappers
    """
    print(banner)


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description='Yosys Netlist Analyzer')
    parser.add_argument('path', help='Path to Verilog/SystemVerilog files')
    parser.add_argument('--top', '-t', help='Top module name', default="")
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    analyzer = YosysNetlistAnalyzer(verbose=args.verbose)
    results = analyzer.analyze_module(args.path, args.top)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[YOSYS ELITE] Results saved to: {args.output}")
    
    print(f"\n[YOSYS ELITE] Analysis complete.")


if __name__ == '__main__':
    main()
