#!/usr/bin/env python3
"""
Yosys Netlist Analyzer v2.0
===========================
Uses sv2v + Yosys to elaborate SystemVerilog designs and find TRUE floating wires.

Pipeline: SystemVerilog -> sv2v -> Verilog -> Yosys -> JSON Netlist -> Analysis

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
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import argparse


BANNER = """
██╗   ██╗ ██████╗ ███████╗██╗   ██╗███████╗    ███████╗██╗     ██╗████████╗███████╗    ██╗   ██╗██████╗ 
╚██╗ ██╔╝██╔═══██╗██╔════╝╚██╗ ██╔╝██╔════╝    ██╔════╝██║     ██║╚══██╔══╝██╔════╝    ██║   ██║╚════██╗
 ╚████╔╝ ██║   ██║███████╗ ╚████╔╝ ███████╗    █████╗  ██║     ██║   ██║   █████╗      ██║   ██║ █████╔╝
  ╚██╔╝  ██║   ██║╚════██║  ╚██╔╝  ╚════██║    ██╔══╝  ██║     ██║   ██║   ██╔══╝      ╚██╗ ██╔╝██╔═══╝ 
   ██║   ╚██████╔╝███████║   ██║   ███████║    ███████╗███████╗██║   ██║   ███████╗     ╚████╔╝ ███████╗
   ╚═╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝    ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝      ╚═══╝  ╚══════╝

Yosys Netlist Analyzer v2.0 - TRUE Floating Wire Detection
Pipeline: SystemVerilog -> sv2v -> Yosys -> Netlist Analysis
"""


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
    LATCH_INFERRED = "LATCH_INFERRED"         # Inferred latch in synthesis
    COMB_LOOP = "COMB_LOOP"                   # Combinational feedback loop
    YOSYS_WARNING = "YOSYS_WARNING"           # Yosys synthesis warning


@dataclass
class NetlistFinding:
    vuln_type: VulnType
    severity: Severity
    net_name: str
    description: str
    module: str = ""
    cell: str = ""
    driver_count: int = 0
    location: str = ""


@dataclass
class NetlistStats:
    modules: int = 0
    cells: int = 0
    wires: int = 0
    ports: int = 0
    undriven_nets: int = 0
    multi_driven_nets: int = 0
    sv2v_files: int = 0
    yosys_warnings: int = 0


class YosysNetlistAnalyzer:
    """Yosys-based netlist analyzer for true floating wire detection"""
    
    # Security-critical signal patterns
    SECURITY_PATTERNS = [
        r'key', r'secret', r'priv', r'auth', r'crypt', r'hash', r'rand',
        r'entropy', r'seed', r'nonce', r'iv', r'salt', r'otp', r'fuse',
        r'lc_state', r'lifecycle', r'debug', r'jtag', r'tap', r'scan',
        r'rom', r'boot', r'secure', r'trust', r'lock', r'protect',
        r'hmac', r'aes', r'sha', r'kmac', r'csrng', r'edn'
    ]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[NetlistFinding] = []
        self.stats = NetlistStats()
        self.work_dir = None
        
    def is_security_critical(self, name: str) -> bool:
        """Check if signal name suggests security criticality"""
        name_lower = name.lower()
        return any(re.search(pat, name_lower) for pat in self.SECURITY_PATTERNS)
    
    def find_sv_files(self, path: str) -> List[str]:
        """Find all SystemVerilog/Verilog files"""
        files = []
        abs_path = Path(path).resolve()
        
        if abs_path.is_file():
            # If single file, also include related prim files from same directory
            files.append(str(abs_path))
            rtl_dir = abs_path.parent
            
            # For prim modules, include all prim dependencies
            if 'prim' in str(rtl_dir):
                # Include common prim dependencies
                prim_deps = [
                    'prim_subreg_arb.sv',
                    'prim_subreg.sv', 
                    'prim_subreg_ext.sv',
                    'prim_subreg_shadow.sv',
                    'prim_flop.sv',
                    'prim_flop_2sync.sv',
                    'prim_buf.sv',
                    'prim_mubi4_dec.sv',
                    'prim_mubi4_sync.sv',
                    'prim_mubi8_dec.sv',
                    'prim_mubi8_sync.sv',
                    'prim_sec_anchor_flop.sv',
                    'prim_sec_anchor_buf.sv',
                ]
                for dep in prim_deps:
                    dep_path = rtl_dir / dep
                    if dep_path.exists() and str(dep_path) not in [str(abs_path)]:
                        files.append(str(dep_path.resolve()))
            
            return sorted(set(files))
            
        for ext in ['*.sv', '*.v']:
            files.extend(abs_path.rglob(ext))
        
        # Return absolute paths, sorted for determinism
        return sorted([str(f.resolve()) for f in files])
    
    def find_package_files(self, base_path: str) -> List[str]:
        """Find all package files (_pkg.sv) that might be needed"""
        pkg_files = []
        abs_path = Path(base_path).resolve()
        
        # Patterns to exclude (verification, testbench, sim)
        exclude_patterns = ['/dv/', '/formal/', '/tb/', '/pre_dv/', '/sim/', 
                          '/test/', '/vendor/', '/syn/', '/fpv/', '/generic_dv/',
                          '_dv/', 'env/']
        
        # If it's a file, look in the rtl directory
        if abs_path.is_file():
            rtl_dir = abs_path.parent
        else:
            rtl_dir = abs_path
            
        # 1. First, add packages from the same rtl directory
        for pkg in rtl_dir.glob('*_pkg.sv'):
            pkg_str = str(pkg.resolve())
            if pkg_str not in pkg_files:
                pkg_files.append(pkg_str)
        
        # 2. Find the hw root to get common packages
        hw_root = None
        test_path = rtl_dir
        for _ in range(6):
            if (test_path / 'ip').exists() and test_path.name == 'hw':
                hw_root = test_path
                break
            test_path = test_path.parent
        
        if hw_root:
            # Add core packages from standard locations
            pkg_locations = [
                hw_root / 'ip' / 'prim' / 'rtl',
                hw_root / 'ip' / 'tlul' / 'rtl',
                hw_root / 'ip' / 'lc_ctrl' / 'rtl',
                hw_root / 'ip' / 'otp_ctrl' / 'rtl',
                hw_root / 'ip' / 'edn' / 'rtl',
                hw_root / 'ip' / 'csrng' / 'rtl',
                hw_root / 'ip' / 'entropy_src' / 'rtl',
                hw_root / 'ip' / 'keymgr' / 'rtl',
                hw_root / 'top_earlgrey' / 'rtl',
            ]
            
            for loc in pkg_locations:
                if loc.exists():
                    for pkg in loc.glob('*_pkg.sv'):
                        pkg_str = str(pkg.resolve())
                        if any(excl in pkg_str for excl in exclude_patterns):
                            continue
                        if pkg_str not in pkg_files:
                            pkg_files.append(pkg_str)
            
        return sorted(pkg_files)
    
    def run_sv2v(self, sv_files: List[str], pkg_files: List[str], output_dir: str) -> List[str]:
        """Run sv2v to convert SystemVerilog to Verilog"""
        converted_files = []
        
        # Find include directories (containing .svh files)
        include_dirs = set()
        for f in sv_files + pkg_files:
            parent = Path(f).parent
            # Check if this directory has .svh files
            if list(parent.glob('*.svh')):
                include_dirs.add(str(parent))
            # Also check for prim/rtl
            prim_rtl = parent.parent / 'prim' / 'rtl'
            if prim_rtl.exists():
                include_dirs.add(str(prim_rtl))
        
        for sv_file in sv_files:
            # Skip package files (they're included as dependencies)
            if '_pkg.sv' in sv_file:
                continue
                
            module_name = Path(sv_file).stem
            output_file = os.path.join(output_dir, f"{module_name}.v")
            
            # Build sv2v command
            cmd = ['sv2v', '--define=SYNTHESIS', '--define=YOSYS']
            
            # Add include paths
            for inc_dir in include_dirs:
                cmd.extend(['-I', inc_dir])
            
            # Add package files first
            for pkg in pkg_files:
                cmd.append(pkg)
            
            # Add include paths (look for prim/rtl)
            sv_path = Path(sv_file).parent
            if 'rtl' in str(sv_path):
                # Find the prim/rtl directory
                prim_search = sv_path
                for _ in range(5):
                    prim_rtl = prim_search / 'prim' / 'rtl'
                    if prim_rtl.exists():
                        cmd.extend(['-I', str(prim_rtl)])
                        break
                    prim_search = prim_search.parent
            
            # Add the source file
            cmd.append(sv_file)
            
            if self.verbose:
                print(f"  [sv2v] Converting {module_name}...")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    with open(output_file, 'w') as f:
                        f.write(result.stdout)
                    converted_files.append(output_file)
                    self.stats.sv2v_files += 1
                else:
                    if self.verbose:
                        print(f"    [WARN] sv2v failed for {module_name}: {result.stderr[:200]}")
                        
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print(f"    [WARN] sv2v timeout for {module_name}")
            except Exception as e:
                if self.verbose:
                    print(f"    [WARN] sv2v error for {module_name}: {e}")
        
        return converted_files
    
    def create_yosys_script(self, verilog_files: List[str], top_module: str = "") -> str:
        """Create Yosys script for elaboration"""
        script_lines = []
        
        # Read all converted Verilog files
        for f in verilog_files:
            script_lines.append(f"read_verilog {f}")
        
        # Elaborate hierarchy
        if top_module:
            script_lines.append(f"hierarchy -top {top_module}")
        else:
            script_lines.append("hierarchy -auto-top")
        
        # Flatten to resolve all hierarchy
        script_lines.append("flatten")
        
        # Process to convert always blocks to gates
        script_lines.append("proc")
        
        # Optimize to clean up
        script_lines.append("opt_clean -purge")
        
        # Check for issues
        script_lines.append("check -noinit -assert")
        
        # Write JSON netlist
        script_lines.append("write_json netlist.json")
        
        # Statistics
        script_lines.append("stat")
        
        return "\n".join(script_lines)
    
    def run_yosys(self, script: str, work_dir: str) -> Tuple[bool, str, str]:
        """Run Yosys with the given script"""
        script_path = os.path.join(work_dir, "analyze.ys")
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                ["yosys", "-q", "-s", script_path],  # -q for quiet mode
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout: Yosys took too long (>5min)"
        except Exception as e:
            return False, "", str(e)
    
    def parse_yosys_output(self, stdout: str, stderr: str):
        """Parse Yosys output for warnings and issues"""
        combined = stdout + stderr
        
        # Track unique signals (Yosys reports per-bit, we want per-signal)
        undriven_signals: Dict[str, int] = {}  # signal_base -> bit_count
        blackbox_signals: Dict[str, int] = {}  # Signals from missing primitives
        
        # Patterns for externally-driven signals (connected at SoC level)
        external_patterns = [
            r'tl_i',           # TL-UL input port
            r'tl_o',           # TL-UL output port  
            r'reg2hw',         # Register file to hardware interface
            r'hw2reg',         # Hardware to register file interface
            r'clk_i',          # Clock input
            r'rst_ni',         # Reset input
            r'_req\b',         # Request signals (externally driven)
            r'_ack\b',         # Acknowledge signals (externally driven)
            r'\bedn_',         # EDN interface (external entropy)
            r'\bkeymgr_',      # Key manager interface
            r'\blc_escalate',  # Lifecycle escalation
            r'alert_rx',       # Alert RX interface
            r'alert_tx',       # Alert TX interface
            r'intr_',          # Interrupt signals
            r'devmode',        # Development mode
            r'idle_o',         # Idle output
        ]
        
        # Patterns for blackbox primitive outputs (missing implementations)
        blackbox_patterns = [
            r'prim_sparse_fsm_flop',  # FSM flop (blackboxed)
            r'prim_flop',             # Generic flop (blackboxed)
            r'prim_buf',              # Buffer (blackboxed)
            r'prim_and2',             # AND gate (blackboxed)
            r'prim_xor2',             # XOR gate (blackboxed)
            r'prim_sec_anchor',       # Security anchor (blackboxed)
            r'gen_flop',              # Generated flop instance
            r'u_state_regs',          # State register instance
        ]
        
        # Look for undriven wire warnings
        undriven_pattern = r'Warning: Wire ([^\s]+?)(?:\s+\[\d+\])?\s+is used but has no driver'
        for match in re.finditer(undriven_pattern, combined):
            full_name = match.group(1)
            base_name = re.sub(r'\[\d+\]$', '', full_name)
            
            # Skip externally-connected signals
            if any(re.search(pat, base_name) for pat in external_patterns):
                continue
            
            # Categorize as blackbox if from a primitive
            is_blackbox = any(re.search(pat, base_name) for pat in blackbox_patterns)
            
            if is_blackbox:
                blackbox_signals[base_name] = blackbox_signals.get(base_name, 0) + 1
            else:
                undriven_signals[base_name] = undriven_signals.get(base_name, 0) + 1
        
        # Create findings for unique undriven signals (TRUE issues)
        for signal, bit_count in undriven_signals.items():
            severity = Severity.CRITICAL if self.is_security_critical(signal) else Severity.HIGH
            self.findings.append(NetlistFinding(
                vuln_type=VulnType.TRUE_FLOATING,
                severity=severity,
                net_name=signal,
                description=f"TRUE FLOATING: {bit_count} bit(s) undriven after hierarchy elaboration"
            ))
            self.stats.undriven_nets += 1
        
        # Create lower-priority findings for blackbox signals
        for signal, bit_count in blackbox_signals.items():
            # These are from missing primitives, still worth noting but lower priority
            severity = Severity.MEDIUM if self.is_security_critical(signal) else Severity.LOW
            self.findings.append(NetlistFinding(
                vuln_type=VulnType.UNCONNECTED_PORT,
                severity=severity,
                net_name=signal,
                description=f"BLACKBOX: {bit_count} bit(s) from missing primitive (requires full prim library)"
            ))
        
        # Look for latch inference warnings
        latch_pattern = r'Warning:.*?[Ll]atch inferred for signal[^`]*`([^`\']+)'
        for match in re.finditer(latch_pattern, combined):
            signal = match.group(1)
            severity = Severity.HIGH if self.is_security_critical(signal) else Severity.MEDIUM
            self.findings.append(NetlistFinding(
                vuln_type=VulnType.LATCH_INFERRED,
                severity=severity,
                net_name=signal,
                description=f"Latch inferred for signal - may cause X-propagation or state retention issues"
            ))
        
        # Look for combinational loop warnings
        loop_pattern = r'(?:combinational loop|feedback).*?`([^`]+)\''
        for match in re.finditer(loop_pattern, combined, re.IGNORECASE):
            signal = match.group(1)
            self.findings.append(NetlistFinding(
                vuln_type=VulnType.COMB_LOOP,
                severity=Severity.HIGH,
                net_name=signal,
                description="Combinational feedback loop detected - may cause oscillation or undefined behavior"
            ))
        
        # Count total warnings
        warning_count = len(re.findall(r'Warning:', combined))
        self.stats.yosys_warnings = warning_count
    
    def parse_netlist_json(self, json_path: str) -> Dict:
        """Parse Yosys JSON netlist"""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"  [WARN] Cannot parse netlist JSON: {e}")
            return {}
    
    def analyze_netlist(self, netlist: Dict):
        """Analyze the JSON netlist for security issues"""
        if not netlist or 'modules' not in netlist:
            return
        
        for module_name, module_data in netlist.get('modules', {}).items():
            self.stats.modules += 1
            
            # Track which nets are driven
            net_drivers: Dict[int, List[str]] = {}  # bit_id -> drivers
            net_readers: Dict[int, List[str]] = {}  # bit_id -> readers
            net_names: Dict[int, str] = {}  # bit_id -> net_name
            
            # Map net names
            netnames = module_data.get('netnames', {})
            for net_name, net_data in netnames.items():
                self.stats.wires += 1
                bits = net_data.get('bits', [])
                for bit in bits:
                    if isinstance(bit, int):
                        net_names[bit] = net_name
            
            # Analyze ports
            ports = module_data.get('ports', {})
            for port_name, port_data in ports.items():
                self.stats.ports += 1
                direction = port_data.get('direction', '')
                bits = port_data.get('bits', [])
                
                for bit in bits:
                    if isinstance(bit, int):
                        if direction == 'input':
                            net_drivers.setdefault(bit, []).append(f"input:{port_name}")
                        elif direction == 'output':
                            net_readers.setdefault(bit, []).append(f"output:{port_name}")
            
            # Analyze cells
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
                            if direction == 'output':
                                net_drivers.setdefault(bit, []).append(f"{cell_type}:{cell_name}.{conn_name}")
                            elif direction == 'input':
                                net_readers.setdefault(bit, []).append(f"{cell_type}:{cell_name}.{conn_name}")
            
            # Find issues
            for bit_id, net_name in net_names.items():
                drivers = net_drivers.get(bit_id, [])
                readers = net_readers.get(bit_id, [])
                
                # Skip if this is an input port (externally driven)
                if any('input:' in d for d in drivers):
                    continue
                
                # Check for undriven nets
                if not drivers and readers:
                    severity = Severity.CRITICAL if self.is_security_critical(net_name) else Severity.HIGH
                    self.findings.append(NetlistFinding(
                        vuln_type=VulnType.TRUE_FLOATING,
                        severity=severity,
                        net_name=net_name,
                        description=f"TRUE FLOATING: Net read by {len(readers)} cell(s) but has no driver",
                        module=module_name,
                        driver_count=0
                    ))
                    self.stats.undriven_nets += 1
                
                # Check for multi-driven nets
                if len(drivers) > 1:
                    severity = Severity.HIGH if self.is_security_critical(net_name) else Severity.MEDIUM
                    self.findings.append(NetlistFinding(
                        vuln_type=VulnType.MULTI_DRIVER,
                        severity=severity,
                        net_name=net_name,
                        description=f"Multiple drivers ({len(drivers)}): {', '.join(drivers[:3])}",
                        module=module_name,
                        driver_count=len(drivers)
                    ))
                    self.stats.multi_driven_nets += 1
    
    def analyze(self, path: str, top_module: str = "") -> List[NetlistFinding]:
        """Run complete analysis pipeline"""
        self.findings = []
        self.stats = NetlistStats()
        
        # Find source files
        sv_files = self.find_sv_files(path)
        if not sv_files:
            print(f"[ERROR] No SystemVerilog/Verilog files found in {path}")
            return []
        
        pkg_files = self.find_package_files(path)
        
        if self.verbose:
            print(f"[YOSYS ELITE] Found {len(sv_files)} source files, {len(pkg_files)} package files")
        
        # Create temp directory
        self.work_dir = tempfile.mkdtemp(prefix="yosys_elite_")
        
        try:
            # Step 1: Convert SV to V using sv2v
            if self.verbose:
                print(f"[YOSYS ELITE] Step 1: Converting SystemVerilog to Verilog with sv2v...")
            
            v_files = self.run_sv2v(sv_files, pkg_files, self.work_dir)
            
            if not v_files:
                print(f"[ERROR] sv2v failed to convert any files")
                return []
            
            if self.verbose:
                print(f"[YOSYS ELITE] Successfully converted {len(v_files)} files")
            
            # Step 2: Run Yosys
            if self.verbose:
                print(f"[YOSYS ELITE] Step 2: Running Yosys elaboration...")
            
            script = self.create_yosys_script(v_files, top_module)
            
            if self.verbose:
                print(f"[YOSYS ELITE] Yosys script:\n{script}")
            
            success, stdout, stderr = self.run_yosys(script, self.work_dir)
            
            # Parse Yosys output for warnings
            self.parse_yosys_output(stdout, stderr)
            
            if not success:
                if self.verbose:
                    print(f"[YOSYS ELITE] Yosys elaboration failed (this may be expected for complex designs)")
                    print(f"[YOSYS ELITE] Stderr: {stderr[:500]}")
            
            # Step 3: Parse netlist JSON
            json_path = os.path.join(self.work_dir, "netlist.json")
            if os.path.exists(json_path):
                if self.verbose:
                    print(f"[YOSYS ELITE] Step 3: Analyzing netlist JSON...")
                netlist = self.parse_netlist_json(json_path)
                self.analyze_netlist(netlist)
            else:
                if self.verbose:
                    print(f"[YOSYS ELITE] No netlist.json generated - using Yosys output warnings only")
        
        finally:
            # Clean up temp directory (unless debugging)
            if not self.verbose:
                shutil.rmtree(self.work_dir, ignore_errors=True)
            else:
                print(f"[DEBUG] Work directory preserved: {self.work_dir}")
        
        return self.findings
    
    def print_report(self):
        """Print analysis report"""
        print("\n" + "="*70)
        print("YOSYS NETLIST ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\nSynthesis Statistics:")
        print(f"  Modules Analyzed: {self.stats.modules}")
        print(f"  Cells: {self.stats.cells}")
        print(f"  Wires: {self.stats.wires}")
        print(f"  Ports: {self.stats.ports}")
        print(f"  Files Converted (sv2v): {self.stats.sv2v_files}")
        print(f"  Yosys Warnings: {self.stats.yosys_warnings}")
        
        print(f"\nNet Analysis:")
        print(f"  TRUE Undriven Nets: {self.stats.undriven_nets}")
        print(f"  Multi-Driven Nets: {self.stats.multi_driven_nets}")
        
        # Count by severity
        severity_counts = {}
        for finding in self.findings:
            severity_counts[finding.severity.value] = severity_counts.get(finding.severity.value, 0) + 1
        
        print(f"\nFindings by Severity:")
        for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = severity_counts.get(sev, 0)
            print(f"  {sev}: {count}")
        
        # Print findings by type
        if self.findings:
            print(f"\n" + "-"*70)
            print("DETAILED FINDINGS")
            print("-"*70)
            
            # Group by type
            by_type = {}
            for f in self.findings:
                by_type.setdefault(f.vuln_type.value, []).append(f)
            
            for vuln_type, findings in by_type.items():
                print(f"\n[{vuln_type}] ({len(findings)} findings)")
                
                # Show top 10 per type
                for f in sorted(findings, key=lambda x: x.severity.value)[:10]:
                    sec_marker = " [SECURITY]" if self.is_security_critical(f.net_name) else ""
                    print(f"  [{f.severity.value}]{sec_marker} {f.net_name}")
                    print(f"    {f.description}")
                
                if len(findings) > 10:
                    print(f"  ... and {len(findings)-10} more")
        
        print("\n[YOSYS ELITE] Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Yosys Netlist Analyzer v2.0 - TRUE Floating Wire Detection"
    )
    parser.add_argument("path", help="Path to SystemVerilog file or directory")
    parser.add_argument("-t", "--top", default="", help="Top module name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", help="Output JSON file for findings")
    
    args = parser.parse_args()
    
    print(BANNER)
    
    # Check for required tools
    for tool in ['sv2v', 'yosys']:
        result = subprocess.run(['which', tool], capture_output=True)
        if result.returncode != 0:
            print(f"[ERROR] Required tool '{tool}' not found. Please install it.")
            sys.exit(1)
    
    print(f"[YOSYS ELITE] Analyzing: {args.path}")
    
    analyzer = YosysNetlistAnalyzer(verbose=args.verbose)
    findings = analyzer.analyze(args.path, args.top)
    
    analyzer.print_report()
    
    # Save JSON output
    if args.output:
        output_data = {
            'stats': {
                'modules': analyzer.stats.modules,
                'cells': analyzer.stats.cells,
                'wires': analyzer.stats.wires,
                'sv2v_files': analyzer.stats.sv2v_files,
                'undriven_nets': analyzer.stats.undriven_nets,
                'multi_driven_nets': analyzer.stats.multi_driven_nets
            },
            'findings': [
                {
                    'type': f.vuln_type.value,
                    'severity': f.severity.value,
                    'net_name': f.net_name,
                    'description': f.description,
                    'module': f.module,
                    'security_critical': analyzer.is_security_critical(f.net_name)
                }
                for f in findings
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[YOSYS ELITE] Results saved to {args.output}")
    
    # Return exit code based on findings
    critical_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
