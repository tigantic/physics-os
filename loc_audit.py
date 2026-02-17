#!/usr/bin/env python3
"""Full forensic LOC audit of HyperTensor-VM repository."""
import os
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).parent
SKIP_DIRS = {'.git', '__pycache__', '.mypy_cache', '.pytest_cache', 'node_modules',
             '.cache', '.cargo', 'target', '.vscode', 'vendor', 'weights',
             'pdb_cache', 'cache', '.cache'}

def count_lines(filepath):
    """Count total lines, code lines, blank lines, comment lines."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except:
        return 0, 0, 0, 0
    total = len(lines)
    blank = sum(1 for l in lines if l.strip() == '')
    comment = sum(1 for l in lines if l.strip().startswith('#') or l.strip().startswith('//') or l.strip().startswith('/*') or l.strip().startswith('*'))
    docstring_lines = 0
    in_docstring = False
    for l in lines:
        stripped = l.strip()
        if '"""' in stripped or "'''" in stripped:
            count = stripped.count('"""') + stripped.count("'''")
            if count >= 2:
                docstring_lines += 1
            else:
                in_docstring = not in_docstring
                docstring_lines += 1
        elif in_docstring:
            docstring_lines += 1
    code = total - blank - comment
    return total, code, blank, comment + docstring_lines

# Extension categories
CODE_EXTS = {'.py', '.rs', '.ts', '.tsx', '.js', '.jsx', '.sol', '.circom',
             '.lean', '.sh', '.bash', '.zsh', '.css', '.html', '.vue',
             '.svelte', '.c', '.cpp', '.h', '.hpp', '.cu', '.wgsl', '.glsl',
             '.vert', '.frag', '.comp'}
DOC_EXTS = {'.md', '.rst', '.txt', '.adoc'}
CONFIG_EXTS = {'.toml', '.yaml', '.yml', '.json', '.cfg', '.ini', '.env',
               '.lock', '.xml'}

results = {
    'by_extension': defaultdict(lambda: {'files': 0, 'total_lines': 0, 'code_lines': 0, 'blank_lines': 0, 'bytes': 0}),
    'by_top_dir': defaultdict(lambda: {'files': 0, 'total_lines': 0, 'code_lines': 0, 'bytes': 0}),
    'tensornet_modules': defaultdict(lambda: {'files': 0, 'total_lines': 0, 'code_lines': 0, 'bytes': 0}),
    'largest_files': [],
    'totals': {'files': 0, 'total_lines': 0, 'code_lines': 0, 'blank_lines': 0, 'bytes': 0}
}

all_files = []

for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    rel_root = Path(root).relative_to(REPO)
    
    for fname in files:
        fpath = Path(root) / fname
        ext = fpath.suffix.lower()
        
        if ext not in CODE_EXTS and ext not in DOC_EXTS and ext not in CONFIG_EXTS:
            continue
            
        try:
            size = fpath.stat().st_size
        except:
            continue
        
        total, code, blank, comment = count_lines(fpath)
        
        # By extension
        results['by_extension'][ext]['files'] += 1
        results['by_extension'][ext]['total_lines'] += total
        results['by_extension'][ext]['code_lines'] += code
        results['by_extension'][ext]['blank_lines'] += blank
        results['by_extension'][ext]['bytes'] += size
        
        # By top-level directory
        parts = rel_root.parts
        top_dir = parts[0] if parts else '(root)'
        results['by_top_dir'][top_dir]['files'] += 1
        results['by_top_dir'][top_dir]['total_lines'] += total
        results['by_top_dir'][top_dir]['code_lines'] += code
        results['by_top_dir'][top_dir]['bytes'] += size
        
        # tensornet submodule breakdown
        if parts and parts[0] == 'tensornet' and len(parts) >= 2:
            submod = parts[1]
            results['tensornet_modules'][submod]['files'] += 1
            results['tensornet_modules'][submod]['total_lines'] += total
            results['tensornet_modules'][submod]['code_lines'] += code
            results['tensornet_modules'][submod]['bytes'] += size
        
        # Track largest
        all_files.append((str(rel_root / fname), total, code, size))
        
        # Totals
        results['totals']['files'] += 1
        results['totals']['total_lines'] += total
        results['totals']['code_lines'] += code
        results['totals']['blank_lines'] += blank
        results['totals']['bytes'] += size

# Sort largest files
all_files.sort(key=lambda x: x[1], reverse=True)
results['largest_files'] = all_files[:50]

# Convert defaultdicts
results['by_extension'] = dict(results['by_extension'])
results['by_top_dir'] = dict(results['by_top_dir'])
results['tensornet_modules'] = dict(results['tensornet_modules'])

# Print report
print("=" * 80)
print("HYPERTENSOR-VM FORENSIC LOC AUDIT")
print("=" * 80)

print(f"\n{'GRAND TOTALS':=^80}")
t = results['totals']
print(f"  Files:       {t['files']:,}")
print(f"  Total Lines: {t['total_lines']:,}")
print(f"  Code Lines:  {t['code_lines']:,}")
print(f"  Blank Lines: {t['blank_lines']:,}")
print(f"  Total Size:  {t['bytes']/1024/1024:.1f} MB")

print(f"\n{'BY FILE EXTENSION (sorted by total lines)':=^80}")
exts = sorted(results['by_extension'].items(), key=lambda x: x[1]['total_lines'], reverse=True)
print(f"  {'Ext':<10} {'Files':>7} {'Total Lines':>12} {'Code Lines':>12} {'Size MB':>10}")
print(f"  {'-'*10} {'-'*7} {'-'*12} {'-'*12} {'-'*10}")
for ext, data in exts:
    print(f"  {ext:<10} {data['files']:>7,} {data['total_lines']:>12,} {data['code_lines']:>12,} {data['bytes']/1024/1024:>10.2f}")

print(f"\n{'BY TOP-LEVEL DIRECTORY (sorted by total lines)':=^80}")
dirs_sorted = sorted(results['by_top_dir'].items(), key=lambda x: x[1]['total_lines'], reverse=True)
print(f"  {'Directory':<30} {'Files':>7} {'Total Lines':>12} {'Code Lines':>12} {'Size MB':>10}")
print(f"  {'-'*30} {'-'*7} {'-'*12} {'-'*12} {'-'*10}")
for d, data in dirs_sorted:
    print(f"  {d:<30} {data['files']:>7,} {data['total_lines']:>12,} {data['code_lines']:>12,} {data['bytes']/1024/1024:>10.2f}")

print(f"\n{'TENSORNET SUBMODULES (sorted by total lines)':=^80}")
mods = sorted(results['tensornet_modules'].items(), key=lambda x: x[1]['total_lines'], reverse=True)
print(f"  {'Module':<25} {'Files':>7} {'Total Lines':>12} {'Code Lines':>12} {'Size MB':>10}")
print(f"  {'-'*25} {'-'*7} {'-'*12} {'-'*12} {'-'*10}")
for m, data in mods:
    print(f"  {m:<25} {data['files']:>7,} {data['total_lines']:>12,} {data['code_lines']:>12,} {data['bytes']/1024/1024:>10.2f}")

total_tn = sum(d['total_lines'] for d in results['tensornet_modules'].values())
total_tn_files = sum(d['files'] for d in results['tensornet_modules'].values())
print(f"  {'TENSORNET TOTAL':<25} {total_tn_files:>7,} {total_tn:>12,}")

print(f"\n{'TOP 50 LARGEST FILES':=^80}")
print(f"  {'#':>3} {'Total':>7} {'Code':>7} {'File'}")
print(f"  {'-'*3} {'-'*7} {'-'*7} {'-'*60}")
for i, (path, total, code, size) in enumerate(results['largest_files'][:50], 1):
    print(f"  {i:>3} {total:>7,} {code:>7,} {path}")

# Language breakdown
print(f"\n{'LANGUAGE BREAKDOWN':=^80}")
py_lines = results['by_extension'].get('.py', {}).get('total_lines', 0)
rs_lines = results['by_extension'].get('.rs', {}).get('total_lines', 0)
ts_lines = sum(results['by_extension'].get(e, {}).get('total_lines', 0) for e in ['.ts', '.tsx'])
js_lines = sum(results['by_extension'].get(e, {}).get('total_lines', 0) for e in ['.js', '.jsx'])
sol_lines = results['by_extension'].get('.sol', {}).get('total_lines', 0)
md_lines = results['by_extension'].get('.md', {}).get('total_lines', 0)
sh_lines = sum(results['by_extension'].get(e, {}).get('total_lines', 0) for e in ['.sh', '.bash'])
other = t['total_lines'] - py_lines - rs_lines - ts_lines - js_lines - sol_lines - md_lines - sh_lines

print(f"  Python:     {py_lines:>10,} lines ({100*py_lines/max(t['total_lines'],1):.1f}%)")
print(f"  Rust:       {rs_lines:>10,} lines ({100*rs_lines/max(t['total_lines'],1):.1f}%)")
print(f"  TypeScript: {ts_lines:>10,} lines ({100*ts_lines/max(t['total_lines'],1):.1f}%)")
print(f"  JavaScript: {js_lines:>10,} lines ({100*js_lines/max(t['total_lines'],1):.1f}%)")
print(f"  Solidity:   {sol_lines:>10,} lines ({100*sol_lines/max(t['total_lines'],1):.1f}%)")
print(f"  Markdown:   {md_lines:>10,} lines ({100*md_lines/max(t['total_lines'],1):.1f}%)")
print(f"  Shell:      {sh_lines:>10,} lines ({100*sh_lines/max(t['total_lines'],1):.1f}%)")
print(f"  Other:      {other:>10,} lines ({100*other/max(t['total_lines'],1):.1f}%)")

# Dump JSON for further analysis
with open(REPO / 'loc_audit_results.json', 'w') as f:
    json.dump({
        'totals': results['totals'],
        'by_extension': {k: dict(v) for k, v in results['by_extension'].items()},
        'by_top_dir': {k: dict(v) for k, v in results['by_top_dir'].items()},
        'tensornet_modules': {k: dict(v) for k, v in results['tensornet_modules'].items()},
        'top50': results['largest_files'][:50]
    }, f, indent=2)

print(f"\nJSON results saved to loc_audit_results.json")
