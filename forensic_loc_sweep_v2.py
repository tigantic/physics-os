#!/usr/bin/env python3
"""
HYPERTENSOR-VM FORENSIC LOC SWEEP v2 — AUTHORED CODE ONLY
Excludes vendored, third-party, and generated data directories.
Usage: python3 forensic_loc_sweep_v2.py > loc_authored_report.txt
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.resolve()

# ============================================================
# SKIP: infrastructure/cache dirs (never source code)
# ============================================================
SKIP_DIRS = {
    '__pycache__', '.git', 'target', 'node_modules', '.mypy_cache',
    '.pytest_cache', '.cache', '.cargo', 'dist', 'build', '.eggs',
    'venv', '.venv', 'env', '.tox', '.nox', 'site-packages',
    '.nox', '.tox', '.ruff_cache',
}

# ============================================================
# EXCLUDE: vendored / third-party top-level dirs
# ============================================================
VENDORED_DIRS = {
    'zk_targets',           # Vendored ZK protocol repos
    'vendor',               # Vendored dependencies
    'clawdbot-main',        # Third-party chatbot
    'worldcoin-id',         # Vendored Worldcoin reference
    'openzeppelin-contracts', # Vendored OZ (may appear nested)
}

# Path-based vendored exclusions (matched against full relative path)
VENDORED_PATHS = {
    '.lake/packages',       # Lean mathlib vendored deps
    'foundry/lib/forge-std', # Foundry standard lib
}

# ============================================================
# EXCLUDE: non-source data/generated
# ============================================================
SKIP_EXTENSIONS = {
    '.lock', '.json', '.csv', '.tsv', '.parquet', '.npy', '.npz',
    '.pkl', '.pickle', '.h5', '.hdf5', '.db', '.sqlite',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
    '.pdf', '.docx', '.xlsx', '.pptx',
    '.woff', '.woff2', '.ttf', '.eot',
    '.mp4', '.mp3', '.wav', '.avi', '.mov',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.bin', '.dat', '.raw', '.so', '.dylib', '.dll', '.exe',
    '.pyc', '.pyo', '.class', '.o', '.obj',
    '.map', '.min.js', '.min.css',
}

# Extension -> language mapping (source code only)
EXT_MAP = {
    '.py': 'Python',
    '.rs': 'Rust',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript',
    '.sol': 'Solidity',
    '.circom': 'Circom',
    '.wgsl': 'WGSL Shader',
    '.glsl': 'GLSL Shader',
    '.lean': 'Lean 4',
    '.md': 'Markdown',
    '.html': 'HTML',
    '.css': 'CSS',
    '.toml': 'TOML',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.sh': 'Shell',
    '.bash': 'Shell',
    '.cu': 'CUDA',
    '.cuh': 'CUDA',
    '.dockerfile': 'Docker',
    '.txt': 'Text',
    '.cfg': 'Config',
    '.ini': 'Config',
}

def count_lines(filepath):
    """Count total and blank lines."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return 0, 0
    total = len(lines)
    blank = sum(1 for l in lines if l.strip() == '')
    return total, blank


def get_module_path(filepath, root):
    """Get the top-level module."""
    rel = filepath.relative_to(root)
    parts = rel.parts
    if len(parts) == 1:
        return '_root'
    top = parts[0]
    if top == 'tensornet' and len(parts) > 2:
        return f"tensornet/{parts[1]}"
    return top


def is_vendored(filepath, root):
    """Check if file is in a vendored directory."""
    rel = filepath.relative_to(root)
    rel_str = str(rel)
    parts = rel.parts
    # Check directory name matches
    for part in parts:
        if part in VENDORED_DIRS:
            return True
    # Check path-based matches
    for vpath in VENDORED_PATHS:
        if vpath in rel_str:
            return True
    return False


def main():
    print("=" * 110)
    print("HYPERTENSOR-VM FORENSIC LOC SWEEP v2 — AUTHORED CODE ONLY")
    print(f"Root: {ROOT}")
    print("=" * 110)
    print(f"\nExcluded vendored dirs: {', '.join(sorted(VENDORED_DIRS))}")
    print(f"Excluded data extensions: {', '.join(sorted(SKIP_EXTENSIONS)[:10])}... (+{len(SKIP_EXTENSIONS)-10} more)")

    # Accumulators
    lang_stats = defaultdict(lambda: {'files': 0, 'total': 0, 'blank': 0, 'code': 0, 'bytes': 0})
    module_stats = defaultdict(lambda: defaultdict(lambda: {'files': 0, 'total': 0, 'blank': 0, 'code': 0, 'bytes': 0}))
    top_files = []
    
    vendored_lines = 0
    vendored_files = 0
    skipped_ext = 0
    skipped_binary = 0

    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Prune infrastructure dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        
        for fname in filenames:
            filepath = Path(dirpath) / fname
            ext = filepath.suffix.lower()
            
            # Skip data/binary extensions
            if ext in SKIP_EXTENSIONS:
                skipped_ext += 1
                continue
            
            if ext not in EXT_MAP:
                if fname == 'Dockerfile':
                    ext = '.dockerfile'
                elif fname == 'Makefile':
                    continue
                else:
                    skipped_binary += 1
                    continue
            
            # Check vendored
            if is_vendored(filepath, ROOT):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        lc = sum(1 for _ in f)
                    vendored_lines += lc
                    vendored_files += 1
                except:
                    pass
                continue
            
            lang = EXT_MAP.get(ext, 'Other')
            module = get_module_path(filepath, ROOT)
            
            try:
                size = filepath.stat().st_size
            except OSError:
                continue
            
            # Skip very large files (likely generated)
            if size > 2_000_000:  # 2MB
                skipped_binary += 1
                continue
            
            total, blank = count_lines(filepath)
            code = total - blank  # code = total minus blank (no comment stripping)
            
            lang_stats[lang]['files'] += 1
            lang_stats[lang]['total'] += total
            lang_stats[lang]['blank'] += blank
            lang_stats[lang]['code'] += code
            lang_stats[lang]['bytes'] += size
            
            module_stats[module][lang]['files'] += 1
            module_stats[module][lang]['total'] += total
            module_stats[module][lang]['blank'] += blank
            module_stats[module][lang]['code'] += code
            module_stats[module][lang]['bytes'] += size
            
            if total > 200:
                top_files.append((total, code, size, str(filepath.relative_to(ROOT))))

    # ============================================================
    # SECTION 1: LANGUAGE BREAKDOWN
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 1: AUTHORED CODE — LANGUAGE BREAKDOWN")
    print(f"{'─'*110}")
    print(f"{'Language':<20} {'Files':>7} {'Total Lines':>12} {'Blank':>10} {'Code':>10} {'Size MB':>10} {'%':>7}")
    print(f"{'─'*20} {'─'*7} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*7}")
    
    grand_total = sum(v['total'] for v in lang_stats.values())
    grand_code = sum(v['code'] for v in lang_stats.values())
    grand_files = sum(v['files'] for v in lang_stats.values())
    grand_bytes = sum(v['bytes'] for v in lang_stats.values())
    
    for lang, stats in sorted(lang_stats.items(), key=lambda x: -x[1]['total']):
        pct = 100 * stats['total'] / grand_total if grand_total else 0
        mb = stats['bytes'] / (1024 * 1024)
        print(f"{lang:<20} {stats['files']:>7,} {stats['total']:>12,} {stats['blank']:>10,} {stats['code']:>10,} {mb:>10.2f} {pct:>6.1f}%")
    
    print(f"{'─'*20} {'─'*7} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*7}")
    gb = sum(v['blank'] for v in lang_stats.values())
    print(f"{'TOTAL':<20} {grand_files:>7,} {grand_total:>12,} {gb:>10,} {grand_code:>10,} {grand_bytes/(1024*1024):>10.2f} {'100%':>7}")
    
    # ============================================================
    # SECTION 2: TOP-LEVEL DIRECTORY BREAKDOWN
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 2: AUTHORED CODE — DIRECTORY BREAKDOWN")
    print(f"{'─'*110}")
    
    dir_totals = {}
    for mod, langs in module_stats.items():
        top = mod.split('/')[0] if '/' in mod else mod
        if top not in dir_totals:
            dir_totals[top] = {'files': 0, 'total': 0, 'code': 0, 'bytes': 0, 'langs': defaultdict(int)}
        for lang, stats in langs.items():
            dir_totals[top]['files'] += stats['files']
            dir_totals[top]['total'] += stats['total']
            dir_totals[top]['code'] += stats['code']
            dir_totals[top]['bytes'] += stats['bytes']
            dir_totals[top]['langs'][lang] += stats['total']
    
    print(f"{'Directory':<32} {'Files':>7} {'Total Lines':>12} {'Code':>10} {'Size MB':>8} {'Primary':>15}")
    print(f"{'─'*32} {'─'*7} {'─'*12} {'─'*10} {'─'*8} {'─'*15}")
    
    for d, stats in sorted(dir_totals.items(), key=lambda x: -x[1]['total']):
        if stats['total'] == 0:
            continue
        primary = max(stats['langs'].items(), key=lambda x: x[1])[0] if stats['langs'] else '?'
        mb = stats['bytes'] / (1024 * 1024)
        print(f"{d:<32} {stats['files']:>7,} {stats['total']:>12,} {stats['code']:>10,} {mb:>8.2f} {primary:>15}")

    # ============================================================
    # SECTION 3: TENSORNET MODULE BREAKDOWN
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 3: TENSORNET/ MODULE BREAKDOWN")
    print(f"{'─'*110}")
    
    tn_modules = {}
    for mod, langs in module_stats.items():
        if not mod.startswith('tensornet/'):
            continue
        submod = mod.split('/', 1)[1]
        all_total = sum(s['total'] for s in langs.values())
        all_code = sum(s['code'] for s in langs.values())
        all_files = sum(s['files'] for s in langs.values())
        tn_modules[submod] = {'files': all_files, 'total': all_total, 'code': all_code}
    
    print(f"{'Module':<28} {'Files':>6} {'Total Lines':>12} {'Code':>10}")
    print(f"{'─'*28} {'─'*6} {'─'*12} {'─'*10}")
    
    tn_total = 0
    tn_code = 0
    tn_files = 0
    for mod, stats in sorted(tn_modules.items(), key=lambda x: -x[1]['total'])[:40]:
        print(f"{mod:<28} {stats['files']:>6} {stats['total']:>12,} {stats['code']:>10,}")
        tn_total += stats['total']
        tn_code += stats['code']
        tn_files += stats['files']
    
    remaining = sorted(tn_modules.items(), key=lambda x: -x[1]['total'])[40:]
    if remaining:
        rt = sum(s['total'] for _, s in remaining)
        rc = sum(s['code'] for _, s in remaining)
        rf = sum(s['files'] for _, s in remaining)
        tn_total += rt
        tn_code += rc
        tn_files += rf
        print(f"{'  ... + ' + str(len(remaining)) + ' more':<28} {rf:>6} {rt:>12,} {rc:>10,}")
    
    print(f"{'─'*28} {'─'*6} {'─'*12} {'─'*10}")
    print(f"{'TENSORNET TOTAL':<28} {tn_files:>6} {tn_total:>12,} {tn_code:>10,}")

    # ============================================================
    # SECTION 4: TOP 30 FILES
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 4: TOP 30 LARGEST AUTHORED FILES")
    print(f"{'─'*110}")
    top_files.sort(reverse=True)
    for i, (total, code, size, path) in enumerate(top_files[:30], 1):
        print(f"{i:>3}. {total:>7,} lines  {size/1024:>7.1f} KB  {path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*110}")
    print(f"EXECUTIVE SUMMARY — AUTHORED CODE ONLY")
    print(f"{'='*110}")
    print(f"  Authored source files:       {grand_files:>10,}")
    print(f"  Authored total lines:        {grand_total:>10,}")
    print(f"  Authored code (non-blank):   {grand_code:>10,}")
    print(f"  Authored size on disk:       {grand_bytes/(1024*1024):>10.2f} MB")
    print(f"  Vendored files excluded:     {vendored_files:>10,}")
    print(f"  Vendored lines excluded:     {vendored_lines:>10,}")
    print(f"  Data/binary files skipped:   {skipped_ext:>10,}")
    print(f"  Other non-source skipped:    {skipped_binary:>10,}")
    print(f"  tensornet/ modules:          {len(tn_modules):>10}")
    print(f"  Languages detected:          {len(lang_stats):>10}")

    # Per-language summary
    print(f"\n  AUTHORED LINES BY LANGUAGE:")
    for lang, stats in sorted(lang_stats.items(), key=lambda x: -x[1]['total']):
        if stats['total'] > 1000:
            print(f"    {lang:<20} {stats['total']:>10,} lines  ({stats['files']:,} files)")

    print(f"\n{'='*110}")


if __name__ == '__main__':
    main()
