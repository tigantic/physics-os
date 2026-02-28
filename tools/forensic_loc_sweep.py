#!/usr/bin/env python3
"""
HYPERTENSOR-VM FORENSIC LOC SWEEP
Runs a single-pass walk of the entire repo and outputs a full metric matrix.
Usage: python3 forensic_loc_sweep.py > loc_report.txt
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.resolve()

# Directories to skip entirely
SKIP_DIRS = {
    '__pycache__', '.git', 'target', 'node_modules', '.mypy_cache',
    '.pytest_cache', '.cache', '.cargo', 'dist', 'build', '.eggs',
    'venv', '.venv', 'env', '.tox', '.nox', 'site-packages',
    'openzeppelin-contracts',  # vendored dependency
}

# Extension -> language mapping
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
    '.json': 'JSON',
    '.sh': 'Shell',
    '.bash': 'Shell',
    '.cu': 'CUDA',
    '.cuh': 'CUDA',
    '.ptx': 'CUDA PTX',
    '.dockerfile': 'Docker',
    '.txt': 'Text',
    '.cfg': 'Config',
    '.ini': 'Config',
    '.lock': 'Lock',
}

def count_lines(filepath):
    """Count non-empty, non-comment lines (approximate SLOC)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return 0, 0, 0
    
    total = len(lines)
    blank = sum(1 for l in lines if l.strip() == '')
    # Rough comment detection
    comment = 0
    in_block = False
    for l in lines:
        s = l.strip()
        if not s:
            continue
        # Python/Shell comments
        if s.startswith('#') and filepath.suffix in ('.py', '.sh', '.bash', '.yaml', '.yml', '.toml'):
            comment += 1
        # Rust/JS/TS/Sol/Circom comments
        elif s.startswith('//') and filepath.suffix in ('.rs', '.ts', '.tsx', '.js', '.jsx', '.sol', '.circom', '.wgsl', '.glsl', '.cu', '.lean'):
            comment += 1
        # Block comments (approximate)
        elif '"""' in s or "'''" in s:
            in_block = not in_block
            comment += 1
        elif in_block:
            comment += 1
    
    sloc = total - blank - comment
    return total, blank, max(0, sloc)


def get_module_path(filepath, root):
    """Get the top-level module (first 2 path components relative to root)."""
    rel = filepath.relative_to(root)
    parts = rel.parts
    if len(parts) == 1:
        return '_root'
    
    top = parts[0]
    # For ontic, go one level deeper
    if top == 'ontic' and len(parts) > 2:
        return f"ontic/{parts[1]}"
    return top


def main():
    print("=" * 110)
    print("HYPERTENSOR-VM FULL FORENSIC LOC MATRIX")
    print(f"Root: {ROOT}")
    print("=" * 110)
    
    # Accumulators
    lang_stats = defaultdict(lambda: {'files': 0, 'total': 0, 'blank': 0, 'sloc': 0, 'bytes': 0})
    module_stats = defaultdict(lambda: defaultdict(lambda: {'files': 0, 'total': 0, 'blank': 0, 'sloc': 0, 'bytes': 0}))
    top_files = []  # (sloc, total, bytes, path)
    
    file_count = 0
    skipped_binary = 0
    
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Prune skip dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        
        for fname in filenames:
            filepath = Path(dirpath) / fname
            ext = filepath.suffix.lower()
            
            if ext not in EXT_MAP:
                # Check for Dockerfile (no extension)
                if fname == 'Dockerfile':
                    ext = '.dockerfile'
                else:
                    skipped_binary += 1
                    continue
            
            lang = EXT_MAP.get(ext, 'Other')
            module = get_module_path(filepath, ROOT)
            
            try:
                size = filepath.stat().st_size
            except OSError:
                continue
            
            # Skip very large files (likely generated/vendored)
            if size > 5_000_000:  # 5MB
                skipped_binary += 1
                continue
            
            total, blank, sloc = count_lines(filepath)
            file_count += 1
            
            lang_stats[lang]['files'] += 1
            lang_stats[lang]['total'] += total
            lang_stats[lang]['blank'] += blank
            lang_stats[lang]['sloc'] += sloc
            lang_stats[lang]['bytes'] += size
            
            module_stats[module][lang]['files'] += 1
            module_stats[module][lang]['total'] += total
            module_stats[module][lang]['blank'] += blank
            module_stats[module][lang]['sloc'] += sloc
            module_stats[module][lang]['bytes'] += size
            
            if sloc > 200:
                top_files.append((sloc, total, size, str(filepath.relative_to(ROOT))))
    
    # ============================================================
    # SECTION 1: LANGUAGE BREAKDOWN
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 1: LANGUAGE BREAKDOWN")
    print(f"{'─'*110}")
    print(f"{'Language':<20} {'Files':>7} {'Total Lines':>12} {'Blank':>10} {'SLOC':>10} {'Size MB':>10} {'% SLOC':>8}")
    print(f"{'─'*20} {'─'*7} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    
    grand_sloc = sum(v['sloc'] for v in lang_stats.values())
    grand_total = sum(v['total'] for v in lang_stats.values())
    grand_files = sum(v['files'] for v in lang_stats.values())
    grand_bytes = sum(v['bytes'] for v in lang_stats.values())
    
    for lang, stats in sorted(lang_stats.items(), key=lambda x: -x[1]['sloc']):
        pct = 100 * stats['sloc'] / grand_sloc if grand_sloc else 0
        mb = stats['bytes'] / (1024 * 1024)
        print(f"{lang:<20} {stats['files']:>7,} {stats['total']:>12,} {stats['blank']:>10,} {stats['sloc']:>10,} {mb:>10.2f} {pct:>7.1f}%")
    
    print(f"{'─'*20} {'─'*7} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    print(f"{'TOTAL':<20} {grand_files:>7,} {grand_total:>12,} {sum(v['blank'] for v in lang_stats.values()):>10,} {grand_sloc:>10,} {grand_bytes/(1024*1024):>10.2f} {'100.0%':>8}")
    
    # ============================================================
    # SECTION 2: TOP-LEVEL DIRECTORY BREAKDOWN
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 2: TOP-LEVEL DIRECTORY BREAKDOWN")
    print(f"{'─'*110}")
    
    dir_totals = {}
    for mod, langs in module_stats.items():
        top = mod.split('/')[0] if '/' in mod else mod
        if top not in dir_totals:
            dir_totals[top] = {'files': 0, 'sloc': 0, 'total': 0, 'bytes': 0, 'langs': defaultdict(int)}
        for lang, stats in langs.items():
            dir_totals[top]['files'] += stats['files']
            dir_totals[top]['sloc'] += stats['sloc']
            dir_totals[top]['total'] += stats['total']
            dir_totals[top]['bytes'] += stats['bytes']
            dir_totals[top]['langs'][lang] += stats['sloc']
    
    print(f"{'Directory':<28} {'Files':>7} {'SLOC':>10} {'Total':>10} {'Size MB':>10} {'Primary Lang':>20}")
    print(f"{'─'*28} {'─'*7} {'─'*10} {'─'*10} {'─'*10} {'─'*20}")
    
    for d, stats in sorted(dir_totals.items(), key=lambda x: -x[1]['sloc']):
        if stats['sloc'] == 0:
            continue
        primary = max(stats['langs'].items(), key=lambda x: x[1])[0] if stats['langs'] else '?'
        mb = stats['bytes'] / (1024 * 1024)
        print(f"{d:<28} {stats['files']:>7,} {stats['sloc']:>10,} {stats['total']:>10,} {mb:>10.2f} {primary:>20}")
    
    # ============================================================
    # SECTION 3: TENSORNET MODULE BREAKDOWN  
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 3: TENSORNET/ MODULE BREAKDOWN (Python SLOC)")
    print(f"{'─'*110}")
    
    tn_modules = {}
    for mod, langs in module_stats.items():
        if not mod.startswith('ontic/'):
            continue
        submod = mod.split('/', 1)[1]
        py_stats = langs.get('Python', {'files': 0, 'sloc': 0, 'total': 0})
        all_sloc = sum(s['sloc'] for s in langs.values())
        all_files = sum(s['files'] for s in langs.values())
        tn_modules[submod] = {
            'files': all_files,
            'py_sloc': py_stats['sloc'],
            'all_sloc': all_sloc,
            'total': sum(s['total'] for s in langs.values()),
        }
    
    print(f"{'Module':<28} {'Files':>6} {'Py SLOC':>9} {'All SLOC':>9} {'Total Ln':>9}")
    print(f"{'─'*28} {'─'*6} {'─'*9} {'─'*9} {'─'*9}")
    
    tn_total_sloc = 0
    tn_total_files = 0
    for mod, stats in sorted(tn_modules.items(), key=lambda x: -x[1]['all_sloc'])[:40]:
        print(f"{mod:<28} {stats['files']:>6} {stats['py_sloc']:>9,} {stats['all_sloc']:>9,} {stats['total']:>9,}")
        tn_total_sloc += stats['all_sloc']
        tn_total_files += stats['files']
    
    remaining = [(m, s) for m, s in sorted(tn_modules.items(), key=lambda x: -x[1]['all_sloc'])[40:]]
    if remaining:
        rem_sloc = sum(s['all_sloc'] for _, s in remaining)
        rem_files = sum(s['files'] for _, s in remaining)
        tn_total_sloc += rem_sloc
        tn_total_files += rem_files
        print(f"{'  ... + ' + str(len(remaining)) + ' more':<28} {rem_files:>6} {'':>9} {rem_sloc:>9,}")
    
    print(f"{'─'*28} {'─'*6} {'─'*9} {'─'*9}")
    print(f"{'TENSORNET TOTAL':<28} {tn_total_files:>6} {'':>9} {tn_total_sloc:>9,}")
    
    # ============================================================
    # SECTION 4: DOMAIN CATEGORY ROLLUPS
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 4: DOMAIN CATEGORY ROLLUPS")
    print(f"{'─'*110}")
    
    categories = {
        "Fluid Dynamics & CFD": ["cfd", "multiphase", "free_surface", "porous_media"],
        "Quantum & Condensed Matter": ["condensed_matter", "quantum", "quantum_mechanics", "qm", "qft", "algorithms", "mps"],
        "Classical Physics": ["plasma", "fusion", "nuclear", "relativity", "em", "optics", "astro", "mechanics", "statmech", "geophysics", "particle", "acoustics", "heat_transfer", "radiation"],
        "Engineering & Applied": ["guidance", "simulation", "defense", "racing", "certification", "manufacturing", "fsi", "energy", "urban", "emergency", "autonomy", "coordination", "robotics_physics", "flight_validation", "digital_twin"],
        "Computational Core": ["core", "qtt", "mpo", "numerics", "adaptive", "gpu", "cuda", "distributed_tn", "distributed", "realtime", "substrate", "fieldops", "mesh_amr"],
        "Platform & Infrastructure": ["platform", "packs", "sdk", "integration", "validation", "provenance", "benchmarks", "types", "deployment"],
        "AI/ML": ["ml_surrogates", "ml_physics", "neural", "hyperenv", "hypersim"],
        "Life Sciences": ["biology", "biomedical", "biophysics", "medical", "membrane_bio", "chemistry", "electronic_structure", "materials", "phase_field"],
        "Blockchain & Security": ["exploit", "oracle", "zk", "cyber", "financial", "discovery"],
        "Visualization & UI": ["visualization", "hypervisual", "sovereign", "gateway", "site", "docs", "fieldos", "intent", "hardware", "hw"],
        "Other Applied": ["multiscale", "environmental", "agri", "semiconductor", "special_applied", "data", "computational_methods", "fuel", "genesis", "shaders"],
    }
    
    cat_rows = []
    for cat, mods in categories.items():
        cat_sloc = sum(tn_modules.get(m, {}).get('all_sloc', 0) for m in mods)
        cat_files = sum(tn_modules.get(m, {}).get('files', 0) for m in mods)
        cat_rows.append((cat, cat_files, cat_sloc))
    
    cat_rows.sort(key=lambda x: -x[2])
    for cat, files, sloc in cat_rows:
        pct = 100 * sloc / tn_total_sloc if tn_total_sloc else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"{cat:<32} {sloc:>8,} SLOC ({pct:>5.1f}%)  {bar}")
    
    # ============================================================
    # SECTION 5: TOP 30 LARGEST FILES
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 5: TOP 30 LARGEST FILES BY SLOC")
    print(f"{'─'*110}")
    
    top_files.sort(reverse=True)
    for i, (sloc, total, size, path) in enumerate(top_files[:30], 1):
        print(f"{i:>3}. {sloc:>6,} SLOC  {total:>7,} total  {size/1024:>7.1f} KB  {path}")
    
    # ============================================================
    # SECTION 6: RUST CRATE BREAKDOWN
    # ============================================================
    print(f"\n{'─'*110}")
    print(f"SECTION 6: RUST CRATE / APP BREAKDOWN")
    print(f"{'─'*110}")
    
    rust_dirs = {}
    for mod, langs in module_stats.items():
        rs = langs.get('Rust', None)
        if not rs or rs['files'] == 0:
            continue
        top = mod.split('/')[0] if '/' in mod else mod
        if top not in rust_dirs:
            rust_dirs[top] = {'files': 0, 'sloc': 0, 'total': 0}
        rust_dirs[top]['files'] += rs['files']
        rust_dirs[top]['sloc'] += rs['sloc']
        rust_dirs[top]['total'] += rs['total']
    
    print(f"{'Crate/Directory':<28} {'Files':>7} {'SLOC':>10} {'Total':>10}")
    print(f"{'─'*28} {'─'*7} {'─'*10} {'─'*10}")
    for d, stats in sorted(rust_dirs.items(), key=lambda x: -x[1]['sloc']):
        print(f"{d:<28} {stats['files']:>7,} {stats['sloc']:>10,} {stats['total']:>10,}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*110}")
    print(f"EXECUTIVE SUMMARY")
    print(f"{'='*110}")
    print(f"Total source files:    {grand_files:>8,}")
    print(f"Total lines (raw):     {grand_total:>8,}")
    print(f"Total SLOC:            {grand_sloc:>8,}")
    print(f"Total size:            {grand_bytes/(1024*1024):>8.2f} MB")
    print(f"Skipped (binary/large):{skipped_binary:>8,}")
    print(f"ontic/ modules:    {len(tn_modules):>8}")
    print(f"Languages detected:    {len(lang_stats):>8}")
    
    py_sloc = lang_stats.get('Python', {}).get('sloc', 0)
    rs_sloc = lang_stats.get('Rust', {}).get('sloc', 0)
    print(f"\nPython SLOC:           {py_sloc:>8,}  ({100*py_sloc/grand_sloc:.1f}%)")
    print(f"Rust SLOC:             {rs_sloc:>8,}  ({100*rs_sloc/grand_sloc:.1f}%)")
    print(f"Other SLOC:            {grand_sloc - py_sloc - rs_sloc:>8,}  ({100*(grand_sloc-py_sloc-rs_sloc)/grand_sloc:.1f}%)")
    print(f"{'='*110}")


if __name__ == '__main__':
    main()
