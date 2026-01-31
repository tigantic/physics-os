#!/usr/bin/env python3
"""
QTT Command-Line Interface
==========================

Usage:
    # Create from text file
    qtt pack corpus.txt -o library.qtt
    
    # Create from JSON array
    qtt pack documents.json -o library.qtt
    
    # Query a container
    qtt query library.qtt "quantum physics"
    
    # Inspect a container
    qtt info library.qtt
    
    # Benchmark
    qtt bench library.qtt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List


def load_texts(path: str) -> List[str]:
    """Load texts from file (supports .txt, .json, .jsonl)."""
    path = Path(path)
    
    if path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict) and 'texts' in data:
                return data['texts']
            else:
                raise ValueError("JSON must be array or {texts: [...]}")
    
    elif path.suffix == '.jsonl':
        texts = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
        return texts
    
    else:  # Plain text: one document per line or paragraph
        with open(path) as f:
            content = f.read()
        
        # If file has blank line separators, split on them
        if '\n\n' in content:
            return [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            return [line.strip() for line in content.split('\n') if line.strip()]


def cmd_pack(args):
    """Pack texts into .qtt container."""
    from qtt import QTTContainer
    
    print(f"Loading texts from {args.input}...")
    texts = load_texts(args.input)
    print(f"Loaded {len(texts):,} documents")
    
    container = QTTContainer.from_text_corpus(
        texts,
        n_subvectors=args.subvectors,
        n_centroids=args.centroids,
        compress_payload=not args.no_compress,
        show_progress=True
    )
    
    container.save(args.output)


def cmd_query(args):
    """Query a .qtt container."""
    from qtt import QTTContainer
    
    with QTTContainer.open(args.container) as container:
        result = container.slice(query=args.query, top_k=args.top_k)
        
        print(f"\nQuery: '{args.query}'")
        print(f"Time: {result.access_time_ms:.1f}ms")
        print(f"\nResults:")
        
        for i, match in enumerate(result.matches):
            text = container.read_text(match)
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"\n{i+1}. [score={match.score:.3f}]")
            print(f"   {preview}")


def cmd_info(args):
    """Show container info."""
    from qtt import QTTContainer
    
    with QTTContainer.open(args.container) as container:
        print(container.info())
        
        # File stats
        path = Path(args.container)
        print(f"\nFile: {path}")
        print(f"Size: {path.stat().st_size:,} bytes")


def cmd_bench(args):
    """Benchmark container performance."""
    from qtt import QTTContainer
    
    queries = [
        "quantum physics",
        "machine learning",
        "climate change",
        "financial markets",
        "ancient history"
    ]
    
    with QTTContainer.open(args.container) as container:
        print(container.info())
        print("\n" + "=" * 50)
        print("BENCHMARK")
        print("=" * 50)
        
        # Warmup
        _ = container.slice(query="warmup")
        
        # Benchmark
        times = []
        for q in queries:
            start = time.perf_counter()
            result = container.slice(query=q, top_k=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  '{q}': {elapsed:.1f}ms")
        
        avg = sum(times) / len(times)
        print(f"\nAverage: {avg:.1f}ms")
        print(f"Throughput: {1000/avg:.0f} QPS")


def cmd_spatial(args):
    """Create spatial container from numpy file."""
    from qtt import QTTContainer
    import numpy as np
    
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    
    if isinstance(data, np.lib.npyio.NpzFile):
        # NPZ file - use first array
        key = list(data.keys())[0]
        data = data[key]
    
    print(f"Shape: {data.shape}, dtype: {data.dtype}")
    
    container = QTTContainer.from_spatial_data(
        data,
        max_rank=args.max_rank,
        show_progress=True
    )
    
    container.save(args.output)


def main():
    parser = argparse.ArgumentParser(
        description="QTT - Universal Random-Access File System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Pack command
    pack_parser = subparsers.add_parser('pack', help='Pack texts into .qtt')
    pack_parser.add_argument('input', help='Input file (.txt, .json, .jsonl)')
    pack_parser.add_argument('-o', '--output', required=True, help='Output .qtt file')
    pack_parser.add_argument('--subvectors', type=int, default=12, help='PQ subvectors')
    pack_parser.add_argument('--centroids', type=int, default=256, help='Centroids per subvector')
    pack_parser.add_argument('--no-compress', action='store_true', help='Disable payload compression')
    pack_parser.set_defaults(func=cmd_pack)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query a .qtt container')
    query_parser.add_argument('container', help='.qtt file')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results')
    query_parser.set_defaults(func=cmd_query)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show container info')
    info_parser.add_argument('container', help='.qtt file')
    info_parser.set_defaults(func=cmd_info)
    
    # Bench command
    bench_parser = subparsers.add_parser('bench', help='Benchmark performance')
    bench_parser.add_argument('container', help='.qtt file')
    bench_parser.set_defaults(func=cmd_bench)
    
    # Spatial command
    spatial_parser = subparsers.add_parser('spatial', help='Create from numpy array')
    spatial_parser.add_argument('input', help='Input .npy or .npz file')
    spatial_parser.add_argument('-o', '--output', required=True, help='Output .qtt file')
    spatial_parser.add_argument('--max-rank', type=int, default=64, help='Maximum TT rank')
    spatial_parser.set_defaults(func=cmd_spatial)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
