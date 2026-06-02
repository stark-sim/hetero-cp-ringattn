#!/usr/bin/env python3
"""Parse HCP Ring Attention worker logs and generate performance summary."""
import sys
import re
from pathlib import Path

def parse_log(log_path):
    """Extract per-layer recv/compute stats from worker log."""
    rounds = []
    current_round = {}
    
    with open(log_path) as f:
        for line in f:
            # Match: round X layer Y: received micro_block N/M, Z bytes
            m = re.search(r'round (\d+) layer (\d+): received micro_block .*?(\d+) bytes', line)
            if m:
                r, layer, bytes_val = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if r not in current_round:
                    current_round[r] = {}
                current_round[r][layer] = {'bytes': bytes_val}
            
            # Match: layer Y Phase 2 summary: N micro blocks, avg recv=Xms, avg compute=Yms, recv/compute=Zx
            m = re.search(r'layer (\d+) Phase 2 summary: (\d+) micro blocks, avg recv=([\d.]+)ms, avg compute=([\d.]+)ms, recv/compute=([\d.]+)x', line)
            if m:
                layer = int(m.group(1))
                # Find which round this layer belongs to
                for r in sorted(current_round.keys(), reverse=True):
                    if layer in current_round[r]:
                        current_round[r][layer].update({
                            'recv_ms': float(m.group(3)),
                            'compute_ms': float(m.group(4)),
                            'ratio': float(m.group(5)),
                        })
                        break
    
    return current_round

def summarize(rounds):
    """Compute summary statistics across all rounds and layers."""
    all_recv = []
    all_compute = []
    all_ratio = []
    all_bytes = []
    
    for r, layers in rounds.items():
        for layer, stats in layers.items():
            if 'recv_ms' in stats:
                all_recv.append(stats['recv_ms'])
                all_compute.append(stats['compute_ms'])
                all_ratio.append(stats['ratio'])
                all_bytes.append(stats.get('bytes', 0))
    
    if not all_recv:
        return None
    
    return {
        'rounds': len(rounds),
        'layers_per_round': len(rounds[0]) if 0 in rounds else 0,
        'avg_recv_ms': sum(all_recv) / len(all_recv),
        'min_recv_ms': min(all_recv),
        'max_recv_ms': max(all_recv),
        'avg_compute_ms': sum(all_compute) / len(all_compute),
        'min_compute_ms': min(all_compute),
        'max_compute_ms': max(all_compute),
        'avg_ratio': sum(all_ratio) / len(all_ratio),
        'min_ratio': min(all_ratio),
        'max_ratio': max(all_ratio),
        'avg_bytes': sum(all_bytes) / len(all_bytes) if all_bytes else 0,
    }

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <worker_log_file>")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    rounds = parse_log(log_path)
    stats = summarize(rounds)
    
    if not stats:
        print("No performance data found in log.")
        sys.exit(1)
    
    print(f"File: {log_path}")
    print(f"Rounds: {stats['rounds']}")
    print(f"Layers per round: {stats['layers_per_round']}")
    print(f"Avg KV block: {stats['avg_bytes'] / 1024 / 1024:.2f} MB")
    print(f"")
    print(f"Recv time:   avg={stats['avg_recv_ms']:.2f}ms, min={stats['min_recv_ms']:.2f}ms, max={stats['max_recv_ms']:.2f}ms")
    print(f"Compute time: avg={stats['avg_compute_ms']:.2f}ms, min={stats['min_compute_ms']:.2f}ms, max={stats['max_compute_ms']:.2f}ms")
    print(f"Ratio:        avg={stats['avg_ratio']:.2f}x, min={stats['min_ratio']:.2f}x, max={stats['max_ratio']:.2f}x")

if __name__ == '__main__':
    main()
