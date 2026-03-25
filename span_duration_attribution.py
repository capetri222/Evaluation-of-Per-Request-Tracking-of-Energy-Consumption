#!/usr/bin/env python3
"""
Span-Duration Attribution Method
=================================
COMPARISON method with known under-attribution issue.

Attribution Logic:
    For each span:
        1. Find all intervals that overlap with span
        2. For each overlapping interval:
            - Calculate power consumed in that interval (E_interval / interval_duration)
            - Calculate span's portion: power * (overlap_duration / span_duration) * (span_duration / measurement_window_duration)
            - E_span += interval_contribution
    
Problem (Under-Attribution):
    - Spans that are shorter than interval duration get proportionally less energy
    - Short spans during high-energy intervals are under-attributed
    - Long idle periods between spans create unattributed energy
    
This method is included for comparison to show why interval-duration is superior.

Usage:
    python3 span_duration_attribution.py preprocessed_data/E3_20260320_012421
    python3 span_duration_attribution.py --input-dir preprocessed_data/E3_20260320_012421 --output-dir attributed_data
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "attributed_data"

# ══════════════════════════════════════════════════════════════════════
# SPAN-DURATION ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════

def attribute_span_duration(intervals: List[Dict], graph: Dict, measurement_window: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Perform span-duration attribution.
    
    For each span:
        1. Find overlapping intervals
        2. Calculate power in each interval (total_energy / interval_duration)
        3. Attribute proportionally: power * (overlap / span_duration) * (span_duration / window_duration)
    
    Returns:
        (span_energies, span_interval_mapping)
        
        span_energies: {
            (trace_id, span_id): {
                "energy_J": 0.123,
                "service": "frontend",
                "operation": "GET /",
                "duration_ms": 45.6,
                "contributions": [
                    {"interval": "2026-03-20T...", "energy_J": 0.001, "overlap_s": 0.5},
                    ...
                ]
            }
        }
        
        span_interval_mapping: [
            {
                "span_id": "abc123",
                "trace_id": "trace_001",
                "service": "frontend",
                "overlapping_intervals": [
                    {"timestamp": "...", "overlap_s": 0.5, "energy_J": 0.001},
                    ...
                ]
            },
            ...
        ]
    """
    # Parse measurement window
    start_time = datetime.fromisoformat(
        measurement_window["start"].replace('Z', '+00:00')
    ).timestamp()
    end_time = datetime.fromisoformat(
        measurement_window["end"].replace('Z', '+00:00')
    ).timestamp()
    window_duration = measurement_window["duration_s"]
    
    # Initialize span energy tracker
    span_energies = {}
    span_interval_mapping = []
    
    # Build interval lookup by timestamp
    interval_lookup = {}
    for interval in intervals:
        ts = datetime.fromisoformat(interval["timestamp"].replace('Z', '+00:00')).timestamp()
        interval_lookup[ts] = interval
    
    # Process each span
    for node in graph["nodes"]:
        trace_id = node["trace_id"]
        span_id = node["span_id"]
        service = node["service"]
        operation = node["operation"]
        
        # Span timing – node timestamps are already absolute Unix seconds
        # (set by preprocess: start_s = start_us / 1_000_000)
        span_start = node["start_time"]
        span_end   = node["end_time"]
        span_duration = span_end - span_start
        
        if span_duration <= 0:
            continue
        
        # Initialize span energy
        key = (trace_id, span_id)
        span_energies[key] = {
            "energy_J": 0.0,
            "service": service,
            "operation": operation,
            "duration_ms": node["duration_ms"],
            "contributions": []
        }
        
        overlapping_intervals = []
        
        # Find all overlapping intervals
        for interval in intervals:
            # Parse interval timing
            interval_ts = datetime.fromisoformat(
                interval["timestamp"].replace('Z', '+00:00')
            ).timestamp()
            interval_duration = interval["interval_duration_s"]
            interval_end = interval_ts + interval_duration
            
            # Check for overlap
            overlap_start = max(span_start, interval_ts)
            overlap_end = min(span_end, interval_end)
            
            if overlap_end <= overlap_start:
                continue  # No overlap
            
            overlap_duration = overlap_end - overlap_start
            
            # Get interval energy for this service
            service_data = interval.get("services", {}).get(service)
            
            if not service_data:
                continue  # No energy data for this service in this interval
            
            # Use request-induced energy (not total)
            interval_energy = service_data.get("induced_J", 0.0)
            
            if interval_energy <= 0:
                continue
            
            # Calculate power in this interval (W = J/s)
            interval_power = interval_energy / interval_duration
            
            # Span's portion of interval energy
            # Formula: power * (overlap / span_duration) * (span_duration / window_duration)
            # 
            # This creates under-attribution because:
            # - overlap / span_duration: reduces contribution if span is longer than overlap
            # - span_duration / window_duration: reduces contribution for short spans
            proportion = (overlap_duration / span_duration) * (span_duration / window_duration)
            span_contribution = interval_power * overlap_duration * proportion
            
            # Add to span energy
            span_energies[key]["energy_J"] += span_contribution
            span_energies[key]["contributions"].append({
                "interval": interval["timestamp"],
                "energy_J": span_contribution,
                "overlap_s": overlap_duration,
                "proportion": proportion
            })
            
            overlapping_intervals.append({
                "timestamp": interval["timestamp"],
                "overlap_s": overlap_duration,
                "energy_J": span_contribution,
                "interval_energy_J": interval_energy,
                "proportion": proportion
            })
        
        # Store span-interval mapping
        if overlapping_intervals:
            span_interval_mapping.append({
                "span_id": span_id,
                "trace_id": trace_id,
                "service": service,
                "operation": operation,
                "duration_s": span_duration,
                "overlapping_intervals": overlapping_intervals,
                "total_energy_J": span_energies[key]["energy_J"]
            })
    
    return span_energies, span_interval_mapping


# ══════════════════════════════════════════════════════════════════════
# TRACE-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def aggregate_trace_energies(span_energies: Dict, graph: Dict) -> List[Dict]:
    """Aggregate span energies to trace level."""
    # Group spans by trace
    traces_dict = defaultdict(lambda: {
        "total_energy_J": 0.0,
        "span_count": 0,
        "services": defaultdict(lambda: {"energy_J": 0.0, "span_count": 0})
    })
    
    for (trace_id, span_id), span_data in span_energies.items():
        traces_dict[trace_id]["total_energy_J"] += span_data["energy_J"]
        traces_dict[trace_id]["span_count"] += 1
        
        service = span_data["service"]
        traces_dict[trace_id]["services"][service]["energy_J"] += span_data["energy_J"]
        traces_dict[trace_id]["services"][service]["span_count"] += 1
    
    # Add trace metadata
    trace_metadata = {t["trace_id"]: t for t in graph["traces"]}
    
    traces = []
    for trace_id, data in traces_dict.items():
        metadata = trace_metadata.get(trace_id, {})
        
        # Find root service
        root_span_id = metadata.get("root_span_id")
        root_service = "unknown"
        
        for node in graph["nodes"]:
            if node["trace_id"] == trace_id and node["span_id"] == root_span_id:
                root_service = node["service"]
                break
        
        traces.append({
            "trace_id": trace_id,
            "total_energy_J": data["total_energy_J"],
            "span_count": data["span_count"],
            "root_service": root_service,
            "duration_ms": metadata.get("duration_ms", 0.0),
            "services": {s: dict(d) for s, d in data["services"].items()}
        })
    
    return sorted(traces, key=lambda t: t["total_energy_J"], reverse=True)


# ══════════════════════════════════════════════════════════════════════
# STATISTICS & COMPARISON WITH INTERVAL METHOD
# ══════════════════════════════════════════════════════════════════════

def compute_attribution_statistics(
    span_energies: Dict,
    trace_energies: List[Dict],
    metadata: Dict
) -> Dict:
    """Compute statistics and highlight under-attribution."""
    
    # Total energies
    total_measured = metadata["statistics"]["total_energy_J"]
    total_idle = metadata["statistics"]["idle_energy_J"]
    total_request_induced = metadata["statistics"]["request_induced_J"]
    
    # Attribution results
    total_attributed = sum(s["energy_J"] for s in span_energies.values())
    total_unattributed = total_request_induced - total_attributed
    
    # Under-attribution rate
    under_attribution_rate = total_unattributed / total_request_induced if total_request_induced > 0 else 0
    
    # Per-service statistics
    service_stats = defaultdict(lambda: {
        "attributed_J": 0.0,
        "span_count": 0
    })
    
    for (trace_id, span_id), span_data in span_energies.items():
        service = span_data["service"]
        service_stats[service]["attributed_J"] += span_data["energy_J"]
        service_stats[service]["span_count"] += 1
    
    # Span energy distribution
    span_energy_values = [s["energy_J"] for s in span_energies.values() if s["energy_J"] > 0]
    
    if span_energy_values:
        span_energy_values_sorted = sorted(span_energy_values)
        n = len(span_energy_values_sorted)
        
        span_energy_stats = {
            "mean": sum(span_energy_values) / n,
            "median": span_energy_values_sorted[n // 2],
            "min": span_energy_values_sorted[0],
            "max": span_energy_values_sorted[-1],
            "p95": span_energy_values_sorted[int(n * 0.95)],
            "p99": span_energy_values_sorted[int(n * 0.99)]
        }
    else:
        span_energy_stats = {
            "mean": 0, "median": 0, "min": 0, "max": 0, "p95": 0, "p99": 0
        }
    
    # Trace energy distribution
    trace_energy_values = [t["total_energy_J"] for t in trace_energies if t["total_energy_J"] > 0]
    
    if trace_energy_values:
        trace_energy_values_sorted = sorted(trace_energy_values)
        n = len(trace_energy_values_sorted)
        
        trace_energy_stats = {
            "mean": sum(trace_energy_values) / n,
            "median": trace_energy_values_sorted[n // 2],
            "min": trace_energy_values_sorted[0],
            "max": trace_energy_values_sorted[-1],
            "p95": trace_energy_values_sorted[int(n * 0.95)],
            "p99": trace_energy_values_sorted[int(n * 0.99)]
        }
    else:
        trace_energy_stats = {
            "mean": 0, "median": 0, "min": 0, "max": 0, "p95": 0, "p99": 0
        }
    
    return {
        "energy_budget": {
            "total_measured_J": total_measured,
            "idle_J": total_idle,
            "request_induced_J": total_request_induced,
            "attributed_J": total_attributed,
            "unattributed_J": total_unattributed
        },
        "attribution_quality": {
            "attribution_rate": total_attributed / total_request_induced if total_request_induced > 0 else 0,
            "under_attribution_rate": under_attribution_rate,
            "note": "EXPECTED: Under-attribution due to method design (span_duration / window_duration factor)"
        },
        "per_service": {s: dict(d) for s, d in service_stats.items()},
        "span_energy_distribution": span_energy_stats,
        "trace_energy_distribution": trace_energy_stats,
        "counts": {
            "spans": len(span_energies),
            "traces": len(trace_energies),
            "spans_with_energy": len([s for s in span_energies.values() if s["energy_J"] > 0])
        }
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN ATTRIBUTION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_span_duration_attribution(input_dir: str, output_dir: str) -> None:
    """
    Run span-duration attribution pipeline.
    
    Input: preprocessed_data/{experiment_name}/
        - graph_structure.json
        - interval_timeseries.json
        - metadata.json
    
    Output: attributed_data/{experiment_name}/
        - span_energies_span_method.json
        - trace_energies_span_method.json
        - span_interval_mapping.json
        - attribution_statistics_span_method.json
    """
    input_path = Path(input_dir)
    experiment_name = input_path.name
    
    print("=" * 70)
    print("SPAN-DURATION ATTRIBUTION (Comparison Method)")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}/{experiment_name}")
    print(f"\n⚠️  Note: This method produces UNDER-ATTRIBUTION by design")
    print(f"    Used for comparison with interval-duration method")
    
    # Load preprocessed data
    print(f"\n[1/4] Loading preprocessed data...")
    
    with open(input_path / "metadata.json") as f:
        metadata = json.load(f)
    
    with open(input_path / "graph_structure.json") as f:
        graph = json.load(f)
    
    with open(input_path / "interval_timeseries.json") as f:
        intervals = json.load(f)
    
    print(f"  ✓ Loaded {len(intervals)} intervals")
    print(f"  ✓ Loaded {len(graph['nodes'])} spans")
    print(f"  ✓ Total request-induced energy: {metadata['statistics']['request_induced_J']:.2f} J")
    
    # Run attribution
    print(f"\n[2/4] Running span-duration attribution...")
    
    span_energies, span_interval_mapping = attribute_span_duration(
        intervals,
        graph,
        metadata["measurement_window"]
    )
    
    total_attributed = sum(s["energy_J"] for s in span_energies.values())
    total_request_induced = metadata["statistics"]["request_induced_J"]
    under_attribution = total_request_induced - total_attributed
    
    print(f"  ✓ Attributed {total_attributed:.2f} J to {len(span_energies)} spans")
    print(f"  ⚠️  Under-attributed: {under_attribution:.2f} J ({under_attribution/total_request_induced*100:.1f}%)")
    
    # Aggregate to trace level
    print(f"\n[3/4] Aggregating to trace level...")
    
    trace_energies = aggregate_trace_energies(span_energies, graph)
    
    print(f"  ✓ Aggregated {len(trace_energies)} traces")
    
    # Compute statistics
    print(f"\n[4/4] Computing statistics...")
    
    statistics = compute_attribution_statistics(
        span_energies,
        trace_energies,
        metadata
    )
    
    print(f"  ✓ Attribution rate: {statistics['attribution_quality']['attribution_rate']*100:.1f}%")
    print(f"  ⚠️  Under-attribution: {statistics['attribution_quality']['under_attribution_rate']*100:.1f}%")
    
    # Save outputs
    print(f"\n[5/5] Saving results...")
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save span energies
    with open(output_path / "span_energies_span_method.json", 'w') as f:
        json.dump({
            "method": "span_duration",
            "note": "COMPARISON METHOD - produces under-attribution",
            "spans": {f"{k[0]}_{k[1]}": v for k, v in span_energies.items()}
        }, f, indent=2)
    print(f"  ✓ Saved span_energies_span_method.json")
    
    # Save trace energies
    with open(output_path / "trace_energies_span_method.json", 'w') as f:
        json.dump({
            "method": "span_duration",
            "note": "COMPARISON METHOD - produces under-attribution",
            "traces": trace_energies
        }, f, indent=2)
    print(f"  ✓ Saved trace_energies_span_method.json")
    
    # Save span-interval mapping
    with open(output_path / "span_interval_mapping.json", 'w') as f:
        json.dump({
            "method": "span_duration",
            "mappings": span_interval_mapping
        }, f, indent=2)
    print(f"  ✓ Saved span_interval_mapping.json")
    
    # Save statistics
    with open(output_path / "attribution_statistics_span_method.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"  ✓ Saved attribution_statistics_span_method.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ SPAN-DURATION ATTRIBUTION COMPLETE")
    print("=" * 70)
    
    print(f"\nEnergy Budget:")
    print(f"  Total Measured:      {statistics['energy_budget']['total_measured_J']:>10.2f} J")
    print(f"  Idle:                {statistics['energy_budget']['idle_J']:>10.2f} J")
    print(f"  Request-Induced:     {statistics['energy_budget']['request_induced_J']:>10.2f} J")
    print(f"  Attributed:          {statistics['energy_budget']['attributed_J']:>10.2f} J")
    print(f"  Unattributed:        {statistics['energy_budget']['unattributed_J']:>10.2f} J")
    
    print(f"\n⚠️  Attribution Quality (EXPECTED UNDER-ATTRIBUTION):")
    print(f"  Attribution Rate:    {statistics['attribution_quality']['attribution_rate']*100:>9.1f} %")
    print(f"  Under-Attribution:   {statistics['attribution_quality']['under_attribution_rate']*100:>9.1f} %")
    
    print(f"\nSpan Energy Distribution:")
    print(f"  Mean:   {statistics['span_energy_distribution']['mean']*1000:>8.3f} mJ")
    print(f"  Median: {statistics['span_energy_distribution']['median']*1000:>8.3f} mJ")
    print(f"  P95:    {statistics['span_energy_distribution']['p95']*1000:>8.3f} mJ")
    print(f"  P99:    {statistics['span_energy_distribution']['p99']*1000:>8.3f} mJ")
    print(f"  Max:    {statistics['span_energy_distribution']['max']*1000:>8.3f} mJ")
    
    print(f"\nTrace Energy Distribution:")
    print(f"  Mean:   {statistics['trace_energy_distribution']['mean']:>8.3f} J")
    print(f"  Median: {statistics['trace_energy_distribution']['median']:>8.3f} J")
    print(f"  P95:    {statistics['trace_energy_distribution']['p95']:>8.3f} J")
    print(f"  P99:    {statistics['trace_energy_distribution']['p99']:>8.3f} J")
    print(f"  Max:    {statistics['trace_energy_distribution']['max']:>8.3f} J")
    
    print(f"\nTop 5 Energy-Intensive Traces:")
    for i, trace in enumerate(trace_energies[:5], 1):
        print(f"  {i}. {trace['trace_id'][:16]}... : {trace['total_energy_J']*1000:>8.2f} mJ ({trace['span_count']} spans)")
    
    print(f"\n💡 Compare with interval-duration method to see under-attribution effect")
    print(f"   Output: {output_path}")
    print()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Span-duration attribution (comparison method with under-attribution)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        help="Preprocessed data directory (e.g., preprocessed_data/E3_20260320_012421)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        run_span_duration_attribution(args.input_dir, args.output_dir)
        return 0
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
