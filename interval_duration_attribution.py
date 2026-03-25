#!/usr/bin/env python3
"""
Interval-Duration Attribution Method
====================================
The MAIN attribution method from the paper.

Attribution Logic (per service, per interval):
    For each span in interval:
        E_span += E_interval_request_induced * (span_overlap / total_overlap)
    
    Where:
        - E_interval_request_induced = Total interval energy - Idle energy
        - span_overlap = Duration of span overlapping with interval
        - total_overlap = Sum of all span overlaps in this interval (per service)

This ensures:
    1. Request-induced energy is fully attributed to spans
    2. Idle energy remains unattributed
    3. Per-service attribution based on active span duration

Usage:
    python3 interval_duration_attribution.py preprocessed_data/E3_20260320_012421
    python3 interval_duration_attribution.py --input-dir preprocessed_data/E3_20260320_012421 --output-dir attributed_data
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "attributed_data"

# ══════════════════════════════════════════════════════════════════════
# INTERVAL-DURATION ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════

def attribute_interval_duration(intervals: List[Dict], graph: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Perform interval-duration attribution.
    
    For each interval:
        1. Get request-induced energy per service
        2. Get active spans per service
        3. Compute total overlap duration per service
        4. Distribute energy proportionally to span overlap
    
    Returns:
        (span_energies, interval_attributions)
        
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
        
        interval_attributions: [
            {
                "timestamp": "2026-03-20T...",
                "total_energy_J": 1.234,
                "idle_energy_J": 0.456,
                "request_induced_J": 0.778,
                "attributed_J": 0.778,
                "unattributed_J": 0.0,
                "services": {
                    "frontend": {
                        "total_J": 0.1,
                        "idle_J": 0.05,
                        "induced_J": 0.05,
                        "attributed_J": 0.05,
                        "span_count": 3,
                        "total_overlap_s": 0.8
                    },
                    ...
                }
            },
            ...
        ]
    """
    # Initialize span energy tracker
    span_energies = {}
    
    # Build span lookup by span_id
    span_lookup = {}
    for node in graph["nodes"]:
        key = (node["trace_id"], node["span_id"])
        span_lookup[key] = node
        span_energies[key] = {
            "energy_J": 0.0,
            "service": node["service"],
            "operation": node["operation"],
            "duration_ms": node["duration_ms"],
            "contributions": []
        }
    
    # Process each interval
    interval_attributions = []
    
    # Track statistics
    intervals_with_any_spans = 0
    intervals_without_any_spans = 0
    total_span_activations = 0
    
    # Per-service statistics
    from collections import defaultdict
    service_stats = defaultdict(lambda: {
        "intervals_without_spans": 0,
        "energy_without_spans_J": 0.0,
        "intervals_with_spans": 0,
        "energy_with_spans_J": 0.0,
        "energy_attributed_J": 0.0,
        "energy_unattributed_J": 0.0,
        "total_induced_J": 0.0  # NEW: Track total induced energy
    })
    
    # DEBUG: Track overall energy flow
    debug_total_induced = 0.0
    debug_total_attributed = 0.0
    debug_total_unattributed = 0.0
    
    for interval in intervals:
        timestamp = interval["timestamp"]
        interval_duration = interval["interval_duration_s"]
        
        # Check if this interval has ANY spans at all
        has_any_spans = len(interval.get("active_spans", [])) > 0
        
        if has_any_spans:
            intervals_with_any_spans += 1
            total_span_activations += len(interval.get("active_spans", []))
        else:
            intervals_without_any_spans += 1
        
        # Per-service attribution
        service_attribution = {}
        
        # Get services with energy in this interval
        services = interval.get("services", {})
        
        for service, energy_data in services.items():
            total_J = energy_data["total_J"]
            idle_J = energy_data["idle_J"]
            induced_J = energy_data["induced_J"]
            
            # DEBUG: Track all induced energy
            debug_total_induced += induced_J
            service_stats[service]["total_induced_J"] += induced_J
            
            # Get active spans for this service in this interval
            active_spans = [
                s for s in interval.get("active_spans", [])
                if s["service"] == service
            ]
            
            if not active_spans:
                # No active spans for THIS SERVICE in this interval
                service_stats[service]["intervals_without_spans"] += 1
                service_stats[service]["energy_without_spans_J"] += induced_J
                
                # DEBUG: This energy should be unattributed
                debug_total_unattributed += induced_J
                
                service_attribution[service] = {
                    "total_J": total_J,
                    "idle_J": idle_J,
                    "induced_J": induced_J,
                    "attributed_J": 0.0,
                    "unattributed_J": induced_J,
                    "span_count": 0,
                    "total_overlap_s": 0.0
                }
                continue
            
            # Calculate total overlap for this service
            total_overlap = sum(s["overlap_duration_s"] for s in active_spans)
            
            if total_overlap <= 0:
                # Edge case: spans present but no overlap (shouldn't happen)
                service_stats[service]["intervals_without_spans"] += 1
                service_stats[service]["energy_without_spans_J"] += induced_J
                
                # DEBUG: This energy should be unattributed
                debug_total_unattributed += induced_J
                
                service_attribution[service] = {
                    "total_J": total_J,
                    "idle_J": idle_J,
                    "induced_J": induced_J,
                    "attributed_J": 0.0,
                    "unattributed_J": induced_J,
                    "span_count": len(active_spans),
                    "total_overlap_s": 0.0
                }
                continue
            
            # Distribute request-induced energy proportionally
            attributed_energy = 0.0
            
            service_stats[service]["intervals_with_spans"] += 1
            service_stats[service]["energy_with_spans_J"] += induced_J
            
            for span in active_spans:
                span_overlap = span["overlap_duration_s"]
                
                # Proportional attribution
                span_energy = induced_J * (span_overlap / total_overlap)
                
                # Add to span's total energy
                key = (span["trace_id"], span["span_id"])
                if key in span_energies:
                    span_energies[key]["energy_J"] += span_energy
                    span_energies[key]["contributions"].append({
                        "interval": timestamp,
                        "energy_J": span_energy,
                        "overlap_s": span_overlap,
                        "attribution_factor": span_overlap / total_overlap
                    })
                    attributed_energy += span_energy
            
            # DEBUG: Track attribution
            debug_total_attributed += attributed_energy
            unattrib = max(0.0, induced_J - attributed_energy)
            debug_total_unattributed += unattrib
            
            service_stats[service]["energy_attributed_J"] += attributed_energy
            service_stats[service]["energy_unattributed_J"] += unattrib
            
            # Store service attribution
            service_attribution[service] = {
                "total_J": total_J,
                "idle_J": idle_J,
                "induced_J": induced_J,
                "attributed_J": attributed_energy,
                "unattributed_J": unattrib,
                "span_count": len(active_spans),
                "total_overlap_s": total_overlap
            }
        
        # Aggregate interval-level attribution
        interval_attr = {
            "timestamp": timestamp,
            "interval_duration_s": interval_duration,
            "total_energy_J": interval["total_energy_J"],
            "idle_energy_J": interval["idle_energy_J"],
            "request_induced_J": interval["request_induced_J"],
            "attributed_J": sum(s["attributed_J"] for s in service_attribution.values()),
            "unattributed_J": sum(s["unattributed_J"] for s in service_attribution.values()),
            "services": service_attribution
        }
        
        interval_attributions.append(interval_attr)
    
    # DEBUG: Print energy flow validation
    print(f"\n  🔍 DEBUG - Energy Flow Validation:")
    print(f"     Total induced (from intervals):  {debug_total_induced:.2f} J")
    print(f"     Total attributed (to spans):     {debug_total_attributed:.2f} J")
    print(f"     Total unattributed (tracked):    {debug_total_unattributed:.2f} J")
    print(f"     Check: {debug_total_attributed:.2f} + {debug_total_unattributed:.2f} = {debug_total_attributed + debug_total_unattributed:.2f} J")
    
    conservation_error = abs((debug_total_attributed + debug_total_unattributed) - debug_total_induced)
    if conservation_error > 0.01:
        print(f"     ⚠️  Conservation error: {conservation_error:.2f} J")
    else:
        print(f"     ✓ Conservation OK")
    
    # Print interval statistics
    total_intervals = len(intervals)
    print(f"\n  📊 Interval Statistics:")
    print(f"     Total intervals: {total_intervals}")
    print(f"     Intervals with spans: {intervals_with_any_spans}")
    print(f"     Intervals without spans: {intervals_without_any_spans}")
    print(f"     Total span activations: {total_span_activations}")
    if total_intervals > 0:
        print(f"     Coverage: {intervals_with_any_spans/total_intervals*100:.1f}% of intervals have spans")
    
    # Print per-service statistics (sorted by unattributed energy)
    print(f"\n  📊 Per-Service Attribution (Top 10 by unattributed energy):")
    print(f"     {'Service':<20s} {'Total Induced':<15s} {'Attributed':<15s} {'Unattributed':<15s}")
    print(f"     {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    
    service_list = sorted(
        service_stats.items(),
        key=lambda x: x[1]["energy_unattributed_J"],
        reverse=True
    )
    
    for service, stats in service_list[:10]:
        total_induced = stats["total_induced_J"]
        attributed = stats["energy_attributed_J"]
        unattrib = stats["energy_unattributed_J"]
        
        print(f"     {service:<20s} {total_induced:>12.2f} J {attributed:>12.2f} J {unattrib:>12.2f} J")
    
    if len(service_list) > 10:
        print(f"     ... ({len(service_list) - 10} more services)")
    
    # Validate per-service conservation
    print(f"\n  🔍 Per-Service Conservation Check:")
    service_conservation_errors = []
    for service, stats in service_stats.items():
        total_induced = stats["total_induced_J"]
        attributed = stats["energy_attributed_J"]
        unattrib = stats["energy_unattributed_J"]
        
        service_sum = attributed + unattrib
        error = abs(service_sum - total_induced)
        
        if error > 0.01:
            service_conservation_errors.append((service, total_induced, service_sum, error))
    
    if service_conservation_errors:
        print(f"     ⚠️  Found {len(service_conservation_errors)} services with conservation errors:")
        for service, induced, summed, error in service_conservation_errors[:5]:
            print(f"        {service}: induced={induced:.2f} J, attributed+unattrib={summed:.2f} J, error={error:.2f} J")
    else:
        print(f"     ✓ All services conserve energy correctly")
    
    return span_energies, interval_attributions


# ══════════════════════════════════════════════════════════════════════
# TRACE-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def aggregate_trace_energies(span_energies: Dict, graph: Dict) -> List[Dict]:
    """
    Aggregate span energies to trace level.
    
    Returns:
        [
            {
                "trace_id": "abc123",
                "total_energy_J": 0.456,
                "span_count": 10,
                "root_service": "frontend",
                "duration_ms": 123.4,
                "services": {
                    "frontend": {"energy_J": 0.1, "span_count": 3},
                    "cart": {"energy_J": 0.05, "span_count": 2},
                    ...
                }
            },
            ...
        ]
    """
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
            "user_action": metadata.get("user_action", "unknown"),
            "services": {s: dict(d) for s, d in data["services"].items()}
        })
    
    return sorted(traces, key=lambda t: t["total_energy_J"], reverse=True)


# ══════════════════════════════════════════════════════════════════════
# STATISTICS & VALIDATION
# ══════════════════════════════════════════════════════════════════════

def compute_attribution_statistics(
    span_energies: Dict,
    interval_attributions: List[Dict],
    trace_energies: List[Dict],
    metadata: Dict
) -> Dict:
    """Compute comprehensive attribution statistics."""
    
    # Total energies
    total_measured = metadata["statistics"]["total_energy_J"]
    total_idle = metadata["statistics"]["idle_energy_J"]
    total_request_induced = metadata["statistics"]["request_induced_J"]
    
    # Attribution results
    total_attributed = sum(s["energy_J"] for s in span_energies.values())
    total_unattributed = sum(i["unattributed_J"] for i in interval_attributions)
    
    # Validation: attributed + unattributed should equal request-induced
    attribution_sum = total_attributed + total_unattributed
    attribution_error = abs(attribution_sum - total_request_induced) / total_request_induced if total_request_induced > 0 else 0
    
    # Per-service statistics
    service_stats = defaultdict(lambda: {
        "attributed_J": 0.0,
        "unattributed_J": 0.0,
        "span_count": 0
    })
    
    for interval in interval_attributions:
        for service, data in interval["services"].items():
            service_stats[service]["attributed_J"] += data["attributed_J"]
            service_stats[service]["unattributed_J"] += data["unattributed_J"]
            service_stats[service]["span_count"] += data["span_count"]
    
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
    
    # Add aliases for plot compatibility
    per_service_out = {}
    for svc, d in service_stats.items():
        entry = dict(d)
        entry["total_induced_J"]    = entry["attributed_J"] + entry["unattributed_J"]
        entry["energy_attributed_J"] = entry["attributed_J"]
        per_service_out[svc] = entry

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
            "unattributed_rate": total_unattributed / total_request_induced if total_request_induced > 0 else 0,
            "conservation_error": attribution_error
        },
        "per_service": per_service_out,
        "span_energy_distribution": span_energy_stats,
        "trace_energy_distribution": trace_energy_stats,
        "counts": {
            "spans": len(span_energies),
            "traces": len(trace_energies),
            "intervals": len(interval_attributions),
            "spans_with_energy": len([s for s in span_energies.values() if s["energy_J"] > 0])
        }
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN ATTRIBUTION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_interval_duration_attribution(input_dir: str, output_dir: str) -> None:
    """
    Run interval-duration attribution pipeline.
    
    Input: preprocessed_data/{experiment_name}/
        - graph_structure.json
        - interval_timeseries.json
        - metadata.json
    
    Output: attributed_data/{experiment_name}/
        - span_energies.json
        - trace_energies.json
        - interval_attributions.json
        - attribution_statistics.json
    """
    input_path = Path(input_dir)
    experiment_name = input_path.name
    
    print("=" * 70)
    print("INTERVAL-DURATION ATTRIBUTION")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}/{experiment_name}")
    
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
    print(f"\n[2/4] Running interval-duration attribution...")
    
    span_energies, interval_attributions = attribute_interval_duration(intervals, graph)
    
    total_attributed = sum(s["energy_J"] for s in span_energies.values())
    print(f"  ✓ Attributed {total_attributed:.2f} J to {len(span_energies)} spans")
    
    # Aggregate to trace level
    print(f"\n[3/4] Aggregating to trace level...")
    
    trace_energies = aggregate_trace_energies(span_energies, graph)
    
    print(f"  ✓ Aggregated {len(trace_energies)} traces")
    
    # Compute statistics
    print(f"\n[4/4] Computing statistics...")
    
    statistics = compute_attribution_statistics(
        span_energies,
        interval_attributions,
        trace_energies,
        metadata
    )
    
    print(f"  ✓ Attribution rate: {statistics['attribution_quality']['attribution_rate']*100:.1f}%")
    print(f"  ✓ Unattributed: {statistics['attribution_quality']['unattributed_rate']*100:.1f}%")
    
    # Save outputs
    print(f"\n[5/5] Saving results...")
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save span energies
    with open(output_path / "span_energies.json", 'w') as f:
        json.dump({
            "method": "interval_duration",
            "spans": {f"{k[0]}_{k[1]}": v for k, v in span_energies.items()}
        }, f, indent=2)
    print(f"  ✓ Saved span_energies.json")
    
    # Save trace energies
    with open(output_path / "trace_energies.json", 'w') as f:
        json.dump({
            "method": "interval_duration",
            "traces": trace_energies
        }, f, indent=2)
    print(f"  ✓ Saved trace_energies.json")
    
    # Save interval attributions
    with open(output_path / "interval_attributions.json", 'w') as f:
        json.dump({
            "method": "interval_duration",
            "intervals": interval_attributions
        }, f, indent=2)
    print(f"  ✓ Saved interval_attributions.json")
    
    # Save statistics
    with open(output_path / "attribution_statistics.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"  ✓ Saved attribution_statistics.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ INTERVAL-DURATION ATTRIBUTION COMPLETE")
    print("=" * 70)
    
    print(f"\nEnergy Budget:")
    print(f"  Total Measured:      {statistics['energy_budget']['total_measured_J']:>10.2f} J")
    print(f"  Idle:                {statistics['energy_budget']['idle_J']:>10.2f} J")
    print(f"  Request-Induced:     {statistics['energy_budget']['request_induced_J']:>10.2f} J")
    print(f"  Attributed:          {statistics['energy_budget']['attributed_J']:>10.2f} J")
    print(f"  Unattributed:        {statistics['energy_budget']['unattributed_J']:>10.2f} J")
    
    print(f"\nAttribution Quality:")
    print(f"  Attribution Rate:    {statistics['attribution_quality']['attribution_rate']*100:>9.1f} %")
    print(f"  Unattributed Rate:   {statistics['attribution_quality']['unattributed_rate']*100:>9.1f} %")
    print(f"  Conservation Error:  {statistics['attribution_quality']['conservation_error']*100:>9.3f} %")
    
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
    
    print(f"\nOutput: {output_path}")
    print()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interval-duration attribution (main method)",
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
        run_interval_duration_attribution(args.input_dir, args.output_dir)
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
