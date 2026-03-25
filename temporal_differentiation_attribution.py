#!/usr/bin/env python3
"""
Temporal Differentiation Attribution Method
===========================================
ONLY for ISOLATED experiments (E1) where requests are sequential.

Attribution Logic:
    For isolated requests (N repetitions of same action):
        1. Measure total energy during execution window (with baselines)
        2. Subtract idle baseline from total energy
        3. Divide by number of requests: E_per_request = E_total / N
    
    This gives the AVERAGE energy per request type.
    
Advantages:
    - Simple and accurate for isolated scenarios
    - No need for trace-span mapping
    - Direct measurement of per-request energy
    
Limitations:
    - ONLY works for isolated/sequential requests
    - Cannot attribute to individual spans/traces
    - Cannot distinguish variability between requests
    - Not applicable to concurrent workloads (E2-E6)

Usage:
    python3 temporal_differentiation_attribution.py cluster_experiments/E1/E1_isolated_browse_N100_20260320_202929
    python3 temporal_differentiation_attribution.py --input-dir cluster_experiments/E1/... --output-dir attributed_data
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "attributed_data"

# ══════════════════════════════════════════════════════════════════════
# TEMPORAL DIFFERENTIATION ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════

def perform_temporal_differentiation(
    manifest: Dict,
    baseline_pre: Dict,
    baseline_post: Dict,
    kepler_metrics_path: str
) -> Dict:
    """
    Perform temporal differentiation for isolated requests.
    
    Steps:
        1. Calculate idle baseline power (average of pre and post)
        2. Calculate total energy during execution (from Kepler metrics)
        3. Calculate request-induced energy (total - idle)
        4. Divide by number of requests: E_per_request = E_request_induced / N
    
    Returns:
        {
            "action": "browse",
            "repetitions": 100,
            "total_energy_J": 123.45,
            "idle_energy_J": 45.67,
            "request_induced_J": 77.78,
            "energy_per_request_J": 0.7778,
            "energy_per_request_mJ": 777.8,
            "per_service": {
                "frontend": {
                    "total_J": 10.0,
                    "idle_J": 2.0,
                    "induced_J": 8.0,
                    "per_request_J": 0.08,
                    "per_request_mJ": 80.0
                },
                ...
            }
        }
    """
    # Get experiment parameters
    action = manifest["experiment"].get("action", "unknown")
    repetitions = manifest["experiment"].get("repetitions", 1)
    
    if repetitions <= 0:
        raise ValueError(f"Invalid repetitions: {repetitions}")
    
    # Calculate idle baseline per service
    pre_baselines = baseline_pre.get("baselines", {})
    post_baselines = baseline_post.get("baselines", {})
    
    idle_power_per_service = {}
    for service in pre_baselines.keys():
        pre = pre_baselines.get(service, 0.0)
        post = post_baselines.get(service, 0.0)
        idle_power_per_service[service] = (pre + post) / 2.0
    
    total_idle_power = sum(idle_power_per_service.values())
    
    # Calculate execution duration
    start_time = datetime.fromisoformat(
        manifest["execution"]["start_time"].replace('Z', '+00:00')
    )
    end_time = datetime.fromisoformat(
        manifest["execution"]["end_time"].replace('Z', '+00:00')
    )
    execution_duration = (end_time - start_time).total_seconds()
    
    # Calculate idle energy during execution
    idle_energy_total = total_idle_power * execution_duration
    idle_energy_per_service = {
        service: power * execution_duration
        for service, power in idle_power_per_service.items()
    }
    
    # Load Kepler metrics to get total energy
    import csv
    
    total_energy_per_service = {}
    
    with open(kepler_metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Build service -> zone -> [values] mapping
        service_metrics = {}
        
        for row in reader:
            service = row['service']
            zone = row['zone']
            joules = float(row['joules_total'])
            
            if service not in service_metrics:
                service_metrics[service] = {}
            if zone not in service_metrics[service]:
                service_metrics[service][zone] = []
            
            service_metrics[service][zone].append(joules)
    
    # Calculate energy delta (last - first) for each service
    for service, zones in service_metrics.items():
        total_delta = 0.0
        
        for zone, values in zones.items():
            if len(values) >= 2:
                # Energy delta = last - first (cumulative counter)
                delta = values[-1] - values[0]
                total_delta += delta
        
        total_energy_per_service[service] = total_delta
    
    total_energy = sum(total_energy_per_service.values())
    
    # Calculate request-induced energy
    request_induced_total = total_energy - idle_energy_total
    request_induced_per_service = {
        service: total_energy_per_service.get(service, 0.0) - idle_energy_per_service.get(service, 0.0)
        for service in total_energy_per_service.keys()
    }
    
    # Per-request energy
    energy_per_request = request_induced_total / repetitions
    energy_per_request_per_service = {
        service: induced / repetitions
        for service, induced in request_induced_per_service.items()
    }
    
    # Build result
    result = {
        "method": "temporal_differentiation",
        "action": action,
        "repetitions": repetitions,
        "execution_duration_s": execution_duration,
        "total_energy_J": total_energy,
        "idle_energy_J": idle_energy_total,
        "request_induced_J": request_induced_total,
        "energy_per_request_J": energy_per_request,
        "energy_per_request_mJ": energy_per_request * 1000,
        "per_service": {
            service: {
                "total_J": total_energy_per_service.get(service, 0.0),
                "idle_J": idle_energy_per_service.get(service, 0.0),
                "induced_J": request_induced_per_service.get(service, 0.0),
                "per_request_J": energy_per_request_per_service.get(service, 0.0),
                "per_request_mJ": energy_per_request_per_service.get(service, 0.0) * 1000
            }
            for service in total_energy_per_service.keys()
        }
    }
    
    return result


# ══════════════════════════════════════════════════════════════════════
# COMPARISON WITH OTHER METHODS
# ══════════════════════════════════════════════════════════════════════

def compare_with_attributed_results(
    temporal_result: Dict,
    attributed_data_dir: Path,
    experiment_name: str
) -> Dict:
    """
    Compare temporal differentiation with interval/span attribution methods.
    
    Returns comparison statistics if attributed data exists.
    """
    comparison = {
        "temporal_differentiation": {
            "total_request_induced_J": temporal_result["request_induced_J"],
            "per_request_mJ": temporal_result["energy_per_request_mJ"]
        }
    }
    
    # Check if interval-duration attribution exists
    interval_stats_path = attributed_data_dir / experiment_name / "attribution_statistics.json"
    if interval_stats_path.exists():
        with open(interval_stats_path) as f:
            interval_stats = json.load(f)
        
        # Calculate average per trace
        trace_count = interval_stats["counts"]["traces"]
        total_attributed = interval_stats["energy_budget"]["attributed_J"]
        
        if trace_count > 0:
            avg_per_trace = total_attributed / trace_count
            
            comparison["interval_duration_method"] = {
                "total_attributed_J": total_attributed,
                "per_trace_avg_mJ": avg_per_trace * 1000,
                "trace_count": trace_count,
                "attribution_rate": interval_stats["attribution_quality"]["attribution_rate"]
            }
            
            # Calculate difference
            temporal_per_req = temporal_result["energy_per_request_mJ"]
            interval_per_trace = avg_per_trace * 1000
            
            diff_mJ = temporal_per_req - interval_per_trace
            diff_pct = (diff_mJ / temporal_per_req * 100) if temporal_per_req > 0 else 0
            
            comparison["comparison"] = {
                "difference_mJ": diff_mJ,
                "difference_pct": diff_pct,
                "note": "Temporal = average, Interval = per-trace (may vary)"
            }
    
    # Check if span-duration attribution exists
    span_stats_path = attributed_data_dir / experiment_name / "attribution_statistics_span_method.json"
    if span_stats_path.exists():
        with open(span_stats_path) as f:
            span_stats = json.load(f)
        
        trace_count = span_stats["counts"]["traces"]
        total_attributed = span_stats["energy_budget"]["attributed_J"]
        
        if trace_count > 0:
            avg_per_trace = total_attributed / trace_count
            
            comparison["span_duration_method"] = {
                "total_attributed_J": total_attributed,
                "per_trace_avg_mJ": avg_per_trace * 1000,
                "trace_count": trace_count,
                "attribution_rate": span_stats["attribution_quality"]["attribution_rate"],
                "under_attribution_rate": span_stats["attribution_quality"].get("under_attribution_rate", 0)
            }
    
    return comparison


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_temporal_differentiation(input_dir: str, output_dir: str) -> None:
    """
    Run temporal differentiation attribution for isolated experiments.
    
    Input: Experiment directory (cluster_experiments/E1/...)
        - manifest.json
        - baseline_pre.json
        - baseline_post.json
        - kepler_metrics.csv
    
    Output: attributed_data/{experiment_name}/
        - temporal_differentiation.json
        - temporal_comparison.json (if other methods exist)
    """
    input_path = Path(input_dir)
    experiment_name = input_path.name
    
    print("=" * 70)
    print("TEMPORAL DIFFERENTIATION ATTRIBUTION")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}/{experiment_name}")
    
    # Validate this is an isolated experiment
    with open(input_path / "manifest.json") as f:
        manifest = json.load(f)
    
    exp_type = manifest["experiment"].get("type", "")
    
    if exp_type != "isolated":
        print(f"\n⚠️  Warning: This method is designed for 'isolated' experiments")
        print(f"   Current experiment type: '{exp_type}'")
        print(f"   Results may not be meaningful for concurrent workloads\n")
    
    # Load data
    print(f"\n[1/3] Loading experiment data...")
    
    with open(input_path / "baseline_pre.json") as f:
        baseline_pre = json.load(f)
    
    with open(input_path / "baseline_post.json") as f:
        baseline_post = json.load(f)
    
    action = manifest["experiment"].get("action", "unknown")
    repetitions = manifest["experiment"].get("repetitions", 1)
    
    print(f"  ✓ Action: {action}")
    print(f"  ✓ Repetitions: {repetitions}")
    print(f"  ✓ Baselines loaded")
    
    # Perform temporal differentiation
    print(f"\n[2/3] Performing temporal differentiation...")
    
    result = perform_temporal_differentiation(
        manifest,
        baseline_pre,
        baseline_post,
        str(input_path / "kepler_metrics.csv")
    )
    
    print(f"  ✓ Total request-induced energy: {result['request_induced_J']:.2f} J")
    print(f"  ✓ Energy per request: {result['energy_per_request_mJ']:.3f} mJ")
    
    # Compare with other methods if they exist
    print(f"\n[3/3] Checking for comparison data...")
    
    output_path = Path(output_dir)
    comparison = compare_with_attributed_results(result, output_path, experiment_name)
    
    if "interval_duration_method" in comparison:
        print(f"  ✓ Found interval-duration attribution for comparison")
    if "span_duration_method" in comparison:
        print(f"  ✓ Found span-duration attribution for comparison")
    
    # Save results
    print(f"\n[4/4] Saving results...")
    
    output_exp_path = output_path / experiment_name
    output_exp_path.mkdir(parents=True, exist_ok=True)
    
    # Save temporal differentiation result
    with open(output_exp_path / "temporal_differentiation.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved temporal_differentiation.json")
    
    # Save comparison if available
    if len(comparison) > 1:  # More than just temporal_differentiation
        with open(output_exp_path / "temporal_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"  ✓ Saved temporal_comparison.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ TEMPORAL DIFFERENTIATION COMPLETE")
    print("=" * 70)
    
    print(f"\nResults for '{action}' (N={repetitions}):")
    print(f"  Execution Duration:  {result['execution_duration_s']:>10.2f} s")
    print(f"  Total Energy:        {result['total_energy_J']:>10.2f} J")
    print(f"  Idle Energy:         {result['idle_energy_J']:>10.2f} J")
    print(f"  Request-Induced:     {result['request_induced_J']:>10.2f} J")
    print(f"\n  Energy per Request:  {result['energy_per_request_mJ']:>10.3f} mJ")
    
    print(f"\nTop 5 Services by Per-Request Energy:")
    services_sorted = sorted(
        result["per_service"].items(),
        key=lambda x: x[1]["per_request_mJ"],
        reverse=True
    )
    
    for i, (service, data) in enumerate(services_sorted[:5], 1):
        pct = (data["per_request_mJ"] / result["energy_per_request_mJ"] * 100) if result["energy_per_request_mJ"] > 0 else 0
        print(f"  {i}. {service:<20s} {data['per_request_mJ']:>8.3f} mJ ({pct:>5.1f}%)")
    
    # Print comparison if available
    if "comparison" in comparison:
        comp = comparison["comparison"]
        print(f"\n📊 Comparison with Interval-Duration Method:")
        print(f"  Temporal (avg):      {result['energy_per_request_mJ']:>10.3f} mJ")
        print(f"  Interval (avg):      {comparison['interval_duration_method']['per_trace_avg_mJ']:>10.3f} mJ")
        print(f"  Difference:          {comp['difference_mJ']:>+10.3f} mJ ({comp['difference_pct']:>+6.1f}%)")
        print(f"\n  Note: {comp['note']}")
    
    print(f"\nOutput: {output_exp_path}")
    print()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Temporal differentiation attribution for isolated experiments (E1)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        help="Experiment directory (e.g., cluster_experiments/E1/E1_isolated_browse_N100_20260320_202929)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        run_temporal_differentiation(args.input_dir, args.output_dir)
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
