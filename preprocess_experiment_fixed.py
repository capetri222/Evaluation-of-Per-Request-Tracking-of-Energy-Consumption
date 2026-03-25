#!/usr/bin/env python3
"""
Experiment Data Preprocessing - FIXED VERSION
==============================================
Prepares experiment data for energy attribution.

FIXES:
1. Uses baseline mean from JSON files (average of pre/post per service)
2. Handles zero-value metrics by using nearest non-zero neighbors
3. Classifies traces by user action (browse, cart, checkout, background, monitoring)

Input: cluster_experiments/{experiment_name}/
    - manifest.json
    - baseline_pre.json
    - baseline_post.json  
    - kepler_metrics.csv
    - raw_traces.json

Output: preprocessed_data/{experiment_name}/
    - graph_structure.json
    - interval_timeseries.json
    - metadata.json

Usage:
    python3 preprocess_experiment_fixed.py cluster_experiments/E3/E3_20260320_012421
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple, Optional

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "preprocessed_data"

TELEMETRY_CONFIGS = {
    "fine": {"scrape_interval": "1s", "interval_length_s": 1.0},
    "medium": {"scrape_interval": "10s", "interval_length_s": 10.0},
    "coarse": {"scrape_interval": "60s", "interval_length_s": 60.0}
}

# User action patterns for classification
USER_ACTION_PATTERNS = {
    "browse": ["/", "/product/", "browse", "BrowseProductCatalog"],
    "cart": ["/cart", "/api/cart", "AddToCart", "GetCart"],
    "checkout": ["/api/checkout", "/checkout", "PlaceOrder", "Checkout"],
    "view_cart": ["/cart", "GetCart", "ViewCart"]
}

MONITORING_SERVICES = {
    "jaeger", "opensearch", "opentelemetry-collector", "prometheus-server",
    "grafana", "loki"
}

# ══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def classify_trace_action(trace: Dict) -> str:
    """
    Classify trace by user action. Single source of truth for the pipeline.

    Priority: checkout > add_to_cart > view_cart > browse > background

    Key insight from data analysis: add_to_cart traces ALWAYS contain
    BOTH AddItem AND GetCart spans (cart is read after writing).
    Therefore HTTP method alone is unreliable. We rely on:
      1. Explicit Locust operation names: user_add_to_cart / user_view_cart
      2. Presence of AddItem span (only in add_to_cart)
      3. GetCart-only (no AddItem) → view_cart
    """
    spans     = trace.get("spans", [])
    processes = trace.get("processes", {})

    operations     = []
    http_targets   = []
    services_seen  = set()

    for span in spans:
        proc_id = span.get("processID", "")
        service = processes.get(proc_id, {}).get("serviceName", "unknown")
        services_seen.add(service)
        operations.append(span.get("operationName", ""))
        for tag in span.get("tags", []):
            if tag.get("key") == "http.target":
                http_targets.append(tag.get("value", ""))

    if services_seen & MONITORING_SERVICES:
        return "monitoring"

    ops_lower  = [o.lower() for o in operations]
    tgts_lower = [t.lower() for t in http_targets]
    all_text   = " ".join(ops_lower + tgts_lower)

    # ── 1. Checkout ───────────────────────────────────────────────────
    if any(p in all_text for p in ["checkout", "placeorder", "/api/checkout"]):
        return "checkout"

    # ── 2. Cart – use explicit Locust markers first, then AddItem ─────
    in_cart = ("/api/cart" in all_text or "/cart" in all_text
               or "cartservice" in all_text or "getcart" in all_text
               or "additem" in all_text)
    if in_cart:
        # Explicit Locust operation names are the most reliable signal
        if "user_add_to_cart" in all_text:
            return "add_to_cart"
        if "user_view_cart" in all_text:
            return "view_cart"
        # AddItem span present → add_to_cart (even if GetCart also present)
        has_additem = any("additem" in o for o in ops_lower)
        if has_additem:
            return "add_to_cart"
        # Only GetCart, no AddItem → view_cart
        return "view_cart"

    # ── 3. Browse ─────────────────────────────────────────────────────
    if any(p in all_text for p in ["browse", "product", "productcatalog",
                                    "recommendation", "/api/products",
                                    "catalogservice"]):
        return "browse"

    # "/" alone is too broad; use it only as last resort for frontend calls
    if any(o in ["/", "get /", "get"] for o in ops_lower):
        if "frontend" in services_seen or "frontend-proxy" in services_seen:
            return "browse"

    return "background"

def calculate_idle_baseline(baseline_pre: Dict, baseline_post: Dict) -> Tuple[Dict[str, float], float]:
    """
    Calculate idle baseline power per service.
    
    Simple average of pre and post baseline values (already aggregated in JSON).
    These are mean power values in Watts per service during idle periods.
    
    Returns: (idle_power_per_service, total_idle_power)
    """
    pre_baselines = baseline_pre.get("baselines", {})
    post_baselines = baseline_post.get("baselines", {})
    
    idle_power_per_service = {}
    
    # All services present in either baseline
    all_services = set(pre_baselines.keys()) | set(post_baselines.keys())
    
    for service in all_services:
        pre = pre_baselines.get(service, 0.0)
        post = post_baselines.get(service, 0.0)
        
        # Average of pre and post
        idle_power_per_service[service] = (pre + post) / 2.0
    
    total_idle_power = sum(idle_power_per_service.values())
    
    return idle_power_per_service, total_idle_power


# ══════════════════════════════════════════════════════════════════════
# KEPLER METRICS LOADING (WITH ZERO HANDLING)
# ══════════════════════════════════════════════════════════════════════

def load_kepler_metrics(csv_path: str) -> Dict:
    """
    Load Kepler metrics with zero-value handling.
    
    Returns: {timestamp: {service: {zone: joules}}}
    """
    metrics = defaultdict(lambda: defaultdict(dict))
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row['timestamp']
            service = row['service']
            zone = row['zone']
            joules = float(row['joules_total'])
            
            metrics[timestamp][service][zone] = joules
    
    return dict(metrics)


def find_nearest_nonzero(
    metrics: Dict,
    timestamps: List[str],
    target_idx: int,
    service: str,
    direction: str = "backward"
) -> Optional[float]:
    """
    Find nearest non-zero metric value.
    
    direction: "backward" (earlier) or "forward" (later)
    """
    if direction == "backward":
        indices = range(target_idx - 1, -1, -1)
    else:
        indices = range(target_idx + 1, len(timestamps))
    
    for idx in indices:
        ts = timestamps[idx]
        service_data = metrics.get(ts, {}).get(service, {})
        total = sum(service_data.values())
        
        if total > 0:
            return total
    
    return None


# ══════════════════════════════════════════════════════════════════════
# INTERVAL POWER COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_intervals_with_zero_handling(
    metrics: Dict,
    idle_power_per_service: Dict[str, float]
) -> List[Dict]:
    """
    Compute interval power with zero-value handling.
    
    For zero values, use nearest non-zero neighbor.
    """
    timestamps = sorted(metrics.keys())
    intervals = []
    
    for i in range(len(timestamps) - 1):
        ts_current = timestamps[i]
        ts_next = timestamps[i + 1]
        
        dt_current = datetime.fromisoformat(ts_current.replace('Z', '+00:00'))
        dt_next = datetime.fromisoformat(ts_next.replace('Z', '+00:00'))
        interval_duration = (dt_next - dt_current).total_seconds()
        
        if interval_duration <= 0:
            continue
        
        interval_data = {
            "timestamp": ts_current,
            "interval_duration_s": interval_duration,
            "total_energy_J": 0.0,
            "idle_energy_J": 0.0,
            "request_induced_J": 0.0,
            "services": {}
        }
        
        current_metrics = metrics[ts_current]
        next_metrics = metrics[ts_next]
        
        all_services = set(current_metrics.keys()) | set(next_metrics.keys())
        
        for service in all_services:
            # Get current and next values
            current_total = sum(current_metrics.get(service, {}).values())
            next_total = sum(next_metrics.get(service, {}).values())
            
            # Handle zeros
            if current_total == 0:
                fallback = find_nearest_nonzero(metrics, timestamps, i, service, "backward")
                if fallback is not None:
                    current_total = fallback
            
            if next_total == 0:
                fallback = find_nearest_nonzero(metrics, timestamps, i + 1, service, "forward")
                if fallback is not None:
                    next_total = fallback
            
            # Calculate energy delta
            energy_delta = next_total - current_total
            
            if energy_delta < 0:
                energy_delta = 0.0  # Counter reset or error
            
            # Idle energy
            idle_power = idle_power_per_service.get(service, 0.0)
            idle_energy = idle_power * interval_duration
            
            # Request-induced (can be negative if idle > total, but we cap at 0)
            request_induced = energy_delta - idle_energy
            
            # For conservation: if request_induced < 0, reduce idle to match total
            if request_induced < 0:
                # This means idle baseline was overestimated for this service/interval
                # Adjust: set idle = total, request = 0
                adjusted_idle = energy_delta
                adjusted_request = 0.0
            else:
                adjusted_idle = idle_energy
                adjusted_request = request_induced
            
            interval_data["services"][service] = {
                "total_J": energy_delta,
                "idle_J": adjusted_idle,
                "induced_J": adjusted_request
            }
            
            interval_data["total_energy_J"] += energy_delta
            interval_data["idle_energy_J"] += adjusted_idle
            interval_data["request_induced_J"] += adjusted_request
        
        intervals.append(interval_data)
    
    return intervals


# ══════════════════════════════════════════════════════════════════════
# SPAN GRAPH BUILDING (WITH USER ACTION CLASSIFICATION)
# ══════════════════════════════════════════════════════════════════════

def build_span_graph(traces: List[Dict]) -> Dict:
    """Build span graph with user action classification."""
    nodes = []
    edges = []
    trace_metadata = []
    
    for trace in traces:
        trace_id = trace.get("traceID", "")
        spans = trace.get("spans", [])
        
        if not spans:
            continue
        
        # Classify trace action
        user_action = classify_trace_action(trace)
        
        # Build process map
        process_map = {}
        for proc_id, proc_data in trace.get("processes", {}).items():
            process_map[proc_id] = proc_data.get("serviceName", "unknown")
        
        # Track trace bounds
        root_span_id = None
        min_start_time = float('inf')
        max_end_time = 0.0
        
        # Process spans
        for span in spans:
            span_id = span.get("spanID", "")
            process_id = span.get("processID", "")
            service = process_map.get(process_id, "unknown")
            operation = span.get("operationName", "")
            start_us = span.get("startTime", 0)
            duration_us = span.get("duration", 0)
            
            # Convert to seconds (absolute Unix timestamps)
            start_s = start_us / 1_000_000.0
            duration_s = duration_us / 1_000_000.0
            end_s = start_s + duration_s
            
            # Track bounds
            if start_s < min_start_time:
                min_start_time = start_s
                root_span_id = span_id
            
            max_end_time = max(max_end_time, end_s)
            
            # Create node
            nodes.append({
                "span_id": span_id,
                "trace_id": trace_id,
                "service": service,
                "operation": operation,
                "start_time": start_s,
                "end_time": end_s,
                "duration_ms": duration_s * 1000,
                "user_action": user_action  # NEW: user action classification
            })
            
            # Create edges
            for ref in span.get("references", []):
                if ref.get("refType") == "CHILD_OF":
                    parent_id = ref.get("spanID", "")
                    edges.append({
                        "from": parent_id,
                        "to": span_id,
                        "trace_id": trace_id
                    })
        
        # Store trace metadata
        trace_metadata.append({
            "trace_id": trace_id,
            "root_span_id": root_span_id,
            "start_time": min_start_time,
            "duration_ms": (max_end_time - min_start_time) * 1000,
            "span_count": len(spans),
            "user_action": user_action  # NEW
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "traces": trace_metadata
    }


# ══════════════════════════════════════════════════════════════════════
# SPAN-INTERVAL MAPPING
# ══════════════════════════════════════════════════════════════════════

def map_spans_to_intervals(intervals: List[Dict], graph: Dict) -> List[Dict]:
    """Map spans to intervals using absolute timestamps."""
    nodes = graph["nodes"]
    
    for interval in intervals:
        ts = datetime.fromisoformat(interval["timestamp"].replace('Z', '+00:00'))
        interval_start = ts.timestamp()
        interval_end = interval_start + interval["interval_duration_s"]
        
        active_spans = []
        
        for node in nodes:
            span_start = node["start_time"]
            span_end = node["end_time"]
            
            # Check overlap (absolute timestamps)
            overlap_start = max(interval_start, span_start)
            overlap_end = min(interval_end, span_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                
                active_spans.append({
                    "span_id": node["span_id"],
                    "trace_id": node["trace_id"],
                    "service": node["service"],
                    "operation": node["operation"],
                    "user_action": node["user_action"],  # NEW
                    "overlap_duration_s": overlap_duration
                })
        
        interval["active_spans"] = active_spans
        interval["active_span_count"] = len(active_spans)
        
        # Group by service and user action
        service_span_counts = defaultdict(int)
        service_total_overlap = defaultdict(float)
        action_span_counts = defaultdict(int)
        
        for span in active_spans:
            service = span["service"]
            action = span["user_action"]
            
            service_span_counts[service] += 1
            service_total_overlap[service] += span["overlap_duration_s"]
            action_span_counts[action] += 1
        
        interval["spans_per_service"] = dict(service_span_counts)
        interval["total_overlap_per_service"] = dict(service_total_overlap)
        interval["spans_per_action"] = dict(action_span_counts)  # NEW
    
    return intervals


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_preprocessing(input_dir: str, output_dir: str) -> None:
    """Run complete preprocessing pipeline with all fixes."""
    input_path = Path(input_dir)
    experiment_name = input_path.name
    
    print("=" * 70)
    print("EXPERIMENT DATA PREPROCESSING (FIXED)")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}/{experiment_name}\n")
    
    print("Fixes applied:")
    print("  ✓ Mean baseline from JSON (avg of pre/post)")
    print("  ✓ Zero-value handling (nearest neighbor)")
    print("  ✓ User action classification")
    
    # Check required files
    required_files = ["manifest.json", "baseline_pre.json", "baseline_post.json",
                      "kepler_metrics.csv", "raw_traces.json"]
    
    for fname in required_files:
        if not (input_path / fname).exists():
            raise FileNotFoundError(f"Missing required file: {fname}")
    
    print(f"\n✓ All required files present")
    
    # Load data
    print(f"\n[1/5] Loading experiment data...")
    
    with open(input_path / "manifest.json") as f:
        manifest = json.load(f)
    
    with open(input_path / "baseline_pre.json") as f:
        baseline_pre = json.load(f)
    
    with open(input_path / "baseline_post.json") as f:
        baseline_post = json.load(f)
    
    with open(input_path / "raw_traces.json") as f:
        trace_data = json.load(f)
        traces = trace_data.get("traces", [])
    
    telemetry = manifest["experiment"].get("telemetry", "fine")
    telemetry_config = TELEMETRY_CONFIGS.get(telemetry, TELEMETRY_CONFIGS["fine"])
    interval_length = telemetry_config["interval_length_s"]
    
    print(f"  ✓ Manifest: {manifest['experiment']['type']} experiment")
    print(f"  ✓ Telemetry: {telemetry} (interval: {interval_length}s)")
    print(f"  ✓ Baselines and traces loaded")
    
    # Calculate idle baseline
    print(f"\n[2/5] Calculating idle baseline...")
    
    idle_power_per_service, total_idle_power = calculate_idle_baseline(
        baseline_pre, baseline_post
    )
    
    print(f"  ✓ Total idle power (mean): {total_idle_power:.6f} W")
    print(f"  ✓ Idle power for {len(idle_power_per_service)} services")
    
    # Process intervals with zero handling
    print(f"\n[3/5] Processing Kepler metrics (with zero handling)...")
    
    metrics = load_kepler_metrics(str(input_path / "kepler_metrics.csv"))
    intervals = compute_intervals_with_zero_handling(metrics, idle_power_per_service)
    
    total_energy = sum(i["total_energy_J"] for i in intervals)
    total_idle = sum(i["idle_energy_J"] for i in intervals)
    total_request = sum(i["request_induced_J"] for i in intervals)
    
    print(f"  ✓ Loaded metrics for {len(metrics)} timestamps")
    print(f"  ✓ Computed {len(intervals)} intervals")
    print(f"\n  📊 Energy Budget:")
    print(f"     Total Energy:      {total_energy:.2f} J")
    print(f"     Idle Energy:       {total_idle:.2f} J")
    print(f"     Request-Induced:   {total_request:.2f} J")
    print(f"     Check: {total_idle:.2f} + {total_request:.2f} = {total_idle + total_request:.2f} J")
    
    conservation_error = abs((total_idle + total_request) - total_energy)
    if conservation_error < 0.01:
        print(f"     ✓ Energy conservation OK (error: {conservation_error:.3f} J)")
    else:
        print(f"     ⚠️  Conservation gap: {conservation_error:.2f} J")
    
    # Build span graph with action classification
    print(f"\n[4/5] Building span graph (with action classification)...")
    
    graph = build_span_graph(traces)
    
    print(f"  ✓ Loaded {len(traces)} traces")
    print(f"  ✓ Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    
    # Show action distribution
    action_counts = defaultdict(int)
    for trace in graph["traces"]:
        action_counts[trace["user_action"]] += 1
    
    print(f"\n  📊 User Action Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(traces) * 100 if traces else 0
        print(f"     {action:15s} {count:>4d} ({pct:>5.1f}%)")
    
    # Map spans to intervals
    print(f"\n[5/5] Mapping spans to intervals...")
    
    intervals = map_spans_to_intervals(intervals, graph)
    
    total_activations = sum(i["active_span_count"] for i in intervals)
    print(f"  ✓ Mapped {total_activations} span activations to intervals")
    
    # Save outputs
    print(f"\n[6/6] Saving preprocessed data...")
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "graph_structure.json", 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"  ✓ Saved graph_structure.json")
    
    with open(output_path / "interval_timeseries.json", 'w') as f:
        json.dump(intervals, f, indent=2)
    print(f"  ✓ Saved interval_timeseries.json")
    
    metadata = {
        "experiment_name": experiment_name,
        "experiment_type": manifest["experiment"]["type"],
        "telemetry": telemetry,
        "interval_length_s": interval_length,
        "measurement_window": {
            "start": manifest["execution"]["start_time"],
            "end": manifest["execution"]["end_time"],
            "duration_s": manifest["execution"]["duration_seconds"]
        },
        "idle_baseline": {
            "method": "mean",  # Average of pre and post baseline
            "total_power_W": total_idle_power,
            "per_service": idle_power_per_service
        },
        "statistics": {
            "trace_count": len(traces),
            "span_count": len(graph["nodes"]),
            "interval_count": len(intervals),
            "total_energy_J": total_energy,
            "idle_energy_J": total_idle,
            "request_induced_J": total_request,
            "action_distribution": dict(action_counts)
        }
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}")
    print(f"\nStatistics:")
    print(f"  Traces:              {len(traces)}")
    print(f"  Spans:               {len(graph['nodes'])}")
    print(f"  Intervals:           {len(intervals)}")
    print(f"  Total Energy:        {total_energy:.2f} J")
    print(f"  Idle Energy:         {total_idle:.2f} J")
    print(f"  Request-induced:     {total_request:.2f} J")
    
    print(f"\nNext steps:")
    print(f"  1. Interval-based attribution: python3 interval_duration_attribution.py {output_path}")
    print(f"  2. Span-based attribution:     python3 span_duration_attribution.py {output_path}")
    print()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess experiment data (FIXED: mean baseline, zero handling, action classification)"
    )
    parser.add_argument(
        "input_dir",
        help="Experiment directory (e.g., cluster_experiments/E3/E3_20260320_012421)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        run_preprocessing(args.input_dir, args.output_dir)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
