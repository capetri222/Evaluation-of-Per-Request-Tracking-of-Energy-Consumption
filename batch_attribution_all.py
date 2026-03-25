#!/usr/bin/env python3
"""
Batch Attribution for ALL Experiments
======================================
Automatically run attribution on all preprocessed experiments.

Attribution strategy:
  - E1 (isolated): Temporal differentiation + Interval-duration + Span-duration
  - E2-E6 (concurrent): Interval-duration + Span-duration

Input:  preprocessed_data/
Output: attributed_data/

Usage:
    python3 batch_attribution_all.py
    python3 batch_attribution_all.py --preprocessed-dir preprocessed_data --output-dir attributed_data
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_PREPROCESSED_DIR = "preprocessed_data"
DEFAULT_OUTPUT_DIR = "attributed_data"

ATTRIBUTION_SCRIPTS = {
    "interval": "interval_duration_attribution.py",
    "span": "span_duration_attribution.py",
    "temporal": "temporal_differentiation_attribution.py"
}

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT DISCOVERY
# ══════════════════════════════════════════════════════════════════════

def find_preprocessed_experiments(preprocessed_dir: str) -> Dict[str, List[Path]]:
    """
    Find all preprocessed experiments organized by type.
    
    Returns: {
        "E1_isolated": [path1, path2, ...],
        "E2_concurrent": [path1],
        ...
    }
    """
    prep_path = Path(preprocessed_dir)
    
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")
    
    experiments = {
        "E1_isolated": [],
        "E2_concurrent": [],
        "E3_concurrent": [],
        "E4_concurrent": [],
        "E5_concurrent": [],
        "E6_concurrent": []
    }
    
    required_files = ["metadata.json", "graph_structure.json", "interval_timeseries.json"]
    
    for exp_dir in prep_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check required files
        if not all((exp_dir / f).exists() for f in required_files):
            print(f"⚠️  Skipping {exp_dir.name} (missing required files)")
            continue
        
        # Read metadata to determine experiment type
        try:
            with open(exp_dir / "metadata.json") as f:
                metadata = json.load(f)
            
            exp_type = metadata.get("experiment_type", "unknown")
            
            # Classify experiment
            if exp_dir.name.startswith("E1_isolated_"):
                experiments["E1_isolated"].append(exp_dir)
            elif exp_dir.name.startswith("E2"):
                experiments["E2_concurrent"].append(exp_dir)
            elif exp_dir.name.startswith("E3"):
                experiments["E3_concurrent"].append(exp_dir)
            elif exp_dir.name.startswith("E4"):
                experiments["E4_concurrent"].append(exp_dir)
            elif exp_dir.name.startswith("E5"):
                experiments["E5_concurrent"].append(exp_dir)
            elif exp_dir.name.startswith("E6"):
                experiments["E6_concurrent"].append(exp_dir)
            else:
                print(f"⚠️  Unknown experiment type: {exp_dir.name}")
                
        except Exception as e:
            print(f"⚠️  Could not read metadata for {exp_dir.name}: {e}")
    
    # Sort experiments
    def e1_sort_key(p):
        parts = p.name.split('_')
        action = '_'.join(parts[2:-2])
        n_value = parts[-2][1:]
        return (action, int(n_value))
    
    experiments["E1_isolated"].sort(key=e1_sort_key)
    
    for key in experiments.keys():
        if key != "E1_isolated":
            experiments[key].sort(key=lambda p: p.name)
    
    return experiments


def get_original_experiment_path(preprocessed_path: Path) -> Path:
    """
    Get original experiment path for temporal differentiation.
    
    preprocessed_data/E1_isolated_browse_N100_20260320_202929
    -> cluster_experiments/E1/E1_isolated_browse_N100_20260320_202929
    """
    exp_name = preprocessed_path.name
    
    if exp_name.startswith("E1_isolated_"):
        return Path("cluster_experiments") / "E1" / exp_name
    else:
        # E2-E6
        exp_type = exp_name.split('_')[0]  # E2, E3, etc.
        return Path("cluster_experiments") / exp_type / exp_name


def run_attribution(
    exp_path: Path,
    output_dir: str,
    method: str,
    script_path: str,
    use_original_path: bool = False
) -> bool:
    """
    Run attribution on a single experiment.
    
    Args:
        exp_path: Path to preprocessed experiment
        output_dir: Output directory for attribution
        method: "interval", "span", or "temporal"
        script_path: Path to attribution script
        use_original_path: For temporal, use original experiment path
    
    Returns True if successful, False otherwise.
    """
    print(f"  [{method:8s}] ", end="", flush=True)
    
    # For temporal differentiation, use original experiment path
    input_path = get_original_experiment_path(exp_path) if use_original_path else exp_path
    
    cmd = [
        "python3",
        script_path,
        str(input_path),
        "--output-dir",
        output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅", flush=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ (exit code {e.returncode})", flush=True)
        if e.stderr:
            print(f"           Error: {e.stderr.strip()[:100]}")
        return False
    except Exception as e:
        print(f"❌ ({e})", flush=True)
        return False


# ══════════════════════════════════════════════════════════════════════
# BATCH ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════

def run_batch_attribution(preprocessed_dir: str, output_dir: str) -> None:
    """Run batch attribution on all preprocessed experiments."""
    
    print("=" * 70)
    print("BATCH ATTRIBUTION - ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Preprocessed Directory: {preprocessed_dir}")
    print(f"Output Directory:       {output_dir}")
    
    # Check scripts exist
    print(f"\n[1/4] Checking attribution scripts...")
    
    missing_scripts = []
    for method, script in ATTRIBUTION_SCRIPTS.items():
        if not Path(script).exists():
            missing_scripts.append(script)
            print(f"  ❌ {script}")
        else:
            print(f"  ✓ {script}")
    
    if missing_scripts:
        print(f"\n❌ Missing scripts. Please ensure they are in the current directory.")
        return
    
    # Find experiments
    print(f"\n[2/4] Finding preprocessed experiments...")
    
    experiments = find_preprocessed_experiments(preprocessed_dir)
    
    total_count = sum(len(exps) for exps in experiments.values())
    
    if total_count == 0:
        print(f"\n❌ No preprocessed experiments found in {preprocessed_dir}")
        print(f"   Run preprocessing first: python3 batch_preprocess_all.py")
        return
    
    print(f"  ✓ Found {total_count} preprocessed experiments")
    
    # Show breakdown
    print(f"\n  Experiments by type:")
    for exp_type, exps in experiments.items():
        if exps:
            print(f"    {exp_type:15s}: {len(exps):>3d} experiment(s)")
    
    # Process each experiment
    print(f"\n[3/4] Running attribution...")
    
    results = []
    processed = 0
    
    for exp_type, exp_list in experiments.items():
        if not exp_list:
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {exp_type} ({len(exp_list)} total)")
        print(f"{'='*70}")
        
        is_isolated = "isolated" in exp_type
        
        for i, exp_path in enumerate(exp_list, 1):
            processed += 1
            print(f"\n[{processed}/{total_count}] {exp_path.name}")
            
            exp_results = {
                "name": exp_path.name,
                "type": exp_type,
                "interval": False,
                "span": False,
                "temporal": False
            }
            
            # Run interval-duration attribution
            success = run_attribution(
                exp_path,
                output_dir,
                "interval",
                ATTRIBUTION_SCRIPTS["interval"]
            )
            exp_results["interval"] = success
            
            # Run span-duration attribution
            success = run_attribution(
                exp_path,
                output_dir,
                "span",
                ATTRIBUTION_SCRIPTS["span"]
            )
            exp_results["span"] = success
            
            # Run temporal differentiation (E1 only, skip if source deleted)
            if is_isolated:
                original_path = get_original_experiment_path(exp_path)
                if not original_path.exists():
                    print(f"  [temporal ] ⚠  skipped – {original_path} not found")
                    exp_results["temporal"] = False
                else:
                    success = run_attribution(
                        exp_path,
                        output_dir,
                        "temporal",
                        ATTRIBUTION_SCRIPTS["temporal"],
                        use_original_path=True
                    )
                    exp_results["temporal"] = success
            
            results.append(exp_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH ATTRIBUTION SUMMARY")
    print("=" * 70)
    
    total_attributions = 0
    successful_attributions = 0
    
    for result in results:
        if "isolated" in result["type"]:
            # E1: 3 methods
            total_attributions += 3
            successful_attributions += sum([
                result["interval"],
                result["span"],
                result["temporal"]
            ])
        else:
            # E2-E6: 2 methods
            total_attributions += 2
            successful_attributions += sum([
                result["interval"],
                result["span"]
            ])
    
    print(f"\nTotal experiments:   {len(results)}")
    print(f"Total attributions:  {total_attributions}")
    print(f"Successful:          {successful_attributions}")
    print(f"Failed:              {total_attributions - successful_attributions}")
    
    # Show results by experiment type
    print(f"\nResults by experiment type:")
    
    for exp_type in ["E1_isolated", "E2_concurrent", "E3_concurrent", 
                     "E4_concurrent", "E5_concurrent", "E6_concurrent"]:
        type_results = [r for r in results if r["type"] == exp_type]
        
        if not type_results:
            continue
        
        is_isolated = "isolated" in exp_type
        
        interval_success = sum(1 for r in type_results if r["interval"])
        span_success = sum(1 for r in type_results if r["span"])
        
        print(f"\n  {exp_type}:")
        print(f"    Interval-duration: {interval_success}/{len(type_results)}")
        print(f"    Span-duration:     {span_success}/{len(type_results)}")
        
        if is_isolated:
            temporal_success = sum(1 for r in type_results if r["temporal"])
            print(f"    Temporal:          {temporal_success}/{len(type_results)}")
    
    # Show failures
    failures = []
    for result in results:
        failed_methods = []
        if not result["interval"]:
            failed_methods.append("interval")
        if not result["span"]:
            failed_methods.append("span")
        if "temporal" in result and not result["temporal"]:
            # Only count as failure if the source data existed
            original_path = get_original_experiment_path(
                Path(result["name"]) if not isinstance(result.get("path"), Path)
                else result["path"]
            )
            if original_path.exists():
                failed_methods.append("temporal")
        
        if failed_methods:
            failures.append((result["name"], failed_methods))
    
    if failures:
        print(f"\n❌ Failed attributions:")
        for name, methods in failures:
            print(f"   {name}: {', '.join(methods)}")
    
    if successful_attributions > 0:
        print(f"\n✅ All attributed data saved to: {output_dir}/")
        print(f"\nNext steps:")
        print(f"  Visualize results or run analysis scripts")
    
    print()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch attribution for all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--preprocessed-dir",
        default=DEFAULT_PREPROCESSED_DIR,
        help=f"Preprocessed data directory (default: {DEFAULT_PREPROCESSED_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        run_batch_attribution(args.preprocessed_dir, args.output_dir)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
