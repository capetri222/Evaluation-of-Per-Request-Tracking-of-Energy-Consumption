#!/usr/bin/env python3
"""
Batch Preprocessing for ALL Experiments (E1-E6)
================================================
Automatically preprocess all experiment directories.

Experiment structure:
  cluster_experiments/
    E1/
      E1_isolated_browse_N100_...
      E1_isolated_add_to_cart_N100_...
      ... (16 experiments)
    E2/
      E2_20260320_...
    E3/
      E3_20260320_...
    E4/
      E4_20260320_...
    E5/
      E5_20260320_...
    E6/
      E6_20260320_...

Usage:
    python3 batch_preprocess_all.py
    python3 batch_preprocess_all.py --experiments-dir cluster_experiments --output-dir preprocessed_data
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_EXPERIMENTS_DIR = "cluster_experiments"
DEFAULT_OUTPUT_DIR = "preprocessed_data"
PREPROCESSING_SCRIPT = "preprocess_experiment_fixed.py"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT DISCOVERY
# ══════════════════════════════════════════════════════════════════════

def find_all_experiments(experiments_dir: str) -> Dict[str, List[Path]]:
    """
    Find all experiments organized by type (E1-E6).
    
    Returns: {
        "E1": [path1, path2, ...],
        "E2": [path1],
        ...
    }
    """
    exp_path = Path(experiments_dir)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    experiments = {
        "E1": [],
        "E2": [],
        "E3": [],
        "E4": [],
        "E5": [],
        "E6": []
    }
    
    required_files = ["manifest.json", "kepler_metrics.csv", "raw_traces.json"]
    
    # Check E1 (isolated experiments - many subdirectories)
    e1_dir = exp_path / "E1"
    if e1_dir.exists():
        for exp_dir in e1_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("E1_isolated_"):
                if all((exp_dir / f).exists() for f in required_files):
                    experiments["E1"].append(exp_dir)
                else:
                    print(f"⚠️  Skipping {exp_dir.name} (missing required files)")
    
    # Check E2-E6 (concurrent experiments - typically one per type)
    for exp_type in ["E2", "E3", "E4", "E5", "E6"]:
        exp_type_dir = exp_path / exp_type
        
        if not exp_type_dir.exists():
            continue
        
        for exp_dir in exp_type_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith(exp_type):
                if all((exp_dir / f).exists() for f in required_files):
                    experiments[exp_type].append(exp_dir)
                else:
                    print(f"⚠️  Skipping {exp_dir.name} (missing required files)")
    
    # Sort E1 experiments
    def e1_sort_key(p):
        parts = p.name.split('_')
        action = '_'.join(parts[2:-2])
        n_value = parts[-2][1:]
        return (action, int(n_value))
    
    experiments["E1"].sort(key=e1_sort_key)
    
    # Sort E2-E6 by timestamp in name
    for exp_type in ["E2", "E3", "E4", "E5", "E6"]:
        experiments[exp_type].sort(key=lambda p: p.name)
    
    return experiments


def preprocess_experiment(exp_path: Path, output_dir: str, script_path: str) -> bool:
    """
    Run preprocessing on a single experiment.
    
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {exp_path.name}")
    print(f"{'='*70}")
    
    cmd = [
        "python3",
        script_path,
        str(exp_path),
        "--output-dir",
        output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n✅ Successfully preprocessed {exp_path.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to preprocess {exp_path.name}")
        print(f"   Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════

def run_batch_preprocessing(experiments_dir: str, output_dir: str, script_path: str) -> None:
    """Run batch preprocessing on all experiments."""
    
    print("=" * 70)
    print("BATCH PREPROCESSING - ALL EXPERIMENTS (E1-E6)")
    print("=" * 70)
    print(f"Experiments Directory: {experiments_dir}")
    print(f"Output Directory:      {output_dir}")
    print(f"Script:                {script_path}")
    
    # Find experiments
    print(f"\n[1/3] Finding experiments...")
    
    experiments = find_all_experiments(experiments_dir)
    
    total_count = sum(len(exps) for exps in experiments.values())
    
    if total_count == 0:
        print(f"\n❌ No valid experiments found in {experiments_dir}")
        return
    
    print(f"  ✓ Found {total_count} experiments")
    
    # Show breakdown
    print(f"\n  Experiments by type:")
    for exp_type in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        count = len(experiments[exp_type])
        if count > 0:
            print(f"    {exp_type}: {count:>3d} experiment(s)")
    
    # Show E1 breakdown
    if experiments["E1"]:
        from collections import defaultdict
        e1_by_action = defaultdict(list)
        
        for exp in experiments["E1"]:
            parts = exp.name.split('_')
            action = '_'.join(parts[2:-2])
            n_value = parts[-2]
            e1_by_action[action].append(n_value)
        
        print(f"\n  E1 breakdown by action:")
        for action, n_values in sorted(e1_by_action.items()):
            print(f"    {action:15s} : {', '.join(n_values)}")
    
    # Process each experiment
    print(f"\n[2/3] Preprocessing experiments...")
    
    results = []
    processed = 0
    
    for exp_type in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        if not experiments[exp_type]:
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {exp_type} experiments ({len(experiments[exp_type])} total)")
        print(f"{'='*70}")
        
        for i, exp_path in enumerate(experiments[exp_type], 1):
            processed += 1
            print(f"\n[{processed}/{total_count}] [{exp_type}] ", end="")
            
            success = preprocess_experiment(exp_path, output_dir, script_path)
            results.append((exp_path.name, exp_type, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH PREPROCESSING SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    
    print(f"\nTotal:      {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    
    # Show results by experiment type
    print(f"\nResults by experiment type:")
    for exp_type in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        type_results = [(name, success) for name, etype, success in results if etype == exp_type]
        if type_results:
            type_success = sum(1 for _, success in type_results if success)
            print(f"  {exp_type}: {type_success}/{len(type_results)} successful")
    
    if failed > 0:
        print(f"\n❌ Failed experiments:")
        for name, exp_type, success in results:
            if not success:
                print(f"   [{exp_type}] {name}")
    
    if successful > 0:
        print(f"\n✅ All preprocessed data saved to: {output_dir}/")
        print(f"\nNext steps:")
        print(f"  Run attribution: python3 batch_attribution_all.py")
    
    print()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocess all experiments (E1-E6)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--experiments-dir",
        default=DEFAULT_EXPERIMENTS_DIR,
        help=f"Experiments directory (default: {DEFAULT_EXPERIMENTS_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--script",
        default=PREPROCESSING_SCRIPT,
        help=f"Preprocessing script path (default: {PREPROCESSING_SCRIPT})"
    )
    
    args = parser.parse_args()
    
    # Check if preprocessing script exists
    if not Path(args.script).exists():
        print(f"❌ Preprocessing script not found: {args.script}")
        print(f"   Make sure {args.script} is in the current directory")
        return 1
    
    try:
        run_batch_preprocessing(args.experiments_dir, args.output_dir, args.script)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
