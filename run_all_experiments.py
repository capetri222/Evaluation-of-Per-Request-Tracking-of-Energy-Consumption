#!/usr/bin/env python3
"""
run_all_experiments.py  –  Automated 3-Run Batch Orchestrator
==============================================================
Runs all experiments 3 times each for statistical validation.

E1 (isolated, sequential mode):
    4 N-values (1, 10, 100, 1000) × 3 runs = 12 E1 runs
    Each run executes all 4 actions (browse, add_to_cart, view_cart, checkout)
    sequentially with shared baselines via --sequential flag.

E2–E6 (concurrent):
    5 experiments × 3 runs = 15 runs

Total: 27 runs

Usage:
    python3 run_all_experiments.py
    python3 run_all_experiments.py --runs 3
    python3 run_all_experiments.py --experiments E1 E3 E5   # subset only
    python3 run_all_experiments.py --dry-run                # print plan only
    python3 run_all_experiments.py --start-from E3          # resume after crash
"""

import argparse
import subprocess
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

ORCHESTRATOR   = "cluster_experiment_orchestrator.py"
LOG_FILE       = "batch_run.log"
STATE_FILE     = "batch_run_state.json"   # tracks completed runs for resume

# E1 N-values to run sequentially
E1_N_VALUES = [1, 10, 100, 1000]

# Concurrent experiments
CONCURRENT_EXPERIMENTS = ["E2", "E3", "E4", "E5", "E6"]

# Estimated durations per run (for ETA calculation)
ESTIMATED_DURATION_MIN = {
    "E1_N1":    8,    # ~8 min: 2min pre-baseline + requests + cooldown + 2min post
    "E1_N10":   8,
    "E1_N100":  10,
    "E1_N1000": 20,   # more requests take longer
    "E2":       18,   # 2min pre + 10min load + cooldown + 2min post
    "E3":       18,
    "E4":       18,
    "E5":       18,
    "E6":       18,
}

DEFAULT_RUNS = 3


# ══════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT  (resume after crash)
# ══════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    p = Path(STATE_FILE)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run_key(exp_id: str, n: int, run_idx: int) -> str:
    if exp_id == "E1":
        return f"E1_N{n}_run{run_idx}"
    return f"{exp_id}_run{run_idx}"


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

def log(msg: str, also_print: bool = True) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line)


# ══════════════════════════════════════════════════════════════════════
# RUN A SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_experiment(exp_id: str, n: int = None, dry_run: bool = False) -> bool:
    """
    Run a single experiment via the orchestrator script.
    Returns True if successful.
    """
    if exp_id == "E1":
        cmd = [
            "python3", ORCHESTRATOR,
            "--experiment", "E1",
            "--sequential",
            "--repetitions", str(n),
        ]
        label = f"E1 --sequential N={n}"
    else:
        cmd = [
            "python3", ORCHESTRATOR,
            "--experiment", exp_id,
        ]
        label = exp_id

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    log(f"Starting: {label}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=False,
            timeout=3600,   # 60min hard timeout per experiment
        )

        elapsed = time.time() - start
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if result.returncode == 0:
            log(f"✅ Completed: {label}  ({elapsed_str})")
            return True
        else:
            log(f"❌ Failed (exit {result.returncode}): {label}  ({elapsed_str})")
            return False

    except subprocess.TimeoutExpired:
        log(f"❌ TIMEOUT after 60min: {label}")
        return False
    except KeyboardInterrupt:
        log(f"⚠️  Interrupted: {label}")
        raise
    except Exception as e:
        log(f"❌ Error: {label}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# BUILD RUN PLAN
# ══════════════════════════════════════════════════════════════════════

def build_plan(experiments: list, n_runs: int) -> list:
    """
    Returns ordered list of (exp_id, n, run_idx, key, est_min).
    Order: all runs of E1 N=1 first, then N=10, etc., then E2–E6.
    This minimises redeploys between runs of the same type.
    """
    plan = []

    # E1: group by N so same-N runs are consecutive (same telemetry config)
    if "E1" in experiments:
        for n in E1_N_VALUES:
            for run_idx in range(1, n_runs + 1):
                key = run_key("E1", n, run_idx)
                est = ESTIMATED_DURATION_MIN.get(f"E1_N{n}", 15)
                plan.append({
                    "exp_id":  "E1",
                    "n":       n,
                    "run_idx": run_idx,
                    "key":     key,
                    "est_min": est,
                    "label":   f"E1 N={n} run {run_idx}/{n_runs}",
                })

    # E2–E6: all runs of each experiment consecutively
    for exp_id in CONCURRENT_EXPERIMENTS:
        if exp_id not in experiments:
            continue
        for run_idx in range(1, n_runs + 1):
            key = run_key(exp_id, None, run_idx)
            est = ESTIMATED_DURATION_MIN.get(exp_id, 18)
            plan.append({
                "exp_id":  exp_id,
                "n":       None,
                "run_idx": run_idx,
                "key":     key,
                "est_min": est,
                "label":   f"{exp_id} run {run_idx}/{n_runs}",
            })

    return plan


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments 3 times each",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_all_experiments.py                    # full 27-run batch
  python3 run_all_experiments.py --dry-run          # preview plan
  python3 run_all_experiments.py --runs 1           # single run (testing)
  python3 run_all_experiments.py --experiments E3 E5 E6
  python3 run_all_experiments.py --start-from E3    # resume after E2 done
  python3 run_all_experiments.py --skip-completed   # use state file to skip
        """
    )
    parser.add_argument("--runs",         type=int, default=DEFAULT_RUNS)
    parser.add_argument("--experiments",  nargs="+",
                        default=["E1"] + CONCURRENT_EXPERIMENTS,
                        help="Experiments to run (default: all)")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--start-from",   default=None,
                        help="Skip all experiments before this one (e.g. E3)")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip runs already recorded in batch_run_state.json")
    args = parser.parse_args()

    # Validate experiments
    valid = {"E1", "E2", "E3", "E4", "E5", "E6"}
    for e in args.experiments:
        if e not in valid:
            print(f"❌ Unknown experiment: {e}")
            sys.exit(1)

    # Build plan
    plan = build_plan(args.experiments, args.runs)

    # Load state for resume
    state = load_state()

    # Apply --start-from filter
    if args.start_from:
        skip_until = args.start_from
        skipping = True
        filtered = []
        for step in plan:
            if skipping and step["exp_id"] == skip_until:
                skipping = False
            if not skipping:
                filtered.append(step)
        if skipping:
            print(f"❌ --start-from '{skip_until}' not found in plan")
            sys.exit(1)
        plan = filtered
        print(f"  Resuming from {skip_until} ({len(plan)} steps remaining)")

    # Print plan
    total_est = sum(s["est_min"] for s in plan)
    total_already_done = sum(
        1 for s in plan if s["key"] in state["completed"])

    print("=" * 65)
    print(f"BATCH EXPERIMENT RUN  –  {len(plan)} total steps")
    print("=" * 65)
    print(f"  Runs per experiment : {args.runs}")
    print(f"  Experiments         : {args.experiments}")
    print(f"  Estimated total     : ~{total_est} min (~{total_est/60:.1f} h)")
    if args.skip_completed and total_already_done:
        print(f"  Already completed   : {total_already_done} (will skip)")
    print()

    if args.dry_run:
        print("DRY RUN – planned steps:")
        for i, step in enumerate(plan, 1):
            done = "✓" if step["key"] in state["completed"] else " "
            print(f"  [{done}] {i:2d}/{len(plan)}  {step['label']:<35}  "
                  f"~{step['est_min']} min")
        print(f"\nTotal: {len(plan)} steps, ~{total_est} min")
        return

    # Confirm if more than 5 steps
    if len(plan) > 5 and not args.dry_run:
        print(f"About to run {len(plan)} experiments "
              f"(~{total_est} min / ~{total_est/60:.1f} h).")
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # Execute
    log(f"=== BATCH START: {len(plan)} steps, {args.runs} runs each ===")

    completed = 0
    failed    = 0
    skipped   = 0
    eta_min   = total_est

    for i, step in enumerate(plan, 1):
        key = step["key"]

        # Skip if already done
        if args.skip_completed and key in state["completed"]:
            print(f"  [{i:2d}/{len(plan)}] SKIP (already done): {step['label']}")
            skipped += 1
            eta_min -= step["est_min"]
            continue

        # ETA
        eta_str = str(timedelta(minutes=eta_min))
        print()
        print(f"{'─'*65}")
        print(f"[{i:2d}/{len(plan)}]  {step['label']}  "
              f"  ETA remaining: ~{eta_str}")
        print(f"{'─'*65}")

        try:
            success = run_experiment(
                step["exp_id"], step["n"], dry_run=False)
        except KeyboardInterrupt:
            log(f"⚠️  Batch interrupted at step {i}/{len(plan)}")
            save_state(state)
            print(f"\n⚠️  Interrupted. Completed {completed} runs.")
            print(f"   Resume with: python3 run_all_experiments.py "
                  f"--skip-completed")
            sys.exit(1)

        eta_min -= step["est_min"]

        if success:
            completed += 1
            state["completed"].append(key)
        else:
            failed += 1
            state["failed"].append(key)
            log(f"  Continuing despite failure...")

        save_state(state)

    # Summary
    print()
    print("=" * 65)
    print("BATCH COMPLETE")
    print("=" * 65)
    log(f"=== BATCH DONE: {completed} completed, {failed} failed, "
        f"{skipped} skipped ===")
    print(f"  Completed : {completed}")
    print(f"  Failed    : {failed}")
    print(f"  Skipped   : {skipped}")

    if state["failed"]:
        print(f"\n  Failed runs:")
        for k in state["failed"]:
            print(f"    ❌ {k}")
        print(f"\n  Re-run failed only:")
        print(f"    python3 run_all_experiments.py --skip-completed")

    print(f"\n  Next steps:")
    print(f"    python3 batch_preprocess_all.py")
    print(f"    python3 batch_attribution_all.py")
    print(f"    python3 generate_plots.py")


if __name__ == "__main__":
    main()
