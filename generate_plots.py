#!/usr/bin/env python3
"""
generate_plots.py  –  Analysis & Visualisation Pipeline
=========================================================
Generates all thesis plots from attributed experiment data.

Usage:
    python3 generate_plots.py
    python3 generate_plots.py --attributed-dir attributed_data --output-dir plots
"""

import argparse
import csv as _csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_ATTRIBUTED_DIR = "attributed_data"
DEFAULT_CLUSTER_DIR    = "cluster_experiments"
DEFAULT_OUTPUT_DIR     = "plots"
DEFAULT_RUNS           = 3

ACTIONS = ["browse", "add_to_cart", "view_cart", "checkout"]
ACTION_LABELS = {
    "browse":      "Browse",
    "add_to_cart": "Add to Cart",
    "view_cart":   "View Cart",
    "checkout":    "Checkout",
}
N_VALUES = [1, 10, 100, 1000]
CONCURRENT_EXPERIMENTS = ["E2", "E3", "E4", "E5", "E6"]

# Human-readable experiment labels for all plots
EXP_LABELS = {
    "E2": "Workload Low",
    "E3": "Workload Medium",
    "E4": "Workload High",
    "E5": "Telemetry Medium",
    "E6": "Telemetry Low",
}
# Short versions for tight axes (e.g. tick labels with config info)
EXP_LABELS_SHORT = {
    "E2": "Wrkld Low",
    "E3": "Wrkld Med",
    "E4": "Wrkld High",
    "E5": "Telem Med",
    "E6": "Telem Low",
}
# E1 N-labels
def e1_label(n):
    return f"Isolated N={n}"
def e1_label_short(n):
    return f"Isol. N={n}"

COLORS = {
    "temporal": "#2196F3",
    "interval": "#4CAF50",
    "span":     "#FF9800",
    "coverage": "#673AB7",
}
METHOD_LABELS = {
    "temporal": "Temporal Diff.",
    "interval": "Interval-Duration",
    "span":     "Span-Duration",
}
ACTION_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
EXP_COLORS    = ["#E91E63", "#FF5722", "#3F51B5", "#009688", "#795548"]
DPI = 150


# ══════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════

def parse_exp_key(name: str) -> str:
    parts = name.split("_")
    if parts[0] == "E1":
        key_parts = []
        for p in parts:
            if len(p) == 8 and p.isdigit():
                break
            key_parts.append(p)
        return "_".join(key_parts)
    return parts[0]


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_attribution_rate(exp_dir: Path) -> float:
    s = load_json(exp_dir / "attribution_statistics.json")
    if s is None:
        return -1.0
    return s["attribution_quality"]["attribution_rate"]


def select_best_runs(attributed_dir: Path, n_runs: int) -> Dict[str, Path]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for exp_dir in attributed_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if not (exp_dir / "attribution_statistics.json").exists():
            continue
        groups[parse_exp_key(exp_dir.name)].append(exp_dir)

    best: Dict[str, Path] = {}
    for key, dirs in groups.items():
        ranked = sorted(dirs, key=load_attribution_rate, reverse=True)
        best[key] = ranked[0]
        rates = [f"{load_attribution_rate(d)*100:.1f}%" for d in ranked]
        tag = f"(rates: {', '.join(rates)})" if len(dirs) > 1 else "(only run)"
        print(f"  {key}: {ranked[0].name} {tag}")
    return best


def e1_temporal(best, action, n) -> Optional[float]:
    exp_dir = best.get(f"E1_isolated_{action}_N{n}")
    if exp_dir is None:
        return None
    d = load_json(exp_dir / "temporal_differentiation.json")
    return d.get("energy_per_request_mJ") if d else None


def e1_interval(best, action, n) -> Optional[float]:
    exp_dir = best.get(f"E1_isolated_{action}_N{n}")
    if exp_dir is None:
        return None
    d = load_json(exp_dir / "attribution_statistics.json")
    return d["trace_energy_distribution"]["mean"] * 1000 if d else None


def e1_span(best, action, n) -> Optional[float]:
    exp_dir = best.get(f"E1_isolated_{action}_N{n}")
    if exp_dir is None:
        return None
    d = load_json(exp_dir / "attribution_statistics_span_method.json")
    return d["trace_energy_distribution"]["mean"] * 1000 if d else None


def load_trace_energies_mJ(exp_dir: Path) -> List[float]:
    d = load_json(exp_dir / "trace_energies.json")
    if d is None:
        return []
    return [t["total_energy_J"] * 1000
            for t in d.get("traces", []) if t["total_energy_J"] > 0]


# Map classifier labels to canonical action names.
# Locust may classify both "add_to_cart" and "view_cart" as "cart"
# (since both hit /api/cart). We keep the raw label when it is already
# specific, and map the ambiguous "cart" to "view_cart" as default.
# After preprocessing fix, actions are: browse, add_to_cart, view_cart,
# checkout, background. The old "cart" label may still appear in older
# preprocessed data - map it to view_cart as fallback.
ACTION_LABEL_MAP = {
    "browse":       "browse",
    "cart":         "view_cart",     # legacy label from old preprocessing
    "add_to_cart":  "add_to_cart",
    "view_cart":    "view_cart",
    "checkout":     "checkout",
    "background":   "background",
}

def load_trace_energies_by_action(exp_dir: Path) -> Dict[str, List[float]]:
    d = load_json(exp_dir / "trace_energies.json")
    if d is None:
        return {}
    result: Dict[str, List[float]] = defaultdict(list)
    for t in d.get("traces", []):
        if t["total_energy_J"] > 0:
            raw = t.get("user_action", "background")
            action = ACTION_LABEL_MAP.get(raw, raw)
            result[action].append(t["total_energy_J"] * 1000)
    return dict(result)


# ══════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════

def save(fig, output_dir: Path, name: str) -> None:
    p = output_dir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {p}")


def annotate_bars(ax, bars, fmt="{:.1f}", offset=(0, 4), fontsize=8):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=offset, textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize)


def grid_style(ax):
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════
# E1 - Average energy by method (N=100)
# ══════════════════════════════════════════════════════════════════════

def plot_e1_action_energy_by_method(best, output_dir):
    print("\n[E1] Average energy per action by method (N=100)...")
    N = 100
    labels   = [ACTION_LABELS[a] for a in ACTIONS]
    temporal = [e1_temporal(best, a, N) or 0.0 for a in ACTIONS]
    interval = [e1_interval(best, a, N) or 0.0 for a in ACTIONS]
    span     = [e1_span(best, a, N)     or 0.0 for a in ACTIONS]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w, temporal, w, label=METHOD_LABELS["temporal"],
                color=COLORS["temporal"], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,     interval, w, label=METHOD_LABELS["interval"],
                color=COLORS["interval"], alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + w, span,     w, label=METHOD_LABELS["span"],
                color=COLORS["span"],     alpha=0.85, edgecolor="white")
    for bars in (b1, b2, b3):
        annotate_bars(ax, bars, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Energy per Request [mJ]")
    ax.set_title("Isolated Requests (N=100): Mean Energy per Action by Attribution Method")
    ax.legend(framealpha=0.9)
    grid_style(ax)
    fig.tight_layout()
    save(fig, output_dir, "e1_action_energy_by_method.png")


# ══════════════════════════════════════════════════════════════════════
# E1 - Linearity check (dual-row: full + zoomed with annotations)
# ══════════════════════════════════════════════════════════════════════

def plot_e1_action_energy_by_n(best, output_dir):
    print("\n[E1] Energy per request vs. N (linearity)...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="row")

    for col, action in enumerate(ACTIONS):
        data = {}
        for n in N_VALUES:
            t  = e1_temporal(best, action, n)
            iv = e1_interval(best, action, n)
            if t is not None and iv is not None:
                data[n] = (t, iv)

        xs = sorted(data.keys())

        for row, zoom in enumerate([False, True]):
            ax = axes[row, col]
            xs_z = [n for n in xs if n >= 100] if zoom else xs
            if not xs_z:
                continue
            yt = [data[n][0] for n in xs_z]
            yi = [data[n][1] for n in xs_z]

            ax.plot(xs_z, yt, "o-", color=COLORS["temporal"],
                    label=METHOD_LABELS["temporal"], lw=2, ms=6)
            ax.plot(xs_z, yi, "s--", color=COLORS["interval"],
                    label=METHOD_LABELS["interval"], lw=2, ms=6)

            for n, vt, vi in zip(xs_z, yt, yi):
                ax.annotate(f"{vt:.0f}", (n, vt),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color=COLORS["temporal"])
                ax.annotate(f"{vi:.1f}", (n, vi),
                            textcoords="offset points", xytext=(0, -12),
                            ha="center", fontsize=7, color=COLORS["interval"])

            ax.set_xscale("log")
            ax.set_xticks(xs_z)
            ax.set_xticklabels([str(n) for n in xs_z])
            ax.set_xlabel("Number of Requests N" if row == 1 else "")
            ax.set_ylabel("Energy per Request [mJ]" if col == 0 else "")
            title = ACTION_LABELS[action]
            ax.set_title(title if row == 0 else f"{title} (N=100/1000 zoom)",
                         fontweight="bold" if row == 0 else "normal", fontsize=9)
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
            grid_style(ax)

    fig.suptitle(
        "Isolated Requests: Energy per Request over N — Linearity Check\n"
        "(top: full scale incl. N=1 idle-dominated; bottom: zoomed N>=10)",
        y=1.01)
    fig.tight_layout()
    save(fig, output_dir, "e1_action_energy_by_n.png")


# ══════════════════════════════════════════════════════════════════════
# E1 - Top-10 traces per action
# ══════════════════════════════════════════════════════════════════════

# def plot_e1_top10_traces(best, output_dir):
#     print("\n[E1] Top-10 traces per action...")
#     for action in ACTIONS:
#         key     = f"E1_isolated_{action}_N1000"
#         exp_dir = best.get(key)
#         if exp_dir is None:
#             print(f"  ⚠  {key} not found")
#             continue
# 
#         td = load_json(exp_dir / "temporal_differentiation.json")
#         te = load_json(exp_dir / "trace_energies.json")
#         if td is None or te is None:
#             continue
# 
#         temp_mJ = td.get("energy_per_request_mJ", 0.0)
#         traces  = sorted(te.get("traces", []),
#                          key=lambda t: t["total_energy_J"], reverse=True)[:10]
#         if not traces:
#             continue
# 
#         vals   = [t["total_energy_J"] * 1000 for t in traces]
#         labels = [f"T{i+1}" for i in range(len(traces))]
# 
#         fig, ax = plt.subplots(figsize=(11, 5))
#         bars = ax.bar(labels, vals, color=COLORS["interval"],
#                       alpha=0.85, edgecolor="white", width=0.6)
#         ax.axhline(temp_mJ, color=COLORS["temporal"], lw=2, ls="--",
#                    label=f"{METHOD_LABELS['temporal']} avg ({temp_mJ:.1f} mJ)")
#         annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7)
#         ax.set_ylabel("Energy [mJ]")
#         ax.set_title(
#             f"E1 - {ACTION_LABELS[action]} (N=1000): "
#             f"Top-10 Traces (Interval-Duration) vs. Temporal Average")
#         ax.legend(framealpha=0.9)
#         grid_style(ax)
#         fig.tight_layout()
#         save(fig, output_dir, f"e1_top10_traces_{action}.png")
# 
# 
# # ══════════════════════════════════════════════════════════════════════
# # ALL - Attribution coverage
# # ══════════════════════════════════════════════════════════════════════
def plot_attribution_coverage(best, output_dir):
    """Attribution coverage: E1 grouped by N (mean across actions), then E2-E6."""
    print("\n[ALL] Attribution coverage...")

    entries = []

    # E1: group by N, mean across all 4 actions
    for n in N_VALUES:
        rates = []
        for action in ACTIONS:
            exp_dir = best.get(f"E1_isolated_{action}_N{n}")
            if exp_dir:
                r = load_attribution_rate(exp_dir)
                if r >= 0:
                    rates.append(r * 100)
        if rates:
            entries.append((f"Isolated N={n}", np.mean(rates), np.std(rates)))

    # E2-E6
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir:
            r = load_attribution_rate(exp_dir)
            if r >= 0:
                entries.append((EXP_LABELS.get(exp_id, exp_id), r * 100, 0))

    if not entries:
        return

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    stds   = [e[2] for e in entries]

    # Color: E1 entries in purple shades, E2-E6 in experiment colors
    colors = []
    for lbl in labels:
        if lbl.startswith("E1"):
            colors.append(COLORS["coverage"])
        else:
            idx = CONCURRENT_EXPERIMENTS.index(lbl) if lbl in CONCURRENT_EXPERIMENTS else 0
            colors.append(EXP_COLORS[idx])

    fig, ax = plt.subplots(figsize=(9, max(5, len(entries) * 0.6)))
    y    = np.arange(len(labels))
    bars = ax.barh(y, values, xerr=stds, color=colors, alpha=0.85,
                   edgecolor="white", capsize=3,
                   error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Attribution Coverage [%]", fontsize=11)
    ax.set_title("Attribution Coverage (Interval-Duration Method)\n"
                 "Isolated Requests: mean across all 4 user actions per N", fontsize=12)
    ax.set_xlim(0, 105)
    ax.axvline(100, color="grey", lw=0.8, ls=":")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Separator line between E1 and E2-E6
    e1_count = sum(1 for l in labels if l.startswith("E1"))
    ax.axhline(e1_count - 0.5, color="#aaa", lw=1, ls="--")

    for bar, val, std in zip(bars, values, stds):
        txt = f"{val:.0f}%" + (f" ±{std:.0f}" if std > 0.5 else "")
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=10)
    fig.tight_layout()
    save(fig, output_dir, "all_attribution_coverage.png")


# ══════════════════════════════════════════════════════════════════════
# E2-E6 - Action energy vs. E1 baseline
# ══════════════════════════════════════════════════════════════════════

def plot_concurrent_action_energy(best, output_dir):
    print("\n[E2-E6] Action energy vs. E1 baseline...")
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        te = load_json(exp_dir / "trace_energies.json")
        if te is None:
            continue

        by_action: Dict[str, List[float]] = defaultdict(list)
        for t in te.get("traces", []):
            raw = t.get("user_action", "background")
            ua  = ACTION_LABEL_MAP.get(raw, raw)
            if ua in ACTIONS:
                by_action[ua].append(t["total_energy_J"] * 1000)

        # Always show all 4 actions explicitly; 0 if no data
        present = ACTIONS
        labels    = [ACTION_LABELS[a] for a in present]
        means     = [np.mean(by_action[a]) if a in by_action and by_action[a] else 0.0
                     for a in present]
        stds      = [np.std(by_action[a])  if a in by_action and by_action[a] else 0.0
                     for a in present]
        baselines = [e1_temporal(best, a, 100) or 0.0 for a in present]
        # Skip experiment if all means are 0
        if all(m == 0 for m in means):
            continue

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        # Clip lower error bar at 0 (negative energy = attribution artefact)
        stds_clipped = [min(s, m) for m, s in zip(means, stds)]
        b1 = ax.bar(x - w/2, means, w, yerr=[stds_clipped, stds], capsize=4,
                    label=f"{EXP_LABELS.get(exp_id, exp_id)} Interval-Duration (mean ± std)",
                    color=COLORS["interval"], alpha=0.85, edgecolor="white",
                    error_kw={"elinewidth": 1.5, "ecolor": "#2e7d32"})
        b2 = ax.bar(x + w/2, baselines, w,
                    label="Isolated N=100 Temporal Diff. (reference)",
                    color=COLORS["temporal"], alpha=0.7, edgecolor="white")
        annotate_bars(ax, b1, fontsize=7)
        annotate_bars(ax, b2, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Energy per Trace [mJ]")
        ax.set_title(f"{EXP_LABELS.get(exp_id, exp_id)}: Per-Request Energy by User Action\n(Interval-Duration vs. E1 Baseline)")
        ax.legend(framealpha=0.9, fontsize=9)
        grid_style(ax)
        fig.tight_layout()
        save(fig, output_dir, f"{exp_id.lower()}_action_energy_vs_baseline.png")


# ══════════════════════════════════════════════════════════════════════
# E2-E6 - Top-10 traces per action
# ══════════════════════════════════════════════════════════════════════

# def plot_concurrent_top10_traces(best, output_dir):
#     print("\n[E2-E6] Top-10 traces per action...")
#     for exp_id in CONCURRENT_EXPERIMENTS:
#         exp_dir = best.get(exp_id)
#         if exp_dir is None:
#             continue
#         te = load_json(exp_dir / "trace_energies.json")
#         if te is None:
#             continue
# 
#         for action in ACTIONS:
#             traces = sorted(
#                 [t for t in te.get("traces", [])
#                  if t.get("user_action") == action],
#                 key=lambda t: t["total_energy_J"], reverse=True)[:10]
#             if not traces:
#                 continue
# 
#             baseline_mJ = e1_temporal(best, action, 100) or 0.0
#             vals   = [t["total_energy_J"] * 1000 for t in traces]
#             labels = [f"T{i+1}" for i in range(len(traces))]
# 
#             fig, ax = plt.subplots(figsize=(10, 4))
#             bars = ax.bar(labels, vals, color=COLORS["interval"],
#                           alpha=0.85, edgecolor="white", width=0.6)
#             if baseline_mJ > 0:
#                 ax.axhline(baseline_mJ, color=COLORS["temporal"], lw=2, ls="--",
#                            label=f"E1 Temporal Avg ({baseline_mJ:.1f} mJ)")
#             annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7)
#             ax.set_ylabel("Energy [mJ]")
#             ax.set_title(f"{exp_id} - {ACTION_LABELS[action]}: "
#                          f"Top-10 Traces (Interval-Duration)")
#             ax.legend(framealpha=0.9, fontsize=9)
#             grid_style(ax)
#             fig.tight_layout()
#             save(fig, output_dir,
#                  f"{exp_id.lower()}_top10_traces_{action}.png")
# 
# 
# # ══════════════════════════════════════════════════════════════════════
# # E3/E5/E6 - Telemetry overhead
# # Uses increase()-based summary; falls back to kepler_metrics.csv
# # ══════════════════════════════════════════════════════════════════════
def plot_telemetry_overhead(best, cluster_dir, output_dir):
    print("\n[E3/E5/E6] Telemetry overhead...")

    TELEMETRY_EXPS = {
        "Workload Medium\n(1s/100%)":   "E3",
        "Telemetry Medium\n(10s/50%)": "E5",
        "Telemetry Low\n(60s/1%)":  "E6",
    }
    OVERHEAD_SVCS = [
        "jaeger", "opentelemetry-collector", "prometheus-server", "kepler"
    ]
    OVERHEAD_ALIASES = {
        "jaeger":                  ["jaeger"],
        "opentelemetry-collector": ["opentelemetry-collector", "collector"],
        "prometheus-server":       ["prometheus-server"],
        "kepler":                  ["kepler", "kepler-exporter"],
    }
    SVC_COLORS = ["#E91E63", "#FF5722", "#FFC107", "#009688"]

    data: Dict[str, Dict[str, float]] = {}

    for label, exp_id in TELEMETRY_EXPS.items():
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            print(f"  ⚠  {exp_id} not found")
            continue

        # 1) increase()-based summary (preferred)
        summary = None
        for candidate in cluster_dir.glob(
                f"{exp_id}/*/telemetry_overhead_summary.json"):
            s = load_json(candidate)
            if s and s.get("method") == "increase()" and \
                    s.get("total_overhead_J", 0) > 0:
                summary = s
                break

        if summary is not None:
            svc_data = {}
            for svc in OVERHEAD_SVCS:
                svc_data[svc] = sum(
                    summary.get("services", {}).get(a, 0.0)
                    for a in OVERHEAD_ALIASES[svc])
            data[label] = svc_data
            continue

        # 2) Fallback: kepler_metrics.csv delta
        csv_path = None
        for candidate in cluster_dir.glob(f"{exp_id}/*/kepler_metrics.csv"):
            csv_path = candidate
            break
        if csv_path is None:
            print(f"  ⚠  no data for {exp_id}")
            continue

        series: Dict[str, list] = defaultdict(list)
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                series[row["service"]].append(
                    (row["timestamp"], float(row["joules_total"])))

        raw: Dict[str, float] = {}
        for svc, pts in series.items():
            pts_s = sorted(pts, key=lambda x: x[0])
            if len(pts_s) >= 2:
                raw[svc] = max(pts_s[-1][1] - pts_s[0][1], 0.0)

        svc_data = {}
        for svc in OVERHEAD_SVCS:
            svc_data[svc] = sum(raw.get(a, 0.0) for a in OVERHEAD_ALIASES[svc])
        data[label] = svc_data

    if not data:
        print("  ⚠  no telemetry overhead data")
        return

    labels  = list(data.keys())
    x       = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    for svc, color in zip(OVERHEAD_SVCS, SVC_COLORS):
        vals = np.array([data[lbl].get(svc, 0.0) for lbl in labels])
        ax.bar(x, vals, 0.5, bottom=bottoms, label=svc,
               color=color, alpha=0.85, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 1:
                ax.text(xi, b + v / 2, f"{v:.0f}J",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += vals

    for xi, tot in enumerate(bottoms):
        ax.text(xi, tot + 1, f"Total\n{tot:.0f}J",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Energy [J]")
    ax.set_title("Telemetry Infrastructure Energy Overhead\n"
                 "(via Prometheus increase() over experiment window)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, max(bottoms) * 1.2 + 5 if len(bottoms) else 10)
    grid_style(ax)
    fig.tight_layout()
    save(fig, output_dir, "telemetry_overhead_e3_e5_e6.png")


# ══════════════════════════════════════════════════════════════════════
# VALIDATION - Std-dev over runs
# ══════════════════════════════════════════════════════════════════════

def plot_validation_std_over_runs(attributed_dir, output_dir):
    print("\n[VALIDATION] Std-dev over runs...")

    groups: Dict[str, List[Path]] = defaultdict(list)
    for exp_dir in attributed_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if not (exp_dir / "attribution_statistics.json").exists():
            continue
        groups[parse_exp_key(exp_dir.name)].append(exp_dir)

    # E1: group by N (mean across 4 actions), then concurrent experiments
    interesting_e1 = [f"E1_isolated_{a}_N{n}"
                      for n in N_VALUES for a in ACTIONS]
    rows = []

    # E1: pool all 4 actions per N → one bar per N
    for n in N_VALUES:
        keys_n = [f"E1_isolated_{a}_N{n}" for a in ACTIONS]
        all_dirs = [d for k in keys_n for d in groups.get(k, [])]
        if len(all_dirs) < 2:
            continue
        rates = [load_attribution_rate(d) * 100 for d in all_dirs]
        means = []
        for d in all_dirs:
            s = load_json(d / "attribution_statistics.json")
            if s:
                means.append(s["trace_energy_distribution"]["mean"] * 1000)
        if means:
            rows.append({
                "key":         f"E1_N{n}",
                "label":       f"Isolated N={n}",
                "rate_mean":   np.mean(rates),
                "rate_std":    np.std(rates),
                "energy_mean": np.mean(means),
                "energy_std":  np.std(means),
                "group":       "E1",
            })

    # Concurrent experiments
    for key in CONCURRENT_EXPERIMENTS:
        dirs = groups.get(key, [])
        if len(dirs) < 2:
            continue
        rates = [load_attribution_rate(d) * 100 for d in dirs]
        means = []
        for d in dirs:
            s = load_json(d / "attribution_statistics.json")
            if s:
                means.append(s["trace_energy_distribution"]["mean"] * 1000)
        if means:
            rows.append({
                "key":         key,
                "label":       EXP_LABELS.get(key, key),
                "rate_mean":   np.mean(rates),
                "rate_std":    np.std(rates),
                "energy_mean": np.mean(means),
                "energy_std":  np.std(means),
                "group":       "concurrent",
            })

    if not rows:
        print("  ⚠  need >=2 runs per experiment")
        return

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
    ax1.bar(x, [r["energy_mean"] for r in rows],
            yerr=[r["energy_std"] for r in rows],
            capsize=5, color=COLORS["interval"], alpha=0.85, edgecolor="white",
            error_kw={"elinewidth": 2})
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Energy per Trace [mJ]")
    ax1.set_title("Stability: Mean per-Trace Energy across Runs (mean \u00b1 std)\n"
                  "Isolated requests: mean across 4 user actions per N")
    ax1.set_yscale("log")
    grid_style(ax1)

    fig.tight_layout()
    save(fig, output_dir, "validation_std_over_runs.png")


# ══════════════════════════════════════════════════════════════════════
# VALIDATION - Per-service attribution rate (E3)
# ══════════════════════════════════════════════════════════════════════

def plot_validation_per_service_coverage(best, output_dir):
    """Per-service attribution rate for ALL experiments (one subplot each)."""
    print("\n[VALIDATION] Per-service attribution rate (all experiments)...")

    exp_ids = ["E3"] + CONCURRENT_EXPERIMENTS
    exp_ids = list(dict.fromkeys(exp_ids))  # deduplicate, keep order

    for exp_id in exp_ids:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            print(f"  {exp_id} not found, skipping")
            continue
        stats = load_json(exp_dir / "attribution_statistics.json")
        if stats is None:
            continue

        rows = []
        for svc, d in stats.get("per_service", {}).items():
            induced    = d.get("total_induced_J", 0.0)
            attributed = d.get("energy_attributed_J", 0.0)
            if induced < 0.05:
                continue
            rate = attributed / induced if induced > 0 else 0.0
            rows.append((svc, rate * 100, induced))

        if not rows:
            print(f"  {exp_id}: no per-service data")
            continue

        rows.sort(key=lambda r: r[1])
        labels  = [r[0] for r in rows]
        rates   = [r[1] for r in rows]
        induced = [r[2] for r in rows]
        colors  = ["#EF5350" if r < 20 else "#FFA726" if r < 50 else "#66BB6A"
                   for r in rates]

        n     = len(rows)
        fig_h = max(5, n * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y    = np.arange(n)
        bars = ax.barh(y, rates, height=0.5, color=colors,
                       alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Attribution Rate [%]")
        ax.set_title(f"{EXP_LABELS.get(exp_id, exp_id)}: Per-Service Attribution Rate\n(Interval-Duration Method)")
        ax.set_xlim(0, 115)
        ax.axvline(100, color="grey", lw=0.8, ls=":")
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar, rate_val, ind in zip(bars, rates, induced):
            w  = bar.get_width()
            cy = bar.get_y() + bar.get_height() / 2
            ax.text(max(w - 1, 1), cy, f"{rate_val:.0f}%",
                    va="center", ha="right", fontsize=7,
                    color="white", fontweight="bold")
            ax.text(w + 1, cy, f"{ind:.1f} J",
                    va="center", ha="left", fontsize=7, color="#555")

        legend_patches = [
            mpatches.Patch(color="#66BB6A", label=">50% attributed"),
            mpatches.Patch(color="#FFA726", label="20-50% attributed"),
            mpatches.Patch(color="#EF5350", label="<20% attributed"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
        fig.tight_layout()
        fname = f"validation_per_service_coverage_{exp_id.lower()}.png"
        save(fig, output_dir, fname)


def plot_validation_coverage_over_runs(attributed_dir, output_dir):
    """Scatter: coverage per run. E1 grouped by N (mean across 4 actions)."""
    print("\n[VALIDATION] Coverage over runs (scatter)...")
    import re as _re

    all_groups = defaultdict(list)
    for exp_dir in attributed_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        r = load_attribution_rate(exp_dir)
        if r >= 0:
            all_groups[parse_exp_key(exp_dir.name)].append(r * 100)

    # Group E1 by N: merge all 4 actions per N into one group
    n_groups = defaultdict(list)
    for key, vals in all_groups.items():
        if key.startswith("E1_isolated_"):
            m = _re.search(r"_N(\d+)$", key)
            if m:
                n_groups[f"Isolated N={m.group(1)}"].extend(vals)
        else:
            # Use human-readable label for concurrent experiments
            n_groups[EXP_LABELS.get(key, key)].extend(vals)

    multi = {k: v for k, v in n_groups.items() if len(v) >= 2}
    if not multi:
        print("  no data with >=2 runs")
        return

    # Order: E1 isolated by N, then concurrent experiments
    order = ([f"Isolated N={n}" for n in N_VALUES
              if f"Isolated N={n}" in multi]
             + [EXP_LABELS[e] for e in CONCURRENT_EXPERIMENTS
                if EXP_LABELS[e] in multi])
    if not order:
        order = sorted(multi.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.1), 5))
    for i, key in enumerate(order):
        vals = sorted(multi[key])
        ax.plot([i] * len(vals), vals, "o",
                color=COLORS["coverage"], alpha=0.6, markersize=7)
        mean_v = np.mean(vals)
        ax.plot([i - 0.2, i + 0.2], [mean_v, mean_v],
                "-", color=COLORS["coverage"], lw=3)
        ax.text(i, mean_v + 1.5, f"{mean_v:.0f}%",
                ha="center", fontsize=7, color=COLORS["coverage"])

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Attribution Coverage [%]")
    ax.set_title("Attribution Coverage across Runs\n"
                 "(dots = individual runs, bar = mean;\n"
                 " Isolated: each group pools all 4 actions at same N)")
    ax.set_ylim(0, 105)
    grid_style(ax)
    fig.tight_layout()
    save(fig, output_dir, "validation_coverage_over_runs.png")


def plot_boxplots_e1(best, output_dir):
    print("\n[BOXPLOT] E1 request energy distribution...")

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, action in zip(axes, ACTIONS):
        all_vals, x_labels = [], []
        temporal_refs = []   # temporal avg per N for reference lines
        for n in N_VALUES:
            exp_dir = best.get(f"E1_isolated_{action}_N{n}")
            if exp_dir is None:
                continue
            vals = load_trace_energies_mJ(exp_dir)
            t_val = e1_temporal(best, action, n)
            if vals:
                all_vals.append(vals)
                x_labels.append(f"Isolated N={n}\n(n={len(vals)})")
                temporal_refs.append(t_val)

        if all_vals:
            bp = ax.boxplot(all_vals, patch_artist=True, notch=False,
                            showfliers=True,
                            flierprops={"marker": ".", "markersize": 3,
                                        "alpha": 0.4})
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS["interval"])
                patch.set_alpha(0.7)
            # Add temporal reference markers (diamond) per N
            for xi, t_val in enumerate(temporal_refs, start=1):
                if t_val is not None:
                    ax.plot(xi, t_val, marker="D", color=COLORS["temporal"],
                            markersize=5, zorder=5,
                            label="Temporal Diff." if xi == 1 else "")

        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(ACTION_LABELS[action], fontweight="bold")
        ax.set_ylabel("Energy [mJ]" if action == ACTIONS[0] else "")
        if action == ACTIONS[0]:
            ax.legend(fontsize=8, loc="upper right")
        grid_style(ax)

    fig.suptitle("Isolated Requests: Per-Request Energy Distribution by Action and N\n"
                 "(Interval-Duration boxplot; diamond = Temporal Diff. mean)")
    fig.tight_layout()
    save(fig, output_dir, "boxplot_e1_by_n.png")

    # ── Log-scale version ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax2, action in zip(axes2, ACTIONS):
        all_vals2, x_labels2 = [], []
        for n in N_VALUES:
            exp_dir = best.get(f"E1_isolated_{action}_N{n}")
            if exp_dir is None:
                continue
            vals = load_trace_energies_mJ(exp_dir)
            if vals:
                all_vals2.append([max(v, 0.01) for v in vals])
                x_labels2.append(f"Isolated N={n}\n(n={len(vals)})")
        if all_vals2:
            bp2 = ax2.boxplot(all_vals2, patch_artist=True, notch=False,
                              showfliers=True,
                              flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
            for patch in bp2["boxes"]:
                patch.set_facecolor(COLORS["interval"])
                patch.set_alpha(0.7)
            for xi, n in enumerate([n for n in N_VALUES
                                     if best.get(f"E1_isolated_{action}_N{n}")], start=1):
                t_val = e1_temporal(best, action, n)
                if t_val is not None and t_val > 0:
                    ax2.plot(xi, t_val, marker="D", color=COLORS["temporal"],
                             markersize=5, zorder=5,
                             label="Temporal Diff." if xi == 1 else "")
            ax2.set_yscale("log")
        ax2.set_xticklabels(x_labels2, fontsize=8)
        ax2.set_title(ACTION_LABELS[action], fontweight="bold")
        ax2.set_ylabel("Energy [mJ] (log)" if action == ACTIONS[0] else "")
        if action == ACTIONS[0]:
            ax2.legend(fontsize=8, loc="upper right")
        grid_style(ax2)
    fig2.suptitle("Isolated Requests: Per-Request Energy Distribution by Action and N (log scale)\n"
                  "(Interval-Duration boxplot; diamond = Temporal Diff. mean)")
    fig2.tight_layout()
    save(fig2, output_dir, "boxplot_e1_by_n_log.png")

    # ── Version without temporal at N=1 (removes N=1 distortion) ─────
    fig3, axes3 = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
    for ax3, action in zip(axes3, ACTIONS):
        all_vals3, x_labels3, t_refs3 = [], [], []
        for n in [10, 100, 1000]:   # skip N=1
            exp_dir = best.get(f"E1_isolated_{action}_N{n}")
            if exp_dir is None:
                continue
            vals = load_trace_energies_mJ(exp_dir)
            t_val = e1_temporal(best, action, n)
            if vals:
                all_vals3.append(vals)
                x_labels3.append(f"Isolated N={n}\n(n={len(vals)})")
                t_refs3.append(t_val)
        if all_vals3:
            bp3 = ax3.boxplot(all_vals3, patch_artist=True, notch=False,
                              showfliers=True,
                              flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
            for patch in bp3["boxes"]:
                patch.set_facecolor(COLORS["interval"])
                patch.set_alpha(0.7)
            for xi, t_val in enumerate(t_refs3, start=1):
                if t_val is not None:
                    ax3.plot(xi, t_val, marker="D", color=COLORS["temporal"],
                             markersize=5, zorder=5,
                             label="Temporal Diff." if xi == 1 else "")
        ax3.set_xticklabels(x_labels3, fontsize=8)
        ax3.set_title(ACTION_LABELS[action], fontweight="bold")
        ax3.set_ylabel("Energy [mJ]" if action == ACTIONS[0] else "")
        if action == ACTIONS[0]:
            ax3.legend(fontsize=8, loc="upper right")
        grid_style(ax3)
    fig3.suptitle("Isolated Requests: Energy Distribution by Action — N=10/100/1000\n"
                  "(N=1 excluded; Interval-Duration boxplot; diamond = Temporal Diff. mean)")
    fig3.tight_layout()
    save(fig3, output_dir, "boxplot_e1_by_n_no_n1.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    all_vals, x_labels = [], []
    for action in ACTIONS:
        exp_dir = best.get(f"E1_isolated_{action}_N1000")
        if exp_dir is None:
            continue
        vals = load_trace_energies_mJ(exp_dir)
        if vals:
            all_vals.append(vals)
            x_labels.append(f"{ACTION_LABELS[action]}\n(n={len(vals)})")

    if all_vals:
        bp = ax.boxplot(all_vals, patch_artist=True, notch=False,
                        showfliers=True,
                        flierprops={"marker": ".", "markersize": 4,
                                    "alpha": 0.5})
        for patch, color in zip(bp["boxes"], ACTION_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Energy per Request [mJ]")
    ax.set_title("Isolated Requests (N=1000): Per-Request Energy Distribution by User Action\n"
                 "(Interval-Duration Method)")
    grid_style(ax)
    fig.tight_layout()
    save(fig, output_dir, "boxplot_e1_actions_n1000.png")


# ══════════════════════════════════════════════════════════════════════
# BOXPLOTS - E2-E6: per action and pooled
# ══════════════════════════════════════════════════════════════════════

def plot_boxplots_concurrent(best, output_dir):
    print("\n[BOXPLOT] E2-E6 request energy distribution...")

    present = [(eid, best[eid])
               for eid in CONCURRENT_EXPERIMENTS if eid in best]
    if not present:
        return

    action_color_map = dict(zip(ACTIONS, ACTION_COLORS))

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (exp_id, exp_dir) in zip(axes, present):
        by_action = load_trace_energies_by_action(exp_dir)
        vals_list, labels_bp, colors_bp = [], [], []
        for action in ACTIONS:
            vals = by_action.get(action, [])
            if vals:
                vals_list.append(vals)
                labels_bp.append(
                    f"{ACTION_LABELS[action]}\n(n={len(vals)})")
                colors_bp.append(action_color_map[action])

        if vals_list:
            bp = ax.boxplot(vals_list, patch_artist=True, notch=False,
                            showfliers=True,
                            flierprops={"marker": ".", "markersize": 3,
                                        "alpha": 0.4})
            for patch, color in zip(bp["boxes"], colors_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
        ax.set_xticklabels(labels_bp, fontsize=8, rotation=15, ha="right")
        ax.set_title(EXP_LABELS.get(exp_id, exp_id), fontweight="bold")
        ax.set_ylabel("Energy [mJ]" if exp_id == present[0][0] else "")
        grid_style(ax)

    fig.suptitle("Concurrent Experiments: Per-Request Energy Distribution by User Action\n"
                 "(Interval-Duration Method)")
    fig.tight_layout()
    save(fig, output_dir, "boxplot_concurrent_by_action.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    all_vals, x_labels = [], []
    for exp_id, exp_dir in present:
        vals = load_trace_energies_mJ(exp_dir)
        if vals:
            all_vals.append(vals)
            x_labels.append(f"{exp_id}\n(n={len(vals)})")

    if all_vals:
        bp = ax.boxplot(all_vals, patch_artist=True, notch=False,
                        showfliers=True,
                        flierprops={"marker": ".", "markersize": 3,
                                    "alpha": 0.4})
        for patch, color in zip(bp["boxes"], EXP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Energy per Trace [mJ]")
    ax.set_title("Per-Trace Energy Distribution — Concurrent Experiments (pooled)\n"
                 "(Interval-Duration Method)")
    grid_style(ax)
    fig.tight_layout()
    save(fig, output_dir, "boxplot_concurrent_all_exps.png")

    # ── Log-scale version ─────────────────────────────────────────────
    if all_vals:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        all_vals_log = [[max(v, 0.01) for v in vals] for vals in all_vals]
        bp2 = ax2.boxplot(all_vals_log, patch_artist=True, notch=False,
                          showfliers=True,
                          flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
        for patch, color in zip(bp2["boxes"], EXP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax2.set_yscale("log")
        ax2.set_xticklabels(x_labels, fontsize=9)
        ax2.set_ylabel("Energy per Trace [mJ] (log scale)")
        ax2.set_title("Per-Trace Energy Distribution — Concurrent Experiments (pooled)\n"
                      "(Interval-Duration Method, log scale)")
        grid_style(ax2)
        fig2.tight_layout()
        save(fig2, output_dir, "boxplot_concurrent_all_exps_log.png")



# ══════════════════════════════════════════════════════════════════════
# NEW: All experiments action energy in one combined plot
# ══════════════════════════════════════════════════════════════════════

def plot_all_experiments_action_energy(best, output_dir):
    """All 4 actions mean energy for E2-E6. Generated twice: N=100 and N=1000 baseline."""
    print("\n[COMBINED] All experiments action energy (N=100 and N=1000 baselines)...")
    from matplotlib.lines import Line2D

    PLOT_ACTIONS  = ACTIONS
    action_colors = dict(zip(ACTIONS, ACTION_COLORS))

    # Experiment config annotations (only for E2-E5, not E6)
    EXP_CONFIGS = {
        "E2": "2 users | 1s / 100%",
        "E3": "10 users | 1s / 100%",
        "E4": "50 users | 1s / 100%",
        "E5": "10 users | 10s / 50%",
    }

    # Pre-compute exp_data once
    exp_data = {}
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        by_action = load_trace_energies_by_action(exp_dir)
        exp_data[exp_id] = {
            a: (np.mean(by_action[a]) if a in by_action and by_action[a] else 0.0,
                np.std(by_action[a])  if a in by_action and by_action[a] else 0.0)
            for a in PLOT_ACTIONS
        }
    if not exp_data:
        return

    def _draw(baseline_n, fname):
        e1_base = {a: (e1_temporal(best, a, baseline_n) or 0.0)
                   for a in PLOT_ACTIONS}
        bl_label = f"Isolated N={baseline_n} Temporal Diff. baseline"

        n_act = len(PLOT_ACTIONS)
        w = 0.18
        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                 gridspec_kw={"width_ratios": [4, 1]})

        for ax_idx, (ax, exps) in enumerate(zip(
                axes, [["E2", "E3", "E4", "E5"], ["E6"]])):
            x_e = np.arange(len(exps))
            for i, action in enumerate(PLOT_ACTIONS):
                offset = (i - n_act / 2 + 0.5) * w * 1.05
                means    = [exp_data.get(e, {}).get(action, (0.0, 0.0))[0] for e in exps]
                errs     = [exp_data.get(e, {}).get(action, (0.0, 0.0))[1] for e in exps]
                errs_low = [min(e, m) for m, e in zip(means, errs)]
                color    = action_colors[action]
                bars = ax.bar(x_e + offset, means, w,
                              yerr=[errs_low, errs], capsize=3,
                              label=ACTION_LABELS[action] if ax_idx == 0 else "",
                              color=color, alpha=0.82, edgecolor="white",
                              error_kw={"elinewidth": 1.2})
                bl = e1_base[action]
                for xi in x_e:
                    ax.plot([xi + offset - w/2, xi + offset + w/2], [bl, bl],
                            color=color, lw=1.5, ls=":", alpha=0.7)
                annotate_bars(ax, bars, fmt="{:.0f}", fontsize=8)

            # Experiment config annotation below x-axis (E2-E5 only, not E6)
            ax.set_xticks(x_e)
            ax.set_xticklabels([EXP_LABELS.get(e, e) for e in exps], fontsize=11)
            if ax_idx == 0:
                for xi, exp_id in enumerate(exps):
                    cfg = EXP_CONFIGS.get(exp_id, "")
                    if cfg:
                        ax.annotate(cfg,
                                    xy=(xi, 0),
                                    xycoords=("data", "axes fraction"),
                                    xytext=(0, -28),
                                    textcoords="offset points",
                                    ha="center", fontsize=8,
                                    color="#555", style="italic")

            ax.set_ylabel("Mean Energy per Trace [mJ]" if ax_idx == 0 else "")
            ax.set_ylim(bottom=0)
            grid_style(ax)
            if ax_idx == 1:
                ax.set_title("Telemetry Low\n(60s / 1%)", fontsize=11)

        handles, lbls = axes[0].get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="grey", lw=1.5, ls=":", alpha=0.7))
        lbls.append(bl_label)
        axes[0].legend(handles, lbls, fontsize=10, framealpha=0.9,
                       loc="upper left", ncol=2)
        axes[0].set_title("Workload Low–High / Telemetry Medium: Mean Per-Request Energy", fontsize=12)
        # suptitle removed per user request
        fig.subplots_adjust(bottom=0.18)
        save(fig, output_dir, fname)

    _draw(100,  "all_exps_action_energy_combined_n100.png")
    _draw(1000, "all_exps_action_energy_combined_n1000.png")


# ══════════════════════════════════════════════════════════════════════
# NEW: Attribution coverage grouped by N
# ══════════════════════════════════════════════════════════════════════

def plot_attribution_coverage_by_n(best, output_dir):
    """Coverage for E1 grouped by N (mean across 4 actions) + E2-E6."""
    print("\n[COVERAGE] Coverage grouped by N...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    n_groups = {n: [] for n in N_VALUES}
    for action in ACTIONS:
        for n in N_VALUES:
            exp_dir = best.get(f"E1_isolated_{action}_N{n}")
            if exp_dir:
                r = load_attribution_rate(exp_dir)
                if r >= 0:
                    n_groups[n].append(r * 100)
    x = np.arange(len(N_VALUES))
    means = [np.mean(n_groups[n]) if n_groups[n] else 0 for n in N_VALUES]
    stds  = [np.std(n_groups[n])  if n_groups[n] else 0 for n in N_VALUES]
    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=5,
                  color=COLORS["coverage"], alpha=0.85, edgecolor="white",
                  error_kw={"elinewidth": 2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{m:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Isolated\nN={n}" for n in N_VALUES])
    ax.set_ylabel("Attribution Coverage [%]")
    ax.set_title("Isolated Requests: Coverage by N\n(mean±std across 4 actions)")
    ax.set_ylim(0, 105)
    grid_style(ax)

    ax2 = axes[1]
    rates, clabels = [], []
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir:
            r = load_attribution_rate(exp_dir)
            if r >= 0:
                rates.append(r*100); clabels.append(exp_id)
    x2 = np.arange(len(clabels))
    bars2 = ax2.bar(x2, rates, 0.5, color=EXP_COLORS[:len(clabels)],
                    alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, rates):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x2); ax2.set_xticklabels(clabels)
    ax2.set_ylabel("Attribution Coverage [%]")
    ax2.set_title("Concurrent: Coverage\n(best run)")
    ax2.set_ylim(0, 105)
    grid_style(ax2)

    fig.suptitle("Attribution Coverage (Interval-Duration Method)", y=1.02, fontweight="bold")
    fig.tight_layout()
    save(fig, output_dir, "attribution_coverage_by_n.png")


# ══════════════════════════════════════════════════════════════════════
# NEW: Duration vs. Energy scatter
# ══════════════════════════════════════════════════════════════════════

def plot_duration_vs_energy(best, output_dir):
    """Scatter: trace duration vs. attributed energy (single plot).
    Individual requests as dots colored by action.
    Group means shown as diamonds with labels.
    Three versions: p99-filtered, IQR no-outlier, log-scale."""
    print("\n[SCATTER] Duration vs. Energy...")

    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        te = load_json(exp_dir / "trace_energies.json")
        if te is None:
            continue

        all_traces = [t for t in te.get("traces", [])
                      if t["total_energy_J"] > 0 and t.get("duration_ms", 0) > 0]
        if not all_traces:
            continue

        action_color_map = dict(zip(ACTIONS, ACTION_COLORS))

        def make_scatter(ax, traces, title):
            """Draw scatter on ax with group-mean diamonds."""
            seen_labels = set()
            for t in traces:
                raw    = t.get("user_action", "unknown")
                action = ACTION_LABEL_MAP.get(raw, raw)
                color  = action_color_map.get(action, "#aaa")
                label  = ACTION_LABELS.get(action, action)
                ax.scatter(t["duration_ms"], t["total_energy_J"]*1000,
                           c=color, alpha=0.35, s=14,
                           label=label if label not in seen_labels else "")
                seen_labels.add(label)

            xs = np.array([t["duration_ms"] for t in traces])
            ys = np.array([t["total_energy_J"]*1000 for t in traces])
            r = 0.0
            if len(xs) > 10:
                z = np.polyfit(xs, ys, 1)
                r = np.corrcoef(xs, ys)[0, 1]
                xs_l = np.linspace(xs.min(), xs.max(), 100)
                ax.plot(xs_l, np.poly1d(z)(xs_l), "k--", lw=1.5, alpha=0.7,
                        label=f"Linear trend (r={r:.2f})")

            # Group-mean diamonds – ALWAYS on top
            for action, color in zip(ACTIONS, ACTION_COLORS):
                at = [t for t in traces
                      if ACTION_LABEL_MAP.get(
                          t.get("user_action",""), t.get("user_action","")) == action]
                if not at:
                    continue
                xs_a = np.array([t["duration_ms"] for t in at])
                ys_a = np.array([t["total_energy_J"]*1000 for t in at])
                ax.scatter(np.mean(xs_a), np.mean(ys_a),
                           c=color, s=14, marker="D",
                           edgecolors="black", linewidths=0.6, zorder=5,
                           label=(f"{ACTION_LABELS.get(action,action)} mean "
                                  f"(n={len(at)}, "
                                  f"μ\u0435={np.mean(ys_a):.0f}mJ, "
                                  f"{np.mean(xs_a):.0f}ms)"))

            handles, labels_l = ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labels_l):
                if l and l not in seen:
                    seen[l] = h
            # legend_outside: if set, place legend below figure instead of inside
            if getattr(ax, "_legend_outside", False):
                ax.get_figure().legend(
                    seen.values(), seen.keys(),
                    fontsize=8, framealpha=0.9, markerscale=1.2,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.08),
                    ncol=min(4, len(seen)),
                )
            else:
                ax.legend(seen.values(), seen.keys(), fontsize=7.5,
                          framealpha=0.9, markerscale=1.5)
            ax.set_xlabel("Trace Duration [ms]")
            ax.set_ylabel("Energy per Trace [mJ]")
            ax.set_title(title)
            grid_style(ax)

        # ── Version 1: p99-filtered ───────────────────────────────────
        e99 = np.percentile([t["total_energy_J"]*1000 for t in all_traces], 99)
        d99 = np.percentile([t["duration_ms"] for t in all_traces], 99)
        traces_p99 = [t for t in all_traces
                      if t["total_energy_J"]*1000 <= e99 and t["duration_ms"] <= d99]
        if len(traces_p99) >= 5:
            fig, ax = plt.subplots(figsize=(9, 5))
            make_scatter(ax, traces_p99,
                         f"{EXP_LABELS.get(exp_id, exp_id)}: Trace Duration vs. Energy\n"
                         f"(p99 filtered, n={len(traces_p99)}; "
                         f"diamonds = mean per user action)")
            fig.tight_layout()
            save(fig, output_dir, f"{exp_id.lower()}_duration_vs_energy.png")

        # ── Version 2: IQR no-outlier ─────────────────────────────────
        all_e = np.array([t["total_energy_J"]*1000 for t in all_traces])
        all_d = np.array([t["duration_ms"] for t in all_traces])
        e_cap = np.percentile(all_e, 75) + 1.5*(np.percentile(all_e,75)-np.percentile(all_e,25))
        d_cap = np.percentile(all_d, 75) + 1.5*(np.percentile(all_d,75)-np.percentile(all_d,25))
        traces_iqr = [t for t in all_traces
                      if t["total_energy_J"]*1000 <= e_cap and t["duration_ms"] <= d_cap]
        if len(traces_iqr) >= 5:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax._legend_outside = True   # legend below plot
            make_scatter(ax, traces_iqr,
                         f"{EXP_LABELS.get(exp_id, exp_id)}: Trace Duration vs. Energy (IQR-filtered)\n"
                         f"(n={len(traces_iqr)}; diamonds = mean per user action)")
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.14)
            save(fig, output_dir, f"{exp_id.lower()}_duration_vs_energy_noout.png")

        # ── Version 3: log-scale, all data ───────────────────────────
        if len(all_traces) >= 5:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax._legend_outside = True   # place legend below plot
            make_scatter(ax, all_traces,
                         f"{EXP_LABELS.get(exp_id, exp_id)}: Trace Duration vs. Energy (log scale)\n"
                         f"(n={len(all_traces)}; diamonds = mean per user action)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Trace Duration [ms] (log scale)")
            ax.set_ylabel("Energy per Trace [mJ] (log scale)")
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.14)
            save(fig, output_dir, f"{exp_id.lower()}_duration_vs_energy_log.png")



# ══════════════════════════════════════════════════════════════════════
# SPAN-LEVEL vs REQUEST-LEVEL: service breakdown per trace
# ══════════════════════════════════════════════════════════════════════

def plot_span_vs_request_energy(best, output_dir):
    """
    Plot 1 – E3: Top-3 most energy-intensive traces per user action.
    Stacked bars show how energy is distributed across services within each trace.

    Plot 2 – E2-E4: Mean per-request energy by user action, stacked by service.
    Shows which services dominate energy across concurrency levels.
    """
    print("\n[SPAN vs REQUEST] Service breakdown plots...")

    # ── Shared colour palette for services ───────────────────────────
    # We use a fixed palette; unknown services get grey
    SERVICE_PALETTE = {
        "frontend":          "#2196F3",
        "frontend-proxy":    "#1565C0",
        "cart":              "#4CAF50",
        "product-catalog":   "#8BC34A",
        "checkout":          "#9C27B0",
        "payment":           "#E91E63",
        "email":             "#F44336",
        "shipping":          "#FF5722",
        "currency":          "#FF9800",
        "recommendation":    "#FFC107",
        "ad":                "#FFEB3B",
        "fraud-detection":   "#795548",
        "accounting":        "#607D8B",
        "quote":             "#9E9E9E",
        "valkey-cart":       "#00BCD4",
        "postgresql":        "#3F51B5",
        "kafka":             "#673AB7",
        "flagd":             "#009688",
    }

    def svc_color(svc):
        return SERVICE_PALETTE.get(svc, "#BDBDBD")

    # ════════════════════════════════════════════════════════════════
    # PLOT 1: E3 – top-3 traces per user action, stacked by service
    # ════════════════════════════════════════════════════════════════
    exp_dir = best.get("E3")
    if exp_dir is not None:
        te = load_json(exp_dir / "trace_energies.json")
        if te is not None:
            # Build per-action top-3
            by_action = defaultdict(list)
            for t in te.get("traces", []):
                ua = ACTION_LABEL_MAP.get(t.get("user_action", ""), "background")
                if ua in ACTIONS and t["total_energy_J"] > 0:
                    by_action[ua].append(t)

            present_actions = [a for a in ACTIONS if a in by_action]
            n_cols = len(present_actions)
            # Collect all services across all actions for consistent legend
            all_svcs_all_actions = set()
            for traces_for_action in by_action.values():
                for t in sorted(traces_for_action,
                                key=lambda t: t["total_energy_J"],
                                reverse=True)[:3]:
                    all_svcs_all_actions.update(t.get("services", {}).keys())

            if n_cols > 0:
                fig, axes = plt.subplots(1, n_cols,
                                         figsize=(4.5 * n_cols, 5),
                                         sharey=False)
                if n_cols == 1:
                    axes = [axes]
                # Build shared legend patches once from full service set
                _svcs_ordered = [s for s in SERVICE_PALETTE
                                  if s in all_svcs_all_actions]
                _svcs_ordered += [s for s in all_svcs_all_actions
                                   if s not in SERVICE_PALETTE]
                _last_legend_patches = [
                    mpatches.Patch(color=svc_color(s), label=s)
                    for s in _svcs_ordered
                ]

                for ax, action in zip(axes, present_actions):
                    top3 = sorted(by_action[action],
                                  key=lambda t: t["total_energy_J"],
                                  reverse=True)[:3]
                    if not top3:
                        continue

                    # Collect all services that appear
                    all_svcs = sorted(
                        {s for t in top3 for s in t.get("services", {})},
                        key=lambda s: -max(
                            t["services"].get(s, {}).get("energy_J", 0)
                            for t in top3))

                    x = np.arange(len(top3))
                    bottoms = np.zeros(len(top3))
                    legend_patches = []

                    # Use fixed palette order for colour consistency across subplots
                    svc_draw_order = [s for s in SERVICE_PALETTE if s in all_svcs]
                    svc_draw_order += [s for s in all_svcs if s not in SERVICE_PALETTE]
                    for svc in svc_draw_order:
                        vals = np.array([
                            t["services"].get(svc, {}).get("energy_J", 0) * 1000
                            for t in top3])
                        color = svc_color(svc)
                        ax.bar(x, vals, bottom=bottoms, color=color,
                               alpha=0.85, edgecolor="white", width=0.6,
                               label=svc)
                        for xi, (v, b) in enumerate(zip(vals, bottoms)):
                            if v > 0.5:
                                ax.text(xi, b + v / 2,
                                        f"{v:.1f}", ha="center",
                                        va="center", fontsize=6.5,
                                        color="white", fontweight="bold")
                        bottoms += vals
                        legend_patches.append(
                            mpatches.Patch(color=color, label=svc))

                    # Total label on top
                    for xi, tot in enumerate(bottoms):
                        ax.text(xi, tot + 0.5, f"{tot:.1f} mJ",
                                ha="center", va="bottom", fontsize=8,
                                fontweight="bold")

                    labels = [f"Trace {i+1}\n({t['duration_ms']:.0f}ms)"
                              for i, t in enumerate(top3)]
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, fontsize=9)
                    ax.set_ylabel("Energy [mJ]" if action == present_actions[0]
                                  else "")
                    ax.set_title(ACTION_LABELS[action], fontweight="bold")
                    # Store for shared legend (set later)
                    _last_legend_patches = legend_patches
                    grid_style(ax)

                fig.suptitle(
                    "Workload Medium: Top-3 Most Energy-Intensive Traces per User Action\n"
                    "(stacked by service, interval-duration attribution)",
                    fontsize=12, y=1.02)
                # Single shared legend outside the plot area
                if _last_legend_patches:
                    fig.legend(handles=_last_legend_patches,
                               title="Service", fontsize=8,
                               loc="lower center",
                               bbox_to_anchor=(0.5, -0.12),
                               ncol=min(6, len(_last_legend_patches)),
                               framealpha=0.9)
                fig.tight_layout()
                save(fig, output_dir, "e3_top3_traces_service_breakdown.png")

    # ════════════════════════════════════════════════════════════════
    # PLOT 2: E2-E4 – mean energy per action, stacked by service
    # ════════════════════════════════════════════════════════════════
    target_exps = ["E2", "E3", "E4"]
    exp_dirs = {e: best[e] for e in target_exps if e in best}

    if exp_dirs:
        # Collect mean energy per service per action per experiment
        # data[exp_id][action] = {service: mean_mJ}
        data = {}
        all_svcs_global = set()

        for exp_id, exp_dir in exp_dirs.items():
            te = load_json(exp_dir / "trace_energies.json")
            if te is None:
                continue
            # Group traces by action
            by_action = defaultdict(list)
            for t in te.get("traces", []):
                ua = ACTION_LABEL_MAP.get(t.get("user_action",""), "background")
                if ua in ACTIONS and t["total_energy_J"] > 0:
                    by_action[ua].append(t)

            data[exp_id] = {}
            for action, traces in by_action.items():
                svc_totals = defaultdict(list)
                for t in traces:
                    for svc, sdata in t.get("services", {}).items():
                        svc_totals[svc].append(
                            sdata.get("energy_J", 0) * 1000)
                data[exp_id][action] = {
                    svc: np.mean(vals)
                    for svc, vals in svc_totals.items()
                    if np.mean(vals) > 0.1}
                all_svcs_global.update(data[exp_id][action].keys())

        if data:
            present_actions = [a for a in ACTIONS
                               if any(a in data[e] for e in data)]
            n_act = len(present_actions)
            fig, axes = plt.subplots(1, n_act,
                                     figsize=(4.5 * n_act, 5),
                                     sharey=True)
            if n_act == 1:
                axes = [axes]

            # Fixed palette draw order – same colour = same service in ALL bars
            svc_draw_order = [s for s in SERVICE_PALETTE if s in all_svcs_global]
            svc_draw_order += [s for s in all_svcs_global if s not in SERVICE_PALETTE]

            # Build shared legend once from ALL services (not filtered per subplot)
            _last_svc_patches = [
                mpatches.Patch(color=svc_color(s), label=s)
                for s in svc_draw_order
            ]

            for ax, action in zip(axes, present_actions):
                x = np.arange(len(exp_dirs))
                bottoms = np.zeros(len(exp_dirs))
                for svc in svc_draw_order:
                    vals = np.array([
                        data.get(e, {}).get(action, {}).get(svc, 0.0)
                        for e in exp_dirs])
                    if vals.sum() < 0.1:
                        continue
                    color = svc_color(svc)
                    ax.bar(x, vals, bottom=bottoms, color=color,
                           alpha=0.85, edgecolor="white", width=0.5)
                    for xi, (v, b) in enumerate(zip(vals, bottoms)):
                        if v > 1.0:
                            ax.text(xi, b + v / 2, f"{v:.0f}",
                                    ha="center", va="center",
                                    fontsize=7, color="white",
                                    fontweight="bold")
                    bottoms += vals

                # Total on top
                for xi, tot in enumerate(bottoms):
                    if tot > 0:
                        ax.text(xi, tot + 0.5, f"{tot:.0f} mJ",
                                ha="center", va="bottom",
                                fontsize=8, fontweight="bold")

                ax.set_xticks(x)
                ax.set_xticklabels([EXP_LABELS.get(e, e) for e in exp_dirs.keys()], fontsize=10)
                ax.set_title(ACTION_LABELS[action], fontweight="bold")
                ax.set_ylabel("Mean Energy per Trace [mJ]"
                              if action == present_actions[0] else "")
                grid_style(ax)

            # suptitle removed per user request
            if _last_svc_patches:
                fig.legend(handles=_last_svc_patches,
                           title="Service", fontsize=8,
                           loc="lower center",
                           bbox_to_anchor=(0.5, -0.12),
                           ncol=min(6, len(_last_svc_patches)),
                           framealpha=0.9)
            fig.tight_layout()
            save(fig, output_dir,
                 "e2_e4_action_energy_service_breakdown.png")


# ══════════════════════════════════════════════════════════════════════
# COVERAGE vs. TEMPORAL ERROR  (E1 only)
# ══════════════════════════════════════════════════════════════════════

def plot_coverage_vs_temporal_error(best, output_dir):
    """
    E1: Coverage vs. deviation from temporal baseline (left panel only).
    Each point = one action × N combination.
    Colour = N value; label = first letter of action (B/A/V/C).
    Shows: higher N → higher coverage AND lower deviation from temporal.
    """
    print("\n[COVERAGE vs ERROR] E1 coverage vs temporal deviation...")

    rows = []
    for action in ACTIONS:
        for n in N_VALUES:
            key     = f"E1_isolated_{action}_N{n}"
            exp_dir = best.get(key)
            if exp_dir is None:
                continue
            rate = load_attribution_rate(exp_dir)
            if rate < 0:
                continue
            stats = load_json(exp_dir / "attribution_statistics.json")
            if stats is None:
                continue
            intv_mean_mJ = stats["trace_energy_distribution"]["mean"] * 1000
            temp_mJ = e1_temporal(best, action, n)
            if temp_mJ is None or temp_mJ <= 0:
                continue
            rel_error = abs(intv_mean_mJ - temp_mJ) / temp_mJ * 100
            rows.append({
                "action":    action,
                "N":         n,
                "coverage":  rate * 100,
                "rel_error": rel_error,
            })

    if not rows:
        print("  ⚠  No E1 data with both interval and temporal results")
        return

    n_colors  = {1: "#EF5350", 10: "#FF9800", 100: "#66BB6A", 1000: "#2196F3"}
    n_markers = {1: "o", 10: "s", 100: "^", 1000: "D"}
    # First letter labels for actions
    action_initials = {"browse": "B", "add_to_cart": "A",
                       "view_cart": "V", "checkout": "C"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for n in N_VALUES:
        subset = [r for r in rows if r["N"] == n]
        if not subset:
            continue
        xs = [r["coverage"]  for r in subset]
        ys = [r["rel_error"] for r in subset]
        ax.scatter(xs, ys, c=n_colors[n], marker=n_markers[n],
                   s=80, alpha=0.88, zorder=4, label=f"N={n}")
        for r in subset:
            initial = action_initials.get(r["action"], "?")
            ax.annotate(initial, (r["coverage"], r["rel_error"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=9, color=n_colors[n], fontweight="bold")

    ax.set_xlabel("Attribution Coverage [%]")
    ax.set_ylabel("Relative Deviation from Temporal Diff. [%]")
    ax.set_title("Isolated Requests: Attribution Coverage vs. Deviation from Temporal Baseline\n"
                 "(ideal: top-right = high coverage; bottom-right = low deviation)")
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.4)
    grid_style(ax)

    # Legend: N values (scatter points)
    n_legend = ax.legend(title="N (requests)", fontsize=10,
                         loc="upper right", framealpha=0.9)
    ax.add_artist(n_legend)

    # Second legend: action initials explanation
    action_patches = [
        mpatches.Patch(color="none",
                       label=f"{initial} = {ACTION_LABELS[a]}")
        for a, initial in action_initials.items()
    ]
    ax.legend(handles=action_patches, title="Labels", fontsize=9,
              loc="lower left", framealpha=0.9)

    fig.tight_layout()
    save(fig, output_dir, "e1_coverage_vs_temporal_error.png")


# ══════════════════════════════════════════════════════════════════════
# TOTAL ABSOLUTE ENERGY per experiment
# ══════════════════════════════════════════════════════════════════════

def plot_total_energy_comparison(best, attributed_dir, cluster_dir, output_dir):
    """
    Compare total measured energy (J) across experiments.

    Two plots:
    1. E2-E6 concurrent: energy with telemetry overhead subtracted.
       Shows idle vs. attributed vs. unattributed request energy.
       User count and telemetry config annotated below bars.

    2. E1 isolated by N: same breakdown for N=1/10/100/1000
       (averaged across 4 actions). Shows how more requests
       increase total energy.
    """
    print("\n[TOTAL ENERGY] Absolute energy comparison...")

    TELEMETRY_SVCS = {"jaeger", "opentelemetry-collector",
                      "prometheus-server", "kepler", "kepler-exporter"}
    TELEMETRY_ALIASES = {
        "jaeger":                  ["jaeger"],
        "opentelemetry-collector": ["opentelemetry-collector", "collector"],
        "prometheus-server":       ["prometheus-server"],
        "kepler":                  ["kepler", "kepler-exporter"],
    }

    def get_telemetry_J(exp_id, cluster_dir):
        """Read telemetry overhead from increase()-based summary or CSV delta."""
        import csv as _csv2
        # Try summary first (increase()-based, accurate)
        for p in cluster_dir.glob(f"{exp_id}/*/telemetry_overhead_summary.json"):
            s = load_json(p)
            if s and s.get("method") == "increase()" and s.get("total_overhead_J", 0) > 0:
                return s["total_overhead_J"]
        # Fallback: sum from kepler_metrics.csv using delta
        for p in cluster_dir.glob(f"{exp_id}/*/kepler_metrics.csv"):
            series = defaultdict(list)
            with open(p) as f:
                for row in _csv2.DictReader(f):
                    if row["service"] in TELEMETRY_SVCS:
                        series[row["service"]].append(
                            (row["timestamp"], float(row["joules_total"])))
            total = 0.0
            for svc, pts in series.items():
                pts_s = sorted(pts, key=lambda x: x[0])
                if len(pts_s) >= 2:
                    total += max(pts_s[-1][1] - pts_s[0][1], 0.0)
            return total
        return 0.0

    # ════════════════════════════════════════════════════════════════
    # PLOT 1: E2-E6 concurrent
    # ════════════════════════════════════════════════════════════════
    USER_COUNTS = {"E2": 2, "E3": 10, "E4": 50, "E5": 10, "E6": 10}
    TELEM_LABELS = {"E2": "1s/100%", "E3": "1s/100%", "E4": "1s/100%",
                    "E5": "10s/50%", "E6": "60s/1%"}

    # Primary (application) services only – exclude telemetry services
    PRIMARY_EXCLUDE = {"jaeger", "opentelemetry-collector", "collector",
                       "prometheus-server", "kepler", "kepler-exporter",
                       "load-generator"}

    rows = []
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        stats = load_json(exp_dir / "attribution_statistics.json")
        if stats is None:
            continue
        budget = stats.get("energy_budget", {})
        ps     = stats.get("per_service", {})

        # Sum only primary services
        primary_total_J  = sum(
            d.get("total_induced_J", 0) + d.get("unattributed_J", 0) +
            d.get("attributed_J", 0)
            for svc, d in ps.items() if svc not in PRIMARY_EXCLUDE
        )
        # Use budget values directly (already exclude nothing, but
        # the attribution pipeline only tracks OTel-visible services)
        induced = budget.get("request_induced_J", 0)
        attr_J  = budget.get("attributed_J", 0)
        idle_J  = budget.get("idle_J", 0)

        # Re-compute idle from per_service excluding telemetry
        idle_primary = sum(
            d.get("unattributed_J", 0)
            for svc, d in ps.items() if svc not in PRIMARY_EXCLUDE
        )
        # Fall back to full idle if per_service doesn't cover it
        idle_use = idle_primary if idle_primary > 0 else idle_J

        rows.append({
            "exp":      exp_id,
            "idle_J":   idle_use,
            "induced":  induced,
            "attr_J":   attr_J,
            "unattr_J": max(induced - attr_J, 0),
        })

    if rows:
        x = np.arange(len(rows))
        w = 0.55
        fig, ax = plt.subplots(figsize=(10, 6))

        idle_vals = np.array([r["idle_J"]   for r in rows])
        attr_vals = np.array([r["attr_J"]   for r in rows])
        unat_vals = np.array([r["unattr_J"] for r in rows])

        ax.bar(x, idle_vals, w, label="Idle baseline (primary services)",
               color="#B0BEC5", alpha=0.85, edgecolor="white")
        ax.bar(x, attr_vals, w, bottom=idle_vals,
               label="Request-induced — attributed",
               color=COLORS["interval"], alpha=0.85, edgecolor="white")
        ax.bar(x, unat_vals, w, bottom=idle_vals + attr_vals,
               label="Request-induced — unattributed",
               color="#A5D6A7", alpha=0.65, edgecolor="white", hatch="//")

        # Total above each bar
        for xi, r in enumerate(rows):
            tot = r["idle_J"] + r["induced"]
            ax.text(xi, tot * 1.01, f"{tot:.0f} J",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        tick_labels = []
        for r in rows:
            u   = USER_COUNTS.get(r["exp"], "?")
            tel = TELEM_LABELS.get(r["exp"], "")
            lbl = EXP_LABELS.get(r["exp"], r["exp"])
            tick_labels.append(f"{lbl}\n{u} users | {tel}")
        ax.set_xticklabels(tick_labels, fontsize=10, ha="center")

        ax.set_ylabel("Energy [J]")
        ax.set_title("Total Energy per Experiment — Primary Services Only\n"
                     "(telemetry services excluded: kepler, prometheus, jaeger, otel-collector)")
        ax.legend(fontsize=9, loc="upper left")
        ax.margins(y=0.12)
        grid_style(ax)
        fig.tight_layout()
        save(fig, output_dir, "total_energy_comparison.png")

    # ════════════════════════════════════════════════════════════════
    # PLOT 2: E1 isolated by N (mean across 4 actions)
    # ════════════════════════════════════════════════════════════════
    e1_rows = []
    for n in N_VALUES:
        action_budgets = []
        for action in ACTIONS:
            exp_dir = best.get(f"E1_isolated_{action}_N{n}")
            if exp_dir is None:
                continue
            stats = load_json(exp_dir / "attribution_statistics.json")
            if stats:
                action_budgets.append(stats.get("energy_budget", {}))
        if not action_budgets:
            continue
        e1_rows.append({
            "N":       n,
            "total_J": np.mean([b.get("total_measured_J", 0) for b in action_budgets]),
            "idle_J":  np.mean([b.get("idle_J", 0)           for b in action_budgets]),
            "induced": np.mean([b.get("request_induced_J", 0) for b in action_budgets]),
            "attr_J":  np.mean([b.get("attributed_J", 0)      for b in action_budgets]),
        })

    if e1_rows:
        x1 = np.arange(len(e1_rows))
        w  = 0.55
        fig2, ax3 = plt.subplots(figsize=(8, 6))

        idle_v  = np.array([r["idle_J"]  for r in e1_rows])
        attr_v  = np.array([r["attr_J"]  for r in e1_rows])
        unat_v  = np.array([max(r["induced"] - r["attr_J"], 0) for r in e1_rows])

        ax3.bar(x1, idle_v, w, label="Idle baseline",
                color="#B0BEC5", alpha=0.85, edgecolor="white")
        ax3.bar(x1, attr_v, w, bottom=idle_v,
                label="Request-induced — attributed",
                color=COLORS["interval"], alpha=0.85, edgecolor="white")
        ax3.bar(x1, unat_v, w, bottom=idle_v + attr_v,
                label="Request-induced — unattributed",
                color="#A5D6A7", alpha=0.65, edgecolor="white", hatch="//")

        for xi, r in enumerate(e1_rows):
            tot = r["total_J"]
            ax3.text(xi, tot * 1.01, f"{tot:.0f} J",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax3.set_xticks(x1)
        ax3.set_xticklabels([f"N={r['N']}" for r in e1_rows], fontsize=11)

        # Annotate request counts below
        for xi, r in enumerate(e1_rows):
            ax3.annotate(f"{r['N']} requests",
                         xy=(xi, 0), xycoords=("data", "axes fraction"),
                         xytext=(0, -22), textcoords="offset points",
                         ha="center", fontsize=8, color="#555", style="italic")

        ax3.set_ylabel("Energy [J]")
        ax3.set_title("Isolated Requests: Total Measured Energy by N\n"
                      "(mean across 4 user actions per N)")
        ax3.legend(fontsize=9, loc="upper left")
        ax3.margins(y=0.12)
        grid_style(ax3)
        fig2.subplots_adjust(bottom=0.15)
        save(fig2, output_dir, "total_energy_comparison_e1.png")


# ══════════════════════════════════════════════════════════════════════
# USER SESSION ENERGY (add_to_cart + checkout only)
# ══════════════════════════════════════════════════════════════════════

def plot_user_session_energy(best, cluster_dir, output_dir):
    """
    For experiments where app.user.id is present in traces:
    group energy by user session. Only add_to_cart and checkout
    traces reliably carry a user ID (Browse does not pass through
    the cart service that sets app.user.id).

    For each experiment (E3/E5 best data), shows:
    - Left: per-user total energy (add_to_cart + checkout combined)
            as horizontal bars, coloured by dominant action
    - Right: scatter of per-trace energy coloured by action,
             with mean ± std annotation
    """
    print("\n[USER SESSION] Energy per user session...")

    SESSION_ACTIONS = {"add_to_cart", "checkout"}
    action_colors   = dict(zip(ACTIONS, ACTION_COLORS))

    def extract_user_traces(cluster_exp_dir):
        """Read raw_traces.json and return list of
        {user_id, action, total_energy_J} matched against attributed data."""
        raw_p = cluster_exp_dir / "raw_traces.json"
        if not raw_p.exists():
            return []
        raw = load_json(raw_p)
        if raw is None:
            return []

        # Build trace_id → user_id map
        tid_to_user = {}
        for trace in raw.get("traces", []):
            uid = None
            for span in trace.get("spans", []):
                for tag in span.get("tags", []):
                    if tag.get("key") in ("app.user.id", "user.id",
                                          "enduser.id"):
                        uid = str(tag.get("value", ""))
                        break
                if uid:
                    break
            if uid:
                tid_to_user[trace.get("traceID", "")] = uid
        return tid_to_user

    # Find the raw cluster_experiments path for each best exp
    def find_cluster_exp_dir(exp_id, exp_dir_path):
        """attributed_data/E3_20260322_032451 →
           cluster_experiments/E3/E3_20260322_032451"""
        name = exp_dir_path.name  # e.g. E3_20260322_032451
        prefix = name.split("_")[0]  # E3
        p = cluster_dir / prefix / name
        return p if p.exists() else None

    results = {}   # exp_id → list of {user_id, action, energy_mJ}

    for exp_id in ["E3", "E4", "E5"]:   # skip E6 (only ~13 traces)
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        te = load_json(exp_dir / "trace_energies.json")
        if te is None:
            continue

        cluster_exp = find_cluster_exp_dir(exp_id, exp_dir)
        if cluster_exp is None:
            print(f"  ⚠  cluster dir not found for {exp_id}")
            continue

        tid_to_user = extract_user_traces(cluster_exp)
        if not tid_to_user:
            print(f"  ⚠  no user IDs found in {exp_id}")
            continue

        user_traces = []
        for t in te.get("traces", []):
            if t["total_energy_J"] <= 0:
                continue
            ua  = ACTION_LABEL_MAP.get(t.get("user_action", ""), "background")
            if ua not in SESSION_ACTIONS:
                continue
            uid = tid_to_user.get(t.get("trace_id", ""))
            if uid is None:
                continue
            user_traces.append({
                "user_id":   uid,
                "action":    ua,
                "energy_mJ": t["total_energy_J"] * 1000,
            })

        if user_traces:
            results[exp_id] = user_traces
            print(f"  {exp_id}: {len(user_traces)} user-tagged traces, "
                  f"{len({t['user_id'] for t in user_traces})} unique users")

    if not results:
        print("  ⚠  no user-tagged traces found in any experiment")
        return

    EXP_CONFIG_LABELS = {
        "E3": "Workload Medium — 10 concurrent users, fine telemetry (1s / 100%)",
        "E4": "Workload High — 50 concurrent users, fine telemetry (1s / 100%)",
        "E5": "Telemetry Medium — 10 concurrent users, medium telemetry (10s / 50%)",
    }

    for exp_id, user_traces in results.items():
        from collections import defaultdict as _dd
        by_user = _dd(lambda: {"add_to_cart": [], "checkout": []})
        for t in user_traces:
            by_user[t["user_id"]][t["action"]].append(t["energy_mJ"])

        # Sort by total energy descending, keep top 10
        user_order = sorted(
            by_user.keys(),
            key=lambda u: sum(v for vl in by_user[u].values() for v in vl),
            reverse=True)[:10]

        def short_uid(uid):
            return uid[:8] + "\u2026"   # truncate to 8 chars + ellipsis

        x = np.arange(len(user_order))
        w = 0.55

        fig, ax = plt.subplots(figsize=(max(10, len(user_order) * 1.1 + 2), 5))

        add_means = np.array([np.mean(by_user[u]["add_to_cart"])
                               if by_user[u]["add_to_cart"] else 0.0
                               for u in user_order])
        chk_means = np.array([np.mean(by_user[u]["checkout"])
                               if by_user[u]["checkout"] else 0.0
                               for u in user_order])
        add_ns = [len(by_user[u]["add_to_cart"]) for u in user_order]
        chk_ns = [len(by_user[u]["checkout"])    for u in user_order]

        # Stacked vertical bars: add_to_cart bottom, checkout on top
        ax.bar(x, add_means, w,
               color=action_colors["add_to_cart"], alpha=0.85,
               edgecolor="white", label="Add to Cart (mean per trace)")
        ax.bar(x, chk_means, w, bottom=add_means,
               color=action_colors["checkout"], alpha=0.85,
               edgecolor="white", label="Checkout (mean per trace)")

        # Total + count annotation above each bar
        for xi, (an, cn) in enumerate(zip(add_ns, chk_ns)):
            total = add_means[xi] + chk_means[xi]
            parts = []
            if an: parts.append(f"add×{an}")
            if cn: parts.append(f"chk×{cn}")
            ax.text(xi, total * 1.01, "\n".join(parts),
                    ha="center", va="bottom", fontsize=7.5, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([short_uid(u) for u in user_order],
                           fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Mean Energy per Trace [mJ]")
        ax.set_xlabel("User ID")
        ax.legend(fontsize=10, loc="upper right")
        grid_style(ax)

        cfg = EXP_CONFIG_LABELS.get(exp_id, exp_id)
        n_total_users = len(by_user)
        fig.suptitle(
            f"{EXP_LABELS.get(exp_id, exp_id)}: Top-10 Energy-Intensive Users — Add to Cart & Checkout",
            fontsize=12)
        fig.text(0.5, -0.04,
                 f"{cfg}  |  "
                 f"Top-10 of {n_total_users} users with app.user.id  |  "
                 f"add_to_cart & checkout traces only",
                 ha="center", fontsize=9, color="#555", style="italic")
        fig.tight_layout()
        save(fig, output_dir, f"{exp_id.lower()}_user_session_energy.png")

# ══════════════════════════════════════════════════════════════════════
# GROUPED ACTION ENERGY: E2/E3/E4 and E3/E5/E6
# ══════════════════════════════════════════════════════════════════════

def plot_grouped_action_energy(best, output_dir):
    """
    Two grouped comparison plots:
    1. E2 / E3 / E4 – same telemetry, increasing concurrency (2/10/50 users)
    2. E3 / E5 / E6 – same concurrency (10 users), decreasing telemetry granularity

    Each plot: 4 user actions on x-axis, grouped bars per experiment,
    with E1 N=1000 temporal baseline as dotted reference lines.
    Legend and annotations placed outside bars.
    """
    print("\n[GROUPED] Action energy grouped comparison plots...")
    from matplotlib.lines import Line2D

    # Config labels shown below x-axis
    EXP_ANNOT = {
        "E2": "Workload Low\n2 users | 1s/100%",
        "E3": "Workload Medium\n10 users | 1s/100%",
        "E4": "Workload High\n50 users | 1s/100%",
        "E5": "Telemetry Medium\n10 users | 10s/50%",
        "E6": "Telemetry Low\n10 users | 60s/1%",
    }
    # Distinct colours per experiment (not per action – actions on x-axis)
    EXP_COLORS_MAP = {
        "E2": "#42A5F5",
        "E3": "#66BB6A",
        "E4": "#FFA726",
        "E5": "#AB47BC",
        "E6": "#EF5350",
    }

    def _make_grouped_plot(exp_ids, fname, title, subtitle):
        # Collect experiment data
        exp_data = {}
        for exp_id in exp_ids:
            exp_dir = best.get(exp_id)
            if exp_dir is None:
                continue
            by_action = load_trace_energies_by_action(exp_dir)
            if all(not by_action.get(a) for a in ACTIONS):
                continue
            exp_data[exp_id] = {
                a: (np.mean(by_action[a]) if by_action.get(a) else 0.0,
                    np.std(by_action[a])  if by_action.get(a) else 0.0,
                    len(by_action.get(a, [])))
                for a in ACTIONS
            }
        if not exp_data:
            return

        present_exps = [e for e in exp_ids if e in exp_data]
        n_exps  = len(present_exps)
        w       = 0.15
        offsets = np.linspace(-(n_exps-1)/2, (n_exps-1)/2, n_exps) * w * 1.2
        x       = np.arange(len(ACTIONS))

        # Both baselines per action
        bl100  = {a: e1_temporal(best, a, 100)  or e1_temporal(best, a, 1000) or 0.0
                  for a in ACTIONS}
        bl1000 = {a: e1_temporal(best, a, 1000) or e1_temporal(best, a, 100)  or 0.0
                  for a in ACTIONS}

        fig, ax = plt.subplots(figsize=(14, 6))

        for exp_id, offset in zip(present_exps, offsets):
            color    = EXP_COLORS_MAP[exp_id]
            means    = [exp_data[exp_id][a][0] for a in ACTIONS]
            stds     = [exp_data[exp_id][a][1] for a in ACTIONS]
            errs_low = [min(s, m) for m, s in zip(means, stds)]

            bars = ax.bar(x + offset, means, w,
                          yerr=[errs_low, stds], capsize=3,
                          label=EXP_LABELS.get(exp_id, exp_id),
                          color=color, alpha=0.85, edgecolor="white",
                          error_kw={"elinewidth": 1.1, "ecolor": "#444"})
            for bar, m in zip(bars, means):
                if m > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            f"{m:.0f}",
                            ha="center", va="bottom",
                            fontsize=7, color="#222")

        # Span of bars per action group (for drawing baseline lines)
        half_span = (n_exps / 2) * w * 1.2 + w / 2

        # N=100 baseline: solid thin line
        for xi, action in enumerate(ACTIONS):
            bl = bl100[action]
            if bl > 0:
                ax.plot([xi - half_span, xi + half_span], [bl, bl],
                        color="#1565C0", lw=1.4, ls="--", alpha=0.8, zorder=3)

        # N=1000 baseline: dotted line
        for xi, action in enumerate(ACTIONS):
            bl = bl1000[action]
            if bl > 0:
                ax.plot([xi - half_span, xi + half_span], [bl, bl],
                        color="#B71C1C", lw=1.4, ls=":", alpha=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([ACTION_LABELS[a] for a in ACTIONS], fontsize=11)
        ax.set_ylabel("Mean Energy per Trace [mJ]")
        ax.set_ylim(bottom=0)
        grid_style(ax)

        # Legend: experiments + both baselines – outside right
        handles, lbls = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="#1565C0", lw=1.4, ls="--", alpha=0.8))
        lbls.append("Isolated N=100 Temporal Diff. baseline")
        handles.append(Line2D([0], [0], color="#B71C1C", lw=1.4, ls=":", alpha=0.8))
        lbls.append("Isolated N=1000 Temporal Diff. baseline")
        ax.legend(handles, lbls, fontsize=10, framealpha=0.92,
                  loc="upper left")

        ax.set_title(title, fontsize=12, pad=10)
        if subtitle:
            fig.suptitle(subtitle, fontsize=10, y=1.01, color="#666")
        fig.tight_layout()
        save(fig, output_dir, fname)

    # ── Plot 1: E2/E3/E4 – concurrency, both baselines ──────────────
    _make_grouped_plot(
        exp_ids  = ["E2", "E3", "E4"],
        fname    = "grouped_action_energy_concurrency.png",
        title    = "Per-Request Energy by User Action across Concurrency Levels",
        subtitle = "",  # no suptitle for this plot
    )

    # ── Plot 2: E3/E5/E6 – telemetry, both baselines ─────────────────
    _make_grouped_plot(
        exp_ids  = ["E3", "E5", "E6"],
        fname    = "grouped_action_energy_telemetry.png",
        title    = "Per-Request Energy by User Action: Decreasing Telemetry Granularity",
        subtitle = "Workload Medium / Telemetry Medium / Telemetry Low — same concurrency (10 users)",
    )

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT OVERVIEW: all metrics in one figure
# ══════════════════════════════════════════════════════════════════════

def plot_experiment_overview(best, output_dir):
    """
    Single figure with 5 stacked subplots sharing the x-axis.
    X-axis: E1 N=1/10/100/1000 (mean across 4 actions) + E2/E3/E4/E5/E6.
    Panels (top to bottom):
      1. Attribution Coverage [%]
      2. Idle Baseline Consumption [J]
      3. Total Energy Primary Services [J]
      4. Mean Energy per Request [mJ]
      5. Std Dev of Energy per Request [mJ]
    Vertical dashed separator between E1 and E2-E6.
    """
    print("\n[OVERVIEW] Experiment overview multi-metric plot...")

    PRIMARY_EXCLUDE = {"jaeger", "opentelemetry-collector", "collector",
                       "prometheus-server", "kepler", "kepler-exporter",
                       "load-generator"}

    EXP_CONFIGS = {
        "Isolated N=1":    "Isolated\nN=1",
        "Isolated N=10":   "Isolated\nN=10",
        "Isolated N=100":  "Isolated\nN=100",
        "Isolated N=1000": "Isolated\nN=1000",
        "E2":        "Wrkld\nLow",
        "E3":        "Wrkld\nMed",
        "E4":        "Wrkld\nHigh",
        "E5":        "Telem\nMed",
        "E6":        "Telem\nLow",
    }

    # ── Collect data ──────────────────────────────────────────────────
    rows = []   # one per x-position

    # E1: mean across 4 actions per N (only N=100 and N=1000 – N=1/10 distort scale)
    for n in [100, 1000]:
        coverage_vals, idle_vals, total_vals, mean_vals, std_vals = [], [], [], [], []
        for action in ACTIONS:
            key     = f"E1_isolated_{action}_N{n}"
            exp_dir = best.get(key)
            if exp_dir is None:
                continue
            stats = load_json(exp_dir / "attribution_statistics.json")
            if stats is None:
                continue
            budget = stats.get("energy_budget", {})
            aq     = stats.get("attribution_quality", {})
            te     = load_json(exp_dir / "trace_energies.json")

            coverage_vals.append(aq.get("attribution_rate", 0) * 100)
            # idle from primary services only (exclude telemetry)
            idle_primary_e1 = sum(
                d.get("unattributed_J", 0)
                for svc, d in stats.get("per_service", {}).items()
                if svc not in PRIMARY_EXCLUDE
            )
            idle_vals.append(idle_primary_e1 if idle_primary_e1 > 0
                             else budget.get("idle_J", 0))

            # Total primary = total - sum of telemetry service energies
            ps = stats.get("per_service", {})
            total_primary = sum(
                d.get("total_induced_J", 0)
                for svc, d in ps.items() if svc not in PRIMARY_EXCLUDE
            ) + budget.get("idle_J", 0)
            total_vals.append(total_primary)

            if te:
                vals_mJ = [t["total_energy_J"] * 1000
                           for t in te.get("traces", [])
                           if t["total_energy_J"] > 0
                           and ACTION_LABEL_MAP.get(
                               t.get("user_action",""), "") in ACTIONS]
                if vals_mJ:
                    mean_vals.append(np.mean(vals_mJ))
                    std_vals.append(np.std(vals_mJ))

        if coverage_vals:
            rows.append({
                "label":    f"Isolated N={n}",
                "n_val":    n,
                "group":    "E1",
                "coverage": np.mean(coverage_vals),
                "idle_J":   np.mean(idle_vals),
                "total_J":  np.mean(total_vals),
                "mean_mJ":  np.mean(mean_vals)  if mean_vals  else 0.0,
                "std_mJ":   np.mean(std_vals)   if std_vals   else 0.0,
            })

    # E2-E6
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        stats = load_json(exp_dir / "attribution_statistics.json")
        if stats is None:
            continue
        budget = stats.get("energy_budget", {})
        aq     = stats.get("attribution_quality", {})
        ps     = stats.get("per_service", {})
        te     = load_json(exp_dir / "trace_energies.json")

        total_primary = sum(
            d.get("total_induced_J", 0)
            for svc, d in ps.items() if svc not in PRIMARY_EXCLUDE
        ) + budget.get("idle_J", 0)

        vals_mJ = []
        if te:
            vals_mJ = [t["total_energy_J"] * 1000
                       for t in te.get("traces", [])
                       if t["total_energy_J"] > 0
                       and ACTION_LABEL_MAP.get(
                           t.get("user_action",""), "") in ACTIONS]

        idle_primary_conc = sum(
            d.get("unattributed_J", 0)
            for svc, d in ps.items() if svc not in PRIMARY_EXCLUDE
        )
        idle_conc = idle_primary_conc if idle_primary_conc > 0 else budget.get("idle_J", 0)

        rows.append({
            "label":    EXP_LABELS.get(exp_id, exp_id),
            "exp_id":   exp_id,   # keep original key for lookups
            "group":    "concurrent",
            "coverage": aq.get("attribution_rate", 0) * 100,
            "idle_J":   idle_conc,
            "total_J":  total_primary,
            "mean_mJ":  np.mean(vals_mJ) if vals_mJ else 0.0,
            "std_mJ":   np.std(vals_mJ)  if vals_mJ else 0.0,
        })

    if not rows:
        print("  ⚠  no data")
        return

    # ── Compute idle power (W) from idle_J / experiment duration ─────
    # E1 isolated: short runs; concurrent: 600s default
    E1_DURATION_S   = 120.0   # approx warm-up + request window for isolated
    CONC_DURATION_S = 600.0
    for r in rows:
        dur = E1_DURATION_S if r["group"] == "E1" else CONC_DURATION_S
        r["idle_W"] = r["idle_J"] / dur

    # ── Layout: 3 panels (coverage, idle power, mean energy per action)
    labels = [r["label"] for r in rows]
    x      = np.arange(len(rows))
    w      = 0.55
    sep_x  = 1.5   # vertical separator after Isolated N=1000 (only N=100/1000)

    EXP_COLOR_MAP = dict(zip(CONCURRENT_EXPERIMENTS, EXP_COLORS))
    n_purple   = [COLORS["coverage"]] * 2  # only N=100 and N=1000
    c_conc     = [EXP_COLOR_MAP.get(r["label"], "#888")
                  for r in rows if r["group"] == "concurrent"]
    bar_colors = n_purple + c_conc

    action_line_colors = dict(zip(ACTIONS, ACTION_COLORS))
    action_markers     = {"browse": "o", "add_to_cart": "s",
                          "view_cart": "^", "checkout": "D"}

    def _vline(ax):
        ax.axvline(sep_x, color="#aaa", lw=1, ls="--", alpha=0.6)

    # ── Collect per-action energy series ─────────────────────────────
    action_ys = {a: [] for a in ACTIONS}
    for r in rows:
        if r["group"] == "E1":
            n_val = r["n_val"]
            for a in ACTIONS:
                # Use interval-duration trace_energies for the specific action
                exp_dir = best.get(f"E1_isolated_{a}_N{n_val}")
                if exp_dir:
                    te = load_json(exp_dir / "trace_energies.json")
                    if te:
                        vals = [t["total_energy_J"] * 1000
                                for t in te.get("traces", [])
                                if t["total_energy_J"] > 0
                                and ACTION_LABEL_MAP.get(
                                    t.get("user_action",""), "") == a]
                        action_ys[a].append(np.mean(vals) if vals else np.nan)
                    else:
                        action_ys[a].append(np.nan)
                else:
                    action_ys[a].append(np.nan)
        else:
            # Use exp_id (original key) for lookup, not display label
            exp_id  = r.get("exp_id", r["label"])
            exp_dir = best.get(exp_id)
            by_a    = load_trace_energies_by_action(exp_dir) if exp_dir else {}
            for a in ACTIONS:
                vals = by_a.get(a, [])
                action_ys[a].append(np.mean(vals) if vals else np.nan)

    coverage_vals = [r["coverage"] for r in rows]
    tick_labels   = [EXP_CONFIGS.get(r["label"], r["label"]) for r in rows]

    # ════════════════════════════════════════════════════════════════
    # PLOT 1: experiment_overview.png
    #   Single panel: 4 action lines (log) + coverage shading right axis
    # ════════════════════════════════════════════════════════════════
    idle_w = [r["idle_W"] for r in rows]  # still used for idle_baseline_overview
    fig, ax_single = plt.subplots(figsize=(14, 6))

    # ── Single panel: 4 action lines (log) + coverage shading ────────
    ax_e  = ax_single                  # left y-axis: energy [mJ]
    ax_cv = ax_e.twinx()               # right y-axis: coverage [%]

    # Coverage background shading (filled area from 0 to coverage %)
    # Use a light purple fill so lines stay readable
    ax_cv.fill_between(x, 0, coverage_vals,
                       color=COLORS["coverage"], alpha=0.12,
                       step=None, label="_nolegend_")
    # Coverage line on top of shading
    ax_cv.plot(x, coverage_vals,
               color=COLORS["coverage"], lw=1.5, ls="--",
               marker="x", markersize=6, alpha=0.75,
               label="Attribution Coverage [%]")
    for xi, cv in enumerate(coverage_vals):
        ax_cv.text(xi, cv + 1.5, f"{cv:.0f}%",
                   ha="center", va="bottom", fontsize=7.5,
                   color=COLORS["coverage"])
    ax_cv.set_ylim(0, 115)
    ax_cv.set_ylabel("Attribution Coverage [%]",
                     color=COLORS["coverage"], fontsize=10)
    ax_cv.tick_params(axis="y", labelcolor=COLORS["coverage"])
    ax_cv.spines["right"].set_edgecolor(COLORS["coverage"])

    # Temporal diff. baseline N=1000 per action (horizontal dotted lines)
    for action in ACTIONS:
        bl = e1_temporal(best, action, 1000) or e1_temporal(best, action, 100)
        if bl and bl > 0:
            ax_e.axhline(bl,
                         color=action_line_colors[action],
                         lw=1.0, ls=":", alpha=0.55, zorder=2)

    # 4 action lines
    first_action = True
    for action in ACTIONS:
        ys    = action_ys[action]
        valid = [(xi, y) for xi, y in enumerate(ys)
                 if y is not None and not np.isnan(y) and y > 0]
        if not valid:
            continue
        xs_v, ys_v = zip(*valid)
        ax_e.plot(xs_v, ys_v,
                  color=action_line_colors[action],
                  marker=action_markers[action],
                  markersize=7, lw=2, alpha=0.9,
                  label=ACTION_LABELS[action])
        # Add baseline label on the right edge (only once per action)
        bl = e1_temporal(best, action, 1000) or e1_temporal(best, action, 100)
        if bl and bl > 0:
            ax_e.annotate(f"baseline {ACTION_LABELS[action][:3]}. {bl:.0f}mJ",
                          xy=(len(rows) - 0.5, bl),
                          xycoords=("data", "data"),
                          fontsize=7, color=action_line_colors[action],
                          alpha=0.7, va="center",
                          annotation_clip=False)
        for xi, y in valid:
            ax_e.annotate(f"{y:.0f}",
                          (xi, y),
                          textcoords="offset points",
                          xytext=(0, 6),
                          ha="center", fontsize=7.5,
                          color=action_line_colors[action])

    ax_e.set_yscale("log")
    ax_e.set_ylabel("Mean Energy per Request [mJ] (log scale)", fontsize=10)
    _vline(ax_e)
    grid_style(ax_e)

    ax_e.set_title("Experiment Overview — Mean Energy per Request & Attribution Coverage",
                  fontsize=13, pad=10)
    ax_e.text(0.5, ax_e.get_ylim()[1] if ax_e.get_yscale() != "log"
              else 10 ** (np.log10(ax_e.get_ylim()[1]) * 0.97),
              "Isolated", ha="center",
              fontsize=9, color="#555", style="italic")
    ax_e.text(4.0, ax_e.get_ylim()[1] if ax_e.get_yscale() != "log"
              else 10 ** (np.log10(ax_e.get_ylim()[1]) * 0.97),
              "Concurrent workload experiments", ha="center",
              fontsize=9, color="#555", style="italic")

    # Combined legend: actions + coverage + baseline marker
    from matplotlib.lines import Line2D as _L2D
    lines_e, lbl_e   = ax_e.get_legend_handles_labels()
    lines_cv, lbl_cv = ax_cv.get_legend_handles_labels()
    baseline_handle  = _L2D([0],[0], color="grey", lw=1.0, ls=":",
                             alpha=0.7, label="Temporal Diff. baseline N=1000")
    ax_e.legend(lines_e + lines_cv + [baseline_handle],
                lbl_e  + lbl_cv  + ["Temporal Diff. baseline N=1000"],
                fontsize=10, framealpha=0.9,
                loc="lower right")

    ax_e.set_xticks(x)
    ax_e.set_xticklabels(tick_labels, fontsize=10, ha="center")

    fig.text(0.5, -0.02,
             "Isolated requests: per-action mean shown separately  ·  "
             "Idle power = idle_J / experiment duration  ·  "
             "Primary app services only (kepler / prometheus / jaeger excluded)",
             ha="center", fontsize=11, color="#444", style="italic")

    fig.tight_layout()
    save(fig, output_dir, "experiment_overview.png")

    # ════════════════════════════════════════════════════════════════
    # PLOT 2: idle_baseline_overview.png  (standalone idle power)
    # ════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    bars2 = ax2.bar(x, idle_w, w, color=bar_colors,
                    alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, idle_w):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     v * 1.01, f"{v:.3f} W",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, fontsize=10, ha="center")
    ax2.set_ylabel("Idle Baseline Power [W]", fontsize=11)
    ax2.set_title("Idle Baseline Power per Experiment\n"
                  "(primary app services; telemetry excluded)",
                  fontsize=12)
    ax2.axvline(sep_x, color="#aaa", lw=1, ls="--", alpha=0.6)
    ax2.text(0.5, ax2.get_ylim()[1] * 0.93,
             "Isolated", ha="center",
             fontsize=9, color="#555", style="italic")
    ax2.text(6.0, ax2.get_ylim()[1] * 0.93,
             "Concurrent experiments", ha="center",
             fontsize=9, color="#555", style="italic")
    grid_style(ax2)
    fig2.tight_layout()
    save(fig2, output_dir, "idle_baseline_overview.png")

# ══════════════════════════════════════════════════════════════════════
# TRACE DISTRIBUTION & INTERVALS WITHOUT SPANS
# ══════════════════════════════════════════════════════════════════════

def plot_trace_distribution(best, output_dir):
    """
    Single stacked bar chart: trace count per experiment (E2-E6),
    stacked by user action. Legend placed outside to avoid overlap.
    """
    print("\n[TRACE DIST] Trace count and distribution per experiment...")

    ALL_CATS   = ACTIONS + ["background"]
    cat_colors = dict(zip(ACTIONS, ACTION_COLORS))
    cat_colors["background"] = "#B0BEC5"

    exp_ids = []
    counts  = {a: [] for a in ALL_CATS}

    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        te = load_json(exp_dir / "trace_energies.json")
        if te is None:
            continue
        cat_count = {a: 0 for a in ALL_CATS}
        for t in te.get("traces", []):
            ua  = ACTION_LABEL_MAP.get(t.get("user_action",""), "background")
            cat = ua if ua in ALL_CATS else "background"
            cat_count[cat] += 1
        if sum(cat_count.values()) > 0:
            exp_ids.append(EXP_LABELS.get(exp_id, exp_id))
            for cat in ALL_CATS:
                counts[cat].append(cat_count[cat])

    if not exp_ids:
        print("  ⚠  no data")
        return

    x      = np.arange(len(exp_ids))
    w      = 0.55
    totals = np.array([sum(counts[c][i] for c in ALL_CATS)
                       for i in range(len(exp_ids))])

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(exp_ids))
    legend_handles = []

    for cat in ALL_CATS:
        vals = np.array(counts[cat])
        if vals.sum() == 0:
            continue
        bars = ax.bar(x, vals, w, bottom=bottoms,
                      color=cat_colors[cat], alpha=0.85,
                      edgecolor="white",
                      label=ACTION_LABELS.get(cat, cat.capitalize()))
        legend_handles.append(bars[0])
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > max(totals) * 0.04:
                ax.text(xi, b + v / 2, str(int(v)),
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    for xi, tot in enumerate(totals):
        ax.text(xi, tot * 1.01, str(int(tot)),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, fontsize=11)
    ax.set_ylabel("Number of Traces")
    ax.set_title("Trace Count per Experiment (E2–E6)\n"
                 "stacked by user action (interval-duration attribution)")
    # Legend outside right
    ax.legend(fontsize=10, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.9)
    grid_style(ax)
    fig.tight_layout()
    save(fig, output_dir, "trace_distribution.png")

def plot_intervals_without_spans(best, output_dir):
    """
    Per experiment (E2-E6): for each service, percentage of Prometheus
    scrape intervals where spans_per_service[svc] == 0 (or key absent).
    Uses interval_timeseries.json (list of interval dicts, each with
    'spans_per_service': {svc: count}).
    """
    print("\n[INTERVALS] Intervals without spans per service...")

    def get_preprocessed_dir(attributed_exp_dir: Path) -> Optional[Path]:
        p = Path("preprocessed_data") / attributed_exp_dir.name
        return p if p.exists() else None

    TELEMETRY_SVCS = {"jaeger", "opentelemetry-collector", "collector",
                      "prometheus-server", "kepler", "kepler-exporter",
                      "load-generator", "flagd-ui", "opensearch",
                      "image-provider"}

    results = {}

    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        pre_dir = get_preprocessed_dir(exp_dir)
        if pre_dir is None:
            print(f"  ⚠  preprocessed_data not found for {exp_id}")
            continue
        its = load_json(pre_dir / "interval_timeseries.json")
        if its is None:
            continue

        # interval_timeseries.json is a list of interval dicts
        interval_list = its if isinstance(its, list) else its.get("intervals", [])

        # Iterate intervals: use Kepler "services" as ground truth of
        # which services had measurable energy, then check OTel span coverage
        # Second pass: for each interval, check per service whether
        # Kepler measured energy (interval["services"][svc]) but no
        # OTel span overlapped (spans_per_service[svc] == 0 or absent)
        svc_counts = {}  # {svc: [n_empty, n_total]}

        for iv in interval_list:
            if not isinstance(iv, dict):
                continue
            kepler_svcs = iv.get("services", {})    # svcs with Kepler energy
            span_sps    = iv.get("spans_per_service", {})  # svcs with spans

            for svc, sdata in kepler_svcs.items():
                if not isinstance(sdata, dict):
                    continue
                # Only count intervals where the service had actual energy
                if sdata.get("total_J", 0) <= 0:
                    continue
                if svc not in svc_counts:
                    svc_counts[svc] = [0, 0]
                svc_counts[svc][1] += 1
                # Empty = no OTel span overlap for this service in this interval
                if span_sps.get(svc, 0) == 0:
                    svc_counts[svc][0] += 1

        if svc_counts:
            results[exp_id] = svc_counts

    if not results:
        print("  ⚠  no interval data found")
        return

    fig, axes = plt.subplots(1, len(results),
                             figsize=(5 * len(results), 7),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (exp_id, svc_counts) in zip(axes, results.items()):
        rows_s = []
        for svc, (n_empty, n_total) in svc_counts.items():
            if svc in TELEMETRY_SVCS or n_total == 0:
                continue
            pct = n_empty / n_total * 100
            rows_s.append((svc, pct, n_empty, n_total))
        rows_s.sort(key=lambda r: r[1], reverse=True)

        if not rows_s:
            ax.set_title(f"{EXP_LABELS.get(exp_id, exp_id)}\n(no data)", fontweight="bold")
            continue

        svcs   = [r[0] for r in rows_s]
        pcts   = [r[1] for r in rows_s]
        y      = np.arange(len(svcs))
        colors = ["#EF5350" if p >= 80 else
                  "#FFA726" if p >= 40 else
                  "#66BB6A" for p in pcts]

        ax.barh(y, pcts, color=colors, alpha=0.85, edgecolor="white")
        for yi, (p, r) in enumerate(zip(pcts, rows_s)):
            ax.text(min(p + 0.5, 102), yi,
                    f"{p:.0f}% ({r[2]}/{r[3]})",
                    va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(svcs, fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Intervals without Spans [%]")
        ax.set_title(EXP_LABELS.get(exp_id, exp_id), fontweight="bold")
        ax.axvline(50, color="#aaa", lw=0.8, ls=":", alpha=0.6)
        grid_style(ax)

    from matplotlib.patches import Patch as _Patch
    legend_els = [
        _Patch(color="#66BB6A", alpha=0.85,
               label="< 40% empty  (well attributed)"),
        _Patch(color="#FFA726", alpha=0.85, label="40–80% empty"),
        _Patch(color="#EF5350", alpha=0.85,
               label="≥ 80% empty  (structurally unattributable)"),
    ]
    fig.legend(handles=legend_els, fontsize=9,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    # suptitle removed per user request
    fig.tight_layout()
    save(fig, output_dir, "intervals_without_spans.png")


# ══════════════════════════════════════════════════════════════════════
# PLAUSIBILITY: concurrent vs isolated ratio plot
# ══════════════════════════════════════════════════════════════════════

def plot_concurrent_vs_isolated_ratio(best, output_dir):
    """
    For each user action: ratio of mean concurrent energy to isolated
    N=1000 temporal-diff baseline. Values near 1.0 indicate the
    concurrent attribution is plausible; systematically high values
    point to load-induced overhead or attribution artefacts.

    Shows E2-E6 as grouped bars per action, with a reference line at 1.0.
    """
    print("\n[PLAUSIBILITY] Concurrent vs. isolated ratio plot...")
    from matplotlib.lines import Line2D

    # Baselines: temporal diff N=100 (N=1000 is unreliable due to
    # baseline measurement noise dominating at high request counts)
    baselines = {}
    for a in ACTIONS:
        bl = e1_temporal(best, a, 100) or e1_temporal(best, a, 1000)
        if bl and bl > 0:
            baselines[a] = bl

    if not baselines:
        print("  ⚠  no temporal baselines found")
        return

    # Concurrent mean per action
    exp_data = {}
    for exp_id in CONCURRENT_EXPERIMENTS:
        exp_dir = best.get(exp_id)
        if exp_dir is None:
            continue
        by_action = load_trace_energies_by_action(exp_dir)
        exp_data[exp_id] = {
            a: np.mean(by_action[a]) if by_action.get(a) else np.nan
            for a in ACTIONS
        }

    if not exp_data:
        return

    present_exps = [e for e in CONCURRENT_EXPERIMENTS if e in exp_data]
    action_colors = dict(zip(ACTIONS, ACTION_COLORS))
    EXP_COLORS_MAP = dict(zip(CONCURRENT_EXPERIMENTS, EXP_COLORS))

    x      = np.arange(len(ACTIONS))
    n_exps = len(present_exps)
    w      = 0.14
    offsets = np.linspace(-(n_exps-1)/2, (n_exps-1)/2, n_exps) * w * 1.15

    fig, ax = plt.subplots(figsize=(12, 5))

    for exp_id, offset in zip(present_exps, offsets):
        ratios = []
        for a in ACTIONS:
            mean_mJ = exp_data[exp_id].get(a, np.nan)
            bl      = baselines.get(a)
            if bl and not np.isnan(mean_mJ) and mean_mJ > 0:
                ratios.append(mean_mJ / bl)
            else:
                ratios.append(np.nan)

        valid_x = [xi for xi, r in enumerate(ratios) if not np.isnan(r)]
        valid_r = [r  for r     in ratios          if not np.isnan(r)]

        bars = ax.bar([xi + offset for xi in valid_x], valid_r, w,
                      color=EXP_COLORS_MAP[exp_id], alpha=0.82,
                      edgecolor="white",
                      label=EXP_LABELS.get(exp_id, exp_id))
        for xi, r in zip(valid_x, valid_r):
            ax.text(xi + offset, r + 0.02,
                    f"{r:.2f}×",
                    ha="center", va="bottom",
                    fontsize=7.5, color=EXP_COLORS_MAP[exp_id])

    # Reference line at 1.0 = perfectly matches isolated baseline
    ax.axhline(1.0, color="#333", lw=1.5, ls="-", alpha=0.7,
               label="Isolated N=100 Temporal Diff. baseline (ratio = 1.0)")
    # Plausibility band: ±50% (0.5 – 1.5)
    ax.axhspan(0.5, 1.5, color="#A5D6A7", alpha=0.12,
               label="Plausibility band (0.5×–1.5×)")

    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABELS[a] for a in ACTIONS], fontsize=11)
    ax.set_ylabel("Energy Ratio (concurrent / isolated N=1000 baseline)")
    ax.set_ylim(bottom=0)
    ax.set_title("Plausibility Check: Concurrent Per-Request Energy\n"
                 "relative to Isolated N=100 Temporal Diff. Baseline",
                 fontsize=12)

    ax.legend(fontsize=10, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.9)
    grid_style(ax)

    fig.text(0.5, -0.04,
             "Ratio = 1.0: concurrent matches Isolated N=100 Temporal Diff. baseline  ·  "
             "1×–2×: plausible load overhead  ·  "
             "> 5×: likely attribution artefact (coarse telemetry or low trace coverage)",
             ha="center", fontsize=9, color="#444", style="italic")

    fig.tight_layout()
    save(fig, output_dir, "plausibility_concurrent_vs_isolated.png")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate all thesis plots from attributed experiment data"
    )
    parser.add_argument("--attributed-dir", default=DEFAULT_ATTRIBUTED_DIR)
    parser.add_argument("--cluster-dir",    default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = parser.parse_args()

    attributed_dir = Path(args.attributed_dir)
    cluster_dir    = Path(args.cluster_dir)
    output_dir     = Path(args.output_dir)

    if not attributed_dir.exists():
        print(f"ERROR: attributed_dir not found: {attributed_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("THESIS PLOT GENERATION")
    print("=" * 65)
    print(f"Attributed data : {attributed_dir}")
    print(f"Cluster data    : {cluster_dir}")
    print(f"Output          : {output_dir}")

    print("\n[0] Selecting best run per experiment...")
    best = select_best_runs(attributed_dir, args.runs)
    print(f"  -> {len(best)} experiment keys loaded")

    plot_e1_action_energy_by_method(best, output_dir)
    plot_e1_action_energy_by_n(best, output_dir)
    # plot_e1_top10_traces(best, output_dir)
    plot_attribution_coverage(best, output_dir)
    plot_concurrent_action_energy(best, output_dir)
    plot_grouped_action_energy(best, output_dir)
    # plot_concurrent_top10_traces(best, output_dir)
    plot_telemetry_overhead(best, cluster_dir, output_dir)
    plot_boxplots_e1(best, output_dir)       # also saves _log and _no_n1 variants
    plot_boxplots_concurrent(best, output_dir)
    plot_all_experiments_action_energy(best, output_dir)
    plot_attribution_coverage_by_n(best, output_dir)
    plot_duration_vs_energy(best, output_dir)
    plot_total_energy_comparison(best, attributed_dir, cluster_dir, output_dir)
    plot_user_session_energy(best, cluster_dir, output_dir)
    plot_span_vs_request_energy(best, output_dir)
    plot_trace_distribution(best, output_dir)
    plot_intervals_without_spans(best, output_dir)
    plot_concurrent_vs_isolated_ratio(best, output_dir)
    plot_experiment_overview(best, output_dir)
    plot_coverage_vs_temporal_error(best, output_dir)
    plot_validation_std_over_runs(attributed_dir, output_dir)
    plot_validation_coverage_over_runs(attributed_dir, output_dir)
    plot_validation_per_service_coverage(best, output_dir)

    plots = sorted(output_dir.glob("*.png"))
    print(f"\n{'='*65}")
    print(f"  {len(plots)} plots saved to {output_dir}/")
    print(f"{'='*65}")
    for p in plots:
        print(f"   {p.name}")


if __name__ == "__main__":
    main()
