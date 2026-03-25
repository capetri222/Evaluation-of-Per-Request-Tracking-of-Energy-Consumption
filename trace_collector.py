#!/usr/bin/env python3
"""
Trace Collector

Reads the collection window from raw_data_meta.json (written by raw_data_collector.py),
queries Jaeger for all traces in that window per service,
saves complete traces to raw_traces/raw_traces_latest.json.

Usage:
    python3 trace_collector.py
    python3 trace_collector.py --meta raw_data/raw_data_meta.json
    python3 trace_collector.py --start 2026-02-10T22:00:00 --end 2026-02-10T22:15:00

Why this changed
----------------
raw_data_collector.py now saves energy data as CSV (raw_data_latest.csv)
and writes a small companion metadata file (raw_data_meta.json) with the
window_start / window_end timestamps.  The old code read these fields from
raw_data_latest.json, which no longer exists.

Jaeger URL notes
----------------
The correct API endpoint depends on your deployment:

  Standard (bare Jaeger):     http://localhost:16686
    -> API calls go to  http://localhost:16686/api/traces

  Behind a reverse proxy / Ingress with /jaeger/ui prefix:
    http://localhost:16686/jaeger/ui
    -> API calls still go to  http://localhost:16686/api/traces
       (the /jaeger/ui path is only for the browser UI, NOT the API)

If you got HTML responses yesterday, the base URL was correct but the
/api/traces suffix was being doubled.  The default below uses the bare
base URL; override with --jaeger-url if needed.
"""

import json
import sys
import requests
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── defaults ──────────────────────────────────────────────────────
# Base URL only — /api/traces is appended automatically.
# Do NOT include /jaeger/ui here; that path is only for the browser UI.
JAEGER_BASE_URL = "http://localhost:16686"
META_FILE       = "raw_data/raw_data_meta.json"
OUTPUT_FILE     = "raw_traces/raw_traces_latest.json"
LIMIT_PER_SVC   = 5000  # Reduced from 10000 to avoid Jaeger timeouts

PRIMARY_SERVICES = [
    "accounting", "ad", "cart", "checkout", "currency",
    "email", "fraud-detection", "frontend", "frontend-proxy",
    "image-provider", "kafka", "payment", "postgresql",
    "product-catalog", "product-reviews", "quote", "recommendation", 
    "shipping", "valkey-cart", "flagd", "flagd-ui", "llm"
]


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is UTC-aware.
    - If already timezone-aware: convert to UTC.
    - If naive: assume UTC (all timestamps in this system are UTC).
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ═══════════════════════════════════════════════════════════════════
# WINDOW EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_window_from_meta(meta_file: str) -> Tuple[datetime, datetime]:
    """
    Read window_start / window_end from raw_data_meta.json.
    This file is written by raw_data_collector.py alongside the CSV.
    """
    print(f"Reading window from: {meta_file}")

    with open(meta_file) as f:
        meta = json.load(f)

    ws_str = meta.get("window_start", "")
    we_str = meta.get("window_end",   "")

    if not ws_str or not we_str:
        raise ValueError(
            f"window_start / window_end missing in {meta_file}.\n"
            f"Contents: {meta}"
        )

    start_dt = ensure_utc(datetime.fromisoformat(ws_str))
    end_dt   = ensure_utc(datetime.fromisoformat(we_str))
    duration = (end_dt - start_dt).total_seconds() / 60.0

    print(f"  Start:    {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  End:      {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Duration: {duration:.1f} min  |  zone: {meta.get('zone','?')}  "
          f"|  services: {meta.get('services', [])}")

    return start_dt, end_dt


def parse_manual_window(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    """Parse --start / --end ISO strings supplied on the command line.
    Naive datetimes are assumed to be UTC."""
    try:
        start_dt = ensure_utc(datetime.fromisoformat(start_str))
        end_dt   = ensure_utc(datetime.fromisoformat(end_str))
        return start_dt, end_dt
    except ValueError as e:
        raise ValueError(f"Invalid datetime format (use ISO 8601, e.g. 2026-02-10T22:00:00): {e}")


# ═══════════════════════════════════════════════════════════════════
# JAEGER QUERY
# ═══════════════════════════════════════════════════════════════════

def probe_jaeger(base_url: str) -> Optional[str]:
    """
    Check which API path Jaeger responds on and return the working base.
    Tries:
      1. {base_url}/api/services          (standard)
      2. {base_url}/jaeger/ui/api/services (some proxy configs)
    Returns the working base URL for API calls, or None if unreachable.
    """
    candidates = [
        base_url.rstrip("/"),
        base_url.rstrip("/") + "/jaeger/ui",
    ]
    # Remove duplicates while preserving order
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)

    for candidate in seen:
        probe_url = f"{candidate}/api/services"
        try:
            r = requests.get(probe_url, timeout=5)
            ct = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "application/json" in ct:
                data = r.json()
                svcs = data.get("data", [])
                print(f"  ✓ Jaeger reachable at {candidate}/api/")
                print(f"    Services found: {len(svcs)}")
                if svcs:
                    sample = ", ".join(svcs[:6])
                    if len(svcs) > 6:
                        sample += f" … +{len(svcs)-6} more"
                    print(f"    Sample: {sample}")
                return candidate
            elif "text/html" in ct:
                print(f"  – {candidate}/api/services → HTML (browser UI, not API base)")
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to {candidate}")
        except Exception as e:
            print(f"  – {candidate}: {e}")

    return None


def collect_all_traces(start_dt: datetime,
                       end_dt:   datetime,
                       api_base: str) -> List[Dict]:
    """Query Jaeger per service, deduplicate by traceID, return all traces."""
    print(f"\n{'='*70}")
    print("COLLECTING TRACES FROM JAEGER")
    print(f"{'='*70}")
    print(f"  API base: {api_base}/api/")
    print(f"  Window:   {start_dt.strftime('%H:%M:%S')} → {end_dt.strftime('%H:%M:%S')} UTC")
    print()

    # Use UTC timestamp — critical fix: timezone-aware datetime.timestamp()
    # correctly returns UTC epoch seconds regardless of local timezone.
    start_us = int(start_dt.timestamp() * 1_000_000)
    end_us   = int(end_dt.timestamp()   * 1_000_000)

    print(f"  start_us: {start_us}")
    print(f"  end_us:   {end_us}")
    print()

    all_traces: Dict[str, Dict] = {}

    for service in PRIMARY_SERVICES:
        url = f"{api_base}/api/traces"
        params = {
            "service":  service,
            "start":    start_us,
            "end":      end_us,
            "limit":    LIMIT_PER_SVC,
            "lookback": "custom",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            ct   = resp.headers.get("Content-Type", "")

            if "text/html" in ct:
                # This should not happen after probe_jaeger(), but guard anyway
                print(f"  ✗ {service:<25s}: HTML — api_base may be wrong")
                continue

            resp.raise_for_status()

            if not resp.text.strip():
                print(f"  - {service:<25s}: empty response")
                continue

            try:
                data = resp.json()
            except json.JSONDecodeError:
                print(f"  ✗ {service:<25s}: JSON decode error")
                continue

            traces = data.get("data", []) if isinstance(data, dict) else data

            new = 0
            for trace in traces:
                tid = trace.get("traceID")
                if tid and tid not in all_traces:
                    all_traces[tid] = trace
                    new += 1

            status = f"✓ {new} new" if new > 0 else "- 0 new"
            print(f"  {status:<6} {service}")

        except requests.exceptions.Timeout:
            print(f"  ✗ {service:<25s}: timeout")
        except requests.exceptions.ConnectionError as e:
            print(f"  ⚠ {service:<25s}: connection error, retrying with limit=1000...")
            # Retry with smaller limit
            try:
                params['limit'] = 1000
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                if resp.text.strip():
                    data = resp.json()
                    traces = data.get("data", []) if isinstance(data, dict) else data
                    new = 0
                    for trace in traces:
                        tid = trace.get("traceID")
                        if tid and tid not in all_traces:
                            all_traces[tid] = trace
                            new += 1
                    status = f"✓ {new} new (retry)" if new > 0 else "- 0 new (retry)"
                    print(f"  {status} {service}")
            except Exception as retry_error:
                print(f"  ✗ {service:<25s}: retry failed — {retry_error}")
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ {service:<25s}: HTTP {e.response.status_code}")
        except Exception as e:
            print(f"  ✗ {service:<25s}: {e}")

    result = list(all_traces.values())
    print(f"\n  Total unique traces: {len(result)}")

    if not result:
        print()
        print("  No traces found — possible causes:")
        print("    1. Load generator was not running during the collection window")
        print("    2. Jaeger retention is shorter than the gap since collection")
        print(f"       (window was {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC — "
              f"check Jaeger UI at {api_base})")
        print("    3. Services emit traces to a different Jaeger instance")
        print("    4. Timezone mismatch — verify start_us/end_us printed above match Jaeger trace timestamps")

    return result


# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════

def save_traces(traces: List[Dict], output_file: str,
                start_dt: datetime, end_dt: datetime) -> None:
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True
    )
    output = {
        "window_start": start_dt.isoformat(),
        "window_end":   end_dt.isoformat(),
        "trace_count":  len(traces),
        "traces":       traces,
    }
    with open(output_file, "w") as f:
        json.dump(output, f)

    size_kb = os.path.getsize(output_file) / 1024
    print(f"\n✓ Saved: {output_file}  ({size_kb:.0f} KB, {len(traces)} traces)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Jaeger traces for the raw_data collection window",
        epilog="""
Examples:
  # Collect last 10 minutes (no meta file needed):
  python3 trace_collector.py --minutes 10

  # Use meta file from raw_data_collector:
  python3 trace_collector.py --meta raw_data/raw_data_meta.json

  # Manual time window:
  python3 trace_collector.py --start 2026-03-10T11:00:00 --end 2026-03-10T11:15:00
        """
    )
    parser.add_argument(
        "--meta", default=None,
        help=f"raw_data_meta.json from raw_data_collector.py"
    )
    parser.add_argument(
        "--start", default=None,
        help="Override window start (ISO 8601, e.g. 2026-02-10T22:00:00 — assumed UTC if no timezone given)"
    )
    parser.add_argument(
        "--end", default=None,
        help="Override window end (ISO 8601, e.g. 2026-02-10T22:15:00 — assumed UTC if no timezone given)"
    )
    parser.add_argument(
        "--minutes", type=int, default=10,
        help="Collect traces from last N minutes (default: 10)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Output file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--jaeger-url", default=JAEGER_BASE_URL,
        help=(
            f"Jaeger BASE URL — do not include /api or /jaeger/ui "
            f"(default: {JAEGER_BASE_URL})"
        )
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TRACE COLLECTOR")
    print("=" * 70)

    # ── 1. Resolve time window ─────────────────────────────────────
    if args.start and args.end:
        print("\n[1/3] Using manual time window...")
        try:
            start_dt, end_dt = parse_manual_window(args.start, args.end)
        except ValueError as e:
            print(f"\n✗ {e}")
            return 1
        print(f"  Start: {start_dt.isoformat()}")
        print(f"  End:   {end_dt.isoformat()}")
    elif args.meta:
        print("\n[1/3] Reading time window from metadata...")
        meta_path = Path(args.meta)
        if not meta_path.exists():
            # Try old JSON file as fallback
            old_json = meta_path.parent / "raw_data_latest.json"
            if old_json.exists():
                print(f"  ⚠  {args.meta} not found, trying {old_json} (old format)")
                args.meta = str(old_json)
            else:
                print(f"\n✗ Not found: {args.meta}")
                print("  Use --minutes N instead, or run raw_data_collector.py first")
                return 1
        try:
            # Try new meta format first
            if args.meta.endswith("_meta.json") or "meta" in Path(args.meta).name:
                start_dt, end_dt = extract_window_from_meta(args.meta)
            else:
                # Old JSON format: window_start/window_end in a large JSON
                start_dt, end_dt = _extract_from_json(args.meta)
        except Exception as e:
            print(f"\n✗ Could not read window: {e}")
            return 1
    else:
        # Default: Use --minutes to collect last N minutes
        print(f"\n[1/3] Using last {args.minutes} minutes...")
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=args.minutes)
        print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  End:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # ── 2. Probe Jaeger ────────────────────────────────────────────
    print("\n[2/3] Probing Jaeger...")
    api_base = probe_jaeger(args.jaeger_url)
    if api_base is None:
        print(f"\n✗ Jaeger unreachable at {args.jaeger_url}")
        print("  Check:")
        print("    kubectl port-forward svc/jaeger 16686:16686 -n christianpetri-otel-demo")
        print(f"    curl {args.jaeger_url}/api/services")
        return 1

    # ── 3. Collect & save ─────────────────────────────────────────
    print("\n[3/3] Collecting traces...")
    traces = collect_all_traces(start_dt, end_dt, api_base)
    save_traces(traces, args.output, start_dt, end_dt)

    print(f"\n{'='*70}")
    print("✓  DONE")
    print(f"  Traces:  {args.output}")
    print(f"\nNext:  python3 trace_cleaner.py")
    print()
    return 0


def _extract_from_json(filepath: str) -> Tuple[datetime, datetime]:
    """Fallback: read window_start/window_end from old raw_data_latest.json."""
    print(f"  Reading window from JSON: {filepath}")
    with open(filepath) as f:
        head = f.read(400)
    ws_key = '"window_start":'
    we_key = '"window_end":'
    ws_idx = head.index(ws_key) + len(ws_key)
    we_idx = head.index(we_key) + len(we_key)
    ws_str = head[ws_idx:ws_idx+32].strip().strip('"').split('"')[0]
    we_str = head[we_idx:we_idx+32].strip().strip('"').split('"')[0]
    return ensure_utc(datetime.fromisoformat(ws_str)), ensure_utc(datetime.fromisoformat(we_str))


if __name__ == "__main__":
    sys.exit(main())
