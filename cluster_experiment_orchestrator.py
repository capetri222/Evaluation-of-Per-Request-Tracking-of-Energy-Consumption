#!/usr/bin/env python3
"""
Cluster Experiment Orchestrator v2
===================================
Executes experiments E1-E6 from Chapter 6 (Evaluation)

Usage:
    python3 cluster_experiment_orchestrator_v2.py --experiment E1 --action browse --repetitions 100
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

NAMESPACE = "christianpetri-otel-demo"
FRONTEND_URL = "http://localhost:8080"
OUTPUT_BASE_DIR = "cluster_experiments"
PROMETHEUS_URL = "http://localhost:9090"

# File paths for config files
OTEL_DEMO_VALUES_FILE = os.path.expanduser("~/mt/otel-demo-values.yaml")
PROMETHEUS_VALUES_FILE = os.path.expanduser("~/mt/prometheus-simple-values.yaml")

# Container lists for Kepler queries
CONTAINERS_E1_E4 = "accounting|ad|cart|checkout|currency|email|fraud-detection|frontend|frontend-proxy|image-provider|kafka|payment|postgresql|product-catalog|product-reviews|quote|recommendation|shipping|valkey-cart|flagd|flagd-ui|llm"

CONTAINERS_E5_E6 = CONTAINERS_E1_E4 + "|jaeger|opensearch|opentelemetry-collector|prometheus-server"

# Telemetry overhead services – collected for E3/E5/E6 to compare
# monitoring energy across fine / medium / coarse telemetry settings.
# Names verified against actual Kepler container_name labels in Prometheus.
TELEMETRY_OVERHEAD_CONTAINERS = "jaeger|opentelemetry-collector|prometheus-server|kepler|kepler-exporter"

# User actions for isolated experiments
USER_ACTIONS = {
    "browse": {
        "method": "GET",
        "path": "/",
        "description": "Browse homepage/product list"
    },
    "add_to_cart": {
        "method": "POST", 
        "path": "/api/cart",
        "payload": '{"productId":"OLJCESPC7Z","quantity":1}',
        "description": "Add product to cart"
    },
    "view_cart": {
        "method": "GET",
        "path": "/api/cart",
        "description": "View cart contents"
    },
    "checkout": {
        "method": "POST",
        "path": "/api/checkout",
        "payload": '{"email":"test@example.com","streetAddress":"123 Main St","city":"Berlin","state":"BE","country":"Germany","zipCode":"10115","creditCardNumber":"4111111111111111","creditCardExpirationMonth":"12","creditCardExpirationYear":"2030","creditCardCvv":"123"}',
        "description": "Complete checkout"
    }
}

# Telemetry configurations
TELEMETRY_CONFIGS = {
    "fine": {
        "sampling_rate": 100,  # percentage
        "scrape_interval": "1s",
        "scrape_timeout": "900ms",
        "step": "1s",
        "description": "Fine-grained telemetry (100% sampling, 1s scrape)"
    },
    "medium": {
        "sampling_rate": 50,
        "scrape_interval": "10s",
        "scrape_timeout": "9s",
        "step": "10s",
        "description": "Medium telemetry (50% sampling, 10s scrape)"
    },
    "coarse": {
        "sampling_rate": 1,
        "scrape_interval": "60s",
        "scrape_timeout": "54s",
        "step": "60s",
        "description": "Coarse telemetry (1% sampling, 60s scrape)"
    }
}

# Experiment configurations
EXPERIMENTS = {
    "E1": {
        "type": "isolated",
        "telemetry": "fine",
        "containers": CONTAINERS_E1_E4,
        "description": "Isolated requests for baseline estimation"
    },
    "E2": {
        "type": "concurrent",
        "users": 2,
        "duration": 600,
        "telemetry": "fine",
        "containers": CONTAINERS_E1_E4,
        "description": "Low concurrency"
    },
    "E3": {
        "type": "concurrent",
        "users": 10,
        "duration": 600,
        "telemetry": "fine",
        "containers": CONTAINERS_E1_E4,
        "description": "Medium concurrency"
    },
    "E4": {
        "type": "concurrent",
        "users": 50,
        "duration": 600,
        "telemetry": "fine",
        "containers": CONTAINERS_E1_E4,
        "description": "High concurrency"
    },
    "E5": {
        "type": "concurrent",
        "users": 10,
        "duration": 600,
        "telemetry": "medium",
        "containers": CONTAINERS_E5_E6,
        "description": "Medium concurrency with medium telemetry"
    },
    "E6": {
        "type": "concurrent",
        "users": 10,
        "duration": 600,
        "telemetry": "coarse",
        "containers": CONTAINERS_E5_E6,
        "description": "Medium concurrency with coarse telemetry"
    }
}

# Timing parameters
BASELINE_PRE_DURATION = 120  # 2 minutes fixed pre-baseline
BASELINE_POST_DURATION = 120  # 2 minutes fixed post-baseline (after cooldown)
FIXED_COOLDOWN = 30  # 30 seconds fixed cooldown (no measurements)
DYNAMIC_COOLDOWN_MAX = 600  # Maximum 10 minutes for dynamic cooldown
STABILITY_THRESHOLD = 0.10  # ±10% for stability check
TELEMETRY_WARMUP = 300  # 5 minutes after helm upgrade


# ══════════════════════════════════════════════════════════════════════
# PORT-FORWARD MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

class PortForwardManager:
    """Manages kubectl port-forward processes for the experiment."""
    
    def __init__(self):
        self.processes = {}  # port -> subprocess.Popen
        self.required_forwards = {
            8080: {"service": "frontend-proxy", "target_port": 8080, "name": "Frontend"},
            9090: {"service": "cpetri-prometheus-server", "target_port": 80, "name": "Prometheus"},
            16686: {"service": "jaeger", "target_port": 16686, "name": "Jaeger"}
        }
    
    def kill_existing_port_forwards(self, port: int) -> None:
        """Kill any existing process using the port."""
        try:
            result = subprocess.run(f"lsof -ti :{port}", shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    print(f"    Killing existing process on port {port} (PID: {pid})")
                    subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
                time.sleep(1)
        except Exception as e:
            print(f"    ⚠️  Could not check port {port}: {e}")
    
    def start_port_forward(self, port: int, service: str, target_port: int, name: str) -> bool:
        """Start a kubectl port-forward and return success status."""
        print(f"  Starting {name} port-forward ({port} → {target_port})...")
        
        # Kill existing processes on this port
        self.kill_existing_port_forwards(port)
        
        # Start port-forward
        cmd = f"kubectl port-forward svc/{service} {port}:{target_port} -n {NAMESPACE}"
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            self.processes[port] = process
            time.sleep(5)  # Give kubectl more time to establish connection
            
            # Check if process is still running
            if process.poll() is None:
                print(f"    ✅ {name} port-forward started")
                return True
            else:
                print(f"    ❌ {name} port-forward failed to start")
                return False
        except Exception as e:
            print(f"    ❌ Error starting {name}: {e}")
            return False
    
    def start_load_generator_forward(self) -> bool:
        """Start port-forward to load-generator pod."""
        print(f"  Starting Load-Generator port-forward...")
        
        # Get pod name
        rc, stdout, stderr = kubectl("get pod -l app.kubernetes.io/name=load-generator -o name", timeout=30)
        if rc != 0 or not stdout.strip():
            print(f"    ⚠️  No load-generator pod found")
            return False
        
        pod_name = stdout.strip().replace('pod/', '')
        
        # Kill existing
        self.kill_existing_port_forwards(8089)
        
        # Start forward
        cmd = f"kubectl port-forward {pod_name} 8089:8089 -n {NAMESPACE}"
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            self.processes[8089] = process
            time.sleep(2)
            
            if process.poll() is None:
                print(f"    ✅ Load-generator port-forward started")
                return True
            else:
                print(f"    ⚠️  Load-generator port-forward failed")
                return False
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    def stop_load_generator_forward(self) -> None:
        """Stop load-generator port-forward."""
        if 8089 in self.processes:
            print(f"  Stopping load-generator port-forward...")
            try:
                self.processes[8089].terminate()
                self.processes[8089].wait(timeout=5)
                del self.processes[8089]
                print(f"    ✅ Stopped")
            except Exception as e:
                print(f"    ⚠️  Error stopping: {e}")
    
    def healthcheck(self) -> bool:
        """Check if all port-forwards are healthy."""
        import requests
        
        print(f"\n{'─'*60}")
        print(f"HEALTHCHECK: Port-Forwards")
        print(f"{'─'*60}")
        
        checks = {
            "Jaeger": {
                "url": "http://localhost:16686/api/services",
                "port": 16686,
                "expected": lambda r: r.status_code == 200 and "data" in r.json()
            },
            "Prometheus": {
                "url": "http://localhost:9090/-/healthy",
                "port": 9090,
                "expected": lambda r: r.status_code == 200
            },
            "Frontend": {
                "url": "http://localhost:8080/",
                "port": 8080,
                "expected": lambda r: r.status_code == 200
            }
        }
        
        # Wait for port-forwards to initialize
        print(f"  Waiting for port-forwards to initialize...")
        time.sleep(5)
        
        all_healthy = True
        timeout = 60  # Increased from 30s
        start = time.time()
        
        for name, check in checks.items():
            print(f"  Checking {name}...", end=" ", flush=True)
            
            healthy = False
            attempts = 0
            while time.time() - start < timeout:
                try:
                    response = requests.get(check["url"], timeout=2)
                    if check["expected"](response):
                        healthy = True
                        break
                except Exception as e:
                    attempts += 1
                    pass
                time.sleep(3)  # Increased from 2s
            
            if healthy:
                print(f"✅")
            else:
                print(f"❌ (timeout after {attempts} attempts)")
                print(f"      Service '{name}' not responding on port {check['port']}")
                print(f"      URL: {check['url']}")
                all_healthy = False
        
        if all_healthy:
            print(f"\n  ✅ All services reachable")
        else:
            print(f"\n  ❌ Some services unreachable")
            print(f"  💡 Tip: Check if pods are running: kubectl get pods -n {NAMESPACE}")
        
        return all_healthy
    
    def start_all(self) -> bool:
        """Start all required port-forwards."""
        print(f"\n{'='*60}")
        print(f"STARTING PORT-FORWARDS")
        print(f"{'='*60}")
        
        success = True
        for port, config in self.required_forwards.items():
            if not self.start_port_forward(port, config["service"], config["target_port"], config["name"]):
                success = False
        
        if not success:
            print(f"\n  ❌ Some port-forwards failed to start")
            return False
        
        print(f"\n  ✅ All port-forwards started")
        return True
    
    def restart_after_redeploy(self) -> bool:
        """Restart port-forwards after pod recreation."""
        print(f"\n{'─'*60}")
        print(f"RESTARTING PORT-FORWARDS (after redeploy)")
        print(f"{'─'*60}")
        
        # Stop existing
        for port in list(self.processes.keys()):
            if port != 8089:  # Keep load-generator separate
                try:
                    self.processes[port].terminate()
                    self.processes[port].wait(timeout=5)
                    del self.processes[port]
                except:
                    pass
        
        # Restart
        return self.start_all()
    
    def stop_all(self) -> None:
        """Stop all port-forwards managed by this instance."""
        print(f"\n{'─'*60}")
        print(f"STOPPING PORT-FORWARDS")
        print(f"{'─'*60}")
        
        for port, process in list(self.processes.items()):
            print(f"  Stopping port {port}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                try:
                    process.kill()
                except:
                    pass
        
        self.processes.clear()
        print(f"  ✅ All port-forwards stopped")


# Global port-forward manager instance
port_manager = PortForwardManager()


# ══════════════════════════════════════════════════════════════════════
# KUBERNETES UTILITIES
# ══════════════════════════════════════════════════════════════════════

def kubectl(cmd: str, namespace: str = NAMESPACE, timeout: int = 60) -> Tuple[int, str, str]:
    """Execute kubectl command with timeout."""
    full_cmd = f"kubectl -n {namespace} {cmd}"
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"


def ensure_load_generator_off() -> None:
    """Ensure load-generator is scaled to 0 and wait for stabilization if needed."""
    print(f"\n{'─'*60}")
    print(f"CHECKING LOAD-GENERATOR STATUS")
    print(f"{'─'*60}")
    
    # Check if any load-generator pods are running
    rc, stdout, stderr = kubectl("get pods -l app.kubernetes.io/name=load-generator -o name", timeout=30)
    
    if rc != 0:
        print(f"  ⚠️  Could not check pods: {stderr}")
        print(f"  Assuming no pods running, continuing...")
        return
    
    pods = [line for line in stdout.strip().split('\n') if line.strip()]
    
    if not pods:
        print(f"  ✅ No load-generator pods running")
        return
    
    # Pods exist - need to scale down and wait
    print(f"  ⚠️  Found {len(pods)} load-generator pod(s)")
    print(f"  Scaling down to 0...")
    
    rc, stdout, stderr = kubectl("scale deployment/load-generator --replicas=0", timeout=60)
    
    if rc == 0:
        print(f"  ✅ Scaled to 0 replicas")
    else:
        print(f"  ⚠️  Scale command had issues, but continuing...")
    
    # Wait 5 minutes for stabilization
    print(f"\n{'─'*60}")
    print(f"STABILIZATION WAIT (5 minutes)")
    print(f"{'─'*60}")
    print(f"  Cluster needs time to stabilize after scale-down...")
    
    for remaining in range(300, 0, -30):
        mins, secs = divmod(remaining, 60)
        print(f"    Time remaining: {mins}m {secs}s", end="\r")
        time.sleep(30)
    
    print("\n  ✅ Stabilization complete")


def start_load_generator(users: int) -> None:
    """Start load-generator with specified number of users."""
    print(f"\n{'─'*60}")
    print(f"STARTING LOAD-GENERATOR")
    print(f"{'─'*60}")
    print(f"  Users: {users}")
    
    # Set environment variable
    print(f"  [1/2] Configuring user count...")
    kubectl(f"set env deployment/load-generator LOCUST_USERS={users}", timeout=30)
    
    # Scale to 1
    print(f"  [2/2] Scaling to 1 replica...")
    rc, stdout, stderr = kubectl("scale deployment/load-generator --replicas=1", timeout=60)
    
    if rc != 0:
        print(f"  ❌ Scale failed: {stderr}")
        sys.exit(1)
    
    # Wait for pod to be ready
    print(f"  ⏳ Waiting for pod to be ready...")
    rc, stdout, stderr = kubectl("wait --for=condition=ready pod -l app.kubernetes.io/name=load-generator --timeout=90s", timeout=120)
    
    if rc != 0:
        print(f"  ⚠️  Pod not ready: {stderr}")
    else:
        print(f"  ✅ Load-generator ready")
    
    # Start load-generator port-forward
    port_manager.start_load_generator_forward()


def stop_load_generator() -> None:
    """Stop load-generator (scale to 0)."""
    print(f"\n{'─'*60}")
    print(f"STOPPING LOAD-GENERATOR")
    print(f"{'─'*60}")
    
    # Stop port-forward first
    port_manager.stop_load_generator_forward()
    
    kubectl("scale deployment/load-generator --replicas=0", timeout=60)
    print(f"  ✅ Scaled to 0 (shutting down)")


# ══════════════════════════════════════════════════════════════════════
# TELEMETRY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

def get_current_sampling_rate() -> Optional[int]:
    """Get current OTel sampling rate from values file."""
    if not os.path.exists(OTEL_DEMO_VALUES_FILE):
        return None
    
    with open(OTEL_DEMO_VALUES_FILE, 'r') as f:
        content = f.read()
    
    match = re.search(r'sampling_percentage:\s*(\d+)', content)
    if match:
        return int(match.group(1))
    return None


def get_current_scrape_interval() -> Optional[str]:
    """Get current Prometheus scrape interval from values file."""
    if not os.path.exists(PROMETHEUS_VALUES_FILE):
        return None
    
    with open(PROMETHEUS_VALUES_FILE, 'r') as f:
        content = f.read()
    
    match = re.search(r'scrape_interval:\s*(\d+[smh])', content)
    if match:
        return match.group(1)
    return None


def configure_telemetry(config_name: str) -> None:
    """
    Configure Prometheus scrape interval and OTel sampling rate.
    Uses sed + helm upgrade as per your instructions.
    """
    print(f"\n{'='*60}")
    print(f"CONFIGURING TELEMETRY: {config_name.upper()}")
    print(f"{'='*60}")
    
    config = TELEMETRY_CONFIGS[config_name]
    target_sampling = config['sampling_rate']
    target_scrape = config['scrape_interval']
    target_timeout = config['scrape_timeout']
    
    print(f"  Target Sampling Rate:   {target_sampling}%")
    print(f"  Target Scrape Interval: {target_scrape}")
    print(f"  Target Scrape Timeout:  {target_timeout}")
    
    # Check current config
    current_sampling = get_current_sampling_rate()
    current_scrape = get_current_scrape_interval()
    
    print(f"\n  Current Sampling Rate:   {current_sampling}%")
    print(f"  Current Scrape Interval: {current_scrape}")
    
    needs_update = (current_sampling != target_sampling or current_scrape != target_scrape)
    
    if not needs_update:
        print(f"\n  ✅ Telemetry already configured correctly")
        return
    
    print(f"\n  ⚠️  Configuration mismatch - updating...")
    
    # 1. Update OTel sampling rate
    if current_sampling != target_sampling:
        print(f"\n[1/4] Updating OTel sampling rate...")
        
        if current_sampling:
            cmd = f"sed -i 's/sampling_percentage: {current_sampling}/sampling_percentage: {target_sampling}/' {OTEL_DEMO_VALUES_FILE}"
        else:
            print(f"  ⚠️  Could not detect current sampling rate, please update manually")
            cmd = None
        
        if cmd:
            print(f"  $ {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ sed failed: {result.stderr}")
                sys.exit(1)
        
        print(f"  Running helm upgrade for otel-demo...")
        cmd = f"helm upgrade otel-demo open-telemetry/opentelemetry-demo -n {NAMESPACE} -f {OTEL_DEMO_VALUES_FILE}"
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ Helm upgrade failed: {result.stderr}")
            sys.exit(1)
        
        print(f"  ✅ OTel sampling rate updated")
    
    # 2. Update Prometheus scrape interval
    if current_scrape != target_scrape:
        print(f"\n[2/4] Updating Prometheus scrape interval...")
        
        if current_scrape:
            # Update scrape_interval
            cmd1 = f"sed -i 's/scrape_interval: {current_scrape}/scrape_interval: {target_scrape}/g' {PROMETHEUS_VALUES_FILE}"
            print(f"  $ {cmd1}")
            result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
            
            # Update scrape_timeout
            current_timeout_match = re.search(r'scrape_timeout:\s*(\S+)', open(PROMETHEUS_VALUES_FILE).read())
            if current_timeout_match:
                current_timeout = current_timeout_match.group(1)
                cmd2 = f"sed -i 's/scrape_timeout: {current_timeout}/scrape_timeout: {target_timeout}/g' {PROMETHEUS_VALUES_FILE}"
                print(f"  $ {cmd2}")
                subprocess.run(cmd2, shell=True)
        
        print(f"  Running helm upgrade for Prometheus...")
        cmd = f"helm upgrade cpetri-prometheus prometheus-community/prometheus -n {NAMESPACE} -f {PROMETHEUS_VALUES_FILE}"
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ Helm upgrade failed: {result.stderr}")
            sys.exit(1)
        
        print(f"  ✅ Prometheus scrape interval updated")
    
    # 3. Ensure load-generator is off (will wait 5min only if it was running)
    print(f"\n[3/4] Ensuring load-generator is off...")
    ensure_load_generator_off()
    
    # 4. Wait for warmup
    print(f"\n[4/4] Warmup period...")
    print(f"  ⏰ Additional 5-minute warmup after helm upgrade...")
    
    for remaining in range(300, 0, -30):
        mins, secs = divmod(remaining, 60)
        print(f"    Warmup: {mins}m {secs}s remaining", end="\r")
        time.sleep(30)
    
    print("\n  ✅ Warmup complete")


# ══════════════════════════════════════════════════════════════════════
# BASELINE MEASUREMENT
# ══════════════════════════════════════════════════════════════════════

def measure_baseline_power(duration: int, containers: str) -> Dict[str, float]:
    """Measure baseline power via Prometheus Kepler metrics."""
    import requests
    
    print(f"  ⏱️  Measuring for {duration}s...")
    
    # Wait with countdown
    for remaining in range(duration, 0, -30):
        if remaining > 30:
            print(f"    Time remaining: {remaining}s", end="\r")
        time.sleep(min(30, remaining))
    
    if duration >= 30:
        print()  # New line after countdown
    
    baseline = {}
    
    # Query per-service power (package + dram)
    for zone in ["package", "dram"]:
        query = f'sum by (container_name) (rate(kepler_container_cpu_joules_total{{container_name=~"{containers}", zone="{zone}"}}[{duration}s]))'
        
        try:
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                for result in data["data"]["result"]:
                    service = result["metric"]["container_name"]
                    power = float(result["value"][1])
                    baseline.setdefault(service, 0.0)
                    baseline[service] += power
        
        except Exception as e:
            print(f"    ⚠️  Could not query {zone}: {e}")
    
    total_power = sum(baseline.values())
    print(f"  ✓ Total baseline power: {total_power:.6f} W")
    
    return baseline


def query_baseline_power_no_wait(duration: int, containers: str) -> Dict[str, float]:
    """
    Query baseline power for the LAST duration seconds without waiting.
    Used for sliding window checks where we query historical data.
    """
    import requests
    
    baseline = {}
    
    # Query per-service power (package + dram) for last duration seconds
    for zone in ["package", "dram"]:
        query = f'sum by (container_name) (rate(kepler_container_cpu_joules_total{{container_name=~"{containers}", zone="{zone}"}}[{duration}s]))'
        
        try:
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                for result in data["data"]["result"]:
                    service = result["metric"]["container_name"]
                    power = float(result["value"][1])
                    baseline.setdefault(service, 0.0)
                    baseline[service] += power
        
        except Exception as e:
            print(f"    ⚠️  Could not query {zone}: {e}")
    
    return baseline
    
    return baseline


def measure_pre_baseline(containers: str) -> Dict:
    """Measure pre-execution baseline."""
    print(f"\n{'='*60}")
    print(f"PRE-BASELINE MEASUREMENT")
    print(f"{'='*60}")
    
    baseline = measure_baseline_power(BASELINE_PRE_DURATION, containers)
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": BASELINE_PRE_DURATION,
        "baselines": baseline,
        "total_power_watts": sum(baseline.values())
    }
    
    print(f"  ✅ Pre-baseline measured")
    return result


def execute_cooldown_and_post_baseline(pre_baseline: Dict, containers: str) -> Tuple[Dict, datetime]:
    """
    Combined cooldown + post-baseline measurement with Dynamic Window Expansion (DWE).
    
    NEW LOGIC: Only close window when post_power ≤ 1.1 * pre_power
    - Higher post-power indicates ongoing activity → keep window open
    - Equal or lower post-power is OK (CPU may have downscaled) → close window
    
    Timeline:
    - t=0-30s: Fixed cooldown (no measurements)
    - t=30s: Post-baseline window STARTS
    - t=30-150s: Post-baseline window measures (120s)
    - t=150s: First stability check
      - If post_power ≤ 1.1 * pre_power → DONE, stabilization_time = 30s
      - If post_power > 1.1 * pre_power → slide window +1s, check every 1s
    - t=151s+: Check last 120s every second until stable or timeout
    
    stabilization_time = 30s + seconds_slid
    
    Returns: (baseline_post_dict, cutoff_timestamp)
    """
    print(f"\n{'='*60}")
    print(f"COOLDOWN + POST-BASELINE")
    print(f"{'='*60}")
    
    pre_power = pre_baseline["total_power_watts"]
    threshold_power = pre_power * 1.1  # +10%
    print(f"  Pre-baseline power: {pre_power:.6f} W")
    print(f"  Closing threshold:  ≤ {threshold_power:.6f} W (+10%)")
    
    # Phase 1: Fixed cooldown (30s) - no measurements
    print(f"\n{'─'*60}")
    print(f"FIXED COOLDOWN ({FIXED_COOLDOWN}s)")
    print(f"{'─'*60}")
    print(f"  Waiting for async workload energy to settle...")
    
    for remaining in range(FIXED_COOLDOWN, 0, -10):
        print(f"    Time remaining: {remaining}s", end="\r")
        time.sleep(10)
    
    print("\n  ✅ Fixed cooldown complete")
    print(f"  Post-baseline window starts NOW")
    
    # Mark the cutoff time (end of cooldown = start of post-baseline window)
    cooldown_end_time = datetime.now(timezone.utc)
    
    # Phase 2: Post-baseline window (120s) - measure the NEXT 120 seconds
    print(f"\n{'─'*60}")
    print(f"POST-BASELINE WINDOW ({BASELINE_POST_DURATION}s)")
    print(f"{'─'*60}")
    print(f"  Measuring baseline over next {BASELINE_POST_DURATION} seconds...")
    
    # measure_baseline_power already does time.sleep + measurement
    baseline_values = measure_baseline_power(BASELINE_POST_DURATION, containers)
    post_power = sum(baseline_values.values())
    
    print(f"  ✓ Window complete: {post_power:.6f} W")
    
    # Phase 3: Stability check
    print(f"\n{'─'*60}")
    print(f"STABILITY CHECK")
    print(f"{'─'*60}")
    print(f"\n  📊 Baseline Comparison:")
    print(f"     Pre-baseline:  {pre_power:.6f} W")
    print(f"     Post-baseline: {post_power:.6f} W")
    print(f"     Difference:    {post_power - pre_power:+.6f} W")
    print(f"     Threshold:     ≤ {threshold_power:.6f} W")
    print(f"")
    
    if post_power <= threshold_power:
        # Stable! Post is at or below threshold
        # Cutoff is at END of cooldown (START of post-baseline window)
        cutoff_timestamp = cooldown_end_time
        
        print(f"  ✅ System stable (post ≤ threshold)!")
        print(f"     Stabilization time: {FIXED_COOLDOWN}s")
        print(f"  📍 Cutoff timestamp: {cutoff_timestamp.isoformat()}")
        
        result = {
            "timestamp": cutoff_timestamp.isoformat(),
            "duration_seconds": BASELINE_POST_DURATION,
            "stabilization_time_seconds": FIXED_COOLDOWN,
            "baselines": baseline_values,
            "total_power_watts": post_power,
            "deviation_from_pre": (post_power - pre_power) / pre_power if pre_power > 0 else 0
        }
        
        return result, cutoff_timestamp
    
    # Not stable - post_power > threshold (still too high)
    print(f"  ⚠️  Post-power too high: {post_power:.6f} W > {threshold_power:.6f} W")
    print(f"  Sliding window (checking every 10s over last 120s)...")
    
    seconds_slid = 0
    max_slide = DYNAMIC_COOLDOWN_MAX - FIXED_COOLDOWN - BASELINE_POST_DURATION
    check_interval = 10  # Check every 10 seconds
    
    while seconds_slid < max_slide:
        # Wait for next check
        time.sleep(check_interval)
        seconds_slid += check_interval
        
        # Query Prometheus for LAST 120 seconds (true sliding window!)
        # This does NOT wait - it queries historical data
        baseline_values = query_baseline_power_no_wait(BASELINE_POST_DURATION, containers)
        post_power = sum(baseline_values.values())
        
        total_time = FIXED_COOLDOWN + seconds_slid
        status = "OK" if post_power <= threshold_power else "HIGH"
        
        # Always print status on each check
        print(f"    +{seconds_slid}s: {post_power:.6f} W [{status}] (stabilization={total_time}s)")
        
        if post_power <= threshold_power:
            # Stable! Post is now at or below threshold
            # Cutoff is at end of cooldown + sliding time
            cutoff_timestamp = cooldown_end_time + timedelta(seconds=seconds_slid)
            stabilization_time = FIXED_COOLDOWN + seconds_slid
            
            print(f"\n  ✅ System stabilized (post ≤ threshold)!")
            print(f"     Stabilization time: {stabilization_time}s (30s fixed + {seconds_slid}s sliding)")
            print(f"  📍 Cutoff timestamp: {cutoff_timestamp.isoformat()}")
            
            result = {
                "timestamp": cutoff_timestamp.isoformat(),
                "duration_seconds": BASELINE_POST_DURATION,
                "stabilization_time_seconds": stabilization_time,
                "baselines": baseline_values,
                "total_power_watts": post_power,
                "deviation_from_pre": (post_power - pre_power) / pre_power if pre_power > 0 else 0
            }
            
            return result, cutoff_timestamp
    
    # Timeout - post_power still above threshold
    cutoff_timestamp = cooldown_end_time + timedelta(seconds=seconds_slid)
    stabilization_time = FIXED_COOLDOWN + seconds_slid
    
    print(f"\n  ⚠️  Timeout after sliding {seconds_slid}s")
    print(f"     Stabilization time: {stabilization_time}s")
    print(f"     Final post-power: {post_power:.6f} W (still > {threshold_power:.6f} W)")
    print(f"  📍 Cutoff timestamp: {cutoff_timestamp.isoformat()}")
    
    result = {
        "timestamp": cutoff_timestamp.isoformat(),
        "duration_seconds": BASELINE_POST_DURATION,
        "stabilization_time_seconds": stabilization_time,
        "baselines": baseline_values,
        "total_power_watts": post_power,
        "deviation_from_pre": (post_power - pre_power) / pre_power if pre_power > 0 else 0,
        "warning": f"Post-power still high after {seconds_slid}s sliding"
    }
    
    return result, cutoff_timestamp


# ══════════════════════════════════════════════════════════════════════
# WORKLOAD EXECUTION
# ══════════════════════════════════════════════════════════════════════

def execute_isolated_request(action: str, repetitions: int) -> Tuple[datetime, datetime]:
    """
    Execute isolated requests with session persistence.
    
    Session Strategy:
    - browse: Creates sessions, saves cookies (cookie_0.txt, cookie_1.txt, ...)
    - add_to_cart: Loads existing cookies, adds items, saves cookies
    - view_cart: Loads existing cookies (carts should have items)
    - checkout: Loads existing cookies, performs checkout
    
    Cookie files stored in /tmp/experiment_cookies/
    """
    print(f"\n{'='*60}")
    print(f"EXECUTING ISOLATED WORKLOAD")
    print(f"{'='*60}")
    print(f"  Action: {action}")
    print(f"  Repetitions: {repetitions}")
    
    action_config = USER_ACTIONS[action]
    method = action_config["method"]
    path = action_config["path"]
    url = f"{FRONTEND_URL}{path}"
    
    print(f"  URL: {url}")
    print(f"  Method: {method}")
    
    # Create cookie directory
    cookie_dir = "/tmp/experiment_cookies"
    os.makedirs(cookie_dir, exist_ok=True)
    
    # Session handling based on action
    if action == "browse":
        print(f"  📝 Creating {repetitions} sessions (saving cookies)")
    elif action in ["add_to_cart", "view_cart", "checkout"]:
        print(f"  🔄 Using existing sessions from cookies")
        print(f"  ⚠️  Ensure 'browse' was run first to create sessions!")
    
    start_time = datetime.now(timezone.utc)
    
    # Execute requests in parallel batches for speed
    # Use batches of 10 to avoid overwhelming the server
    batch_size = 10
    total_batches = (repetitions + batch_size - 1) // batch_size
    completed = 0
    
    def execute_single_request(i):
        """Execute a single request with session handling."""
        cookie_file = f"{cookie_dir}/cookie_{i}.txt"
        
        if action == "browse":
            # Create new session, save cookie
            cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -c {cookie_file} {url}"
            
        elif action == "add_to_cart":
            # Load cookie (from browse), add item, save cookie
            payload = action_config.get("payload", "{}")
            
            if os.path.exists(cookie_file):
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -b {cookie_file} -c {cookie_file} -X POST -H 'Content-Type: application/json' -d '{payload}' {url}"
            else:
                # Fallback: create session first
                subprocess.run(f"curl -s -o /dev/null -c {cookie_file} {FRONTEND_URL}/", 
                             shell=True, capture_output=True)
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -b {cookie_file} -c {cookie_file} -X POST -H 'Content-Type: application/json' -d '{payload}' {url}"
            
        elif action == "view_cart":
            # Load cookie (should have items from add_to_cart)
            if os.path.exists(cookie_file):
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -b {cookie_file} {url}"
            else:
                # Fallback: create session + add item first
                subprocess.run(f"curl -s -o /dev/null -c {cookie_file} {FRONTEND_URL}/", 
                             shell=True, capture_output=True)
                add_payload = '{"productId":"OLJCESPC7Z","quantity":1}'
                subprocess.run(f"curl -s -o /dev/null -b {cookie_file} -c {cookie_file} -X POST -H 'Content-Type: application/json' -d '{add_payload}' {FRONTEND_URL}/api/cart",
                             shell=True, capture_output=True)
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -b {cookie_file} {url}"
            
        elif action == "checkout":
            # Load cookie (should have items), checkout
            payload = action_config.get("payload", "{}")
            
            if os.path.exists(cookie_file):
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -b {cookie_file} -X POST -H 'Content-Type: application/json' -d '{payload}' {url}"
            else:
                # Fallback: create session + add item first
                subprocess.run(f"curl -s -o /dev/null -c {cookie_file} {FRONTEND_URL}/", 
                             shell=True, capture_output=True)
                add_payload = '{"productId":"OLJCESPC7Z","quantity":1}'
                subprocess.run(f"curl -s -o /dev/null -b {cookie_file} -c {cookie_file} -X POST -H 'Content-Type: application/json' -d '{add_payload}' {FRONTEND_URL}/api/cart",
                             shell=True, capture_output=True)
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -b {cookie_file} -X POST -H 'Content-Type: application/json' -d '{payload}' {url}"
        
        else:
            # Default: no session handling
            if method == "GET":
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {url}"
            else:
                payload = action_config.get("payload", "{}")
                cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -X POST -H 'Content-Type: application/json' -d '{payload}' {url}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return i, result.returncode
    
    # Execute in parallel batches
    for batch_start in range(0, repetitions, batch_size):
        batch_end = min(batch_start + batch_size, repetitions)
        batch_indices = range(batch_start, batch_end)
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(execute_single_request, i): i for i in batch_indices}
            for future in as_completed(futures):
                i, returncode = future.result()
                completed += 1
                if completed % 10 == 0 or completed == repetitions:
                    print(f"    [{completed}/{repetitions}]", end="\r")
        
        # Small delay between batches to avoid overwhelming server
        if batch_end < repetitions:
            time.sleep(0.1)
    
    end_time = datetime.now(timezone.utc)
    
    print(f"\n  ✅ Completed {repetitions} requests")
    print(f"  Duration: {(end_time - start_time).total_seconds():.2f}s")
    
    if action == "browse":
        print(f"  📝 Saved {repetitions} session cookies to {cookie_dir}/")
    
    return start_time, end_time


def execute_concurrent_workload(users: int, duration: int) -> Tuple[datetime, datetime]:
    """Execute concurrent workload via Locust."""
    print(f"\n{'='*60}")
    print(f"EXECUTING CONCURRENT WORKLOAD")
    print(f"{'='*60}")
    print(f"  Users: {users}")
    print(f"  Duration: {duration}s")
    
    # Start load-generator
    start_load_generator(users)
    
    # Run workload
    print(f"\n  Running load for {duration}s...")
    start_time = datetime.now(timezone.utc)
    
    for remaining in range(duration, 0, -10):
        print(f"    Time remaining: {remaining}s", end="\r")
        time.sleep(10)
    
    end_time = datetime.now(timezone.utc)
    
    print(f"\n  ✅ Load generation complete")
    
    # Stop load-generator
    stop_load_generator()
    
    return start_time, end_time


# ══════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════

def collect_kepler_metrics(start_time: datetime, end_time: datetime, output_dir: str, 
                          step: str, containers: str) -> str:
    """Collect Kepler metrics from Prometheus as CSV."""
    print(f"\n{'─'*60}")
    print(f"COLLECTING KEPLER METRICS")
    print(f"{'─'*60}")
    
    import requests
    
    print(f"  Time range: {start_time} → {end_time}")
    print(f"  Step: {step}")
    print(f"  Containers: {len(containers.split('|'))} services")
    
    output_file = os.path.join(output_dir, "kepler_metrics.csv")
    
    # Write CSV header
    with open(output_file, "w") as f:
        f.write("timestamp,service,zone,joules_total\n")
    
    total_services_found = 0
    total_datapoints = 0
    
    # Query each zone
    for zone in ["package", "dram"]:
        print(f"\n  Querying zone={zone}...")
        
        query = f'sum by (container_name) (kepler_container_cpu_joules_total{{container_name=~"{containers}", zone="{zone}"}})'
        
        params = {
            "query": query,
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
            "step": step
        }
        
        print(f"    Query: {query[:80]}...")
        print(f"    Params: start={params['start']}, end={params['end']}, step={step}")
        
        try:
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=60)
            
            if resp.status_code != 200:
                print(f"    ❌ Query failed: HTTP {resp.status_code}")
                print(f"       Response: {resp.text[:200]}")
                continue
            
            data = resp.json()
            
            if data.get("status") != "success":
                print(f"    ❌ Query not successful: {data.get('error', 'unknown error')}")
                continue
            
            results = data["data"]["result"]
            
            if not results:
                print(f"    ⚠️  No data returned (0 services)")
                continue
            
            # Write data points for ALL services
            datapoints_this_zone = 0
            with open(output_file, "a") as f:
                for result in results:
                    service = result["metric"]["container_name"]
                    values = result["values"]
                    
                    for timestamp, value in values:
                        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                        f.write(f"{dt},{service},{zone},{value}\n")
                        datapoints_this_zone += 1
            
            total_services_found += len(results)
            total_datapoints += datapoints_this_zone
            
            print(f"    ✓ {len(results)} services, {datapoints_this_zone} data points")
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  ✅ Metrics saved: {output_file}")
    print(f"  📊 Total: {total_services_found} services, {total_datapoints} data points")
    
    return output_file


def collect_telemetry_overhead(start_time: datetime, end_time: datetime,
                               output_dir: str, step: str) -> str:
    """
    Collect energy of telemetry services using Prometheus increase() over the
    exact experiment window. Avoids cumulative-counter artefacts from services
    that run continuously across experiments (kepler, prometheus-server).

    Output: telemetry_overhead_summary.json (authoritative)
            telemetry_overhead.csv           (raw timeseries, reference only)
    """
    print(f"\n{'─'*60}")
    print(f"COLLECTING TELEMETRY OVERHEAD METRICS")
    print(f"{'─'*60}")

    import requests
    from collections import defaultdict

    output_file  = os.path.join(output_dir, "telemetry_overhead.csv")
    summary_file = os.path.join(output_dir, "telemetry_overhead_summary.json")

    duration_s = int((end_time - start_time).total_seconds())

    # ── increase() at end of window → exact J consumed during window ─────────
    svc_energy: Dict[str, float] = defaultdict(float)

    for zone in ["package", "dram"]:
        query = (
            f'sum by (container_name) ('
            f'increase(kepler_container_cpu_joules_total{{'
            f'container_name=~"{TELEMETRY_OVERHEAD_CONTAINERS}",'
            f'zone="{zone}"'
            f'}}[{duration_s}s])' 
            f')' 
        )
        try:
            resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query, "time": int(end_time.timestamp())},
                timeout=60
            )
            if resp.status_code != 200:
                print(f"  ❌ zone={zone} failed: HTTP {resp.status_code}")
                continue
            data = resp.json()
            if data.get("status") != "success":
                print(f"  ❌ zone={zone}: {data.get('error')}")
                continue
            for r in data["data"]["result"]:
                svc = r["metric"].get("container_name", "unknown")
                val = float(r["value"][1])
                if val > 0:
                    svc_energy[svc] += val
            print(f"  ✓ zone={zone}: {len(data['data']['result'])} services via increase()")
        except Exception as e:
            print(f"  ❌ zone={zone} error: {e}")

    # ── Normalize: merge kepler + kepler-exporter ─────────────────────────────
    def normalize_svc(name: str) -> str:
        return "kepler" if "kepler" in name else name

    merged: Dict[str, float] = defaultdict(float)
    for svc, j in svc_energy.items():
        merged[normalize_svc(svc)] += j

    total_overhead_J = sum(merged.values())

    # ── Raw timeseries CSV (reference, not used for summary values) ───────────
    with open(output_file, "w") as f:
        f.write("timestamp,service,zone,joules_total\n")
    for zone in ["package", "dram"]:
        query = (
            f'sum by (container_name) ('
            f'kepler_container_cpu_joules_total{{'
            f'container_name=~"{TELEMETRY_OVERHEAD_CONTAINERS}",zone="{zone}"}}'
            f')' 
        )
        try:
            resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params={"query": query, "start": int(start_time.timestamp()),
                        "end": int(end_time.timestamp()), "step": step},
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    with open(output_file, "a") as f:
                        for r in data["data"]["result"]:
                            svc = r["metric"].get("container_name", "unknown")
                            for ts, val in r["values"]:
                                dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                                f.write(f"{dt},{svc},{zone},{val}\n")
        except Exception:
            pass

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "method": "increase()",
        "telemetry_step": step,
        "window_start":   start_time.isoformat(),
        "window_end":     end_time.isoformat(),
        "duration_s":     duration_s,
        "services":       dict(merged),
        "total_overhead_J": total_overhead_J,
        "note": (
            "Energy via Prometheus increase() over exact experiment window. "
            "kepler and kepler-exporter are merged into kepler."
        ),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  📊 Telemetry Overhead (increase method):")
    for svc, j in sorted(merged.items(), key=lambda x: -x[1]):
        print(f"     {svc:<35} {j:>8.2f} J")
    print(f"     {'TOTAL':<35} {total_overhead_J:>8.2f} J")
    print(f"\n  ✅ {summary_file}")

    return output_file

def collect_traces(start_time: datetime, end_time: datetime, output_dir: str) -> str:
    """Collect traces using trace_collector.py."""
    print(f"\n{'─'*60}")
    print(f"COLLECTING TRACES")
    print(f"{'─'*60}")
    
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    output_file = os.path.join(output_dir, "raw_traces.json")
    
    cmd = [
        "python3", "trace_collector.py",
        "--start", start_str,
        "--end", end_str,
        "--output", output_file
    ]
    
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    # Show output for debugging
    if result.stdout:
        print(f"\n  Output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"    {line}")
    
    if result.returncode != 0:
        print(f"\n  ⚠️  Trace collection failed (exit code {result.returncode})")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        # Create empty traces file
        with open(output_file, 'w') as f:
            json.dump({
                "window_start": start_str,
                "window_end": end_str,
                "trace_count": 0,
                "traces": [],
                "error": "Trace collection failed"
            }, f, indent=2)
    else:
        # Check if file was created
        if os.path.exists(output_file):
            # Check trace count
            with open(output_file, 'r') as f:
                trace_data = json.load(f)
            trace_count = trace_data.get("trace_count", 0)
            if trace_count > 0:
                print(f"  ✅ Traces saved: {output_file} ({trace_count} traces)")
            else:
                print(f"  ⚠️  No traces found in time window")
                print(f"      File: {output_file}")
        else:
            print(f"  ⚠️  Trace file not created: {output_file}")
    
    return output_file


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT EXECUTION
# ══════════════════════════════════════════════════════════════════════

def create_experiment_directory(experiment_id: str, **kwargs) -> str:
    """Create experiment output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_id == "E1":
        action = kwargs.get("action", "unknown")
        repetitions = kwargs.get("repetitions", 0)
        dir_name = f"{experiment_id}_isolated_{action}_N{repetitions}_{timestamp}"
    else:
        dir_name = f"{experiment_id}_{timestamp}"
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, experiment_id, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT DIRECTORY")
    print(f"{'='*60}")
    print(f"  {output_dir}")
    
    return output_dir


def save_manifest(output_dir: str, experiment_config: Dict, start_time: datetime, 
                 end_time: datetime) -> None:
    """Save experiment manifest (without baseline duplication)."""
    manifest = {
        "experiment": experiment_config,
        "execution": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        },
        "output_directory": output_dir
    }
    
    manifest_file = os.path.join(output_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n  ✅ Manifest saved: {manifest_file}")


def run_experiment(experiment_id: str, **kwargs) -> None:
    """Execute complete experiment with clean redeploy."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT {experiment_id}")
    print(f"{'#'*60}")
    
    # Get experiment config
    experiment_config = EXPERIMENTS[experiment_id].copy()
    experiment_config.update(kwargs)
    
    print(f"\nConfiguration:")
    for key, value in experiment_config.items():
        if key != "containers":  # too long
            print(f"  {key}: {value}")
    
    # REDEPLOY FOR CLEAN STATE
    print(f"\n{'='*60}")
    print(f"REDEPLOYING OTEL-DEMO")
    print(f"{'='*60}")
    
    # Get telemetry config
    telemetry = experiment_config["telemetry"]
    telemetry_config = TELEMETRY_CONFIGS[telemetry]
    target_sampling = telemetry_config["sampling_rate"]
    target_scrape = telemetry_config["scrape_interval"]
    
    # Check current config
    current_sampling = get_current_sampling_rate()
    current_scrape = get_current_scrape_interval()
    needs_update = (current_sampling != target_sampling or current_scrape != target_scrape)
    
    if needs_update:
        print(f"  Telemetry will be updated:")
        print(f"    Sampling: {current_sampling}% → {target_sampling}%")
        print(f"    Scrape:   {current_scrape} → {target_scrape}")
    
    # Uninstall (check if running first)
    print(f"\n  [1/5] Checking if otel-demo is running...")
    check_result = subprocess.run(
        f"kubectl get pods -n {NAMESPACE} -l app.kubernetes.io/instance=otel-demo --no-headers 2>/dev/null | wc -l",
        shell=True, capture_output=True, text=True
    )
    
    pod_count = int(check_result.stdout.strip()) if check_result.stdout.strip().isdigit() else 0
    
    if pod_count > 0:
        print(f"      Found {pod_count} otel-demo pods, uninstalling...")
        subprocess.run(f"helm uninstall otel-demo -n {NAMESPACE}", shell=True, capture_output=True)
        print(f"      Waiting 30s for cleanup...")
        time.sleep(30)
        print(f"      ✅ Uninstalled")
    else:
        print(f"      ✅ No otel-demo pods running, skipping uninstall")
    
    # Update configs if needed
    if needs_update:
        print(f"  [2/5] Updating configs...")
        if current_sampling and current_sampling != target_sampling:
            subprocess.run(f"sed -i 's/sampling_percentage: {current_sampling}/sampling_percentage: {target_sampling}/' ~/mt/otel-demo-values.yaml", shell=True)
        if current_scrape and current_scrape != target_scrape:
            subprocess.run(f"sed -i 's/scrape_interval: {current_scrape}/scrape_interval: {target_scrape}/g' ~/mt/prometheus-simple-values.yaml", shell=True)
        print(f"      ✅ Configs updated")
    else:
        print(f"  [2/5] Configs correct, skipping")
    
    # Install
    print(f"  [3/5] Installing...")
    result = subprocess.run(
        f"helm install otel-demo open-telemetry/opentelemetry-demo -n {NAMESPACE} -f ~/mt/otel-demo-values.yaml",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"      ❌ Failed: {result.stderr}")
        sys.exit(1)
    
    # Wait and scale down load-generator
    print(f"  [4/5] Scaling down load-generator...")
    time.sleep(10)  # Brief wait for deployment to register
    kubectl("scale deployment/load-generator --replicas=0", timeout=30)
    print(f"      ✅ Load-generator scaled to 0")
    
    # Warmup
    print(f"  [5/5] Warmup (3 minutes)...")
    time.sleep(180)
    print(f"  ✅ Deployment ready")
    
    # Start port-forwards (pods are ready now)
    if not port_manager.start_all():
        print("\n❌ Port-forwards failed to start")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_experiment_directory(experiment_id, **kwargs)
    
    # Get containers
    containers = experiment_config["containers"]
    
    # Pre-baseline
    baseline_pre = measure_pre_baseline(containers)
    baseline_pre_end = datetime.now(timezone.utc)
    
    with open(os.path.join(output_dir, "baseline_pre.json"), "w") as f:
        json.dump(baseline_pre, f, indent=2)
    
    # Execute workload
    if experiment_config["type"] == "isolated":
        action = kwargs.get("action")
        repetitions = kwargs.get("repetitions")
        start_time, end_time = execute_isolated_request(action, repetitions)
    else:
        users = experiment_config["users"]
        duration = experiment_config["duration"]
        start_time, end_time = execute_concurrent_workload(users, duration)
    
    # Combined cooldown + post-baseline
    baseline_post, cutoff_timestamp = execute_cooldown_and_post_baseline(baseline_pre, containers)
    
    with open(os.path.join(output_dir, "baseline_post.json"), "w") as f:
        json.dump(baseline_post, f, indent=2)
    
    print(f"\n{'─'*60}")
    print(f"MEASUREMENT WINDOW")
    print(f"{'─'*60}")
    print(f"  Start:  {baseline_pre_end.isoformat()} (end of pre-baseline)")
    print(f"  End:    {cutoff_timestamp.isoformat()} (post-baseline cutoff)")
    print(f"  Duration: {(cutoff_timestamp - baseline_pre_end).total_seconds():.1f}s")
    
    # Collect data using exact measurement window
    traces_file = collect_traces(baseline_pre_end, cutoff_timestamp, output_dir)
    metrics_file = collect_kepler_metrics(baseline_pre_end, cutoff_timestamp, output_dir,
                                         telemetry_config["step"], containers)

    # Collect telemetry overhead for E3, E5, E6 (enables cross-config comparison)
    if experiment_id in ("E3", "E5", "E6"):
        collect_telemetry_overhead(
            baseline_pre_end, cutoff_timestamp,
            output_dir, telemetry_config["step"]
        )
    
    # Save manifest (without baseline duplication)
    save_manifest(output_dir, experiment_config, start_time, end_time)
    
    print(f"\n{'='*60}")
    print(f"✅ EXPERIMENT {experiment_id} COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    
    # Cleanup
    cleanup_experiment()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

# Main function is defined after E1 sequential workflow


# ══════════════════════════════════════════════════════════════════════
# E1 SEQUENTIAL WORKFLOW (Optimized with shared baselines)
# ══════════════════════════════════════════════════════════════════════

def run_e1_sequential_workflow(repetitions: int) -> None:
    """
    Run E1 sequential workflow: browse → add_to_cart → view_cart → checkout
    
    Optimization: Post-baseline from action N becomes pre-baseline for action N+1
    Time savings: ~24 min → ~10 min
    """
    print(f"\n{'⚡'*30}")
    print(f"E1 SEQUENTIAL WORKFLOW (Optimized)")
    print(f"{'⚡'*30}")
    print(f"  Actions: browse → add_to_cart → view_cart → checkout")
    print(f"  Repetitions: {repetitions} per action")
    print(f"  Optimization: Sharing baselines")
    
    # Check telemetry config BEFORE redeploy
    print(f"\n{'='*60}")
    print(f"CHECKING TELEMETRY CONFIGURATION")
    print(f"{'='*60}")
    
    current_sampling = get_current_sampling_rate()
    current_scrape = get_current_scrape_interval()
    target_sampling = 100
    target_scrape = "1s"
    
    needs_telemetry_update = (current_sampling != target_sampling or current_scrape != target_scrape)
    
    if needs_telemetry_update:
        print(f"  Current: {current_sampling}% sampling, {current_scrape} scrape")
        print(f"  Target:  {target_sampling}% sampling, {target_scrape} scrape")
        print(f"  ⚠️  Will update during redeploy")
    else:
        print(f"  ✅ Telemetry already correct ({target_sampling}%, {target_scrape})")
    
    # Redeploy for clean state
    print(f"\n{'='*60}")
    print(f"REDEPLOYING OTEL-DEMO")
    print(f"{'='*60}")
    
    # Uninstall (check if running first)
    print(f"  [1/5] Checking if otel-demo is running...")
    check_result = subprocess.run(
        f"kubectl get pods -n {NAMESPACE} -l app.kubernetes.io/instance=otel-demo --no-headers 2>/dev/null | wc -l",
        shell=True, capture_output=True, text=True
    )
    
    pod_count = int(check_result.stdout.strip()) if check_result.stdout.strip().isdigit() else 0
    
    if pod_count > 0:
        print(f"      Found {pod_count} otel-demo pods, uninstalling...")
        subprocess.run(f"helm uninstall otel-demo -n {NAMESPACE}", shell=True, capture_output=True)
        print(f"      Waiting 30s for cleanup...")
        time.sleep(30)
        print(f"      ✅ Uninstalled")
    else:
        print(f"      ✅ No otel-demo pods running, skipping uninstall")
    
    # Update telemetry configs if needed
    if needs_telemetry_update:
        print(f"  [2/5] Updating telemetry configs...")
        
        # Update OTel sampling
        if current_sampling and current_sampling != target_sampling:
            cmd = f"sed -i 's/sampling_percentage: {current_sampling}/sampling_percentage: {target_sampling}/' ~/mt/otel-demo-values.yaml"
            subprocess.run(cmd, shell=True)
        
        # Update Prometheus scrape interval
        if current_scrape and current_scrape != target_scrape:
            cmd = f"sed -i 's/scrape_interval: {current_scrape}/scrape_interval: {target_scrape}/g' ~/mt/prometheus-simple-values.yaml"
            subprocess.run(cmd, shell=True)
        
        print(f"  ✅ Configs updated")
    else:
        print(f"  [2/5] Telemetry configs already correct, skipping update")
    
    # Install
    print(f"  [3/5] Installing fresh deployment...")
    result = subprocess.run(
        f"helm install otel-demo open-telemetry/opentelemetry-demo -n {NAMESPACE} -f ~/mt/otel-demo-values.yaml",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr}")
        sys.exit(1)
    
    # Scale down load-generator immediately
    print(f"  [4/5] Scaling down load-generator...")
    time.sleep(10)  # Brief wait for deployment to register
    rc, stdout, stderr = kubectl("scale deployment/load-generator --replicas=0", timeout=30)
    if rc == 0:
        print(f"  ✅ Load-generator scaled to 0")
    else:
        print(f"  ⚠️  Could not scale load-generator (might not exist yet)")
    
    # Warmup
    print(f"  [5/5] Warmup (3 minutes)...")
    time.sleep(180)
    print(f"  ✅ Deployment ready")
    
    # Start port-forwards (pods have new names after redeploy)
    if not port_manager.start_all():
        print("\n❌ Port-forwards failed after redeploy")
        sys.exit(1)
    
    # Get config
    experiment_config = EXPERIMENTS["E1"].copy()
    telemetry_config = TELEMETRY_CONFIGS["fine"]
    containers = experiment_config["containers"]
    
    # Clean cookie directory (with error handling)
    cookie_dir = "/tmp/experiment_cookies"
    try:
        if os.path.exists(cookie_dir):
            subprocess.run(f"rm -rf {cookie_dir}", shell=True, check=False)
        os.makedirs(cookie_dir, exist_ok=True)
        print(f"\n  🍪 Cookie directory ready: {cookie_dir}/")
    except Exception as e:
        print(f"\n  ⚠️  Cookie directory setup: {e}")
        print(f"     Continuing anyway...")
    
    # Actions to run
    actions = ["browse", "add_to_cart", "view_cart", "checkout"]
    shared_baseline = None  # Will hold post-baseline to reuse
    
    for i, action in enumerate(actions):
        is_first = (i == 0)
        is_last = (i == len(actions) - 1)
        
        print(f"\n{'='*60}")
        print(f"ACTION {i+1}/{len(actions)}: {action.upper()}")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = create_experiment_directory("E1", action=action, repetitions=repetitions)
        
        # Pre-baseline (first action measures, others reuse)
        if is_first:
            print(f"\n📊 Measuring pre-baseline...")
            baseline_pre = measure_pre_baseline(containers)
            baseline_pre_end = datetime.now(timezone.utc)
        else:
            print(f"\n♻️  Reusing post-baseline from previous action as pre-baseline")
            baseline_pre = shared_baseline
            baseline_pre_end = datetime.now(timezone.utc)
        
        with open(os.path.join(output_dir, "baseline_pre.json"), "w") as f:
            json.dump(baseline_pre, f, indent=2)
        
        # Execute workload
        start_time, end_time = execute_isolated_request(action, repetitions)
        
        # Post-baseline
        baseline_post, cutoff_timestamp = execute_cooldown_and_post_baseline(baseline_pre, containers)
        
        with open(os.path.join(output_dir, "baseline_post.json"), "w") as f:
            json.dump(baseline_post, f, indent=2)
        
        # Save post-baseline for next action (unless last)
        if not is_last:
            shared_baseline = baseline_post
        
        # Measurement window
        print(f"\n{'─'*60}")
        print(f"MEASUREMENT WINDOW")
        print(f"{'─'*60}")
        print(f"  Start:  {baseline_pre_end.isoformat()}")
        print(f"  End:    {cutoff_timestamp.isoformat()}")
        print(f"  Duration: {(cutoff_timestamp - baseline_pre_end).total_seconds():.1f}s")
        
        # Collect data
        collect_traces(baseline_pre_end, cutoff_timestamp, output_dir)
        collect_kepler_metrics(baseline_pre_end, cutoff_timestamp, output_dir,
                              telemetry_config["step"], containers)
        
        # Save manifest
        config_copy = experiment_config.copy()
        config_copy["action"] = action
        config_copy["repetitions"] = repetitions
        save_manifest(output_dir, config_copy, start_time, end_time)
        
        print(f"\n  ✅ '{action}' complete → {output_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ E1 SEQUENTIAL WORKFLOW COMPLETE")
    print(f"{'='*60}")
    print(f"  All 4 actions measured with {repetitions} repetitions")
    print(f"  Time saved: ~14 minutes (shared baselines)")
    print(f"  Session cookies preserved in {cookie_dir}/")
    
    # Cleanup
    cleanup_experiment()


# Update main() to support --sequential flag


def cleanup_experiment():
    """Cleanup after experiment: stop port-forwards and delete otel-demo."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CLEANUP")
    print(f"{'='*60}")
    
    # Stop port-forwards
    port_manager.stop_all()
    
    # Delete otel-demo deployment
    print(f"\n  Deleting otel-demo deployment...")
    subprocess.run(f"helm uninstall otel-demo -n {NAMESPACE}", shell=True, capture_output=True)
    
    # Wait for pods to be deleted (check every 5s)
    print(f"  ⏳ Waiting for pods to terminate...")
    max_wait = 60  # Max 60s
    waited = 0
    while waited < max_wait:
        result = subprocess.run(
            f"kubectl get pods -n {NAMESPACE} -l app.kubernetes.io/instance=otel-demo --no-headers 2>/dev/null | wc -l",
            shell=True, capture_output=True, text=True
        )
        pod_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        
        if pod_count == 0:
            print(f"  ✅ All pods deleted")
            break
        
        time.sleep(5)
        waited += 5
        if waited % 10 == 0:
            print(f"      {pod_count} pods still terminating... ({waited}s)", end="\r")
    
    if pod_count > 0:
        print(f"\n  ⚠️  {pod_count} pods still running after {max_wait}s")
        print(f"      Run: kubectl delete pods -n {NAMESPACE} -l app.kubernetes.io/instance=otel-demo --force")
    
    print(f"\n  💡 Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Cluster Experiment Orchestrator v2")
    parser.add_argument("--experiment", required=True, choices=["E1", "E2", "E3", "E4", "E5", "E6"])
    parser.add_argument("--action", choices=list(USER_ACTIONS.keys()))
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--sequential", action="store_true",
                       help="E1 only: Run browse→add_to_cart→view_cart→checkout with shared baselines")
    
    args = parser.parse_args()
    
    try:
        # Sequential workflow mode (E1 only)
        if args.sequential:
            if args.experiment != "E1":
                parser.error("--sequential only works with --experiment E1")
            if not args.repetitions:
                parser.error("--sequential requires --repetitions")
            
            run_e1_sequential_workflow(args.repetitions)
            return
        
        # Regular mode
        if args.experiment == "E1":
            if not args.action or not args.repetitions:
                parser.error("E1 requires --action and --repetitions (or use --sequential)")
        
        run_experiment(args.experiment, action=args.action, repetitions=args.repetitions)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cleanup_experiment()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_experiment()
        sys.exit(1)


if __name__ == "__main__":
    main()
