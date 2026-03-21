#!/usr/bin/env python3
"""Runs PLINK2 and Regenie baselines, captures execution time and hardware specs.

Generates a JSON report.
"""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cpuinfo
import psutil

try:
    import GPUtil

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def check_dependencies():
    missing = []

    plink_bin = os.environ.get("PLINK2_BIN", "plink2")
    try:
        subprocess.run([plink_bin, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        missing.append(f"plink2 (checked via '{plink_bin}')")

    regenie_bin = os.environ.get("REGENIE_BIN", "regenie")
    try:
        subprocess.run([regenie_bin, "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        missing.append(f"regenie (checked via '{regenie_bin}')")

    if missing:
        print("Warning: The following required baseline tools are missing in PATH:")
        for tool in missing:
            print(f"  - {tool}")
        print("You can specify their paths using PLINK2_BIN and REGENIE_BIN environment variables.")
        print("Benchmarks will fail if these are not available.")

    return plink_bin, regenie_bin


def get_hardware_specs():
    specs = {
        "cpu_info": cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU"),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpus": [],
    }

    if HAS_GPU:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            specs["gpus"].append({"name": gpu.name, "memory_total_mb": gpu.memoryTotal})

    return specs


def run_command(name, cmd_list):
    print(f"\n--- Running {name} ---")
    cmd_str = " ".join(cmd_list)
    print(f"Command: {cmd_str}")

    start_time = time.time()
    try:
        subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Success: {name} completed in {duration:.2f} seconds.")
        return {"success": True, "duration_seconds": duration, "command": cmd_str}
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Error running {name}:")
        print(e.stderr)
        return {
            "success": False,
            "duration_seconds": duration,
            "error": e.stderr,
            "command": cmd_str,
        }


def main():
    plink_bin, regenie_bin = check_dependencies()

    data_dir = Path("data")
    base_out = data_dir / "baselines"
    base_out.mkdir(exist_ok=True)

    bfile = str(data_dir / "1kg_chr22_full")
    bgen_file = str(data_dir / "1kg_chr22_full.bgen")
    pheno_cont = str(data_dir / "pheno_cont.txt")
    pheno_bin = str(data_dir / "pheno_bin.txt")
    covar = str(data_dir / "covariates.txt")

    if not Path(f"{bfile}.bed").exists():
        print(f"Error: Missing {bfile}.bed. Please run fetch_1kg.py first.")
        sys.exit(1)

    if not Path(pheno_bin).exists():
        print(f"Error: Missing {pheno_bin}. Please run simulate_phenos.py first.")
        sys.exit(1)

    print("Gathering hardware specs...")
    hw_specs = get_hardware_specs()

    results = {}

    plink_cont_cmd = [
        plink_bin,
        "--bfile",
        bfile,
        "--pheno",
        pheno_cont,
        "--covar",
        covar,
        "--glm",
        "allow-no-covars",
        "--out",
        str(base_out / "plink_cont"),
    ]
    results["plink_cont"] = run_command("PLINK2 Continuous", plink_cont_cmd)

    plink_bin_cmd = [
        plink_bin,
        "--bfile",
        bfile,
        "--pheno",
        pheno_bin,
        "--covar",
        covar,
        "--glm",
        "firth-fallback",
        "allow-no-covars",
        "--out",
        str(base_out / "plink_bin"),
    ]
    results["plink_bin"] = run_command("PLINK2 Binary", plink_bin_cmd)

    regenie_step1_cmd = [
        regenie_bin,
        "--step",
        "1",
        "--bed",
        bfile,
        "--phenoFile",
        pheno_bin,
        "--covarFile",
        covar,
        "--bt",
        "--bsize",
        "1000",
        "--out",
        str(base_out / "regenie_step1"),
    ]
    results["regenie_step1"] = run_command("Regenie Step 1", regenie_step1_cmd)

    regenie_step2_cmd = [
        regenie_bin,
        "--step",
        "2",
        "--bgen",
        bgen_file,
        "--phenoFile",
        pheno_bin,
        "--covarFile",
        covar,
        "--bt",
        "--firth",
        "--approx",
        "--pred",
        str(base_out / "regenie_step1_pred.list"),
        "--out",
        str(base_out / "regenie_step2"),
    ]

    if results.get("regenie_step1", {}).get("success"):
        results["regenie_step2"] = run_command("Regenie Step 2", regenie_step2_cmd)
    else:
        print("\nSkipping Regenie Step 2 because Step 1 failed.")
        results["regenie_step2"] = {"success": False, "error": "Step 1 failed"}

    report = {"timestamp": datetime.now(UTC).isoformat(), "hardware": hw_specs, "results": results}

    report_path = data_dir / "benchmark_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=4)

    print(f"\nBenchmark complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()
