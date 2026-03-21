#!/usr/bin/env python3
"""Downloads and prepares 1000 Genomes Phase 3 (Chr 22) for GWAS benchmarking."""

import subprocess
import sys
import urllib.request
from pathlib import Path

# Configuration
# DropBox links from cog-genomics for Phase 3
URLS = {
    "pgen.zst": "https://www.dropbox.com/s/w9wwua4pe9em280/chr22_phase3.pgen.zst?dl=1",
    "pvar.zst": "https://www.dropbox.com/s/3acsdd1sqlj2pa8/chr22_phase3_noannot.pvar.zst?dl=1",
    "psam": "https://www.dropbox.com/s/6ppo144ikdzery5/phase3_corrected.psam?dl=1",
}

DIR_DATA = Path("data")
PREFIX_FULL = DIR_DATA / "1kg_chr22_full"
PREFIX_TOY = DIR_DATA / "1kg_chr22_toy"


def check_plink2():
    try:
        subprocess.run(["plink2", "--version"], capture_output=True, check=True)
    except FileNotFoundError, subprocess.CalledProcessError:
        print("Error: plink2 could not be found. Please ensure it is installed and in your PATH.")
        sys.exit(1)


def download_file(url, dest):
    if not dest.exists():
        print(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"File {dest.name} already exists. Skipping download.")


def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(result.returncode)


def main():
    check_plink2()
    DIR_DATA.mkdir(exist_ok=True)

    # Download raw data
    for ext, url in URLS.items():
        dest = Path(f"{PREFIX_FULL}.{ext}")
        download_file(url, dest)

    # Convert to PLINK 1 binary (BED/BIM/FAM)
    bed_file = Path(f"{PREFIX_FULL}.bed")
    if not bed_file.exists():
        print("Converting to BED/BIM/FAM format for PLINK/Regenie step 1...")
        cmd_bed = [
            "plink2",
            "--pgen",
            f"{PREFIX_FULL}.pgen.zst",
            "--pvar",
            f"{PREFIX_FULL}.pvar.zst",
            "--psam",
            f"{PREFIX_FULL}.psam",
            "--make-bed",
            "--out",
            str(PREFIX_FULL),
        ]
        run_cmd(cmd_bed)

    # Convert to BGEN format for Regenie step 2
    bgen_file = Path(f"{PREFIX_FULL}.bgen")
    if not bgen_file.exists():
        print("Generating BGEN format for Regenie step 2...")
        cmd_bgen = ["plink2", "--bfile", str(PREFIX_FULL), "--export", "bgen-1.2", "bits=8", "--out", str(PREFIX_FULL)]
        run_cmd(cmd_bgen)

    # Create toy slice
    toy_bed = Path(f"{PREFIX_TOY}.bed")
    if not toy_bed.exists():
        print("Creating a toy slice (5000 variants)...")
        cmd_toy = [
            "plink2",
            "--bfile",
            str(PREFIX_FULL),
            "--thin-count",
            "5000",
            "--make-bed",
            "--out",
            str(PREFIX_TOY),
        ]
        run_cmd(cmd_toy)

    print("Data fetching and preparation complete.")


if __name__ == "__main__":
    main()
