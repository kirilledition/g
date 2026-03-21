import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    data_dir = Path("data")

    fam_path = data_dir / "1kg_chr22_full.fam"

    if not fam_path.exists():
        print(f"Error: Could not find {fam_path}. Run fetch_1kg.py first.")
        sys.exit(1)

    print(f"Reading {fam_path}...")
    fam_df = pd.read_csv(fam_path, sep=r"\s+", header=None, names=["FID", "IID", "PID", "MID", "Sex", "Pheno"])

    n_samples = len(fam_df)
    print(f"Loaded {n_samples} samples.")
    print("Generating simulated phenotypes...")

    fam_df["pheno_cont"] = rng.standard_normal(n_samples)
    fam_df["pheno_bin"] = rng.binomial(n=1, p=0.3, size=n_samples)
    fam_df["Age"] = rng.normal(loc=50, scale=10, size=n_samples).astype(int)
    fam_df["Sex_cov"] = fam_df["Sex"].replace(0, rng.choice([1, 2]))

    cont_path = data_dir / "pheno_cont.txt"
    bin_path = data_dir / "pheno_bin.txt"
    cov_path = data_dir / "covariates.txt"

    fam_df[["FID", "IID", "pheno_cont"]].to_csv(cont_path, sep="\t", index=False)
    fam_df[["FID", "IID", "pheno_bin"]].to_csv(bin_path, sep="\t", index=False)
    fam_df[["FID", "IID", "Age", "Sex_cov"]].to_csv(cov_path, sep="\t", index=False)

    print(f"Saved {cont_path}")
    print(f"Saved {bin_path}")
    print(f"Saved {cov_path}")
    print("Phenotype simulation complete.")


if __name__ == "__main__":
    main()
