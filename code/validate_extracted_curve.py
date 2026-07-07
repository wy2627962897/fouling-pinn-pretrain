"""
Validate and preview a digitized fouling curve CSV.

Usage:
    python validate_extracted_curve.py ../data/real/curves/source_001_fig5_rf_exp.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_extracted_curve.py <curve_csv>")
        return 2

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if list(df.columns[:2]) != ["time", "value"]:
        print("Expected first two columns: time,value")
        print(f"Actual columns: {list(df.columns)}")
        return 1

    df = df[["time", "value"]].dropna().sort_values("time")
    if df.empty:
        print("No valid rows after dropping missing values.")
        return 1

    monotonic_time = df["time"].is_monotonic_increasing
    negative_steps = (df["value"].diff() < 0).sum()

    print(f"File: {csv_path}")
    print(f"Points: {len(df)}")
    print(f"Time range: {df['time'].min():.6g} to {df['time'].max():.6g}")
    print(f"Value range: {df['value'].min():.6g} to {df['value'].max():.6g}")
    print(f"Time monotonic: {monotonic_time}")
    print(f"Negative value steps: {negative_steps}")

    out_dir = csv_path.parent / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_preview.png"

    plt.figure(figsize=(6, 4))
    plt.plot(df["time"], df["value"], "o-", markersize=3)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(csv_path.stem)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Preview saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

