#!/usr/bin/env python
"""Plot Fig. 4 from the reference paper CSVs, or from a new calculation.

    python plot.py
    python plot.py --data-dir output --out-dir output

This script needs only NumPy and Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


HERE = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "reference")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = HERE / "output" / "reference" if args.data_dir.resolve() == (HERE / "reference").resolve() else HERE / "output"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "font.size": 13.0, "mathtext.fontset": "stix", "axes.labelsize": 15.0,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6.5, "pdf.fonttype": 42,
    })
    rows = read_csv(args.data_dir / "g0w0_takada_curve.csv")
    n = np.array([float(row["n_cm3"]) for row in rows])
    order = np.argsort(n)
    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    methods = (
        ("G0W0_no_trunc_cellavg", "#000000", "-", "o", r"$G_0W_0$, full $(k,{\rm i}\omega_n)$"),
        ("G0W0_trunc", "#0072B2", "--", "s", r"$G_0W_0$, frequency truncated"),
    )
    for method, color, linestyle, marker, label in methods:
        lam = np.array([float(row[method + "_lambda"]) for row in rows])
        ax.semilogx(n[order], lam[order], color=color, ls=linestyle,
                    marker=marker, label=label)
        failed = np.array([int(row[method + "_gap_converged"]) != 1 for row in rows])
        if np.any(failed):
            ax.scatter(n[failed], lam[failed], marker="x", s=85,
                       color="crimson", zorder=5, label=method + " unconverged")
    projected = read_csv(args.data_dir / "takada_curve.csv")
    tn = np.array([float(row["n_cm3"]) for row in projected])
    tlam = np.array([float(row["lambda"]) for row in projected])
    order = np.argsort(tn)
    ax.semilogx(tn[order], tlam[order], color="#D55E00", ls="-.", marker="^",
                label="Takada projection")
    ax.axhline(1.0, color="0.4", lw=1.0, ls=(0, (5, 3)))
    ax.set_xlabel(r"$n$ [cm$^{-3}$]")
    ax.set_ylabel(r"$\lambda$")
    settings_file = args.data_dir / "settings.json"
    temperature = (json.loads(settings_file.read_text(encoding="utf-8"))["temperature"]
                   if settings_file.exists() else 1.0)
    ax.text(0.04, 0.92, r"$T=%g$ K" % temperature,
            transform=ax.transAxes, fontsize=12)
    ax.legend(loc="center left", fontsize=11)
    for suffix in ("pdf", "png"):
        path = args.out_dir / ("fig_projection." + suffix)
        fig.savefig(path, dpi=200)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
