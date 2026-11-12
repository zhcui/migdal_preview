#!/usr/bin/env python
"""Draw Fig. 3 from reference CSVs or a recomputed output directory.

    python data/arxiv_260718757/fig3_mechanism/plot.py
    python data/arxiv_260718757/fig3_mechanism/plot.py --data-dir data/arxiv_260718757/fig3_mechanism/output
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LO_MEV = (21.801250872312576, 55.718662629463765, 102.13755738509244)
CURVES = (
    ("G0W0", r"$G_0W_0$", "black", "o", "-"),
    ("scGW", r"sc$GW$", "#D55E00", "D", "-."),
    ("G0V", r"$G_0\mathcal{V}$ ($\Pi=0$)", "#009E73", "*", ":"),
    ("GV", r"$G\mathcal{V}$ ($\Pi=0$)", "#CC79A7", "D", "--"),
    ("G0W0_no_phonon", r"$G_0W_0$, no phonon", "0.45", "x", ":"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def converged(row, prefix=""):
    return (int(float(row[prefix + "gap_converged"])) == 1
            and int(float(row[prefix + "scf_converged"])) != 0)


def plot_controls(ax, data_dir):
    wide = read_csv(data_dir / "screening_phonon_controls.csv")
    for method, label, color, marker, style in CURVES:
        if method in ("G0V", "GV"):
            rows = read_csv(data_dir / ("g0v_scan.csv" if method == "G0V" else "gv_scan.csv"))
            prefix = ""
        else:
            rows, prefix = wide, method + "_"
        rows = sorted(rows, key=lambda row: float(row["n_cm3"]))
        x = np.array([float(row["n_cm3"]) for row in rows])
        y = np.array([float(row[prefix + "lambda"]) for row in rows])
        valid = np.array([converged(row, prefix) for row in rows])
        ax.semilogx(x, np.where(valid, y, np.nan), color=color, ls=style,
                    marker=marker, ms=5, label=label)
        if np.any(~valid):
            ax.semilogx(x[~valid], y[~valid], ls="none", marker="x", color=color, ms=7)
    ax.axhline(1, color="0.4", lw=1, ls="--")
    ax.set(xlabel=r"$n$ [cm$^{-3}$]", ylabel=r"$\lambda$")
    ax.legend(loc="upper right", fontsize=10)


def plot_gaps(ax, data_dir):
    rows = read_csv(data_dir / "gw0_scgw_gap_function.csv")
    scales = read_csv(data_dir / "gw0_scgw_qavg_w_scales.csv")
    rows = [row for row in rows if row["method"] == "scGW"]
    if not rows:
        raise ValueError("the gap panel requires scGW rows")
    densities = sorted({float(row["n_cm3"]) for row in rows})
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for i, density in enumerate(densities):
        selected = sorted([row for row in rows if float(row["n_cm3"]) == density],
                          key=lambda row: float(row["omega_meV"]))
        scale = next(row for row in scales if row["method"] == "scGW"
                     and np.isclose(float(row["n_cm3"]), density, rtol=1e-10, atol=0))
        color = colors[i % len(colors)]
        exponent = int(np.floor(np.log10(density)))
        coefficient = density / 10.0**exponent
        number = r"10^{%d}" % exponent if np.isclose(coefficient, 1) else r"%.3g\times10^{%d}" % (coefficient, exponent)
        label = r"$%s$ cm$^{-3}$" % number
        if not converged(scale):
            label += " (unconverged)"
        ax.semilogx([float(row["omega_meV"]) for row in selected],
                    [float(row["gap_kavg_norm"]) for row in selected],
                    color=color, lw=2, ls="-" if converged(scale) else "--", label=label)
        for key, style in (("E_F_meV", "-"), ("E_plasmon_meV", "--")):
            ax.axvline(float(scale[key]), color=color, ls=style, lw=1.8, ymax=0.18)
    for energy in LO_MEV:
        ax.axvline(energy, color="black", ls=":", lw=1.1, alpha=0.6)
    ax.axhline(0, color="0.4", lw=0.9)
    ax.text(LO_MEV[-1] * 1.35, -0.42, "LO poles", fontsize=11)
    ax.set(xlabel=r"$\omega_n$ [meV]", ylabel=r"$\langle\Phi(k,{\rm i}\omega_n)\rangle_k$ [norm.]",
           xlim=(0.2, 3e4), ylim=(-1.05, 1.1))
    ax.legend(loc="upper left", fontsize=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "reference")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = HERE / "output" / "reference" if args.data_dir.resolve() == (HERE / "reference").resolve() else HERE / "output"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "font.size": 12,
                         "mathtext.fontset": "stix", "axes.linewidth": 1,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    plot_controls(axes[0], args.data_dir)
    plot_gaps(axes[1], args.data_dir)
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.13, 1.03, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.out_dir / ("fig_mechanism." + suffix)
        fig.savefig(path, dpi=220)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
