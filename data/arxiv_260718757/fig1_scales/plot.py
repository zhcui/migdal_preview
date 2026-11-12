#!/usr/bin/env python
"""Draw Figure 1 from the saved CSV files.

    python data/arxiv_260718757/fig1_scales/plot.py
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
TO_MEV = np.array([1.2385747182565898, 22.388405634920773, 73.24152018061872])
LO_MEV = np.array([21.801250872312576, 55.718662629463765, 102.13755738509244])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "reference")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = HERE / "output" / "reference" if args.data_dir.resolve() == (HERE / "reference").resolve() else HERE / "output"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n, e_F, omega_pl, e_TF = np.loadtxt(args.data_dir / "scales.csv", delimiter=",", skiprows=1, ndmin=2).T
    nu, factor = np.loadtxt(args.data_dir / "interaction_factor.csv", delimiter=",", skiprows=1, ndmin=2).T
    n_star = np.interp(LO_MEV[-1], e_F, n)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 14,
        "legend.frameon": False, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
    })
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    axa.axvspan(n[0], n_star, color="0.85", alpha=0.45, lw=0)
    axa.loglog(n, e_F, color="#0072B2", label=r"$E_{\rm F}$")
    axa.loglog(n, omega_pl, color="#D55E00", ls="--", label=r"$\omega_{\rm pl}$")
    axa.loglog(n, e_TF, color="#009E73", ls="-.", label=r"$E_{\rm TF}$")
    axa.axhline(LO_MEV[-1], color="k", ls=":", lw=1.3)
    axa.axhline(TO_MEV[0], color="0.55", ls=":", lw=1.1)
    axa.axvline(n_star, color="0.55", lw=1.2)
    axa.text(1.5e14, LO_MEV[-1] * 1.35, r"$\omega_{\rm LO,max}$")
    axa.text(1.5e14, TO_MEV[0] * 1.35, r"$\omega_{\rm TO,min}$", color="0.4")
    axa.annotate(
        r"$E_{\rm F}=\omega_{\rm LO,max}$", xy=(n_star, 2.0e-3),
        xytext=(2.0e18, 5.0e-4), fontsize=11, color="0.35",
        arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 1.0},
    )
    axa.set(xlabel=r"$n$ [cm$^{-3}$]", ylabel="energy [meV]",
            xlim=(1.0e14, 1.0e23), ylim=(1.0e-4, 1.0e4))
    axa.legend(loc="lower right")

    axb.loglog(nu, factor, color="#0072B2", lw=2.1)
    axb.axhline(1.0, color="0.45", lw=1)
    axb.axhline(6.163085088 / 23000.0, color="0.45", ls=":", lw=1)
    for lo in LO_MEV:
        axb.axvline(lo, color="0.2", ls="--", lw=1, alpha=0.65)
    arrow = {"arrowstyle": "->", "color": "0.35", "lw": 1.0}
    axb.annotate(
        r"$\mathcal{V}(q,0)=\frac{4\pi}{\epsilon_0 q^2}$",
        xy=(0.02, np.interp(0.02, nu, factor)), xytext=(0.05, 1.7e-3),
        fontsize=11, arrowprops=arrow,
    )
    axb.annotate(
        r"$\mathcal{V}(q,i\infty)=\frac{4\pi}{\epsilon_\infty q^2}$",
        xy=(2200, np.interp(2200, nu, factor)), xytext=(180, 0.21),
        fontsize=11, arrowprops=arrow,
    )
    axb.text(0.69, 0.38, "LO phonons", transform=axb.transAxes, ha="center", fontsize=11)
    axb.set(xlabel=r"$\nu_n$ [meV]", ylabel=r"$F_3(\nu_n)$",
            xlim=(1.0e-2, 1.0e4), ylim=(1.0e-4, 1.4))
    for ax, label in zip((axa, axb), "AB"):
        ax.text(-0.17, 1.02, label, transform=ax.transAxes, fontsize=17, fontweight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(args.out_dir / ("fig_scales_3mode." + suffix), dpi=180)
    plt.close(fig)
    print("wrote", args.out_dir / "fig_scales_3mode.pdf")


if __name__ == "__main__":
    main()
