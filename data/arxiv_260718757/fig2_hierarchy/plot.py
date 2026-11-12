#!/usr/bin/env python
"""Draw Figure 2 from the saved CSV files; hollow markers are unconverged points.

    python data/arxiv_260718757/fig2_hierarchy/plot.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
STYLES = {
    "G0W0": {"color": "#000000", "ls": "-", "marker": "o"},
    "GW0": {"color": "#0072B2", "ls": "--", "marker": "s"},
    "scGW": {"color": "#D55E00", "ls": "-.", "marker": "D"},
}
LABELS = {"G0W0": r"$G_0W_0$", "GW0": r"$GW_0$", "scGW": r"sc$GW$"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_points(ax, x, y, converged, label, style):
    order = np.argsort(x)
    x, y, converged = np.asarray(x)[order], np.asarray(y)[order], np.asarray(converged)[order]
    ax.semilogx(x, y, label=label, markersize=5, **style)
    ax.semilogx(x[~converged], y[~converged], ls="none", marker=style["marker"],
               color=style["color"], markerfacecolor="white", markersize=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "reference")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = HERE / "output" / "reference" if args.data_dir.resolve() == (HERE / "reference").resolve() else HERE / "output"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.data_dir / "g0w0_gw0_scgw_curve.csv")
    z_rows = read_csv(args.data_dir / "z_scan.csv")
    n = np.array([float(row["n_cm3"]) for row in rows])
    lam = {method: np.array([float(row[method + "_lambda"]) for row in rows]) for method in STYLES}
    ok = {method: np.array([
        float(row[method + "_gap_converged"]) == 1
        and (method == "G0W0" or float(row[method + "_scf_converged"]) == 1)
        for row in rows
    ]) for method in STYLES}
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 14,
        "legend.frameon": False, "legend.fontsize": 10,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
    })
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(8.8, 3.0), constrained_layout=True)
    for method, style in STYLES.items():
        plot_points(axa, n, lam[method], ok[method], LABELS[method], style)
    axa.axhline(1.0, color="0.4", lw=1, ls="--")
    axa.set_ylabel(r"$\lambda$")
    temperature = float(rows[0].get("temperature_K", 1.0))
    axa.text(0.05, 0.06, "$T=%g$ K" % temperature, transform=axa.transAxes)
    axa.legend(loc="upper right")

    z = {}
    for method in ("GW0", "scGW"):
        selected = [row for row in z_rows if row["method"] == method]
        nz = np.array([float(row["n_cm3"]) for row in selected])
        zv = np.array([float(row["Z_kF"]) for row in selected])
        z_ok = np.array([float(row["scf_converged"]) == 1 for row in selected])
        z[method] = nz, zv, z_ok
        plot_points(axb, nz, zv, z_ok, LABELS[method], STYLES[method])
    axb.set_ylabel(r"$Z(k_{\rm F},\,{\rm i}\pi T)$")
    axb.legend()

    plot_points(axc, n, lam["G0W0"] / lam["GW0"], ok["G0W0"] & ok["GW0"],
                r"$\lambda_{G_0W_0}/\lambda_{GW_0}$", STYLES["G0W0"])
    nz, zv, z_ok = z["GW0"]
    plot_points(axc, nz, zv**2, z_ok, r"$Z^2$ ($GW_0$)", STYLES["GW0"])
    axc.set_ylabel("kernel suppression")
    axc.legend(loc="lower left")
    for ax, label in zip((axa, axb, axc), "ABC"):
        ax.set_xlabel(r"$n$ [cm$^{-3}$]")
        ax.text(-0.17, 1.02, label, transform=ax.transAxes, fontsize=17, fontweight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(args.out_dir / ("fig_hierarchy_3mode." + suffix), dpi=180)
    plt.close(fig)
    print("wrote", args.out_dir / "fig_hierarchy_3mode.pdf")


if __name__ == "__main__":
    main()
