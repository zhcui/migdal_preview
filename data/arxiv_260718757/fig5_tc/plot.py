#!/usr/bin/env python
"""Draw Fig. 5 from the reference CSVs, or pass --data-dir output to use a run."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from tc import read_csv


HERE = Path(__file__).resolve().parent
STYLES = {
    "G0W0": dict(color="black", ls="-", marker="o"),
    "GW0": dict(color="#0072B2", ls="--", marker="s"),
    "scGW": dict(color="#D55E00", ls="-.", marker="D"),
}
LABELS = {"G0W0": r"$G_0W_0$", "GW0": r"$GW_0$", "scGW": r"sc$GW$"}


def plot_value(row):
    status = row["status"]
    if status in ("crossed", "suspect_crossed"):
        return float(row["tc_K"])
    if int(row["excluded_points"]):
        return np.nan
    if status == "below_window":
        return float(row["tc_upper_bound_K"])
    if status == "above_window":
        return float(row["tc_lower_bound_K"])
    return np.nan


def plot_experiment(ax, directory):
    rows = read_csv(directory / "lin14_fig1b_digitized.csv")
    for series, face in (("reduced", "0.42"), ("nb_doped", "white")):
        points = sorted((float(r["n_cm3"]), float(r["tc_K"]))
                        for r in rows if r["series"] == series)
        ax.loglog(*np.array(points).T, ls="none", marker="s", color="0.4",
                  mfc=face, mec="0.4", ms=4.1, alpha=0.58, zorder=3.2)
    rows = read_csv(directory / "lin14_fig1c_digitized.csv")
    points = sorted((float(r["x_per_fu"]) / (3.905e-8)**3, float(r["tc_K"]))
                    for r in rows if r["series"] == "schooley")
    ax.loglog(*np.array(points).T, ls="none", marker="o", color="0.55",
              mfc="0.8", mec="0.55", ms=4, alpha=0.58, zorder=3.1)
    rows = read_csv(directory / "tomioka22_nb_ba_fig4c.csv")
    points = sorted((float(r["n_cm3"]), float(r["tc_mid_K"]))
                    for r in rows if float(r["tc_mid_K"]) > 0)
    ax.loglog(*np.array(points).T, ls="none", marker="x", color="0.4",
              ms=4.4, mew=1.15, alpha=0.65, zorder=3.6)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=HERE / "reference")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--exp-dir", type=Path, default=HERE / "reference" / "experiment")
    p.add_argument("--no-experiment", action="store_true")
    args = p.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = HERE / "output" / "reference" if args.data_dir.resolve() == (HERE / "reference").resolve() else HERE / "output"
    data = args.data_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "font.size": 12, "mathtext.fontset": "stix", "axes.labelsize": 15,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.9, 3.55), constrained_layout=True)
    rows = read_csv(data / "tc_vs_density.csv")
    handles = []
    for method, style in STYLES.items():
        points = sorted((float(r["n_cm3"]), plot_value(r), r["status"])
                        for r in rows if r["method"] == method and np.isfinite(plot_value(r)))
        if not points:
            continue
        ax.loglog([r[0] for r in points], [r[1] for r in points],
                  color=style["color"], ls=style["ls"], lw=2)
        for density, temperature, status in points:
            ax.loglog(density, temperature, ls="none", marker=style["marker"],
                      color=style["color"], ms=6.5,
                      mfc=style["color"] if status == "crossed" else "none")
        handles.append(Line2D([], [], **style, label=LABELS[method]))
    z_rows = read_csv(data / "efstar_line.csv")
    if z_rows:
        z_points = sorted((float(r["n_cm3"]), float(r["Z_kF_i0"])) for r in z_rows)
        n = np.geomspace(1e13, 1e23, 200)
        z = np.interp(np.log(n), np.log([r[0] for r in z_points]), [r[1] for r in z_points])
        kf = (3 * np.pi**2 * n * (0.52917721067e-8)**3)**(1 / 3)
        efstar = kf**2 / (2 * 2.1) * 315775.02480407 / z
        ax.loglog(n, efstar, color="0.55", lw=1.4)
        ax.fill_between(n, efstar, 300, color="0.93", zorder=0)
        ax.text(8e17, 6.6, r"$T=E_{\rm F}^{\ast}$", fontsize=11, color="0.4", rotation=52)
        ax.text(2.5e13, 4.2, "$T>E_{\\rm F}^{\\ast}$:\nstiffness\nlimited",
                fontsize=10.5, color="0.45", va="top")
    if not args.no_experiment:
        plot_experiment(ax, args.exp_dir)
        handles.append(Line2D([], [], color="0.45", ls="none", marker="s",
                              mfc="0.72", label=r"exp. $T_{\rm c}$"))
    ax.set(xlim=(1e13, 2e23), ylim=(0.04, 300),
           xlabel=r"$n$ [cm$^{-3}$]", ylabel="temperature [K]")
    ax.legend(handles=handles, loc="upper left", fontsize=10.2, handlelength=1.55)
    for suffix in ("pdf", "png"):
        path = args.out_dir / ("fig_tc." + suffix)
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
