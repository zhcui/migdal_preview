#!/usr/bin/env python

"""Regenerate example-08 figures from saved CSV files.

This script does not run G0W0, GW0, scGW, sparse-IR generation, SCF, or gap
solves.  It only reads existing ``lambda_vs_temperature.csv`` and
``tc_vs_density.csv`` data, then redraws the two PNG figures.

Run from the repository root:

    python examples/08_tc_vs_density/plot.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tc_common as tc  # noqa: E402

REQUIRED_LAMBDA_COLUMNS = (
    "method",
    "n_cm3",
    "temperature_K",
    "lambda",
    "gap_converged",
    "scf_converged",
)
REQUIRED_TC_COLUMNS = (
    "method",
    "n_cm3",
    "tc_K",
    "tc_lower_bound_K",
    "tc_upper_bound_K",
    "status",
)
PLOT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 8.0,
    "mathtext.fontset": "stix",
    "mathtext.default": "it",
    "axes.labelsize": 8.0,
    "axes.titlesize": 9.2,
    "axes.titlepad": 7.0,
    "axes.labelpad": 2.0,
    "axes.linewidth": 0.65,
    "axes.formatter.use_mathtext": True,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 1.45,
    "ytick.minor.size": 1.45,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "lines.linewidth": 1.05,
    "lines.markersize": 3.4,
    "lines.markeredgewidth": 0.65,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.45,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
METHOD_ORDER = tc.METHODS
METHOD_LABELS = {
    "G0W0": r"$G_0W_0$",
    "GW0": r"$GW_0$",
    "scGW": r"$\mathrm{sc}GW$",
}
METHOD_STYLES = {
    "G0W0": {"color": "#000000", "ls": "-", "marker": "o"},
    "GW0": {"color": "#0072B2", "ls": "--", "marker": "s"},
    "scGW": {"color": "#D55E00", "ls": "-.", "marker": "D"},
}
STATUS_MARKERS = {
    "crossed": {"marker": None, "face": "color", "label": r"$T_{\mathrm{c}}$"},
    "suspect_crossed": {
        "marker": "D",
        "face": "none",
        "label": r"suspect $T_{\mathrm{c}}$",
    },
    "below_window": {
        "marker": None,
        "face": "none",
        "label": r"$T_{\mathrm{c}}<T_{\mathrm{min}}$",
    },
    "above_window": {
        "marker": None,
        "face": "none",
        "label": r"$T_{\mathrm{c}}>T_{\mathrm{max}}$",
    },
}


def build_parser():
    """Return the plotting-only argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(SCRIPT_DIR),
        help="Directory containing lambda_vs_temperature.csv and tc_vs_density.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for regenerated PNG files. Defaults to --data-dir.",
    )
    parser.add_argument(
        "--lambda-csv",
        default="lambda_vs_temperature.csv",
        help="Lambda CSV path, relative to --data-dir unless absolute.",
    )
    parser.add_argument(
        "--tc-csv",
        default="tc_vs_density.csv",
        help="Tc CSV path, relative to --data-dir unless absolute.",
    )
    parser.add_argument(
        "--rebuild-tc-csv",
        action="store_true",
        help=(
            "Rebuild tc_vs_density.csv from lambda_vs_temperature.csv before "
            "plotting. This is post-processing only and does not rerun GW/SCF."
        ),
    )
    parser.add_argument(
        "--tc-allow-unconverged",
        action="store_true",
        help=(
            "When rebuilding Tc rows, include unconverged gap or GW0/scGW SCF "
            "rows if lambda is finite and positive."
        ),
    )
    return parser


def resolve_path(base_dir, path):
    """Return ``path`` resolved against ``base_dir`` when relative."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def load_csv_rows(path, required_columns):
    """Load CSV rows and validate the columns needed by the plot helpers."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or ())
        missing = [col for col in required_columns if col not in fieldnames]
        if missing:
            raise ValueError("%s is missing columns: %s" % (path, ", ".join(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("%s does not contain any data rows." % path)
    return rows


def load_or_build_tc_rows(args, lambda_rows, tc_csv, out_dir):
    """Return Tc rows, optionally rebuilding them from saved lambda rows."""
    rebuild = args.rebuild_tc_csv or args.tc_allow_unconverged or not tc_csv.exists()
    if not rebuild:
        return load_csv_rows(tc_csv, REQUIRED_TC_COLUMNS), "loaded %s" % tc_csv

    tc_rows = tc.summarize_tc(
        lambda_rows,
        allow_unconverged=args.tc_allow_unconverged,
    )
    if args.rebuild_tc_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        rebuilt_csv = out_dir / Path(args.tc_csv).name
        tc.write_tc_csv(rebuilt_csv, tc_rows)
        return tc_rows, "rebuilt %s" % rebuilt_csv
    if not tc_csv.exists():
        return tc_rows, "built Tc rows in memory because %s is absent" % tc_csv
    return tc_rows, "rebuilt Tc rows in memory for --tc-allow-unconverged"


def density_key(row):
    """Return a stable density key for grouping CSV rows."""
    return "%.12e" % float(row["n_cm3"])


def method_sort_key(method):
    """Return the preferred method order used in legends and line drawing."""
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def method_style(method):
    """Return line style settings for one many-body method."""
    return METHOD_STYLES.get(
        method,
        {"color": "0.35", "ls": "-", "marker": "o"},
    )


def method_label(method):
    """Return a mathtext display label for one many-body method."""
    return METHOD_LABELS.get(method, method)


def finite_positive(value):
    """Return whether ``value`` can be shown on a log-scale axis."""
    return np.isfinite(value) and value > 0.0


def tc_plot_value(row):
    """Return the y-value used for one Tc row, including window bounds."""
    status = row.get("status", "")
    if status in ("crossed", "suspect_crossed"):
        value = tc.as_float(row, "tc_K")
    elif status == "below_window":
        value = tc.as_float(row, "tc_upper_bound_K")
    elif status == "above_window":
        value = tc.as_float(row, "tc_lower_bound_K")
    else:
        return np.nan
    return value if finite_positive(value) else np.nan


def polish_log_axes(ax):
    """Apply compact publication-style tick and spine settings."""
    ax.grid(True, which="major", color="0.84", lw=0.45)
    ax.grid(True, which="minor", color="0.92", lw=0.30, ls=":")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_tc_vs_density(plt, out_dir, tc_rows):
    """Write the Tc-vs-density figure with connected window-bound points."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.0, 3.45), constrained_layout=True)
    methods = sorted({row["method"] for row in tc_rows}, key=method_sort_key)

    for method in methods:
        style = method_style(method)
        rows = [row for row in tc_rows if row["method"] == method]
        points = [
            (float(row["n_cm3"]), tc_plot_value(row), row.get("status", ""))
            for row in rows
        ]
        points = [(n_cm3, y, status) for n_cm3, y, status in points if finite_positive(y)]
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        ax.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            color=style["color"],
            ls=style["ls"],
            lw=1.15,
            zorder=1,
        )
        for status in sorted({item[2] for item in points}):
            status_points = [item for item in points if item[2] == status]
            if not status_points:
                continue
            marker_info = STATUS_MARKERS.get(status, STATUS_MARKERS["crossed"])
            marker = marker_info["marker"] or style["marker"]
            face = (
                style["color"]
                if marker_info["face"] == "color"
                else marker_info["face"]
            )
            ax.scatter(
                [item[0] for item in status_points],
                [item[1] for item in status_points],
                marker=marker,
                s=24,
                facecolors=face,
                edgecolors=style["color"],
                linewidths=0.7,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$ [cm$^{-3}$]")
    ax.set_ylabel(r"$T_{\mathrm{c}}$ [K]")
    ax.set_title(r"$T_{\mathrm{c}}(n)$ from $\lambda(T)=1$")
    polish_log_axes(ax)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=method_style(method)["color"],
            ls=method_style(method)["ls"],
            marker=method_style(method)["marker"],
            ms=3.6,
            lw=1.15,
            label=method_label(method),
        )
        for method in methods
    ]
    handles = method_handles
    if handles:
        ax.legend(handles=handles, loc="best", ncol=1)
    fig.savefig(out_dir / "tc_vs_density.png", dpi=400)
    plt.close(fig)


def plot_lambda_vs_temperature(plt, out_dir, lambda_rows, allow_unconverged=False):
    """Write the lambda-vs-temperature figure with method-specific markers."""
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(5.6, 3.65), constrained_layout=True)
    densities = sorted({float(row["n_cm3"]) for row in lambda_rows}, reverse=True)
    methods = sorted({row["method"] for row in lambda_rows}, key=method_sort_key)
    cmap = plt.get_cmap("viridis")
    if len(densities) == 1:
        norm = LogNorm(
            vmin=densities[0] / np.sqrt(10.0),
            vmax=densities[0] * np.sqrt(10.0),
        )
    else:
        norm = LogNorm(vmin=min(densities), vmax=max(densities))
    density_colors = {
        "%.12e" % n_cm3: cmap(norm(n_cm3))
        for n_cm3 in densities
    }
    has_excluded = False

    for n_cm3 in densities:
        n_key = "%.12e" % n_cm3
        for method in methods:
            style = method_style(method)
            rows = [
                row
                for row in lambda_rows
                if row["method"] == method and density_key(row) == n_key
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: float(row["temperature_K"]))
            usable = [
                row
                for row in rows
                if tc.usable_for_tc(row, allow_unconverged=allow_unconverged)
            ]
            excluded = [
                row
                for row in rows
                if not tc.usable_for_tc(row, allow_unconverged=allow_unconverged)
            ]
            if usable:
                ax.plot(
                    [float(row["temperature_K"]) for row in usable],
                    [float(row["lambda"]) for row in usable],
                    marker=style["marker"],
                    ms=3.0,
                    lw=0.9,
                    ls=style["ls"],
                    color=density_colors[n_key],
                    markeredgewidth=0.6,
                )
            excluded_xy = [
                (float(row["temperature_K"]), tc.as_float(row, "lambda"))
                for row in excluded
                if finite_positive(tc.as_float(row, "lambda"))
            ]
            if excluded_xy:
                ax.scatter(
                    [temp for temp, _ in excluded_xy],
                    [lam for _, lam in excluded_xy],
                    marker="x",
                    s=20,
                    color=density_colors[n_key],
                    alpha=0.9,
                    linewidths=0.7,
                )
                has_excluded = True

    ax.axhline(1.0, lw=0.65, color="0.35", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\lambda(T)$")
    ax.set_title(r"Cooling scan of $\lambda(T)$")
    polish_log_axes(ax)

    method_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            ls=method_style(method)["ls"],
            marker=method_style(method)["marker"],
            ms=3.4,
            lw=1.0,
            label=method_label(method),
        )
        for method in methods
    ]
    if has_excluded:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                ls="",
                marker="x",
                ms=4.0,
                label="excluded",
            )
        )
    ax.legend(
        handles=method_handles,
        loc="lower left",
        framealpha=0.9,
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.015,
        fraction=0.052,
    )
    colorbar.set_label(r"$n$ [cm$^{-3}$]")
    colorbar.ax.tick_params(which="both", direction="in")
    fig.savefig(out_dir / "lambda_vs_temperature.png", dpi=400)
    plt.close(fig)


def plot_outputs(out_dir, lambda_rows, tc_rows, allow_unconverged=False):
    """Write example-08 figures from saved lambda and Tc rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = tc.common.pyplot(out_dir)
    with plt.rc_context(PLOT_RC):
        plot_tc_vs_density(plt, out_dir, tc_rows)
        plot_lambda_vs_temperature(
            plt,
            out_dir,
            lambda_rows,
            allow_unconverged=allow_unconverged,
        )


def main(argv=None):
    """Regenerate the example-08 figures from saved CSV data."""
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir
    lambda_csv = resolve_path(data_dir, args.lambda_csv)
    tc_csv = resolve_path(data_dir, args.tc_csv)

    lambda_rows = load_csv_rows(lambda_csv, REQUIRED_LAMBDA_COLUMNS)
    tc_rows, tc_status = load_or_build_tc_rows(args, lambda_rows, tc_csv, out_dir)

    plot_outputs(
        out_dir,
        lambda_rows,
        tc_rows,
        allow_unconverged=args.tc_allow_unconverged,
    )

    print("loaded %s" % lambda_csv, flush=True)
    print(tc_status, flush=True)
    print("wrote %s" % (out_dir / "lambda_vs_temperature.png"), flush=True)
    print("wrote %s" % (out_dir / "tc_vs_density.png"), flush=True)


if __name__ == "__main__":
    main()
