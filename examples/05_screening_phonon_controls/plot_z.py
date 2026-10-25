#!/usr/bin/env python

"""Plot Example-05 Matsubara Z diagnostics from the saved CSV file.

This script is post-processing only.  It reads
``screening_phonon_controls_z.csv`` and redraws the Z comparison figure without
rerunning G0W0, GW0, scGW, sparse-IR generation, SCF, or gap solves.

Run from the repository root:

    python examples/05_screening_phonon_controls/plot_z.py
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

from examples import _common as common  # noqa: E402


Z_CSV = "screening_phonon_controls_z.csv"
Z_PLOT_STEM = "screening_phonon_controls_z"
REQUIRED_COLUMNS = (
    "method",
    "n_cm3",
    "Z_kF_i0",
)
PLOT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 8.0,
    "mathtext.fontset": "stix",
    "axes.labelsize": 8.0,
    "axes.titlesize": 9.2,
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


def build_parser():
    """Return the plotting-only argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(SCRIPT_DIR),
        help="Directory containing screening_phonon_controls_z.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for regenerated PNG/PDF files. Defaults to --data-dir.",
    )
    parser.add_argument(
        "--input",
        default=Z_CSV,
        help="Z CSV path, relative to --data-dir unless absolute.",
    )
    parser.add_argument(
        "--stem",
        default=Z_PLOT_STEM,
        help="Output filename stem for the PNG and PDF figures.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated subset of method names to plot.",
    )
    return parser


def resolve_path(base_dir, path):
    """Return ``path`` resolved against ``base_dir`` when relative."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def parse_methods(text):
    """Parse a comma-separated list of methods, preserving order."""
    if text is None:
        return None
    methods = [item.strip() for item in str(text).split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    return tuple(dict.fromkeys(methods))


def load_z_rows(path):
    """Load Z diagnostic rows and validate the columns needed for plotting."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or ())
        missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError("%s is missing columns: %s" % (path, ", ".join(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("%s does not contain any data rows." % path)
    return rows


def _row_method(row):
    return str(row["method"])


def _ordered_methods(rows, method_order=None, method_filter=None):
    """Return methods in the requested legend order."""
    present = list(dict.fromkeys(_row_method(row) for row in rows))
    if method_filter is not None:
        missing = [name for name in method_filter if name not in present]
        if missing:
            raise ValueError("Z CSV has no rows for methods: %s" % ", ".join(missing))
        present = [name for name in present if name in method_filter]
    ordered = []
    if method_order is not None:
        ordered.extend([name for name in method_order if name in present])
    ordered.extend([name for name in present if name not in ordered])
    return ordered


def _style_for(method, method_styles=None):
    """Return ``(color, marker, linestyle)`` for one method."""
    if method_styles is None:
        return None, "o", "-"
    style = method_styles.get(method)
    if style is None:
        return None, "o", "-"
    if isinstance(style, dict):
        return (
            style.get("color"),
            style.get("marker", "o"),
            style.get("ls", style.get("linestyle", "-")),
        )
    color, marker, linestyle = style
    return color, marker, linestyle


def _label_for(method, method_labels=None):
    """Return the legend label for one method."""
    if method_labels is None:
        return method
    return method_labels.get(method, method)


def _method_arrays(rows, method):
    """Return sorted finite density and Z arrays for one method."""
    points = [
        (float(row["n_cm3"]), float(row["Z_kF_i0"]))
        for row in rows
        if _row_method(row) == method
    ]
    if not points:
        return np.array([]), np.array([])
    arr = np.asarray(points, dtype=float)
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[finite]
    if arr.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1]


def plot_z_rows(
    rows,
    out_dir,
    *,
    method_order=None,
    method_filter=None,
    method_labels=None,
    method_styles=None,
    stem=Z_PLOT_STEM,
):
    """Plot ``Z_kF_i0`` versus carrier density for the requested methods."""
    rows = list(rows)
    if not rows:
        raise ValueError("no Z rows were provided.")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = common.pyplot(out_dir)
    methods = _ordered_methods(
        rows, method_order=method_order, method_filter=method_filter)
    if not methods:
        raise ValueError("no methods are available for plotting.")

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(3.35, 2.55))
        fig.patch.set_facecolor("white")
        all_z = []
        for method in methods:
            density, z_vals = _method_arrays(rows, method)
            if density.size == 0:
                continue
            all_z.extend(z_vals.tolist())
            color, marker, linestyle = _style_for(method, method_styles)
            ax.plot(
                density,
                z_vals,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=_label_for(method, method_labels),
            )
        if not all_z:
            raise ValueError("no finite Z values are available for plotting.")

        ax.set_xscale("log")
        ax.set_xlabel(r"carrier density $n$ [cm$^{-3}$]")
        ax.set_ylabel(r"$Z(k_{\mathrm{F}},\mathrm{i}0)$")
        ax.set_title("Example 05 Matsubara Z")
        ax.axhline(1.0, color="#8a8f98", linestyle=":", linewidth=0.75, zorder=0)
        ax.grid(True, which="major", color="#d0d5dd", linewidth=0.45, alpha=0.7)
        z_min = float(np.min(all_z))
        z_max = float(np.max(all_z))
        pad = 0.08 * max(z_max - z_min, abs(z_max), 1.0)
        ax.set_ylim(z_min - pad, z_max + pad)
        ax.legend(loc="best", ncol=1)
        fig.tight_layout(pad=0.25)
        png = out_dir / ("%s.png" % stem)
        pdf = out_dir / ("%s.pdf" % stem)
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
    return png, pdf


def main(argv=None):
    """Regenerate Example-05 Z figures from the saved Z CSV file."""
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    out_dir = data_dir if args.out_dir is None else resolve_path(data_dir, args.out_dir)
    rows = load_z_rows(resolve_path(data_dir, args.input))
    method_filter = parse_methods(args.methods)

    import run as example05

    method_order = list(example05.METHOD_LABELS)
    png, pdf = plot_z_rows(
        rows,
        out_dir,
        method_order=method_order,
        method_filter=method_filter,
        method_labels=example05.METHOD_LABELS,
        method_styles=example05.METHOD_STYLES,
        stem=args.stem,
    )
    print("Wrote %s" % png)
    print("Wrote %s" % pdf)


if __name__ == "__main__":
    main()
