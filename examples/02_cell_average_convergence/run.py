#!/usr/bin/env python

"""Tutorial: why diagonal cell averaging helps radial convergence.

The example uses no-truncation G0W0 at one density and compares lambda on a
sequence of radial k grids, with and without diagonal cell averages.

Run from the repository root, preferably under MPI:

    mpirun -n 2 python examples/02_cell_average_convergence/run.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402


Q_INTERP_MODE = "legacy"
K_GRID_PAD = 4
DIAG_CASES = (
    ("point_diag", 0, "point diagonal", "o", "#2f6f8f"),
    ("cell_average", 1, "cell average", "s", "#5f8d4e"),
)
CSV_FIELDS = (
    "nside",
    "treatment",
    "diag_average",
    "nk",
    "lambda",
    "rel_error_to_finest_cellavg",
    "gap_converged",
    "matvecs",
)


def parse_args(argv=None):
    """Parse diagonal-cell convergence example arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density",
        type=float,
        default=1.0e20,
        help="Carrier density in cm^-3 for the one-point convergence study.",
    )
    parser.add_argument(
        "--ir-file",
        default=str(common.legacy_ir_file()),
        help="Sparse-IR h5 file. Uses the repository cache by default.",
    )
    parser.add_argument(
        "--resolutions",
        default="50,100,200",
        help=(
            "Comma-separated k-left/right values. k-shell-min is matched to "
            "each value."
        ),
    )
    common.add_output_arg(parser, __file__)
    return parser.parse_args(argv)


def parse_resolutions(text):
    """Return the requested radial k-grid side sizes."""
    nsides = [int(x) for x in str(text).split(",") if x.strip()]
    if len(nsides) < 2:
        raise ValueError("--resolutions must contain at least two values.")
    if any(n <= 0 for n in nsides):
        raise ValueError("--resolutions values must be positive integers.")
    return nsides


def display_path(path):
    """Return a readable path without machine-specific prefixes."""
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, Path(__file__).resolve().parents[2])


def driver_kwargs(args, nside, diag_average):
    """Return only the non-default G0W0 settings needed for one run."""
    return {
        "densities": [args.density],
        "ir_file": args.ir_file,
        "diag_average": int(diag_average),
        "q_interp_mode": Q_INTERP_MODE,
        "kgrid_n_pad": K_GRID_PAD,
        "kgrid_n_left": nside,
        "kgrid_n_right": nside,
        "kgrid_n_shell_min": nside,
        "kgrid_n_shell_max": max(4 * nside, nside + 8),
        "verbose": 0,
    }


def make_driver(G0W0, args, nside, diag_average):
    """Build the no-truncation G0W0 driver for one grid and diagonal rule."""
    return G0W0(**driver_kwargs(args, nside, diag_average))


def run_convergence_scan(args, nsides, mpi):
    """Run point-diagonal and cell-average cases on each radial grid."""
    from migdal.g0w0 import G0W0

    records = []
    total = len(nsides) * len(DIAG_CASES)
    step = 0
    for nside in nsides:
        for key, diag_average, label, _, _ in DIAG_CASES:
            step += 1
            common.rank0_print(
                mpi.rank,
                "step %d/%d: %s, nside=%d" % (step, total, label, nside),
            )
            row = make_driver(G0W0, args, nside, diag_average).kernel()
            records.append(
                {
                    "nside": nside,
                    "treatment": key,
                    "diag_average": diag_average,
                    "row": row,
                }
            )
    return records


def reference_record(records):
    """Return the finest-grid cell-average record used as the reference."""
    cellavg = [r for r in records if r["diag_average"]]
    return max(cellavg, key=lambda r: r["nside"])


def table_rows(records):
    """Convert driver results to CSV/report rows with relative errors."""
    ref_lam = reference_record(records)["row"].lam
    rows = []
    for rec in records:
        row = rec["row"]
        rel = abs(row.lam - ref_lam) / max(abs(ref_lam), 1.0e-300)
        rows.append(
            {
                "nside": rec["nside"],
                "treatment": rec["treatment"],
                "diag_average": rec["diag_average"],
                "nk": row.nk,
                "lambda": row.lam,
                "rel_error_to_finest_cellavg": rel,
                "gap_converged": int(row.gap_converged),
                "matvecs": row.gap.matvecs,
            }
        )
    return rows


def write_csv(path, rows):
    """Write the convergence table with tutorial-friendly treatment labels."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(path, out_dir, rows):
    """Plot lambda and relative error for both diagonal treatments."""
    plt = common.pyplot(out_dir)
    fig, (ax_lam, ax_err) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.05, 1.0)},
    )
    fig.patch.set_facecolor("white")

    for key, _, label, marker, color in DIAG_CASES:
        sub = sorted(
            (r for r in rows if r["treatment"] == key),
            key=lambda r: r["nk"],
        )
        nk = np.asarray([r["nk"] for r in sub], dtype=float)
        lam = np.asarray([r["lambda"] for r in sub], dtype=float)
        rel = np.asarray(
            [r["rel_error_to_finest_cellavg"] for r in sub],
            dtype=float,
        )
        ax_lam.plot(
            nk,
            lam,
            marker=marker,
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax_err.plot(nk, rel, marker=marker, color=color, linewidth=1.8)

    ax_lam.set_ylabel(r"leading eigenvalue $\lambda$")
    ax_lam.set_title(
        r"Example 02: diagonal cell averaging improves $G_0W_0$ convergence"
    )
    ax_lam.grid(True, which="both", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    ax_lam.legend(frameon=False, loc="best")

    ax_err.set_xscale("log")
    ax_err.set_xlabel(r"number of radial $k$ points, $N_k$")
    ax_err.set_ylabel(r"relative error to $\lambda_{\mathrm{ref}}$")
    ax_err.grid(True, which="both", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    fig.text(
        0.1,
        0.025,
        (
            r"Reference: finest cell-average run; q_interp_mode='legacy' "
            "keeps point-diagonal runs finite."
        ),
        va="bottom",
        ha="left",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.savefig(path, dpi=200)


def format_rel(value):
    """Return a compact relative-error string."""
    return "%.3e" % float(value)


def tutorial_report(args, nsides, rows):
    """Build the text tutorial printed to the terminal and saved beside outputs."""
    ref = [r for r in rows if r["diag_average"] and r["nside"] == max(nsides)][0]
    point_ref_grid = [
        r for r in rows if not r["diag_average"] and r["nside"] == max(nsides)
    ][0]
    coarse_point = [
        r for r in rows if not r["diag_average"] and r["nside"] == min(nsides)
    ][0]
    coarse_cell = [
        r for r in rows if r["diag_average"] and r["nside"] == min(nsides)
    ][0]
    improvement = (
        coarse_point["rel_error_to_finest_cellavg"]
        / max(coarse_cell["rel_error_to_finest_cellavg"], 1.0e-300)
    )

    lines = [
        "Example 02: diagonal cell-average convergence tutorial",
        "",
        "Question:",
        "  How much radial-grid resolution is saved by replacing the singular",
        "  same-cell diagonal block with a finite cell average?",
        "",
        "Experiment:",
        "  Run no-truncation G0W0 at one density for several radial k grids.",
        "  For each grid, compare two diagonal treatments:",
        "    point_diag: sample the diagonal W(k, k) value directly.",
        "    cell_average: replace the same-cell block by its finite cell integral.",
        "",
        "Why this example uses q_interp_mode='legacy':",
        "  The direct W interpolation keeps the point-diagonal branch finite.",
        "  The q_singular mode represents the Coulomb logarithm exactly and",
        "  therefore requires diagonal cell averages.",
        "",
        "Run settings:",
        "  density=%.6e cm^-3, q_interp_mode=%s, k-pad=%d"
        % (args.density, Q_INTERP_MODE, K_GRID_PAD),
        "  resolutions=%s" % ", ".join(str(n) for n in nsides),
        "  sparse-IR file=%s" % display_path(args.ir_file),
        "",
        "Result snapshot:",
        "  reference lambda=%.6g from cell_average at nside=%d"
        % (ref["lambda"], ref["nside"]),
        "  coarsest-grid error: point_diag=%s, cell_average=%s"
        % (
            format_rel(coarse_point["rel_error_to_finest_cellavg"]),
            format_rel(coarse_cell["rel_error_to_finest_cellavg"]),
        ),
        "  cell averaging reduces the coarsest-grid error by about %.1fx"
        % improvement,
        "  point_diag error at the finest grid is still %s"
        % format_rel(point_ref_grid["rel_error_to_finest_cellavg"]),
        "",
        "Per-grid errors:",
    ]
    for nside in nsides:
        point = [
            r for r in rows if r["nside"] == nside and not r["diag_average"]
        ][0]
        cell = [r for r in rows if r["nside"] == nside and r["diag_average"]][0]
        lines.append(
            "  nside=%d, nk=%d: point_diag=%s, cell_average=%s"
            % (
                nside,
                point["nk"],
                format_rel(point["rel_error_to_finest_cellavg"]),
                format_rel(cell["rel_error_to_finest_cellavg"]),
            )
        )
    lines.extend(
        [
            "",
            "Files:",
            "  cell_average_convergence.csv",
            "  cell_average_convergence.png",
            "  cell_average_convergence.txt",
        ]
    )
    return lines


def main(argv=None):
    """Run the no-truncation G0W0 cell-average convergence scan."""
    args = parse_args(argv)
    from mpi4pyscf.lib import logger  # noqa: F401
    from mpi4pyscf.tools import mpi

    out_dir = common.prepare_out_dir(args.out_dir)
    nsides = parse_resolutions(args.resolutions)
    common.rank0_print(
        mpi.rank,
        "Example 02: density=%.3e cm^-3, resolutions=%s"
        % (args.density, ",".join(str(n) for n in nsides)),
    )
    common.rank0_print(
        mpi.rank,
        "Using q_interp_mode='legacy' to compare point diagonal vs cell average.",
    )

    records = run_convergence_scan(args, nsides, mpi)

    if mpi.rank == 0:
        rows = table_rows(records)
        write_csv(out_dir / "cell_average_convergence.csv", rows)
        plot_convergence(out_dir / "cell_average_convergence.png", out_dir, rows)
        report = tutorial_report(args, nsides, rows)
        common.write_lines(out_dir / "cell_average_convergence.txt", report)
        print(
            "\n".join(report + ["", "Output directory:", "  %s" % out_dir]),
            flush=True,
        )


if __name__ == "__main__":
    main()
