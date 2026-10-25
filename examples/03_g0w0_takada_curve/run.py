#!/usr/bin/env python

"""Tutorial: compare three G0W0-style pairing estimates.

Run from the repository root, preferably under MPI:

    mpirun -n 2 python examples/03_g0w0_takada_curve/run.py

The example scans carrier density and compares the leading pairing eigenvalue
lambda from three closely related approximations:

* truncated sparse-IR G0W0,
* no-truncation sparse-IR G0W0 with the driver's automatic diagonal cell
  average, and
* Takada's projected dense kernel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402


G0W0_METHODS = (
    (
        "G0W0_trunc",
        "sparse-IR G0W0 with Matsubara-frequency truncation",
        {"truncate": True},
        r"$G_0W_0$, truncated",
        "o",
        "#2f6f8f",
    ),
    (
        "G0W0_no_trunc_cellavg",
        (
            "no-truncation sparse-IR G0W0 with automatic diagonal treatment"
        ),
        {},
        r"$G_0W_0$, no truncation + cell average",
        "s",
        "#5f8d4e",
    ),
)


def parse_args(argv=None):
    """Parse density-curve tutorial arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_density_args(parser, default_n=6)
    common.add_radial_args(parser)
    common.add_output_arg(parser, __file__)
    return parser.parse_args(argv)


def density_summary(densities):
    """Return one readable line describing the density grid."""
    densities = np.asarray(densities, dtype=float)
    return "%d points from %.3e down to %.3e cm^-3" % (
        len(densities),
        float(np.max(densities)),
        float(np.min(densities)),
    )


def display_path(path):
    """Return a readable path without machine-specific prefixes."""
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, Path(__file__).resolve().parents[2])


def build_g0w0_jobs(G0W0, densities, radial_kwargs):
    """Create the two sparse-IR G0W0 curves used by the tutorial."""
    jobs = []
    for name, note, overrides, _, _, _ in G0W0_METHODS:
        driver_kwargs = dict(radial_kwargs)
        driver_kwargs.update(overrides)
        jobs.append(
            (
                name,
                note,
                G0W0(densities=densities, **driver_kwargs),
            )
        )
    return jobs


def g0w0_treatment_lines(args):
    """Describe the actual sparse-IR W and diagonal treatment in this run."""
    if args.q_interp_mode == "q_singular":
        return [
            "  The sparse-IR G0W0 curves use q_interp_mode='q_singular' and",
            "  diag_average='auto'. q_singular requires diagonal cell averages,",
            "  so the truncated and no-truncation curves share that treatment.",
        ]
    return [
        "  The sparse-IR G0W0 curves use q_interp_mode='legacy' and",
        "  diag_average='auto'. The truncated curve stays point-diagonal,",
        "  while the no-truncation curve uses diagonal cell averaging.",
    ]


def g0w0_caption_line(args):
    """Return a one-line plot caption matching the actual W setting."""
    if args.q_interp_mode == "q_singular":
        return "The G0W0 curves use q_singular W and auto diagonal cell averages."
    return "The G0W0 curves use q_interp_mode='legacy' and auto diagonal handling."


def takada_rows_to_array(rows):
    """Return the compact table written for the projected Takada solver."""
    return np.asarray(
        [(r.n_cm3, r.lam, r.eps_F_eV, r.k_F_A, r.symmetry_error) for r in rows],
        dtype=float,
    )


def plot_density_curves(path, out_dir, curves, takada_rows, args):
    """Plot lambda(n) with tutorial-style labels and math formatting."""
    plt = common.pyplot(out_dir)
    takada_arr = takada_rows_to_array(takada_rows)

    def plot_array(arr, label, marker, color):
        order = np.argsort(arr[:, 0])
        ax.plot(
            arr[order, 0],
            arr[order, 1],
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=5,
            label=label,
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.24)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    ax.axhline(0.0, color="#667085", linewidth=0.9, alpha=0.7)

    for (_, _, _, label, marker, color), (_, rows) in zip(
        G0W0_METHODS, curves
    ):
        plot_array(common.rows_to_array(rows), label, marker, color)
    plot_array(takada_arr, "Takada projection", "^", "#b65f5f")

    ax.set_xscale("log")
    ax.set_xlabel(r"carrier density $n$ [cm$^{-3}$]")
    ax.set_ylabel(r"leading pairing eigenvalue $\lambda$")
    ax.set_title(r"Example 03: density dependence of $\lambda$")
    ax.grid(True, which="major", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#e4e7ec", linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False, loc="best")
    fig.text(
        0.11,
        0.055,
        "\n".join(
            [
                r"Note: $\lambda$ is the leading gap-kernel eigenvalue.",
                g0w0_caption_line(args),
                "Convergence flags are stored in the CSV.",
            ]
        ),
        va="bottom",
        ha="left",
        fontsize=9,
    )
    fig.savefig(path, dpi=200)


def lambda_range(rows):
    """Return a compact min/max lambda string for a result table."""
    vals = np.asarray([r.lam for r in rows], dtype=float)
    return "%.6g to %.6g" % (float(np.min(vals)), float(np.max(vals)))


def tutorial_report(args, densities, curves, takada_rows):
    """Build the text report printed to the terminal and saved beside outputs."""
    lines = [
        "Example 03: G0W0/Takada density-curve tutorial",
        "",
        "Question:",
        "  How does the leading pairing eigenvalue lambda(n) change when the",
        "  gap kernel is truncated, kept untruncated, or projected following Takada?",
        "",
        "Density scan:",
        "  %s" % density_summary(densities),
        "",
        "Methods:",
    ]
    for name, note, _, _, _, _ in G0W0_METHODS:
        lines.append("  %s: %s." % (name, note))
    lines.extend(
        [
            "  Takada_projection: dense projected solver in x = xi / eps_F.",
            "",
            "Driver settings used:",
        ]
        + g0w0_treatment_lines(args)
        + [
            "  TakadaProjection uses its built-in grid defaults:",
            "  n_q=20, n_theta=12, n_neg=48, n_pos=52, x_max=1e3.",
            "",
            "Run settings:",
            "  radial nq=%d, k-left=%d, k-right=%d, k-shell=[%d, %d]"
            % (
                args.nq,
                args.k_left,
                args.k_right,
                args.k_shell_min,
                args.k_shell_max,
            ),
            "  q_interp_mode=%s, diag_average_mode=%s"
            % (args.q_interp_mode, args.diag_average_mode),
            "  sparse-IR lambda=%.6g, ir_file=%s"
            % (args.lamb, display_path(args.ir_file)),
            "",
            "Result snapshot:",
        ]
    )
    for name, rows in curves:
        lines.append("  %s lambda range: %s" % (name, lambda_range(rows)))
    lines.append("  Takada_projection lambda range: %s" % lambda_range(takada_rows))
    lines.extend(
        [
            "",
            "Files:",
            "  g0w0_takada_curve.csv",
            "  takada_curve.csv",
            "  g0w0_takada_curve.png",
            "  g0w0_takada_curve.txt",
        ]
    )
    return lines


def main(argv=None):
    """Run the G0W0/Takada density-curve tutorial."""
    args = parse_args(argv)
    from mpi4pyscf.lib import logger  # noqa: F401
    from mpi4pyscf.tools import mpi

    from migdal.g0w0 import G0W0
    from migdal.takada import TakadaProjection

    dens = common.density_points(args)
    out_dir = common.prepare_out_dir(args.out_dir)
    kwargs = common.radial_kwargs(args)
    common.rank0_print(mpi.rank, "Example 03: %s" % density_summary(dens))
    common.rank0_print(
        mpi.rank,
        "Using %s W interpolation for G0W0 and built-in defaults for Takada."
        % args.q_interp_mode,
    )

    curves = []
    jobs = build_g0w0_jobs(G0W0, dens, kwargs)
    n_jobs = len(jobs) + 1
    for idx, (name, note, drv) in enumerate(jobs, start=1):
        common.rank0_print(
            mpi.rank,
            "step %d/%d: %s - %s" % (idx, n_jobs, name, note),
        )
        curves.append((name, drv.scan()))

    common.rank0_print(
        mpi.rank,
        "step %d/%d: Takada_projection - projected dense kernel"
        % (n_jobs, n_jobs),
    )
    takada = TakadaProjection(densities=dens)
    takada_rows = takada.scan()

    if mpi.rank == 0:
        common.save_curve(out_dir / "g0w0_takada_curve.csv", curves)
        np.savetxt(
            out_dir / "takada_curve.csv",
            takada_rows_to_array(takada_rows),
            header="n_cm3 lambda eps_F_eV k_F_A^-1 symmetry_error",
        )
        plot_density_curves(
            out_dir / "g0w0_takada_curve.png",
            out_dir,
            curves,
            takada_rows,
            args,
        )
        report = tutorial_report(args, dens, curves, takada_rows)
        common.write_lines(out_dir / "g0w0_takada_curve.txt", report)
        print(
            "\n".join(report + ["", "Output directory:", "  %s" % out_dir]),
            flush=True,
        )


if __name__ == "__main__":
    main()
