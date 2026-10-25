#!/usr/bin/env python

"""Tutorial: compare G0W0, GW0, and scGW density curves.

The example scans carrier density and compares the leading pairing eigenvalue
from three radial drivers:

* G0W0: fixed noninteracting Green's function and fixed W0,
* GW0: self-consistent normal-state G with fixed W0, and
* scGW: self-consistent normal-state G and W.

Run from the repository root, preferably under MPI:

    mpirun -n 2 python examples/04_g0w0_gw0_scgw_curve/run.py
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


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_STYLES = {
    "G0W0": ("#2f6f8f", "o"),
    "GW0": ("#5f8d4e", "s"),
    "scGW": ("#b65f5f", "^"),
}
METHOD_LABELS = {
    "G0W0": r"$G_0W_0$",
    "GW0": r"$GW_0$",
    "scGW": "scGW",
}
METHOD_NOTES = {
    "G0W0": "fixed G0 and fixed RPA W0",
    "GW0": "self-consistent normal-state G with fixed W0",
    "scGW": "self-consistent normal-state G and W",
}


def default_ir_file():
    """Return the preferred sparse-IR cache path for this example."""
    candidates = (
        SCRIPT_DIR / "ir.h5",
        SCRIPT_DIR / "ir.5",
    )
    for path in candidates:
        if path.exists():
            return path
    return SCRIPT_DIR / "ir.5"


def parse_args(argv=None):
    """Parse example-04 density-curve arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_density_args(parser, default_n=6)
    common.add_radial_args(
        parser,
        default_ir_file=default_ir_file(),
        ir_help=(
            "Sparse-IR h5 cache file. Defaults to this example's ir.h5 when "
            "it exists, otherwise this example's ir.5."
        ),
    )
    parser.add_argument("--diag-ncell", type=int, default=12)
    parser.add_argument(
        "--gw0-scf-max-cycle",
        type=int,
        default=300,
        help="GW0 normal-state SCF cycle limit.",
    )
    parser.add_argument("--gw0-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--gw0-continuation-max-dex",
        type=float,
        default=1.0,
        help=(
            "Maximum density step, in decades, for inserted GW0 continuation "
            "points when --restart-scf is enabled."
        ),
    )
    parser.add_argument("--scgw-nq", type=int, default=500)
    parser.add_argument("--scgw-scf-max-cycle", type=int, default=300)
    parser.add_argument("--scgw-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--scgw-scf-damp", type=float, default=0.98)
    parser.add_argument("--scgw-scf-diis-space", type=int, default=8)
    parser.add_argument("--scgw-scf-diis-start-cycle", type=int, default=0)
    parser.add_argument(
        "--scgw-continuation-max-dex",
        type=float,
        default=1.0,
        help=(
            "Maximum density step, in decades, for inserted scGW continuation "
            "points when --restart-scf is enabled."
        ),
    )
    parser.add_argument(
        "--restart-scf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Warm-start GW0/scGW density scans from the previous density point.",
    )
    parser.add_argument(
        "--scf-fs-pin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-scf-fs-pin to run the unpinned SCF gauge.",
    )
    parser.add_argument("--allow-no-pi-tail", action="store_true")
    parser.add_argument(
        "--fail-unconverged",
        action="store_true",
        help=(
            "Raise if GW0/scGW SCF does not converge; default writes flags "
            "and continues."
        ),
    )
    parser.add_argument("--scgw-n-tail", type=int, default=100)
    parser.add_argument("--pi-imag-tol", type=float, default=1.0e-5)
    parser.add_argument("--pi-imag-policy", choices=("warn", "fail"), default="warn")
    parser.add_argument(
        "--sigma-tail-policy",
        choices=("exchange_static", "exchange_t0", "zero"),
        default="exchange_static",
    )
    common.add_output_arg(parser, __file__)
    return parser.parse_args(argv)


def radial_kwargs(args):
    """Return shared radial kwargs, leaving driver defaults in place."""
    kwargs = common.radial_kwargs(args)
    kwargs["save_ir"] = True
    kwargs["verbose"] = 5
    return kwargs


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


def base_driver_kwargs(args, densities):
    """Return common driver settings for this no-truncation comparison."""
    kwargs = radial_kwargs(args)
    kwargs.update(
        densities=densities,
        diag_ncell=args.diag_ncell,
    )
    return kwargs


def gw0_kwargs(args):
    """Return GW0 SCF controls exposed by this tutorial."""
    kwargs = common.changed_driver_kwargs(args, "gw0")
    if not args.fail_unconverged:
        kwargs["allow_unconverged"] = True
    return kwargs


def scgw_kwargs(args):
    """Return scGW SCF and W-update controls exposed by this tutorial."""
    kwargs = common.changed_driver_kwargs(args, "scgw")
    if not args.fail_unconverged:
        kwargs["allow_unconverged"] = True
    return kwargs


def uses_scf_restart(restart_scf):
    """Return whether GW0/scGW density scans should warm-start SCF."""
    return bool(restart_scf)


def make_jobs(args, densities):
    """Construct the three radial drivers compared by the tutorial."""
    from migdal.g0w0 import G0W0
    from migdal.gw0 import GW0
    from migdal.scgw import ScGW

    base = base_driver_kwargs(args, densities)
    scgw_base = dict(base)
    scgw_base["nq"] = args.scgw_nq
    return [
        ("G0W0", G0W0(**base)),
        ("GW0", GW0(**base, **gw0_kwargs(args))),
        ("scGW", ScGW(**scgw_base, **scgw_kwargs(args))),
    ]


def scan_jobs(jobs, mpi, restart_scf=False):
    """Run each driver and return ``(method, rows)`` curves."""
    curves = []
    for idx, (name, drv) in enumerate(jobs, start=1):
        common.rank0_print(
            mpi.rank,
            "step %d/%d: %s - %s" % (
                idx,
                len(jobs),
                name,
                METHOD_NOTES[name],
            ),
        )
        if name in ("GW0", "scGW"):
            rows = drv.scan(
                restart=restart_scf,
                return_intermediate=False,
            )
        else:
            rows = drv.scan()
        curves.append((name, rows))
    return curves


def lambda_range(rows):
    """Return a compact min/max lambda string for one method."""
    vals = np.asarray([row.lam for row in rows], dtype=float)
    return "%.6g to %.6g" % (float(np.min(vals)), float(np.max(vals)))


def convergence_line(name, rows):
    """Summarize gap and SCF convergence for one method."""
    gap_ok = sum(int(row.gap_converged) for row in rows)
    scf_rows = [row for row in rows if row.scf_converged is not None]
    if not scf_rows:
        return "  %s: gap %d/%d, no normal-state SCF loop" % (
            name,
            gap_ok,
            len(rows),
        )
    scf_ok = sum(int(row.scf_converged) for row in scf_rows)
    cycles = [int(row.scf_cycles) for row in scf_rows]
    return "  %s: gap %d/%d, SCF %d/%d, cycles %d-%d" % (
        name,
        gap_ok,
        len(rows),
        scf_ok,
        len(scf_rows),
        min(cycles),
        max(cycles),
    )


def plot_density_curves(path, out_dir, curves, args):
    """Plot lambda curves and SCF-cycle diagnostics."""
    plt = common.pyplot(out_dir)
    fig, (ax_lam, ax_scf) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.15, 0.85)},
    )
    fig.patch.set_facecolor("white")

    for name, rows in curves:
        color, marker = METHOD_STYLES[name]
        arr = common.rows_to_array(rows)
        order = np.argsort(arr[:, 0])
        density = arr[order, 0]
        lam = arr[order, 1]
        ax_lam.plot(
            density,
            lam,
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=5,
            label=METHOD_LABELS[name],
        )
        scf_cycles = arr[order, 4]
        mask = scf_cycles >= 0
        if np.any(mask):
            ax_scf.plot(
                density[mask],
                scf_cycles[mask],
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=5,
                label=METHOD_LABELS[name],
            )

    ax_lam.set_xscale("log")
    ax_lam.set_ylabel(r"leading eigenvalue $\lambda$")
    ax_lam.set_title(r"Example 04: $G_0W_0$, $GW_0$, and scGW density curves")
    ax_lam.grid(True, which="both", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    ax_lam.legend(frameon=False, loc="best")

    ax_scf.set_xscale("log")
    ax_scf.set_xlabel(r"carrier density $n$ [cm$^{-3}$]")
    ax_scf.set_ylabel("SCF cycles")
    ax_scf.grid(True, which="both", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    fig.text(
        0.1,
        0.03,
        (
            "No truncation; diag_average='auto' selects diagonal cell averaging.\n"
            "q_interp_mode=%s; diag_average_mode=%s."
        )
        % (args.q_interp_mode, args.diag_average_mode),
        va="bottom",
        ha="left",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(path, dpi=200)


def tutorial_report(args, densities, curves):
    """Build the tutorial text printed to terminal and saved beside outputs."""
    lines = [
        "Example 04: G0W0/GW0/scGW density-curve tutorial",
        "",
        "Question:",
        "  How does the leading pairing eigenvalue lambda(n) change when the",
        "  normal-state Green's function and screened interaction are updated",
        "  self-consistently?",
        "",
        "Density scan:",
        "  %s" % density_summary(densities),
        "",
        "Methods:",
    ]
    for name, _ in curves:
        lines.append("  %s: %s." % (name, METHOD_NOTES[name]))
    lines.extend(
        [
            "",
            "Driver settings used:",
            "  All three drivers use no-truncation gap solves and",
            "  diag_average='auto', which selects diagonal cell averaging",
            "  for this no-truncation comparison.",
            "  q_interp_mode=%s, diag_average_mode=%s"
            % (args.q_interp_mode, args.diag_average_mode),
            "  sparse-IR file=%s" % display_path(args.ir_file),
            "",
            "SCF settings:",
            "  GW0 inserted-continuation max_dex=%s, restart=%s, max_cycle=%s."
            % (
                args.gw0_continuation_max_dex,
                "on" if uses_scf_restart(args.restart_scf) else "off",
                args.gw0_scf_max_cycle,
            ),
            "  scGW inserted-continuation max_dex=%s, restart=%s."
            % (
                args.scgw_continuation_max_dex,
                "on" if uses_scf_restart(args.restart_scf) else "off",
            ),
            "  scGW damp=%s, DIIS space=%d."
            % (
                args.scgw_scf_damp,
                args.scgw_scf_diis_space,
            ),
            "  scGW DIIS start=%d, max_cycle=%d."
            % (
                args.scgw_scf_diis_start_cycle,
                args.scgw_scf_max_cycle,
            ),
            "  Fermi-surface pin: %s; unconverged rows are %s."
            % (
                "on" if args.scf_fs_pin else "off",
                "rejected" if args.fail_unconverged else "recorded with flags",
            ),
            "",
            "Result snapshot:",
        ]
    )
    for name, rows in curves:
        lines.append("  %s lambda range: %s" % (name, lambda_range(rows)))
    lines.extend(["", "Convergence summary:"])
    for name, rows in curves:
        lines.append(convergence_line(name, rows))
    lines.extend(
        [
            "",
            "Files:",
            "  g0w0_gw0_scgw_curve.csv",
            "  g0w0_gw0_scgw_curve.png",
            "  g0w0_gw0_scgw_curve.txt",
        ]
    )
    return lines


def main(argv=None):
    """Run no-truncation G0W0, GW0, and scGW density curves."""
    args = parse_args(argv)
    from mpi4pyscf.lib import logger  # noqa: F401
    from mpi4pyscf.tools import mpi

    dens = common.density_points(args)
    out_dir = common.prepare_out_dir(args.out_dir)
    common.rank0_print(
        mpi.rank,
        "Example 04: %s" % density_summary(dens),
    )
    common.rank0_print(
        mpi.rank,
        "Using q_interp_mode=%s with tuned GW0/scGW SCF settings."
        % args.q_interp_mode,
    )

    curves = scan_jobs(make_jobs(args, dens), mpi, restart_scf=args.restart_scf)

    if mpi.rank == 0:
        common.save_curve(out_dir / "g0w0_gw0_scgw_curve.csv", curves)
        plot_density_curves(
            out_dir / "g0w0_gw0_scgw_curve.png",
            out_dir,
            curves,
            args,
        )
        report = tutorial_report(args, dens, curves)
        common.write_lines(out_dir / "g0w0_gw0_scgw_curve.txt", report)
        print(
            "\n".join(report + ["", "Output directory:", "  %s" % out_dir]),
            flush=True,
        )


if __name__ == "__main__":
    main()
