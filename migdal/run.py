"""Command-line entry point for Migdal radial drivers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from migdal import log as mlog, memory, mpi_wrap


def build_parser():
    """Build the command-line parser for all Migdal driver methods."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="method", required=True)
    for name in ("g0w0", "gw0", "scgw", "takada"):
        p = sub.add_parser(name)
        p.add_argument("--densities", default="default")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--restart", type=int, choices=(0, 1),
                       default=0)
        if name != "takada":
            p.add_argument(
                "--verbose",
                type=int,
                default=4,
                help="PySCF-style verbosity level: 3 summary, 4 sections, 5 cycles.",
            )
            p.add_argument("--lamb", type=float, default=1.0e9)
            p.add_argument(
                "--ir-file",
                help=(
                    "Sparse-IR HDF5 cache path. Use this to reuse or create "
                    "a cache instead of rebuilding IR in memory."
                ),
            )
            p.add_argument(
                "--save-ir",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Write a generated sparse-IR cache; rank 0 writes under "
                    "MPI. Use --no-save-ir to disable."
                ),
            )
            p.add_argument("--nq", type=int, default=500)
            p.add_argument(
                "--n-2kf-min",
                type=int,
                default=101,
                help="Minimum q-grid points in the optional 2*kF refinement window.",
            )
            p.add_argument("--truncate", type=int, choices=(0, 1), default=0)
            p.add_argument("--diag-average", default="auto")
            p.add_argument("--diag-ncell", type=int, default=12)
            p.add_argument(
                "--diag-average-mode",
                choices=("auto", "gauss", "analytic"),
                default="auto",
                help=(
                    "Diagonal cell averaging rule; auto uses analytic for "
                    "q_singular and gauss otherwise."
                ),
            )
            p.add_argument("--kgrid-n-pad", type=int, default=5)
            p.add_argument("--kgrid-n-left", type=int, default=200)
            p.add_argument("--kgrid-n-right", type=int, default=200)
            if name != "scgw":
                p.add_argument(
                    "--kgrid-n-tail",
                    type=int,
                    default=0,
                    help=(
                        "Must remain 0 for G0W0/GW0 gap grids; scGW uses "
                        "--n-tail for the separate Pi integration tail."
                    ),
                )
            p.add_argument("--kgrid-n-shell-min", type=int, default=50)
            p.add_argument("--kgrid-n-shell-max", type=int, default=5000)
            p.add_argument("--kmax-factor", type=float, default=4.0)
            p.add_argument("--kmax-lo-factor", type=float, default=2.0)
            p.add_argument("--qmax-factor", type=float, default=2.0)
            p.add_argument("--wcut-factor", type=float, default=1.0)
            p.add_argument("--wcut-min-ev", type=float, default=0.0)
            p.add_argument(
                "--denom-policy",
                choices=("legacy_clip", "sign_regularize", "fail", "none"),
                default="sign_regularize",
            )
            p.add_argument("--denom-min", type=float, default=1.0e-10)
            p.add_argument(
                "--q-interp-mode",
                choices=("legacy", "q_singular"),
                default="q_singular",
                help=(
                    "Radial W primitive mode. Defaults to q_singular, which "
                    "handles Coulomb tails and needs diagonal cell averages."
                ),
            )
            p.add_argument(
                "--q-singular-head-policy",
                choices=("pi_head", "fit_w"),
                default="pi_head",
                help=(
                    "Source for q_singular screened Coulomb head; fit_w "
                    "fits q^2 W directly from sampled low-q rows."
                ),
            )
            if name in ("g0w0", "gw0"):
                p.add_argument(
                    "--pi0-mode",
                    choices=("finite_T", "0K", "GG"),
                    default="finite_T",
                    help=(
                        "Polarization used to build fixed W0. finite_T uses "
                        "the finite-temperature Lindhard formula, 0K uses "
                        "the zero-temperature Lindhard formula, and GG "
                        "builds the bubble from G0(tau)G0(beta-tau)."
                    ),
                )
            p.add_argument("--n0-check-rtol", type=float, default=1.0e-4)
            p.add_argument("--n0-check-atol", type=float, default=1.0e-12)
            p.add_argument("--davidson-tol", type=float, default=1.0e-6)
            p.add_argument("--davidson-max-cycle", type=int, default=300)
            p.add_argument("--davidson-max-space", type=int, default=20)
            p.add_argument("--nroots", type=int, default=1)
            p.add_argument("--gap-precond", choices=("even_diag", "ones"), default="even_diag")
            p.add_argument("--gap-cache", choices=("auto", "off"), default="auto")
            p.add_argument(
                "--max-memory",
                type=float,
                default=memory.DEFAULT_MAX_MEMORY,
                help="PySCF-style per-rank memory limit in MB.",
            )
        if name in ("gw0", "scgw"):
            p.add_argument("--scf-conv-tol", type=float, default=1.0e-6)
            p.add_argument("--scf-max-cycle", type=int, default=300)
            p.add_argument("--scf-diis-space", type=int, default=8)
            p.add_argument("--scf-damp", type=float, default=0.98)
            p.add_argument("--scf-diis-start-cycle", type=int, default=0)
            p.add_argument("--scf-continuation-max-dex", type=float, default=1.0)
            p.add_argument(
                "--scf-mu-bracket",
                choices=("nearest_positive", "extreme"),
                default="nearest_positive",
            )
            p.add_argument(
                "--scf-fs-pin",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Pin the normal-state self-energy at the Fermi surface. "
                    "Use --no-scf-fs-pin to recover the unpinned gauge."
                ),
            )
            p.add_argument("--allow-unconverged", action="store_true")
        if name == "scgw":
            p.add_argument(
                "--n-tail",
                type=int,
                default=100,
                help=(
                    "Minimum number of passive k points for the separate "
                    "scGW Pi integration tail."
                ),
            )
            p.add_argument(
                "--sigma-tail-policy",
                choices=("exchange_static", "exchange_t0", "zero"),
                default="exchange_static",
            )
            p.add_argument("--static-tail-xcut", type=float, default=60.0)
            p.add_argument("--pi-imag-tol", type=float, default=1.0e-5)
            p.add_argument("--pi-imag-policy", choices=("warn", "fail"), default="warn")
            p.add_argument("--pi-positive-tol", type=float, default=1.0e-12)
            p.add_argument("--pi-positive-policy", choices=("fail", "warn"), default="warn")
            p.add_argument(
                "--pi-positive-action",
                choices=("keep", "zero_small"),
                default="keep",
            )
            p.add_argument(
                "--pi-low-q-policy",
                choices=("asymptotic", "none"),
                default="asymptotic",
            )
            p.add_argument("--q-low-factor", type=float, default=2.0e-2)
            p.add_argument("--q-static-low-factor", type=float)
            p.add_argument("--allow-no-pi-tail", action="store_true")
            p.add_argument(
                "--legacy-scf-density",
                action="store_true",
                help=(
                    "Use the direct -2 G(beta) endpoint density in scGW mu "
                    "searches instead of the high-frequency corrected density."
                ),
            )
        if name == "takada":
            p.add_argument("--n-q", type=int, default=20)
            p.add_argument("--n-theta", type=int, default=12)
            p.add_argument("--x-core", type=float, default=1.0e-5)
            p.add_argument("--x-max", type=float, default=1.0e3)
            p.add_argument("--n-neg", type=int, default=48)
            p.add_argument("--n-pos", type=int, default=52)
    return parser


def _gap_kwargs(args):
    return {
        "precond": args.gap_precond,
        "cache": args.gap_cache,
        "davidson_tol": args.davidson_tol,
        "davidson_max_cycle": args.davidson_max_cycle,
        "davidson_max_space": args.davidson_max_space,
        "nroots": args.nroots,
    }


def main():
    """Parse CLI arguments, run the selected density scan, and write output."""
    args = build_parser().parse_args()
    mpi_wrap.get_mpi(import_logger=True)

    from migdal.g0w0 import G0W0
    from migdal.gw0 import GW0
    from migdal.scgw import ScGW
    from migdal.takada import TakadaProjection

    if args.method == "takada":
        drv = TakadaProjection(
            densities=args.densities,
            temperature=args.temperature,
            n_q=args.n_q,
            n_theta=args.n_theta,
            x_core=args.x_core,
            x_max=args.x_max,
            n_neg=args.n_neg,
            n_pos=args.n_pos,
        )
    else:
        common = dict(
            densities=args.densities,
            temperature=args.temperature,
            verbose=args.verbose,
            lamb=args.lamb,
            ir_file=args.ir_file,
            save_ir=args.save_ir,
            nq=args.nq,
            n_2kf_min=args.n_2kf_min,
            truncate=bool(args.truncate),
            diag_average=args.diag_average,
            diag_ncell=args.diag_ncell,
            diag_average_mode=args.diag_average_mode,
            kgrid_n_pad=args.kgrid_n_pad,
            kgrid_n_left=args.kgrid_n_left,
            kgrid_n_right=args.kgrid_n_right,
            kgrid_n_tail=getattr(args, "kgrid_n_tail", 0),
            kgrid_n_shell_min=args.kgrid_n_shell_min,
            kgrid_n_shell_max=args.kgrid_n_shell_max,
            kmax_factor=args.kmax_factor,
            kmax_lo_factor=args.kmax_lo_factor,
            qmax_factor=args.qmax_factor,
            wcut_factor=args.wcut_factor,
            wcut_min_ev=args.wcut_min_ev,
            n0_check_rtol=args.n0_check_rtol,
            n0_check_atol=args.n0_check_atol,
            denom_policy=args.denom_policy,
            denom_min=args.denom_min,
            q_interp_mode=args.q_interp_mode,
            q_singular_head_policy=args.q_singular_head_policy,
            max_memory=args.max_memory,
            **_gap_kwargs(args),
        )
        if args.method in ("g0w0", "gw0"):
            common["pi0_mode"] = args.pi0_mode
        if args.method == "g0w0":
            drv = G0W0(**common)
        elif args.method == "gw0":
            drv = GW0(
                **common,
                scf_conv_tol=args.scf_conv_tol,
                scf_max_cycle=args.scf_max_cycle,
                scf_damp=args.scf_damp,
                scf_diis_space=args.scf_diis_space,
                scf_diis_start_cycle=args.scf_diis_start_cycle,
                scf_mu_bracket=args.scf_mu_bracket,
                scf_fs_pin=args.scf_fs_pin,
                scf_continuation_max_dex=args.scf_continuation_max_dex,
                allow_unconverged=args.allow_unconverged,
            )
        elif args.method == "scgw":
            drv = ScGW(
                **common,
                n_tail=args.n_tail,
                sigma_tail_policy=args.sigma_tail_policy,
                static_tail_xcut=args.static_tail_xcut,
                pi_imag_tol=args.pi_imag_tol,
                pi_imag_policy=args.pi_imag_policy,
                pi_positive_tol=args.pi_positive_tol,
                pi_positive_policy=args.pi_positive_policy,
                pi_positive_action=args.pi_positive_action,
                pi_low_q_policy=args.pi_low_q_policy,
                q_low_factor=args.q_low_factor,
                q_static_low_factor=args.q_static_low_factor,
                allow_no_pi_tail=args.allow_no_pi_tail,
                scf_conv_tol=args.scf_conv_tol,
                scf_max_cycle=args.scf_max_cycle,
                scf_damp=args.scf_damp,
                scf_diis_space=args.scf_diis_space,
                scf_diis_start_cycle=args.scf_diis_start_cycle,
                scf_continuation_max_dex=args.scf_continuation_max_dex,
                scf_mu_bracket=args.scf_mu_bracket,
                scf_fs_pin=args.scf_fs_pin,
                scf_high_freq_density=not args.legacy_scf_density,
                allow_unconverged=args.allow_unconverged,
            )
        else:
            raise AssertionError(args.method)
    drv.scan(restart=bool(args.restart))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    drv.write(args.output)


def _main_with_error_box():
    try:
        main()
    except Exception as err:
        mlog.error(
            mlog.new_logger(verbose=mlog.QUIET, stdout=sys.stderr),
            str(err) or err.__class__.__name__,
            footer="Execution stopped.",
        )
        sys.exit(1)


if __name__ == "__main__":
    _main_with_error_box()
