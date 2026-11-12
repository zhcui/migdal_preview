#!/usr/bin/env python
"""Recompute Fig. 4: full/truncated G0W0 and the Takada projection.

    python run.py
    python plot.py --data-dir output

Defaults are the paper grids at T=1 K. For a quick calculation, use one
density and reduce both the radial and projected grids; see --help.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from migdal import ir_cache, model
from migdal.mpi_wrap import mpi
from migdal.g0w0 import G0W0
from migdal.takada import TakadaProjection


HERE = Path(__file__).resolve().parent
PARAMS = model.STOParams(
    m_eff=2.1,
    eps_inf=6.163085088,
    eps_0=23000.0,
    phonon_to_poles_cm=(9.98977881246349, 180.5746693840041, 590.7326992132146),
    phonon_lo_zeros_cm=(175.83894684689574, 449.4013216886174, 823.7949569635838),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output")
    parser.add_argument("--densities", help="Comma-separated densities in cm^-3.")
    parser.add_argument("--n-density", type=int, default=11,
                        help="Log-spaced points from 1e23 down to 1e16 cm^-3.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=1.0e9)
    parser.add_argument("--ir-file", type=Path, default=None,
                        help="Use an IR cache at the requested temperature and Lambda.")
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--n-2kf-min", type=int, default=101)
    parser.add_argument("--k-pad", type=int, default=5)
    parser.add_argument("--k-left", type=int, default=200)
    parser.add_argument("--k-right", type=int, default=200)
    parser.add_argument("--k-shell-min", type=int, default=50)
    parser.add_argument("--k-shell-max", type=int, default=5000)
    parser.add_argument("--pi0-mode", choices=("0K", "finite_T", "GG"), default="finite_T")
    parser.add_argument("--davidson-tol", type=float, default=1.0e-6)
    parser.add_argument("--davidson-max-cycle", type=int, default=300)
    parser.add_argument("--max-memory", type=float, default=102400.0,
                        help="Memory budget in MB per rank, following PySCF.")
    parser.add_argument("--takada-nq", type=int, default=20)
    parser.add_argument("--takada-n-theta", type=int, default=12)
    parser.add_argument("--takada-n-neg", type=int, default=48)
    parser.add_argument("--takada-n-pos", type=int, default=52)
    return parser.parse_args(argv)


def save_g0w0_curves(path, curves):
    """Write lambda curves in the reference CSV format."""
    columns = [np.array([row.n_cm3 for row in curves[0][1]])]
    names = ["n_cm3"]
    formats = ["%.18e"]
    for method, rows in curves:
        values = np.array([
            (row.lam, int(row.gap_converged),
             -1 if row.scf_converged is None else int(row.scf_converged),
             -1 if row.scf_cycles is None else row.scf_cycles, row.nk, row.nq)
            for row in rows
        ])
        columns.extend(values.T)
        names.extend(method + "_" + key for key in (
            "lambda", "gap_converged", "scf_converged", "scf_cycles", "nk", "nq"
        ))
        formats.extend(["%.18e", "%d", "%d", "%d", "%d", "%d"])
    np.savetxt(path, np.column_stack(columns), delimiter=",", comments="",
               header=",".join(names), fmt=formats)


def main(argv=None):
    args = parse_args(argv)
    if args.n_density < 1:
        raise ValueError("--n-density must be positive.")
    densities = (model.density_grid(args.densities) if args.densities
                 else np.geomspace(1.0e23, 1.0e16, args.n_density))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.ir_file is None:
        args.ir_file = out_dir / "cache" / ("ir_L%.12g_T%.12gK.h5" % (args.lamb, args.temperature))
        if mpi.rank == 0 and args.lamb == 1e9:
            ir_cache.write_scaled_sparse_ir_h5(
                HERE.parent / "ir.h5", args.ir_file,
                PARAMS.beta_ha(args.temperature), overwrite=True)

    radial_settings = dict(
        params=PARAMS, densities=densities, temperature=args.temperature,
        lamb=args.lamb, ir_file=args.ir_file, save_ir=True,
        nq=args.nq, n_2kf_min=args.n_2kf_min,
        kgrid_n_pad=args.k_pad, kgrid_n_left=args.k_left,
        kgrid_n_right=args.k_right, kgrid_n_shell_min=args.k_shell_min,
        kgrid_n_shell_max=args.k_shell_max,
        kmax_factor=4.0, kmax_lo_factor=2.0, qmax_factor=2.0,
        pi0_mode=args.pi0_mode, q_interp_mode="q_singular",
        q_singular_head_policy="pi_head", diag_average="auto",
        diag_average_mode="auto", diag_ncell=12,
        davidson_tol=args.davidson_tol, davidson_max_cycle=args.davidson_max_cycle,
        davidson_max_space=20, precond="even_diag", cache="auto",
        n0_check_atol=1.0e-12, max_memory=args.max_memory, verbose=4,
    )
    curves = []
    for method, truncate in (("G0W0_trunc", True), ("G0W0_no_trunc_cellavg", False)):
        driver = G0W0(**radial_settings, truncate=truncate)
        curves.append((method, driver.scan()))
    save_g0w0_curves(out_dir / "g0w0_takada_curve.csv", curves)

    projected = TakadaProjection(
        params=PARAMS, densities=densities, temperature=args.temperature,
        n_q=args.takada_nq, n_theta=args.takada_n_theta,
        n_neg=args.takada_n_neg, n_pos=args.takada_n_pos,
        x_core=1.0e-5, x_max=1.0e3, max_symmetry_error=1.0e-10,
    ).scan()
    np.savetxt(
        out_dir / "takada_curve.csv",
        [(row.n_cm3, row.lam, row.eps_F_eV, row.k_F_A, row.symmetry_error)
         for row in projected], delimiter=",", comments="",
        header="n_cm3,lambda,eps_F_eV,k_F_A^-1,symmetry_error", fmt="%.18e",
    )
    settings = dict(vars(args), densities_cm3=list(map(float, densities)),
                    model=vars(PARAMS))
    (out_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
