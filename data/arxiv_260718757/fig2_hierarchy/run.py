#!/usr/bin/env python
"""Figure 2: continued pairing curves and independent cold Z(k_F, i*pi*T).

    mpirun -n 2 python data/arxiv_260718757/fig2_hierarchy/run.py
    python data/arxiv_260718757/fig2_hierarchy/plot.py --data-dir data/arxiv_260718757/fig2_hierarchy/output
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from migdal import G0W0, GW0, ScGW, ir_cache, memory, model, scf

METHODS = ("G0W0", "GW0", "scGW")
CURVE_FIELDS = ("lambda", "gap_converged", "scf_converged", "scf_cycles", "nk", "nq")
Z_FIELDS = ("n_cm3", "method", "lambda", "Z_kF", "mu_ha", "scf_converged",
            "scf_cycles", "wall_s", "gap_converged", "temperature_K")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output")
    parser.add_argument("--part", choices=("all", "curves", "z"), default="all",
                        help="Run continued pairing curves, independent cold Z, or both.")
    parser.add_argument("--densities", help="Comma-separated densities in cm^-3.")
    parser.add_argument("--n-density", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--k-left", type=int, default=200)
    parser.add_argument("--k-right", type=int, default=200)
    parser.add_argument("--k-shell-min", type=int, default=50)
    parser.add_argument("--k-shell-max", type=int, default=5000)
    parser.add_argument("--lamb", type=float, default=1.0e9)
    parser.add_argument("--ir-tol", type=float, default=1.0e-14)
    parser.add_argument("--ir-file", type=Path,
                        help="Use an IR cache at the requested temperature and Lambda.")
    parser.add_argument("--pi0-mode", choices=("finite_T", "0K", "GG"), default="finite_T")
    parser.add_argument("--diag-ncell", type=int, default=12)
    parser.add_argument("--n-tail", type=int, default=100)
    parser.add_argument("--n0-check-atol", type=float, default=1.0e-12)
    parser.add_argument("--scf-max-cycle", type=int, default=300)
    parser.add_argument("--scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--davidson-max-cycle", type=int, default=300)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-6)
    parser.add_argument("--max-memory", type=float, default=memory.DEFAULT_MAX_MEMORY,
                        help="Memory budget in MB per rank, following PySCF.")
    parser.add_argument("--restart-scf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-unconverged", action="store_true")
    parser.add_argument("--verbose", type=int, default=4)
    return parser.parse_args(argv)


def make_drivers(args, densities, methods=METHODS):
    params = model.STOParams(
        m_eff=2.1, eps_inf=6.163085088, eps_0=23000.0,
        phonon_to_poles_cm=(9.98977881246349, 180.5746693840041, 590.7326992132146),
        phonon_lo_zeros_cm=(175.83894684689574, 449.4013216886174, 823.7949569635838),
    )
    radial = dict(
        params=params, densities=densities, temperature=args.temperature,
        lamb=args.lamb, ir_tol=args.ir_tol,
        ir_file=args.ir_file, save_ir=True,
        nq=args.nq, n_2kf_min=101, truncate=False,
        q_interp_mode="q_singular", q_singular_head_policy="pi_head",
        diag_average="auto", diag_average_mode="auto", diag_ncell=args.diag_ncell,
        kgrid_n_pad=5, kgrid_n_left=args.k_left, kgrid_n_right=args.k_right,
        kgrid_n_shell_min=args.k_shell_min, kgrid_n_shell_max=args.k_shell_max,
        n0_check_atol=args.n0_check_atol,
        davidson_tol=args.davidson_tol, davidson_max_cycle=args.davidson_max_cycle,
        davidson_max_space=20, precond="even_diag", cache="auto",
        max_memory=args.max_memory, verbose=args.verbose,
    )
    normal = dict(
        scf_max_cycle=args.scf_max_cycle, scf_conv_tol=args.scf_conv_tol,
        scf_damp=0.98, scf_diis_space=8, scf_diis_start_cycle=0,
        scf_fs_pin=True, scf_high_freq_density=True,
        allow_unconverged=not args.fail_unconverged,
    )
    # Construct each method only after the previous method has finished.
    if "G0W0" in methods:
        yield G0W0(**radial, pi0_mode=args.pi0_mode)
    if "GW0" in methods:
        yield GW0(**radial, **normal, pi0_mode=args.pi0_mode)
    if "scGW" in methods:
        yield ScGW(**radial, **normal, n_tail=args.n_tail,
                   sigma_tail_policy="exchange_static", pi_imag_tol=1.0e-5,
                   pi_imag_policy="warn", pi_low_q_policy="asymptotic")


def z_at_fermi_surface(driver):
    """Interpolate Z at the lowest positive Matsubara frequency, as in the paper."""
    ctx = driver._last_ctx
    sigma = driver._last_scf.sigma
    k_F = driver._last_point_data["k_F"]
    i0 = np.flatnonzero(ctx.ws > 0)[0]
    Z = scf.get_Z_matsubara(ctx, sigma)
    return float(np.interp(k_F, driver._last_k_grid, Z[:, i0]))


def write_curves(path, curves, temperature):
    header = ["n_cm3"] + [name + "_" + field for name in METHODS for field in CURVE_FIELDS]
    header.append("temperature_K")
    rows = []
    for points in zip(*(curves[method] for method in METHODS)):
        values = [points[0].n_cm3]
        for row in points:
            values.extend((row.lam, int(row.gap_converged),
                           -1 if row.scf_converged is None else int(row.scf_converged),
                           -1 if row.scf_cycles is None else row.scf_cycles, row.nk, row.nq))
        rows.append(values + [temperature])
    formats = ["%.18e"] + ["%.18e", "%d", "%d", "%d", "%d", "%d"] * len(METHODS) + ["%.18e"]
    temporary = path.with_name(path.name + ".tmp")
    if rows:
        np.savetxt(temporary, rows, delimiter=",", header=",".join(header), comments="", fmt=formats)
    else:
        temporary.write_text(",".join(header) + "\n")
    temporary.replace(path)


def write_z(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=Z_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    from mpi4pyscf.tools import mpi

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.ir_file is None:
        args.ir_file = args.out_dir / "cache" / ("ir_L%.12g_T%.12gK.h5" % (args.lamb, args.temperature))
        if mpi.rank == 0 and args.lamb == 1e9:
            ir_cache.write_scaled_sparse_ir_h5(
                HERE.parent / "ir.h5", args.ir_file,
                model.STOParams().beta_ha(args.temperature), overwrite=True)
    densities = (model.density_grid(args.densities) if args.densities
                 else np.geomspace(1.0e23, 1.0e16, args.n_density))
    if mpi.rank == 0:
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != "out_dir"}
        settings["densities_cm3"] = densities.tolist()
        settings["q_interp_mode"] = "q_singular"
        settings["q_singular_head_policy"] = "pi_head"
        settings["scf_fs_pin"] = True
        settings["density_restart"] = "direct sigma0, mu0, sigma0_k, fs_pin_shift0; cold retry if SCF fails"
        settings["z_initialization"] = "cold at each density; independent of curve states"
        (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    curves, z_rows = {method: [] for method in METHODS}, []
    if args.part in ("all", "curves"):
        for method, driver in zip(METHODS, make_drivers(args, densities)):
            points, seed = curves[method], {}
            if method == "G0W0":
                scan_path, keep = densities, np.ones(len(densities), dtype=bool)
            else:
                max_dex = driver.scf_continuation_max_dex if args.restart_scf else 0.0
                scan_path, keep = driver._scan_path(densities, max_step_dex=max_dex)
            for n_cm3, keep_row in zip(scan_path, keep):
                row = driver.kernel(float(n_cm3), **seed)
                if seed and not row.scf_converged:
                    row = driver.kernel(float(n_cm3))
                if keep_row:
                    points.append(row)
                if method != "G0W0":
                    result = driver._last_scf
                    seed = (dict(sigma0=result.sigma, mu0=result.mu,
                                 sigma0_k=driver._last_k_grid,
                                 fs_pin_shift0=result.extra.get("fs_pin_shift"))
                            if args.restart_scf and result.converged else {})
                if mpi.rank == 0 and points:
                    driver.results = points
                    path = args.out_dir / (method.lower() + ".csv")
                    temporary = path.with_name(path.name + ".tmp")
                    driver.write(temporary)
                    temporary.replace(path)
                    # Wide rows require all methods to finish the requested density.
                    write_curves(args.out_dir / "g0w0_gw0_scgw_curve.csv", curves, args.temperature)

    # The manuscript Z scan solved each density cold, independently of the
    # continued lambda curves.
    if args.part in ("all", "z"):
        for method, driver in zip(METHODS[1:], make_drivers(args, densities, METHODS[1:])):
            for n_cm3 in densities:
                start = time.perf_counter()
                row = driver.kernel(float(n_cm3))
                z_rows.append(dict(
                    n_cm3=float(n_cm3), method=method, **{"lambda": row.lam},
                    Z_kF=z_at_fermi_surface(driver), mu_ha=driver._last_scf.mu,
                    scf_converged=int(row.scf_converged), scf_cycles=row.scf_cycles,
                    wall_s=time.perf_counter() - start,
                    gap_converged=int(row.gap_converged), temperature_K=args.temperature,
                ))
                if mpi.rank == 0:
                    write_z(args.out_dir / "z_scan.csv", z_rows)

    if mpi.rank == 0:
        print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
