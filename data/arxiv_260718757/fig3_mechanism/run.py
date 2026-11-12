#!/usr/bin/env python
"""Recompute Fig. 3 controls and gap functions. 

    mpirun -n 2 python data/arxiv_260718757/fig3_mechanism/run.py

Parameters are in K, cm^-3, and cm^-1; the solvers use atomic units.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from migdal.memory import DEFAULT_MAX_MEMORY
from migdal import ir_cache

TO_CM = (9.98977881246349, 180.5746693840041, 590.7326992132146)
LO_CM = (175.83894684689574, 449.4013216886174, 823.7949569635838)
CONTROL_METHODS = ("G0W0", "scGW", "G0W0_no_phonon", "G0V", "GV")
SUMMARY_FIELDS = ("lambda", "gap_converged", "scf_converged", "scf_cycles", "nk", "nq")
SCALE_FIELDS = (
    "method", "n_cm3", "k_F_Ainv", "kmax_Ainv", "qmax_Ainv", "E_F_meV",
    "phonon_scale_meV", "E_plasmon_meV", "E_TF_meV", "lambda",
    "gap_converged", "gap_cycles", "scf_converged", "scf_cycles",
    "scf_res_sigma", "scf_dn", "nk", "nq", "n_active_w", "used_min_pair",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output")
    parser.add_argument("--part", choices=("all", "controls", "gap"), default="all")
    parser.add_argument("--densities", help="Comma-separated densities; default is the 11-point log grid.")
    parser.add_argument("--gap-densities", default="1e17,4e20,1e23")
    parser.add_argument("--gap-methods", default="G0W0,GW0,scGW")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=1e9)
    parser.add_argument("--ir-file", type=Path,
                        help="Use an IR cache at the requested temperature and Lambda.")
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--n-2kf-min", type=int, default=101)
    parser.add_argument("--k-left", type=int, default=200)
    parser.add_argument("--k-right", type=int, default=200)
    parser.add_argument("--k-shell-min", type=int, default=50)
    parser.add_argument("--k-shell-max", type=int, default=5000)
    parser.add_argument("--n-tail", type=int, default=100)
    parser.add_argument("--scf-max-cycle", type=int, default=300)
    parser.add_argument("--davidson-max-cycle", type=int, default=300)
    parser.add_argument("--max-memory", type=float, default=DEFAULT_MAX_MEMORY,
                        help="Memory budget in MB per rank, following PySCF.")
    parser.add_argument("--n0-check-atol", type=float, default=1e-12)
    parser.add_argument("--pi0-mode", choices=("finite_T", "0K", "GG"), default="finite_T")
    parser.add_argument("--allow-unconverged", action="store_true", help="Write failed convergence flags for diagnostic runs.")
    parser.add_argument("--verbose", type=int, default=4)
    args = parser.parse_args(argv)
    for name in ("densities", "gap_densities"):
        text = getattr(args, name)
        values = (np.geomspace(1e23, 1e16, 11) if text is None
                  else np.array([float(v) for v in text.split(",")]))
        if not np.all(np.isfinite(values) & (values > 0)):
            parser.error("%s must contain positive finite densities" % name)
        setattr(args, name, values)
    args.gap_methods = args.gap_methods.split(",")
    if any(m not in ("G0W0", "GW0", "scGW") for m in args.gap_methods):
        parser.error("--gap-methods must contain G0W0, GW0, or scGW")
    return args


def paper_params():
    from migdal.model import STOParams

    return STOParams(m_eff=2.1, eps_inf=6.163085088, eps_0=23000.0,
                     phonon_to_poles_cm=TO_CM, phonon_lo_zeros_cm=LO_CM)


class LatticeOnly:
    """Set Pi=0, retaining the same Coulomb head and diagonal cell averages.
    """

    def _build_W0(self, a, mu, mu0, k_grid, k_wts, G0, Vph, q_grid):
        from migdal import model, radial

        vc = model.get_Vc(q_grid, self.params.eps_inf, q0_treat=False,
                          tol=np.nextafter(0.0, 1.0))
        vtot = vc[:, None] * Vph
        return radial.get_W_from_Pi(
            self.ctx, np.zeros_like(vtot), q_grid, vc, vtot,
            denom_policy=self.denom_policy, denom_min=self.denom_min,
            q_interp_mode="q_singular", q_singular_head_policy="pi_head",
            static_singular_coeff=self._static_coulomb_head(),
            vtot_head=self._vtot_head(self.ctx), return_metrics=True,
        )


def make_driver(method, args, params, gap_function=False):
    from migdal.g0w0 import G0W0
    from migdal.gw0 import GW0
    from migdal.scgw import ScGW

    class G0V(LatticeOnly, G0W0):
        pass

    class GV(LatticeOnly, GW0):
        pass

    if method == "G0W0_no_phonon":
        params = replace(params, eps_0=params.eps_inf,
                         phonon_to_poles_cm=(), phonon_lo_zeros_cm=())
    cls = {"G0W0": G0W0, "GW0": GW0, "scGW": ScGW,
           "G0W0_no_phonon": G0W0, "G0V": G0V, "GV": GV}[method]
    kwargs = dict(
        params=params, temperature=args.temperature, lamb=args.lamb,
        ir_tol=1e-14, ir_file=args.ir_file, save_ir=True,
        nq=args.nq, n_2kf_min=args.n_2kf_min,
        kgrid_n_pad=5, kgrid_n_left=args.k_left, kgrid_n_right=args.k_right,
        kgrid_n_shell_min=args.k_shell_min, kgrid_n_shell_max=args.k_shell_max,
        kmax_factor=4.0, kmax_lo_factor=2.0, qmax_factor=2.0,
        truncate=False, q_interp_mode="q_singular", q_singular_head_policy="pi_head",
        diag_average=1, diag_ncell=12, diag_average_mode="auto",
        pi0_mode=args.pi0_mode, denom_policy="sign_regularize", denom_min=1e-10,
        n0_check_atol=args.n0_check_atol, davidson_tol=1e-6,
        davidson_max_cycle=args.davidson_max_cycle, davidson_max_space=20,
        precond="even_diag", cache="auto", max_memory=args.max_memory,
        return_gap_function=gap_function,
        verbose=args.verbose,
    )
    if method in ("GW0", "scGW", "GV"):
        kwargs.update(scf_conv_tol=1e-6, scf_max_cycle=args.scf_max_cycle,
                      scf_damp=0.98, scf_diis_space=8, scf_diis_start_cycle=0,
                      scf_fs_pin=True, scf_high_freq_density=True,
                      allow_unconverged=args.allow_unconverged)
    if method == "scGW":
        kwargs.pop("pi0_mode")
        kwargs.update(n_tail=args.n_tail, sigma_tail_policy="exchange_static",
                      pi_imag_tol=1e-5, pi_imag_policy="warn")
    return cls(**kwargs)


def summary(row):
    return dict(zip(SUMMARY_FIELDS, (
        row.lam, int(row.gap_converged),
        -1 if row.scf_converged is None else int(row.scf_converged),
        -1 if row.scf_cycles is None else row.scf_cycles, row.nk, row.nq,
    )))


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_controls(out_dir, curves):
    """Save finished points; wide rows require all three baseline methods."""
    fields = ["n_cm3", *SUMMARY_FIELDS, "wall_s"]
    for method, rows in curves.items():
        if rows:
            write_csv(out_dir / (method.lower() + "_scan.csv"), fields, rows)
    methods = CONTROL_METHODS[:3]
    wide = []
    for points in zip(*(curves[method] for method in methods)):
        row = {"n_cm3": points[0]["n_cm3"]}
        for method, point in zip(methods, points):
            row.update({method + "_" + key: point[key] for key in SUMMARY_FIELDS})
        wide.append(row)
    fields = ["n_cm3"] + [method + "_" + key for method in methods for key in SUMMARY_FIELDS]
    write_csv(out_dir / "screening_phonon_controls.csv", fields, wide)


def gap_rows(driver, row, method):
    """Match the original dk average and maximum-amplitude phase convention."""
    from migdal import model
    from pyscf.data.nist import BOHR, HARTREE2EV

    k = driver._last_k_grid
    weights = driver._last_k_wts
    phi = row.gap.phi_tilde / k[:, None]
    average = np.tensordot(weights, phi, axes=(0, 0)) / np.sum(weights)
    pos = driver.ctx.ws > 0
    values = average[pos]
    values *= np.exp(-1j * np.angle(values[np.argmax(np.abs(values))]))
    values /= max(float(np.max(np.abs(values.real))), 1e-300)
    rows = [dict(method=method, n_cm3=row.n_cm3, omega_meV=float(w * HARTREE2EV * 1e3),
                 gap_kavg_norm=float(v.real), gap_kavg_imag_norm=float(v.imag))
            for w, v in zip(driver.ctx.ws[pos], values)]
    data, params = driver._last_point_data, driver.params
    kf = data["k_F"]
    ha_to_mev = HARTREE2EV * 1e3
    scales = dict(
        method=method, n_cm3=row.n_cm3, k_F_Ainv=kf / BOHR,
        kmax_Ainv=data["kmax"] / BOHR, qmax_Ainv=max(driver._last_W.q) / BOHR,
        E_F_meV=row.mu0 * ha_to_mev,
        phonon_scale_meV=model.phonon_grid_ha(params) * ha_to_mev,
        E_plasmon_meV=np.sqrt(4 * np.pi * row.n_bohr3 / (params.m_eff * params.eps_inf)) * ha_to_mev,
        E_TF_meV=2 * kf / (np.pi * params.eps_inf) * ha_to_mev,
        **summary(row), gap_cycles=row.gap.davidson_cycles,
        scf_res_sigma=np.nan if row.scf_res_sigma is None else row.scf_res_sigma,
        scf_dn=np.nan if row.scf_dn is None else row.scf_dn,
        n_active_w=row.gap.n_active_w, used_min_pair=int(row.gap.used_min_pair),
    )
    return rows, scales


def main(argv=None):
    args = parse_args(argv)
    from mpi4pyscf.tools import mpi

    args.out_dir = args.out_dir.resolve()
    if mpi.rank == 0:
        (args.out_dir / "cache").mkdir(parents=True, exist_ok=True)
    params = paper_params()
    if args.ir_file is None:
        args.ir_file = args.out_dir / "cache" / ("ir_L%.12g_T%.12gK.h5" % (args.lamb, args.temperature))
        if mpi.rank == 0 and args.lamb == 1e9:
            ir_cache.write_scaled_sparse_ir_h5(
                HERE.parent / "ir.h5", args.ir_file,
                params.beta_ha(args.temperature), overwrite=True)
    if mpi.rank == 0:
        metadata = {key: value.tolist() if isinstance(value, np.ndarray) else str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}
        metadata["params"] = asdict(params)
        metadata["q_interp_mode"] = "q_singular"
        metadata["control_scgw_initialization"] = "previous converged density with cold retry if SCF fails; other controls and gaps cold"
        metadata["gap_average"] = "dk; phase from maximum magnitude; normalized by maximum real amplitude"
        (args.out_dir / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.part in ("all", "controls"):
        curves = {method: [] for method in CONTROL_METHODS}
        for method in CONTROL_METHODS:
            drv = make_driver(method, args, params)
            rows = curves[method]
            seed = {}
            if method == "scGW":
                drv.allow_unconverged = True
                scan_path, keep = drv._scan_path(args.densities, max_step_dex=drv.scf_continuation_max_dex)
            else:
                scan_path, keep = args.densities, np.ones(len(args.densities), dtype=bool)
            for density, keep_row in zip(scan_path, keep):
                start = time.monotonic()
                result = drv.kernel(float(density), **seed)
                if method == "scGW":
                    if seed and not result.scf_converged:
                        result = drv.kernel(float(density))
                    if not result.scf_converged and not args.allow_unconverged:
                        raise RuntimeError("scGW did not converge at n=%.6g cm^-3" % density)
                if keep_row:
                    rows.append(dict(n_cm3=float(density), **summary(result), wall_s=time.monotonic() - start))
                if method == "scGW":
                    normal = drv._last_scf
                    seed = (dict(sigma0=normal.sigma, mu0=normal.mu,
                                 sigma0_k=drv._last_k_grid,
                                 fs_pin_shift0=normal.extra.get("fs_pin_shift"))
                            if normal.converged else {})
                if mpi.rank == 0:
                    write_controls(args.out_dir, curves)
                if not result.gap_converged and not args.allow_unconverged:
                    raise RuntimeError("%s gap did not converge at n=%g" % (method, density))
    if args.part in ("all", "gap"):
        rows, scales = [], []
        for method in args.gap_methods:
            drv = make_driver(method, args, params, gap_function=True)
            for density in args.gap_densities:
                result = drv.kernel(float(density))
                gap_data, scale_data = gap_rows(drv, result, method)
                rows.extend(gap_data)
                scales.append(scale_data)
                if mpi.rank == 0:
                    write_csv(args.out_dir / "gw0_scgw_gap_function.csv",
                              ("method", "n_cm3", "omega_meV", "gap_kavg_norm", "gap_kavg_imag_norm"), rows)
                    write_csv(args.out_dir / "gw0_scgw_qavg_w_scales.csv", SCALE_FIELDS, scales)
                if not result.gap_converged and not args.allow_unconverged:
                    raise RuntimeError("%s gap did not converge at n=%g" % (method, density))
    if mpi.rank == 0:
        print("wrote Fig. 3 %s data to %s" % (args.part, args.out_dir))


if __name__ == "__main__":
    main()
