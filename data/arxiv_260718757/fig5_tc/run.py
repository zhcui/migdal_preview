#!/usr/bin/env python
"""Fig. 5: scan lambda(T), seeding each scGW point with same-temperature GW0.

    mpirun -n 2 python data/arxiv_260718757/fig5_tc/run.py
    python data/arxiv_260718757/fig5_tc/plot.py --data-dir data/arxiv_260718757/fig5_tc/output
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from migdal import G0W0, GW0, ScGW, ir_cache, model, radial, scf
from migdal.freq import FreqGrid
from migdal.mpi_wrap import mpi

import tc


HERE = Path(__file__).resolve().parent
HA_TO_K = 315775.02480407
PARAMS = model.STOParams(
    m_eff=2.1, eps_inf=6.163085088, eps_0=23000.0,
    phonon_to_poles_cm=tuple(np.array([
        1.2385747182565898, 22.388405634920773, 73.24152018061872,
    ]) / (1000 * model.CM1_TO_EV)),
    phonon_lo_zeros_cm=tuple(np.array([
        21.801250872312576, 55.718662629463765, 102.13755738509244,
    ]) / (1000 * model.CM1_TO_EV)),
)
DENSITIES = np.sort(np.r_[
    np.geomspace(1e23, 1e16, 11), np.geomspace(10**13.2, 10**15.3, 4),
])[::-1]
LOW_DENSITIES = DENSITIES[-2:]
GW0_RETRY_SCF_TOL = 1e-8
TEMPERATURES = np.geomspace(150, 0.1, 20)
LAMBDA_COLUMNS = (
    "method", "n_cm3", "temperature_K", "lambda", "gap_converged",
    "gap_cycles", "scf_converged", "scf_cycles", "scf_res_sigma", "scf_dn",
    "nk", "nq", "mu_Ha", "mu0_Ha", "ir_file",
)
SEED_COLUMNS = (
    "n_cm3", "temperature_K", "seed_method", "seed_temperature_K",
    "scgw_gw0_seed_used", "gw0_temperature_restart_used",
    "gw0_retry_without_restart", "scgw_cold_gw0_seed_retry",
    "seed_scf_converged", "seed_scf_cycles", "seed_scf_tol", "scgw_scf_converged",
    "scgw_scf_cycles", "lambda",
)
Z_COLUMNS = (
    "n_cm3", "temperature_K", "lambda", "Z_kF_i0", "E_F_K", "E_F_star_K",
    "sigma_symm_err", "scf_converged", "scf_cycles", "nk", "nq",
    "k_F_bohr_inv", "mu_Ha", "ir_file",
)
COOLING_COLUMNS = (
    "method", "n_cm3", "temperature_K", "restart_used",
    "restart_retry_without_restart", "restart_updated",
    "restart_source_temperature_K", "lambda", "gap_converged",
    "scf_converged", "scf_cycles",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--densities", default=",".join("%.12g" % n for n in DENSITIES))
    p.add_argument("--temperatures", default=",".join("%.12g" % t for t in TEMPERATURES))
    p.add_argument("--methods", nargs="+", choices=tc.METHODS, default=list(tc.METHODS))
    p.add_argument("--out-dir", type=Path, default=HERE / "output")
    p.add_argument("--resume", action="store_true", help="Keep completed density chains; rerun the unfinished chain.")
    p.add_argument("--max-memory", type=float, default=102400,
                   help="Cache memory budget in MB per rank.")
    p.add_argument("--lamb", type=float, default=1e9)
    p.add_argument("--nq", type=int, default=600)
    p.add_argument("--scgw-nq", type=int, default=600)
    p.add_argument("--k-left", type=int, default=200)
    p.add_argument("--k-right", type=int, default=200)
    p.add_argument("--k-shell-min", type=int, default=50)
    p.add_argument("--k-shell-max", type=int, default=5000)
    p.add_argument("--scf-max-cycle", type=int, default=300)
    p.add_argument("--scf-damp", type=float, default=0.98)
    p.add_argument("--scgw-n-tail", type=int, default=100)
    p.add_argument("--davidson-tol", type=float, default=1e-6)
    p.add_argument("--davidson-nroots", type=int, default=3)
    p.add_argument("--davidson-max-space", type=int, default=40)
    p.add_argument("--davidson-max-cycle", type=int, default=300)
    p.add_argument("--n0-check-atol", type=float, default=1e-12)
    p.add_argument("--pi0-mode", choices=("finite_T", "0K", "GG"), default="finite_T")
    p.add_argument("--low-density-patch", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the published finer grids at the two lowest densities.")
    p.add_argument("--stop-after-crossing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", type=int, default=4)
    args = p.parse_args(argv)
    args.densities = [float(n) for n in args.densities.split(",")]
    args.temperatures = sorted({float(t) for t in args.temperatures.split(",")}, reverse=True)
    return args


def low_density(n, args):
    return args.low_density_patch and np.any(np.isclose(n, LOW_DENSITIES, rtol=1e-10, atol=0))


def make_driver(method, n, temperature, cache, args, seed=False):
    # The GW0 seed uses the same fine k grid as scGW at the two dilute points.
    patched = low_density(n, args) and (method == "scGW" or seed)
    settings = dict(
        params=PARAMS, densities=[n], temperature=temperature,
        lamb=args.lamb, ir_file=str(cache), save_ir=True,
        nq=args.scgw_nq if method == "scGW" else args.nq,
        kgrid_n_pad=5, kgrid_n_left=args.k_left, kgrid_n_right=args.k_right,
        kgrid_n_shell_min=max(200, args.k_shell_min) if patched else args.k_shell_min,
        kgrid_n_shell_max=args.k_shell_max,
        truncate=False, diag_average=1, diag_ncell=12, diag_average_mode="auto",
        q_interp_mode="q_singular", q_singular_head_policy="pi_head",
        pi0_mode=args.pi0_mode, n0_check_atol=args.n0_check_atol,
        max_memory=args.max_memory,
        davidson_tol=args.davidson_tol, davidson_max_cycle=args.davidson_max_cycle,
        nroots=args.davidson_nroots, davidson_max_space=args.davidson_max_space,
        precond="even_diag", verbose=args.verbose,
    )
    if method == "G0W0":
        return G0W0(**settings)
    settings.update(
        scf_max_cycle=900 if patched and method == "scGW" else args.scf_max_cycle,
        scf_damp=0.995 if patched and method == "scGW" else args.scf_damp,
        scf_conv_tol=1e-6,
        scf_diis_space=8, scf_diis_start_cycle=0,
        scf_fs_pin=True, allow_unconverged=True,
    )
    if method == "GW0":
        return GW0(**settings)
    settings.pop("pi0_mode")
    if patched:
        settings["nq"] = max(750, args.scgw_nq)
    return ScGW(**settings, n_tail=args.scgw_n_tail,
                pi_imag_tol=1e-5, pi_imag_policy="warn", sigma_tail_policy="exchange_static")


def lambda_row(method, point, cache):
    return dict(
        method=method, n_cm3=point.n_cm3, temperature_K=point.temperature_K,
        **{"lambda": point.lam}, gap_converged=int(point.gap_converged),
        gap_cycles=point.gap.davidson_cycles,
        scf_converged=-1 if point.scf_converged is None else int(point.scf_converged),
        scf_cycles=-1 if point.scf_cycles is None else point.scf_cycles,
        scf_res_sigma="" if point.scf_res_sigma is None else point.scf_res_sigma,
        scf_dn="" if point.scf_dn is None else point.scf_dn,
        nk=point.nk, nq=point.nq, mu_Ha=point.mu, mu0_Ha=point.mu0,
        ir_file=str(Path("cache") / cache.name),
    )


def z_row(driver, point, cache):
    data = driver._last_point_data
    k = data["k_scf"]
    sigma = driver._last_scf.sigma
    z = scf.low_freq_Z_at_k(driver.ctx, sigma, k, data["k_F"])
    if not np.isfinite(z) or z <= 0:
        raise FloatingPointError("invalid Z(k_F, i0) = %s" % z)
    ef = data["k_F"]**2 / (2 * PARAMS.m_eff) * HA_TO_K
    return dict(
        n_cm3=point.n_cm3, temperature_K=point.temperature_K,
        **{"lambda": point.lam}, Z_kF_i0=z, E_F_K=ef, E_F_star_K=ef / z,
        sigma_symm_err=radial.check_sigma_symm(driver.ctx, sigma),
        scf_converged=int(point.scf_converged), scf_cycles=point.scf_cycles,
        nk=point.nk, nq=point.nq, k_F_bohr_inv=data["k_F"], mu_Ha=point.mu,
        ir_file=str(Path("cache") / cache.name),
    )


def solve_gw0(n, temperature, cache, args, state, seed=False):
    driver = make_driver("GW0", n, temperature, cache, args, seed=seed)
    point = driver.kernel(restart_state=state)
    retried = state is not None and not point.scf_converged
    if retried:
        driver = make_driver("GW0", n, temperature, cache, args, seed=seed)
        point = driver.kernel()
    return driver, point, retried


def retry_scgw(n, temperature, cache, args, cold_seed_failed=False):
    """Retry scGW with cold GW0 seeds, tightening the final seed to 1e-8."""
    for seed_tol in (1e-6, GW0_RETRY_SCF_TOL):
        if cold_seed_failed and seed_tol == 1e-6:
            continue
        gw = make_driver("GW0", n, temperature, cache, args, seed=low_density(n, args))
        gw.scf_conv_tol = seed_tol
        gp = gw.kernel()
        if not gp.scf_converged:
            continue
        sg = make_driver("scGW", n, temperature, cache, args)
        sp = sg.kernel(restart_state=gw.last_scf_restart_state())
        if sp.scf_converged:
            return gw, gp, sg, sp
    return None


def main(argv=None):
    args = parse_args(argv)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    rank = mpi.rank
    settings = {k: v for k, v in vars(args).items() if k not in ("out_dir", "resume")}
    settings["gw0_retry_scf_tol"] = GW0_RETRY_SCF_TOL
    settings["ir_source"] = "../ir.h5" if args.lamb == 1e9 else "sparse-ir"
    settings_path = out / "settings.json"
    completed = []
    if args.resume and (out / "progress.json").exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("resume settings differ from output/settings.json")
        completed = json.loads((out / "progress.json").read_text())["completed_densities"]
        def load(name):
            if not completed:
                return []
            return [r for r in tc.read_csv(out / name) if float(r["n_cm3"]) in completed]
        rows = load("lambda_vs_temperature.csv")
        seed_rows = load("scgw_gw0_seed_log.csv")
        z_rows = load("efstar_vs_temperature.csv")
        cooling_rows = load("cooling_restart_log.csv")
    else:
        rows, seed_rows, z_rows, cooling_rows = [], [], [], []
    if rank == 0:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    def save_progress():
        if rank == 0:
            temporary = out / "progress.json.tmp"
            temporary.write_text(json.dumps({"completed_densities": completed}, indent=2) + "\n")
            temporary.replace(out / "progress.json")

    save_progress()

    base_ir = out / "cache" / ("ir_L%.12g_T1K.h5" % args.lamb)
    if any(n not in completed for n in args.densities):
        if rank == 0 and args.lamb == 1e9:
            ir_cache.write_scaled_sparse_ir_h5(
                HERE.parent / "ir.h5", base_ir, PARAMS.beta_ha(1.0), overwrite=True)
        FreqGrid(PARAMS.beta_ha(1.0), lamb=args.lamb, ir_file=base_ir,
                 save_ir=True, verbose=args.verbose)
    cached_temperatures = {1.0}

    def save():
        if rank == 0:
            summary = tc.summarize(rows)
            tc.write_csv(out / "lambda_vs_temperature.csv", rows, LAMBDA_COLUMNS)
            tc.write_csv(out / "tc_vs_density.csv", summary, tc.TC_COLUMNS)
            tc.write_csv(out / "scgw_gw0_seed_log.csv", seed_rows, SEED_COLUMNS)
            tc.write_csv(out / "cooling_restart_log.csv", cooling_rows, COOLING_COLUMNS)
            tc.write_csv(out / "efstar_vs_temperature.csv", z_rows, Z_COLUMNS)
            tc.write_csv(out / "efstar_line.csv", tc.fermi_line(summary, z_rows), tc.EFSTAR_COLUMNS)

    def record(method, point, cache, restarted=False, retried=False, source_t=""):
        row = lambda_row(method, point, cache)
        rows.append(row)
        cooling_rows.append(dict(
            method=method, n_cm3=point.n_cm3, temperature_K=point.temperature_K,
            restart_used=int(restarted), restart_retry_without_restart=int(retried),
            restart_updated=int(method == "GW0" and bool(point.scf_converged)),
            restart_source_temperature_K=source_t,
            **{key: row[key] for key in ("lambda", "gap_converged", "scf_converged", "scf_cycles")},
        ))

    for n in args.densities:
        if n in completed:
            continue
        stopped = set()
        gw_state, gw_temperature = None, ""
        fine_state = None
        for temperature in args.temperatures:
            active = [m for m in args.methods if m not in stopped]
            if not active:
                break
            cache = out / "cache" / ("ir_L%.12g_T%.12gK.h5" % (args.lamb, temperature))
            if temperature not in cached_temperatures:
                if rank == 0:
                    ir_cache.write_scaled_sparse_ir_h5(
                        base_ir, cache, PARAMS.beta_ha(temperature), overwrite=True)
                cached_temperatures.add(temperature)
            if "G0W0" in active:
                point = make_driver("G0W0", n, temperature, cache, args).kernel()
                record("G0W0", point, cache)
                save()
            fine_seed = low_density(n, args) and "scGW" in active
            if "GW0" in active or ("scGW" in active and not fine_seed):
                restarted = gw_state is not None
                seed_t = gw_temperature if restarted else ""
                gw, gp, retried = solve_gw0(n, temperature, cache, args, gw_state)
                if gp.scf_converged:
                    gw_state = gw.last_scf_restart_state()
                    gw_temperature = temperature
                if "GW0" in active:
                    record("GW0", gp, cache, restarted, retried, seed_t)
                    save()
            if "scGW" in active:
                if fine_seed:
                    restarted = fine_state is not None
                    gw, gp, retried = solve_gw0(n, temperature, cache, args, fine_state, seed=True)
                    if gp.scf_converged:
                        fine_state = gw.last_scf_restart_state()
                if not gp.scf_converged:
                    raise RuntimeError("GW0 seed did not converge at n=%.6g cm^-3, T=%.6g K" % (n, temperature))
                sg = make_driver("scGW", n, temperature, cache, args)
                sp = sg.kernel(restart_state=gw.last_scf_restart_state())
                cold_seed_retry = False
                if not sp.scf_converged:
                    recovered = retry_scgw(n, temperature, cache, args,
                                           cold_seed_failed=not restarted or retried)
                    if recovered is not None:
                        gw, gp, sg, sp = recovered
                        if fine_seed:
                            fine_state = gw.last_scf_restart_state()
                        else:
                            gw_state = gw.last_scf_restart_state()
                        cold_seed_retry = True
                record("scGW", sp, cache, True, source_t=temperature)
                z_rows.append(z_row(sg, sp, cache))
                seed_rows.append(dict(
                    n_cm3=n, temperature_K=temperature, seed_method="GW0",
                    seed_temperature_K=temperature, scgw_gw0_seed_used=1,
                    gw0_temperature_restart_used=int(restarted and not retried and not cold_seed_retry),
                    gw0_retry_without_restart=int(retried or cold_seed_retry),
                    scgw_cold_gw0_seed_retry=int(cold_seed_retry),
                    seed_scf_converged=1, seed_scf_cycles=gp.scf_cycles,
                    seed_scf_tol=gw.scf_conv_tol,
                    scgw_scf_converged=int(sp.scf_converged), scgw_scf_cycles=sp.scf_cycles,
                    **{"lambda": sp.lam},
                ))
                save()
            for row in tc.summarize(r for r in rows if r["n_cm3"] == n):
                patched_scgw = row["method"] == "scGW" and low_density(n, args)
                if args.stop_after_crossing and not patched_scgw and row["status"] == "crossed":
                    stopped.add(row["method"])
        completed.append(n)
        save_progress()
    if rank == 0:
        for row in tc.summarize(rows):
            value = row["tc_K"] if row["status"] == "crossed" else row["status"]
            print("%s n=%.6e: %s" % (row["method"], row["n_cm3"], value))


if __name__ == "__main__":
    main()
