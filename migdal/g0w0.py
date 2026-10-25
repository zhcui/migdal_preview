"""G0W0 radial gap driver for the STO one-band model.

The driver prepares one density point at a time: convert density units, build
radial k/q grids, form the fixed RPA interaction ``W0``, and pass the resulting
normal Green's function and interaction to :func:`migdal.gap.solve_gap`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import sys

import numpy as np

from migdal import gap, grid, log as mlog, memory, model, radial
from migdal.freq import FreqGrid


@dataclass
class PointResult:
    """One density point from a Migdal scan.

    Densities are reported both in the public input unit ``cm^-3`` and in the
    internal radial unit ``Bohr^-3``.  ``lam`` is the leading gap eigenvalue
    returned by :func:`migdal.gap.solve_gap`.
    """

    n_cm3: float
    lam: float
    gap_converged: bool
    mu: float
    mu0: float
    n_bohr3: float
    n_re: float
    nk: int
    nq: int
    gap: gap.GapResult
    temperature_K: float | None = None
    scf_converged: bool | None = None
    scf_cycles: int | None = None
    scf_res_sigma: float | None = None
    scf_dn: float | None = None
    denom: radial.DenomMetrics | None = None
    pi: object | None = None
    fs_pin_shift: float | None = None
    fs_pin_residual: float | None = None
    mu_abs: float | None = None
    timings: dict[str, dict[str, int | float]] = field(default_factory=dict)


def _density_label(densities):
    try:
        vals = model._density_values(densities)
    except Exception:
        return str(densities)
    if len(vals) == 1:
        return "%.6e cm^-3" % float(vals[0])
    return "%d points from %.6e to %.6e cm^-3" % (
        len(vals), float(vals[0]), float(vals[-1])
    )


def _validate_gap_k_tail(n_tail):
    n_tail = int(n_tail)
    if n_tail != 0:
        raise ValueError(
            "kgrid_n_tail extends a separate Pi integration grid and is only "
            "valid for scGW via n_tail; G0W0/GW0 gap grids require "
            "kgrid_n_tail=0."
        )
    return n_tail


def _reject_removed_gap_kwargs(kwargs):
    if "cache_mb" in kwargs:
        raise ValueError("cache_mb was removed; use max_memory instead.")


def _validate_n_2kf_min(n_2kf_min):
    n_2kf_min = int(n_2kf_min)
    if n_2kf_min < 0:
        raise ValueError("n_2kf_min must be non-negative.")
    return n_2kf_min


def _validate_driver_q_singular_head_policy(policy):
    policy = radial._validate_q_singular_head_policy(policy)
    if policy == "provided":
        raise ValueError(
            "q_singular_head_policy='provided' requires a supplied screened "
            "singular coefficient and is only supported by the lower-level "
            "radial W builders. Radial drivers support 'pi_head' and 'fit_w'."
        )
    return policy


def _validate_pi0_mode(mode):
    key = str(mode).strip().replace("-", "_").lower()
    aliases = {
        "finite": "finite_T",
        "finite_t": "finite_T",
        "finite_temperature": "finite_T",
        "ft": "finite_T",
        "0k": "0K",
        "0_k": "0K",
        "t0": "0K",
        "zero": "0K",
        "zero_t": "0K",
        "zero_temperature": "0K",
        "gg": "GG",
    }
    if key not in aliases:
        raise ValueError(
            "unknown pi0_mode %r; expected one of finite_T, 0K, GG" % mode
        )
    return aliases[key]


def _beta_tau_index(ctx):
    ts = np.asarray(ctx.ts_b, dtype=float)
    beta_index = np.empty(len(ts), dtype=int)
    for it, tau in enumerate(ts):
        beta_index[it] = int(np.argmin(np.abs(ts - (ctx.beta - tau))))
    return beta_index


def _debug_grid(drv, n_cm3, data):
    log = mlog.new_logger(drv)
    q_grid = np.asarray(data["q_grid"], dtype=float)
    k_key = "k_grid" if "k_grid" in data else "k_scf"
    k_grid = np.asarray(data[k_key], dtype=float)
    info = data.get("grid_info", {})
    rows = [
        ("density", "%.6e cm^-3" % n_cm3),
        ("k_F", "%.6e Bohr^-1" % data["k_F"]),
        ("kmax", "%.6e Bohr^-1" % data["kmax"]),
        ("qmax", "%.6e Bohr^-1" % np.max(q_grid)),
        ("nk / nq / nw", "%d / %d / %d" % (len(k_grid), len(q_grid), drv.ctx.nw)),
        ("mu0", "%.6e Ha" % data["mu0"]),
        ("mu", "%.6e Ha" % data["mu"]),
        ("wcut", "%.6e Ha" % data["wcut"]),
        ("particle number error", "%.3e" % abs(data["n_re"] - data["n_ref"])),
    ]
    if "k_pi" in data:
        rows.append(("k_pi max", "%.6e Bohr^-1" % np.max(data["k_pi"])))
    if "dk_shell_min" in info:
        rows.append(("minimum shell spacing", "%.3e" % info["dk_shell_min"]))
    if data.get("denom") is not None:
        denom = data["denom"]
        rows.extend(
            (
                ("denominator regularizations", "%d" % denom.count),
                ("minimum |denominator|", "%.3e" % denom.min_abs),
                ("maximum |W|", "%.3e" % denom.max_abs_W),
            )
        )
        rows.extend(_singular_fit_rows(denom))
    mlog.kv_table(log, "Setup", rows, level=mlog.INFO)


def _singular_fit_rows(denom):
    fit = getattr(denom, "singular_fit", None)
    if fit is None:
        return ()
    fit_label = "singular direct-fit"
    if fit.coefficient_source != "fit_w":
        fit_label = "singular unused direct-fit"
    return (
        ("singular head source", fit.coefficient_source),
        ("%s rel residual" % fit_label, "%.3e" % fit.max_rel_residual),
        ("%s abs residual" % fit_label, "%.3e" % fit.max_abs_residual),
        ("%s worst v" % fit_label, "%.6e" % fit.worst_v),
        ("singular head min |denom|", "%.3e" % fit.head_min_abs_denom),
        ("singular head regularizations", "%d" % fit.head_regularized_count),
        ("singular |A_W| max", "%.3e" % fit.max_abs_w_coeff),
        ("%s |A_W| max" % fit_label, "%.3e" % fit.max_abs_fit_w_coeff),
        ("singular |A_dyn| max", "%.3e" % fit.max_abs_dynamic_coeff),
    )


def _debug_result(drv, row):
    log = mlog.new_logger(drv)
    rows = [
        ("density", "%.6e cm^-3" % row.n_cm3),
        ("lambda", "%.12e" % row.lam),
        ("gap converged", mlog.format_bool(row.gap_converged)),
        ("gap Davidson cycles", "%d" % row.gap.davidson_cycles),
        ("gap matvecs", "%d" % row.gap.matvecs),
        ("mu", "%.12e Ha" % row.mu),
    ]
    if row.scf_converged is not None:
        rows.extend(
            (
                ("SCF converged", mlog.format_bool(row.scf_converged)),
                ("SCF cycles", "%d" % row.scf_cycles),
                ("SCF residual", "%.3e" % row.scf_res_sigma),
                ("SCF density error", "%.3e" % row.scf_dn),
            )
        )
        if not row.scf_converged:
            log.warn(
                "SCF did not converge for n=%.6e cm^-3: cycles=%s res=%.3e dn=%.3e",
                row.n_cm3,
                row.scf_cycles,
                row.scf_res_sigma,
                row.scf_dn,
            )
    if not row.gap_converged:
        log.warn(
            "Gap Davidson did not converge for n=%.6e cm^-3: cycles=%s matvecs=%s",
            row.n_cm3,
            row.gap.davidson_cycles,
            row.gap.matvecs,
        )
    mlog.kv_table(log, "Point result", rows, level=mlog.NOTE)


def _debug_scan_summary(drv, rows):
    log = mlog.new_logger(drv)
    if not rows or not mlog.enabled(log, mlog.NOTE):
        return
    mlog.section(log, "Scan summary", level=mlog.NOTE)
    mlog.emit(
        log,
        mlog.NOTE,
        "  %5s %14s %14s %6s %6s %6s %6s %5s %5s",
        "point",
        "n_cm^-3",
        "lambda",
        "gap",
        "scf",
        "scfcyc",
        "gapcyc",
        "nk",
        "nq",
    )
    for idx, row in enumerate(rows, start=1):
        scf_flag = "n/a" if row.scf_converged is None else (
            "yes" if row.scf_converged else "no"
        )
        scf_cycles = -1 if row.scf_cycles is None else int(row.scf_cycles)
        mlog.emit(
            log,
            mlog.NOTE,
            "  %5d %14.6e %14.6e %6s %6s %6d %6d %5d %5d",
            idx,
            row.n_cm3,
            row.lam,
            "yes" if row.gap_converged else "no",
            scf_flag,
            scf_cycles,
            row.gap.davidson_cycles,
            row.nk,
            row.nq,
        )


def _scan_scf_continuation(drv, densities=None, restart=False,
                           continuation_max_dex=None,
                           return_intermediate=False):
    """Run the shared GW0/scGW density-continuation scan."""
    if drv.ctx is None:
        drv.build()
    densities = drv.densities if densities is None else densities
    requested_max_dex = (
        drv.scf_continuation_max_dex
        if continuation_max_dex is None else continuation_max_dex
    )
    max_dex = requested_max_dex if restart else 0.0
    path, keep = drv._scan_path(densities, max_step_dex=max_dex)
    mlog.kv_table(
        mlog.new_logger(drv),
        "Density scan",
        (
            ("requested points", "%d" % len(drv._density_values(densities))),
            ("continuation path points", "%d" % len(path)),
            ("restart", mlog.format_bool(restart)),
            ("max step", "%.3g dex" % max_dex),
        ),
        level=mlog.INFO,
    )
    rows = []
    kept_rows = []
    sigma0 = None
    mu0 = None
    fs_pin_shift0 = None
    old_k = None
    for idx, (n_cm3, keep_row) in enumerate(zip(path, keep), start=1):
        mlog.emit(
            mlog.new_logger(drv),
            mlog.NOTE,
            "Density scan point %d/%d: n = %.6e cm^-3 keep=%s",
            idx,
            len(path),
            n_cm3,
            bool(keep_row),
        )
        row = drv.kernel(
            float(n_cm3),
            sigma0=sigma0 if restart else None,
            mu0=mu0 if restart else None,
            sigma0_k=old_k if restart else None,
            fs_pin_shift0=fs_pin_shift0 if restart else None,
        )
        rows.append(row)
        if keep_row:
            kept_rows.append(row)
        sres = drv._last_scf
        if restart and sres.converged:
            sigma0 = sres.sigma
            mu0 = sres.mu
            fs_pin_shift0 = sres.extra.get("fs_pin_shift")
            old_k = drv._last_k_grid
        elif restart:
            sigma0 = None
            mu0 = None
            fs_pin_shift0 = None
            old_k = None
    drv.results = rows if return_intermediate else kept_rows
    mlog.kv_table(
        mlog.new_logger(drv),
        "Continuation scan result",
        (
            ("returned points", "%d" % len(drv.results)),
            ("intermediate points", "%d" % len(rows)),
        ),
        level=mlog.INFO,
    )
    _debug_scan_summary(drv, drv.results)
    return drv.results


def _temperature_values(temperatures):
    try:
        vals = np.asarray(list(temperatures), dtype=float)
    except TypeError:
        vals = np.asarray([temperatures], dtype=float)
    if (
        vals.ndim != 1
        or vals.size == 0
        or not np.all(np.isfinite(vals))
        or np.any(vals <= 0.0)
    ):
        raise ValueError(
            "temperatures must be a non-empty sequence of finite positives."
        )
    return vals


def _scan_temp_density(drv, n_cm3):
    if n_cm3 is not None:
        return drv._single_density(n_cm3)
    vals = drv._density_values(drv.densities)
    if len(vals) != 1:
        raise ValueError(
            "scan_temp requires one fixed density; pass n_cm3 or construct "
            "the driver with a single density."
        )
    return float(vals[0])


def _prepare_temp_point(drv, temperature, ir_file_for_temperature):
    drv.temperature = float(temperature)
    if ir_file_for_temperature is not None:
        ir_file = ir_file_for_temperature(float(temperature))
        drv.ir_file = None if ir_file is None else str(ir_file)
    drv.ctx = None
    return drv.ir_file


def _temp_scan_event(point, temperature, ir_file, restart_used,
                     retry_without_restart, restart_updated, source_temperature):
    return {
        "n_cm3": point.n_cm3,
        "temperature_K": float(temperature),
        "ir_file": "" if ir_file is None else str(ir_file),
        "restart_used": int(bool(restart_used)),
        "restart_retry_without_restart": int(bool(retry_without_restart)),
        "restart_updated": int(bool(restart_updated)),
        "restart_source_temperature_K": (
            "" if source_temperature is None else float(source_temperature)
        ),
        "lambda": point.lam,
        "gap_converged": int(point.gap_converged),
        "scf_converged": (
            -1 if point.scf_converged is None else int(point.scf_converged)
        ),
        "scf_cycles": -1 if point.scf_cycles is None else int(point.scf_cycles),
    }


def _scan_temp_continuation(
    drv,
    temperatures,
    n_cm3=None,
    restart=False,
    retry_without_restart=True,
    ir_file_for_temperature=None,
    return_restart_log=False,
    stop_callback=None,
):
    """Run a fixed-density temperature scan with optional SCF continuation."""
    temps = _temperature_values(temperatures)
    n_cm3 = _scan_temp_density(drv, n_cm3)
    can_restart = hasattr(drv, "last_scf_restart_state")
    restart = bool(restart and can_restart)
    retry_without_restart = bool(retry_without_restart and can_restart)
    mlog.kv_table(
        mlog.new_logger(drv),
        "Temperature scan",
        (
            ("density", "%.6e cm^-3" % n_cm3),
            ("temperature points", "%d" % len(temps)),
            ("restart", mlog.format_bool(restart)),
            ("retry without restart", mlog.format_bool(retry_without_restart)),
        ),
        level=mlog.INFO,
    )

    rows = []
    restart_log = []
    restart_state = None
    restart_state_temperature = None
    log = mlog.new_logger(drv)
    for idx, temperature in enumerate(temps, start=1):
        ir_file = _prepare_temp_point(drv, temperature, ir_file_for_temperature)
        restart_used = bool(restart and restart_state is not None)
        source_temperature = restart_state_temperature if restart_used else None
        mlog.emit(
            log,
            mlog.NOTE,
            "Temperature scan point %d/%d: T = %.6g K restart=%s",
            idx,
            len(temps),
            float(temperature),
            mlog.format_bool(restart_used),
        )

        retried = False
        if restart_used:
            row = drv.kernel(float(n_cm3), restart_state=restart_state)
            if (
                row.scf_converged is False
                and retry_without_restart
            ):
                retried = True
                mlog.warning(
                    log,
                    "Temperature restart failed at n=%.6e cm^-3 T=%.6g K; "
                    "retrying without restart.",
                    n_cm3,
                    float(temperature),
                )
                drv.ctx = None
                row = drv.kernel(float(n_cm3))
        else:
            row = drv.kernel(float(n_cm3))

        if row.temperature_K is None:
            row.temperature_K = float(temperature)
        restart_updated = False
        if can_restart and row.scf_converged:
            restart_state = drv.last_scf_restart_state()
            restart_state_temperature = float(temperature)
            restart_updated = True
        event = _temp_scan_event(
            row,
            temperature,
            ir_file,
            restart_used,
            retried,
            restart_updated,
            source_temperature,
        )
        rows.append(row)
        restart_log.append(event)
        if stop_callback is not None and stop_callback(row, event):
            break

    drv.results = rows
    _debug_scan_summary(drv, rows)
    if return_restart_log:
        return rows, restart_log
    return rows


class G0W0:
    """G0W0 pairing driver for the STO radial one-band model.

    The driver uses atomic units internally.  Public densities are spin-summed
    electron densities in cm^-3; they are converted to Bohr^-3 before building
    the radial k grid.  The fixed interaction is the RPA ``W0`` built from the
    selected ``pi0_mode`` polarization and the phonon dielectric factor
    ``Vph(q, i*nu)``.  The default ``pi0_mode='finite_T'`` uses the
    finite-temperature Lindhard function; ``'0K'`` uses the zero-temperature
    Lindhard function, and ``'GG'`` builds the bubble numerically from
    ``G0(tau)G0(beta-tau)``.

    Notes
    -----
    ``diag_average="auto"`` uses diagonal cell averages in ``q_singular`` mode.
    When ``q_interp_mode='legacy'``, it means a diagonal cell average for
    no-truncation solves and no diagonal cell average for explicitly truncated
    solves.
    The q grid is built to cover the radial convolution domain up to
    ``2*kmax``; `_check_q_coverage` raises if a custom parameter set violates
    this requirement.
    """

    def __init__(
        self,
        params: model.STOParams | None = None,
        densities="default",
        temperature=1.0,
        lamb=1.0e9,
        ir_tol=1.0e-14,
        ir_file=None,
        save_ir=True,
        cache_transforms=True,
        nq=500,
        n_2kf_min=101,
        truncate=False,
        diag_average="auto",
        diag_ncell=12,
        diag_average_mode="auto",
        kmax_factor=4.0,
        kmax_lo_factor=2.0,
        qmax_factor=2.0,
        kgrid_n_pad=5,
        kgrid_n_left=200,
        kgrid_n_right=200,
        kgrid_n_tail=0,
        kgrid_n_shell_min=50,
        kgrid_n_shell_max=5000,
        wcut_factor=1.0,
        wcut_min_ev=0.0,
        denom_policy="sign_regularize",
        denom_min=1.0e-10,
        pi0_mode="finite_T",
        q_interp_mode="q_singular",
        q_singular_head_policy="pi_head",
        n0_check_rtol=1.0e-4,
        n0_check_atol=1.0e-12,
        max_memory=memory.DEFAULT_MAX_MEMORY,
        verbose=4,
        stdout=sys.stdout,
        **gap_kwargs,
    ):
        _reject_removed_gap_kwargs(gap_kwargs)
        self.params = model.STOParams() if params is None else params
        self.densities = densities
        self.temperature = temperature
        self.lamb = lamb
        self.ir_tol = ir_tol
        self.ir_file = ir_file
        self.save_ir = save_ir
        self.cache_transforms = bool(cache_transforms)
        self.nq = nq
        self.n_2kf_min = _validate_n_2kf_min(n_2kf_min)
        self.truncate = bool(truncate)
        self.diag_average = diag_average
        self.diag_ncell = diag_ncell
        self.diag_average_mode = radial._validate_diag_average_mode(
            diag_average_mode)
        self.kmax_factor = kmax_factor
        self.kmax_lo_factor = kmax_lo_factor
        self.qmax_factor = qmax_factor
        self.kgrid_n_pad = kgrid_n_pad
        self.kgrid_n_left = kgrid_n_left
        self.kgrid_n_right = kgrid_n_right
        self.kgrid_n_tail = _validate_gap_k_tail(kgrid_n_tail)
        self.kgrid_n_shell_min = kgrid_n_shell_min
        self.kgrid_n_shell_max = kgrid_n_shell_max
        self.wcut_factor = wcut_factor
        self.wcut_min_ev = wcut_min_ev
        self.denom_policy = denom_policy
        self.denom_min = denom_min
        self.pi0_mode = _validate_pi0_mode(pi0_mode)
        self.q_interp_mode = radial._validate_w_interp_mode(q_interp_mode)
        self.q_singular_head_policy = _validate_driver_q_singular_head_policy(
            q_singular_head_policy)
        self.n0_check_rtol = n0_check_rtol
        self.n0_check_atol = n0_check_atol
        self.max_memory = float(max_memory)
        if not np.isfinite(self.max_memory) or self.max_memory < 0.0:
            raise ValueError("max_memory must be a non-negative finite MB value.")
        self.verbose = verbose
        self.stdout = stdout
        self.gap_kwargs = gap_kwargs
        self.ctx = None
        self.results = []
        self._last_point_data = None
        self._last_W = None
        self._last_diag_cells = None
        self._last_k_grid = None
        self._last_k_wts = None
        self._timings = None

    def set(self, **kwargs):
        """Set driver attributes in PySCF style and return ``self``."""
        _reject_removed_gap_kwargs(kwargs)
        for key, val in kwargs.items():
            if key == "kgrid_n_tail":
                val = _validate_gap_k_tail(val)
            elif key == "n_2kf_min":
                val = _validate_n_2kf_min(val)
            elif key == "q_interp_mode":
                val = radial._validate_w_interp_mode(val)
            elif key == "q_singular_head_policy":
                val = _validate_driver_q_singular_head_policy(val)
            elif key == "pi0_mode":
                val = _validate_pi0_mode(val)
            elif key == "diag_average_mode":
                val = radial._validate_diag_average_mode(val)
            elif key == "max_memory":
                val = float(val)
                if not np.isfinite(val) or val < 0.0:
                    raise ValueError(
                        "max_memory must be a non-negative finite MB value."
                    )
            setattr(self, key, val)
        return self

    def build(self):
        """Build or load the sparse-IR frequency context for this driver."""
        beta = self.params.beta_ha(self.temperature)
        log = mlog.new_logger(self)
        mlog.banner(
            log,
            "Migdal %s radial calculation" % self.__class__.__name__,
            level=mlog.INFO,
        )
        self.dump_flags(log)
        mlog.kv_table(
            log,
            "Sparse-IR setup",
            (
                ("temperature", "%.6g K" % self.temperature),
                ("beta", "%.12g Ha^-1" % beta),
                ("lambda", "%.12g" % self.lamb),
                ("IR file", self.ir_file),
                ("save IR cache", mlog.format_bool(self.save_ir)),
            ),
            level=mlog.INFO,
        )
        self.ctx = FreqGrid(
            beta,
            lamb=self.lamb,
            tol=self.ir_tol,
            ir_file=self.ir_file,
            save_ir=self.save_ir,
            cache_transforms=self.cache_transforms,
            verbose=self.verbose,
            stdout=self.stdout,
        )
        mlog.kv_table(
            log,
            "Sparse-IR grid",
            (
                ("fermion frequencies", "%d" % self.ctx.nw),
                ("boson frequencies", "%d" % self.ctx.nv),
                ("fermion tau points", "%d" % self.ctx.nt_f),
                ("boson tau points", "%d" % self.ctx.nt_b),
                ("wmax", "%.6e Ha" % self.ctx.wmax),
            ),
            level=mlog.INFO,
        )
        return self

    def dump_flags(self, verbose=None):
        """Print the current driver and gap-solver settings."""
        log = mlog.new_logger(self, verbose)
        mlog.kv_table(
            log,
            "Driver flags",
            self._flag_rows(),
            level=mlog.INFO,
        )
        return self

    def _flag_rows(self):
        rows = [
            ("method", self.__class__.__name__),
            ("densities", _density_label(self.densities)),
            ("temperature", "%.6g K" % self.temperature),
            ("nq", "%d" % self.nq),
            ("2kF q-grid min points", "%d" % self.n_2kf_min),
            ("truncate gap frequencies", mlog.format_bool(self.truncate)),
            ("diagonal cell average", self.diag_average),
            ("diagonal cells", "%d" % self.diag_ncell),
            ("diagonal cell mode", self.diag_average_mode),
            ("kmax factor", "%.6g" % self.kmax_factor),
            ("kmax low-density factor", "%.6g" % self.kmax_lo_factor),
            ("qmax factor", "%.6g" % self.qmax_factor),
            ("k grid pad/left/right/tail", "%d / %d / %d / %d" % (
                self.kgrid_n_pad,
                self.kgrid_n_left,
                self.kgrid_n_right,
                self.kgrid_n_tail,
            )),
            ("k shell min/max", "%d / %d" % (
                self.kgrid_n_shell_min,
                self.kgrid_n_shell_max,
            )),
            ("wcut factor", "%.6g" % self.wcut_factor),
            ("wcut floor", "%.6g eV" % self.wcut_min_ev),
            ("denominator policy", self.denom_policy),
            ("denominator minimum", "%.3e" % self.denom_min),
            ("q interpolation mode", self.q_interp_mode),
            ("n0 check rtol/atol", "%.3e / %.3e" % (
                self.n0_check_rtol,
                self.n0_check_atol,
            )),
            ("max memory", "%.6g MB" % self.max_memory),
        ]
        if self._uses_fixed_W0_pi0_mode():
            rows.insert(-2, ("W0 polarization", self.pi0_mode))
        if self.q_interp_mode == "q_singular":
            rows.append(("q singular head policy", self.q_singular_head_policy))
        rows.extend(self._gap_flag_rows())
        rows.extend(self._extra_flag_rows())
        return rows

    def _gap_flag_rows(self):
        defaults = gap.GapConfig()
        opts = dict(self.gap_kwargs)
        return [
            ("gap preconditioner", opts.get("precond", defaults.precond)),
            ("gap cache", opts.get("cache", defaults.cache)),
            ("Davidson tolerance", "%.3e" % opts.get(
                "davidson_tol", defaults.davidson_tol)),
            ("Davidson max cycle", "%d" % opts.get(
                "davidson_max_cycle", defaults.davidson_max_cycle)),
            ("Davidson max space", "%d" % opts.get(
                "davidson_max_space", defaults.davidson_max_space)),
            ("Davidson roots", "%d" % opts.get("nroots", defaults.nroots)),
        ]

    def _extra_flag_rows(self):
        return []

    def _uses_fixed_W0_pi0_mode(self):
        return True

    def _scan_path(self, densities, max_step_dex=0.0):
        return model.refine_density_path(
            self._density_values(densities), max_step_dex=max_step_dex)

    def run(self, *args, **kwargs):
        """Run :meth:`scan` after applying optional keyword updates."""
        restart = kwargs.pop("restart", None)
        if kwargs:
            self.set(**kwargs)
        if restart is None:
            self.scan(*args)
        else:
            self.scan(*args, restart=restart)
        return self

    def _densities(self):
        return self._density_values(self.densities)

    def _density_values(self, densities):
        return model._density_values(densities)

    def _single_density(self, n_cm3):
        return model._single_density(self.densities, n_cm3)

    def _eps(self, k, mu):
        return self.params.band_a_ha_bohr2 * np.asarray(k) ** 2 - mu

    def _static_coulomb_head(self):
        return 4.0 * np.pi / float(self.params.eps_inf)

    def _vtot_head(self, ctx):
        vph0 = model.get_Vph_q(np.array([0.0]), ctx.vs, self.params)[0]
        return self._static_coulomb_head() * vph0

    def _check_q_coverage(self, q_grid, k_upper, label):
        q_max = float(np.max(q_grid))
        required = 2.0 * float(k_upper)
        if q_max < required * (1.0 - 1.0e-10):
            raise ValueError(
                "%s q grid does not cover radial convolution: q_max=%s required=%s"
                % (label, q_max, required)
            )

    def _build_W0(self, a, mu, mu0, k_grid, k_wts, G0, Vph, q_grid):
        p = self.params
        ctx = self.ctx
        pi0_mode = _validate_pi0_mode(self.pi0_mode)
        common = dict(
            q0_treat=False,
            tol=1.0e-10,
            denom_policy=self.denom_policy,
            denom_min=self.denom_min,
            q_interp_mode=self.q_interp_mode,
            q_singular_head_policy=self.q_singular_head_policy,
            vtot_head=(
                self._vtot_head(ctx) if self.q_interp_mode == "q_singular"
                else None
            ),
            return_metrics=True,
            warning_log=mlog.new_logger(self),
        )
        if pi0_mode == "finite_T":
            return radial.get_W0(
                ctx,
                model.get_Pi0_finite_T,
                model.get_Vc,
                Vph,
                p.eps_inf,
                q_grid,
                kwargs_get_Pi={"beta": ctx.beta, "mu": mu, "a": a},
                **common,
            )
        if pi0_mode == "0K":
            return radial.get_W0(
                ctx,
                model.get_Pi,
                model.get_Vc,
                Vph,
                p.eps_inf,
                q_grid,
                kwargs_get_Pi={"a": a, "mu": mu0},
                **common,
            )

        G_tau = ctx.w2t_f2b(G0)
        Pi_tau = radial.get_Pi(
            None,
            ctx,
            G_tau.real,
            k_grid,
            k_wts,
            q_grid,
            beta_index=_beta_tau_index(ctx),
            q_zero_tol=1.0e-10,
        )
        Pi_w = ctx.t2w_b(Pi_tau)
        Pi = radial.symm_Pi(ctx, Pi_w).real
        # Match radial.get_W0: q_singular needs the Coulomb tail at every
        # positive production q point, however small.
        vc_tol = (
            np.nextafter(0.0, 1.0)
            if self.q_interp_mode == "q_singular"
            else 1.0e-10
        )
        Vc = model.get_Vc(q_grid, p.eps_inf, q0_treat=False, tol=vc_tol)
        Vtot = Vc[:, None] * Vph
        return radial.get_W_from_Pi(
            ctx,
            Pi,
            q_grid,
            Vc,
            Vtot,
            denom_policy=self.denom_policy,
            denom_min=self.denom_min,
            q_interp_mode=self.q_interp_mode,
            static_singular_coeff=4.0 * np.pi / float(p.eps_inf),
            q_singular_head_policy=self.q_singular_head_policy,
            vtot_head=(
                self._vtot_head(ctx) if self.q_interp_mode == "q_singular"
                else None
            ),
            return_metrics=True,
            warning_log=mlog.new_logger(self),
        )

    def _setup_point(self, n_cm3):
        if self.ctx is None:
            self.build()
        p = self.params
        ctx = self.ctx
        a = p.band_a_ha_bohr2
        n_bohr3 = model.n_cm3_to_bohr3(n_cm3)
        mu0 = model.get_mu(n_bohr3, beta=np.inf, a=a)
        mu = model.get_mu(n_bohr3, beta=ctx.beta, a=a)
        k_F = np.sqrt(mu0 / a)
        k_ph = np.sqrt((max(mu0, 0.0) + model.phonon_grid_ha(p)) / a)
        kmax = max(self.kmax_factor * k_F, self.kmax_lo_factor * k_ph)
        qmax = self.qmax_factor * kmax
        q_grid, _ = grid.get_q_grid(
            0.5 * qmax,
            k_F=k_F,
            nq=self.nq,
            frac_log=0.35,
            n_2kf_min=self.n_2kf_min,
        )
        k_grid, k_wts, grid_info = grid.get_k_grid(
            k_F, 1.0 / ctx.beta, q_grid, kmax,
            lambda k: self._eps(k, mu0),
            n_pad=self.kgrid_n_pad,
            n_left=self.kgrid_n_left,
            n_right=self.kgrid_n_right,
            n_tail=self.kgrid_n_tail,
            n_shell_min=self.kgrid_n_shell_min,
            n_shell_max=self.kgrid_n_shell_max,
        )
        self._check_q_coverage(q_grid, kmax, "G0W0")
        e_k = self._eps(k_grid, 0.0)
        G0 = ctx.get_G0(e_k=e_k, mu=mu)
        rdm1 = ctx.make_rdm1(G=G0, e_k=e_k, mu=mu)
        n_re = np.dot(k_grid**2 * k_wts, rdm1).real / (2.0 * np.pi**2)
        n_ref = model.get_n(mu, ctx.beta, a, tol=1.0e-15)
        err = abs(n_re - n_ref)
        if err > self.n0_check_atol + self.n0_check_rtol * abs(n_ref):
            raise ValueError(
                "initial particle-number check failed: n_re=%s n_ref=%s diff=%s"
                % (n_re, n_ref, err)
            )
        screen_t0 = mlog.wall_clock() if self._timings is not None else None
        Vph = model.get_Vph_q(q_grid, ctx.vs, p)
        W, denom = self._build_W0(a, mu, mu0, k_grid, k_wts, G0, Vph, q_grid)
        if self.q_interp_mode == "q_singular":
            diag_average = model.bool_auto(self.diag_average, default=True)
            if not diag_average:
                raise ValueError("q_singular mode requires diagonal cell averages.")
        else:
            diag_average = model.bool_auto(self.diag_average, default=not self.truncate)
        diag_cells = None
        if diag_average:
            diag_cells = radial.get_diag_cells(
                None, W, k_grid, k_wts, kmax, self.diag_ncell,
                cell_mode=self.diag_average_mode)
        if self._timings is not None:
            mlog.record_wall(self._timings, "screening", screen_t0)
        wcut = max(mu0, mu) * self.wcut_factor
        wcut = max(wcut, self.wcut_min_ev * model.EV_TO_HA)
        return {
            "n_bohr3": n_bohr3,
            "mu0": mu0,
            "mu": mu,
            "k_F": k_F,
            "kmax": kmax,
            "q_grid": q_grid,
            "k_grid": k_grid,
            "k_wts": k_wts,
            "grid_info": grid_info,
            "e_k": e_k,
            "G0": G0,
            "W": W,
            "denom": denom,
            "diag_cells": diag_cells,
            "wcut": wcut,
            "n_re": n_re,
            "n_ref": n_ref,
        }

    def _gap_config(self, wcut):
        kwargs = dict(self.gap_kwargs)
        kwargs.setdefault("truncate", self.truncate)
        kwargs.setdefault("wcut", wcut)
        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("stdout", self.stdout)
        kwargs.setdefault("max_memory", self.max_memory)
        return gap.GapConfig(**kwargs)

    def kernel(self, n_cm3=None):
        """Run one density point and return a :class:`PointResult`."""
        log = mlog.new_logger(self)
        self._timings = mlog.new_timings(log)
        init_t0 = mlog.wall_clock() if self._timings is not None else None
        if self.ctx is None:
            self.build()
        n_cm3 = self._single_density(n_cm3)
        mlog.section(
            log,
            "Density point: n = %.6e cm^-3" % n_cm3,
            level=mlog.INFO,
        )
        data = self._setup_point(n_cm3)
        _debug_grid(self, n_cm3, data)
        if self._timings is not None:
            mlog.record_wall(self._timings, "initialization", init_t0)
        mlog.section(log, "Gap solve", level=mlog.INFO)
        gap_t0 = mlog.wall_clock() if self._timings is not None else None
        gres = gap.solve_gap(
            self.ctx,
            data["k_grid"],
            data["k_wts"],
            data["G0"],
            data["W"],
            config=self._gap_config(data["wcut"]),
            diag_cells=data["diag_cells"],
            mu_scale=data["mu0"],
        )
        if self._timings is not None:
            mlog.record_wall(self._timings, "gap_solver", gap_t0)
        row = PointResult(
            n_cm3=n_cm3,
            lam=gres.lam,
            gap_converged=gres.converged,
            mu=data["mu"],
            mu0=data["mu0"],
            n_bohr3=data["n_ref"],
            n_re=data["n_re"],
            nk=len(data["k_grid"]),
            nq=len(data["q_grid"]),
            gap=gres,
            temperature_K=float(self.temperature),
            denom=data["denom"],
            timings=mlog.snapshot_timings(self._timings),
        )
        self._last_point_data = data
        self._last_W = data["W"]
        self._last_diag_cells = data["diag_cells"]
        self._last_k_grid = data["k_grid"]
        self._last_k_wts = data["k_wts"]
        self.results = [row]
        mlog.timing_summary(log, self._timings)
        _debug_result(self, row)
        return row

    def scan(self, densities=None, restart=False):
        """Run a density scan. ``restart`` is accepted for API symmetry."""
        if self.ctx is None:
            self.build()
        densities = self.densities if densities is None else densities
        dens_values = self._density_values(densities)
        mlog.kv_table(
            mlog.new_logger(self),
            "Density scan",
            (
                ("requested points", "%d" % len(dens_values)),
                ("restart requested", mlog.format_bool(restart)),
            ),
            level=mlog.INFO,
        )
        rows = []
        for idx, n_cm3 in enumerate(dens_values, start=1):
            mlog.emit(
                mlog.new_logger(self),
                mlog.NOTE,
                "Density scan point %d/%d: n = %.6e cm^-3",
                idx,
                len(dens_values),
                n_cm3,
            )
            rows.append(self.kernel(float(n_cm3)))
        self.results = rows
        _debug_scan_summary(self, rows)
        return rows

    def scan_temp(
        self,
        temperatures,
        n_cm3=None,
        *,
        restart=False,
        retry_without_restart=False,
        ir_file_for_temperature=None,
        return_restart_log=False,
        stop_callback=None,
    ):
        """Run a fixed-density temperature scan without SCF continuation."""
        return _scan_temp_continuation(
            self,
            temperatures,
            n_cm3=n_cm3,
            restart=False,
            retry_without_restart=False,
            ir_file_for_temperature=ir_file_for_temperature,
            return_restart_log=return_restart_log,
            stop_callback=stop_callback,
        )

    def write(self, filename):
        """Write density, lambda, convergence, and diagnostics as CSV."""
        if not self.results:
            raise ValueError("no results available; call kernel() first.")
        include_fs_pin = any(
            r.fs_pin_shift is not None
            or r.fs_pin_residual is not None
            or r.mu_abs is not None
            for r in self.results
        )
        include_pi = any(getattr(r, "pi", None) is not None for r in self.results)
        include_singular = any(
            getattr(getattr(r, "denom", None), "singular_fit", None) is not None
            for r in self.results
        )
        header = [
            "method",
            "temperature_K",
            "n_cm3",
            "lambda",
            "gap_converged",
            "scf_converged",
            "scf_cycles",
            "scf_res_sigma",
            "scf_dn",
            "denom_policy",
            "denom_reg_count",
            "denom_undetermined_count",
            "denom_min_abs",
            "denom_max_abs_W",
            "denom_worst_q",
            "denom_worst_v",
            "denom_worst_old",
            "denom_worst_new",
            "mu_Ha",
            "mu0_Ha",
            "n_bohr3",
            "n_re",
            "gap_cycles",
            "gap_matvecs",
            "gap_n_active_w",
            "gap_used_min_pair",
        ]
        if include_fs_pin:
            header.extend(["fs_pin_shift", "fs_pin_residual", "mu_abs"])
        if include_pi:
            header.extend(
                [
                    "pi_g_tau_imag",
                    "pi_w_imag",
                    "pi_symm_err",
                    "pi_positive_count",
                    "pi_max_positive",
                ]
            )
        if include_singular:
            header.extend(
                [
                    "singular_head_source",
                    "singular_direct_fit_rel_residual",
                    "singular_direct_fit_abs_residual",
                    "singular_direct_fit_worst_v",
                    "singular_head_min_abs_denom",
                    "singular_head_reg_count",
                    "singular_max_abs_A_W",
                    "singular_direct_fit_max_abs_A_W",
                    "singular_max_abs_A_dyn",
                ]
            )
        header.extend(["nk", "nq"])

        def opt(value):
            return np.nan if value is None else value

        rows = []
        for r in self.results:
            denom = r.denom
            gap_res = r.gap
            scf_converged = -1 if r.scf_converged is None else int(r.scf_converged)
            scf_cycles = -1 if r.scf_cycles is None else int(r.scf_cycles)
            row = [
                self.__class__.__name__,
                self.temperature if r.temperature_K is None else r.temperature_K,
                r.n_cm3, r.lam, int(r.gap_converged), scf_converged,
                scf_cycles, opt(r.scf_res_sigma), opt(r.scf_dn),
                "" if denom is None else denom.policy,
                np.nan if denom is None else denom.count,
                np.nan if denom is None else denom.undetermined_count,
                np.nan if denom is None else denom.min_abs,
                np.nan if denom is None else denom.max_abs_W,
                np.nan if denom is None else denom.worst_q,
                np.nan if denom is None else denom.worst_v,
                "" if denom is None else denom.worst_old,
                "" if denom is None else denom.worst_new,
                r.mu, r.mu0, r.n_bohr3, r.n_re,
                -1 if gap_res is None else gap_res.davidson_cycles,
                -1 if gap_res is None else gap_res.matvecs,
                -1 if gap_res is None else gap_res.n_active_w,
                -1 if gap_res is None else int(gap_res.used_min_pair),
            ]
            if include_fs_pin:
                row.extend((
                    opt(r.fs_pin_shift),
                    opt(r.fs_pin_residual),
                    opt(r.mu_abs),
                ))
            if include_pi:
                pi = getattr(r, "pi", None)
                row.extend(
                    (
                        np.nan if pi is None else pi.g_tau_imag,
                        np.nan if pi is None else pi.pi_w_imag,
                        np.nan if pi is None else pi.pi_symm_err,
                        np.nan if pi is None else pi.positive_pi_count,
                        np.nan if pi is None else pi.max_positive_pi,
                    )
                )
            if include_singular:
                fit = getattr(denom, "singular_fit", None)
                row.extend(
                    (
                        "" if fit is None else fit.coefficient_source,
                        np.nan if fit is None else fit.max_rel_residual,
                        np.nan if fit is None else fit.max_abs_residual,
                        np.nan if fit is None else fit.worst_v,
                        np.nan if fit is None else fit.head_min_abs_denom,
                        np.nan if fit is None else fit.head_regularized_count,
                        np.nan if fit is None else fit.max_abs_w_coeff,
                        np.nan if fit is None else fit.max_abs_fit_w_coeff,
                        np.nan if fit is None else fit.max_abs_dynamic_coeff,
                    )
                )
            row.extend((r.nk, r.nq))
            rows.append(row)
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return self
