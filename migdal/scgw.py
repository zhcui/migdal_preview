"""Self-consistent GW radial driver for the STO one-band model.

scGW updates both the normal Green's function and the screened interaction.
Compared with GW0, each SCF map evaluates a polarization bubble on an extended
``k_pi`` grid, builds a new RPA ``W``, and then recomputes ``Sigma`` on the
active SCF grid before the shared gap solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import lib

from migdal import gap, grid, log as mlog, model, radial, scf
from migdal.g0w0 import (
    G0W0,
    PointResult,
    _debug_grid,
    _debug_result,
    _scan_scf_continuation,
    _scan_temp_continuation,
    _singular_fit_rows,
)
from migdal.gw0 import _interp_sigma, _reject_unknown_scf_kwargs


_SCGW_SCF_SET_KEYS = frozenset(
    (
        "scf_conv_tol",
        "scf_max_cycle",
        "scf_damp",
        "scf_diis_space",
        "scf_diis_start_cycle",
        "scf_continuation_max_dex",
        "scf_mu_bracket",
        "scf_fs_pin",
        "scf_high_freq_density",
        "scf_conv_rtol_nelec",
        "scf_conv_atol_nelec",
        "scf_chkfile",
        "scf_restart_chkfile",
    )
)


@dataclass
class PiMetrics:
    """Diagnostics for a scGW polarization build."""

    g_tau_imag: float = 0.0
    pi_w_imag: float = 0.0
    pi_symm_err: float = 0.0
    positive_pi_count: int = 0
    max_positive_pi: float = 0.0


_PI_IMAG_POLICIES = ("warn", "fail")
_PI_POSITIVE_POLICIES = ("warn", "fail")
_PI_POSITIVE_ACTIONS = ("keep", "zero_small")
_PI_LOW_Q_POLICIES = ("asymptotic", "none")


def _validate_choice(name, value, choices):
    value = str(value)
    if value not in choices:
        raise ValueError("unknown %s %r; expected one of %s" % (name, value, choices))
    return value


def _validate_pi_imag_policy(policy):
    return _validate_choice("pi_imag_policy", policy, _PI_IMAG_POLICIES)


def _validate_pi_positive_policy(policy):
    return _validate_choice("pi_positive_policy", policy, _PI_POSITIVE_POLICIES)


def _validate_pi_positive_action(action):
    return _validate_choice("pi_positive_action", action, _PI_POSITIVE_ACTIONS)


def _validate_pi_low_q_policy(policy):
    return _validate_choice("pi_low_q_policy", policy, _PI_LOW_Q_POLICIES)


def _array_value_key(arr):
    arr = np.ascontiguousarray(arr)
    return arr.shape, arr.dtype.str, arr.tobytes()


def _static_diag_key(W):
    mode = getattr(W, "interp_mode", None)
    if mode == "legacy":
        q_grid = getattr(W, "q", None)
        yS = getattr(W, "yS", None)
        if q_grid is None or yS is None:
            return None
        return (
            "legacy",
            _array_value_key(q_grid),
            _array_value_key(yS),
        )
    if mode != "q_singular":
        return None
    q_grid = getattr(W, "q", None)
    yS_reg = getattr(W, "yS_reg", None)
    singular_sta = getattr(W, "singular_sta", None)
    if q_grid is None or yS_reg is None or singular_sta is None:
        return None
    return (
        "q_singular",
        _array_value_key(q_grid),
        _array_value_key(yS_reg),
        _array_value_key(np.asarray(singular_sta)),
    )


def _q_singular_static_diag_key(W):
    if getattr(W, "interp_mode", None) != "q_singular":
        return None
    return _static_diag_key(W)


def _diag_cell_static_cache_key(data, ncell, cell_mode="auto"):
    cell_mode = radial._validate_diag_average_mode(cell_mode)
    geometry = data.get("diag_geometry")
    if geometry is not None and all(
        hasattr(geometry, attr)
        for attr in ("k_grid", "k_wts", "k_upper", "bounds", "xw", "ncell")
    ):
        x, w = geometry.xw
        return (
            "geometry",
            cell_mode,
            int(ncell),
            int(geometry.ncell),
            _array_value_key(geometry.k_grid),
            _array_value_key(geometry.k_wts),
            float(geometry.k_upper),
            _array_value_key(geometry.bounds),
            _array_value_key(x),
            _array_value_key(w),
        )
    return (
        "inputs",
        cell_mode,
        int(ncell),
        _array_value_key(data["k_scf"]),
        _array_value_key(data["k_wts_scf"]),
        float(data["kmax"]),
        None if geometry is None else id(geometry),
    )


class ScGW(G0W0):
    r"""Self-consistent GW driver followed by the shared radial gap solve.

    The SCF particle-number constraint defaults to the high-frequency corrected
    density ``2 f(e_k - mu) - 2 [G - G0](beta^-)``.  This convention keeps the
    analytically known noninteracting occupation outside the sparse-IR endpoint
    transform and is more stable for low-density radial grids.  Set
    ``scf_high_freq_density=False`` only for direct ``-2 G(beta^-)``
    comparisons.
    """

    def __init__(
        self,
        *args,
        n_tail=100,
        sigma_tail_policy="exchange_static",
        static_tail_xcut=60.0,
        scf_conv_tol=1.0e-6,
        scf_max_cycle=300,
        scf_damp=0.98,
        scf_diis_space=8,
        scf_diis_start_cycle=0,
        scf_continuation_max_dex=1.0,
        scf_mu_bracket="nearest_positive",
        scf_fs_pin=True,
        scf_high_freq_density=True,
        scf_conv_rtol_nelec=1.0e-4,
        scf_conv_atol_nelec=1.0e-14,
        scf_chkfile=None,
        scf_restart_chkfile=None,
        denom_policy="sign_regularize",
        denom_min=1.0e-10,
        pi_imag_tol=1.0e-5,
        pi_imag_policy="warn",
        pi_positive_tol=1.0e-12,
        pi_positive_policy="warn",
        pi_positive_action="keep",
        pi_low_q_policy="asymptotic",
        q_low_factor=2.0e-2,
        q_static_low_factor=None,
        allow_unconverged=False,
        allow_no_pi_tail=False,
        **kwargs,
    ):
        if "pi0_mode" in kwargs:
            raise ValueError(
                "pi0_mode is only used by fixed-W0 G0W0/GW0; "
                "scGW builds Pi from GG self-consistently."
            )
        _reject_unknown_scf_kwargs(kwargs)
        _reject_removed_pi_k_interp_mode(kwargs)
        kwargs.setdefault("truncate", False)
        kwargs.setdefault("diag_average", "auto")
        super().__init__(*args, denom_policy=denom_policy, denom_min=denom_min, **kwargs)
        allow_no_pi_tail = bool(allow_no_pi_tail)
        self.n_tail = _validate_n_tail(n_tail, allow_no_pi_tail)
        self.sigma_tail_policy = sigma_tail_policy
        self.static_tail_xcut = static_tail_xcut
        self.scf_conv_tol = scf_conv_tol
        self.scf_max_cycle = scf_max_cycle
        self.scf_damp = scf_damp
        self.scf_diis_space = scf_diis_space
        self.scf_diis_start_cycle = scf_diis_start_cycle
        self.scf_continuation_max_dex = scf_continuation_max_dex
        self.scf_mu_bracket = scf_mu_bracket
        self.scf_fs_pin = bool(scf_fs_pin)
        self.scf_high_freq_density = bool(scf_high_freq_density)
        self.scf_conv_rtol_nelec = scf_conv_rtol_nelec
        self.scf_conv_atol_nelec = scf_conv_atol_nelec
        self.scf_chkfile = scf_chkfile
        self.scf_restart_chkfile = scf_restart_chkfile
        self.pi_imag_tol = pi_imag_tol
        self.pi_imag_policy = _validate_pi_imag_policy(pi_imag_policy)
        self.pi_positive_tol = pi_positive_tol
        self.pi_positive_policy = _validate_pi_positive_policy(pi_positive_policy)
        self.pi_positive_action = _validate_pi_positive_action(pi_positive_action)
        self.pi_low_q_policy = _validate_pi_low_q_policy(pi_low_q_policy)
        self.q_low_factor = q_low_factor
        self.q_static_low_factor = q_static_low_factor
        self.allow_unconverged = allow_unconverged
        self.allow_no_pi_tail = allow_no_pi_tail

    def set(self, **kwargs):
        """Set driver attributes and revalidate coupled ``n_tail`` options."""
        if "pi0_mode" in kwargs:
            raise ValueError(
                "pi0_mode is only used by fixed-W0 G0W0/GW0; "
                "scGW builds Pi from GG self-consistently."
            )
        _reject_unknown_scf_kwargs(kwargs, _SCGW_SCF_SET_KEYS)
        _reject_removed_pi_k_interp_mode(kwargs)
        if "allow_no_pi_tail" in kwargs:
            kwargs["allow_no_pi_tail"] = bool(kwargs["allow_no_pi_tail"])
        allow_no_pi_tail = kwargs.get(
            "allow_no_pi_tail", getattr(self, "allow_no_pi_tail", False)
        )
        if "n_tail" in kwargs or "allow_no_pi_tail" in kwargs:
            n_tail = kwargs.get("n_tail", getattr(self, "n_tail", 100))
            kwargs["n_tail"] = _validate_n_tail(n_tail, allow_no_pi_tail)
        if "pi_imag_policy" in kwargs:
            kwargs["pi_imag_policy"] = _validate_pi_imag_policy(
                kwargs["pi_imag_policy"])
        if "pi_positive_policy" in kwargs:
            kwargs["pi_positive_policy"] = _validate_pi_positive_policy(
                kwargs["pi_positive_policy"])
        if "pi_positive_action" in kwargs:
            kwargs["pi_positive_action"] = _validate_pi_positive_action(
                kwargs["pi_positive_action"])
        if "pi_low_q_policy" in kwargs:
            kwargs["pi_low_q_policy"] = _validate_pi_low_q_policy(
                kwargs["pi_low_q_policy"])
        return super().set(**kwargs)

    def build(self):
        """Build the inherited sparse-IR context."""
        return super().build()

    def _uses_fixed_W0_pi0_mode(self):
        return False

    def _extra_flag_rows(self):
        return [
            ("SCF tolerance", "%.3e" % self.scf_conv_tol),
            ("SCF max cycle", "%d" % int(self.scf_max_cycle)),
            ("SCF damping", "%.6g" % self.scf_damp),
            ("SCF DIIS space", "%d" % int(self.scf_diis_space)),
            ("SCF DIIS start", "%d" % self.scf_diis_start_cycle),
            ("SCF mu bracket", self.scf_mu_bracket),
            ("SCF FS pin", mlog.format_bool(self.scf_fs_pin)),
            ("SCF density convention", "high-frequency corrected"
             if self.scf_high_freq_density else "direct -2 G(beta)"),
            ("SCF continuation max dex", "%.6g" % self.scf_continuation_max_dex),
            ("SCF checkpoint file", mlog.format_optional(self.scf_chkfile)),
            ("SCF restart checkpoint", mlog.format_optional(
                self.scf_restart_chkfile)),
            ("allow unconverged SCF", mlog.format_bool(self.allow_unconverged)),
            ("Pi tail min points", "%d" % self.n_tail),
            ("Sigma tail policy", self.sigma_tail_policy),
            ("static tail xcut", "%.6g" % self.static_tail_xcut),
            ("Pi imaginary tolerance", "%.3e" % self.pi_imag_tol),
            ("Pi imaginary policy", self.pi_imag_policy),
            ("Pi positive tolerance", "%.3e" % self.pi_positive_tol),
            ("Pi positive policy", self.pi_positive_policy),
            ("Pi positive action", self.pi_positive_action),
            ("Pi low-q policy", self.pi_low_q_policy),
            ("q low factor", "%.6g" % self.q_low_factor),
            ("q static low factor", mlog.format_optional(
                self.q_static_low_factor, "%.6g")),
            ("allow no Pi tail", mlog.format_bool(self.allow_no_pi_tail)),
        ]

    def _setup_scgw_point(self, n_cm3):
        if self.ctx is None:
            self.build()
        self.n_tail = _validate_n_tail(self.n_tail, self.allow_no_pi_tail)
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
        k_pi, k_wts_pi, info = grid.get_k_grid(
            k_F, 1.0 / ctx.beta, q_grid, kmax,
            lambda k: self._eps(k, mu0),
            n_pad=self.kgrid_n_pad,
            n_left=self.kgrid_n_left,
            n_right=self.kgrid_n_right,
            n_tail=self.n_tail,
            n_shell_min=self.kgrid_n_shell_min,
            n_shell_max=self.kgrid_n_shell_max,
        )
        active_k = k_pi <= kmax
        active_idx = np.flatnonzero(active_k)
        if not np.array_equal(active_idx, np.arange(len(active_idx))):
            raise ValueError("k_scf must be the leading block of k_pi.")
        k_scf = k_pi[active_k]
        k_wts_scf = k_wts_pi[active_k]
        if int(info.get("nk_active", len(k_scf))) != len(k_scf):
            raise RuntimeError("grid active-block metadata is inconsistent.")
        active_atol = 1.0e-12 * max(1.0, abs(kmax))
        pi_atol = 1.0e-12 * max(1.0, abs(info["kmax_pi"]))
        if not np.isclose(np.sum(k_wts_scf), kmax, rtol=1.0e-10, atol=active_atol):
            raise RuntimeError("k_scf weights do not cover the Sigma/gap domain.")
        if not np.isclose(np.sum(k_wts_pi), info["kmax_pi"], rtol=1.0e-10, atol=pi_atol):
            raise RuntimeError("k_pi weights do not cover the Pi domain.")
        self._check_q_coverage(q_grid, kmax, "scGW")
        e_scf = self._eps(k_scf, 0.0)
        e_pi = self._eps(k_pi, 0.0)
        G0 = ctx.get_G0(e_k=e_scf, mu=mu)
        n_re = scf.particle_number(
            ctx, G0, k_scf, k_wts_scf,
            e_k=e_scf, mu=mu, high_freq_correction=True,
        )
        n_ref = model.get_n(mu, ctx.beta, a, tol=1.0e-15)
        err = abs(n_re - n_ref)
        if err > self.n0_check_atol + self.n0_check_rtol * abs(n_ref):
            raise ValueError(
                "initial particle-number check failed: n_re=%s n_ref=%s diff=%s"
                % (n_re, n_ref, err)
            )
        vc_tol = (
            np.nextafter(0.0, 1.0)
            if self.q_interp_mode == "q_singular"
            else 1.0e-10
        )
        Vc = model.get_Vc(q_grid, p.eps_inf, q0_treat=False, tol=vc_tol)
        Vtot = Vc[:, None] * model.get_Vph_q(q_grid, ctx.vs, p)
        vtot_head = self._vtot_head(ctx)
        wcut = max(mu0, mu) * self.wcut_factor
        wcut = max(wcut, self.wcut_min_ev * model.EV_TO_HA)
        diag_average = model.bool_auto(self.diag_average, default=True)
        if self.q_interp_mode == "q_singular" and not diag_average:
            raise ValueError("q_singular mode requires diagonal cell averages.")
        diag_geometry = None
        if diag_average:
            diag_geometry = radial.get_diag_cell_geometry(
                k_scf, k_wts_scf, kmax, self.diag_ncell
            )
        return {
            "n_ref": n_ref,
            "n_re": n_re,
            "mu0": mu0,
            "mu": mu,
            "k_F": k_F,
            "kmax": kmax,
            "q_grid": q_grid,
            "k_scf": k_scf,
            "k_wts_scf": k_wts_scf,
            "k_pi": k_pi,
            "k_wts_pi": k_wts_pi,
            "e_scf": e_scf,
            "e_pi": e_pi,
            "Vc": Vc,
            "Vtot": Vtot,
            "vtot_head": vtot_head,
            "wcut": wcut,
            "grid_info": info,
            "diag_average": diag_average,
            "diag_average_mode": self.diag_average_mode,
            "diag_geometry": diag_geometry,
        }

    def kernel(
        self, n_cm3=None, sigma0=None, mu0=None, sigma0_k=None,
        fs_pin_shift0=None, restart_state=None, scf_restart_chkfile=None,
    ):
        """Run one scGW density point and return a :class:`PointResult`."""
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
        data = self._setup_scgw_point(float(n_cm3))
        _debug_grid(self, n_cm3, data)
        restart_chkfile = (
            self.scf_restart_chkfile
            if scf_restart_chkfile is None else scf_restart_chkfile
        )
        restart_cycle0 = 0
        has_scf_seed = any(
            x is not None for x in (sigma0, mu0, sigma0_k, fs_pin_shift0)
        )
        if restart_state is not None:
            if has_scf_seed:
                raise ValueError(
                    "restart_state cannot be combined with "
                    "sigma0/mu0/sigma0_k/fs_pin_shift0."
                )
            guess = scf.remap_scf_restart_state(
                restart_state,
                self.ctx,
                data["k_scf"],
                k_F=data["k_F"],
                fs_pin=self.scf_fs_pin,
            )
            sigma0 = guess.sigma0
            mu0 = guess.mu0
            fs_pin_shift0 = guess.fs_pin_shift0
        elif restart_chkfile and not has_scf_seed:
            guess = scf.load_scf_cycle_checkpoint(
                restart_chkfile,
                method="scGW",
                ctx=self.ctx,
                k_grid=data["k_scf"],
                fs_pin=self.scf_fs_pin,
            )
            sigma0 = guess.sigma0
            mu0 = guess.mu0
            fs_pin_shift0 = guess.fs_pin_shift0
            restart_cycle0 = int(guess.diagnostics["cycle"])
            mlog.kv_table(
                mlog.new_logger(self),
                "SCF checkpoint restart",
                (
                    ("file", str(restart_chkfile)),
                    ("cycle", "%d" % guess.diagnostics["cycle"]),
                    ("converged", mlog.format_bool(
                        guess.diagnostics["converged"])),
                    ("residual Sigma-mu", "%.3e" % guess.diagnostics["res_sigma"]),
                    ("density error", "%.3e" % guess.diagnostics["dn"]),
                ),
                level=mlog.INFO,
            )
        elif restart_chkfile and scf_restart_chkfile is not None:
            raise ValueError(
                "scf_restart_chkfile cannot be combined with "
                "sigma0/mu0/sigma0_k/fs_pin_shift0."
            )
        elif sigma0 is not None and sigma0_k is not None:
            sigma0 = _interp_sigma(sigma0_k, sigma0, data["k_scf"])
        sigma_tail = self._get_tail(data)
        conv_tol_nelec = model.default_nelec_tol(
            data["n_ref"],
            rtol=self.scf_conv_rtol_nelec,
            atol=self.scf_conv_atol_nelec,
        )
        if self._timings is not None:
            mlog.record_wall(self._timings, "initialization", init_t0)
        sres, W, diag_cells = self._solve_scgw_point(
            data, sigma0=sigma0, mu0=mu0, sigma_tail=sigma_tail,
            conv_tol_nelec=conv_tol_nelec, fs_pin_shift0=fs_pin_shift0,
            chk_metadata={
                "n_cm3": float(n_cm3),
                "n0": float(data["n_ref"]),
                "temperature_K": float(self.temperature),
                "k_F": float(data["k_F"]),
            },
            cycle0=restart_cycle0,
        )
        if not sres.converged and not self.allow_unconverged:
            raise RuntimeError(
                "scGW self-energy did not converge for n_cm3=%s: "
                "cycles=%d res_sigma=%s dn=%s"
                % (n_cm3, sres.cycles, sres.res_sigma, sres.dn)
            )
        pi_metrics = sres.extra.get("pi")
        denom_metrics = sres.extra.get("denom")
        rows = [
            ("converged", mlog.format_bool(sres.converged)),
            ("cycles", "%d" % sres.cycles),
            ("residual Sigma-mu", "%.3e" % sres.res_sigma),
            ("density error", "%.3e" % sres.dn),
            ("mu", "%.12e Ha" % sres.mu),
        ]
        if pi_metrics is not None:
            rows.extend(
                (
                    ("max Im Pi(iw)", "%.3e" % pi_metrics.pi_w_imag),
                    ("positive Pi count", "%d" % pi_metrics.positive_pi_count),
                    ("max positive Pi", "%.3e" % pi_metrics.max_positive_pi),
                )
            )
        if denom_metrics is not None:
            rows.extend(
                (
                    ("denominator regularizations", "%d" % denom_metrics.count),
                    ("minimum |denominator|", "%.3e" % denom_metrics.min_abs),
                    ("maximum |W|", "%.3e" % denom_metrics.max_abs_W),
                )
            )
            rows.extend(_singular_fit_rows(denom_metrics))
        if self.scf_fs_pin:
            rows.extend(
                (
                    ("FS pin shift", "%.12e Ha" % sres.extra["fs_pin_shift"]),
                    ("FS residual", "%.3e Ha" % sres.extra["fs_pin_residual"]),
                    ("absolute mu", "%.12e Ha" % sres.extra["mu_abs"]),
                )
            )
        mlog.kv_table(
            mlog.new_logger(self),
            "Normal-state SCF result",
            rows,
            level=mlog.INFO,
        )
        G_pair = self.ctx.get_G(
            e_k=data["e_scf"], mu=sres.mu, sigma=sres.sigma)
        mlog.section(log, "Gap solve", level=mlog.INFO)
        gap_t0 = mlog.wall_clock() if self._timings is not None else None
        gres = gap.solve_gap(
            self.ctx,
            data["k_scf"],
            data["k_wts_scf"],
            G_pair,
            W,
            config=self._gap_config(data["wcut"]),
            diag_cells=diag_cells,
            mu_scale=data["mu0"],
        )
        if self._timings is not None:
            mlog.record_wall(self._timings, "gap_solver", gap_t0)
        row = PointResult(
            n_cm3=float(n_cm3),
            lam=gres.lam,
            gap_converged=gres.converged,
            mu=sres.extra.get("mu_abs", sres.mu),
            mu0=data["mu0"],
            n_bohr3=data["n_ref"],
            n_re=sres.n,
            nk=len(data["k_scf"]),
            nq=len(data["q_grid"]),
            gap=gres,
            temperature_K=float(self.temperature),
            scf_converged=sres.converged,
            scf_cycles=sres.cycles,
            scf_res_sigma=sres.res_sigma,
            scf_dn=sres.dn,
            denom=sres.extra.get("denom"),
            pi=sres.extra.get("pi"),
            fs_pin_shift=sres.extra.get("fs_pin_shift"),
            fs_pin_residual=sres.extra.get("fs_pin_residual"),
            mu_abs=sres.extra.get("mu_abs"),
            timings=mlog.snapshot_timings(self._timings),
        )
        self._last_point_data = data
        self._last_W = W
        self._last_diag_cells = diag_cells
        self._last_k_wts = data["k_wts_scf"]
        self._last_scf = sres
        self._last_ctx = self.ctx
        self._last_k_grid = data["k_scf"]
        self.results = [row]
        mlog.timing_summary(log, self._timings)
        _debug_result(self, row)
        return row

    def last_scf_restart_state(self, require_converged=True):
        """Return the last converged SCF state for temperature continuation."""
        sres = getattr(self, "_last_scf", None)
        if sres is None:
            raise ValueError("no previous scGW SCF result is available.")
        if require_converged and not sres.converged:
            raise ValueError("last scGW SCF result is not converged.")
        return scf.restart_state_from_result(
            getattr(self, "_last_ctx", self.ctx), self._last_k_grid, sres
        )

    def scan(self, densities=None, restart=False, continuation_max_dex=None,
             return_intermediate=False):
        """Run a scGW density scan with optional density-continuation restart."""
        return _scan_scf_continuation(
            self, densities=densities, restart=restart,
            continuation_max_dex=continuation_max_dex,
            return_intermediate=return_intermediate,
        )

    def scan_temp(
        self,
        temperatures,
        n_cm3=None,
        *,
        restart=True,
        retry_without_restart=True,
        ir_file_for_temperature=None,
        return_restart_log=False,
        stop_callback=None,
    ):
        """Run a fixed-density temperature scan with SCF continuation."""
        return _scan_temp_continuation(
            self,
            temperatures,
            n_cm3=n_cm3,
            restart=restart,
            retry_without_restart=retry_without_restart,
            ir_file_for_temperature=ir_file_for_temperature,
            return_restart_log=return_restart_log,
            stop_callback=stop_callback,
        )

    def _get_tail(self, data):
        tail_k = data["k_pi"][len(data["k_scf"]):]
        if self.sigma_tail_policy == "zero":
            return None
        if self.sigma_tail_policy == "exchange_static":
            return radial.get_static_tail(
                None, tail_k, data["mu"], self.ctx.beta,
                self.params.band_a_ha_bohr2, self.params.eps_inf,
                xcut=self.static_tail_xcut,
            )
        if self.sigma_tail_policy == "exchange_t0":
            return radial.sigma_x_t0(tail_k, data["k_F"], self.params.eps_inf)
        raise ValueError("unknown sigma_tail_policy %r" % self.sigma_tail_policy)

    def _solve_scgw_point(
        self, data, sigma0, mu0, sigma_tail, conv_tol_nelec,
        fs_pin_shift0=None, chk_metadata=None, cycle0=0,
    ):
        ctx = self.ctx
        mu_bracket_policy = scf._normalize_mu_bracket_policy(self.scf_mu_bracket)
        cycle0 = int(cycle0)
        if cycle0 < 0:
            raise ValueError("cycle0 must be non-negative.")
        if self.scf_max_cycle < 1:
            raise ValueError("scf_max_cycle must be at least 1.")
        if sigma0 is None:
            sigma = np.zeros((len(data["k_scf"]), ctx.nw), dtype=complex)
        else:
            sigma = np.asarray(sigma0, dtype=complex)
        if sigma.shape != (len(data["k_scf"]), ctx.nw):
            raise ValueError("sigma0 shape is inconsistent with scGW k grid.")
        fs_pin = bool(self.scf_fs_pin)
        if fs_pin:
            sigma, mu0, fs_shift = scf.fs_pin_initial_state(
                ctx, sigma, mu0, data["k_scf"], data["k_F"],
                shift0=fs_pin_shift0,
            )
        else:
            fs_shift = 0.0
        if int(self.scf_diis_space) > 0:
            adiis = lib.diis.DIIS()
            adiis.space = int(self.scf_diis_space)
        else:
            adiis = None
        if mu0 is None and self.scf_high_freq_density:
            mu = data["mu"] - fs_shift
        else:
            mu = mu0
        W = None
        diag_cells = None
        converged = False
        res_sigma = np.inf
        dn = np.inf
        n = np.nan
        pi_metrics = None
        denom_metrics = None
        prev_res_sigma = None
        log = mlog.new_logger(self)
        mlog.kv_table(
            log,
            "Normal-state SCF",
            (
                ("type", "self-consistent GW"),
                ("nk / nk_pi / nq / nw", "%d / %d / %d / %d" % (
                    len(data["k_scf"]),
                    len(data["k_pi"]),
                    len(data["q_grid"]),
                    ctx.nw,
                )),
                ("max cycle", "%d" % int(self.scf_max_cycle)),
                ("Sigma-mu tolerance", "%.3e" % self.scf_conv_tol),
                ("mu density tolerance", "%.3e" % conv_tol_nelec),
                ("mu bracket policy", mu_bracket_policy),
                ("Fermi-surface pin", mlog.format_bool(fs_pin)),
                ("damping", "%.6g" % self.scf_damp),
                ("DIIS space", "%d" % int(self.scf_diis_space)),
                ("Sigma tail policy", self.sigma_tail_policy),
            ),
            level=mlog.INFO,
        )
        collect_cycle_pi_metrics = int(getattr(self, "verbose", 0)) >= mlog.DEBUG1
        timings = self._timings
        if collect_cycle_pi_metrics:
            cycle_header_fmt = scf._SCF_CYCLE_PI_HEADER_FMT
            cycle_row_fmt = scf._SCF_CYCLE_PI_ROW_FMT
            cycle_labels = (
                "cycle", "mu", "n", "dn", "res(S-mu)", "ImPi", "posPi", "rate"
            )
        else:
            cycle_header_fmt = scf._SCF_CYCLE_HEADER_FMT
            cycle_row_fmt = scf._SCF_CYCLE_ROW_FMT
            cycle_labels = ("cycle", "mu", "n", "dn", "res(S-mu)", "rate")
        mlog.table_header(
            log,
            "scGW SCF cycles",
            cycle_header_fmt,
            cycle_labels,
            level=mlog.DEBUG,
        )

        def dump_cycle(cycle_number, sigma_state, mu_state, shift_state, conv_state):
            scf.dump_scf_cycle_checkpoint(
                self.scf_chkfile,
                method="scGW",
                cycle=cycle_number,
                ctx=ctx,
                k_grid=data["k_scf"],
                sigma=sigma_state,
                mu=mu_state,
                fs_pin=fs_pin,
                fs_pin_shift=shift_state if fs_pin else None,
                converged=conv_state,
                res_sigma=res_sigma,
                dn=dn,
                n=n,
                metadata=chk_metadata,
            )

        for cycle in range(int(self.scf_max_cycle)):
            cycle_label = cycle0 + cycle + 1
            if fs_pin:
                mu_new = scf.fs_pin_get_mu(
                    ctx, sigma, fs_shift,
                    data["e_scf"], data["n_ref"],
                    data["k_scf"], data["k_wts_scf"], mu0=mu,
                    tol=conv_tol_nelec,
                    high_freq_correction=self.scf_high_freq_density,
                    bracket_policy=mu_bracket_policy,
                    warning_log=log,
                )
            else:
                mu_new = scf.get_mu(
                    ctx, sigma, data["e_scf"], data["n_ref"],
                    data["k_scf"], data["k_wts_scf"], mu0=mu,
                    tol=conv_tol_nelec,
                    high_freq_correction=self.scf_high_freq_density,
                    bracket_policy=mu_bracket_policy,
                    warning_log=log)
            G_scf = ctx.get_G(e_k=data["e_scf"], mu=mu_new, sigma=sigma)
            if fs_pin:
                n = scf.fs_pin_particle_number(
                    ctx, G_scf, data["k_scf"], data["k_wts_scf"],
                    e_k=data["e_scf"], mu=mu_new, shift=fs_shift,
                    high_freq_correction=self.scf_high_freq_density)
            else:
                n = scf.particle_number(
                    ctx, G_scf, data["k_scf"], data["k_wts_scf"],
                    e_k=data["e_scf"], mu=mu_new,
                    high_freq_correction=self.scf_high_freq_density)
            screen_t0 = mlog.wall_clock() if timings is not None else None
            W, diag_cells, pi_metrics, denom_metrics = self._update_W(
                data, sigma, mu_new, sigma_tail, fs_pin_shift=fs_shift,
                collect_pi_metrics=collect_cycle_pi_metrics,
                emit_pi_warnings=False)
            if timings is not None:
                mlog.record_wall(timings, "screening", screen_t0)
            sigma_t0 = mlog.wall_clock() if timings is not None else None
            sigma_raw_abs = scf.get_sigma(
                ctx, data["k_scf"], data["k_wts_scf"], G_scf, W,
                diag_cells=diag_cells,
            )
            if timings is not None:
                mlog.record_wall(timings, "self_energy", sigma_t0)
            if fs_pin:
                sigma_raw, mu_raw_pin, fs_shift_raw = scf.fs_pin_project_with_shift(
                    ctx, sigma_raw_abs, mu_new, fs_shift,
                    data["k_scf"], data["k_F"],
                )
                # Mu is re-solved before each map evaluation.  DIIS sees the
                # gauge-invariant Sigma-mu residual, while the extrapolated
                # state carries only pinned Sigma plus the auxiliary shift.
                sigma_mu_err = scf.sigma_mu_error(
                    sigma_raw, mu_raw_pin, sigma, mu_new
                )
            else:
                sigma_raw = sigma_raw_abs
                mu_raw_pin = mu_new
                fs_shift_raw = fs_shift
                # The evaluated input Green's function used the density-solved mu.
                sigma_mu_err = scf.sigma_mu_error(
                    sigma_raw, mu_new, sigma, mu_new
                )
            res_sigma = radial.max_abs(sigma_mu_err)
            dn = abs(n - data["n_ref"])
            candidate_converged = (
                res_sigma < self.scf_conv_tol and dn <= conv_tol_nelec
            )
            res_rate = scf._format_res_rate(res_sigma, prev_res_sigma)
            if collect_cycle_pi_metrics:
                cycle_values = (
                    cycle_label,
                    mu_new,
                    n,
                    dn,
                    res_sigma,
                    np.nan if pi_metrics is None else pi_metrics.pi_w_imag,
                    -1 if pi_metrics is None else pi_metrics.positive_pi_count,
                    res_rate,
                )
            else:
                cycle_values = (
                    cycle_label,
                    mu_new,
                    n,
                    dn,
                    res_sigma,
                    res_rate,
                )
            mlog.table_row(log, cycle_row_fmt, cycle_values, level=mlog.DEBUG)
            if candidate_converged:
                mu = mu_new
                converged = True
                dump_cycle(cycle_label, sigma, mu, fs_shift, converged)
                self._warn_pi_metrics(pi_metrics)
                break
            if cycle + 1 == int(self.scf_max_cycle):
                mu = mu_new
                dump_cycle(cycle_label, sigma, mu, fs_shift, converged)
                self._warn_pi_metrics(pi_metrics)
                break
            prev_res_sigma = res_sigma
            sigma_new = sigma_raw
            if fs_pin:
                old_shift = fs_shift
                shift_new = fs_shift_raw
                if self.scf_damp != 0.0:
                    sigma_new = (
                        sigma_new * (1.0 - self.scf_damp)
                        + sigma * self.scf_damp
                    )
                    shift_new = (
                        shift_new * (1.0 - self.scf_damp)
                        + fs_shift * self.scf_damp
                    )
                if adiis is not None and cycle >= int(self.scf_diis_start_cycle):
                    state = scf.pack_fs_pin_state(sigma_new, shift_new)
                    state_diis = adiis.update(state, xerr=sigma_mu_err)
                    if np.all(np.isfinite(state_diis)):
                        sigma_new, shift_new = scf.unpack_fs_pin_state(
                            state_diis, sigma.shape
                        )
                    else:
                        mlog.warning(
                            log,
                            "FS-pin DIIS produced a non-finite state; using "
                            "linearly mixed state for this cycle.",
                        )
                sigma, shift_new, delta = scf.fs_pin_repin_state(
                    ctx, sigma_new, shift_new, data["k_scf"], data["k_F"]
                )
                fs_shift = shift_new
                mu = scf.fs_pin_transport_mu_guess(mu_new, old_shift, fs_shift)
            else:
                if self.scf_damp != 0.0:
                    sigma_new = sigma_new * (1.0 - self.scf_damp) + sigma * self.scf_damp
                if adiis is not None and cycle >= int(self.scf_diis_start_cycle):
                    sigma_new = adiis.update(sigma_new, xerr=sigma_mu_err)
                sigma = radial.symm_sigma(ctx, sigma_new)
                mu = mu_new
            dump_cycle(cycle_label, sigma, mu, fs_shift, converged)
        extra = {"pi": pi_metrics, "denom": denom_metrics}
        if fs_pin:
            extra.update(
                fs_pin_shift=float(fs_shift),
                fs_pin_residual=scf.fs_pin_shift(
                    self.ctx, sigma, data["k_scf"], data["k_F"]
                ),
                mu_abs=float(mu) + float(fs_shift),
            )
        return scf.SCFResult(
            sigma=sigma,
            converged=bool(converged),
            cycles=cycle_label,
            res_sigma=float(res_sigma),
            dn=float(dn),
            mu=float(mu),
            n=float(n),
            extra=extra,
        ), W, diag_cells

    def _diag_cells_for_W_update(self, data, W):
        diag_average = data["diag_average"] if "diag_average" in data else (
            model.bool_auto(self.diag_average, default=True)
        )
        if not diag_average:
            return None

        cell_mode = data.get(
            "diag_average_mode",
            getattr(self, "diag_average_mode", "auto"),
        )
        cell_mode = radial._resolve_diag_average_mode(W, cell_mode)
        geometry = data.get("diag_geometry")
        diag_args = (
            None, W, data["k_scf"], data["k_wts_scf"],
            data["kmax"], self.diag_ncell,
        )
        sta_key = "_diag_sta_avg"
        bounds_key = "_diag_bounds"
        static_key_key = "_diag_static_key"
        W_static_key = _static_diag_key(W)
        if W_static_key is None:
            return radial.get_diag_cells(
                *diag_args, geometry=geometry, cell_mode=cell_mode)
        static_key = (
            W_static_key,
            _diag_cell_static_cache_key(data, self.diag_ncell, cell_mode),
        )
        sta_avg = data.get(sta_key)
        if sta_avg is None or data.get(static_key_key) != static_key:
            dyn_avg, sta_avg_new, bounds = radial.get_diag_cells(
                *diag_args, geometry=geometry, cell_mode=cell_mode)
            sta_avg = np.array(sta_avg_new, copy=True)
            sta_avg.setflags(write=False)
            cached_bounds = np.array(bounds, copy=True)
            cached_bounds.setflags(write=False)
            data[sta_key] = sta_avg
            data[bounds_key] = cached_bounds
            data[static_key_key] = static_key
            return dyn_avg, sta_avg, cached_bounds

        sta_avg = np.asarray(sta_avg)
        if sta_avg.shape != (len(data["k_scf"]),):
            raise RuntimeError("cached static diagonal cells are stale.")
        dyn_avg, bounds = radial.get_diag_cells_dyn(
            *diag_args, geometry=geometry, cell_mode=cell_mode)
        cached_bounds = data.get(bounds_key)
        if cached_bounds is None or not np.array_equal(bounds, cached_bounds):
            raise RuntimeError("cached diagonal-cell bounds are stale.")
        return dyn_avg, sta_avg, cached_bounds

    def _warn_pi_metrics(self, metrics):
        """Emit final Pi diagnostics warnings once for one density point."""
        if metrics is None:
            return
        messages = []
        if (
            self.pi_imag_policy == "warn"
            and metrics.g_tau_imag > float(self.pi_imag_tol)
        ):
            messages.append(
                "G(tau) imaginary part before Pi is %s" % metrics.g_tau_imag
            )
        if (
            self.pi_imag_policy == "warn"
            and metrics.pi_w_imag > float(self.pi_imag_tol)
        ):
            messages.append("Pi(iw) imaginary part is %s" % metrics.pi_w_imag)
        if (
            self.pi_positive_policy == "warn"
            and metrics.max_positive_pi > float(self.pi_positive_tol)
        ):
            messages.append(
                "Pi has positive entries above tolerance; max=%s"
                % metrics.max_positive_pi
            )
        if messages:
            mlog.warning(mlog.new_logger(self), "\n".join(messages))

    def _update_W(
        self, data, sigma, mu, sigma_tail, fs_pin_shift=0.0,
        collect_pi_metrics=True, emit_pi_warnings=True,
    ):
        sigma_pi = _build_sigma_pi(
            sigma, data["k_scf"], data["k_pi"], self.sigma_tail_policy,
            k_F=data["k_F"], eps_inf=self.params.eps_inf, sigma_tail=sigma_tail,
            shift=fs_pin_shift if self.scf_fs_pin else 0.0,
        )
        G_pi = self.ctx.get_G(e_k=data["e_pi"], mu=mu, sigma=sigma_pi)
        Pi, metrics = _get_Pi_scgw(
            self.ctx, G_pi, data["k_pi"], data["k_wts_pi"], data["q_grid"],
            data["k_F"], self.pi_imag_tol,
            self.pi_imag_policy,
            self.q_low_factor, self.q_static_low_factor,
            self.pi_positive_tol, self.pi_positive_policy,
            pi_positive_action=self.pi_positive_action,
            pi_low_q_policy=self.pi_low_q_policy,
            collect_metrics=collect_pi_metrics,
            emit_warnings=emit_pi_warnings,
            warning_log=mlog.new_logger(self),
        )
        W, denom_metrics = radial.get_W_from_Pi(
            self.ctx, Pi, data["q_grid"], data["Vc"], data["Vtot"],
            denom_policy=self.denom_policy, denom_min=self.denom_min,
            q_interp_mode=self.q_interp_mode,
            q_singular_head_policy=self.q_singular_head_policy,
            static_singular_coeff=4.0 * np.pi / float(self.params.eps_inf),
            vtot_head=(
                data["vtot_head"] if self.q_interp_mode == "q_singular"
                else None
            ),
            return_metrics=True,
            warning_log=mlog.new_logger(self),
        )
        diag_cells = self._diag_cells_for_W_update(data, W)
        return W, diag_cells, metrics, denom_metrics


def _reject_removed_pi_k_interp_mode(kwargs):
    if "pi_k_interp_mode" not in kwargs:
        return
    raise ValueError(
        "pi_k_interp_mode was removed; the scGW polarization bubble uses the "
        "radial G primitive."
    )


def _validate_n_tail(n_tail, allow_no_pi_tail):
    n_tail = int(n_tail)
    if n_tail < 0:
        raise ValueError("n_tail must be non-negative.")
    if n_tail == 0 and not allow_no_pi_tail:
        raise ValueError(
            "scGW Pi requires n_tail > 0 so the radial bubble has high-k "
            "coverage. Pass allow_no_pi_tail=True only for explicit debug "
            "comparisons."
        )
    return n_tail


def _build_sigma_pi(sigma, k_scf, k_pi, policy, k_F=None, eps_inf=None,
                    sigma_tail=None, shift=0.0):
    sigma = np.asarray(sigma)
    k_scf = np.asarray(k_scf, dtype=float)
    k_pi = np.asarray(k_pi, dtype=float)
    n_scf = len(k_scf)
    if not np.allclose(k_pi[:n_scf], k_scf, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_scf must be the leading block of k_pi.")
    out = np.zeros((len(k_pi), sigma.shape[1]), dtype=complex)
    out[:n_scf] = sigma
    shift = float(shift)
    if len(k_pi) == n_scf:
        return out
    if policy == "zero":
        if shift != 0.0:
            out[n_scf:] = -shift
        return out
    if policy == "exchange_static":
        sigma_tail = np.asarray(sigma_tail)
        if sigma_tail.shape != (len(k_pi) - n_scf,):
            raise ValueError("sigma_tail shape is inconsistent with k_pi.")
        out[n_scf:] = sigma_tail[:, None] - shift
        return out
    if policy == "exchange_t0":
        out[n_scf:] = radial.sigma_x_t0(k_pi[n_scf:], k_F, eps_inf)[:, None] - shift
        return out
    raise ValueError("unknown sigma tail policy %r" % policy)


def _beta_tau_index(ctx, tau_tol=1.0e-5):
    ts = np.asarray(ctx.ts_b, dtype=float)
    pair_err = radial.max_abs(ts + ts[::-1] - ctx.beta)
    if pair_err <= tau_tol:
        return None
    beta_index = np.empty(len(ts), dtype=int)
    for it, tau in enumerate(ts):
        beta_index[it] = int(np.argmin(np.abs(ts - (ctx.beta - tau))))
    return beta_index


def _get_Pi_scgw(ctx, G_pi, k_pi, k_wts_pi, q_grid, k_F, pi_imag_tol,
                 pi_imag_policy, q_low_factor, q_static_low_factor,
                 pi_positive_tol, pi_positive_policy, *,
                 pi_positive_action="keep", pi_low_q_policy="asymptotic",
                 collect_metrics=True, emit_warnings=True, warning_log=None):
    """Build the scGW polarization and optionally collect diagnostics."""
    pi_imag_policy = _validate_pi_imag_policy(pi_imag_policy)
    pi_positive_policy = _validate_pi_positive_policy(pi_positive_policy)
    pi_positive_action = _validate_pi_positive_action(pi_positive_action)
    pi_low_q_policy = _validate_pi_low_q_policy(pi_low_q_policy)
    G_tau = ctx.w2t_f2b(G_pi)
    need_imag_check = bool(collect_metrics) or pi_imag_policy == "fail"
    g_imag = radial.max_abs(G_tau.imag) if need_imag_check else 0.0
    if g_imag > pi_imag_tol and (
        pi_imag_policy == "fail" or bool(emit_warnings)
    ):
        _handle_pi_imag(
            "G(tau) imaginary part before Pi is %s" % g_imag,
            pi_imag_policy,
            warning_log=warning_log,
        )
    Pi_tau = radial.get_Pi(
        None, ctx, G_tau.real, k_pi, k_wts_pi, q_grid,
        beta_index=_beta_tau_index(ctx), q_zero_tol=1.0e-10,
    )
    Pi_w = ctx.t2w_b(Pi_tau)
    pi_imag = radial.max_abs(Pi_w.imag) if need_imag_check else 0.0
    if pi_imag > pi_imag_tol and (
        pi_imag_policy == "fail" or bool(emit_warnings)
    ):
        _handle_pi_imag(
            "Pi(iw) imaginary part is %s" % pi_imag,
            pi_imag_policy,
            warning_log=warning_log,
        )
    symm_err = radial.check_Pi_symm(ctx, Pi_w) if collect_metrics else 0.0
    Pi = radial.symm_Pi(ctx, Pi_w).real
    Pi = _apply_Pi_low_q(
        ctx, Pi, q_grid, k_F, q_low_factor, q_static_low_factor,
        pi_low_q_policy,
    )
    need_positive_check = (
        bool(collect_metrics)
        or pi_positive_policy == "fail"
        or pi_positive_action == "zero_small"
    )
    if need_positive_check:
        Pi, pos_count, max_pos = _handle_Pi_positive(
            Pi,
            pi_positive_tol,
            pi_positive_policy,
            pi_positive_action,
            emit_warnings=emit_warnings,
            warning_log=warning_log,
        )
    else:
        pos_count = 0
        max_pos = 0.0
    metrics = None
    if collect_metrics:
        metrics = PiMetrics(
            g_tau_imag=float(g_imag),
            pi_w_imag=float(pi_imag),
            pi_symm_err=float(symm_err),
            positive_pi_count=int(pos_count),
            max_positive_pi=float(max_pos),
        )
    return Pi, metrics


def _warn_with_logger(msg, warning_log=None):
    mlog.warning(warning_log, msg)


def _handle_pi_imag(msg, policy, warning_log=None):
    if policy == "fail":
        raise FloatingPointError(msg)
    if policy != "warn":
        raise ValueError("unknown pi_imag_policy %r" % policy)
    _warn_with_logger(msg, warning_log=warning_log)


def _apply_Pi_low_q(ctx, Pi, q_grid, k_F, q_low_factor, q_static_low_factor,
                    pi_low_q_policy):
    """Apply the optional low-q asymptotic continuation to Pi."""
    pi_low_q_policy = _validate_pi_low_q_policy(pi_low_q_policy)
    if pi_low_q_policy == "none":
        return np.asarray(Pi)
    Pi = np.array(Pi, copy=True)
    q_grid = np.asarray(q_grid, dtype=float)
    if q_static_low_factor is None:
        q_static_low_factor = q_low_factor
    finite_v = np.abs(ctx.vs) > 1.0e-12
    static_v = ~finite_v
    q_ref = float(q_low_factor) * k_F
    if q_ref > 0.0 and np.any(finite_v):
        idx_ref = int(np.searchsorted(q_grid, q_ref, side="left"))
        if 0 < idx_ref < len(q_grid):
            small = np.arange(idx_ref)
            scale = (q_grid[small, None] / q_grid[idx_ref]) ** 2
            Pi[np.ix_(small, finite_v)] = scale * Pi[idx_ref, finite_v][None, :]
    q_static_ref = float(q_static_low_factor) * k_F
    if q_static_ref > 0.0 and np.any(static_v):
        idx_ref = int(np.searchsorted(q_grid, q_static_ref, side="left"))
        if 0 < idx_ref < len(q_grid):
            small = np.arange(idx_ref)
            Pi[np.ix_(small, static_v)] = Pi[idx_ref, static_v][None, :]
    return Pi


def _handle_Pi_positive(Pi, pi_positive_tol, pi_positive_policy,
                        pi_positive_action, emit_warnings=True,
                        warning_log=None):
    """Diagnose positive Pi entries and optionally zero only small positives."""
    pi_positive_policy = _validate_pi_positive_policy(pi_positive_policy)
    pi_positive_action = _validate_pi_positive_action(pi_positive_action)
    pos = Pi.real > 0.0
    pos_count = int(np.count_nonzero(pos))
    max_pos = float(np.max(Pi.real[pos])) if pos_count else 0.0
    if np.any(pos & (Pi.real > pi_positive_tol)):
        msg = "Pi has positive entries above tolerance; max=%s" % max_pos
        if pi_positive_policy == "fail":
            raise FloatingPointError(msg)
        if emit_warnings:
            _warn_with_logger(msg, warning_log=warning_log)
    if pi_positive_action == "zero_small":
        small_pos = pos & (Pi.real <= pi_positive_tol)
        if np.any(small_pos):
            Pi = np.array(Pi, copy=True)
            Pi[small_pos] = 0.0
    return Pi, pos_count, max_pos
