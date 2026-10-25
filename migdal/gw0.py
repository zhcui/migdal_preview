"""GW0 radial driver for the STO one-band model.

GW0 reuses the G0W0 setup and fixed ``W0`` table, but replaces the
noninteracting Green's function by a self-consistent normal-state ``G`` before
solving the same gap eigenproblem.
"""

from __future__ import annotations

from migdal import gap, log as mlog, model, scf
from migdal.g0w0 import (
    G0W0,
    PointResult,
    _debug_grid,
    _debug_result,
    _scan_scf_continuation,
    _scan_temp_continuation,
)


_GW0_SCF_SET_KEYS = frozenset(
    (
        "scf_conv_tol",
        "scf_max_cycle",
        "scf_damp",
        "scf_diis_space",
        "scf_diis_start_cycle",
        "scf_mu_bracket",
        "scf_fs_pin",
        "scf_conv_rtol_nelec",
        "scf_conv_atol_nelec",
        "scf_high_freq_density",
        "scf_continuation_max_dex",
        "scf_chkfile",
        "scf_restart_chkfile",
    )
)


def _reject_unknown_scf_kwargs(kwargs, allowed=()):
    allowed = set(allowed)
    for key in kwargs:
        if key.startswith("scf_") and key not in allowed:
            raise ValueError("unknown SCF option %r" % key)


class GW0(G0W0):
    """GW0 driver: self-consistent G with fixed W0, then shared gap solve.

    The normal-state loop is implemented by :func:`migdal.scf.solve_fixed_W`.
    Density continuation can reuse the previous converged self-energy and
    chemical potential, interpolating the self-energy when the k grid changes.
    """

    def __init__(
        self,
        *args,
        scf_conv_tol=1.0e-6,
        scf_max_cycle=300,
        scf_damp=0.98,
        scf_diis_space=8,
        scf_diis_start_cycle=0,
        scf_mu_bracket="nearest_positive",
        scf_fs_pin=True,
        scf_conv_rtol_nelec=1.0e-4,
        scf_conv_atol_nelec=1.0e-14,
        scf_high_freq_density=True,
        scf_continuation_max_dex=1.0,
        scf_chkfile=None,
        scf_restart_chkfile=None,
        allow_unconverged=False,
        **kwargs,
    ):
        _reject_unknown_scf_kwargs(kwargs)
        super().__init__(*args, **kwargs)
        self.scf_conv_tol = scf_conv_tol
        self.scf_max_cycle = scf_max_cycle
        self.scf_damp = scf_damp
        self.scf_diis_space = scf_diis_space
        self.scf_diis_start_cycle = scf_diis_start_cycle
        self.scf_mu_bracket = scf_mu_bracket
        self.scf_fs_pin = bool(scf_fs_pin)
        self.scf_conv_rtol_nelec = scf_conv_rtol_nelec
        self.scf_conv_atol_nelec = scf_conv_atol_nelec
        self.scf_high_freq_density = bool(scf_high_freq_density)
        self.scf_continuation_max_dex = scf_continuation_max_dex
        self.scf_chkfile = scf_chkfile
        self.scf_restart_chkfile = scf_restart_chkfile
        self.allow_unconverged = allow_unconverged

    def _scf_max_cycle(self):
        return int(self.scf_max_cycle)

    def _scf_diis_space(self):
        return int(self.scf_diis_space)

    def set(self, **kwargs):
        _reject_unknown_scf_kwargs(kwargs, _GW0_SCF_SET_KEYS)
        return super().set(**kwargs)

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
        ]

    def kernel(
        self, n_cm3=None, sigma0=None, mu0=None, sigma0_k=None,
        fs_pin_shift0=None, restart_state=None, scf_restart_chkfile=None,
    ):
        """Run one GW0 density point and return a :class:`PointResult`."""
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
        data = self._setup_point(float(n_cm3))
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
                data["k_grid"],
                k_F=data["k_F"],
                fs_pin=self.scf_fs_pin,
            )
            sigma0 = guess.sigma0
            mu0 = guess.mu0
            fs_pin_shift0 = guess.fs_pin_shift0
        elif restart_chkfile and not has_scf_seed:
            guess = scf.load_scf_cycle_checkpoint(
                restart_chkfile,
                method="GW0",
                ctx=self.ctx,
                k_grid=data["k_grid"],
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
            sigma0 = _interp_sigma(sigma0_k, sigma0, data["k_grid"])
        conv_tol_nelec = model.default_nelec_tol(
            data["n_ref"],
            rtol=self.scf_conv_rtol_nelec,
            atol=self.scf_conv_atol_nelec,
        )
        if self._timings is not None:
            mlog.record_wall(self._timings, "initialization", init_t0)
        sres = scf.solve_fixed_W(
            self.ctx,
            data["k_grid"],
            data["k_wts"],
            data["e_k"],
            data["W"],
            data["n_ref"],
            sigma0=sigma0,
            mu0=mu0,
            fix_mu=False,
            conv_tol=self.scf_conv_tol,
            conv_tol_nelec=conv_tol_nelec,
            max_cycle=self._scf_max_cycle(),
            damp=self.scf_damp,
            diis_space=self._scf_diis_space(),
            diis_start_cycle=self.scf_diis_start_cycle,
            mu_bracket_policy=self.scf_mu_bracket,
            diag_cells=data["diag_cells"],
            high_freq_correction=self.scf_high_freq_density,
            fs_pin=self.scf_fs_pin,
            fs_pin_k_F=data["k_F"],
            fs_pin_mu0=data["mu"],
            fs_pin_shift0=fs_pin_shift0,
            verbose=self.verbose,
            stdout=self.stdout,
            chkfile=self.scf_chkfile,
            chk_metadata={
                "n_cm3": float(n_cm3),
                "n0": float(data["n_ref"]),
                "temperature_K": float(self.temperature),
                "k_F": float(data["k_F"]),
            },
            cycle0=restart_cycle0,
            timings=self._timings,
        )
        if not sres.converged and not self.allow_unconverged:
            raise RuntimeError(
                "GW0 self-energy did not converge for n_cm3=%s: "
                "cycles=%d res_sigma=%s dn=%s"
                % (n_cm3, sres.cycles, sres.res_sigma, sres.dn)
            )
        mlog.kv_table(
            mlog.new_logger(self),
            "Normal-state SCF result",
            (
                ("converged", mlog.format_bool(sres.converged)),
                ("cycles", "%d" % sres.cycles),
                ("residual Sigma-mu", "%.3e" % sres.res_sigma),
                ("density error", "%.3e" % sres.dn),
                ("mu", "%.12e Ha" % sres.mu),
            ),
            level=mlog.INFO,
        )
        if self.scf_fs_pin:
            mlog.kv_table(
                mlog.new_logger(self),
                "Fermi-surface pin result",
                (
                    ("shift", "%.12e Ha" % sres.extra["fs_pin_shift"]),
                    ("FS residual", "%.3e Ha" % sres.extra["fs_pin_residual"]),
                    ("absolute mu", "%.12e Ha" % sres.extra["mu_abs"]),
                ),
                level=mlog.INFO,
            )
        G_pair = self.ctx.get_G(e_k=data["e_k"], mu=sres.mu, sigma=sres.sigma)
        mlog.section(log, "Gap solve", level=mlog.INFO)
        gap_t0 = mlog.wall_clock() if self._timings is not None else None
        gres = gap.solve_gap(
            self.ctx,
            data["k_grid"],
            data["k_wts"],
            G_pair,
            data["W"],
            config=self._gap_config(data["wcut"]),
            diag_cells=data["diag_cells"],
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
            nk=len(data["k_grid"]),
            nq=len(data["q_grid"]),
            gap=gres,
            temperature_K=float(self.temperature),
            scf_converged=sres.converged,
            scf_cycles=sres.cycles,
            scf_res_sigma=sres.res_sigma,
            scf_dn=sres.dn,
            denom=data["denom"],
            fs_pin_shift=sres.extra.get("fs_pin_shift"),
            fs_pin_residual=sres.extra.get("fs_pin_residual"),
            mu_abs=sres.extra.get("mu_abs"),
            timings=mlog.snapshot_timings(self._timings),
        )
        self._last_point_data = data
        self._last_W = data["W"]
        self._last_diag_cells = data["diag_cells"]
        self._last_k_wts = data["k_wts"]
        self._last_scf = sres
        self._last_ctx = self.ctx
        self._last_k_grid = data["k_grid"]
        self.results = [row]
        mlog.timing_summary(log, self._timings)
        _debug_result(self, row)
        return row

    def last_scf_restart_state(self, require_converged=True):
        """Return the last converged SCF state for temperature continuation."""
        sres = getattr(self, "_last_scf", None)
        if sres is None:
            raise ValueError("no previous GW0 SCF result is available.")
        if require_converged and not sres.converged:
            raise ValueError("last GW0 SCF result is not converged.")
        return scf.restart_state_from_result(
            getattr(self, "_last_ctx", self.ctx), self._last_k_grid, sres
        )

    def scan(self, densities=None, restart=False, continuation_max_dex=None,
             return_intermediate=False):
        """Run a GW0 density scan with optional density-continuation restart."""
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


def _interp_sigma(old_k, old_sigma, new_k):
    import numpy as np

    old_k = np.asarray(old_k, dtype=float)
    old_sigma = np.asarray(old_sigma)
    new_k = np.asarray(new_k, dtype=float)
    order = np.argsort(old_k)
    old_k = old_k[order]
    old_sigma = old_sigma[order]
    sigma = np.empty((len(new_k), old_sigma.shape[1]), dtype=complex)
    for iw in range(old_sigma.shape[1]):
        sigma[:, iw] = (
            np.interp(new_k, old_k, old_sigma[:, iw].real)
            + 1j * np.interp(new_k, old_k, old_sigma[:, iw].imag)
        )
    return sigma
