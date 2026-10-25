r"""Shared one-band SCF helpers for GW0 and scGW.

The normal-state loops solve for a self-energy and chemical potential subject
to a fixed spin-summed density.  Convergence is measured in the gauge-invariant
combination ``Sigma - mu`` because

.. math::

    G^{-1} = i\omega - e_k - (\Sigma - \mu)

is unchanged by adding the same real constant to both ``Sigma`` and ``mu``.
Optional Fermi-surface pinning chooses the gauge where
``Re Sigma(k_F, i0) = 0`` while carrying an auxiliary absolute shift for
reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import warnings

import numpy as np
from pyscf import lib
from scipy.optimize import brentq, minimize_scalar

from migdal import log as mlog, model, mpi_wrap, radial

mpi = radial.mpi


_MU_SCAN_POINTS = 101
_MU_SCAN_HALF_WIDTH_MIN = 2.5
_MU_BRACKET_DEFAULT = "nearest_positive"
_MU_BRACKET_POLICIES = (_MU_BRACKET_DEFAULT, "extreme")
_FS_PIN_FIT_POINTS = 4
_DENSITY_PREF = 1.0 / (2.0 * np.pi**2)
_SCF_CYCLE_HEADER_FMT = "  %5s %11s %11s %11s %11s %11s"
_SCF_CYCLE_ROW_FMT = "  %5d %11.5g %11.5g %11.5g %11.5g %11s"
_SCF_CYCLE_PI_HEADER_FMT = "  %5s %11s %11s %11s %11s %11s %7s %11s"
_SCF_CYCLE_PI_ROW_FMT = "  %5d %11.5g %11.5g %11.5g %11.5g %11.5g %7d %11s"
_SCF_CHK_GROUP = "scf/latest"
_SCF_CHK_SCHEMA_VERSION = 1


def _format_res_rate(res_sigma, prev_res_sigma):
    """Format the residual ratio between consecutive SCF cycles."""
    if prev_res_sigma is None:
        return "-"
    prev = float(prev_res_sigma)
    if not np.isfinite(prev) or prev <= 0.0:
        return "-"
    rate = float(res_sigma) / prev
    if not np.isfinite(rate):
        return "-"
    return "%.5g" % rate


def _warn_with_logger(msg, warning_log=None):
    mlog.warning(warning_log, msg)


@dataclass
class SCFResult:
    """Convergence information for a normal-state self-energy solve.

    ``res_sigma`` is the max-norm residual of ``Sigma - mu`` for SCF loops that
    solve the chemical potential, matching the gauge-invariant combination in
    ``G^-1 = iw - e_k - (Sigma - mu)``.  When the loop solves ``mu``,
    convergence also requires the density residual ``dn`` to satisfy the
    requested electron-count tolerance.  In fixed-mu comparisons, ``dn`` is only
    diagnostic because the loop cannot adjust the particle number.
    """

    sigma: np.ndarray
    converged: bool
    cycles: int
    res_sigma: float
    dn: float
    mu: float
    n: float
    extra: dict = field(default_factory=dict)


@dataclass
class SCFRestartState:
    """Converged SCF state that can seed another temperature calculation."""

    ctx: object
    k_grid: np.ndarray
    sigma: np.ndarray
    mu: float
    fs_pin_shift: float | None = None
    converged: bool = True


@dataclass
class SCFInitialGuess:
    """Initial SCF guess produced from a previous restart state."""

    sigma0: np.ndarray
    mu0: float
    fs_pin_shift0: float | None = None
    diagnostics: dict = field(default_factory=dict)


def _decode_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _scf_checkpoint_should_write():
    """Return whether this process should write an SCF checkpoint."""
    if not mpi_wrap.loaded():
        if mpi_wrap.world_size_hint() <= 1:
            return True
        for key in (
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
            "MV2_COMM_WORLD_RANK",
            "SLURM_PROCID",
        ):
            val = os.environ.get(key)
            if val is not None:
                try:
                    return int(val) == 0
                except ValueError:
                    return False
        return False
    mpi_mod = mpi_wrap.get_mpi()
    return int(getattr(mpi_mod, "rank", 0)) == 0


def _save_optional_h5_attr(attrs, key, value):
    if value is None:
        return
    try:
        attrs[key] = value
    except TypeError:
        attrs[key] = repr(value)


def dump_scf_cycle_checkpoint(
    chkfile,
    *,
    method,
    cycle,
    ctx,
    k_grid,
    sigma,
    mu,
    fs_pin=False,
    fs_pin_shift=None,
    converged=False,
    res_sigma=np.nan,
    dn=np.nan,
    n=np.nan,
    metadata=None,
):
    """Save the latest normal-state SCF cycle for restart.

    Only rank 0 writes when MPI has been initialized through
    :mod:`migdal.mpi_wrap`.  The checkpoint stores the state that should be used
    as the next SCF input; DIIS history and derived W/Pi tables are intentionally
    not serialized.
    """
    if chkfile is None:
        return False
    if not _scf_checkpoint_should_write():
        return False

    import h5py

    path = Path(chkfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name("%s.tmp.%d" % (path.name, os.getpid()))
    sigma = np.asarray(sigma)
    k_grid = np.asarray(k_grid, dtype=float)
    ws = np.asarray(ctx.ws, dtype=float)
    shift = np.nan if fs_pin_shift is None else float(fs_pin_shift)
    mu_val = np.nan if mu is None else float(mu)
    with h5py.File(tmp_path, "w") as f:
        root = f.create_group("scf")
        g = root.create_group("latest")
        g.attrs["schema_version"] = _SCF_CHK_SCHEMA_VERSION
        g.attrs["method"] = str(method)
        g.attrs["cycle"] = int(cycle)
        g.attrs["converged"] = bool(converged)
        g.attrs["fs_pin"] = bool(fs_pin)
        g.attrs["res_sigma"] = float(res_sigma)
        g.attrs["dn"] = float(dn)
        g.attrs["n"] = float(n)
        _save_optional_h5_attr(g.attrs, "beta", getattr(ctx, "beta", None))
        _save_optional_h5_attr(g.attrs, "lamb", getattr(ctx, "lamb", None))
        g.attrs["nw"] = int(getattr(ctx, "nw", ws.size))
        g.attrs["nk"] = int(k_grid.size)
        if metadata:
            for key, value in metadata.items():
                _save_optional_h5_attr(g.attrs, "meta_" + str(key), value)
        g["sigma"] = sigma
        g["mu"] = mu_val
        g["fs_pin_shift"] = shift
        g["k_grid"] = k_grid
        g["ws"] = ws
    tmp_path.replace(path)
    return True


def load_scf_cycle_checkpoint(chkfile, *, method, ctx, k_grid, fs_pin=False):
    """Load the latest SCF checkpoint as an initial guess."""
    import h5py

    with h5py.File(chkfile, "r") as f:
        if "scf" not in f or "latest" not in f["scf"]:
            raise ValueError("SCF checkpoint does not contain scf/latest.")
        g = f[_SCF_CHK_GROUP]
        version = int(g.attrs.get("schema_version", 0))
        if version != _SCF_CHK_SCHEMA_VERSION:
            raise ValueError(
                "unsupported SCF checkpoint schema version %s." % version
            )
        chk_method = _decode_h5_attr(g.attrs.get("method", ""))
        if str(chk_method) != str(method):
            raise ValueError(
                "SCF checkpoint method %r does not match %r."
                % (chk_method, method)
            )
        chk_fs_pin = bool(g.attrs.get("fs_pin", False))
        if chk_fs_pin != bool(fs_pin):
            raise ValueError("SCF checkpoint FS-pin setting does not match.")
        sigma = np.asarray(g["sigma"][()], dtype=complex)
        mu = float(np.asarray(g["mu"][()]))
        fs_shift = float(np.asarray(g["fs_pin_shift"][()]))
        chk_k_grid = np.asarray(g["k_grid"][()], dtype=float)
        chk_ws = np.asarray(g["ws"][()], dtype=float)
        diagnostics = {
            "chkfile": str(chkfile),
            "method": str(chk_method),
            "cycle": int(g.attrs.get("cycle", 0)),
            "converged": bool(g.attrs.get("converged", False)),
            "res_sigma": float(g.attrs.get("res_sigma", np.nan)),
            "dn": float(g.attrs.get("dn", np.nan)),
            "n": float(g.attrs.get("n", np.nan)),
        }
        for key, value in g.attrs.items():
            if str(key).startswith("meta_"):
                diagnostics[str(key)[5:]] = _decode_h5_attr(value)

    k_grid = np.asarray(k_grid, dtype=float)
    ws = np.asarray(ctx.ws, dtype=float)
    if sigma.shape != (len(k_grid), int(ctx.nw)):
        raise ValueError("SCF checkpoint sigma shape does not match this grid.")
    if chk_k_grid.shape != k_grid.shape or not np.allclose(
        chk_k_grid, k_grid, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("SCF checkpoint k grid does not match this point.")
    if chk_ws.shape != ws.shape or not np.allclose(
        chk_ws, ws, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("SCF checkpoint Matsubara grid does not match.")
    if not np.isfinite(mu):
        raise ValueError("SCF checkpoint chemical potential is not finite.")
    if bool(fs_pin) and not np.isfinite(fs_shift):
        raise ValueError("SCF checkpoint FS-pin shift is not finite.")
    return SCFInitialGuess(
        sigma0=sigma,
        mu0=mu,
        fs_pin_shift0=(fs_shift if bool(fs_pin) else None),
        diagnostics=diagnostics,
    )


def restart_state_from_result(ctx, k_grid, result):
    """Return a copied temperature-restart snapshot from an SCF result."""
    return SCFRestartState(
        ctx=ctx,
        k_grid=np.asarray(k_grid, dtype=float).copy(),
        sigma=np.asarray(result.sigma, dtype=complex).copy(),
        mu=float(result.mu),
        fs_pin_shift=result.extra.get("fs_pin_shift"),
        converged=bool(result.converged),
    )


def sigma_mu_error(sigma_new, mu_new, sigma_old, mu_old):
    """Return ``(Sigma_new - mu_new) - (Sigma_old - mu_old)``.

    When ``mu_old`` is ``None``, the old self-energy has no separately solved
    chemical potential yet.  In that first-cycle case the current ``mu_new`` is
    paired with both self-energies, so the residual reduces to ``Delta Sigma``.
    """
    if mu_old is None:
        mu_old = mu_new
    return np.asarray(sigma_new) - np.asarray(sigma_old) - (
        float(mu_new) - float(mu_old)
    )


def sigma_mu_residual(sigma_new, mu_new, sigma_old, mu_old):
    """Return the max-norm residual of ``Sigma - mu``."""
    return radial.max_abs(sigma_mu_error(sigma_new, mu_new, sigma_old, mu_old))


def low_freq_real_at_k(
    ctx,
    table,
    k_grid,
    k_ref,
    nfit=_FS_PIN_FIT_POINTS,
):
    r"""Return ``Re table(k_ref, i0)`` from paired Matsubara samples.

    The real part of a normal self-energy is even on the imaginary axis.  The
    zero-frequency value is estimated by interpolating each Matsubara row to
    ``k_ref`` and fitting the lowest ``|omega|`` pairs to ``a + b omega**2``.
    If only one pair is available, the nearest-frequency value is returned.
    """
    table = np.asarray(table)
    k_grid = np.asarray(k_grid, dtype=float)
    if table.ndim != 2:
        raise ValueError("table must have shape (nk, nw).")
    if k_grid.ndim != 1 or table.shape[0] != k_grid.size:
        raise ValueError("k_grid must match table axis 0.")
    ws = np.asarray(getattr(ctx, "ws", None), dtype=float)
    if ws.ndim != 1 or ws.size != table.shape[1]:
        raise ValueError("ctx.ws must match table axis 1.")
    if k_grid.size == 0:
        raise ValueError("k_grid must not be empty.")
    k_ref = float(k_ref)
    if k_ref < k_grid[0] or k_ref > k_grid[-1]:
        raise ValueError("k_ref must lie inside k_grid.")

    vals = np.empty(table.shape[1], dtype=float)
    real_table = table.real
    for iw in range(table.shape[1]):
        vals[iw] = np.interp(k_ref, k_grid, real_table[:, iw])

    finite = np.isfinite(ws) & np.isfinite(vals)
    if not np.any(finite):
        raise ValueError("no finite Matsubara samples for low-frequency fit.")
    abs_w = np.abs(ws[finite])
    vals = vals[finite]
    order = np.argsort(abs_w)
    abs_w = abs_w[order]
    vals = vals[order]

    xs = []
    ys = []
    i = 0
    while i < abs_w.size and len(xs) < int(nfit):
        w0 = abs_w[i]
        scale = max(1.0, w0)
        j = i + 1
        while j < abs_w.size and abs(abs_w[j] - w0) <= 1.0e-12 * scale:
            j += 1
        xs.append(w0 * w0)
        ys.append(float(np.mean(vals[i:j])))
        i = j

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if y.size == 0:
        raise ValueError("no Matsubara samples for low-frequency fit.")
    if y.size == 1 or radial.max_abs(x - x[0]) == 0.0:
        return float(y[0])
    mat = np.column_stack((np.ones_like(x), x))
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coef[0])


def get_Z_matsubara(
    ctx,
    sigma,
    *,
    imag_atol=1.0e-10,
    imag_rtol=1.0e-8,
    sigma_symm_atol=1.0e-8,
    sigma_symm_rtol=1.0e-8,
    return_complex=False,
):
    r"""Return the imaginary-axis Eliashberg renormalization factor.

    Parameters
    ----------
    ctx : :class:`migdal.freq.FreqGrid`
        Sparse-IR Matsubara grid context.  ``ctx.ws`` must be reverse-index
        paired and contain no zero fermionic frequencies.
    sigma : (nk, nw) ndarray
        Normal self-energy on ``ctx.ws`` in Hartree, using Migdal's Dyson
        convention ``G^-1 = i*omega - (e_k - mu) - Sigma``.
    imag_atol, imag_rtol : float, optional
        Absolute and relative tolerances for the residual imaginary part of
        ``Z`` when ``return_complex=False``.
    sigma_symm_atol, sigma_symm_rtol : float, optional
        Absolute and relative tolerances for the normal self-energy symmetry
        ``Sigma(k, iw) = conj(Sigma(k, -iw))``.  The check is skipped when
        ``return_complex=True`` so callers can inspect raw nonsymmetric input.
    return_complex : bool, optional
        If true, return the complex table without checking that it is real.

    Returns
    -------
    Z : (nk, nw) ndarray
        Matsubara mass-renormalization factor.  By default the returned array is
        real.  It is dimensionless and is defined at the sampled Matsubara
        frequencies, not on the real axis.

    Notes
    -----
    The input is checked against the normal self-energy symmetry before the
    real ``Z`` table is returned.  The frequency-odd part of the self-energy is

    .. math::

        \Sigma_\mathrm{odd}(k, i\omega_n)
        = [\Sigma(k, i\omega_n) - \Sigma(k, -i\omega_n)] / 2.

    With the Dyson convention used by this code,

    .. math::

        \Sigma_\mathrm{odd}(k, i\omega_n)
        = i\omega_n [1 - Z(k, i\omega_n)].

    Therefore ``Z = 1 - Sigma_odd / (1j * omega)``.  For a perfectly symmetric
    normal self-energy this is equivalent to ``1 - Im(Sigma) / omega``.  The
    function uses the odd-part formula directly so it remains valid on
    non-uniform sparse-IR Matsubara samples.

    This quantity is not the real-axis quasiparticle residue
    ``[1 - d Re Sigma^R / d omega]^-1``.
    """
    sigma = np.asarray(sigma)
    ws = np.asarray(getattr(ctx, "ws", None), dtype=float)
    if sigma.ndim != 2:
        raise ValueError("sigma must have shape (nk, nw).")
    if ws.ndim != 1 or sigma.shape[1] != ws.size:
        raise ValueError("ctx.ws must match sigma axis 1.")
    radial.check_w_pair(ctx)
    if np.any(ws == 0.0):
        raise ValueError("fermionic Matsubara frequencies must be nonzero.")
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(ws)):
        raise ValueError("sigma and ctx.ws must contain only finite values.")
    imag_atol = float(imag_atol)
    imag_rtol = float(imag_rtol)
    sigma_symm_atol = float(sigma_symm_atol)
    sigma_symm_rtol = float(sigma_symm_rtol)
    if (
        not np.isfinite(imag_atol)
        or not np.isfinite(imag_rtol)
        or not np.isfinite(sigma_symm_atol)
        or not np.isfinite(sigma_symm_rtol)
        or imag_atol < 0.0
        or imag_rtol < 0.0
        or sigma_symm_atol < 0.0
        or sigma_symm_rtol < 0.0
    ):
        raise ValueError("Z diagnostic tolerances must be non-negative and finite.")
    if not return_complex:
        sigma_symm_err = radial.check_sigma_symm(ctx, sigma)
        sigma_scale = max(radial.max_abs(sigma), sigma_symm_err, 1.0e-300)
        if (
            sigma_symm_err > sigma_symm_atol
            and sigma_symm_err / sigma_scale > sigma_symm_rtol
        ):
            raise FloatingPointError(
                "normal self-energy symmetry error is too large for Matsubara Z."
            )
    pair = np.arange(ws.size - 1, -1, -1)
    sigma_odd = 0.5 * (sigma - sigma[:, pair])
    Z = 1.0 - sigma_odd / (1j * ws[None, :])
    if return_complex:
        return Z

    imag = radial.max_abs(Z.imag)
    scale = max(radial.max_abs(Z.real), imag, 1.0e-300)
    if imag > imag_atol and imag / scale > imag_rtol:
        raise FloatingPointError(
            "Matsubara Z has an unexpectedly large imaginary component."
        )
    return Z.real


def low_freq_Z_at_k(
    ctx,
    sigma,
    k_grid,
    k_ref,
    nfit=_FS_PIN_FIT_POINTS,
    *,
    imag_atol=1.0e-10,
    imag_rtol=1.0e-8,
    sigma_symm_atol=1.0e-8,
    sigma_symm_rtol=1.0e-8,
):
    r"""Return the low-frequency Matsubara ``Z(k_ref, i0)`` estimate.

    The function first evaluates :func:`get_Z_matsubara`, then interpolates in
    ``k`` and fits the lowest paired Matsubara samples to ``a + b*omega**2``
    using the same convention as :func:`low_freq_real_at_k`.
    """
    Z = get_Z_matsubara(
        ctx,
        sigma,
        imag_atol=imag_atol,
        imag_rtol=imag_rtol,
        sigma_symm_atol=sigma_symm_atol,
        sigma_symm_rtol=sigma_symm_rtol,
    )
    return low_freq_real_at_k(ctx, Z, k_grid, k_ref, nfit=nfit)


def fs_pin_shift(ctx, sigma, k_grid, k_F):
    """Return the real constant that pins ``Re Sigma(k_F, i0)`` to zero."""
    return low_freq_real_at_k(ctx, sigma, k_grid, k_F)


def fs_pin_project(ctx, sigma_abs, mu_abs, k_grid, k_F):
    """Project an absolute-gauge pair to the Fermi-surface-pinned gauge."""
    shift = fs_pin_shift(ctx, sigma_abs, k_grid, k_F)
    sigma_pin = radial.symm_sigma(ctx, np.asarray(sigma_abs) - shift)
    return sigma_pin, float(mu_abs) - shift, float(shift)


def fs_pin_project_with_shift(ctx, sigma_abs, mu_pin, shift_in, k_grid, k_F):
    """Project a raw absolute self-energy generated from a pinned input pair."""
    shift_raw = fs_pin_shift(ctx, sigma_abs, k_grid, k_F)
    sigma_pin = radial.symm_sigma(ctx, np.asarray(sigma_abs) - shift_raw)
    mu_raw_pin = float(mu_pin) + float(shift_in) - shift_raw
    return sigma_pin, float(mu_raw_pin), float(shift_raw)


def fs_pin_initial_state(ctx, sigma, mu, k_grid, k_F, shift0=None):
    """Return a pinned initial state and its absolute-gauge shift."""
    if shift0 is None:
        delta = fs_pin_shift(ctx, sigma, k_grid, k_F)
        shift = delta
    else:
        shift = float(shift0)
        delta = fs_pin_shift(ctx, sigma, k_grid, k_F)
        shift += delta
    sigma_pin = radial.symm_sigma(ctx, np.asarray(sigma) - delta)
    mu_pin = None if mu is None else float(mu) - delta
    return sigma_pin, mu_pin, float(shift)


def fs_pin_repin_state(ctx, sigma, shift, k_grid, k_F):
    """Remove numerical pin drift from a pinned-state self-energy."""
    delta = fs_pin_shift(ctx, sigma, k_grid, k_F)
    sigma_pin = radial.symm_sigma(ctx, np.asarray(sigma) - delta)
    return sigma_pin, float(shift) + delta, float(delta)


def fs_pin_absolute_sigma(sigma_pin, shift):
    """Return the absolute-gauge self-energy represented by an FS-pin state."""
    return np.asarray(sigma_pin) + float(shift)


def fs_pin_absolute_mu(mu_pin, shift):
    """Return the absolute-gauge chemical potential for an FS-pin state."""
    return float(mu_pin) + float(shift)


def fs_pin_transport_mu_guess(mu_pin, old_shift, new_shift):
    """Move a pinned chemical-potential guess between two pin gauges."""
    if mu_pin is None:
        return None
    return float(mu_pin) + float(old_shift) - float(new_shift)


def _check_temperature_restart_inputs(state, ctx_new, k_grid_new):
    ctx_old = state.ctx
    k_old = np.asarray(state.k_grid, dtype=float)
    k_new = np.asarray(k_grid_new, dtype=float)
    if k_old.ndim != 1 or k_new.ndim != 1:
        raise ValueError("temperature restart k grids must be one-dimensional.")
    if k_old.size > 1 and np.any(np.diff(k_old) <= 0.0):
        raise ValueError("temperature restart old k grid must be strictly increasing.")
    if k_new.size > 1 and np.any(np.diff(k_new) <= 0.0):
        raise ValueError("temperature restart new k grid must be strictly increasing.")

    nw_old = int(getattr(ctx_old, "nw", len(getattr(ctx_old, "ws", ()))))
    nw_new = int(getattr(ctx_new, "nw", len(getattr(ctx_new, "ws", ()))))
    if nw_old != nw_new:
        raise ValueError("temperature restart requires identical fermion nw.")

    lamb_old = getattr(ctx_old, "lamb", None)
    lamb_new = getattr(ctx_new, "lamb", None)
    if lamb_old is not None and lamb_new is not None:
        if not np.isclose(
            float(lamb_old), float(lamb_new), rtol=1.0e-12, atol=1.0e-12
        ):
            raise ValueError("temperature restart requires identical IR lambda.")

    sigma = np.asarray(state.sigma, dtype=complex)
    if sigma.shape != (len(k_old), nw_old):
        raise ValueError(
            "restart sigma shape is inconsistent with old k_grid and ctx.nw."
        )
    same_k = k_old.shape == k_new.shape and np.allclose(
        k_old, k_new, rtol=0.0, atol=1.0e-12
    )
    if not same_k:
        _check_temperature_restart_k_domains(k_old, k_new)
    return ctx_old, k_old, k_new, sigma, same_k


def _check_temperature_restart_k_domains(k_old, k_new):
    old = np.asarray(k_old, dtype=float)
    new = np.asarray(k_new, dtype=float)
    if old.size < 2:
        raise ValueError(
            "temperature restart k interpolation requires >=2 old k points."
        )
    scale = max(1.0, abs(old[0]), abs(old[-1]))
    left_tol = max(1.0e-10 * scale, 0.25 * (old[1] - old[0]))
    right_tol = max(1.0e-10 * scale, 0.25 * (old[-1] - old[-2]))
    if new[0] < old[0] - left_tol or new[-1] > old[-1] + right_tol:
        warnings.warn(
            "temperature restart k interpolation target extends outside "
            "source k range.",
            RuntimeWarning,
            stacklevel=3,
        )


def _interp_table_k(old_k, table, new_k):
    old_k = np.asarray(old_k, dtype=float)
    new_k = np.asarray(new_k, dtype=float)
    table = np.asarray(table, dtype=complex)
    idx = np.searchsorted(old_k, new_k, side="right") - 1
    idx = np.clip(idx, 0, old_k.size - 2)
    t = (new_k - old_k[idx]) / (old_k[idx + 1] - old_k[idx])
    return (1.0 - t)[:, None] * table[idx] + t[:, None] * table[idx + 1]


def _ir_coefficients(ctx, table):
    return np.tensordot(np.asarray(table), ctx.F_w_f_inv, axes=((1,), (1,)))


def _ir_evaluate(ctx, coeff):
    out = np.tensordot(np.asarray(coeff), ctx.F_w_f, axes=((-1,), (1,)))
    return np.moveaxis(out, -1, 1)


def _check_reduced_matsubara_match(ctx_old, ctx_new):
    red_old = np.asarray(ctx_old.ws, dtype=float) * float(ctx_old.beta)
    red_new = np.asarray(ctx_new.ws, dtype=float) * float(ctx_new.beta)
    if red_old.shape != red_new.shape or not np.allclose(
        red_old, red_new, rtol=1.0e-10, atol=1.0e-10
    ):
        raise ValueError(
            "temperature restart requires matching reduced Matsubara grid."
        )


def remap_scf_restart_state(
    state,
    ctx_new,
    k_grid_new,
    *,
    k_F=None,
    fs_pin=False,
    mu_abs_guess=None,
    require_converged=True,
):
    """Return a temperature-remapped SCF initial guess.

    The remap acts on the gauge-invariant table ``X = Sigma - mu`` by copying
    same reduced-Matsubara-index IR coefficients with the required sparse-IR
    normalization correction.
    """
    if require_converged and not bool(state.converged):
        raise ValueError("temperature restart state is not converged.")
    ctx_old, k_old, k_new, sigma_old, same_k = _check_temperature_restart_inputs(
        state, ctx_new, k_grid_new
    )

    mu_old = float(state.mu)
    x_old = radial.symm_sigma(ctx_old, sigma_old - mu_old)
    coeff_old = _ir_coefficients(ctx_old, x_old)
    if coeff_old.shape[-1] != ctx_new.F_w_f.shape[1]:
        raise ValueError("temperature restart requires matching IR basis size.")

    _check_reduced_matsubara_match(ctx_old, ctx_new)
    scale = np.sqrt(float(ctx_new.beta) / float(ctx_old.beta))
    x_new = _ir_evaluate(ctx_new, coeff_old / scale)
    if not same_k:
        x_new = _interp_table_k(k_old, x_new, k_new)
    x_new = radial.symm_sigma(ctx_new, x_new)

    if mu_abs_guess is None:
        if state.fs_pin_shift is None:
            mu_abs_guess = mu_old
        else:
            mu_abs_guess = mu_old + float(state.fs_pin_shift)
    mu_abs_guess = float(mu_abs_guess)

    diagnostics = {
        "mode": "ir_reduced",
        "beta_old": float(ctx_old.beta),
        "beta_new": float(ctx_new.beta),
        "k_interpolated": not same_k,
        "nk_old": int(len(k_old)),
        "nk_new": int(len(k_new)),
        "fs_pin": bool(fs_pin),
    }
    if fs_pin:
        if k_F is None:
            raise ValueError("fs_pin=True requires k_F.")
        x_kf_i0 = low_freq_real_at_k(ctx_new, x_new, k_new, float(k_F))
        mu_pin = -float(x_kf_i0)
        shift = mu_abs_guess - mu_pin
        sigma_pin = radial.symm_sigma(ctx_new, x_new + mu_pin)
        diagnostics["x_kF_i0"] = float(x_kf_i0)
        return SCFInitialGuess(
            sigma0=sigma_pin,
            mu0=float(mu_pin),
            fs_pin_shift0=float(shift),
            diagnostics=diagnostics,
        )

    return SCFInitialGuess(
        sigma0=radial.symm_sigma(ctx_new, x_new + mu_abs_guess),
        mu0=mu_abs_guess,
        fs_pin_shift0=None,
        diagnostics=diagnostics,
    )


def fs_pin_get_mu(
    ctx,
    sigma_pin,
    shift,
    e_k,
    n0,
    k_grid,
    k_wts,
    mu0=None,
    tol=1.0e-10,
    high_freq_correction=True,
    bracket_policy=_MU_BRACKET_DEFAULT,
    check_mu_monotonic=False,
    warning_log=None,
):
    """Solve the density constraint in absolute gauge and return pinned mu."""
    shift = float(shift)
    mu0_abs = None if mu0 is None else fs_pin_absolute_mu(mu0, shift)
    mu_abs = get_mu(
        ctx,
        fs_pin_absolute_sigma(sigma_pin, shift),
        e_k,
        n0,
        k_grid,
        k_wts,
        mu0=mu0_abs,
        tol=tol,
        high_freq_correction=high_freq_correction,
        bracket_policy=bracket_policy,
        check_mu_monotonic=check_mu_monotonic,
        warning_log=warning_log,
    )
    return float(mu_abs) - shift


def fs_pin_particle_number(
    ctx,
    G,
    k_grid,
    k_wts,
    e_k=None,
    mu=None,
    shift=0.0,
    high_freq_correction=True,
):
    """Evaluate the density diagnostic in the absolute gauge of a pin state."""
    mu_abs = None if mu is None else fs_pin_absolute_mu(mu, shift)
    return particle_number(
        ctx,
        G,
        k_grid,
        k_wts,
        e_k=e_k,
        mu=mu_abs,
        high_freq_correction=high_freq_correction,
    )


def pack_fs_pin_state(sigma, shift):
    """Pack pinned ``sigma`` and its auxiliary absolute shift for DIIS."""
    sigma = np.asarray(sigma)
    out = np.empty(sigma.size + 1, dtype=np.result_type(sigma, complex))
    out[:-1] = sigma.ravel()
    out[-1] = complex(float(shift), 0.0)
    return out


def unpack_fs_pin_state(state, sigma_shape):
    """Unpack a :func:`pack_fs_pin_state` vector."""
    state = np.asarray(state)
    expected = int(np.prod(sigma_shape)) + 1
    if state.size != expected:
        raise ValueError("packed FS-pin state has inconsistent size.")
    sigma = state[:-1].reshape(sigma_shape)
    shift = float(np.real(state[-1]))
    return sigma, shift


def particle_number(ctx, G, k_grid, k_wts, e_k=None, mu=None,
                    high_freq_correction=True):
    r"""Return spin-summed radial density in Bohr^-3.

    By default this uses the high-frequency corrected sparse-IR convention

    .. math::

        n_k = 2 f(\epsilon_k - \mu)
            - 2 [G_k - G^0_k](\tau=\beta^-),

    which evaluates the noninteracting contribution analytically and only
    transforms the correlated difference.  This is less sensitive to endpoint
    reconstruction errors at low carrier density.  Passing
    ``high_freq_correction=False`` uses the direct endpoint formula

    .. math::

        n_k = -2 G_k(\tau=\beta^-),

    reconstructed directly from the sparse-IR frequency table.
    """
    if high_freq_correction:
        if e_k is None or mu is None:
            raise ValueError(
                "high_freq_correction=True requires both e_k and mu."
            )
        rdm1 = ctx.make_rdm1(G=G, e_k=e_k, mu=mu)
    else:
        rdm1 = ctx.make_rdm1(G=G)
    return np.dot(np.asarray(k_grid) ** 2 * np.asarray(k_wts), rdm1).real / (
        2.0 * np.pi**2
    )


def _density_weights(k_grid, k_wts):
    """Return radial density integration weights including ``1/(2*pi**2)``."""
    k = np.asarray(k_grid, dtype=float)
    w = np.asarray(k_wts, dtype=float)
    if k.ndim != 1 or w.shape != k.shape:
        raise ValueError("k_grid and k_wts must be one-dimensional matches.")
    return (k * k * w) * _DENSITY_PREF


def _prepare_high_freq_density(ctx, sigma, e_k, k_grid, k_wts):
    """Precompute immutable pieces for high-frequency corrected ``n(mu)``."""
    e_k = np.asarray(e_k, dtype=float)
    ws = np.asarray(ctx.ws, dtype=float)
    if e_k.ndim != 1 or ws.ndim != 1:
        raise ValueError("e_k and ctx.ws must be one-dimensional.")
    if sigma is None:
        sigma = np.zeros((e_k.size, ws.size), dtype=np.complex128)
    else:
        sigma = np.asarray(sigma)
    if sigma.shape != (e_k.size, ws.size):
        raise ValueError("sigma must have shape (len(e_k), ctx.nw).")
    beta_row = np.asarray(ctx.w2beta_row_f())
    if beta_row.shape != (ws.size,):
        raise ValueError("ctx.w2beta_row_f() has inconsistent shape.")
    weights = _density_weights(k_grid, k_wts)
    if weights.shape != e_k.shape:
        raise ValueError("k_grid/k_wts must match e_k.")
    base0 = 1j * ws[None, :] - e_k[:, None]
    return sigma, e_k, base0, float(ctx.beta), beta_row, weights


def _particle_number_high_freq_from_prepared(prepared, mu):
    r"""Return high-frequency corrected density from ``Sigma`` without ``G0``.

    This is algebraically identical to
    ``particle_number(ctx, ctx.get_G(...), ..., e_k=e_k, mu=mu)`` because

    .. math::

        G - G_0 = \Sigma / [(i\omega-\epsilon+\mu-\Sigma)
                            (i\omega-\epsilon+\mu)].

    It avoids constructing ``G0``, ``G-G0``, and the full ``rdm1`` array for
    every chemical-potential trial.
    """
    sigma, e_k, base0, beta, beta_row, weights = prepared
    mu = float(mu)
    base = base0 + mu
    delta = base - sigma
    delta *= base
    np.divide(sigma, delta, out=delta)
    dG_beta = delta.dot(beta_row)
    occ0 = model.f_occ(e_k - mu, beta)
    return float((2.0 * (np.dot(weights, occ0) - np.dot(weights, dG_beta))).real)


def _particle_number_from_sigma_mu(
    ctx, sigma, e_k, k_grid, k_wts, mu,
):
    """Convenience wrapper for one high-frequency corrected density value."""
    return _particle_number_high_freq_from_prepared(
        _prepare_high_freq_density(ctx, sigma, e_k, k_grid, k_wts), mu
    )


@radial._lazy_parallel_call
def _scan_mu_grid(_, ctx, sigma, e_k, k_grid, k_wts, mus,
                  high_freq_correction=True):
    """Evaluate ``n(mu)`` on a fixed mu grid over MPI rank blocks."""
    mus = np.asarray(mus, dtype=float)
    mloc0, mloc1 = radial._task_location(len(mus))
    ns = np.empty(mloc1 - mloc0, dtype=float)
    if high_freq_correction:
        prepared = _prepare_high_freq_density(ctx, sigma, e_k, k_grid, k_wts)
        for local_i, mu in enumerate(mus[mloc0:mloc1]):
            ns[local_i] = _particle_number_high_freq_from_prepared(prepared, mu)
        return radial._gather_rank0(ns)

    for local_i, mu in enumerate(mus[mloc0:mloc1]):
        G = ctx.get_G(e_k=e_k, mu=mu, sigma=sigma)
        ns[local_i] = particle_number(
            ctx, G, k_grid, k_wts,
            e_k=e_k,
            mu=mu,
            high_freq_correction=high_freq_correction,
        )
    return radial._gather_rank0(ns)


def _positive_slope_bracket(mus, ns, n0, mu_guess, tol=0.0):
    """Return the local positive-slope sign-change nearest ``mu_guess``."""
    mus = np.asarray(mus, dtype=float)
    ns = np.asarray(ns, dtype=float)
    vals = ns - float(n0)
    tol = max(float(tol), 0.0)
    brackets = []
    for i in range(len(mus) - 1):
        if (
            not np.isfinite(mus[i])
            or not np.isfinite(mus[i + 1])
            or not np.isfinite(ns[i])
            or not np.isfinite(ns[i + 1])
        ):
            continue
        if ns[i + 1] <= ns[i]:
            continue
        if abs(vals[i]) <= tol:
            brackets.append((float(mus[i]), float(mus[i])))
        if abs(vals[i + 1]) <= tol:
            brackets.append((float(mus[i + 1]), float(mus[i + 1])))
        if vals[i] * vals[i + 1] < 0.0:
            brackets.append((float(mus[i]), float(mus[i + 1])))
    if not brackets:
        return None
    return min(brackets, key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - mu_guess))


def _extreme_bracket(mus, ns, n0, tol=0.0):
    """Return a sign-changing bracket between the sampled density extrema."""
    mus = np.asarray(mus, dtype=float)
    ns = np.asarray(ns, dtype=float)
    vals = ns - float(n0)
    finite = np.isfinite(mus) & np.isfinite(ns)
    if not np.any(finite):
        return None
    idx = np.flatnonzero(finite)
    idx_min = idx[int(np.argmin(ns[finite]))]
    idx_max = idx[int(np.argmax(ns[finite]))]
    tol = max(float(tol), 0.0)
    if abs(vals[idx_min]) <= tol:
        return float(mus[idx_min]), float(mus[idx_min])
    if abs(vals[idx_max]) <= tol:
        return float(mus[idx_max]), float(mus[idx_max])
    if vals[idx_min] * vals[idx_max] >= 0.0:
        return None
    lval, rval = sorted((float(mus[idx_min]), float(mus[idx_max])))
    return lval, rval


def _normalize_mu_bracket_policy(policy):
    policy = str(policy).lower()
    if policy not in _MU_BRACKET_POLICIES:
        raise ValueError(
            "unknown mu bracket policy %r; expected one of %s"
            % (policy, ", ".join(_MU_BRACKET_POLICIES))
        )
    return policy


def _mu_bracket(mus, ns, n0, mu_guess, tol, policy=_MU_BRACKET_DEFAULT):
    """Choose a scan bracket according to the requested policy."""
    policy = _normalize_mu_bracket_policy(policy)
    if policy == "nearest_positive":
        return _positive_slope_bracket(mus, ns, n0, mu_guess, tol=tol)
    if policy == "extreme":
        return _extreme_bracket(mus, ns, n0, tol=tol)
    raise AssertionError(policy)


def get_mu(
    ctx,
    sigma,
    e_k,
    n0,
    k_grid,
    k_wts,
    mu0=None,
    tol=1.0e-10,
    high_freq_correction=True,
    bracket_policy=_MU_BRACKET_DEFAULT,
    check_mu_monotonic=False,
    warning_log=None,
):
    r"""Find the chemical potential for a target radial density.

    Parameters
    ----------
    ctx : :class:`migdal.freq.FreqGrid`
        Sparse-IR Matsubara grid context.
    sigma : (nk, nw) ndarray
        Normal self-energy on ``ctx.ws`` in Hartree.
    e_k : (nk,) ndarray
        One-band dispersion in Hartree, excluding the chemical potential.
    n0 : float
        Target spin-summed density in Bohr^-3.
    k_grid, k_wts : (nk,) ndarray
        Radial momenta in Bohr^-1 and quadrature weights in Bohr^-1.
    mu0 : float, optional
        Current or previous chemical potential branch.  It is checked first;
        if the particle number is already within ``tol`` it is returned.
    tol : float, optional
        Absolute density tolerance in Bohr^-3.
    high_freq_correction : bool, optional
        If true, evaluate ``n(mu)`` with
        ``2 f(e_k - mu) - 2 [G - G0](beta^-)``.  The default ``True`` is the
        recommended convention for sparse-IR grids.  Pass ``False`` only for
        direct ``-2 G(beta^-)`` endpoint comparisons.
    bracket_policy : {"nearest_positive", "extreme"}, optional
        Policy used to select a Brent bracket from the 101-point scan.  The
        default chooses the closest local positive-slope bracket.  ``"extreme"``
        uses the sampled minimum- and maximum-density points instead.
    check_mu_monotonic : bool, optional
        If true, run the expensive local monotonicity diagnostic before
        returning a converged root.  The default is false because SCF loops call
        this routine many times and the diagnostic requires three extra density
        evaluations per successful solve.
    warning_log : PySCF logger, optional
        Logger used for fallback/diagnostic warnings.  If omitted, warnings are
        emitted to a default WARN-level stream logger.

    Returns
    -------
    mu : float
        Chemical potential in Hartree.

    Notes
    -----
    The high-frequency corrected density is less sensitive to sparse-IR
    endpoint reconstruction errors at low carrier density.  It does not change
    the Green's-function equation being solved; it only changes how the
    particle-number constraint is evaluated.

    The solver first tests ``mu0``.  If it is outside the density tolerance, it
    evaluates 101 uniformly spaced trial chemical potentials over
    ``mu0 +/- max(2.5 Ha, 2 / beta)``; this scan is distributed over MPI ranks.
    By default, the Brent bracket is the local positive-slope sign-changing
    section whose midpoint is nearest the input ``mu0``.  Passing
    ``bracket_policy="extreme"`` instead brackets between the sampled density
    extrema.  If the selected policy does not provide a bracket, the code warns
    and returns the bounded squared-residual minimizer over
    ``mu_guess +/- 0.5 Ha``.
    """

    prepared_density = None
    if high_freq_correction:
        prepared_density = _prepare_high_freq_density(
            ctx, sigma, e_k, k_grid, k_wts
        )

    def get_n(mu):
        if prepared_density is not None:
            return _particle_number_high_freq_from_prepared(prepared_density, mu)
        G = ctx.get_G(e_k=e_k, mu=mu, sigma=sigma)
        return particle_number(
            ctx, G, k_grid, k_wts,
            e_k=e_k,
            mu=mu,
            high_freq_correction=False,
        )

    def cost(mu):
        return get_n(mu) - n0

    if mu0 is None:
        mu_guess = float(np.average(np.asarray(e_k).real))
    else:
        mu_guess = float(mu0)
    bracket_policy = _normalize_mu_bracket_policy(bracket_policy)

    if abs(cost(mu_guess)) < tol:
        return float(mu_guess)

    half_width = max(_MU_SCAN_HALF_WIDTH_MIN, 2.0 / ctx.beta)
    xs = np.linspace(mu_guess - half_width, mu_guess + half_width, _MU_SCAN_POINTS)
    ns = _scan_mu_grid(
        None, ctx, sigma, e_k, k_grid, k_wts, xs,
        high_freq_correction=high_freq_correction,
    )
    tol_root = max(float(tol), 5.0 * np.finfo(float).eps)

    bracket = _mu_bracket(
        xs, ns, n0, mu_guess, tol=float(tol), policy=bracket_policy
    )
    if bracket is not None:
        lval, rval = bracket
        if lval == rval:
            if check_mu_monotonic:
                _warn_mu_nonmonotonic(get_n, lval, warning_log=warning_log)
            return float(lval)
        try:
            res = brentq(
                cost, lval, rval, xtol=tol_root, rtol=tol_root,
                maxiter=1000, full_output=True, disp=False,
            )
        except (RuntimeError, ValueError) as err:
            bracket_error = err
        else:
            if res[1].converged:
                if check_mu_monotonic:
                    _warn_mu_nonmonotonic(get_n, res[0], warning_log=warning_log)
                return float(res[0])
            bracket_error = RuntimeError(
                "chemical-potential brentq did not converge."
            )
    else:
        bracket_error = None

    def sq(mu):
        val = cost(float(np.atleast_1d(mu)[0]))
        if not np.isfinite(val):
            return np.inf
        return val * val

    if bracket_error is None:
        bracket_label = (
            "positive-slope" if bracket_policy == "nearest_positive" else "extreme"
        )
        msg = (
            "n(mu) scan did not find a %s bracket for the target "
            "density in [%s, %s]"
        ) % (bracket_label, xs[0], xs[-1])
    else:
        bracket_label = (
            "positive-slope" if bracket_policy == "nearest_positive" else "extreme"
        )
        msg = "chemical-potential %s bracket solve failed: %s" % (
            bracket_label, bracket_error)
    bounds = (mu_guess - 0.5, mu_guess + 0.5)
    _warn_with_logger(
        "%s; using bounded squared density residual minimization in [%s, %s]."
        % (msg, bounds[0], bounds[1]),
        warning_log=warning_log,
    )
    res = minimize_scalar(
        sq,
        bounds=bounds,
        method="bounded",
        options={"xatol": tol_root},
    )
    return float(res.x)


def _warn_mu_nonmonotonic(get_n, mu, step=1.0e-6, warning_log=None):
    n_l = get_n(float(mu) - step)
    n = get_n(float(mu))
    n_r = get_n(float(mu) + step)
    if n_l > n or n > n_r:
        _warn_with_logger(
            "n(mu) is locally non-monotonic near mu=%s: n_l=%s n=%s n_r=%s. "
            "A denser frequency/k grid may be required."
            % (mu, n_l, n, n_r),
            warning_log=warning_log,
        )


def get_sigma(ctx, k_grid, k_wts, G, W, diag_cells=None):
    """Return physical unscaled ``Sigma(k, iw)`` from a fixed interaction."""
    if diag_cells is None:
        sigma = radial.get_sigma(
            None, ctx, k_grid, k_wts, G, W,
            S_has_k=False, return_scaled=False, tol=1.0e-10,
        )
    else:
        sigma = radial.get_sigma_diag(
            None, ctx, k_grid, k_wts, G, W, diag_cells,
            S_has_k=False, return_scaled=False, tol=1.0e-10,
        )
    radial.check_sigma_symm(ctx, sigma)
    return radial.symm_sigma(ctx, sigma)


def solve_fixed_W(
    ctx,
    k_grid,
    k_wts,
    e_k,
    W,
    n0,
    sigma0=None,
    mu0=None,
    fix_mu=False,
    conv_tol=1.0e-6,
    conv_tol_nelec=None,
    max_cycle=100,
    damp=0.8,
    diis_space=0,
    diis_start_cycle=0,
    diis_file=None,
    mu_max_jump=None,
    mu_bracket_policy=_MU_BRACKET_DEFAULT,
    diag_cells=None,
    high_freq_correction=True,
    fs_pin=False,
    fs_pin_k_F=None,
    fs_pin_mu0=None,
    fs_pin_shift0=None,
    verbose=0,
    stdout=None,
    chkfile=None,
    chk_metadata=None,
    cycle0=0,
    timings=None,
):
    """GW0 normal-state self-energy loop with fixed ``W``.

    Parameters
    ----------
    ctx : :class:`migdal.freq.FreqGrid`
        Sparse-IR transform context.
    k_grid, k_wts : ndarray
        Active radial momenta and integration weights.
    e_k : ndarray
        Band energies excluding the chemical potential.
    W : radial W precomputation object
        Fixed screened interaction table, for example
        :class:`migdal.radial.WPrecomp` or
        :class:`migdal.radial_qW.QWPrecomp`.
    n0 : float
        Target spin-summed density in Bohr^-3.
    timings : dict, optional
        DEBUG1 wall-time accumulator supplied by a driver.  When omitted, the
        SCF loop does not read the clock or update timing data.

    Notes
    -----
    ``high_freq_correction`` selects the density convention used both for
    solving ``mu`` and for the diagnostic electron-count residual ``dn``.
    ``fix_mu=True`` is retained for comparisons where the input branch should
    not be updated by the density solver.  When ``fs_pin=True``, DIIS packs the
    pinned self-energy together with its auxiliary absolute-gauge shift.
    If ``chkfile`` is set, the latest cycle input state is overwritten after
    each cycle on rank 0.
    """
    if conv_tol_nelec is None:
        conv_tol_nelec = model.default_nelec_tol(n0)
    mu_bracket_policy = _normalize_mu_bracket_policy(mu_bracket_policy)
    cycle0 = int(cycle0)
    if cycle0 < 0:
        raise ValueError("cycle0 must be non-negative.")
    if max_cycle < 1:
        raise ValueError("max_cycle must be at least 1.")
    if sigma0 is None:
        sigma = np.zeros((len(k_grid), ctx.nw), dtype=complex)
    else:
        sigma = np.asarray(sigma0, dtype=complex)
    if sigma.shape != (len(k_grid), ctx.nw):
        raise ValueError("sigma0 shape is inconsistent with k_grid and ctx.nw.")
    fs_pin = bool(fs_pin)
    if fs_pin:
        if fs_pin_k_F is None:
            raise ValueError("fs_pin=True requires fs_pin_k_F.")
        sigma, mu0, fs_shift = fs_pin_initial_state(
            ctx, sigma, mu0, k_grid, float(fs_pin_k_F), shift0=fs_pin_shift0
        )
        if mu0 is None and fs_pin_mu0 is not None:
            mu0 = float(fs_pin_mu0) - fs_shift
    else:
        fs_shift = 0.0
    if int(diis_space) > 0:
        adiis = lib.diis.DIIS(filename=diis_file)
        adiis.space = int(diis_space)
    else:
        adiis = None
    mu = mu0
    converged = False
    res_sigma = np.inf
    dn = np.inf
    n = np.nan
    prev_res_sigma = None
    log = mlog.new_logger(verbose=verbose, stdout=stdout)
    mlog.kv_table(
        log,
        "Normal-state SCF",
        (
            ("type", "GW0 fixed W"),
            ("nk / nw", "%d / %d" % (len(k_grid), ctx.nw)),
            ("max cycle", "%d" % int(max_cycle)),
            ("Sigma-mu tolerance", "%.3e" % conv_tol),
            ("mu density tolerance", "%.3e" % conv_tol_nelec),
            ("mu bracket policy", mu_bracket_policy),
            ("Fermi-surface pin", mlog.format_bool(fs_pin)),
            ("damping", "%.6g" % damp),
            ("DIIS space", "%d" % int(diis_space)),
            ("density convention", "high-frequency corrected"
             if high_freq_correction else "direct -2 G(beta)"),
        ),
        level=mlog.INFO,
    )
    mlog.table_header(
        log,
        "GW0 SCF cycles",
        _SCF_CYCLE_HEADER_FMT,
        ("cycle", "mu", "n", "dn", "res(S-mu)", "rate"),
        level=mlog.DEBUG,
    )

    def dump_cycle(cycle_number, sigma_state, mu_state, shift_state, conv_state):
        dump_scf_cycle_checkpoint(
            chkfile,
            method="GW0",
            cycle=cycle_number,
            ctx=ctx,
            k_grid=k_grid,
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

    for cycle in range(int(max_cycle)):
        cycle_label = cycle0 + cycle + 1
        if fix_mu:
            if mu0 is None:
                raise ValueError("fix_mu=True requires mu0.")
            mu_new = float(mu0)
        else:
            if fs_pin:
                mu_new = fs_pin_get_mu(
                    ctx, sigma, fs_shift, e_k, n0, k_grid, k_wts,
                    mu0=mu, tol=conv_tol_nelec,
                    high_freq_correction=high_freq_correction,
                    bracket_policy=mu_bracket_policy,
                    warning_log=log,
                )
            else:
                mu_new = get_mu(ctx, sigma, e_k, n0, k_grid, k_wts,
                                mu0=mu, tol=conv_tol_nelec,
                                high_freq_correction=high_freq_correction,
                                bracket_policy=mu_bracket_policy,
                                warning_log=log)
            if mu_max_jump is not None and mu is not None:
                dmu = abs(mu_new - mu)
                if dmu > float(mu_max_jump):
                    raise RuntimeError(
                        "chemical-potential branch jump detected: "
                        "|dmu|=%s > %s" % (dmu, mu_max_jump)
                    )
        G = ctx.get_G(e_k=e_k, mu=mu_new, sigma=sigma)
        if fs_pin:
            n = fs_pin_particle_number(
                ctx, G, k_grid, k_wts,
                e_k=e_k,
                mu=mu_new,
                shift=fs_shift,
                high_freq_correction=high_freq_correction,
            )
        else:
            n = particle_number(
                ctx, G, k_grid, k_wts,
                e_k=e_k,
                mu=mu_new,
                high_freq_correction=high_freq_correction,
            )
        sigma_t0 = mlog.wall_clock() if timings is not None else None
        sigma_raw = get_sigma(ctx, k_grid, k_wts, G, W, diag_cells=diag_cells)
        if timings is not None:
            mlog.record_wall(timings, "self_energy", sigma_t0)
        if fs_pin:
            sigma_raw_pin, mu_raw_pin, fs_shift_raw = fs_pin_project_with_shift(
                ctx, sigma_raw, mu_new, fs_shift, k_grid, float(fs_pin_k_F)
            )
            # Pair the input self-energy with the density-solved input mu.
            # Mu is a branch/search guess, not an independent mixer variable;
            # the residual is the gauge-invariant Sigma-mu map error.
            sigma_mu_err = sigma_mu_error(
                sigma_raw_pin, mu_raw_pin, sigma, mu_new
            )
        else:
            sigma_raw_pin = sigma_raw
            mu_raw_pin = mu_new
            fs_shift_raw = fs_shift
            # The evaluated input Green's function used the density-solved mu.
            sigma_mu_err = sigma_mu_error(sigma_raw, mu_new, sigma, mu_new)
        res_sigma = radial.max_abs(sigma_mu_err)
        dn = abs(n - n0)
        converged = (
            res_sigma < conv_tol and (fix_mu or dn <= conv_tol_nelec)
        )
        res_rate = _format_res_rate(res_sigma, prev_res_sigma)
        mlog.table_row(
            log,
            _SCF_CYCLE_ROW_FMT,
            (
                cycle_label,
                mu_new,
                n,
                dn,
                res_sigma,
                res_rate,
            ),
            level=mlog.DEBUG,
        )
        if converged:
            mu = mu_new
            dump_cycle(cycle_label, sigma, mu, fs_shift, converged)
            break
        if cycle + 1 == int(max_cycle):
            mu = mu_new
            dump_cycle(cycle_label, sigma, mu, fs_shift, converged)
            break
        prev_res_sigma = res_sigma
        if fs_pin:
            sigma_new = sigma_raw_pin
            old_shift = fs_shift
            shift_new = fs_shift_raw
            if damp != 0.0:
                sigma_new = sigma_new * (1.0 - damp) + sigma * damp
                shift_new = shift_new * (1.0 - damp) + fs_shift * damp
            if adiis is not None and cycle >= int(diis_start_cycle):
                state = pack_fs_pin_state(sigma_new, shift_new)
                state_diis = adiis.update(state, xerr=sigma_mu_err)
                if np.all(np.isfinite(state_diis)):
                    sigma_new, shift_new = unpack_fs_pin_state(
                        state_diis, sigma.shape
                    )
                else:
                    mlog.warning(
                        log,
                        "FS-pin DIIS produced a non-finite state; "
                        "using linearly mixed state for this cycle.",
                    )
            sigma, shift_new, delta = fs_pin_repin_state(
                ctx, sigma_new, shift_new, k_grid, float(fs_pin_k_F)
            )
            fs_shift = shift_new
            mu = fs_pin_transport_mu_guess(mu_new, old_shift, fs_shift)
        else:
            sigma_new = sigma_raw
            if damp != 0.0:
                sigma_new = sigma_new * (1.0 - damp) + sigma * damp
            if adiis is not None and cycle >= int(diis_start_cycle):
                sigma_new = adiis.update(sigma_new, xerr=sigma_mu_err)
            sigma = radial.symm_sigma(ctx, sigma_new)
            mu = mu_new
        dump_cycle(cycle_label, sigma, mu, fs_shift, converged)
    extra = {}
    if fs_pin:
        extra.update(
            fs_pin_shift=float(fs_shift),
            fs_pin_residual=fs_pin_shift(ctx, sigma, k_grid, float(fs_pin_k_F)),
            mu_abs=float(mu) + float(fs_shift),
        )
    return SCFResult(
        sigma=sigma,
        converged=bool(converged),
        cycles=cycle_label,
        res_sigma=float(res_sigma),
        dn=float(dn),
        mu=float(mu),
        n=float(n),
        extra=extra,
    )
