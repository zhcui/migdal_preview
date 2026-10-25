r"""Pure-Python qW primitives for Coulomb-singular radial interactions.

The radial self-energy kernels need interval integrals of ``q * W(q)`` over
``[a, b] = [|k - p|, k + p]``.  A Coulomb tail is singular in ``W`` but only
logarithmic in that integrand:

.. math::

    W(q) = A / q^2 + R(q), \qquad q W(q) = A / q + q R(q).

``QWPrecomp`` stores ``A`` separately, linearly interpolates the regular
integrand ``h(q) = q R(q)``, and adds the exact ``A * (log(b) - log(a))``
contribution for non-diagonal source/target pairs.  The ``a = 0`` diagonal
block is finite only after cell averaging, so callers must provide
``diag_cells`` for genuinely singular interactions.

This module deliberately has no MPI dependency; ``radial.py`` owns the MPI
dispatch and imports this file only for the q-singular primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import weakref

import numpy as np
from numpy.polynomial.legendre import leggauss

from migdal import log as mlog


__all__ = [
    "SingularFitDiagnostics",
    "estimate_screened_singular_coeff_from_pi",
    "fit_singular_coeff",
    "build_q_singular_precomp",
    "clear_qw_static_block_caches",
    "QWPrecomp",
]


_QW_PRECOMP_INSTANCES = weakref.WeakSet()


def clear_qw_static_block_caches():
    """Clear static block caches held by live qW precomp objects."""
    cleared = 0
    for precomp in list(_QW_PRECOMP_INSTANCES):
        cleared += precomp.clear_static_block_cache()
    return cleared


@dataclass
class SingularFitDiagnostics:
    """Diagnostics for the low-q ``A/q**2`` split used by ``q_singular`` W.

    ``fit_singular_coeff`` fits ``q**2 * values(q)`` and interprets the
    intercept as ``A = lim(q -> 0) q**2 values(q)``.  The residual fields measure
    how well the first ``nfit`` q points are represented by that low-q
    polynomial model.
    """

    nfit: int = 0
    degree: int = 0
    max_rel_residual: float = 0.0
    max_abs_residual: float = 0.0
    worst_index: tuple[int, ...] = ()
    worst_v: float = np.nan
    static_coulomb_coeff: complex = 0.0
    static_w_coeff: complex = 0.0
    max_abs_w_coeff: float = 0.0
    max_abs_fit_w_coeff: float = 0.0
    max_abs_dynamic_coeff: float = 0.0
    coefficient_source: str = "fit_w"
    head_min_abs_denom: float = np.inf
    head_regularized_count: int = 0
    _rel_residual: np.ndarray | None = field(default=None, repr=False)
    _abs_residual: np.ndarray | None = field(default=None, repr=False)


def max_abs(a):
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a)))


def _result_dtype(*values):
    dtypes = [np.asarray(value).dtype for value in values if value is not None]
    return np.result_type(*dtypes)


def fit_singular_coeff(q_grid, values, nfit=6, degree=1,
                       rtol=1.0e-10, atol=0.0,
                       return_diagnostics=False):
    r"""Fit the Coulomb coefficient ``A = lim(q -> 0) q**2 * values(q)``.

    The fit is done on ``z(q) = q**2 * values(q)`` as a polynomial in
    ``x = q**2`` over the first positive q points.  The returned coefficient is
    the polynomial intercept.  Scalar static tables and frequency-dependent
    ``(nq, nv)`` tables use the same first-axis convention.
    """
    q = np.asarray(q_grid, dtype=float)
    y = np.asarray(values)
    if q.ndim != 1 or y.shape[0] != q.size:
        raise ValueError("values must have q_grid as its first axis.")
    pos = np.flatnonzero(q > 0.0)
    if pos.size == 0:
        raise ValueError("singular coefficient fit requires positive q points.")
    nfit = min(max(int(nfit), 1), pos.size)
    idx = pos[:nfit]
    x = q[idx] ** 2
    z = (q[idx] ** 2).reshape((nfit,) + (1,) * (y.ndim - 1)) * y[idx]
    degree = min(max(int(degree), 0), nfit - 1)
    z2d = z.reshape(nfit, -1)
    if degree == 0:
        coeff_flat = np.mean(z2d, axis=0)
        pred = np.broadcast_to(coeff_flat[None, :], z2d.shape)
    else:
        mat = np.vander(x, degree + 1, increasing=True)
        sol = np.linalg.lstsq(mat, z2d, rcond=None)[0]
        coeff_flat = sol[0]
        pred = mat @ sol
    coeff = coeff_flat.reshape(y.shape[1:])
    scale = np.max(np.abs(z2d), axis=0)
    scale_shaped = scale.reshape(y.shape[1:])
    abs_resid = np.max(np.abs(z2d - pred), axis=0)
    rel_resid = np.zeros_like(abs_resid, dtype=float)
    nonzero = scale > 0.0
    rel_resid[nonzero] = abs_resid[nonzero] / scale[nonzero]
    rel_resid[~nonzero] = np.where(abs_resid[~nonzero] == 0.0, 0.0, np.inf)
    coeff = np.asarray(coeff)
    tiny = np.abs(coeff) <= (float(atol) + float(rtol) * scale_shaped)
    if np.any(tiny):
        coeff = np.array(coeff, copy=True)
        coeff[tiny] = 0.0
    if return_diagnostics:
        if rel_resid.size:
            worst_flat = int(np.argmax(rel_resid))
            coeff_shape = coeff.shape if coeff.shape else (1,)
            worst_index = np.unravel_index(worst_flat, coeff_shape)
            if coeff.shape == ():
                worst_index = ()
            max_rel_residual = float(rel_resid[worst_flat])
            max_abs_residual = float(np.max(abs_resid))
        else:
            worst_index = ()
            max_rel_residual = 0.0
            max_abs_residual = 0.0
        diagnostics = SingularFitDiagnostics(
            nfit=int(nfit),
            degree=int(degree),
            max_rel_residual=max_rel_residual,
            max_abs_residual=max_abs_residual,
            worst_index=tuple(int(i) for i in worst_index),
            _rel_residual=rel_resid.reshape(y.shape[1:]),
            _abs_residual=abs_resid.reshape(y.shape[1:]),
        )
        return coeff, diagnostics
    return coeff


def _fit_lowq_intercept(q_grid, values, nfit=6, degree=1):
    """Fit the low-q intercept of ``values(q)`` as a polynomial in ``q**2``."""
    q = np.asarray(q_grid, dtype=float)
    y = np.asarray(values)
    if q.ndim != 1 or y.shape[0] != q.size:
        raise ValueError("values must have q_grid as its first axis.")
    pos = np.flatnonzero(q > 0.0)
    if pos.size == 0:
        raise ValueError("low-q intercept fit requires positive q points.")
    nfit = min(max(int(nfit), 1), pos.size)
    idx = pos[:nfit]
    x = q[idx] ** 2
    y2d = y[idx].reshape(nfit, -1)
    degree = min(max(int(degree), 0), nfit - 1)
    if degree == 0:
        coeff_flat = np.mean(y2d, axis=0)
    else:
        mat = np.vander(x, degree + 1, increasing=True)
        coeff_flat = np.linalg.lstsq(mat, y2d, rcond=None)[0][0]
    return coeff_flat.reshape(y.shape[1:])


def _regularize_head_denominator(denom, policy, denom_min):
    """Apply the RPA denominator policy to low-q head denominators."""
    denom = np.asarray(denom)
    out = np.array(denom, copy=True)
    if policy == "none":
        return out, 0
    if policy not in ("fail", "legacy_clip", "sign_regularize"):
        raise ValueError("unknown denom_policy %r" % policy)
    denom_min = float(denom_min)
    if denom_min <= 0.0:
        raise ValueError("denom_min must be positive.")
    small = np.abs(out) < denom_min
    count = int(np.count_nonzero(small))
    if not count:
        return out, 0
    if policy == "fail":
        raise FloatingPointError(
            "q_singular head denominator below denom_min: min_abs=%s"
            % float(np.min(np.abs(out)))
        )
    if policy == "legacy_clip":
        out[small] = denom_min
        return out, count
    vals = out[small]
    if np.iscomplexobj(out):
        direction = np.ones_like(vals, dtype=out.dtype)
        nonzero = np.abs(vals) > 0.0
        direction[nonzero] = vals[nonzero] / np.abs(vals[nonzero])
        out[small] = direction * denom_min
    else:
        direction = np.sign(vals)
        direction[direction == 0.0] = 1.0
        out[small] = direction * denom_min
    return out, count


def estimate_screened_singular_coeff_from_pi(
    q_grid,
    Pi,
    Vtot,
    vs,
    *,
    vtot_head=None,
    nfit=6,
    degree=1,
    denom_policy="fail",
    denom_min=1.0e-10,
    static_v_tol=1.0e-12,
    static_rtol=1.0e-10,
    static_atol=0.0,
):
    r"""Estimate ``A_W = lim(q -> 0) q**2 W(q, i nu)`` from the Pi head.

    The low-q input is

    ``Vtot(q, i nu) = B(i nu) / q**2 + regular`` and
    ``Pi(q, i nu) = P0(i nu) + C(i nu) q**2 + ...``.  Static screening with
    nonzero ``P0`` removes the ``1/q**2`` tail and ``A_W = 0``.  Otherwise,

    ``A_W(i nu) = B(i nu) / (1 - C(i nu) * B(i nu))``.

    If ``vtot_head`` is supplied, it is used as ``B`` directly.  Otherwise ``B``
    is fitted from the first positive-q rows of ``q**2 * Vtot`` as a generic
    fallback for callers that only have tabulated interactions.

    Charge conservation normally makes ``P0=0`` at nonzero bosonic Matsubara
    frequencies; production code follows that constraint instead of fitting a
    finite-frequency ``P0`` that would be sensitive to curvature and numerical
    noise.
    """
    q = np.asarray(q_grid, dtype=float)
    Pi = np.asarray(Pi)
    Vtot = np.asarray(Vtot)
    vs = np.asarray(vs, dtype=float)
    expected = (q.size, vs.size)
    if Pi.shape != expected:
        raise ValueError("Pi must have shape %s; got %s." % (expected, Pi.shape))
    if Vtot.shape != expected:
        raise ValueError("Vtot must have shape %s; got %s." % (expected, Vtot.shape))
    if np.any(q <= 0.0):
        raise ValueError("q_singular head estimate requires positive q_grid.")

    dtype = np.result_type(Pi, Vtot, 0.0 if vtot_head is None else vtot_head)
    if vtot_head is None:
        B = fit_singular_coeff(q, Vtot, nfit=nfit, degree=degree)
    else:
        B = np.asarray(vtot_head)
        if B.shape == ():
            B = np.full(vs.size, B, dtype=dtype)
        if B.shape != (vs.size,):
            raise ValueError("vtot_head must have shape (%d,)." % vs.size)
    B = np.asarray(B, dtype=dtype)
    A_w = np.zeros(vs.size, dtype=dtype)
    q2 = q * q
    reg_count = 0
    min_abs_denom = np.inf

    finite_v = np.abs(vs) > float(static_v_tol)
    if np.any(finite_v):
        C = _fit_lowq_intercept(
            q, Pi[:, finite_v] / q2[:, None], nfit=nfit, degree=degree)
        den = 1.0 - C * B[finite_v]
        den, count = _regularize_head_denominator(
            den, denom_policy, denom_min)
        reg_count += count
        min_abs_denom = min(min_abs_denom, float(np.min(np.abs(den))))
        A_w[finite_v] = B[finite_v] / den

    static_v = ~finite_v
    if np.any(static_v):
        Pi_static = Pi[:, static_v]
        P0 = _fit_lowq_intercept(q, Pi_static, nfit=nfit, degree=degree)
        nfit_eff = min(max(int(nfit), 1), q.size)
        scale = np.max(np.abs(Pi_static[:nfit_eff]), axis=0)
        screened = np.abs(P0) > (
            float(static_atol) + float(static_rtol) * scale
        )
        unscreened = ~screened
        if np.any(unscreened):
            static_idx = np.flatnonzero(static_v)
            Pi_q2 = (
                Pi_static[:, unscreened]
                - P0[unscreened][None, :]
            ) / q2[:, None]
            C = _fit_lowq_intercept(q, Pi_q2, nfit=nfit, degree=degree)
            den = 1.0 - C * B[static_idx[unscreened]]
            den, count = _regularize_head_denominator(
                den, denom_policy, denom_min)
            reg_count += count
            min_abs_denom = min(min_abs_denom, float(np.min(np.abs(den))))
            A_w[static_idx[unscreened]] = B[static_idx[unscreened]] / den

    return A_w, {
        "head_min_abs_denom": float(min_abs_denom),
        "head_regularized_count": int(reg_count),
    }


def build_q_singular_precomp(ctx, q_grid, W, Vc, *,
                             static_singular_coeff=None,
                             screened_singular_coeff=None,
                             singular_head_source="fit_w",
                             singular_fit_n=6, singular_fit_degree=1,
                             singular_fit_warn_rtol=1.0e-3,
                             singular_fit_warn_atol=1.0e-10,
                             warning_log=None):
    r"""Build a Coulomb-split qW precomputation from frequency-domain ``W``.

    The screened table is decomposed as

    .. math::

        \begin{aligned}
        V_c(q) &= A_c / q^2 + V_{\mathrm{reg}}(q),\\
        W(q, i\nu) - V_c(q) &= A_d(i\nu) / q^2
            + D_{\mathrm{reg}}(q, i\nu).
        \end{aligned}

    where ``A_d(i nu) = A_W(i nu) - A_c`` and
    ``A_W(i nu) = lim(q -> 0) q**2 W(q, i nu)``.  The regular dynamic table and
    the singular coefficient are transformed to tau separately; this avoids
    letting the sparse-IR transform see a large, subtractive ``1/q**2`` tail.
    ``warning_log`` controls the optional fit-residual warning; callers can
    pass a quiet logger to silence it.
    """
    singular_head_source = str(singular_head_source)
    if singular_head_source not in ("fit_w", "pi_head", "provided"):
        raise ValueError("unknown singular_head_source %r" % singular_head_source)
    q_grid = np.asarray(q_grid, dtype=float)
    W = np.asarray(W)
    Vc = np.asarray(Vc)
    if q_grid.ndim != 1:
        raise ValueError("q_grid must be one-dimensional.")
    expected_shape = (q_grid.size, len(ctx.vs))
    if W.shape != expected_shape:
        raise ValueError("W must have shape %s; got %s." % (expected_shape, W.shape))
    if Vc.shape != (q_grid.size,):
        raise ValueError("Vc must have shape (%d,); got %s." % (q_grid.size, Vc.shape))
    if np.any(q_grid <= 0.0):
        raise ValueError("q_singular mode requires strictly positive q_grid.")
    if static_singular_coeff is None:
        static_singular_coeff = fit_singular_coeff(
            q_grid, Vc, nfit=singular_fit_n, degree=singular_fit_degree)
    A_w_fit, diagnostics = fit_singular_coeff(
        q_grid, W, nfit=singular_fit_n, degree=singular_fit_degree,
        return_diagnostics=True)
    dtype = _result_dtype(W, Vc, static_singular_coeff, A_w_fit,
                          screened_singular_coeff)
    W = np.asarray(W, dtype=dtype)
    Vc = np.asarray(Vc, dtype=dtype)
    A_c = np.asarray(static_singular_coeff, dtype=dtype).reshape(())
    A_w_fit = np.asarray(A_w_fit, dtype=dtype)
    if screened_singular_coeff is None:
        A_w = A_w_fit
        singular_head_source = "fit_w"
    else:
        A_w = np.asarray(screened_singular_coeff, dtype=dtype)
        if A_w.shape != (len(ctx.vs),):
            raise ValueError("screened_singular_coeff must have shape (nv,).")
        if singular_head_source == "fit_w":
            singular_head_source = "provided"
    A_d = A_w - A_c
    if diagnostics.worst_index and len(ctx.vs) > diagnostics.worst_index[0]:
        diagnostics.worst_v = float(ctx.vs[diagnostics.worst_index[0]])
    elif len(ctx.vs):
        diagnostics.worst_v = float(ctx.vs[0])
    zero_iv = int(np.argmin(np.abs(ctx.vs))) if len(ctx.vs) else 0
    diagnostics.static_coulomb_coeff = complex(A_c)
    diagnostics.static_w_coeff = complex(np.asarray(A_w).reshape(-1)[zero_iv])
    diagnostics.max_abs_w_coeff = max_abs(A_w)
    diagnostics.max_abs_fit_w_coeff = max_abs(A_w_fit)
    diagnostics.max_abs_dynamic_coeff = max_abs(A_d)
    diagnostics.coefficient_source = str(singular_head_source)
    if diagnostics.coefficient_source == "fit_w":
        _warn_bad_singular_fit(
            diagnostics, singular_fit_warn_rtol, singular_fit_warn_atol,
            warning_log=warning_log)

    q2 = q_grid * q_grid
    sta_reg = Vc - A_c / q2
    dyn_reg_w = (q2[:, None] * W - A_w[None, :]) / q2[:, None] - sta_reg[:, None]
    dyn_reg_tau = ctx.w2t_b2f(dyn_reg_w)
    A_d_tau = ctx.w2t_b2f(A_d[None, :])[0]
    dyn_full_tau = dyn_reg_tau + A_d_tau[None, :] / q2[:, None]
    precomp = QWPrecomp(
        q_grid,
        dyn_full_tau,
        Vc,
        singular_dyn=A_d_tau,
        singular_sta=A_c,
        residual_dyn=dyn_reg_tau,
        residual_sta=sta_reg,
        singular_dyn_w=A_d,
    )
    return precomp, diagnostics


def _warn_bad_singular_fit(diagnostics, warn_rtol, warn_atol, warning_log=None):
    if diagnostics is None:
        return
    rel = getattr(diagnostics, "_rel_residual", None)
    abs_resid = getattr(diagnostics, "_abs_residual", None)
    if rel is not None and abs_resid is not None:
        bad = (np.asarray(rel) > float(warn_rtol)) & (
            np.asarray(abs_resid) > float(warn_atol)
        )
        if not np.any(bad):
            return
    else:
        if diagnostics.max_rel_residual <= float(warn_rtol):
            return
        if diagnostics.max_abs_residual <= float(warn_atol):
            return
    mlog.warning(
        warning_log,
        "q_singular low-q singular coefficient fit has a large residual: "
        "max_rel_residual=%.3e max_abs_residual=%.3e worst_v=%.6e. "
        "Inspect DenomMetrics.singular_fit and consider more low-q points "
        "or a safer q grid."
        % (
            diagnostics.max_rel_residual,
            diagnostics.max_abs_residual,
            diagnostics.worst_v,
        ),
    )


def _insert_zero_grid(grid):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("primitive grid must contain at least two points.")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("primitive grid must be strictly increasing.")
    if grid[0] == 0.0:
        return grid, False
    if grid[0] < 0.0:
        raise ValueError("primitive grid must be non-negative.")
    return np.concatenate(([0.0], grid)), True


def _array_value_key(arr):
    arr = np.ascontiguousarray(arr)
    return arr.shape, arr.dtype.str, arr.tobytes()


def _safe_z_log_abs_minus_z(z):
    """Return ``F(z) = z log(abs(z)) - z`` with the continuous value ``F(0)=0``."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    mask = z != 0.0
    out[mask] = z[mask] * np.log(np.abs(z[mask])) - z[mask]
    return out


def _singular_cell_average(coeff, p, lo, hi):
    r"""Average the singular interval ``A log((k+p)/|k-p|)`` over a k cell.

    For a diagonal source point ``p`` and target cell ``[lo, hi]``, the
    singular radial block is

    .. math::

        \frac{A}{hi-lo} \int_{lo}^{hi}
        \left[\log(k+p) - \log|k-p|\right] dk.

    ``_safe_z_log_abs_minus_z`` supplies the antiderivative
    ``F(z) = z log|z| - z`` on both sides of ``k = p``.
    """
    coeff = np.asarray(coeff)
    if max_abs(coeff) == 0.0:
        return np.zeros_like(coeff)
    factor = _singular_cell_average_factor(p, lo, hi)
    return coeff * factor


def _singular_cell_average_factor(p, lo, hi):
    """Return the coefficient-free singular diagonal-cell average."""
    p = np.asarray(p, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    if np.any(width <= 0.0):
        raise ValueError("diagonal cell must have positive width.")
    val = (
        _safe_z_log_abs_minus_z(hi + p)
        - _safe_z_log_abs_minus_z(lo + p)
        - _safe_z_log_abs_minus_z(hi - p)
        + _safe_z_log_abs_minus_z(lo - p)
    )
    return val / width


def _integrate_cell_average(func, kp, lo, hi, ncell, xw=None):
    """Average a regular radial block, splitting the quadrature at ``k = kp``."""
    total = None
    x, w = leggauss(ncell) if xw is None else xw
    for a, b in ((lo, min(kp, hi)), (max(kp, lo), hi)):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        vals = func(mid + half * x, kp)
        contrib = np.tensordot(half * w, vals, axes=(0, 0))
        total = contrib if total is None else total + contrib
    if total is None:
        raise ValueError("empty diagonal cell for k=%.16e" % kp)
    return total / (hi - lo)


class QWPrecomp:
    r"""qW primitives for ``W(q) = A/q**2 + R(q)`` radial kernels.

    ``QWPrecomp`` has the same public protocol as ``radial.WPrecomp`` but stores
    the q-singular representation explicitly:

    .. math::

        \begin{aligned}
        D(q,\tau) &= A_D(\tau) / q^2 + D_{\mathrm{reg}}(q,\tau),\\
        V_c(q) &= A_c / q^2 + V_{\mathrm{reg}}(q).
        \end{aligned}

    The radial kernels integrate ``q * D`` and ``q * V_c``.  This class
    therefore builds primitives of ``h(q) = q * D_reg(q)`` and
    ``h(q) = q * V_reg(q)``, plus exact logarithmic primitives for
    ``A_D / q`` and ``A_c / q``.  ``dW_tau`` and ``Vc`` are the full tables for
    interface compatibility; callers that already know the split should pass
    ``residual_dyn`` and ``residual_sta`` so the regular primitives are built
    from the residual tables directly.  Production code should normally enter
    through ``build_q_singular_precomp``.
    """

    def __init__(self, q_grid, dW_tau, Vc, *, singular_dyn=None,
                 singular_sta=0.0, residual_dyn=None, residual_sta=None,
                 singular_dyn_w=None):
        dtype = _result_dtype(
            dW_tau, Vc, singular_dyn, singular_sta,
            residual_dyn, residual_sta, singular_dyn_w,
        )
        self.q = np.asarray(q_grid, dtype=float)
        self.yD = np.asarray(dW_tau, dtype=dtype)
        if self.q.ndim != 1 or self.yD.ndim != 2 or self.yD.shape[0] != self.q.size:
            raise ValueError("dW_tau must be (nq, nt) with nq=len(q_grid).")
        self.nq, self.nt = self.yD.shape
        self.yS = np.asarray(Vc, dtype=dtype)
        if self.yS.shape != (self.nq,):
            raise ValueError("Vc must have shape (nq,).")
        self.interp_mode = "q_singular"
        self.dyn_dtype = self.yD.dtype
        self.sta_dtype = self.yS.dtype
        self._static_block_cache = {}
        self._cython_cache = None
        self._singular_dyn_w = None
        self.singular_dyn = np.zeros(self.nt, dtype=self.yD.dtype)
        if singular_dyn is not None:
            self.singular_dyn = singular_dyn
        self.singular_sta = singular_sta
        self.singular_dyn_w = singular_dyn_w
        self.qL = self.q[:-1]
        self.qR = self.q[1:]
        self.dq = self.qR - self.qL
        if not np.all(self.dq > 0.0):
            raise ValueError("q_grid must be strictly increasing.")
        if np.any(self.q <= 0.0):
            raise ValueError("q_singular mode requires a strictly positive q_grid.")
        self.q2 = self.q * self.q
        self.yD_reg, self.yS_reg = self._regular_tables(residual_dyn, residual_sta)
        self._init_regular_primitives()
        _QW_PRECOMP_INSTANCES.add(self)

    @property
    def singular_dyn(self):
        return self._singular_dyn

    @singular_dyn.setter
    def singular_dyn(self, value):
        arr = np.asarray(value, dtype=self.yD.dtype)
        if arr.shape != (self.nt,):
            raise ValueError("singular_dyn must have shape (nt,).")
        arr = np.array(arr, dtype=self.yD.dtype, copy=True)
        arr.setflags(write=False)
        self._singular_dyn = arr
        self._has_singular_dyn = max_abs(arr) > 0.0
        self._singular_dyn_w = None
        self._cython_cache = None

    @property
    def singular_dyn_w(self):
        return self._singular_dyn_w

    @singular_dyn_w.setter
    def singular_dyn_w(self, value):
        if value is None:
            self._singular_dyn_w = None
            return
        arr = np.array(np.asarray(value, dtype=self.yD.dtype), copy=True)
        if arr.ndim != 1:
            raise ValueError("singular_dyn_w must be one-dimensional.")
        arr.setflags(write=False)
        self._singular_dyn_w = arr

    @property
    def singular_sta(self):
        return self._singular_sta

    @singular_sta.setter
    def singular_sta(self, value):
        arr = np.array(
            np.asarray(value, dtype=self.yD.dtype).reshape(()),
            dtype=self.yD.dtype,
            copy=True,
        )
        arr.setflags(write=False)
        self._singular_sta = arr
        self._has_singular_sta = abs(arr) > 0.0
        self._cython_cache = None
        self.clear_static_block_cache()

    @property
    def uses_legacy_primitives(self):
        return False

    @property
    def has_singularity(self):
        return self._has_singular_dyn or self._has_singular_sta

    @property
    def requires_diag_cells(self):
        return self.has_singularity

    def _regular_tables(self, residual_dyn, residual_sta):
        """Return regular residual tables after removing ``A/q**2`` tails."""
        q2 = self.q2
        if residual_dyn is None:
            residual_dyn = self.yD - self.singular_dyn[None, :] / q2[:, None]
        if residual_sta is None:
            residual_sta = self.yS - self.singular_sta / q2
        yD_reg = np.asarray(residual_dyn, dtype=self.yD.dtype)
        yS_reg = np.asarray(residual_sta, dtype=self.yD.dtype)
        if yD_reg.shape != self.yD.shape:
            raise ValueError("residual_dyn has inconsistent shape.")
        if yS_reg.shape != self.yS.shape:
            raise ValueError("residual_sta has inconsistent shape.")
        return yD_reg, yS_reg

    def _init_regular_primitives(self):
        """Build primitives of the regular qW integrands ``h(q)=q*R(q)``."""
        self.qh, inserted = _insert_zero_grid(self.q)
        if inserted:
            # After the Coulomb tail is removed, isotropic analytic kernels have
            # finite ``R(0)`` and hence ``h(0)=q*R(q)|_{q=0}=0``.
            yD = np.vstack([
                np.zeros((1, self.nt), dtype=self.yD.dtype),
                self.yD_reg,
            ])
            yS = np.concatenate([
                np.zeros(1, dtype=self.yD.dtype),
                self.yS_reg,
            ])
        else:
            yD = self.yD_reg
            yS = self.yS_reg
        self.hD = self.qh[:, None] * yD
        self.hS = self.qh * yS
        dh = np.diff(self.qh)
        self.shD = (self.hD[1:] - self.hD[:-1]) / dh[:, None]
        self.shS = (self.hS[1:] - self.hS[:-1]) / dh
        segD = self.hD[:-1] * dh[:, None] + 0.5 * self.shD * dh[:, None] ** 2
        segS = self.hS[:-1] * dh + 0.5 * self.shS * dh ** 2
        self.PhD = np.zeros_like(self.hD)
        self.PhS = np.zeros_like(self.hS)
        self.PhD[1:] = np.cumsum(segD, axis=0)
        self.PhS[1:] = np.cumsum(segS)
        qsegD = (
            self.PhD[:-1] * dh[:, None]
            + 0.5 * self.hD[:-1] * dh[:, None] ** 2
            + (self.shD / 6.0) * dh[:, None] ** 3
        )
        qsegS = (
            self.PhS[:-1] * dh
            + 0.5 * self.hS[:-1] * dh ** 2
            + (self.shS / 6.0) * dh ** 3
        )
        self.QhD = np.zeros_like(self.hD)
        self.QhS = np.zeros_like(self.hS)
        self.QhD[1:] = np.cumsum(qsegD, axis=0)
        self.QhS[1:] = np.cumsum(qsegS)
        self._qh_min = self.qh[0]
        self._qh_max_clip = np.nextafter(self.qh[-1], -np.inf)
        self._qh_last_segment = len(self.qh) - 2
        self._cython_cache = None

    def _primitive_grid_args(self, q_star):
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self._qh_min, self._qh_max_clip))
        m = np.searchsorted(self.qh, q, side="right") - 1
        m = np.clip(m, 0, self._qh_last_segment)
        dx = q - self.qh[m]
        return q, m, dx, scalar

    def _second_primitive_grid_args(self, q_star):
        q_raw = np.asarray(q_star, dtype=float)
        scalar = q_raw.ndim == 0
        q_raw = np.atleast_1d(q_raw)
        q = np.clip(q_raw, self._qh_min, self._qh_max_clip)
        m = np.searchsorted(self.qh, q, side="right") - 1
        m = np.clip(m, 0, self._qh_last_segment)
        dx = q - self.qh[m]
        return q_raw, q, m, dx, scalar

    def _primitive_h_dyn(self, q_star):
        _, m, dx, scalar = self._primitive_grid_args(q_star)
        dx_col = dx[:, None]
        out = self.PhD[m] + dx_col * self.hD[m]
        out += 0.5 * dx_col * dx_col * self.shD[m]
        return out[0] if scalar else out

    def _primitive_h_sta(self, q_star):
        _, m, dx, scalar = self._primitive_grid_args(q_star)
        out = self.PhS[m] + dx * self.hS[m] + 0.5 * dx * dx * self.shS[m]
        return out[0] if scalar else out

    def _primitive_h_dyn_sta(self, q_star):
        _, m, dx, scalar = self._primitive_grid_args(q_star)
        dx_col = dx[:, None]
        dyn = self.PhD[m] + dx_col * self.hD[m]
        dyn += 0.5 * dx_col * dx_col * self.shD[m]
        sta = self.PhS[m] + dx * self.hS[m] + 0.5 * dx * dx * self.shS[m]
        if scalar:
            return dyn[0], sta[0]
        return dyn, sta

    def _second_primitive_h_dyn(self, q_star):
        q_raw, q, m, dx, scalar = self._second_primitive_grid_args(q_star)
        dx_col = dx[:, None]
        prim = self.PhD[m] + dx_col * self.hD[m]
        prim += 0.5 * dx_col * dx_col * self.shD[m]
        out = self.QhD[m] + dx_col * self.PhD[m]
        out += 0.5 * dx_col * dx_col * self.hD[m]
        out += (dx_col * dx_col * dx_col / 6.0) * self.shD[m]
        out += (q_raw - q)[:, None] * prim
        return out[0] if scalar else out

    def _second_primitive_h_sta(self, q_star):
        q_raw, q, m, dx, scalar = self._second_primitive_grid_args(q_star)
        prim = self.PhS[m] + dx * self.hS[m] + 0.5 * dx * dx * self.shS[m]
        out = self.QhS[m] + dx * self.PhS[m]
        out += 0.5 * dx * dx * self.hS[m]
        out += (dx * dx * dx / 6.0) * self.shS[m]
        out += (q_raw - q) * prim
        return out[0] if scalar else out

    def _second_primitive_h_dyn_sta(self, q_star):
        q_raw, q, m, dx, scalar = self._second_primitive_grid_args(q_star)
        dx_col = dx[:, None]
        prim_dyn = self.PhD[m] + dx_col * self.hD[m]
        prim_dyn += 0.5 * dx_col * dx_col * self.shD[m]
        dyn = self.QhD[m] + dx_col * self.PhD[m]
        dyn += 0.5 * dx_col * dx_col * self.hD[m]
        dyn += (dx_col * dx_col * dx_col / 6.0) * self.shD[m]
        dyn += (q_raw - q)[:, None] * prim_dyn
        prim_sta = self.PhS[m] + dx * self.hS[m] + 0.5 * dx * dx * self.shS[m]
        sta = self.QhS[m] + dx * self.PhS[m]
        sta += 0.5 * dx * dx * self.hS[m]
        sta += (dx * dx * dx / 6.0) * self.shS[m]
        sta += (q_raw - q) * prim_sta
        if scalar:
            return dyn[0], sta[0]
        return dyn, sta

    def _primitive_dyn(self, q_star):
        return self._primitive_h_dyn(q_star)

    def _primitive_sta(self, q_star):
        return self._primitive_h_sta(q_star)

    def _singular_log_dyn(self, a, b):
        scalar = np.asarray(a).ndim == 0
        if not self._has_singular_dyn:
            out = np.zeros((np.size(a), self.nt), dtype=self.yD.dtype)
            return out[0] if scalar else out
        log_interval = np.atleast_1d(self._singular_log_interval(a, b))
        out = log_interval[:, None] * self.singular_dyn[None, :]
        return out[0] if scalar else out

    def _singular_log_sta(self, a, b):
        scalar = np.asarray(a).ndim == 0
        if not self._has_singular_sta:
            out = np.zeros(np.size(a), dtype=self.yD.dtype)
            return out[0] if scalar else out
        return self._singular_log_interval(a, b) * self.singular_sta

    def _singular_log_interval(self, a, b):
        """Return ``log(b) - log(a)`` for the non-diagonal singular interval."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        scalar = a.ndim == 0
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)
        if np.any(a <= 0.0):
            raise FloatingPointError(
                "singular radial diagonal block requires diag_cells."
            )
        with np.errstate(divide="raise", invalid="raise"):
            out = np.log(b) - np.log(a)
        return out[0] if scalar else out

    def integral_block_dyn(self, k_vec, kp):
        """Return ``int_a^b dq q*D(q,tau)`` with the exact singular log term."""
        k = np.asarray(k_vec, dtype=float)
        a = np.abs(k - kp)
        b = k + kp
        out = self.integral_block_dyn_regular(k, kp)
        if self._has_singular_dyn:
            out = out + self._singular_log_dyn(a, b)
        return out

    def integral_block_sta(self, k_vec, kp):
        """Return ``int_a^b dq q*Vc(q)`` with the exact singular log term."""
        k = np.asarray(k_vec, dtype=float)
        a = np.abs(k - kp)
        b = k + kp
        out = self.integral_block_sta_regular(k, kp)
        if self._has_singular_sta:
            out = out + self._singular_log_sta(a, b)
        return out

    def integral_block_dyn_regular(self, k_vec, kp):
        """Return the regular part of ``int_a^b dq q*D_reg(q,tau)``."""
        k = np.asarray(k_vec, dtype=float)
        return self._primitive_dyn(k + kp) - self._primitive_dyn(np.abs(k - kp))

    def integral_block_sta_regular(self, k_vec, kp):
        """Return the regular part of ``int_a^b dq q*V_reg(q)``."""
        k = np.asarray(k_vec, dtype=float)
        return self._primitive_sta(k + kp) - self._primitive_sta(np.abs(k - kp))

    def integral_blocks_regular(self, k_vec, kp):
        """Return dynamic and static regular radial blocks for one source point."""
        k = np.asarray(k_vec, dtype=float)
        q_hi = k + kp
        q_lo = np.abs(k - kp)
        dyn_hi, sta_hi = self._primitive_h_dyn_sta(q_hi)
        dyn_lo, sta_lo = self._primitive_h_dyn_sta(q_lo)
        return dyn_hi - dyn_lo, sta_hi - sta_lo

    def integral_blocks(self, k_vec, kp):
        """Return full dynamic and static radial blocks for one source point."""
        k = np.asarray(k_vec, dtype=float)
        a = np.abs(k - kp)
        b = k + kp
        dyn_hi, sta_hi = self._primitive_h_dyn_sta(b)
        dyn_lo, sta_lo = self._primitive_h_dyn_sta(a)
        dyn = dyn_hi - dyn_lo
        sta = sta_hi - sta_lo
        if self.has_singularity:
            scalar = a.ndim == 0
            log_interval = self._singular_log_interval(a, b)
            if self._has_singular_dyn:
                if scalar:
                    dyn = dyn + log_interval * self.singular_dyn
                else:
                    dyn = dyn + log_interval[:, None] * self.singular_dyn[None, :]
            if self._has_singular_sta:
                sta = sta + log_interval * self.singular_sta
        return dyn, sta

    def integral_block_dyn_with_diag(self, k_vec, kp, diag_idx, diag_value):
        k = np.asarray(k_vec, dtype=float)
        diag_idx = int(diag_idx)
        if k.ndim != 1 or not (0 <= diag_idx < k.size):
            raise ValueError("diag_idx is inconsistent with k_vec.")
        a = np.abs(k - kp)
        b = k + kp
        out = self._primitive_dyn(b) - self._primitive_dyn(a)
        if self._has_singular_dyn:
            offdiag = a > 0.0
            zero = ~offdiag
            zero[diag_idx] = False
            if np.any(zero):
                raise FloatingPointError(
                    "multiple zero-length singular radial blocks need cell averages."
                )
            with np.errstate(divide="raise", invalid="raise"):
                log_interval = np.log(b[offdiag]) - np.log(a[offdiag])
            out[offdiag] = (
                out[offdiag]
                + log_interval[:, None] * self.singular_dyn[None, :]
            )
        out[diag_idx] = diag_value
        return out

    def integral_block_sta_with_diag(self, k_vec, kp, diag_idx, diag_value):
        k = np.asarray(k_vec, dtype=float)
        diag_idx = int(diag_idx)
        if k.ndim != 1 or not (0 <= diag_idx < k.size):
            raise ValueError("diag_idx is inconsistent with k_vec.")
        a = np.abs(k - kp)
        b = k + kp
        out = self._primitive_sta(b) - self._primitive_sta(a)
        if self._has_singular_sta:
            offdiag = a > 0.0
            zero = ~offdiag
            zero[diag_idx] = False
            if np.any(zero):
                raise FloatingPointError(
                    "multiple zero-length singular radial blocks need cell averages."
                )
            with np.errstate(divide="raise", invalid="raise"):
                log_interval = np.log(b[offdiag]) - np.log(a[offdiag])
            out[offdiag] = out[offdiag] + log_interval * self.singular_sta
        out[diag_idx] = diag_value
        return out

    def _integral_block_with_diag(self, k_vec, kp, diag_idx, diag_value, dynamic):
        if dynamic:
            return self.integral_block_dyn_with_diag(
                k_vec, kp, diag_idx, diag_value)
        return self.integral_block_sta_with_diag(
            k_vec, kp, diag_idx, diag_value)

    def integral_blocks_with_diag(
        self, k_vec, kp, diag_idx, diag_dyn_value, diag_sta_value,
    ):
        """Return dynamic and static blocks sharing qW geometry and log terms."""
        k = np.asarray(k_vec, dtype=float)
        diag_idx = int(diag_idx)
        if k.ndim != 1 or not (0 <= diag_idx < k.size):
            raise ValueError("diag_idx is inconsistent with k_vec.")
        a = np.abs(k - kp)
        b = k + kp
        dyn_hi, sta_hi = self._primitive_h_dyn_sta(b)
        dyn_lo, sta_lo = self._primitive_h_dyn_sta(a)
        dyn = dyn_hi - dyn_lo
        sta = sta_hi - sta_lo
        if self.has_singularity:
            offdiag = a > 0.0
            zero = ~offdiag
            zero[diag_idx] = False
            if np.any(zero):
                raise FloatingPointError(
                    "multiple zero-length singular radial blocks need cell averages."
                )
            with np.errstate(divide="raise", invalid="raise"):
                log_interval = np.log(b[offdiag]) - np.log(a[offdiag])
            if self._has_singular_dyn:
                dyn[offdiag] = (
                    dyn[offdiag]
                    + log_interval[:, None] * self.singular_dyn[None, :]
                )
            if self._has_singular_sta:
                sta[offdiag] = sta[offdiag] + log_interval * self.singular_sta
        dyn[diag_idx] = diag_dyn_value
        sta[diag_idx] = diag_sta_value
        return dyn, sta

    def static_blocks_with_diag(
        self, k_vec, diag_values, row_start=0, row_stop=None,
    ):
        """Return cached static radial blocks for rank-local target rows."""
        k = np.asarray(k_vec, dtype=float)
        diag_values = np.asarray(diag_values, dtype=self.sta_dtype)
        if k.ndim != 1 or diag_values.shape != k.shape:
            raise ValueError("static block cache inputs are inconsistent.")
        row_start = int(row_start)
        row_stop = k.size if row_stop is None else int(row_stop)
        if row_start < 0 or row_stop < row_start or row_stop > k.size:
            raise ValueError("invalid static block row range.")
        key = (
            row_start,
            row_stop,
            _array_value_key(k),
            _array_value_key(diag_values),
        )
        cached = self._static_block_cache.get(key)
        if cached is not None:
            return cached
        blocks = np.empty((row_stop - row_start, k.size), dtype=self.sta_dtype)
        for local_j, j in enumerate(range(row_start, row_stop)):
            blocks[local_j] = self.integral_block_sta_with_diag(
                k, k[j], j, diag_values[j])
        self._static_block_cache.clear()
        self._static_block_cache[key] = blocks
        return blocks

    def clear_static_block_cache(self):
        """Clear cached rank-local static radial blocks for this qW object."""
        ncache = len(self._static_block_cache)
        self._static_block_cache.clear()
        return ncache

    def _cell_average_points(self, kp, lo, hi):
        kp = np.asarray(kp, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        kp, lo, hi = np.broadcast_arrays(kp, lo, hi)
        width = hi - lo
        if np.any(width <= 0.0):
            raise ValueError("diagonal cell must have positive width.")
        points = np.stack(
            (
                hi + kp,
                lo + kp,
                np.maximum(kp - lo, 0.0),
                np.maximum(kp - hi, 0.0),
                np.maximum(hi - kp, 0.0),
                np.maximum(lo - kp, 0.0),
            ),
            axis=-1,
        )
        coeff = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
        return points, coeff, width

    def _regular_cell_average_dyn_analytic(self, kp, lo, hi):
        """Average the regular dynamic block exactly for linear ``q*R`` panels."""
        points, coeff, width = self._cell_average_points(kp, lo, hi)
        shape = width.shape
        flat_points = points.reshape((-1, 6))
        out = None
        # Keep the primitive temporaries at (nk, nt), not (6*nk, nt).
        for idx, fac in enumerate(coeff):
            vals = self._second_primitive_h_dyn(flat_points[:, idx])
            out = fac * vals if out is None else out + fac * vals
        out = out / width.reshape(-1)[:, None]
        if shape:
            return out.reshape(shape + (self.nt,))
        return out[0]

    def _regular_cell_average_sta_analytic(self, kp, lo, hi):
        """Average the regular static block exactly for linear ``q*R`` panels."""
        points, coeff, width = self._cell_average_points(kp, lo, hi)
        shape = width.shape
        flat_points = points.reshape((-1, 6))
        out = None
        for idx, fac in enumerate(coeff):
            vals = self._second_primitive_h_sta(flat_points[:, idx])
            out = fac * vals if out is None else out + fac * vals
        out = out / width.reshape(-1)
        if shape:
            return out.reshape(shape)
        return out[0]

    def _regular_cell_averages_analytic(self, kp, lo, hi):
        """Average regular dynamic/static blocks exactly in one primitive pass."""
        points, coeff, width = self._cell_average_points(kp, lo, hi)
        shape = width.shape
        flat_points = points.reshape((-1, 6))
        dyn = None
        sta = None
        for idx, fac in enumerate(coeff):
            dyn_vals, sta_vals = self._second_primitive_h_dyn_sta(
                flat_points[:, idx])
            if dyn is None:
                dyn = fac * dyn_vals
                sta = fac * sta_vals
            else:
                dyn = dyn + fac * dyn_vals
                sta = sta + fac * sta_vals
        dyn = dyn / width.reshape(-1)[:, None]
        sta = sta / width.reshape(-1)
        if shape:
            return dyn.reshape(shape + (self.nt,)), sta.reshape(shape)
        return dyn[0], sta[0]

    def cell_average_dyn_analytic(self, kp, lo, hi):
        """Return the analytic diagonal-cell average of the dynamic block."""
        dyn = self._regular_cell_average_dyn_analytic(kp, lo, hi)
        if self._has_singular_dyn:
            factor = _singular_cell_average_factor(kp, lo, hi)
            if np.asarray(factor).ndim:
                dyn = dyn + factor[..., None] * self.singular_dyn
            else:
                dyn = dyn + factor * self.singular_dyn
        return dyn

    def cell_average_sta_analytic(self, kp, lo, hi):
        """Return the analytic diagonal-cell average of the static block."""
        sta = self._regular_cell_average_sta_analytic(kp, lo, hi)
        if self._has_singular_sta:
            sta = sta + _singular_cell_average_factor(kp, lo, hi) * self.singular_sta
        return sta

    def cell_averages_analytic(self, kp, lo, hi):
        """Return analytic dynamic and static diagonal-cell averages."""
        dyn, sta = self._regular_cell_averages_analytic(kp, lo, hi)
        if self.has_singularity:
            factor = _singular_cell_average_factor(kp, lo, hi)
            if self._has_singular_dyn:
                if np.asarray(factor).ndim:
                    dyn = dyn + factor[..., None] * self.singular_dyn
                else:
                    dyn = dyn + factor * self.singular_dyn
            if self._has_singular_sta:
                sta = sta + factor * self.singular_sta
        return dyn, sta

    def cell_average_dyn(self, kp, lo, hi, ncell, xw=None):
        """Return the diagonal-cell average of the dynamic radial block."""
        out = _integrate_cell_average(
            self.integral_block_dyn_regular, kp, lo, hi, ncell, xw=xw)
        if self._has_singular_dyn:
            out = out + _singular_cell_average(self.singular_dyn, kp, lo, hi)
        return out

    def cell_average_sta(self, kp, lo, hi, ncell, xw=None):
        """Return the diagonal-cell average of the static radial block."""
        out = _integrate_cell_average(
            self.integral_block_sta_regular, kp, lo, hi, ncell, xw=xw)
        if self._has_singular_sta:
            out = out + _singular_cell_average(self.singular_sta, kp, lo, hi)
        return out

    def cell_averages(self, kp, lo, hi, ncell, xw=None):
        """Return dynamic and static diagonal-cell averages in one quadrature."""
        total_dyn = None
        total_sta = None
        x, w = leggauss(ncell) if xw is None else xw
        for a, b in ((lo, min(kp, hi)), (max(kp, lo), hi)):
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            vals_dyn, vals_sta = self.integral_blocks_regular(
                mid + half * x, kp)
            weight = half * w
            contrib_dyn = np.tensordot(weight, vals_dyn, axes=(0, 0))
            contrib_sta = np.tensordot(weight, vals_sta, axes=(0, 0))
            total_dyn = contrib_dyn if total_dyn is None else total_dyn + contrib_dyn
            total_sta = contrib_sta if total_sta is None else total_sta + contrib_sta
        if total_dyn is None:
            raise ValueError("empty diagonal cell for k=%.16e" % kp)
        width = float(hi) - float(lo)
        dyn = total_dyn / width
        sta = total_sta / width
        if self.has_singularity:
            factor = _singular_cell_average_factor(kp, lo, hi)
            if self._has_singular_dyn:
                dyn = dyn + factor * self.singular_dyn
            if self._has_singular_sta:
                sta = sta + factor * self.singular_sta
        return dyn, sta

    def eval_dyn(self, q_star):
        """Evaluate the full dynamic kernel ``D(q,tau)`` implied by qW primitives."""
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(
            np.clip(q, self.qh[0], np.nextafter(self.qh[-1], -np.inf)))
        out = self._eval_reg_dyn(q)
        if max_abs(self.singular_dyn) > 0.0:
            sing = np.zeros((q.size, self.nt), dtype=self.yD.dtype)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(
                    self.singular_dyn[None, :],
                    q[:, None] ** 2,
                    out=sing,
                    where=(self.singular_dyn[None, :] != 0.0),
                )
            out = out + sing
        return out[0] if scalar else out

    def eval_sta(self, q_star):
        """Evaluate the full static kernel ``Vc(q)`` implied by qW primitives."""
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(
            np.clip(q, self.qh[0], np.nextafter(self.qh[-1], -np.inf)))
        out = self._eval_reg_sta(q)
        if abs(self.singular_sta) > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out + self.singular_sta / (q * q)
        return out[0] if scalar else out

    def _eval_reg_dyn(self, q_star):
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self.qh[0], np.nextafter(self.qh[-1], -np.inf)))
        m = np.searchsorted(self.qh, q, side="right") - 1
        m = np.clip(m, 0, len(self.qh) - 2)
        dx = q - self.qh[m]
        h = self.hD[m] + dx[:, None] * self.shD[m]
        out = np.empty_like(h)
        nz = q != 0.0
        out[nz] = h[nz] / q[nz, None]
        if np.any(~nz):
            out[~nz] = self.shD[0]
        return out[0] if scalar else out

    def _eval_reg_sta(self, q_star):
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self.qh[0], np.nextafter(self.qh[-1], -np.inf)))
        m = np.searchsorted(self.qh, q, side="right") - 1
        m = np.clip(m, 0, len(self.qh) - 2)
        dx = q - self.qh[m]
        h = self.hS[m] + dx * self.shS[m]
        out = np.empty_like(h)
        nz = q != 0.0
        out[nz] = h[nz] / q[nz]
        if np.any(~nz):
            out[~nz] = self.shS[0]
        return out[0] if scalar else out

    def eval_dyn_weighted(self, q_star, weights, q2_weight_limit=None):
        return self._eval_weighted(q_star, weights, self._eval_reg_dyn,
                                   self.singular_dyn, q2_weight_limit)

    def eval_sta_weighted(self, q_star, weights, q2_weight_limit=None):
        return self._eval_weighted(q_star, weights, self._eval_reg_sta,
                                   self.singular_sta, q2_weight_limit)

    def _eval_weighted(self, q_star, weights, eval_reg, coeff, q2_weight_limit):
        """Evaluate ``weights * (R(q) + A/q**2)`` with a supplied q=0 limit."""
        q_in = np.asarray(q_star, dtype=float)
        scalar = q_in.ndim == 0
        q = np.atleast_1d(q_in)
        weights_in = np.asarray(weights)
        if scalar and weights_in.shape == (1,):
            weights_in = weights_in.reshape(())
        weights = np.broadcast_to(weights_in, q.shape)
        reg = eval_reg(q)
        out = weights[(slice(None),) + (None,) * (np.ndim(reg) - 1)] * reg
        if max_abs(coeff) == 0.0:
            return out[0] if scalar else out
        q_clip = np.clip(q, self.qh[0], np.nextafter(self.qh[-1], -np.inf))
        ratio = np.empty_like(q_clip, dtype=float)
        nz = q_clip != 0.0
        ratio[nz] = weights[nz] / (q_clip[nz] * q_clip[nz])
        if np.any(~nz):
            if q2_weight_limit is None:
                raise FloatingPointError(
                    "weighted singular eval at q=0 requires q2_weight_limit."
                )
            limit_in = np.asarray(q2_weight_limit, dtype=float)
            if scalar and limit_in.shape == (1,):
                limit_in = limit_in.reshape(())
            limit = np.broadcast_to(limit_in, q.shape)
            ratio[~nz] = limit[~nz]
        if np.ndim(coeff):
            out = out + ratio[:, None] * coeff[None, :]
        else:
            out = out + ratio * coeff
        return out[0] if scalar else out

    def frequency_table(self, ctx):
        """Reconstruct the full frequency-domain ``W(q, i nu_n)`` table."""
        inv = getattr(ctx, "_migdal_F_t_b2f_inv", None)
        if inv is None:
            inv = np.linalg.pinv(np.asarray(ctx.F_t_b2f))
            setattr(ctx, "_migdal_F_t_b2f_inv", inv)
        dW_reg = ctx.trans(np.asarray(self.yD_reg), inv, ctx.F_w_b)
        if self.singular_dyn_w is None:
            singular_dyn_w = ctx.trans(
                self.singular_dyn[None, :], inv, ctx.F_w_b)[0]
        else:
            singular_dyn_w = np.asarray(self.singular_dyn_w, dtype=dW_reg.dtype)
        if singular_dyn_w.shape != (dW_reg.shape[1],):
            raise ValueError("singular_dyn_w has inconsistent frequency shape.")
        q2 = self.q2[:, None]
        return dW_reg + singular_dyn_w[None, :] / q2 + (
            self.yS_reg + self.singular_sta / self.q2
        )[:, None]
