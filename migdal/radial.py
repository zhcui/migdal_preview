"""MPI radial convolution kernels for the one-band Migdal drivers.

The sparse-IR G0W0/GW0/scGW drivers call these kernels after reducing isotropic
3D momentum integrals to radial k and q grids.  Unless noted otherwise, momenta
are in Bohr^-1, energies are in Hartree, and angular averages use the
``d^3 k / (2*pi)^3`` Fourier normalization.

For a radial target momentum ``k`` and source momentum ``p``, the angular part
of an isotropic convolution is represented by an integral over
``q = |k - p| ... k + p``.  ``WPrecomp`` stores piecewise-linear W primitives,
``migdal.radial_qW.QWPrecomp`` stores Coulomb-split qW primitives, and
``GPrecomp`` stores polarization-bubble primitives so the hot kernels can
evaluate interval integrals without repeated quadrature.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

from migdal import log as mlog, memory, model, mpi_wrap
from migdal.radial_qW import (
    SingularFitDiagnostics,
    build_q_singular_precomp,
    clear_qw_static_block_caches,
    estimate_screened_singular_coeff_from_pi,
)


_SIGMA_KERNEL_CACHES = {}
_SIGMA_PAIR_GEOMETRY_CACHES = OrderedDict()
_QW_SIGMA_PAIR_GEOMETRY_CACHES = OrderedDict()
_QW_DIAG_CELL_GEOMETRY_CACHES = OrderedDict()
_PI_PAIR_GEOMETRY_CACHES = OrderedDict()
_SIGMA_PAIR_GEOMETRY_CACHE_MAX = int(os.environ.get(
    "MIGDAL_SIGMA_PAIR_GEOMETRY_CACHE_MAX", "4"
))
_SIGMA_PAIR_GEOMETRY_CACHE_MB = float(os.environ.get(
    "MIGDAL_SIGMA_PAIR_GEOMETRY_CACHE_MB", "512.0"
))
# Setting MIGDAL_QW_SIGMA_PAIR_GEOMETRY_CACHE_MAX=0 skips the materialized qW
# pair geometry cache; the Cython qW sigma path can still stream pair
# coefficients row-by-row.
_QW_SIGMA_PAIR_GEOMETRY_CACHE_MAX = int(os.environ.get(
    "MIGDAL_QW_SIGMA_PAIR_GEOMETRY_CACHE_MAX",
    str(_SIGMA_PAIR_GEOMETRY_CACHE_MAX),
))
_QW_SIGMA_PAIR_GEOMETRY_CACHE_MB = float(os.environ.get(
    "MIGDAL_QW_SIGMA_PAIR_GEOMETRY_CACHE_MB",
    str(_SIGMA_PAIR_GEOMETRY_CACHE_MB),
))
_QW_DIAG_CELL_GEOMETRY_CACHE_MAX = int(os.environ.get(
    "MIGDAL_QW_DIAG_CELL_GEOMETRY_CACHE_MAX", "8"
))
_QW_DIAG_CELL_GEOMETRY_CACHE_MB = float(os.environ.get(
    "MIGDAL_QW_DIAG_CELL_GEOMETRY_CACHE_MB", "64.0"
))
_PI_PAIR_GEOMETRY_CACHE_MAX = int(os.environ.get(
    "MIGDAL_PI_PAIR_GEOMETRY_CACHE_MAX", "4"
))
_PI_PAIR_GEOMETRY_CACHE_MB = float(os.environ.get(
    "MIGDAL_PI_PAIR_GEOMETRY_CACHE_MB", "512.0"
))
_SIGMA_TRIANGULAR_DISABLED = os.environ.get(
    "MIGDAL_DISABLE_SIGMA_TRIANGULAR", ""
).lower() in ("1", "true", "yes", "on")
_SIGMA_TRIANGULAR_TILE = max(1, int(os.environ.get(
    "MIGDAL_SIGMA_TRIANGULAR_TILE", "32"
)))
_RADIAL_OPENMP_DISABLED = os.environ.get(
    "MIGDAL_DISABLE_RADIAL_OPENMP", ""
).lower() in ("1", "true", "yes", "on")
_RADIAL_OPENMP_MIN_WORK = max(0, int(os.environ.get(
    "MIGDAL_RADIAL_OPENMP_MIN_WORK", "200000"
)))
_SIGMA_OMP_PRIVATE_MB = float(os.environ.get(
    "MIGDAL_SIGMA_OMP_PRIVATE_MB", "256.0"
))
_QW_STATIC_BLOCK_CACHE_MB = float(os.environ.get(
    "MIGDAL_QW_STATIC_BLOCK_CACHE_MB", "128.0"
))
_CYTHON_RADIAL_DISABLED = os.environ.get(
    "MIGDAL_DISABLE_CYTHON_RADIAL", ""
).lower() in ("1", "true", "yes", "on")

_get_mpi = mpi_wrap.get_mpi
_mpi_world_size_hint = mpi_wrap.world_size_hint
_lazy_parallel_call = mpi_wrap.parallel_call
mpi = mpi_wrap.mpi

if _CYTHON_RADIAL_DISABLED:
    _radial_kernels = None
    _qw_radial_kernels = None
else:
    try:
        from migdal import _radial_kernels
    except ImportError:
        _radial_kernels = None
    else:
        if getattr(_radial_kernels, "RADIAL_KERNEL_API_VERSION", 0) < 3:
            _radial_kernels = None
    try:
        from migdal import _qw_radial_kernels
    except ImportError:
        _qw_radial_kernels = None
    else:
        if getattr(_qw_radial_kernels, "QW_RADIAL_KERNEL_API_VERSION", 0) < 3:
            _qw_radial_kernels = None


@dataclass
class DenomMetrics:
    """Diagnostics for RPA denominator regularization."""

    policy: str
    count: int = 0
    undetermined_count: int = 0
    min_abs: float = np.inf
    max_abs_W: float = 0.0
    worst_q: float = np.nan
    worst_v: float = np.nan
    worst_pi: complex = 0.0
    worst_vtot: complex = 0.0
    worst_old: complex = 0.0
    worst_new: complex = 0.0
    singular_fit: SingularFitDiagnostics | None = None


@dataclass
class DiagCellGeometry:
    """Reusable k-grid geometry for diagonal cell averages."""

    k_grid: np.ndarray
    k_wts: np.ndarray
    k_upper: float
    bounds: np.ndarray
    xw: tuple[np.ndarray, np.ndarray]
    ncell: int


def max_abs(a):
    """Return ``max(abs(a))`` as a Python float, with zero for empty arrays."""
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a)))


def _task_location(n, task=None):
    if task is None:
        task = mpi.rank
    neach, extras = divmod(int(n), mpi.pool.size)
    section_sizes = [0] + extras * [neach + 1] + (mpi.pool.size - extras) * [neach]
    div_points = np.cumsum(section_sizes)
    return int(div_points[task]), int(div_points[task + 1])


def _gather_rank0(arr):
    """Gather rank-local row blocks; only rank 0 receives the full array."""
    return mpi_wrap.gather(arr)


def symm_sigma(ctx, sigma):
    """Project ``Sigma(k, iw)`` onto reverse-frequency conjugate symmetry."""
    sigma = np.asarray(sigma)
    out = np.zeros_like(sigma)
    left = sigma[:, : ctx.nw // 2]
    right = sigma[:, ctx.nw // 2:][:, ::-1].conj()
    avg = 0.5 * (left + right)
    out[:, : ctx.nw // 2] = avg
    out[:, ctx.nw // 2:] = avg[:, ::-1].conj()
    return out


def check_w_pair(ctx, atol=1e-10):
    """Validate reverse-index Matsubara pairing."""
    if ctx.nw % 2:
        raise ValueError("fermion frequency grid must contain paired points.")
    err = max_abs(np.asarray(ctx.ws) + np.asarray(ctx.ws)[::-1])
    if err > atol:
        raise ValueError("ctx.ws is not paired by reverse indexing.")
    return err


def check_sigma_symm(ctx, sigma):
    """Return the reverse-frequency conjugate-symmetry error of ``Sigma``.

    A normal Matsubara self-energy should satisfy
    ``Sigma(k, iw) = conj(Sigma(k, -iw))`` on the paired sparse-IR grid.
    The function validates the frequency pairing and returns the max absolute
    mismatch; callers decide whether the mismatch is acceptable.
    """
    check_w_pair(ctx)
    sigma_1 = sigma[:, : ctx.nw // 2]
    sigma_2 = sigma[:, ctx.nw // 2:][:, ::-1].conj()
    return max_abs(sigma_1 - sigma_2)


def check_Pi_symm(ctx, Pi):
    """Return the bosonic even-frequency symmetry error of ``Pi``."""
    idx0 = int(ctx.idx_v0[0])
    Pi_1 = Pi[:, :idx0]
    Pi_2 = Pi[:, idx0 + 1:][:, ::-1].conj()
    return max_abs(Pi_1 - Pi_2)


def symm_Pi(ctx, Pi):
    """Project bosonic response onto real even Matsubara symmetry."""
    idx0 = int(ctx.idx_v0[0])
    out = np.array(Pi, copy=True)
    left = out[:, :idx0]
    right = out[:, idx0 + 1:][:, ::-1].conj()
    avg = 0.5 * (left + right)
    out[:, :idx0] = avg
    out[:, idx0 + 1:] = avg[:, ::-1].conj()

    left = out[:, :idx0]
    right = out[:, idx0 + 1:][:, ::-1]
    avg = 0.5 * (left + right)
    out[:, :idx0] = avg
    out[:, idx0 + 1:] = avg[:, ::-1]
    return out


def _validate_w_interp_mode(mode):
    mode = str(mode)
    if mode not in ("legacy", "q_singular"):
        raise ValueError("unknown W interpolation mode %r" % mode)
    return mode


def _validate_q_singular_head_policy(policy):
    policy = str(policy)
    if policy not in ("pi_head", "fit_w", "provided"):
        raise ValueError("unknown q_singular head policy %r" % policy)
    return policy


def _validate_diag_average_mode(mode):
    mode = str(mode)
    if mode not in ("auto", "gauss", "analytic"):
        raise ValueError("unknown diagonal-cell average mode %r" % mode)
    return mode


def _resolve_diag_average_mode(precomp, mode):
    mode = _validate_diag_average_mode(mode)
    if mode == "auto":
        if getattr(precomp, "interp_mode", None) == "q_singular":
            return "analytic"
        return "gauss"
    return mode


class WPrecomp:
    r"""Piecewise-linear W primitives for radial angle averages.

    This representation linearly interpolates the sampled kernels
    ``D(q,tau)`` and ``Vc(q)``, then analytically integrates
    ``q * D_linear(q)`` and ``q * Vc_linear(q)`` over each angular interval.
    Coulomb-split qW primitives live in :class:`migdal.radial_qW.QWPrecomp`.
    """

    def __init__(self, q_grid, dW_tau, Vc, *, interp_mode="legacy"):
        self.interp_mode = _validate_w_interp_mode(interp_mode)
        if self.interp_mode != "legacy":
            raise ValueError(
                "WPrecomp supports only interp_mode='legacy'; use "
                "radial_qW.QWPrecomp for q_singular W."
            )
        self.q = np.asarray(q_grid, dtype=float)
        self.yD = np.asarray(dW_tau)
        if self.q.ndim != 1 or self.yD.ndim != 2 or self.yD.shape[0] != self.q.size:
            raise ValueError("dW_tau must be (nq, nt) with nq=len(q_grid).")
        self.nq, self.nt = self.yD.shape
        self.yS = np.asarray(Vc, dtype=self.yD.dtype)
        if self.yS.shape != (self.nq,):
            raise ValueError("Vc must have shape (nq,).")

        self.qL = self.q[:-1]
        self.qR = self.q[1:]
        self.dq = self.qR - self.qL
        if not np.all(self.dq > 0.0):
            raise ValueError("q_grid must be strictly increasing.")
        self.q2 = self.q * self.q
        self.q3 = self.q2 * self.q

        qR2 = self.qR * self.qR
        qR3 = qR2 * self.qR
        qL2 = self.qL * self.qL
        qL3 = qL2 * self.qL
        dQ2 = 0.5 * (qR2 - qL2)
        dQ3 = (qR3 - qL3) / 3.0 - self.qL * dQ2

        self.sD = (self.yD[1:] - self.yD[:-1]) / self.dq[:, None]
        JD = self.yD[:-1] * dQ2[:, None] + self.sD * dQ3[:, None]
        self.PD = np.zeros((self.nq, self.nt), dtype=self.yD.dtype)
        self.PD[1:] = np.cumsum(JD, axis=0)

        self.sS = (self.yS[1:] - self.yS[:-1]) / self.dq
        JS = dQ2 * self.yS[:-1] + dQ3 * self.sS
        self.PS = np.zeros(self.nq, dtype=self.yD.dtype)
        self.PS[1:] = np.cumsum(JS)
        self.dyn_dtype = self.yD.dtype
        self.sta_dtype = self.PS.dtype
        self._cython_cache = None

    @property
    def uses_legacy_primitives(self):
        return True

    @property
    def has_singularity(self):
        return False

    @property
    def requires_diag_cells(self):
        return False

    def _primitive_dyn(self, q_star):
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self.q[0], np.nextafter(self.q[-1], -np.inf)))
        m = np.searchsorted(self.q, q, side="right") - 1
        m = np.clip(m, 0, self.nq - 2)
        q2 = q * q
        q3 = q2 * q
        qL = self.q[m]
        tmp = 0.5 * (q2 - self.q2[m])
        out = tmp[:, None] * self.yD[m]
        out += ((q3 - self.q3[m]) / 3.0 - qL * tmp)[:, None] * self.sD[m]
        out += self.PD[m]
        return out[0] if scalar else out

    def _primitive_sta(self, q_star):
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self.q[0], np.nextafter(self.q[-1], -np.inf)))
        m = np.searchsorted(self.q, q, side="right") - 1
        m = np.clip(m, 0, self.nq - 2)
        q2 = q * q
        q3 = q2 * q
        qL = self.q[m]
        tmp = 0.5 * (q2 - self.q2[m])
        out = tmp * self.yS[m]
        out += ((q3 - self.q3[m]) / 3.0 - qL * tmp) * self.sS[m]
        out += self.PS[m]
        return out[0] if scalar else out

    def integral_block_dyn(self, k_vec, kp):
        """Return dynamic interval integrals for all ``k_vec`` against ``kp``."""
        k = np.asarray(k_vec, dtype=float)
        return self._primitive_dyn(k + kp) - self._primitive_dyn(np.abs(k - kp))

    def integral_block_sta(self, k_vec, kp):
        """Return static Coulomb interval integrals for all ``k_vec`` vs ``kp``."""
        k = np.asarray(k_vec, dtype=float)
        return self._primitive_sta(k + kp) - self._primitive_sta(np.abs(k - kp))

    def integral_block_dyn_with_diag(self, k_vec, kp, diag_idx, diag_value):
        return self._integral_block_with_diag(
            k_vec, kp, diag_idx, diag_value, dynamic=True)

    def integral_block_sta_with_diag(self, k_vec, kp, diag_idx, diag_value):
        return self._integral_block_with_diag(
            k_vec, kp, diag_idx, diag_value, dynamic=False)

    def _integral_block_with_diag(self, k_vec, kp, diag_idx, diag_value, dynamic):
        k = np.asarray(k_vec, dtype=float)
        diag_idx = int(diag_idx)
        if k.ndim != 1 or not (0 <= diag_idx < k.size):
            raise ValueError("diag_idx is inconsistent with k_vec.")
        out = self.integral_block_dyn(k, kp) if dynamic else self.integral_block_sta(k, kp)
        out[diag_idx] = diag_value
        return out

    def cell_average_dyn(self, kp, lo, hi, ncell, xw=None):
        return _integrate_cell_average(
            self.integral_block_dyn, kp, lo, hi, ncell, xw=xw)

    def cell_average_sta(self, kp, lo, hi, ncell, xw=None):
        return _integrate_cell_average(
            self.integral_block_sta, kp, lo, hi, ncell, xw=xw)

    def eval_dyn(self, q_star):
        """Linearly interpolate the dynamic tau table at ``q_star``."""
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self.q[0], np.nextafter(self.q[-1], -np.inf)))
        m = np.searchsorted(self.q, q, side="right") - 1
        m = np.clip(m, 0, self.nq - 2)
        t = (q - self.q[m]) / self.dq[m]
        out = (1.0 - t)[:, None] * self.yD[m] + t[:, None] * self.yD[m + 1]
        return out[0] if scalar else out

    def eval_sta(self, q_star):
        """Linearly interpolate the static Coulomb table at ``q_star``."""
        q = np.asarray(q_star, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(np.clip(q, self.q[0], np.nextafter(self.q[-1], -np.inf)))
        m = np.searchsorted(self.q, q, side="right") - 1
        m = np.clip(m, 0, self.nq - 2)
        t = (q - self.q[m]) / self.dq[m]
        out = (1.0 - t) * self.yS[m] + t * self.yS[m + 1]
        return out[0] if scalar else out

    def frequency_table(self, ctx):
        """Reconstruct ``W(q, i nu_n)`` from this precomputed table."""
        inv = getattr(ctx, "_migdal_F_t_b2f_inv", None)
        if inv is None:
            inv = np.linalg.pinv(np.asarray(ctx.F_t_b2f))
            setattr(ctx, "_migdal_F_t_b2f_inv", inv)
        dW = ctx.trans(np.asarray(self.yD), inv, ctx.F_w_b)
        return dW + np.asarray(self.yS)[:, None]


class GPrecomp:
    r"""Piecewise-linear k primitive for the radial polarization bubble.

    The primitive integrates ``k * G(beta-tau, k)`` over k intervals.  It is
    used in the same angular reduction as the W precomputation objects, but
    for the particle-hole bubble ``Pi(q, tau)``.
    """

    def __init__(self, k_grid, G_beta_tau):
        self.k = np.asarray(k_grid, dtype=float)
        self.yD = np.asarray(G_beta_tau)
        if self.k.ndim != 1 or self.yD.ndim != 2 or self.yD.shape[0] != self.k.size:
            raise ValueError("G_beta_tau must be (nk, nt).")
        self.nk, self.nt = self.yD.shape
        self.kL = self.k[:-1]
        self.kR = self.k[1:]
        self.dk = self.kR - self.kL
        if not np.all(self.dk > 0.0):
            raise ValueError("k_grid must be strictly increasing.")
        self.k2 = self.k * self.k
        self.k3 = self.k2 * self.k
        kR2 = self.kR * self.kR
        kR3 = kR2 * self.kR
        kL2 = self.kL * self.kL
        kL3 = kL2 * self.kL
        dK2 = 0.5 * (kR2 - kL2)
        dK3 = (kR3 - kL3) / 3.0 - self.kL * dK2
        self.sD = (self.yD[1:] - self.yD[:-1]) / self.dk[:, None]
        JD = self.yD[:-1] * dK2[:, None] + self.sD * dK3[:, None]
        self.PD = np.zeros((self.nk, self.nt), dtype=self.yD.dtype)
        self.PD[1:] = np.cumsum(JD, axis=0)
        self._cython_cache = None

    def _primitive_dyn(self, k_star):
        k = np.asarray(k_star, dtype=float)
        k = np.clip(k, self.k[0], np.nextafter(self.k[-1], -np.inf))
        m = np.searchsorted(self.k, k, side="right") - 1
        m = np.clip(m, 0, self.nk - 2)
        k2 = k * k
        k3 = k2 * k
        kL = self.k[m]
        tmp = 0.5 * (k2 - self.k2[m])
        out = tmp[:, None] * self.yD[m]
        out += ((k3 - self.k3[m]) / 3.0 - kL * tmp)[:, None] * self.sD[m]
        out += self.PD[m]
        return out

    def integral_block_dyn(self, k_vec, q):
        """Return particle-hole interval integrals for all ``k_vec`` at ``q``."""
        k = np.asarray(k_vec, dtype=float)
        return self._primitive_dyn(k + q) - self._primitive_dyn(np.abs(k - q))


def get_W_from_Pi(ctx, Pi, q_grid, Vc, Vtot, denom_policy="fail",
                  denom_min=1e-10, return_metrics=False,
                  q_interp_mode="legacy", static_singular_coeff=None,
                  q_singular_head_policy="pi_head",
                  screened_singular_coeff=None, vtot_head=None,
                  singular_fit_n=6, singular_fit_degree=1,
                  singular_fit_warn_rtol=1.0e-3,
                  singular_fit_warn_atol=1.0e-10,
                  warning_log=None):
    r"""Build a radial W precomputation object from Pi and a bare interaction table.

    Parameters
    ----------
    ctx : :class:`migdal.freq.FreqGrid`
        Sparse-IR context defining the bosonic Matsubara grid ``ctx.vs``.
    Pi : (nq, nv) array_like
        Polarization on the imaginary axis.  Migdal uses the convention where
        the electron gas has ``Pi <= 0``.
    q_grid : (nq,) array_like
        Positive radial momenta in Bohr^-1.
    Vc : (nq,) array_like
        Static Coulomb table.  The q=0 placeholder, if present, is zero.
    Vtot : (nq, nv) array_like
        Bare Coulomb times phonon dielectric factor.
    q_interp_mode : {"legacy", "q_singular"}, optional
        ``legacy`` uses piecewise-linear W primitives.  ``q_singular`` builds
        :class:`migdal.radial_qW.QWPrecomp` after separating the Coulomb
        ``A_W(i nu)/q**2`` head from a regular residual.  By default
        ``A_W`` is derived from the low-q expansion of ``Pi`` and ``Vtot``.
    warning_log : PySCF logger, optional
        Logger used for low-q fit diagnostics.  If omitted, warnings are
        emitted to a default WARN-level stream logger.
    static_singular_coeff : scalar, optional
        Coefficient ``A_c`` for ``Vc(q) = A_c / q**2 + V_reg(q)``.  If omitted
        in ``q_singular`` mode, it is fitted from ``Vc``.
    q_singular_head_policy : {"pi_head", "fit_w", "provided"}, optional
        Source for ``A_W(i nu) = lim(q -> 0) q**2 W(q, i nu)`` in
        ``q_singular`` mode.  The default ``pi_head`` derives the screened head
        from the low-q expansion of ``Pi`` and ``Vtot``.  ``fit_w`` fits
        ``q**2 W`` directly from the sampled rows.  ``provided`` uses
        ``screened_singular_coeff``.
    screened_singular_coeff : (nv,) array_like, optional
        User-provided screened head coefficient for ``provided`` policy.
    vtot_head : (nv,) array_like, optional
        Known bare-interaction head ``B(i nu) = lim(q -> 0) q**2 Vtot(q, i nu)``.
        If omitted in ``pi_head`` mode, the generic fallback fits it from
        ``Vtot``.

    Notes
    -----
    The screened interaction is

    .. math::

        W(q, i\nu) = \frac{V_{\mathrm{tot}}(q, i\nu)}
                          {1 - \Pi(q, i\nu) V_{\mathrm{tot}}(q, i\nu)}.

    ``denom_policy`` controls only small-denominator handling; it does not
    alter the sign convention for ``Pi`` or the Fourier normalization.

    In ``q_singular`` mode, the dynamic table is split as
    ``W - Vc = (A_W - A_c) / q**2 + D_reg``.  The singular coefficient and
    regular residual are transformed to tau separately before constructing the
    qW primitives.
    """
    q_interp_mode = _validate_w_interp_mode(q_interp_mode)
    q_singular_head_policy = _validate_q_singular_head_policy(
        q_singular_head_policy)
    if denom_policy not in ("none", "fail", "legacy_clip", "sign_regularize"):
        raise ValueError("unknown denom_policy %r" % denom_policy)
    if denom_policy != "none" and denom_min <= 0.0:
        raise ValueError("denom_min must be positive.")
    Pi = np.asarray(Pi)
    q_grid = np.asarray(q_grid, dtype=float)
    Vc = np.asarray(Vc)
    Vtot = np.asarray(Vtot)
    if q_interp_mode == "q_singular" and np.any(q_grid <= 0.0):
        raise ValueError("q_singular mode requires strictly positive q_grid.")
    denom = 1.0 - Pi * Vtot
    expected_shape = (len(q_grid), len(ctx.vs))
    if denom.shape != expected_shape:
        raise ValueError(
            "RPA denominator shape %s != expected %s" % (denom.shape, expected_shape)
        )
    if not np.all(np.isfinite(denom)):
        raise FloatingPointError("non-finite entries in RPA denominator.")
    abs_denom = np.abs(denom)
    worst = np.unravel_index(np.argmin(abs_denom), denom.shape)
    metrics = DenomMetrics(
        policy=denom_policy,
        min_abs=float(abs_denom[worst]),
        worst_q=float(q_grid[worst[0]]),
        worst_v=float(ctx.vs[worst[1]]),
        worst_pi=complex(Pi[worst]),
        worst_vtot=complex(Vtot[worst]),
        worst_old=complex(denom[worst]),
        worst_new=complex(denom[worst]),
    )
    if denom_policy == "none":
        metrics.count = 0
    else:
        small = np.abs(denom) < float(denom_min)
        metrics.count = int(np.count_nonzero(small))
        if np.any(small):
            if denom_policy == "fail":
                iq, iv = np.argwhere(small)[0]
                raise FloatingPointError(
                    "RPA denominator below denom_min at q=%s iv=%s Pi=%s "
                    "Vtot=%s denom=%s"
                    % (q_grid[iq], ctx.vs[iv], Pi[iq, iv], Vtot[iq, iv], denom[iq, iv])
                )
            denom = np.array(denom, copy=True)
            if denom_policy == "legacy_clip":
                denom[small] = float(denom_min)
            else:
                complex_phase = np.iscomplexobj(denom) and max_abs(denom.imag) > 0.0
                for iq, iv in zip(*np.where(small)):
                    old = denom[iq, iv]
                    if abs(old) > 0.0:
                        if complex_phase:
                            direction = old / abs(old)
                            undetermined = False
                        else:
                            direction = np.sign(old.real)
                            undetermined = False
                            if direction == 0.0:
                                direction, undetermined = _neighbor_direction(
                                    denom, iq, iv, complex_phase=False)
                    else:
                        direction, undetermined = _neighbor_direction(
                            denom, iq, iv, complex_phase=complex_phase)
                    if undetermined:
                        metrics.undetermined_count += 1
                    denom[iq, iv] = direction * float(denom_min)
            metrics.worst_new = complex(denom[worst])
    W = Vtot / denom
    if not np.all(np.isfinite(W)):
        raise FloatingPointError("non-finite entries in W.")
    metrics.max_abs_W = max_abs(W)
    if q_interp_mode == "q_singular":
        head_source = q_singular_head_policy
        if q_singular_head_policy == "pi_head":
            if screened_singular_coeff is not None:
                raise ValueError(
                    "screened_singular_coeff requires "
                    "q_singular_head_policy='provided'."
                )
            screened_singular_coeff, head_info = (
                estimate_screened_singular_coeff_from_pi(
                    q_grid,
                    Pi,
                    Vtot,
                    ctx.vs,
                    vtot_head=vtot_head,
                    nfit=singular_fit_n,
                    degree=singular_fit_degree,
                    denom_policy=denom_policy,
                    denom_min=denom_min,
                )
            )
        elif q_singular_head_policy == "provided":
            if screened_singular_coeff is None:
                raise ValueError(
                    "q_singular_head_policy='provided' requires "
                    "screened_singular_coeff."
                )
            head_info = {
                "head_min_abs_denom": np.inf,
                "head_regularized_count": 0,
            }
        else:
            if screened_singular_coeff is not None:
                raise ValueError(
                    "screened_singular_coeff is incompatible with "
                    "q_singular_head_policy='fit_w'."
                )
            head_info = {
                "head_min_abs_denom": np.inf,
                "head_regularized_count": 0,
            }
        precomp, fit_diag = build_q_singular_precomp(
            ctx,
            q_grid,
            W,
            Vc,
            static_singular_coeff=static_singular_coeff,
            screened_singular_coeff=screened_singular_coeff,
            singular_head_source=head_source,
            singular_fit_n=singular_fit_n,
            singular_fit_degree=singular_fit_degree,
            singular_fit_warn_rtol=singular_fit_warn_rtol,
            singular_fit_warn_atol=singular_fit_warn_atol,
            warning_log=warning_log,
        )
        fit_diag.head_min_abs_denom = head_info["head_min_abs_denom"]
        fit_diag.head_regularized_count = head_info["head_regularized_count"]
        metrics.singular_fit = fit_diag
    else:
        dW_tau = ctx.w2t_b2f(W - Vc[:, None])
        precomp = WPrecomp(q_grid, dW_tau, Vc, interp_mode=q_interp_mode)
    if return_metrics:
        return precomp, metrics
    return precomp


def _neighbor_direction(arr, iq, iv, complex_phase=False):
    nq, nv = arr.shape
    for radius in (1, 2):
        vals = []
        for dq in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if dq == 0 and dv == 0:
                    continue
                jq = iq + dq
                jv = iv + dv
                if 0 <= jq < nq and 0 <= jv < nv:
                    val = arr[jq, jv]
                    if np.isfinite(val) and abs(val) > 0.0:
                        vals.append(val)
        if vals:
            vals = np.asarray(vals)
            if complex_phase:
                phase = np.sum(vals / np.abs(vals))
                if abs(phase) > 0.0:
                    return phase / abs(phase), False
            else:
                signs = np.sign(vals.real)
                signs = signs[signs != 0.0]
                if signs.size:
                    sign_sum = np.sum(signs)
                    if sign_sum != 0.0:
                        return np.sign(sign_sum), False
    return (1.0 + 0.0j) if complex_phase else 1.0, True


def _validate_Pi0_block_shape(Pi, q_grid, v_grid):
    expected = (len(q_grid), len(v_grid))
    if Pi.shape != expected:
        raise ValueError(
            "Pi0 builder returned shape %s for q block length %d; expected %s."
            % (Pi.shape, len(q_grid), expected)
        )
    return Pi


def _call_Pi0_builder(get_Pi0, q_grid, v_grid, kwargs_get_Pi, tol):
    """Call a model Pi0 builder on a q block and the full bosonic grid."""
    Pi = np.asarray(
        get_Pi0(q_grid, v_grid, tol=tol, **kwargs_get_Pi),
        dtype=float,
    )
    return _validate_Pi0_block_shape(Pi, q_grid, v_grid)


@_lazy_parallel_call
def _get_Pi0_q_blocks_mpi(_, get_Pi0, q_grid, v_grid, kwargs_get_Pi=None,
                          tol=1e-12):
    kwargs_get_Pi = {} if kwargs_get_Pi is None else dict(kwargs_get_Pi)
    qv = np.asarray(q_grid, dtype=float)
    vv = np.asarray(v_grid, dtype=float)
    qloc0, qloc1 = _task_location(len(qv))
    local = _call_Pi0_builder(
        get_Pi0,
        qv[qloc0:qloc1],
        vv,
        kwargs_get_Pi,
        tol,
    )
    return _gather_rank0(local)


def get_Pi0_q_blocks(_, get_Pi0, q_grid, v_grid, kwargs_get_Pi=None,
                     tol=1e-12):
    r"""Return ``Pi0(q, i nu)`` with optional MPI q blocking.

    ``model.get_Pi`` and ``model.get_Pi0_finite_T`` are pure formula builders;
    this wrapper adds the radial execution policy used by ``get_W0``.  The q
    axis is distributed under MPI because output rows, Coulomb factors, phonon
    factors, and q-singular head diagnostics are all q-major.
    """
    kwargs_get_Pi = {} if kwargs_get_Pi is None else dict(kwargs_get_Pi)
    qv = np.asarray(q_grid, dtype=float)
    vv = np.asarray(v_grid, dtype=float)
    # Keep ordinary serial W0 construction from initializing MPI, while still
    # honoring explicit mpi4pyscf imports and common launcher world-size hints.
    if (
        not mpi_wrap.loaded()
        and "mpi4pyscf.tools.mpi" not in sys.modules
        and _mpi_world_size_hint() <= 1
    ):
        return _call_Pi0_builder(get_Pi0, qv, vv, kwargs_get_Pi, tol)
    return _get_Pi0_q_blocks_mpi(
        None, get_Pi0, qv, vv, kwargs_get_Pi=kwargs_get_Pi, tol=tol)


def get_W0(ctx, get_Pi0, get_Vc, Vph, eps_inf, q_grid,
           kwargs_get_Pi=None, q0_treat=False, tol=1e-12,
           denom_policy="fail", denom_min=1e-10, return_metrics=False,
           q_interp_mode="legacy", static_singular_coeff=None,
           q_singular_head_policy="pi_head", screened_singular_coeff=None,
           vtot_head=None,
           singular_fit_n=6, singular_fit_degree=1,
           singular_fit_warn_rtol=1.0e-3,
           singular_fit_warn_atol=1.0e-10,
           warning_log=None):
    r"""Build fixed RPA ``W0`` tables for G0W0/GW0.

    Parameters
    ----------
    ctx : :class:`migdal.freq.FreqGrid`
        Sparse-IR context defining the bosonic Matsubara grid.
    get_Pi0, get_Vc : callable
        Model-specific polarization and Coulomb table builders.
    Vph : (nv,) or (nq, nv) array_like
        Phonon dielectric factor on ``ctx.vs``.  One-dimensional input is
        treated as q independent; two-dimensional input is ``Vph(q, i*nu)``.
        Two-dimensional input can represent a q-dependent phonon factor; see
        ``vtot_head`` for the q-singular head handling.
    eps_inf : float
        High-frequency dielectric constant.
    q_grid : (nq,) array_like
        Radial momentum grid in Bohr^-1.
    q0_treat : bool, optional
        Unused q=0 special-case switch; passing ``True`` raises.
    q_interp_mode : {"legacy", "q_singular"}, optional
        W interpolation mode.  ``q_singular`` separates the Coulomb
        ``A/q**2`` head before building radial primitives.
    q_singular_head_policy : {"pi_head", "fit_w", "provided"}, optional
        Source for the screened ``q_singular`` head.  The default ``pi_head``
        derives it from the low-q expansion of ``Pi`` and ``Vtot``.
    vtot_head : (nv,) array_like, optional
        Analytic bare-interaction head
        ``B(i nu) = lim(q -> 0) q**2 Vtot(q, i nu)``.  For a regular
        q-dependent phonon table this is
        ``(4*pi/eps_inf) * Vph(q=0, i nu)``.  If omitted with a two-dimensional
        ``Vph`` table in ``q_singular``/``pi_head`` mode, ``get_W0`` warns and
        falls back to fitting ``B`` from the sampled low-q rows.
    warning_log : PySCF logger, optional
        Logger used for warnings about low-q head fitting.  If omitted,
        warnings are emitted to a default WARN-level stream logger.

    Notes
    -----
    The screened interaction is evaluated as

    .. math::

        W_0(q, i\nu) =
        \frac{V_c(q) V_{\mathrm{ph}}(q, i\nu)}
             {1 - \Pi_0(q, i\nu) V_c(q) V_{\mathrm{ph}}(q, i\nu)}.

    Passing ``q0_treat=True`` raises because Migdal fixes the explicit q=0
    Coulomb table value to zero and treats the singular limit through radial
    integration, not through a special grid point.
    """
    if q0_treat:
        raise ValueError(
            "q0_treat=True is not implemented: Migdal fixes the q=0 Coulomb "
            "grid value to zero, while the "
            "singular limit is treated by radial integration."
        )
    q_interp_mode = _validate_w_interp_mode(q_interp_mode)
    q_singular_head_policy = _validate_q_singular_head_policy(
        q_singular_head_policy)
    kwargs_get_Pi = {} if kwargs_get_Pi is None else dict(kwargs_get_Pi)
    # q_singular needs the Coulomb tail at every positive production q point;
    # keep model.get_Vc from zeroing the smallest sampled q by tolerance.
    vc_tol = np.nextafter(0.0, 1.0) if q_interp_mode == "q_singular" else tol
    Vc = get_Vc(q_grid, eps_inf, q0_treat=False, tol=vc_tol)
    Vph = np.asarray(Vph)
    nq = len(q_grid)
    nv = len(ctx.vs)
    shape_error = "Vph must have shape (%d,) or (%d, %d); got %s." % (
        nv, nq, nv, Vph.shape
    )
    if Vph.ndim == 1:
        if Vph.shape != (nv,):
            raise ValueError(shape_error)
        Vph = Vph[None, :]
    elif Vph.ndim == 2:
        if Vph.shape != (nq, nv):
            raise ValueError(shape_error)
    else:
        raise ValueError(shape_error)
    Vtot = Vc[:, None] * Vph
    Pi = get_Pi0_q_blocks(
        None, get_Pi0, q_grid, ctx.vs, kwargs_get_Pi=kwargs_get_Pi, tol=tol)
    if q_interp_mode == "q_singular" and static_singular_coeff is None:
        static_singular_coeff = 4.0 * np.pi / float(eps_inf)
    # A 1D Vph table is q independent, so its q=0 value is known here.  A 2D
    # table only gives sampled q rows; callers with an analytic q->0 limit
    # should pass vtot_head explicitly to avoid a low-q fit of the bare head.
    if (
        q_interp_mode == "q_singular"
        and vtot_head is None
        and Vph.shape[0] == 1
    ):
        vtot_head = np.asarray(static_singular_coeff) * Vph[0]
    elif (
        q_interp_mode == "q_singular"
        and q_singular_head_policy == "pi_head"
        and vtot_head is None
        and Vph.shape[0] != 1
    ):
        mlog.warning(
            warning_log,
            "q_singular pi_head received a two-dimensional Vph table without "
            "vtot_head; "
            "falling back to a low-q fit of B = lim(q -> 0) q**2 Vtot. "
            "Pass vtot_head for an analytic bare-interaction head.",
        )
    return get_W_from_Pi(
        ctx, Pi, q_grid, Vc, Vtot,
        denom_policy=denom_policy, denom_min=denom_min,
        return_metrics=return_metrics,
        q_interp_mode=q_interp_mode,
        static_singular_coeff=static_singular_coeff,
        q_singular_head_policy=q_singular_head_policy,
        screened_singular_coeff=screened_singular_coeff,
        vtot_head=vtot_head,
        singular_fit_n=singular_fit_n,
        singular_fit_degree=singular_fit_degree,
        singular_fit_warn_rtol=singular_fit_warn_rtol,
        singular_fit_warn_atol=singular_fit_warn_atol,
        warning_log=warning_log,
    )


@_lazy_parallel_call
def get_Pi(_, ctx, G_tau, k_grid, k_wts, q_grid, beta_index=None,
           q_zero_tol=1e-10):
    r"""Return the radial bubble ``Pi(q, tau)`` over MPI q blocks.

    The input ``G_tau`` is sampled on the fermionic-to-bosonic tau grid with
    shape ``(nk, nt_b)``.  The result has shape ``(nq, nt_b)`` and uses the
    spin-summed isotropic convention

    .. math::

        \Pi(q,\tau) = -\frac{1}{2\pi^2 q}
            \int dk\, k\, G(k,\tau)
            \int_{|k-q|}^{k+q} dp\, p\, G(p,\beta-\tau),

    with the analytic ``q -> 0`` limit used for ``abs(q) <= q_zero_tol``.
    """
    k = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    qv = np.asarray(q_grid, dtype=float)
    Gt = np.asarray(G_tau)
    nk, nt = Gt.shape
    if k_wts.shape != k.shape or nk != k.size:
        raise ValueError("inconsistent k_grid/k_wts/G_tau shapes.")

    Gb = Gt[:, ::-1] if beta_index is None else Gt[:, np.asarray(beta_index, int)]
    precomp = GPrecomp(k, Gb)
    k2GG_sum = np.dot(k * k * k_wts, Gt * Gb) * 2.0
    kG = (k_wts * k)[:, None] * Gt
    pref = -1.0 / (2.0 * np.pi**2)

    qloc0, qloc1 = _task_location(len(qv))
    if _cython_radial_available():
        k_c = np.ascontiguousarray(k, dtype=np.float64)
        qv_c = np.ascontiguousarray(qv, dtype=np.float64)
        kG_c = np.ascontiguousarray(kG, dtype=np.complex128)
        k2GG_sum_c = np.ascontiguousarray(k2GG_sum, dtype=np.complex128)
        gargs = _gprecomp_cython_args(precomp)
        geom = _pi_pair_geometry(k_c, qv_c, precomp, qloc0, qloc1)
        if geom is not None and hasattr(_radial_kernels, "pi_kernel_geometry"):
            omp_threads = _radial_openmp_threads(
                (qloc1 - qloc0) * len(k_c) * nt
            )
            if omp_threads > 1 and hasattr(_radial_kernels, "pi_kernel_geometry_omp"):
                Pi_tau = _radial_kernels.pi_kernel_geometry_omp(
                    qv_c,
                    kG_c,
                    k2GG_sum_c,
                    *gargs[3:],
                    *geom,
                    qloc0,
                    float(q_zero_tol),
                    omp_threads,
                )
            else:
                Pi_tau = _radial_kernels.pi_kernel_geometry(
                    qv_c,
                    kG_c,
                    k2GG_sum_c,
                    *gargs[3:],
                    *geom,
                    qloc0,
                    float(q_zero_tol),
                )
        else:
            Pi_tau = _radial_kernels.pi_kernel(
                k_c,
                qv_c,
                kG_c,
                k2GG_sum_c,
                *gargs,
                qloc0,
                qloc1,
                float(q_zero_tol),
            )
        return _gather_rank0(Pi_tau * pref)

    Pi_tau = np.zeros((qloc1 - qloc0, nt), dtype=complex)
    for local_i, iq in enumerate(range(qloc0, qloc1)):
        q = qv[iq]
        if abs(q) <= q_zero_tol:
            Pi_tau[local_i] = k2GG_sum
            continue
        dI = precomp.integral_block_dyn(k, q) / q
        Pi_tau[local_i] = np.einsum("kt,kt->t", kG, dI, optimize=True)
    return _gather_rank0(Pi_tau * pref)


def _weighted_sources(k_grid, k_wts, S_t, S_beta, S_has_k):
    if S_has_k:
        weight = np.asarray(k_wts, dtype=float)
    else:
        weight = np.asarray(k_grid, dtype=float) * np.asarray(k_wts, dtype=float)
    return S_t * weight[:, None], S_beta * weight


def _origin_sigma_row(ctx, k_grid, k_wts, S_t, S_beta, precomp, S_has_k):
    if S_has_k:
        row0_weight = (k_grid * k_wts)[:, None]
        q2_limit = None
    else:
        row0_weight = (k_grid**2 * k_wts)[:, None]
        q2_limit = k_wts
    if precomp.interp_mode == "q_singular":
        if S_has_k and precomp.has_singularity and np.any(k_grid == 0.0):
            raise FloatingPointError(
                "scaled-source origin row with q_singular W needs k_grid > 0."
            )
        dv_tau_at_k = precomp.eval_dyn_weighted(
            k_grid, row0_weight[:, 0], q2_weight_limit=q2_limit)
        vc_at_k = precomp.eval_sta_weighted(
            k_grid, row0_weight[:, 0], q2_weight_limit=q2_limit)
    else:
        dv_tau_at_k = precomp.eval_dyn(k_grid) * row0_weight
        vc_at_k = precomp.eval_sta(k_grid) * row0_weight[:, 0]
    sigma_row0 = 2.0 * np.einsum(
        "kt,kt->t", S_t, dv_tau_at_k, optimize=True
    )
    sigma_row0 = ctx.t2w_f(sigma_row0[None])[0]
    sigma0_row0 = 2.0 * np.dot(S_beta, vc_at_k)
    return (sigma_row0 + sigma0_row0) * (-1.0 / ((2.0 * np.pi) ** 2))


def _cython_radial_available():
    """Return whether the optional compiled radial kernels are importable."""
    return _radial_kernels is not None


def _qw_cython_available():
    """Return whether the optional compiled qW radial kernels are importable."""
    return _qw_radial_kernels is not None


def _openmp_threads_for_module(work_items, kernel_module):
    if (
        _RADIAL_OPENMP_DISABLED
        or kernel_module is None
        or not hasattr(kernel_module, "openmp_max_threads")
        or int(work_items) < _RADIAL_OPENMP_MIN_WORK
    ):
        return 1

    raw = os.environ.get("MIGDAL_RADIAL_OPENMP_THREADS")
    if raw is None:
        raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return 1
    raw = str(raw).split(",", 1)[0].strip()
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        return 1
    if requested <= 1:
        return 1
    try:
        max_threads = int(kernel_module.openmp_max_threads())
    except Exception:
        max_threads = requested
    return max(1, min(requested, max_threads))


def _radial_openmp_threads(work_items):
    return _openmp_threads_for_module(
        work_items, _radial_kernels if _cython_radial_available() else None
    )


def _qw_radial_openmp_threads(work_items):
    return _openmp_threads_for_module(
        work_items, _qw_radial_kernels if _qw_cython_available() else None
    )


def _sigma_row_openmp_threads(nrows, nk, nt):
    work_items = int(nrows) * max(int(nk), 1) * max(int(nt), 1)
    return _radial_openmp_threads(work_items)


def _qw_sigma_row_openmp_threads(nrows, nk, nt):
    work_items = int(nrows) * max(int(nk), 1) * max(int(nt), 1)
    return _qw_radial_openmp_threads(work_items)


def _pair_geometry_openmp_threads(nrows, ncols):
    work_items = int(nrows) * max(int(ncols), 1)
    return _radial_openmp_threads(work_items)


def _qw_pair_geometry_openmp_threads(nrows, ncols):
    work_items = int(nrows) * max(int(ncols), 1)
    return _qw_radial_openmp_threads(work_items)


def _triangular_openmp_threads(nk, nt, kernel_module, omp_kernel_name, thread_func):
    """Return a safe OpenMP thread count for private-output triangular kernels."""
    if kernel_module is None or not hasattr(kernel_module, omp_kernel_name):
        return 1
    pair_work = (int(nk) * (int(nk) + 1) // 2) * max(int(nt), 1)
    threads = thread_func(pair_work)
    if threads <= 1:
        return 1
    private_bytes = (
        threads * int(nk) * (int(nt) + 1) * np.dtype(np.complex128).itemsize
    )
    limit_bytes = int(_SIGMA_OMP_PRIVATE_MB * 1024.0 * 1024.0)
    if limit_bytes > 0 and private_bytes > limit_bytes:
        return 1
    return threads


def _sigma_triangular_openmp_threads(nk, nt):
    return _triangular_openmp_threads(
        nk, nt, _radial_kernels, "sigma_kernel_geometry_triangular_omp",
        _radial_openmp_threads,
    )


def _qw_sigma_triangular_openmp_threads(nk, nt):
    return _triangular_openmp_threads(
        nk, nt, _qw_radial_kernels, "qw_sigma_kernel_geometry_triangular_omp",
        _qw_radial_openmp_threads,
    )


def _empty_diag_arrays():
    return (
        np.empty((1, 1), dtype=np.complex128),
        np.empty(1, dtype=np.complex128),
    )


def _writable_complex128_array(arr):
    out = np.asarray(arr, dtype=np.complex128)
    if not out.flags.c_contiguous or not out.flags.writeable:
        out = np.array(out, dtype=np.complex128, copy=True, order="C")
    return out


def _wprecomp_cython_args(precomp):
    if not precomp.uses_legacy_primitives:
        raise TypeError("Cython W kernels only support legacy WPrecomp objects.")
    cache = getattr(precomp, "_cython_cache", None)
    if cache is None:
        cache = (
            np.ascontiguousarray(precomp.q, dtype=np.float64),
            np.ascontiguousarray(precomp.q2, dtype=np.float64),
            np.ascontiguousarray(precomp.q3, dtype=np.float64),
            np.ascontiguousarray(precomp.yD, dtype=np.complex128),
            np.ascontiguousarray(precomp.sD, dtype=np.complex128),
            np.ascontiguousarray(precomp.PD, dtype=np.complex128),
            np.ascontiguousarray(precomp.yS, dtype=np.complex128),
            np.ascontiguousarray(precomp.sS, dtype=np.complex128),
            np.ascontiguousarray(precomp.PS, dtype=np.complex128),
        )
        precomp._cython_cache = cache
    return cache


def _qwprecomp_cython_args(precomp):
    if getattr(precomp, "interp_mode", None) != "q_singular":
        raise TypeError("Cython qW kernels only support q_singular precomp objects.")
    cache = getattr(precomp, "_cython_cache", None)
    if cache is None:
        singular_sta = complex(np.asarray(precomp.singular_sta).reshape(()).item())
        cache = (
            np.ascontiguousarray(precomp.qh, dtype=np.float64),
            np.ascontiguousarray(precomp.hD, dtype=np.complex128),
            np.ascontiguousarray(precomp.shD, dtype=np.complex128),
            np.ascontiguousarray(precomp.PhD, dtype=np.complex128),
            np.ascontiguousarray(precomp.hS, dtype=np.complex128),
            np.ascontiguousarray(precomp.shS, dtype=np.complex128),
            np.ascontiguousarray(precomp.PhS, dtype=np.complex128),
            _writable_complex128_array(precomp.singular_dyn),
            singular_sta,
            bool(getattr(precomp, "_has_singular_dyn", max_abs(precomp.singular_dyn) > 0.0)),
            bool(getattr(precomp, "_has_singular_sta", abs(precomp.singular_sta) > 0.0)),
        )
        precomp._cython_cache = cache
    return cache


def _gprecomp_cython_args(precomp):
    cache = getattr(precomp, "_cython_cache", None)
    if cache is None:
        cache = (
            np.ascontiguousarray(precomp.k, dtype=np.float64),
            np.ascontiguousarray(precomp.k2, dtype=np.float64),
            np.ascontiguousarray(precomp.k3, dtype=np.float64),
            np.ascontiguousarray(precomp.yD, dtype=np.complex128),
            np.ascontiguousarray(precomp.sD, dtype=np.complex128),
            np.ascontiguousarray(precomp.PD, dtype=np.complex128),
        )
        precomp._cython_cache = cache
    return cache


def _array_value_key(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.shape, arr.tobytes()


def _pair_geometry_bytes(nloc, nsource):
    return int(
        int(nloc) * int(nsource) * (
            2 * np.dtype(np.intc).itemsize + 4 * np.dtype(np.float64).itemsize
        )
    )


def _qw_pair_geometry_bytes(nloc, nsource):
    return int(
        int(nloc) * int(nsource) * (
            2 * np.dtype(np.intc).itemsize + 5 * np.dtype(np.float64).itemsize
        )
    )


def _qw_diag_cell_geometry_bytes(nk, max_points):
    return int(
        int(nk) * (
            np.dtype(np.intc).itemsize + np.dtype(np.float64).itemsize
        )
        + int(nk) * int(max_points) * (
            2 * np.dtype(np.intc).itemsize + 5 * np.dtype(np.float64).itemsize
        )
    )


def _pair_geometry_nbytes(geom):
    return sum(int(getattr(arr, "nbytes", 0)) for arr in geom)


def _trim_pair_geometry_cache(cache, max_items, max_mb):
    limit_bytes = int(float(max_mb) * 1024.0 * 1024.0)
    total_bytes = sum(_pair_geometry_nbytes(geom) for geom in cache.values())
    while cache and (len(cache) > int(max_items) or (
        limit_bytes > 0 and total_bytes > limit_bytes
    )):
        _, geom = cache.popitem(last=False)
        total_bytes -= _pair_geometry_nbytes(geom)


def _sigma_pair_geometry(k_grid, precomp, kloc0, kloc1):
    if (
        not _cython_radial_available()
        or not hasattr(_radial_kernels, "build_sigma_pair_geometry")
        or _SIGMA_PAIR_GEOMETRY_CACHE_MAX <= 0
    ):
        return None
    k_grid = np.ascontiguousarray(k_grid, dtype=np.float64)
    q, q2, q3, *_ = _wprecomp_cython_args(precomp)
    if (
        k_grid.ndim != 1
        or q.ndim != 1
        or (k_grid.size > 1 and not np.all(k_grid[1:] >= k_grid[:-1]))
    ):
        return None

    nloc = int(kloc1) - int(kloc0)
    geom_bytes = _pair_geometry_bytes(nloc, len(k_grid))
    limit_bytes = int(_SIGMA_PAIR_GEOMETRY_CACHE_MB * 1024.0 * 1024.0)
    if limit_bytes > 0 and geom_bytes > limit_bytes:
        return None

    key = (
        mpi.rank,
        int(kloc0),
        int(kloc1),
        _array_value_key(k_grid),
        _array_value_key(q),
    )
    geom = _SIGMA_PAIR_GEOMETRY_CACHES.get(key)
    if geom is not None:
        _SIGMA_PAIR_GEOMETRY_CACHES.move_to_end(key)
        return geom

    geom = _radial_kernels.build_sigma_pair_geometry(
        k_grid, q, q2, q3, int(kloc0), int(kloc1),
        _pair_geometry_openmp_threads(nloc, len(k_grid)),
    )
    _SIGMA_PAIR_GEOMETRY_CACHES[key] = geom
    _SIGMA_PAIR_GEOMETRY_CACHES.move_to_end(key)
    _trim_pair_geometry_cache(
        _SIGMA_PAIR_GEOMETRY_CACHES,
        _SIGMA_PAIR_GEOMETRY_CACHE_MAX,
        _SIGMA_PAIR_GEOMETRY_CACHE_MB,
    )
    return geom


def _sigma_full_pair_geometry(k_grid, precomp):
    if (
        _SIGMA_TRIANGULAR_DISABLED
        or not _cython_radial_available()
        or not hasattr(_radial_kernels, "sigma_kernel_geometry_triangular")
    ):
        return None
    return _sigma_pair_geometry(k_grid, precomp, 0, len(k_grid))


def _qw_sigma_pair_geometry(k_grid, precomp, kloc0, kloc1):
    if (
        not _qw_cython_available()
        or not hasattr(_qw_radial_kernels, "build_qw_sigma_pair_geometry")
        or _QW_SIGMA_PAIR_GEOMETRY_CACHE_MAX <= 0
    ):
        return None
    k_grid = np.ascontiguousarray(k_grid, dtype=np.float64)
    qh = _qwprecomp_cython_args(precomp)[0]
    if k_grid.ndim != 1 or qh.ndim != 1:
        return None
    if k_grid.size > 1:
        dk = np.diff(k_grid)
        if np.any(dk < 0.0):
            return None
        if precomp.has_singularity and np.any(dk <= 0.0):
            return None

    nloc = int(kloc1) - int(kloc0)
    geom_bytes = _qw_pair_geometry_bytes(nloc, len(k_grid))
    limit_bytes = int(_QW_SIGMA_PAIR_GEOMETRY_CACHE_MB * 1024.0 * 1024.0)
    if limit_bytes > 0 and geom_bytes > limit_bytes:
        return None

    key = (
        mpi.rank,
        int(kloc0),
        int(kloc1),
        _array_value_key(k_grid),
        _array_value_key(qh),
    )
    geom = _QW_SIGMA_PAIR_GEOMETRY_CACHES.get(key)
    if geom is not None:
        _QW_SIGMA_PAIR_GEOMETRY_CACHES.move_to_end(key)
        return geom

    geom = _qw_radial_kernels.build_qw_sigma_pair_geometry(
        k_grid, qh, int(kloc0), int(kloc1),
        _qw_pair_geometry_openmp_threads(nloc, len(k_grid)),
    )
    _QW_SIGMA_PAIR_GEOMETRY_CACHES[key] = geom
    _QW_SIGMA_PAIR_GEOMETRY_CACHES.move_to_end(key)
    _trim_pair_geometry_cache(
        _QW_SIGMA_PAIR_GEOMETRY_CACHES,
        _QW_SIGMA_PAIR_GEOMETRY_CACHE_MAX,
        _QW_SIGMA_PAIR_GEOMETRY_CACHE_MB,
    )
    return geom


def _qw_sigma_full_pair_geometry(k_grid, precomp):
    if (
        _SIGMA_TRIANGULAR_DISABLED
        or not _qw_cython_available()
        or not hasattr(_qw_radial_kernels, "qw_sigma_kernel_geometry_triangular")
    ):
        return None
    return _qw_sigma_pair_geometry(k_grid, precomp, 0, len(k_grid))


def _qw_diag_cell_geometry(k_grid, bounds, xw, precomp):
    if (
        not _qw_cython_available()
        or getattr(precomp, "interp_mode", None) != "q_singular"
        or not hasattr(_qw_radial_kernels, "build_qw_diag_cell_geometry")
    ):
        return None
    qh = _qwprecomp_cython_args(precomp)[0]
    x_nodes, w_nodes = xw
    k_grid = np.ascontiguousarray(k_grid, dtype=np.float64)
    bounds = np.ascontiguousarray(bounds, dtype=np.float64)
    x_nodes = np.ascontiguousarray(x_nodes, dtype=np.float64)
    w_nodes = np.ascontiguousarray(w_nodes, dtype=np.float64)
    if (
        k_grid.ndim != 1
        or bounds.shape != (len(k_grid), 2)
        or x_nodes.ndim != 1
        or w_nodes.shape != x_nodes.shape
        or qh.ndim != 1
    ):
        return None

    geom_bytes = _qw_diag_cell_geometry_bytes(len(k_grid), 2 * len(x_nodes))
    limit_bytes = int(_QW_DIAG_CELL_GEOMETRY_CACHE_MB * 1024.0 * 1024.0)
    if limit_bytes > 0 and geom_bytes > limit_bytes:
        return None

    key = (
        _array_value_key(k_grid),
        _array_value_key(bounds),
        _array_value_key(x_nodes),
        _array_value_key(w_nodes),
        _array_value_key(qh),
    )
    if _QW_DIAG_CELL_GEOMETRY_CACHE_MAX > 0:
        geom = _QW_DIAG_CELL_GEOMETRY_CACHES.get(key)
        if geom is not None:
            _QW_DIAG_CELL_GEOMETRY_CACHES.move_to_end(key)
            return geom

    geom = _qw_radial_kernels.build_qw_diag_cell_geometry(
        k_grid, bounds, x_nodes, w_nodes, qh,
        _qw_pair_geometry_openmp_threads(len(k_grid), 2 * len(x_nodes)),
    )
    if _QW_DIAG_CELL_GEOMETRY_CACHE_MAX > 0:
        _QW_DIAG_CELL_GEOMETRY_CACHES[key] = geom
        _QW_DIAG_CELL_GEOMETRY_CACHES.move_to_end(key)
        _trim_pair_geometry_cache(
            _QW_DIAG_CELL_GEOMETRY_CACHES,
            _QW_DIAG_CELL_GEOMETRY_CACHE_MAX,
            _QW_DIAG_CELL_GEOMETRY_CACHE_MB,
        )
    return geom


def _qw_diag_cell_openmp_threads(nrows, max_points, nt):
    work_items = int(nrows) * max(int(max_points), 1) * max(int(nt), 1)
    return _qw_radial_openmp_threads(work_items)


def _qw_diag_cell_cython_context(
    precomp, k_grid, bounds, xw, kloc0, kloc1, kernel_name
):
    if not _qw_cython_available() or not hasattr(_qw_radial_kernels, kernel_name):
        return None
    geom = _qw_diag_cell_geometry(k_grid, bounds, xw, precomp)
    if geom is None:
        return None
    qargs = _qwprecomp_cython_args(precomp)
    threads = _qw_diag_cell_openmp_threads(
        int(kloc1) - int(kloc0), geom[1].shape[1], precomp.nt,
    )
    return geom, qargs, threads


def _safe_qw_diag_cell_dtype_cast(arr, dtype, name):
    dtype = np.dtype(dtype)
    arr = np.asarray(arr)
    if arr.dtype == dtype and arr.flags.c_contiguous:
        return arr

    out = arr
    if dtype.kind != "c" and np.iscomplexobj(out):
        imag_abs = max_abs(out.imag)
        if imag_abs > 0.0:
            real_scale = max(max_abs(out.real), 1.0)
            tol = 100.0 * np.finfo(np.float64).eps * real_scale
            if imag_abs > tol:
                raise ValueError(
                    "qW Cython %s diagonal cell result has imaginary part "
                    "%.3e and cannot be safely cast to %s."
                    % (name, imag_abs, dtype.name)
                )
        out = out.real

    out = out.astype(dtype, copy=False)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out


def _diag_cells_qw_cython(precomp, k_grid, bounds, xw, kloc0, kloc1):
    context = _qw_diag_cell_cython_context(
        precomp, k_grid, bounds, xw, kloc0, kloc1, "qw_diag_cells_kernel"
    )
    if context is None:
        return None
    geom, qargs, threads = context
    _qh, hD, shD, PhD, hS, shS, PhS, singular_dyn, singular_sta, has_dyn, has_sta = (
        qargs
    )
    dyn, sta = _qw_radial_kernels.qw_diag_cells_kernel(
        hD,
        shD,
        PhD,
        hS,
        shS,
        PhS,
        singular_dyn,
        singular_sta,
        has_dyn,
        has_sta,
        *geom,
        int(kloc0),
        int(kloc1),
        threads,
    )
    dyn = _safe_qw_diag_cell_dtype_cast(dyn, precomp.dyn_dtype, "dynamic")
    sta = _safe_qw_diag_cell_dtype_cast(sta, precomp.sta_dtype, "static")
    return dyn, sta


def _diag_cells_dyn_legacy_cython(precomp, k_grid, bounds, xw, kloc0, kloc1):
    if (
        not _cython_radial_available()
        or not precomp.uses_legacy_primitives
        or not hasattr(_radial_kernels, "diag_cells_dyn_kernel")
    ):
        return None
    x, w = xw
    dyn = _radial_kernels.diag_cells_dyn_kernel(
        np.ascontiguousarray(k_grid, dtype=np.float64),
        np.ascontiguousarray(bounds, dtype=np.float64),
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w, dtype=np.float64),
        *_wprecomp_cython_args(precomp)[:6],
        int(kloc0),
        int(kloc1),
    )
    return dyn


def _diag_cells_dyn_qw_cython(precomp, k_grid, bounds, xw, kloc0, kloc1):
    context = _qw_diag_cell_cython_context(
        precomp, k_grid, bounds, xw, kloc0, kloc1, "qw_diag_cells_dyn_kernel"
    )
    if context is None:
        return None
    geom, qargs, threads = context
    _qh, hD, shD, PhD, hS, shS, PhS, singular_dyn, singular_sta, has_dyn, has_sta = (
        qargs
    )
    dyn = _qw_radial_kernels.qw_diag_cells_dyn_kernel(
        hD,
        shD,
        PhD,
        singular_dyn,
        has_dyn,
        *geom,
        int(kloc0),
        int(kloc1),
        threads,
    )
    dyn = _safe_qw_diag_cell_dtype_cast(dyn, precomp.dyn_dtype, "dynamic")
    return dyn


def _reduce_sum_rank0_array(arr):
    return mpi_wrap.reduce(arr)


def _worker_result_discarded():
    return mpi.pool.size > 1 and mpi.rank != 0


def _static_blocks_with_diag_cache_feasible(
    precomp, k_grid, sta_avg, row_start, row_stop
):
    static_blocks_with_diag = getattr(precomp, "static_blocks_with_diag", None)
    if static_blocks_with_diag is None or _QW_STATIC_BLOCK_CACHE_MB <= 0.0:
        return False
    nrows = int(row_stop) - int(row_start)
    if nrows < 0:
        return False
    sta_dtype = np.dtype(getattr(precomp, "sta_dtype", np.complex128))
    cache_bytes = nrows * len(k_grid) * sta_dtype.itemsize
    cache_limit = int(_QW_STATIC_BLOCK_CACHE_MB * 1024.0 * 1024.0)
    return cache_limit > 0 and cache_bytes <= cache_limit


def _static_blocks_with_diag_cached(precomp, k_grid, sta_avg, row_start, row_stop):
    if not _static_blocks_with_diag_cache_feasible(
        precomp, k_grid, sta_avg, row_start, row_stop
    ):
        return None
    return precomp.static_blocks_with_diag(
        k_grid, sta_avg, row_start=int(row_start), row_stop=int(row_stop))


def _should_use_static_blocks_with_diag_cache(
    precomp, k_grid, sta_avg, row_start, row_stop
):
    if not _static_blocks_with_diag_cache_feasible(
        precomp, k_grid, sta_avg, row_start, row_stop
    ):
        return False
    if getattr(precomp, "_static_block_cache", None):
        return True
    calls = int(getattr(precomp, "_sigma_diag_static_cache_calls", 0)) + 1
    setattr(precomp, "_sigma_diag_static_cache_calls", calls)
    return calls >= 2


_SIGMA_BACKEND_PYTHON = "python"
_SIGMA_BACKEND_LEGACY_TRIANGULAR = "legacy_triangular"
_SIGMA_BACKEND_LEGACY_ROW = "legacy_row"
_SIGMA_BACKEND_QW_TRIANGULAR = "qw_triangular"
_SIGMA_BACKEND_QW_ROW = "qw_row"


def _module_has(module, name):
    return module is not None and hasattr(module, name)


def _local_sigma_backend_flags(precomp, k_grid, kloc0, kloc1, use_diag=False):
    flags = {
        _SIGMA_BACKEND_LEGACY_TRIANGULAR: False,
        _SIGMA_BACKEND_LEGACY_ROW: False,
        _SIGMA_BACKEND_QW_TRIANGULAR: False,
        _SIGMA_BACKEND_QW_ROW: False,
    }
    k_grid_c = np.ascontiguousarray(k_grid, dtype=np.float64)

    if _cython_radial_available() and precomp.uses_legacy_primitives:
        flags[_SIGMA_BACKEND_LEGACY_TRIANGULAR] = (
            _module_has(_radial_kernels, "sigma_kernel_geometry_triangular")
            and _sigma_full_pair_geometry(k_grid_c, precomp) is not None
        )
        row_geom = _sigma_pair_geometry(k_grid_c, precomp, kloc0, kloc1)
        flags[_SIGMA_BACKEND_LEGACY_ROW] = (
            (
                _module_has(_radial_kernels, "sigma_kernel_geometry")
                and row_geom is not None
            )
            or _module_has(_radial_kernels, "sigma_kernel")
        )

    qw_ready = (
        _qw_cython_available()
        and getattr(precomp, "interp_mode", None) == "q_singular"
        and not (precomp.has_singularity and not use_diag)
    )
    if qw_ready:
        flags[_SIGMA_BACKEND_QW_TRIANGULAR] = (
            _module_has(_qw_radial_kernels, "qw_sigma_kernel_geometry_triangular")
            and _qw_sigma_full_pair_geometry(k_grid_c, precomp) is not None
        )
        row_geom = _qw_sigma_pair_geometry(k_grid_c, precomp, kloc0, kloc1)
        flags[_SIGMA_BACKEND_QW_ROW] = (
            (
                _module_has(_qw_radial_kernels, "qw_sigma_kernel_geometry")
                and row_geom is not None
            )
            or (
                _module_has(_qw_radial_kernels, "qw_sigma_kernel")
                and not (
                    precomp.has_singularity
                    and np.unique(k_grid_c).size != k_grid_c.size
                )
            )
        )
    return flags


def _choose_sigma_backend(precomp, k_grid, kloc0, kloc1, use_diag=False):
    local = _local_sigma_backend_flags(precomp, k_grid, kloc0, kloc1, use_diag)
    if mpi.pool.size <= 1:
        infos = [local]
    elif mpi.rank == 0:
        infos = mpi.comm.gather(local, root=0)
    else:
        mpi.comm.gather(local, root=0)
        infos = None

    choice = None
    if mpi.rank == 0:
        for backend in (
            _SIGMA_BACKEND_LEGACY_TRIANGULAR,
            _SIGMA_BACKEND_LEGACY_ROW,
            _SIGMA_BACKEND_QW_TRIANGULAR,
            _SIGMA_BACKEND_QW_ROW,
        ):
            if all(info[backend] for info in infos):
                choice = backend
                break
        if choice is None:
            choice = _SIGMA_BACKEND_PYTHON
    if mpi.pool.size > 1:
        choice = mpi.comm.bcast(choice, root=0)
    return choice


def _sigma_qw_cython(
    k_grid, weighted_S_t, weighted_S_beta, precomp, kloc0, kloc1,
    use_diag=False, dyn_avg=None, sta_avg=None, backend=None,
    skip_static=False,
):
    """Try the compiled qW sigma backend, returning ``None`` on fallback.

    Successful calls return ``(Sigma_t, Sigma0, full_output)`` before the final
    Matsubara transform/static prefactor step.  ``full_output=True`` means the
    triangular kernel has already produced the full k grid and performed the
    MPI sum, so callers must not allgather it again.  ``full_output=False`` is a
    rank-local rowwise result for ``kloc0:kloc1``.

    q-singular inputs with a nonzero singular tail require averaged diagonal
    cells; without them this helper deliberately declines the compiled path so
    the Python fallback preserves the existing singular-cell behavior.
    """
    if (
        not _qw_cython_available()
        or getattr(precomp, "interp_mode", None) != "q_singular"
    ):
        return None
    if precomp.has_singularity and not use_diag:
        return None

    k_grid_c = np.ascontiguousarray(k_grid, dtype=np.float64)
    weighted_S_t_c = np.ascontiguousarray(weighted_S_t, dtype=np.complex128)
    weighted_S_beta_c = np.ascontiguousarray(weighted_S_beta, dtype=np.complex128)
    if use_diag:
        dyn_avg_c = _writable_complex128_array(dyn_avg)
        sta_avg_c = _writable_complex128_array(sta_avg)
    else:
        dyn_avg_c, sta_avg_c = _empty_diag_arrays()
    qargs = _qwprecomp_cython_args(precomp)

    full_geom = None
    if backend in (None, _SIGMA_BACKEND_QW_TRIANGULAR):
        full_geom = _qw_sigma_full_pair_geometry(k_grid_c, precomp)
    if full_geom is not None:
        omp_threads = _qw_sigma_triangular_openmp_threads(
            len(k_grid_c), weighted_S_t_c.shape[1]
        )
        if (
            omp_threads > 1
            and hasattr(_qw_radial_kernels, "qw_sigma_kernel_geometry_triangular_omp")
        ):
            Sigma_t, Sigma0 = _qw_radial_kernels.qw_sigma_kernel_geometry_triangular_omp(
                weighted_S_t_c,
                weighted_S_beta_c,
                *qargs[1:],
                *full_geom,
                mpi.rank,
                mpi.pool.size,
                _SIGMA_TRIANGULAR_TILE,
                bool(use_diag),
                dyn_avg_c,
                sta_avg_c,
                bool(skip_static),
                omp_threads,
            )
        else:
            Sigma_t, Sigma0 = _qw_radial_kernels.qw_sigma_kernel_geometry_triangular(
                weighted_S_t_c,
                weighted_S_beta_c,
                *qargs[1:],
                *full_geom,
                mpi.rank,
                mpi.pool.size,
                _SIGMA_TRIANGULAR_TILE,
                bool(use_diag),
                dyn_avg_c,
                sta_avg_c,
                bool(skip_static),
            )
        Sigma_t = _reduce_sum_rank0_array(Sigma_t)
        Sigma0 = _reduce_sum_rank0_array(Sigma0)
        return Sigma_t, Sigma0, True

    if backend == _SIGMA_BACKEND_QW_TRIANGULAR:
        return None

    geom = _qw_sigma_pair_geometry(k_grid_c, precomp, kloc0, kloc1)
    row_threads = _qw_sigma_row_openmp_threads(
        int(kloc1) - int(kloc0), len(k_grid_c), weighted_S_t_c.shape[1]
    )
    if geom is not None and hasattr(_qw_radial_kernels, "qw_sigma_kernel_geometry"):
        Sigma_t, Sigma0 = _qw_radial_kernels.qw_sigma_kernel_geometry(
            weighted_S_t_c,
            weighted_S_beta_c,
            *qargs[1:],
            *geom,
            kloc0,
            bool(use_diag),
            dyn_avg_c,
            sta_avg_c,
            bool(skip_static),
            row_threads,
        )
        return Sigma_t, Sigma0, False
    if not hasattr(_qw_radial_kernels, "qw_sigma_kernel"):
        return None
    if precomp.has_singularity and np.unique(k_grid_c).size != k_grid_c.size:
        return None
    Sigma_t, Sigma0 = _qw_radial_kernels.qw_sigma_kernel(
        k_grid_c,
        weighted_S_t_c,
        weighted_S_beta_c,
        *qargs,
        int(kloc0),
        int(kloc1),
        bool(use_diag),
        dyn_avg_c,
        sta_avg_c,
        bool(skip_static),
        row_threads,
    )
    return Sigma_t, Sigma0, False


def _pi_pair_geometry(source_k, q_grid, precomp, qloc0, qloc1):
    if (
        not _cython_radial_available()
        or not hasattr(_radial_kernels, "build_pi_pair_geometry")
        or _PI_PAIR_GEOMETRY_CACHE_MAX <= 0
    ):
        return None
    source_k = np.ascontiguousarray(source_k, dtype=np.float64)
    q_grid = np.ascontiguousarray(q_grid, dtype=np.float64)
    primitive_k, primitive_k2, primitive_k3, *_ = _gprecomp_cython_args(precomp)
    if (
        source_k.ndim != 1
        or q_grid.ndim != 1
        or primitive_k.ndim != 1
        or (source_k.size > 1 and not np.all(source_k[1:] >= source_k[:-1]))
    ):
        return None

    nloc = int(qloc1) - int(qloc0)
    geom_bytes = _pair_geometry_bytes(nloc, len(source_k))
    limit_bytes = int(_PI_PAIR_GEOMETRY_CACHE_MB * 1024.0 * 1024.0)
    if limit_bytes > 0 and geom_bytes > limit_bytes:
        return None

    key = (
        mpi.rank,
        int(qloc0),
        int(qloc1),
        _array_value_key(source_k),
        _array_value_key(q_grid),
        _array_value_key(primitive_k),
    )
    geom = _PI_PAIR_GEOMETRY_CACHES.get(key)
    if geom is not None:
        _PI_PAIR_GEOMETRY_CACHES.move_to_end(key)
        return geom

    geom = _radial_kernels.build_pi_pair_geometry(
        source_k,
        q_grid,
        primitive_k,
        primitive_k2,
        primitive_k3,
        int(qloc0),
        int(qloc1),
        _pair_geometry_openmp_threads(nloc, len(source_k)),
    )
    _PI_PAIR_GEOMETRY_CACHES[key] = geom
    _PI_PAIR_GEOMETRY_CACHES.move_to_end(key)
    _trim_pair_geometry_cache(
        _PI_PAIR_GEOMETRY_CACHES,
        _PI_PAIR_GEOMETRY_CACHE_MAX,
        _PI_PAIR_GEOMETRY_CACHE_MB,
    )
    return geom


@_lazy_parallel_call
def get_sigma(_, ctx, k_grid, k_wts, S, precomp, S_has_k=True,
              return_scaled=True, tol=1e-12):
    r"""Apply the radial self-energy convolution over MPI k blocks.

    ``S`` is either an unscaled source ``S(k, iw)`` or the scaled source
    ``k*S(k, iw)`` depending on ``S_has_k``.  With ``return_scaled=True`` the
    returned array is ``k*Sigma(k, iw)``; this is the variable used by the gap
    equation to avoid a removable ``1/k`` singularity at the origin.  With
    ``return_scaled=False`` the physical ``Sigma(k, iw)`` is returned.

    Notes
    -----
    For nonzero target momentum the unscaled physical form represented by this
    routine is

    .. math::

        \Sigma(k) =
        -\frac{1}{(2\pi)^2 k}
        \int dp\, p\, S(p)
        \int_{|k-p|}^{k+p} dq\, q W(q),

    with the Matsubara/tau transforms and static endpoint contribution applied
    exactly as in the code.  The ``k=0`` row uses the analytic limit of the
    angular interval instead of dividing by zero.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    if getattr(ctx, "symm", False):
        S = symm_sigma(ctx, S)
    S_t = ctx.w2t_f(S)
    S_beta = -ctx.w2beta_f(S)
    nk, nt = S_t.shape
    if precomp.nt != nt:
        raise ValueError("tau-grid mismatch: precomp %d vs S_t %d" % (precomp.nt, nt))

    kloc0, kloc1 = _task_location(nk)
    weighted_S_t, weighted_S_beta = _weighted_sources(
        k_grid, k_wts, S_t, S_beta, S_has_k
    )
    backend = _choose_sigma_backend(precomp, k_grid, kloc0, kloc1, use_diag=False)
    full_output = False
    use_python_fallback = backend == _SIGMA_BACKEND_PYTHON
    if backend in (_SIGMA_BACKEND_LEGACY_TRIANGULAR, _SIGMA_BACKEND_LEGACY_ROW):
        empty_dyn, empty_sta = _empty_diag_arrays()
        k_grid_c = np.ascontiguousarray(k_grid, dtype=np.float64)
        weighted_S_t_c = np.ascontiguousarray(weighted_S_t, dtype=np.complex128)
        weighted_S_beta_c = np.ascontiguousarray(
            weighted_S_beta, dtype=np.complex128
        )
        wargs = _wprecomp_cython_args(precomp)
        if backend == _SIGMA_BACKEND_LEGACY_TRIANGULAR:
            full_geom = _sigma_full_pair_geometry(k_grid_c, precomp)
            if full_geom is None:
                raise RuntimeError("selected legacy triangular backend is unavailable.")
            omp_threads = _sigma_triangular_openmp_threads(len(k_grid_c), nt)
            if (
                omp_threads > 1
                and hasattr(_radial_kernels, "sigma_kernel_geometry_triangular_omp")
            ):
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel_geometry_triangular_omp(
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs[3:],
                    *full_geom,
                    mpi.rank,
                    mpi.pool.size,
                    _SIGMA_TRIANGULAR_TILE,
                    False,
                    empty_dyn,
                    empty_sta,
                    omp_threads,
                )
            else:
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel_geometry_triangular(
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs[3:],
                    *full_geom,
                    mpi.rank,
                    mpi.pool.size,
                    _SIGMA_TRIANGULAR_TILE,
                    False,
                    empty_dyn,
                    empty_sta,
                )
            Sigma_t = _reduce_sum_rank0_array(Sigma_t)
            Sigma0 = _reduce_sum_rank0_array(Sigma0)
            full_output = True
        else:
            geom = _sigma_pair_geometry(k_grid_c, precomp, kloc0, kloc1)
            row_threads = _sigma_row_openmp_threads(kloc1 - kloc0, nk, nt)
            if geom is not None and hasattr(_radial_kernels, "sigma_kernel_geometry"):
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel_geometry(
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs[3:],
                    *geom,
                    kloc0,
                    False,
                    empty_dyn,
                    empty_sta,
                    row_threads,
                )
            else:
                if not hasattr(_radial_kernels, "sigma_kernel"):
                    raise RuntimeError("selected legacy row backend is unavailable.")
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel(
                    k_grid_c,
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs,
                    kloc0,
                    kloc1,
                    False,
                    empty_dyn,
                    empty_sta,
                    row_threads,
                )
    elif backend in (_SIGMA_BACKEND_QW_TRIANGULAR, _SIGMA_BACKEND_QW_ROW):
        qw_result = _sigma_qw_cython(
            k_grid, weighted_S_t, weighted_S_beta, precomp, kloc0, kloc1,
            use_diag=False, backend=backend,
        )
        if qw_result is None:
            raise RuntimeError("selected qW sigma backend is unavailable.")
        else:
            Sigma_t, Sigma0, full_output = qw_result

    if use_python_fallback:
        Sigma_t = np.zeros((kloc1 - kloc0, nt), dtype=complex)
        Sigma0 = np.zeros(kloc1 - kloc0, dtype=complex)
        for local_j, j in enumerate(range(kloc0, kloc1)):
            kp = k_grid[j]
            dI_dyn = precomp.integral_block_dyn(k_grid, kp)
            Sigma_t[local_j] = np.einsum("kt,kt->t", weighted_S_t, dI_dyn,
                                         optimize=True)
            dI_sta = precomp.integral_block_sta(k_grid, kp)
            Sigma0[local_j] = np.dot(weighted_S_beta, dI_sta)
    if full_output and _worker_result_discarded():
        return None
    Sigma_w_tilde = ctx.t2w_f(Sigma_t)
    Sigma_w_tilde += Sigma0[:, None]
    Sigma_w_tilde *= -1.0 / ((2.0 * np.pi) ** 2)

    if full_output:
        if return_scaled:
            return Sigma_w_tilde

        Sigma_w = np.zeros_like(Sigma_w_tilde)
        idx_0 = np.abs(k_grid) < tol
        idx_n0 = ~idx_0
        Sigma_w[idx_n0] = Sigma_w_tilde[idx_n0] / k_grid[idx_n0, None]
        if np.any(idx_0):
            Sigma_w[idx_0] = _origin_sigma_row(
                ctx, k_grid, k_wts, S_t, S_beta, precomp, S_has_k)
        return Sigma_w

    if return_scaled:
        return _gather_rank0(Sigma_w_tilde)

    k_cur = k_grid[kloc0:kloc1]
    Sigma_w = np.zeros_like(Sigma_w_tilde)
    idx_0 = np.abs(k_cur) < tol
    idx_n0 = ~idx_0
    Sigma_w[idx_n0] = Sigma_w_tilde[idx_n0] / k_cur[idx_n0, None]
    if np.any(idx_0):
        Sigma_w[idx_0] = _origin_sigma_row(
            ctx, k_grid, k_wts, S_t, S_beta, precomp, S_has_k)
    return _gather_rank0(Sigma_w)


def diag_cell_bounds(k_grid, k_wts, k_upper):
    """Build width-preserving cells centered on each radial k point."""
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    k_upper = float(k_upper)
    if k_grid.ndim != 1 or k_wts.shape != k_grid.shape:
        raise ValueError("k_grid and k_wts must be one-dimensional arrays.")
    if not np.all(k_wts > 0.0):
        raise ValueError("k_wts must be positive.")
    if k_upper <= 0.0:
        raise ValueError("k_upper must be positive.")
    width_bad = np.flatnonzero(k_wts > k_upper)
    if width_bad.size:
        raise ValueError(
            "diagonal cell width exceeds k range at index %d"
            % int(width_bad[0])
        )
    lo = k_grid - 0.5 * k_wts
    hi = k_grid + 0.5 * k_wts
    under = lo < 0.0
    if np.any(under):
        hi[under] -= lo[under]
        lo[under] = 0.0
    over = hi > k_upper
    if np.any(over):
        lo[over] -= hi[over] - k_upper
        hi[over] = k_upper
    invalid = (
        (lo < -1.0e-14)
        | (hi > k_upper * (1.0 + 1.0e-14))
        | ~(hi > lo)
    )
    invalid_idx = np.flatnonzero(invalid)
    if invalid_idx.size:
        raise ValueError("invalid diagonal cell at index %d" % int(invalid_idx[0]))
    bounds = np.empty((len(k_grid), 2), dtype=float)
    bounds[:, 0] = np.maximum(lo, 0.0)
    bounds[:, 1] = np.minimum(hi, k_upper)
    return bounds


def get_diag_cell_geometry(k_grid, k_wts, k_upper, ncell):
    """Precompute geometry shared by diagonal cell averages on one k grid."""
    ncell = int(ncell)
    if ncell < 1:
        raise ValueError("ncell must be at least 1.")
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    k_upper = float(k_upper)
    bounds = diag_cell_bounds(k_grid, k_wts, k_upper)
    xw = leggauss(ncell)
    return DiagCellGeometry(
        k_grid=k_grid.copy(),
        k_wts=k_wts.copy(),
        k_upper=k_upper,
        bounds=bounds,
        xw=xw,
        ncell=ncell,
    )


def _resolve_diag_cell_geometry(k_grid, k_wts, k_upper, ncell, geometry):
    ncell = int(ncell)
    if ncell < 1:
        raise ValueError("ncell must be at least 1.")
    k_grid = np.asarray(k_grid, dtype=float)
    if geometry is None:
        geometry = get_diag_cell_geometry(k_grid, k_wts, k_upper, ncell)
    if int(geometry.ncell) != ncell:
        raise ValueError("diagonal cell geometry ncell mismatch.")
    k_wts = np.asarray(k_wts, dtype=float)
    if (
        not np.array_equal(np.asarray(geometry.k_grid, dtype=float), k_grid)
        or not np.array_equal(np.asarray(geometry.k_wts, dtype=float), k_wts)
        or float(geometry.k_upper) != float(k_upper)
    ):
        raise ValueError("diagonal cell geometry is inconsistent with the k grid.")
    bounds = np.asarray(geometry.bounds, dtype=float)
    if bounds.shape != (len(k_grid), 2):
        raise ValueError("diagonal cell geometry shape is inconsistent with k_grid.")
    return ncell, k_grid, k_wts, bounds, geometry.xw


def _integrate_cell_average(func, kp, lo, hi, ncell, xw=None):
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


def _analytic_qw_diag_cells(precomp, k_grid, bounds, kloc0, kloc1):
    cell_averages = getattr(precomp, "cell_averages_analytic", None)
    if getattr(precomp, "interp_mode", None) != "q_singular" or cell_averages is None:
        raise TypeError(
            "analytic diagonal-cell averages currently require q_singular W."
        )
    local_bounds = bounds[int(kloc0):int(kloc1)]
    dyn, sta = cell_averages(
        k_grid[int(kloc0):int(kloc1)],
        local_bounds[:, 0],
        local_bounds[:, 1],
    )
    dyn = np.ascontiguousarray(dyn, dtype=precomp.dyn_dtype)
    sta = np.ascontiguousarray(sta, dtype=precomp.sta_dtype)
    return dyn, sta


def _analytic_qw_diag_cells_dyn(precomp, k_grid, bounds, kloc0, kloc1):
    cell_average_dyn = getattr(precomp, "cell_average_dyn_analytic", None)
    if getattr(precomp, "interp_mode", None) != "q_singular" or cell_average_dyn is None:
        raise TypeError(
            "analytic diagonal-cell averages currently require q_singular W."
        )
    local_bounds = bounds[int(kloc0):int(kloc1)]
    dyn = cell_average_dyn(
        k_grid[int(kloc0):int(kloc1)],
        local_bounds[:, 0],
        local_bounds[:, 1],
    )
    return np.ascontiguousarray(dyn, dtype=precomp.dyn_dtype)


@_lazy_parallel_call
def get_diag_cells(_, precomp, k_grid, k_wts, k_upper, ncell, geometry=None,
                   cell_mode="auto"):
    """Precompute diagonal radial-kernel cell averages over MPI."""
    cell_mode = _resolve_diag_average_mode(precomp, cell_mode)
    ncell, k_grid, _, bounds, xw = _resolve_diag_cell_geometry(
        k_grid, k_wts, k_upper, ncell, geometry)
    kloc0, kloc1 = _task_location(len(k_grid))
    if cell_mode == "analytic":
        dyn, sta = _analytic_qw_diag_cells(precomp, k_grid, bounds, kloc0, kloc1)
        return _gather_rank0(dyn), _gather_rank0(sta), bounds
    if _cython_radial_available() and precomp.uses_legacy_primitives:
        x, w = xw
        dyn, sta = _radial_kernels.diag_cells_kernel(
            np.ascontiguousarray(k_grid, dtype=np.float64),
            np.ascontiguousarray(bounds, dtype=np.float64),
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64),
            *_wprecomp_cython_args(precomp),
            kloc0,
            kloc1,
        )
        return _gather_rank0(dyn), _gather_rank0(sta), bounds
    if _qw_cython_available() and getattr(precomp, "interp_mode", None) == "q_singular":
        result = _diag_cells_qw_cython(precomp, k_grid, bounds, xw, kloc0, kloc1)
        if result is not None:
            dyn, sta = result
            return _gather_rank0(dyn), _gather_rank0(sta), bounds

    dyn = np.empty((kloc1 - kloc0, precomp.nt), dtype=precomp.dyn_dtype)
    sta = np.empty(kloc1 - kloc0, dtype=precomp.sta_dtype)
    cell_averages = getattr(precomp, "cell_averages", None)
    for local_ik, ik in enumerate(range(kloc0, kloc1)):
        kp = k_grid[ik]
        lo, hi = bounds[ik]
        if cell_averages is not None:
            dyn[local_ik], sta[local_ik] = cell_averages(
                kp, lo, hi, ncell, xw=xw)
        else:
            dyn[local_ik] = precomp.cell_average_dyn(kp, lo, hi, ncell, xw=xw)
            sta[local_ik] = precomp.cell_average_sta(kp, lo, hi, ncell, xw=xw)
    return _gather_rank0(dyn), _gather_rank0(sta), bounds


@_lazy_parallel_call
def get_diag_cells_dyn(_, precomp, k_grid, k_wts, k_upper, ncell, geometry=None,
                       cell_mode="auto"):
    """Precompute only dynamic diagonal radial-kernel cell averages over MPI."""
    cell_mode = _resolve_diag_average_mode(precomp, cell_mode)
    ncell, k_grid, _, bounds, xw = _resolve_diag_cell_geometry(
        k_grid, k_wts, k_upper, ncell, geometry)
    kloc0, kloc1 = _task_location(len(k_grid))
    if cell_mode == "analytic":
        dyn = _analytic_qw_diag_cells_dyn(precomp, k_grid, bounds, kloc0, kloc1)
        return _gather_rank0(dyn), bounds
    if _cython_radial_available() and precomp.uses_legacy_primitives:
        dyn = _diag_cells_dyn_legacy_cython(
            precomp, k_grid, bounds, xw, kloc0, kloc1)
        if dyn is not None:
            return _gather_rank0(dyn), bounds
    if _qw_cython_available() and getattr(precomp, "interp_mode", None) == "q_singular":
        dyn = _diag_cells_dyn_qw_cython(precomp, k_grid, bounds, xw, kloc0, kloc1)
        if dyn is not None:
            return _gather_rank0(dyn), bounds

    dyn = np.empty((kloc1 - kloc0, precomp.nt), dtype=precomp.dyn_dtype)
    for local_ik, ik in enumerate(range(kloc0, kloc1)):
        kp = k_grid[ik]
        lo, hi = bounds[ik]
        dyn[local_ik] = precomp.cell_average_dyn(kp, lo, hi, ncell, xw=xw)
    return _gather_rank0(dyn), bounds


def _origin_cell_mask(k_grid, bounds, tol):
    # ``bounds`` are source-cell geometry; the k->0 limit is a target property.
    edge_tol = max(float(tol), 10.0 * np.finfo(float).eps)
    return np.abs(k_grid) <= edge_tol


@_lazy_parallel_call
def get_sigma_diag(_, ctx, k_grid, k_wts, S, precomp, diag_cells,
                   S_has_k=True, return_scaled=True, tol=1e-12):
    """Apply ``get_sigma`` with averaged diagonal radial blocks.

    ``diag_cells`` replaces the singular self-cell angular block by a
    width-preserving cell average.  This changes only the diagonal radial
    quadrature treatment, not the Matsubara convention or the source scaling.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    if getattr(ctx, "symm", False):
        S = symm_sigma(ctx, S)
    dyn_avg, sta_avg, bounds = diag_cells
    S_t = ctx.w2t_f(S)
    S_beta = -ctx.w2beta_f(S)
    nk, nt = S_t.shape
    if precomp.nt != nt:
        raise ValueError("tau-grid mismatch.")
    kloc0, kloc1 = _task_location(nk)
    weighted_S_t, weighted_S_beta = _weighted_sources(
        k_grid, k_wts, S_t, S_beta, S_has_k
    )
    backend = _choose_sigma_backend(precomp, k_grid, kloc0, kloc1, use_diag=True)
    full_output = False
    use_python_fallback = backend == _SIGMA_BACKEND_PYTHON
    if backend in (_SIGMA_BACKEND_LEGACY_TRIANGULAR, _SIGMA_BACKEND_LEGACY_ROW):
        k_grid_c = np.ascontiguousarray(k_grid, dtype=np.float64)
        weighted_S_t_c = np.ascontiguousarray(weighted_S_t, dtype=np.complex128)
        weighted_S_beta_c = np.ascontiguousarray(
            weighted_S_beta, dtype=np.complex128
        )
        dyn_avg_c = _writable_complex128_array(dyn_avg)
        sta_avg_c = _writable_complex128_array(sta_avg)
        wargs = _wprecomp_cython_args(precomp)
        if backend == _SIGMA_BACKEND_LEGACY_TRIANGULAR:
            full_geom = _sigma_full_pair_geometry(k_grid_c, precomp)
            if full_geom is None:
                raise RuntimeError("selected legacy triangular backend is unavailable.")
            omp_threads = _sigma_triangular_openmp_threads(len(k_grid_c), nt)
            if (
                omp_threads > 1
                and hasattr(_radial_kernels, "sigma_kernel_geometry_triangular_omp")
            ):
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel_geometry_triangular_omp(
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs[3:],
                    *full_geom,
                    mpi.rank,
                    mpi.pool.size,
                    _SIGMA_TRIANGULAR_TILE,
                    True,
                    dyn_avg_c,
                    sta_avg_c,
                    omp_threads,
                )
            else:
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel_geometry_triangular(
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs[3:],
                    *full_geom,
                    mpi.rank,
                    mpi.pool.size,
                    _SIGMA_TRIANGULAR_TILE,
                    True,
                    dyn_avg_c,
                    sta_avg_c,
                )
            Sigma_t = _reduce_sum_rank0_array(Sigma_t)
            Sigma0 = _reduce_sum_rank0_array(Sigma0)
            full_output = True
        else:
            geom = _sigma_pair_geometry(k_grid_c, precomp, kloc0, kloc1)
            row_threads = _sigma_row_openmp_threads(kloc1 - kloc0, nk, nt)
            if geom is not None and hasattr(_radial_kernels, "sigma_kernel_geometry"):
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel_geometry(
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs[3:],
                    *geom,
                    kloc0,
                    True,
                    dyn_avg_c,
                    sta_avg_c,
                    row_threads,
                )
            else:
                if not hasattr(_radial_kernels, "sigma_kernel"):
                    raise RuntimeError("selected legacy row backend is unavailable.")
                Sigma_t, Sigma0 = _radial_kernels.sigma_kernel(
                    k_grid_c,
                    weighted_S_t_c,
                    weighted_S_beta_c,
                    *wargs,
                    kloc0,
                    kloc1,
                    True,
                    dyn_avg_c,
                    sta_avg_c,
                    row_threads,
                )
    elif backend in (_SIGMA_BACKEND_QW_TRIANGULAR, _SIGMA_BACKEND_QW_ROW):
        static_row_start = 0 if backend == _SIGMA_BACKEND_QW_TRIANGULAR else kloc0
        static_row_stop = nk if backend == _SIGMA_BACKEND_QW_TRIANGULAR else kloc1
        skip_static = _should_use_static_blocks_with_diag_cache(
            precomp, k_grid, sta_avg, static_row_start, static_row_stop)
        qw_result = _sigma_qw_cython(
            k_grid, weighted_S_t, weighted_S_beta, precomp, kloc0, kloc1,
            use_diag=True, dyn_avg=dyn_avg, sta_avg=sta_avg, backend=backend,
            skip_static=skip_static,
        )
        if qw_result is None:
            raise RuntimeError("selected qW sigma backend is unavailable.")
        else:
            Sigma_t, Sigma0, full_output = qw_result
        if skip_static and not (full_output and _worker_result_discarded()):
            row_start = 0 if full_output else kloc0
            row_stop = nk if full_output else kloc1
            sta_blocks = _static_blocks_with_diag_cached(
                precomp, k_grid, sta_avg, row_start, row_stop)
            if sta_blocks is None:
                raise RuntimeError("qW static block cache unexpectedly unavailable.")
            Sigma0 = np.einsum("k,jk->j", weighted_S_beta, sta_blocks,
                               optimize=True)

    if use_python_fallback:
        Sigma_t = np.zeros((kloc1 - kloc0, nt), dtype=complex)
        Sigma0 = np.zeros(kloc1 - kloc0, dtype=complex)
        integral_blocks_with_diag = getattr(precomp, "integral_blocks_with_diag", None)
        static_blocks_with_diag = getattr(precomp, "static_blocks_with_diag", None)
        sta_blocks = None
        if static_blocks_with_diag is not None and _QW_STATIC_BLOCK_CACHE_MB > 0.0:
            sta_cache_bytes = (
                (kloc1 - kloc0) * nk * np.dtype(precomp.sta_dtype).itemsize
            )
            sta_cache_limit = int(_QW_STATIC_BLOCK_CACHE_MB * 1024.0 * 1024.0)
        else:
            sta_cache_bytes = sta_cache_limit = 0
        if static_blocks_with_diag is not None and (
            sta_cache_limit > 0 and sta_cache_bytes <= sta_cache_limit
        ):
            sta_blocks = static_blocks_with_diag(
                k_grid, sta_avg, row_start=kloc0, row_stop=kloc1)
        for local_j, j in enumerate(range(kloc0, kloc1)):
            kp = k_grid[j]
            if sta_blocks is not None:
                dI_dyn = precomp.integral_block_dyn_with_diag(
                    k_grid, kp, j, dyn_avg[j])
                dI_sta = sta_blocks[local_j]
            elif integral_blocks_with_diag is not None:
                dI_dyn, dI_sta = integral_blocks_with_diag(
                    k_grid, kp, j, dyn_avg[j], sta_avg[j])
            else:
                dI_dyn = precomp.integral_block_dyn_with_diag(
                    k_grid, kp, j, dyn_avg[j])
                dI_sta = precomp.integral_block_sta_with_diag(
                    k_grid, kp, j, sta_avg[j])
            Sigma_t[local_j] = np.einsum("kt,kt->t", weighted_S_t, dI_dyn,
                                         optimize=True)
            Sigma0[local_j] = np.dot(weighted_S_beta, dI_sta)
    if full_output and _worker_result_discarded():
        return None
    Sigma_w_tilde = ctx.t2w_f(Sigma_t)
    Sigma_w_tilde += Sigma0[:, None]
    Sigma_w_tilde *= -1.0 / ((2.0 * np.pi) ** 2)
    if full_output:
        if return_scaled:
            return Sigma_w_tilde

        Sigma_w = np.zeros_like(Sigma_w_tilde)
        idx_0 = _origin_cell_mask(k_grid, bounds, tol)
        idx_n0 = ~idx_0
        Sigma_w[idx_n0] = Sigma_w_tilde[idx_n0] / k_grid[idx_n0, None]
        if np.any(idx_0):
            Sigma_w[idx_0] = _origin_sigma_row(
                ctx, k_grid, k_wts, S_t, S_beta, precomp, S_has_k)
        return Sigma_w

    if return_scaled:
        return _gather_rank0(Sigma_w_tilde)

    k_cur = k_grid[kloc0:kloc1]
    Sigma_w = np.zeros_like(Sigma_w_tilde)
    idx_0 = _origin_cell_mask(k_cur, bounds[kloc0:kloc1], tol)
    idx_n0 = ~idx_0
    Sigma_w[idx_n0] = Sigma_w_tilde[idx_n0] / k_cur[idx_n0, None]
    if np.any(idx_0):
        Sigma_w[idx_0] = _origin_sigma_row(
            ctx, k_grid, k_wts, S_t, S_beta, precomp, S_has_k)
    return _gather_rank0(Sigma_w)


def _cache_memory_bytes(nk_local, nk, nt, dyn_dtype, sta_dtype):
    return int(
        nk_local * nk * nt * np.dtype(dyn_dtype).itemsize
        + nk_local * nk * np.dtype(sta_dtype).itemsize
    )


@_lazy_parallel_call
def prepare_cache(_, key, precomp, k_grid, diag_cells=None,
                  mode="auto", max_memory=memory.DEFAULT_MAX_MEMORY):
    """Prepare rank-local radial sigma blocks for repeated gap matvecs."""
    if mode not in ("auto", "off"):
        raise ValueError("cache mode must be 'auto' or 'off'.")
    key = str(key)
    _SIGMA_KERNEL_CACHES.pop(key, None)
    if mode == "off":
        return {"enabled": False, "reason": "off", "key": key,
                "max_local_bytes": 0, "total_bytes": 0,
                "required_mb": 0.0, "total_mb": 0.0,
                "effective_limit_mb": 0.0}
    if float(max_memory) < 0.0:
        raise ValueError("max_memory must be non-negative.")
    if precomp.requires_diag_cells and diag_cells is None:
        raise FloatingPointError(
            "q_singular W has a Coulomb diagonal singularity; build diag_cells "
            "before preparing the radial sigma cache."
        )
    k_grid = np.asarray(k_grid, dtype=float)
    nk = len(k_grid)
    kloc0, kloc1 = _task_location(nk)
    use_cython = _cython_radial_available() and precomp.uses_legacy_primitives
    cache_dyn_dtype = np.complex128 if use_cython else precomp.dyn_dtype
    cache_sta_dtype = np.complex128 if use_cython else precomp.sta_dtype
    local_bytes = _cache_memory_bytes(
        kloc1 - kloc0, nk, precomp.nt, cache_dyn_dtype, cache_sta_dtype)
    max_local_bytes = int(mpi.comm.allreduce(local_bytes, op=mpi.MPI.MAX))
    total_bytes = int(mpi.comm.allreduce(local_bytes, op=mpi.MPI.SUM))
    required_mb = max_local_bytes / memory.MB
    total_mb = total_bytes / memory.MB
    memory_info = memory.available_memory_info(max_memory, comm=mpi.comm)
    rank_effective_mb = float(memory_info["effective_limit_mb"])
    effective_mb = float(mpi.comm.allreduce(rank_effective_mb, op=mpi.MPI.MIN))
    memory_info["rank_effective_limit_mb"] = rank_effective_mb
    memory_info["effective_limit_mb"] = effective_mb
    if effective_mb < rank_effective_mb:
        memory_info["limiting_source"] = "mpi global minimum"
    if required_mb > effective_mb:
        memory_info.update(required_mb=required_mb, total_mb=total_mb)
        return {"enabled": False, "reason": "memory", "key": key,
                "max_local_bytes": max_local_bytes, "total_bytes": total_bytes,
                **memory_info}

    dyn_avg = sta_avg = None
    if diag_cells is not None:
        dyn_avg, sta_avg, _ = diag_cells
    if use_cython:
        if diag_cells is None:
            empty_dyn, empty_sta = _empty_diag_arrays()
            dyn_avg_c, sta_avg_c, use_diag = empty_dyn, empty_sta, False
        else:
            dyn_avg_c = _writable_complex128_array(dyn_avg)
            sta_avg_c = _writable_complex128_array(sta_avg)
            use_diag = True
        dyn, sta = _radial_kernels.prepare_cache_kernel(
            np.ascontiguousarray(k_grid, dtype=np.float64),
            *_wprecomp_cython_args(precomp),
            kloc0,
            kloc1,
            use_diag,
            dyn_avg_c,
            sta_avg_c,
            _sigma_row_openmp_threads(kloc1 - kloc0, nk, precomp.nt),
        )
    else:
        dyn = np.empty((kloc1 - kloc0, nk, precomp.nt), dtype=precomp.dyn_dtype)
        sta = np.empty((kloc1 - kloc0, nk), dtype=precomp.sta_dtype)
        integral_blocks = getattr(precomp, "integral_blocks", None)
        integral_blocks_with_diag = getattr(precomp, "integral_blocks_with_diag", None)
        for local_j, j in enumerate(range(kloc0, kloc1)):
            kp = k_grid[j]
            if diag_cells is not None:
                if integral_blocks_with_diag is not None:
                    dyn_block, sta_block = integral_blocks_with_diag(
                        k_grid, kp, j, dyn_avg[j], sta_avg[j])
                else:
                    dyn_block = precomp.integral_block_dyn_with_diag(
                        k_grid, kp, j, dyn_avg[j])
                    sta_block = precomp.integral_block_sta_with_diag(
                        k_grid, kp, j, sta_avg[j])
            else:
                if integral_blocks is not None:
                    dyn_block, sta_block = integral_blocks(k_grid, kp)
                else:
                    dyn_block = precomp.integral_block_dyn(k_grid, kp)
                    sta_block = precomp.integral_block_sta(k_grid, kp)
            dyn[local_j] = dyn_block
            sta[local_j] = sta_block
    _SIGMA_KERNEL_CACHES[key] = {
        "dyn": dyn, "sta": sta, "nk": nk, "nt": precomp.nt,
        "kloc0": kloc0, "kloc1": kloc1,
    }
    memory_info.update(required_mb=required_mb, total_mb=total_mb)
    return {"enabled": True, "reason": "ok", "key": key,
            "max_local_bytes": max_local_bytes, "total_bytes": total_bytes,
            **memory_info}


@_lazy_parallel_call
def clear_cache(_, key=None):
    """Clear one cached radial kernel, or all cached kernels."""
    if key is None:
        _SIGMA_KERNEL_CACHES.clear()
        _SIGMA_PAIR_GEOMETRY_CACHES.clear()
        _QW_SIGMA_PAIR_GEOMETRY_CACHES.clear()
        _QW_DIAG_CELL_GEOMETRY_CACHES.clear()
        _PI_PAIR_GEOMETRY_CACHES.clear()
        clear_qw_static_block_caches()
    else:
        _SIGMA_KERNEL_CACHES.pop(str(key), None)
    return len(_SIGMA_KERNEL_CACHES)


@_lazy_parallel_call
def get_sigma_cache(_, ctx, k_grid, k_wts, S, cache_key,
                    S_has_k=True, return_scaled=True):
    """Apply a cached rank-local radial sigma kernel."""
    if not return_scaled:
        raise ValueError("cached radial sigma kernel only supports return_scaled=True.")
    cache_key = str(cache_key)
    if cache_key not in _SIGMA_KERNEL_CACHES:
        raise KeyError("radial sigma kernel cache %r is not prepared." % cache_key)
    cache = _SIGMA_KERNEL_CACHES[cache_key]
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    if cache["nk"] != len(k_grid):
        raise ValueError("cached radial sigma kernel has incompatible k-grid size.")
    if getattr(ctx, "symm", False):
        S = symm_sigma(ctx, S)
    S_t = ctx.w2t_f(S)
    S_beta = -ctx.w2beta_f(S)
    weighted_S_t, weighted_S_beta = _weighted_sources(
        k_grid, k_wts, S_t, S_beta, S_has_k
    )
    Sigma_t = np.einsum("kt,jkt->jt", weighted_S_t, cache["dyn"], optimize=True)
    Sigma0 = np.einsum("k,jk->j", weighted_S_beta, cache["sta"], optimize=True)
    Sigma_w = ctx.t2w_f(Sigma_t)
    Sigma_w += Sigma0[:, None]
    Sigma_w *= -1.0 / ((2.0 * np.pi) ** 2)
    return _gather_rank0(Sigma_w)


@_lazy_parallel_call
def get_precond(_, ctx, k_grid, k_wts, G_pair_sq, precomp, diag_cells=None):
    """Build the Jacobi diagonal for the real-even-frequency gap subspace."""
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    G_pair_sq = np.asarray(G_pair_sq)
    nk, nw = G_pair_sq.shape
    if k_grid.shape != (nk,) or k_wts.shape != (nk,):
        raise ValueError("k_grid, k_wts, and G_pair_sq shapes are inconsistent.")
    check_w_pair(ctx)

    pair = np.arange(nw - 1, -1, -1)
    B = np.dot(ctx.F_t_f, ctx.F_w_f_inv)
    C = np.dot(ctx.F_w_f, ctx.F_t_f_inv)
    beta_row = ctx.w2beta_row_f()
    kloc0, kloc1 = _task_location(nk)
    k_cur = k_grid[kloc0:kloc1]
    if diag_cells is None:
        if precomp.has_singularity:
            raise FloatingPointError(
                "q_singular W has a Coulomb diagonal singularity; pass diag_cells "
                "to build the gap preconditioner."
            )
        q_zero = np.zeros_like(k_cur)
        I_dyn = precomp._primitive_dyn(2.0 * k_cur) - precomp._primitive_dyn(q_zero)
        I_sta = precomp._primitive_sta(2.0 * k_cur) - precomp._primitive_sta(q_zero)
    else:
        I_dyn, I_sta, _ = diag_cells
        I_dyn = I_dyn[kloc0:kloc1]
        I_sta = I_sta[kloc0:kloc1]
    G_cur = G_pair_sq[kloc0:kloc1]
    dyn_same = np.einsum("mt,kt,tm->km", C, I_dyn, B, optimize=True)
    dyn_cross = np.einsum("mt,kt,tm->km", C, I_dyn, B[:, pair], optimize=True)
    pref = (k_wts[kloc0:kloc1] / ((2.0 * np.pi) ** 2))[:, None]
    diag_same = pref * G_cur * (dyn_same - I_sta[:, None] * beta_row[None, :])
    diag_cross = pref * G_cur[:, pair] * (
        dyn_cross - I_sta[:, None] * beta_row[pair][None, :]
    )
    diag = 0.5 * (
        diag_same + diag_cross + diag_same[:, pair].conj()
        + diag_cross[:, pair].conj()
    ).real
    diag = _gather_rank0(diag)
    if not np.all(np.isfinite(diag)):
        raise FloatingPointError("non-finite entries in gap preconditioner.")
    if max_abs(diag - diag[:, pair]) > 1.0e-8 * max(max_abs(diag), 1.0):
        raise FloatingPointError("gap preconditioner is not pair symmetric.")
    return diag


def f_occ_k(k, mu, beta, a):
    r"""Return Fermi occupation for the radial dispersion ``a*k**2 - mu``.

    Parameters are in atomic units: ``k`` in Bohr^-1, ``mu`` in Hartree,
    ``beta`` in Hartree^-1, and ``a`` in Hartree Bohr^2.
    """
    a = float(a)
    beta = float(beta)
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    eps = a * np.asarray(k, dtype=float) ** 2 - float(mu)
    return model.f_occ(eps, beta)


def sigma_x_t0(k, k_F, eps_inf, tol=1e-12):
    r"""Static T=0 one-spin Coulomb exchange for radial momentum ``k``.

    The returned value is the exchange self-energy in Hartree for a 3D
    parabolic band with screened Coulomb interaction ``4*pi/(eps_inf*q^2)``.
    The closed form is evaluated with small- and large-``k/k_F`` branches to
    avoid cancellation near the logarithmic limits.
    """
    k_F = float(k_F)
    eps_inf = float(eps_inf)
    if k_F < 0.0:
        raise ValueError("k_F must be non-negative.")
    if eps_inf <= 0.0:
        raise ValueError("eps_inf must be positive.")
    k_arr = np.asarray(k, dtype=float)
    scalar = k_arr.ndim == 0
    kk = np.ravel(k_arr)
    out = np.empty_like(kk, dtype=float)
    if k_F == 0.0:
        out.fill(0.0)
        out = out.reshape(k_arr.shape)
        return float(out) if scalar else out
    scale = max(k_F, 1.0)
    if np.any(kk < -tol * scale):
        raise ValueError("radial momenta k must be non-negative.")
    kk = np.abs(kk)
    small = kk <= tol * scale
    out[small] = -2.0 * k_F / (np.pi * eps_inf)
    mask = ~small
    if np.any(mask):
        x = kk[mask] / k_F
        bracket = np.empty_like(x)
        near_one = np.abs(x - 1.0) <= 1.0e-8
        bracket[near_one] = 1.0
        small_x = (x < 1.0e-5) & (~near_one)
        if np.any(small_x):
            xs = x[small_x]
            x2 = xs * xs
            bracket[small_x] = 2.0 - (2.0 / 3.0) * x2 - (2.0 / 15.0) * x2 * x2
        large_x = (x >= 1.0e4) & (~near_one)
        if np.any(large_x):
            y2 = 1.0 / (x[large_x] * x[large_x])
            bracket[large_x] = (
                (2.0 / 3.0) * y2
                + (2.0 / 15.0) * y2 * y2
                + (2.0 / 35.0) * y2 * y2 * y2
            )
        general = ~(near_one | small_x | large_x)
        if np.any(general):
            xg = x[general]
            log_ratio = np.empty_like(xg)
            below = xg < 1.0
            if np.any(below):
                xb = xg[below]
                log_ratio[below] = np.log1p(xb) - np.log1p(-xb)
            if np.any(~below):
                y = 1.0 / xg[~below]
                log_ratio[~below] = np.log1p(y) - np.log1p(-y)
            bracket[general] = 1.0 + ((1.0 - xg * xg) / (2.0 * xg)) * log_ratio
        out[mask] = -(k_F / (np.pi * eps_inf)) * bracket
    out = out.reshape(k_arr.shape)
    return float(out) if scalar else out


def _static_exchange_p_cut(mu, beta, a, xcut):
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if xcut <= 0.0:
        raise ValueError("xcut must be positive.")
    if np.isinf(beta):
        return np.sqrt(max(float(mu), 0.0) / a)
    return np.sqrt((max(float(mu), 0.0) + float(xcut) / beta) / a)


def _exchange_log_ratio(k, p, tol=1e-14):
    if p == 0.0:
        return 0.0
    scale = max(1.0, abs(k), abs(p))
    if abs(k - p) <= tol * scale:
        return 0.0
    large = max(k, p)
    small = min(k, p)
    y = small / large
    return np.log1p(y) - np.log1p(-y)


def _log_abs_integral_to_cut(k, p_cut):
    left = k * np.log(k) - k
    right_len = p_cut - k
    right = 0.0 if right_len <= 0.0 else right_len * np.log(right_len) - right_len
    return left + right


def sigma_x_static(k, mu, beta, a, eps_inf, xcut=60.0,
                   epsabs=1e-11, epsrel=1e-9, limit=200, tol=1e-12):
    """Finite-temperature static one-spin Coulomb exchange.

    The radial momentum integral is truncated where
    ``beta * (a*p**2 - mu)`` exceeds ``xcut``.  This is a numerical cutoff on
    the Fermi tail, not a change in the Coulomb kernel.
    """
    k = float(k)
    mu = float(mu)
    beta = float(beta)
    a = float(a)
    eps_inf = float(eps_inf)
    if k < -tol:
        raise ValueError("radial momentum k must be non-negative.")
    if beta <= 0.0 or a <= 0.0 or eps_inf <= 0.0:
        raise ValueError("beta, a, and eps_inf must be positive.")
    k = abs(k)
    if np.isinf(beta):
        return sigma_x_t0(k, np.sqrt(max(mu, 0.0) / a), eps_inf, tol=tol)
    p_cut = _static_exchange_p_cut(mu, beta, a, xcut)
    if p_cut <= 0.0:
        return 0.0
    scale = max(1.0, p_cut)
    if k <= tol * scale:
        val, _ = integrate.quad(
            lambda p: f_occ_k(p, mu, beta, a), 0.0, p_cut,
            epsabs=epsabs, epsrel=epsrel, limit=limit)
        return -2.0 * val / (np.pi * eps_inf)

    points = []
    p_mu = np.sqrt(max(mu, 0.0) / a)
    for point in (p_mu, k):
        if 0.0 < point < p_cut:
            points.append(point)
    points = sorted(set(points))
    if k <= p_cut:
        gk = k * f_occ_k(k, mu, beta, a)
        log_abs_int = _log_abs_integral_to_cut(k, p_cut)

        def integrand(p):
            gp = p * f_occ_k(p, mu, beta, a)
            delta = abs(p - k)
            diff_log = 0.0 if delta <= tol * scale else (gp - gk) * np.log(delta)
            return gp * np.log(k + p) - diff_log

        val, _ = integrate.quad(
            integrand, 0.0, p_cut, points=points if points else None,
            epsabs=epsabs, epsrel=epsrel, limit=limit)
        val -= gk * log_abs_int
        return -val / (np.pi * eps_inf * k)

    def integrand(p):
        return p * f_occ_k(p, mu, beta, a) * _exchange_log_ratio(k, p)

    val, _ = integrate.quad(
        integrand, 0.0, p_cut, points=points if points else None,
        epsabs=epsabs, epsrel=epsrel, limit=limit)
    return -val / (np.pi * eps_inf * k)


@_lazy_parallel_call
def get_static_tail(_, k_tail, mu, beta, a, eps_inf, xcut=60.0):
    """Precompute finite-temperature static exchange on passive high-k rows."""
    k_tail = np.asarray(k_tail, dtype=float)
    if k_tail.ndim != 1:
        raise ValueError("k_tail must be one-dimensional.")
    kloc0, kloc1 = _task_location(len(k_tail))
    tail = np.asarray([
        sigma_x_static(k, mu, beta, a, eps_inf, xcut=xcut)
        for k in k_tail[kloc0:kloc1]
    ], dtype=float)
    return _gather_rank0(tail)
