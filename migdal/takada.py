"""Takada on-shell/static projected G0W0 driver.

This module is intentionally separate from the sparse-IR radial drivers.  It
uses eV/Angstrom units and solves a dense projected pairing equation on the
dimensionless energy variable ``x = xi / eps_F``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from migdal import model, mpi_wrap


@dataclass(frozen=True)
class EnergyGrid:
    """Dimensionless Takada energy grid in ``x = xi / eps_F``."""

    x: np.ndarray
    dx: np.ndarray


@dataclass
class TakadaResult:
    """One density point from the projected Takada solver.

    ``eps_F_eV`` and ``k_F_A`` are reported in eV and Angstrom^-1 because the
    projected Takada formulas in this module use eV/Angstrom units.
    """

    n_cm3: float
    lam: float
    eps_F_eV: float
    k_F_A: float
    symmetry_error: float


def get_x_grid(x_core=1.0e-5, x_max=1.0e3, n_neg=48, n_pos=52):
    """Return the log midpoint grid on ``x = xi / eps_F``.

    The grid covers occupied states down to ``x=-1`` and unoccupied states up
    to ``x_max``.  A finite cell around ``x=0`` keeps the Fermi-surface kernel
    integrable in the projected eigenproblem.
    """
    if x_core <= 0.0:
        raise ValueError("x_core must be positive.")
    if x_max <= x_core:
        raise ValueError("x_max must be larger than x_core.")
    if int(n_neg) < 1 or int(n_pos) < 1:
        raise ValueError("n_neg and n_pos must be positive.")
    neg_edges = -np.geomspace(1.0, x_core, int(n_neg) + 1)
    pos_edges = np.geomspace(x_core, x_max, int(n_pos) + 1)
    x_neg = 0.5 * (neg_edges[:-1] + neg_edges[1:])
    dx_neg = neg_edges[1:] - neg_edges[:-1]
    x_mid = np.array([0.0])
    dx_mid = np.array([2.0 * x_core])
    x_pos = 0.5 * (pos_edges[:-1] + pos_edges[1:])
    dx_pos = pos_edges[1:] - pos_edges[:-1]
    return EnergyGrid(
        x=np.concatenate([x_neg, x_mid, x_pos]),
        dx=np.concatenate([dx_neg, dx_mid, dx_pos]),
    )


def _m_nat_and_alpha(m_eff):
    alpha = model.HBAR2_OVER_2ME_EV_A2 / float(m_eff)
    return 1.0 / (2.0 * alpha), alpha


def _ln_ratio_static(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    near_one = np.isclose(z, 1.0, rtol=1.0e-13, atol=1.0e-15)
    out[near_one] = 0.0
    left = (~near_one) & (z < 1.0)
    if np.any(left):
        out[left] = np.log1p(z[left]) - np.log(1.0 - z[left])
    right = (~near_one) & (z > 1.0)
    if np.any(right):
        out[right] = np.log1p(z[right]) - np.log(z[right] - 1.0)
    return out


def get_Pi0(q_A, omega_eV, n_cm3, m_eff):
    r"""Return T=0 Lindhard ``Pi0(q, i*omega)`` in eV/Angstrom units.

    ``q_A`` is in Angstrom^-1, ``omega_eV`` in eV, and ``n_cm3`` is a
    spin-summed density in cm^-3.  The returned polarization is non-positive
    with spin degeneracy included.
    """
    m_nat, _ = _m_nat_and_alpha(m_eff)
    k_F, _ = model.fermi_quantities_ev(n_cm3, m_eff)
    q_A, omega_eV = np.broadcast_arrays(
        np.asarray(q_A, dtype=float), np.asarray(omega_eV, dtype=float)
    )
    z = q_A / (2.0 * k_F)
    bracket = np.empty_like(z)
    q_zero = np.abs(q_A) < 1.0e-14 * max(k_F, 1.0)
    omega_zero = np.abs(omega_eV) < 1.0e-14
    bracket[q_zero & omega_zero] = 2.0
    bracket[q_zero & ~omega_zero] = 0.0
    static = (~q_zero) & omega_zero
    if np.any(static):
        zs = z[static]
        term = ((1.0 - zs**2) / (2.0 * zs)) * _ln_ratio_static(zs)
        term = np.where(np.isclose(zs, 1.0, rtol=1.0e-13, atol=1.0e-15), 0.0, term)
        bracket[static] = 1.0 + term
    dynamic = ~(q_zero | omega_zero)
    if np.any(dynamic):
        zd = z[dynamic]
        ud = m_nat * omega_eV[dynamic] / (q_A[dynamic] * k_F)
        den = (1.0 - zd) ** 2 + ud**2
        log_ratio = np.log1p(4.0 * zd / den)
        atan_term = np.arctan2(2.0 * ud, ud**2 + zd**2 - 1.0)
        bracket[dynamic] = (
            1.0 + ((1.0 - zd**2 + ud**2) / (4.0 * zd)) * log_ratio
            - ud * atan_term
        )
    return -(m_nat * k_F / (2.0 * np.pi**2)) * bracket


def get_eps(q_A, omega_eV, n_cm3, m_eff, params):
    """Return the STO RPA dielectric function ``eps(q, i*omega)``.

    This uses the same phonon dielectric convention as
    :func:`migdal.model.get_Vph_q`, but in eV/Angstrom units.
    """
    q_A, omega_eV = np.broadcast_arrays(
        np.asarray(q_A, dtype=float), np.asarray(omega_eV, dtype=float)
    )
    if np.any(q_A <= 0.0):
        raise ValueError("q_A must be positive for the Coulomb dielectric function.")
    Pi = get_Pi0(q_A, omega_eV, n_cm3, m_eff)
    lattice = model.get_lattice_eps_iw_ev(q_A, omega_eV, params)
    electronic = -(4.0 * np.pi * model.E2_EV_A / q_A**2) * Pi
    return lattice + electronic


def get_W(q_A, omega_eV, n_cm3, m_eff, params):
    """Return screened Coulomb interaction in eV Angstrom units."""
    eps = get_eps(q_A, omega_eV, n_cm3, m_eff, params)
    return 4.0 * np.pi * model.E2_EV_A / (q_A**2 * eps)


def _omega_breaks(eps_F, params):
    if model.has_multipole_phonon(params):
        phonon = np.array(params.phonon_to_poles_cm + params.phonon_lo_zeros_cm)
        phonon = phonon * model.CM1_TO_EV
    else:
        wt0 = model.CM1_TO_EV * params.omega_t0_cm
        wl0 = wt0 * math.sqrt(params.eps_0 / params.eps_inf)
        phonon = np.array([wt0, wl0])
    vals = np.concatenate(([0.25 * eps_F, eps_F, 4.0 * eps_F, 16.0 * eps_F], phonon))
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    return np.unique(np.sort(vals))


def _lorentz_edges(s, breaks):
    s = np.asarray(s, dtype=float)
    edges = [np.zeros_like(s)]
    for omega in breaks:
        ratio = np.divide(float(omega), s, out=np.full_like(s, np.inf), where=s > 0.0)
        edges.append((2.0 / np.pi) * np.arctan(ratio))
    edges.append(np.ones_like(s))
    return edges


def _omega_projected_W(q, s, n_cm3, m_eff, params, breaks, n_theta):
    yx, yw = leggauss(int(n_theta))
    edges = _lorentz_edges(s, breaks)
    out = np.zeros_like(q, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        width = hi - lo
        if np.max(width) <= 0.0:
            continue
        mid = 0.5 * (hi + lo)
        half = 0.5 * width
        y = mid[..., None] + half[..., None] * yx[None, None, :]
        omega = s[..., None] * np.tan(0.5 * np.pi * y)
        vals = get_W(q[..., None], omega[:, :, None, :], n_cm3, m_eff, params)
        out += np.sum(half[..., None, None] * yw[None, None, None, :] * vals, axis=-1)
    return out


def _kernel_rows(x_left, x_right, n_cm3, m_eff, params, n_q, n_theta):
    k_F, eps_F = model.fermi_quantities_ev(n_cm3, m_eff)
    m_nat, _ = _m_nat_and_alpha(m_eff)
    y_left = np.sqrt(np.maximum(0.0, 1.0 + x_left))[:, None]
    y_right = np.sqrt(np.maximum(0.0, 1.0 + x_right))[None, :]
    p_left = k_F * y_left
    p_right = k_F * y_right
    s = eps_F * (np.abs(x_left)[:, None] + np.abs(x_right)[None, :])
    q_min = np.abs(p_left - p_right)
    q_max = p_left + p_right
    tq, wq = leggauss(int(n_q))
    q_mid = 0.5 * (q_max + q_min)
    q_half = 0.5 * (q_max - q_min)
    q = q_mid[..., None] + q_half[..., None] * tq[None, None, :]
    dq = q_half[..., None] * wq[None, None, :]
    q = np.maximum(q, 1.0e-12 * max(k_F, 1.0e-8))
    theta_int = _omega_projected_W(
        q, s, n_cm3, m_eff, params, _omega_breaks(eps_F, params), int(n_theta)
    )
    q_int = np.sum(dq * q * theta_int, axis=-1)
    pref = m_nat / (4.0 * np.pi**2 * np.maximum(p_left, 1.0e-300))
    return pref * q_int


def _task_location(n, size, task):
    neach, extras = divmod(int(n), int(size))
    section_sizes = [0] + extras * [neach + 1] + (int(size) - extras) * [neach]
    div_points = np.cumsum(section_sizes)
    return int(div_points[task]), int(div_points[task + 1])


def _gather_rank0(arr):
    return mpi_wrap.gather(arr)


def _kernel_matrix_impl(_, x, n_cm3, m_eff, params, n_q, n_theta):
    x = np.asarray(x, dtype=float)
    mpi = mpi_wrap.mpi
    loc0, loc1 = _task_location(len(x), mpi.pool.size, mpi.rank)
    block = _kernel_rows(x[loc0:loc1], x, n_cm3, m_eff, params, n_q, n_theta)
    return _gather_rank0(block)


def _parallel_kernel_matrix():
    return mpi_wrap.parallel_call(_kernel_matrix_impl)


_kernel_matrix_mpi = None


def _kernel_matrix(x, n_cm3, m_eff, params, n_q, n_theta):
    global _kernel_matrix_mpi
    if _kernel_matrix_mpi is None:
        _kernel_matrix_mpi = _parallel_kernel_matrix()
    return _kernel_matrix_mpi(None, x, n_cm3, m_eff, params, n_q, n_theta)


def _diag_average_block(
    x, dx, n_cm3, m_eff, params, n_q, n_theta, n_x_diag, loc0, loc1,
):
    if n_x_diag % 2:
        n_x_diag += 1
    tx, wx = leggauss(int(n_x_diag))
    diag = np.empty(int(loc1) - int(loc0), dtype=float)
    for local_i, i in enumerate(range(int(loc0), int(loc1))):
        xi = x[i]
        dxi = dx[i]
        lo = xi - 0.5 * dxi
        hi = xi + 0.5 * dxi
        x_nodes = 0.5 * (hi + lo) + 0.5 * dxi * tx
        vals = _kernel_rows(
            np.array([xi]), x_nodes, n_cm3, m_eff, params, n_q, n_theta
        )[0]
        diag[local_i] = 0.5 * np.dot(wx, vals)
    return diag


def _diag_average_impl(_, x, dx, n_cm3, m_eff, params, n_q, n_theta, n_x_diag=8):
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    mpi = mpi_wrap.mpi
    loc0, loc1 = _task_location(len(x), mpi.pool.size, mpi.rank)
    diag = _diag_average_block(
        x, dx, n_cm3, m_eff, params, n_q, n_theta, n_x_diag, loc0, loc1,
    )
    return _gather_rank0(diag)


def _parallel_diag_average():
    return mpi_wrap.parallel_call(_diag_average_impl)


_diag_average_mpi = None


def _diag_average(x, dx, n_cm3, m_eff, params, n_q, n_theta, n_x_diag=8):
    global _diag_average_mpi
    if _diag_average_mpi is None:
        _diag_average_mpi = _parallel_diag_average()
    return _diag_average_mpi(
        None, x, dx, n_cm3, m_eff, params, n_q, n_theta, n_x_diag,
    )


def _thermal_weight(x, eps_F, temperature_K):
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive.")
    temp_ev = model.KB_EV_PER_K * float(temperature_K)
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    arg = eps_F * ax / (2.0 * temp_ev)
    g = np.empty_like(ax)
    nz = ax > 1.0e-15
    g[nz] = np.tanh(arg[nz]) / (2.0 * ax[nz])
    g[~nz] = eps_F / (4.0 * temp_ev)
    return g


def _solve_dense(H):
    vals = np.linalg.eigvalsh(np.asarray(H))
    return float(vals[-1])


class TakadaProjection:
    """Takada projected G0W0 driver for the STO radial model.

    Unlike the sparse-IR G0W0/GW0/scGW drivers, this projected solver works in
    dimensionless energy ``x = xi / eps_F`` and eV/Angstrom units.  It builds a
    symmetrized dense kernel and returns the leading pairing eigenvalue.
    """

    def __init__(
        self,
        params: model.STOParams | None = None,
        densities="default",
        temperature=1.0,
        m_eff=None,
        n_q=20,
        n_theta=12,
        x_core=1.0e-5,
        x_max=1.0e3,
        n_neg=48,
        n_pos=52,
        max_symmetry_error=1.0e-10,
    ):
        self.params = model.STOParams() if params is None else params
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if (self.params.m_eff if m_eff is None else m_eff) <= 0.0:
            raise ValueError("m_eff must be positive.")
        if int(n_q) < 1 or int(n_theta) < 1:
            raise ValueError("n_q and n_theta must be positive.")
        if max_symmetry_error < 0.0:
            raise ValueError("max_symmetry_error must be non-negative.")
        self.densities = densities
        self.temperature = temperature
        self.m_eff = self.params.m_eff if m_eff is None else m_eff
        self.n_q = n_q
        self.n_theta = n_theta
        self.x_core = x_core
        self.x_max = x_max
        self.n_neg = n_neg
        self.n_pos = n_pos
        self.max_symmetry_error = max_symmetry_error
        self.grid = None
        self.results = []

    def set(self, **kwargs):
        """Set projection-driver attributes in PySCF style and return ``self``."""
        for key, val in kwargs.items():
            setattr(self, key, val)
        return self

    def build(self):
        """Build the dimensionless energy grid used by the dense kernel."""
        self.grid = get_x_grid(self.x_core, self.x_max, self.n_neg, self.n_pos)
        return self

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

    def kernel(self, n_cm3=None):
        """Run one density point and return a :class:`TakadaResult`."""
        if self.grid is None:
            self.build()
        n_cm3 = self._single_density(n_cm3)
        x = self.grid.x
        dx = self.grid.dx
        k_F, eps_F = model.fermi_quantities_ev(float(n_cm3), self.m_eff)
        p = k_F * np.sqrt(np.maximum(0.0, 1.0 + x))
        K = _kernel_matrix(
            x, float(n_cm3), self.m_eff, self.params, self.n_q, self.n_theta)
        diag = _diag_average(
            x, dx, float(n_cm3), self.m_eff, self.params,
            self.n_q, self.n_theta,
        )
        np.fill_diagonal(K, diag)
        raw_sym = np.sqrt(p[:, None] / p[None, :]) * K
        sym_err = _max_abs(raw_sym - raw_sym.T) / max(_max_abs(raw_sym), 1.0e-300)
        if sym_err > self.max_symmetry_error:
            raise AssertionError("Takada kernel symmetry error %.3e" % sym_err)
        sym_K = 0.5 * (raw_sym + raw_sym.T)
        tw = _thermal_weight(x, eps_F, self.temperature)
        w = np.sqrt(dx * tw)
        H = -(w[:, None]) * sym_K * w[None, :]
        lam = _solve_dense(H)
        row = TakadaResult(
            n_cm3=float(n_cm3),
            lam=lam,
            eps_F_eV=eps_F,
            k_F_A=k_F,
            symmetry_error=float(sym_err),
        )
        self.results = [row]
        return row

    def scan(self, densities=None, restart=False):
        """Run a density scan. ``restart`` is accepted for API symmetry."""
        if self.grid is None:
            self.build()
        densities = self.densities if densities is None else densities
        rows = []
        for n_cm3 in self._density_values(densities):
            rows.append(self.kernel(float(n_cm3)))
        self.results = rows
        return rows

    def write(self, filename):
        """Write density, projected lambda, Fermi scales, and symmetry error."""
        if not self.results:
            raise ValueError("no results available; call kernel() first.")
        arr = np.asarray([
            (r.n_cm3, r.lam, r.eps_F_eV, r.k_F_A, r.symmetry_error)
            for r in self.results
        ])
        np.savetxt(
            filename,
            arr,
            header="n_cm3 lambda eps_F_eV k_F_Angstrom_inv symmetry_error",
        )
        return self


def _max_abs(a):
    """Return ``max(abs(a))`` as a Python float, with zero for empty arrays."""
    a = np.asarray(a)
    return 0.0 if a.size == 0 else float(np.max(np.abs(a)))
