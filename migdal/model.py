r"""STO radial one-band model formulas and unit conversions.

The G0W0, GW0, and scGW drivers use atomic units internally: Hartree for
energy, Bohr for length, and Bohr^-1 for momentum.  Public densities are
spin-summed carrier densities in cm^-3; the drivers convert them to Bohr^-3
before building the radial Fermi sphere.

The normal radial band is parabolic,

.. math::

    \epsilon(k) = a k^2,\qquad a = \frac{1}{2 m_\mathrm{eff}},

with ``m_eff`` in electron-mass units.  The zero-temperature density convention
used throughout the code is

.. math::

    n = \frac{k_F^3}{3\pi^2}.

The interaction helpers use the sign convention where the electron-gas
polarization is non-positive, ``Pi <= 0``.  The screened interaction is later
formed as ``W = Vtot / (1 - Pi * Vtot)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import warnings

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from pyscf.data.nist import BOHR, HARTREE2EV


CM1_TO_EV = 1.239841984e-4
EV_TO_HA = 1.0 / HARTREE2EV
KB_EV_PER_K = 8.617333262145e-5
KB_HA_PER_K = KB_EV_PER_K * EV_TO_HA
# eV/Angstrom couplings are derived from the same NIST constants as the
# Hartree/Bohr drivers (e^2 = 1 Ha*Bohr, hbar^2/(2 m_e) = 0.5 Ha*Bohr^2), so
# the Takada eV/Angstrom path and the atomic-unit radial path share one unit
# system exactly.
E2_EV_A = HARTREE2EV * BOHR
HBAR2_OVER_2ME_EV_A2 = 0.5 * HARTREE2EV * BOHR**2


@dataclass(frozen=True)
class STOParams:
    """Parameters for the SrTiO3-like radial one-band model.

    Parameters
    ----------
    a_ang : float
        Cubic lattice constant in Angstrom.  It is used for reporting and unit
        conversions; the radial integrals use the density in Bohr^-3.
    m_eff : float
        Effective mass in electron-mass units.  The dispersion is
        ``eps(k) = k**2 / (2*m_eff)`` in atomic units.
    eps_inf, eps_0 : float
        High-frequency and static dielectric constants.
    omega_t0_cm : float
        Transverse optical mode energy in cm^-1.
    omega_t_disp_cm_a2 : float
        Optional transverse-mode dispersion coefficient for ``omega_T(q)``
        with q in Angstrom^-1.  This legacy linear-frequency convention is
        used only by the single-pole model.
    phonon_to_poles_cm, phonon_lo_zeros_cm : tuple of float
        Optional multipole phonon dielectric representation.  When both are
        provided, the lattice factor is
        ``prod_j (nu^2 + TO_j^2) / (nu^2 + LO_j^2)`` instead of the legacy
        single-pole Lyddane-Sachs-Teller factor.
    phonon_to_omega2_disp_cm2_a2 : tuple of float
        Optional multipole TO-mode squared-frequency dispersion coefficients.
        When supplied, ``omega_TO,j(q)^2 = omega_TO,j(0)^2 + beta_j*q_A^2``,
        with q in Angstrom^-1.
    phonon_to_omega2_cap_cm2 : tuple of float
        Optional caps for multipole TO-mode squared frequencies, in cm^-2.
        Finite caps replace the uncapped parabola by a smooth rational form
        with the same small-q slope and a large-q limit below the matching LO
        zero.  Use ``math.inf`` for uncapped modes.

    Notes
    -----
    In the legacy single-pole model, ``omega_l_ha`` is not an independent
    input.  It is the zone-center value from the Lyddane-Sachs-Teller ratio
    ``omega_L / omega_T = sqrt(eps_0 / eps_inf)``.  With the q-dependent
    ``eps_0(q)`` convention below, the longitudinal mode remains equal to this
    zone-center value unless ``eps_0 == eps_inf`` is used as the exact
    no-phonon limit.  The sparse-IR radial drivers use Hartree/Bohr units,
    while the Takada projected formulas in :mod:`migdal.takada` convert to
    eV/Angstrom units.
    """

    a_ang: float = 3.901
    m_eff: float = 2.0
    eps_inf: float = 5.2
    eps_0: float = 23000.0
    omega_t0_cm: float = 6.34
    omega_t_disp_cm_a2: float = 0.0
    phonon_to_poles_cm: tuple[float, ...] = ()
    phonon_lo_zeros_cm: tuple[float, ...] = ()
    phonon_to_omega2_disp_cm2_a2: tuple[float, ...] = ()
    phonon_to_omega2_cap_cm2: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        to_poles = tuple(float(x) for x in self.phonon_to_poles_cm)
        lo_zeros = tuple(float(x) for x in self.phonon_lo_zeros_cm)
        to_disp = tuple(float(x) for x in self.phonon_to_omega2_disp_cm2_a2)
        to_cap = tuple(float(x) for x in self.phonon_to_omega2_cap_cm2)
        object.__setattr__(self, "phonon_to_poles_cm", to_poles)
        object.__setattr__(self, "phonon_lo_zeros_cm", lo_zeros)
        object.__setattr__(self, "phonon_to_omega2_disp_cm2_a2", to_disp)
        object.__setattr__(self, "phonon_to_omega2_cap_cm2", to_cap)

        if bool(to_poles) != bool(lo_zeros):
            raise ValueError("multipole phonons require both TO poles and LO zeros")
        if not to_poles:
            if to_disp:
                raise ValueError(
                    "phonon_to_omega2_disp_cm2_a2 requires multipole phonons"
                )
            if to_cap:
                raise ValueError(
                    "phonon_to_omega2_cap_cm2 requires multipole phonons"
                )
            return
        if len(to_poles) != len(lo_zeros):
            raise ValueError("multipole TO poles and LO zeros must have the same length")
        if any(freq <= 0.0 for freq in to_poles + lo_zeros):
            raise ValueError("multipole phonon frequencies must be positive")
        if float(self.eps_inf) <= 0.0:
            raise ValueError("eps_inf must be positive")
        if float(self.eps_0) < float(self.eps_inf):
            raise ValueError("eps_0 must be greater than or equal to eps_inf")
        if float(self.omega_t_disp_cm_a2) != 0.0:
            raise ValueError(
                "omega_t_disp_cm_a2 is only for single-pole phonons; "
                "use phonon_to_omega2_disp_cm2_a2 for multipole q-dependence"
            )
        if to_disp and len(to_disp) != len(to_poles):
            raise ValueError(
                "multipole TO squared-frequency dispersion must match "
                "the number of TO poles"
            )
        if to_cap:
            if not to_disp:
                raise ValueError(
                    "multipole TO squared-frequency caps require "
                    "squared-frequency dispersion"
                )
            if len(to_cap) != len(to_poles):
                raise ValueError(
                    "multipole TO squared-frequency caps must match "
                    "the number of TO poles"
                )
            to0 = np.asarray(to_poles, dtype=float)
            lo = np.asarray(lo_zeros, dtype=float)
            beta = np.asarray(to_disp, dtype=float)
            cap2 = np.asarray(to_cap, dtype=float)
            finite = np.isfinite(cap2)
            positive_inf = np.isinf(cap2) & (cap2 > 0.0)
            invalid = ~(finite | positive_inf)
            if np.any(invalid) or np.any(cap2[finite] <= 0.0):
                raise ValueError(
                    "multipole TO squared-frequency caps must be positive "
                    "finite values or math.inf"
                )
            if np.any(cap2[finite] <= to0[finite] ** 2):
                raise ValueError(
                    "finite multipole TO squared-frequency caps must exceed "
                    "the matching TO pole"
                )
            if np.any(cap2[finite] >= lo[finite] ** 2):
                raise ValueError(
                    "finite multipole TO squared-frequency caps must stay "
                    "below the matching LO zero"
                )
            if np.any(beta[finite] <= 0.0):
                raise ValueError(
                    "finite multipole TO squared-frequency caps require "
                    "positive dispersion coefficients"
                )

        ratio = np.prod((np.asarray(lo_zeros) / np.asarray(to_poles)) ** 2)
        target = float(self.eps_0) / float(self.eps_inf)
        if not np.isclose(ratio, target, rtol=1e-6, atol=1e-12):
            raise ValueError(
                "multipole phonons must satisfy generalized LST: "
                "prod((LO/TO)^2) == eps_0/eps_inf"
            )

    @property
    def a_bohr(self) -> float:
        """Cubic lattice constant in Bohr."""
        return self.a_ang / BOHR

    @property
    def band_a_ha_bohr2(self) -> float:
        """Parabolic-band coefficient ``a = 1 / (2*m_eff)`` in Ha Bohr^2."""
        return 1.0 / (2.0 * self.m_eff)

    @property
    def omega_t_ha(self) -> float:
        """Zone-center transverse optical phonon energy in Hartree."""
        return self.omega_t0_cm * CM1_TO_EV * EV_TO_HA

    @property
    def omega_l_ha(self) -> float:
        """Zone-center longitudinal phonon energy from the LST ratio."""
        return self.omega_t_ha * math.sqrt(self.eps_0 / self.eps_inf)

    def beta_ha(self, temperature_K: float) -> float:
        """Return inverse temperature ``1/(k_B T)`` in Hartree^-1."""
        if temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive.")
        return 1.0 / (KB_HA_PER_K * float(temperature_K))


def density_grid(text: str) -> np.ndarray:
    """Return a positive spin-summed density scan in cm^-3.

    ``"default"`` means the production-style logarithmic scan from
    ``1e23`` down to ``1e16`` cm^-3.  A comma-separated string is parsed
    literally and validated as positive densities.
    """
    if text == "default":
        return np.geomspace(1.0e23, 1.0e16, 11)
    vals = np.asarray([float(x) for x in text.split(",")], dtype=float)
    if vals.ndim != 1 or vals.size == 0 or np.any(vals <= 0.0):
        raise ValueError("densities must be a non-empty positive list.")
    return vals


def _density_values(densities) -> np.ndarray:
    """Return a one-dimensional positive density array in cm^-3.

    This private helper keeps the driver APIs consistent while preserving the
    public ``density_grid("default")`` convention used by the CLI and examples.
    """
    if isinstance(densities, str):
        return density_grid(densities)
    vals = np.asarray(densities, dtype=float)
    if vals.ndim != 1 or vals.size == 0 or np.any(vals <= 0.0):
        raise ValueError("densities must be positive.")
    return vals


def _single_density(densities, n_cm3=None) -> float:
    """Return one positive density in cm^-3 for ``kernel``-style calls."""
    if n_cm3 is not None:
        n_cm3 = float(n_cm3)
        if n_cm3 <= 0.0:
            raise ValueError("density must be positive.")
        return n_cm3
    vals = _density_values(densities)
    if len(vals) != 1:
        raise ValueError("kernel() is single-point; pass n_cm3 or call scan().")
    return float(vals[0])


def refine_density_path(densities, max_step_dex=0.0):
    """Insert log-spaced continuation points and mark requested densities.

    Parameters
    ----------
    densities : array_like or str
        Requested densities in cm^-3.
    max_step_dex : float, optional
        Maximum log10 spacing between adjacent continuation points.  Nonpositive
        values leave the requested path unchanged.

    Returns
    -------
    path : ndarray
        Density path including inserted continuation points.
    keep : ndarray of bool
        Mask selecting the originally requested points.  GW0/scGW scans use
        the inserted points only as SCF warm starts unless intermediate output
        is explicitly requested.
    """
    vals = _density_values(densities)
    max_step_dex = float(max_step_dex or 0.0)
    if max_step_dex <= 0.0 or len(vals) <= 1:
        return vals, np.ones(len(vals), dtype=bool)
    logs_requested = np.log10(vals)
    path = [float(vals[0])]
    keep = [True]
    for left, right in zip(logs_requested[:-1], logs_requested[1:]):
        nseg = max(1, int(np.ceil(abs(right - left) / max_step_dex)))
        logs = np.linspace(left, right, nseg + 1)
        for i, logn in enumerate(logs[1:], start=1):
            path.append(float(10.0**logn))
            keep.append(i == nseg)
    return np.asarray(path), np.asarray(keep, dtype=bool)


def bool_auto(text: str | int | bool, default: bool) -> bool:
    """Parse 0/1/auto booleans used by the CLI."""
    if isinstance(text, bool):
        return text
    if str(text).lower() == "auto":
        return bool(default)
    return bool(int(text))


def int_or_auto(text: str | int) -> int | str:
    """Parse integer-or-auto CLI values."""
    if isinstance(text, int):
        return text
    if str(text).lower() == "auto":
        return "auto"
    return int(text)


def n_cm3_to_bohr3(n_cm3: float) -> float:
    """Convert spin-summed number density from cm^-3 to Bohr^-3."""
    if n_cm3 <= 0.0:
        raise ValueError("density must be positive.")
    return float(n_cm3) * (BOHR * 1.0e-8) ** 3


def n_bohr3_to_cm3(n_bohr3: float) -> float:
    """Convert spin-summed number density from Bohr^-3 to cm^-3."""
    return float(n_bohr3) / (BOHR * 1.0e-8) ** 3


def e(k, a):
    """Parabolic band energy ``a*k**2`` in Hartree."""
    return a * np.asarray(k) ** 2


def f_occ(eps, beta):
    r"""Return the Fermi occupation ``f(eps)``.

    Parameters
    ----------
    eps : array_like
        One-particle energy measured relative to the chemical potential, in the
        same energy unit as ``beta^-1``.
    beta : float
        Inverse temperature.  ``np.inf`` gives the zero-temperature step
        function.

    Notes
    -----
    The implementation evaluates

    .. math::

        f(\epsilon) = \frac{1}{\exp(\beta\epsilon) + 1}

    with explicit large-argument branches to avoid overflow.  The return value
    has the same shape as ``eps``.
    """
    eps = np.asarray(eps, dtype=float)
    if np.isinf(beta):
        return (eps <= 0.0).astype(float)
    x = float(beta) * eps
    out = np.empty_like(x, dtype=float)
    high = x > 60.0
    low = x < -60.0
    mid = ~(high | low)
    out[high] = 0.0
    out[low] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def get_n(mu, beta, a, tol=1e-15):
    r"""Return spin-summed density for ``eps(k)=a*k**2``.

    The returned density includes spin degeneracy.  The zero-temperature
    convention is ``n = k_F**3/(3*pi**2)`` with ``k_F = sqrt(mu/a)``.
    At finite temperature the radial integral is

    .. math::

        n = \frac{1}{\pi^2}\int_0^\infty dk\, k^2
            f(a k^2 - \mu).
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if np.isinf(beta):
        if mu <= 0.0:
            return 0.0
        k_F = math.sqrt(mu / a)
        return k_F**3 / (3.0 * math.pi**2)

    def fn(k):
        return k * k * f_occ(a * k * k - mu, beta)

    val, _ = integrate.quad(fn, 0.0, np.inf, limit=1000, epsabs=tol)
    return val / (math.pi**2)


def get_mu(n, beta, a, tol=1e-15):
    """Return chemical potential for a target spin-summed density.

    This is the analytic inverse at zero temperature and a scalar Brent solve
    at finite temperature.  It is used for noninteracting reference quantities;
    interacting GW0/scGW density constraints are solved in :mod:`migdal.scf`
    because they depend on the current self-energy.
    """
    if n < 0.0:
        raise ValueError("n must be non-negative.")
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if np.isinf(beta):
        k_F = (3.0 * math.pi**2 * n) ** (1.0 / 3.0)
        return a * k_F**2

    def cost(mu):
        return get_n(mu, beta, a) - n

    left = -1.0 / beta
    right = 1.0
    while cost(left) > 0.0:
        left -= max(1.0, 10.0 / beta)
    while cost(right) < 0.0:
        right *= 2.0
        if right < 1.0:
            right = 1.0
    res = brentq(cost, left, right, xtol=tol, rtol=tol, maxiter=10000,
                 full_output=True, disp=False)
    if not res[1].converged:
        raise RuntimeError("chemical-potential root solve did not converge.")
    return float(res[0])


def get_Vc(q, eps_inf, q0_treat=False, tol=1e-12):
    r"""Return the radial Coulomb table ``4*pi/(eps_inf*q**2)``.

    Parameters
    ----------
    q : array_like
        Radial momentum in Bohr^-1.
    eps_inf : float
        High-frequency dielectric constant.
    q0_treat : bool, optional
        Unused q=0 special-case switch.  Passing ``True`` raises a
        ``ValueError`` because Migdal fixes the grid value at ``q=0`` to zero;
        the Coulomb singularity is handled by radial integral primitives, not
        by a special point value.
    tol : float, optional
        Values with ``q < tol`` are treated as the ``q=0`` placeholder.

    Returns
    -------
    Vc : ndarray
        Coulomb table in Hartree Bohr^3 units on the supplied radial q grid.
    """
    if q0_treat:
        raise ValueError(
            "q0_treat=True is not implemented: Migdal fixes the q=0 Coulomb "
            "grid value to zero, while the "
            "singular limit is treated by radial integration."
        )
    q = np.asarray(q, dtype=float)
    V = np.zeros_like(q)
    mask = q >= tol
    V[mask] = (4.0 * math.pi / float(eps_inf)) / (q[mask] ** 2)
    return V


def _ln_ratio_static(z):
    """Stable ``log((1+z)/abs(1-z))`` for z >= 0."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    eq1 = z == 1.0
    lt1 = z < 1.0
    gt1 = z > 1.0
    out[eq1] = 0.0
    if np.any(lt1):
        d = 1.0 - z[lt1]
        out[lt1] = np.log(2.0) - np.log(d) + np.log1p(-d / 2.0)
    if np.any(gt1):
        d = z[gt1] - 1.0
        out[gt1] = np.log(2.0) - np.log(d) + np.log1p(d / 2.0)
    return out


def get_Pi0(q, v, k_F, m_eff, tol=1e-12):
    r"""T=0 3D Lindhard ``Pi0(q, i*v)`` on the imaginary axis.

    ``Pi0`` is real and non-positive with spin degeneracy included in the
    standard ``-m*k_F/pi**2`` density-of-states limit.
    One-dimensional ``q`` and ``v`` inputs are interpreted as an outer
    ``(nq, nv)`` grid.
    """
    if k_F <= 0.0:
        raise ValueError("k_F must be positive.")
    if m_eff <= 0.0:
        raise ValueError("m_eff must be positive.")
    q = np.asarray(q, dtype=float)
    v = np.abs(np.asarray(v, dtype=float))
    if q.ndim == 1 and v.ndim == 1:
        q = q[:, None]
        v = v[None, :]
    q, v = np.broadcast_arrays(q, v)

    pref = (m_eff * k_F) / (2.0 * math.pi**2)
    Pi = np.empty_like(q, dtype=float)
    q0 = q == 0.0
    v0 = v == 0.0
    small_q = q < tol
    small_v = v < tol

    Pi[q0 & v0] = 2.0 * pref
    Pi[q0 & (~v0)] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        u_abs = (m_eff * v) / (q * k_F)
    static = (~q0) & (small_v | (np.abs(u_abs) < tol))
    if np.any(static):
        z = q[static] / (2.0 * k_F)
        term = ((1.0 - z * z) * _ln_ratio_static(z)) / (2.0 * z)
        Pi[static] = pref * (1.0 + term)

    remaining = ~(q0 & v0) & ~(q0 & (~v0)) & ~static
    small_nonzero_q = remaining & small_q & (~q0)
    general = remaining & (~small_q)
    if np.any(small_nonzero_q):
        u = (m_eff * v[small_nonzero_q]) / (q[small_nonzero_q] * k_F)
        theta = np.arctan2(2.0 * u, u * u - 1.0)
        Pi[small_nonzero_q] = pref * (2.0 - u * theta)
    if np.any(general):
        qg = q[general]
        vg = v[general]
        z = qg / (2.0 * k_F)
        u = (m_eff * vg) / (qg * k_F)
        den = (1.0 - z) ** 2 + u * u
        log_ratio = np.log1p(4.0 * z / den)
        term = ((1.0 - z * z + u * u) * log_ratio) / (4.0 * z)
        theta = np.arctan2(2.0 * u, u * u + z * z - 1.0)
        Pi[general] = pref * (1.0 + term - u * theta)

    Pi = -Pi
    if Pi.shape == ():
        return float(Pi)
    return Pi


@lru_cache(maxsize=None)
def _leggauss_cached(n):
    return np.polynomial.legendre.leggauss(int(n))


def _fermi_window(x):
    """Return ``f(x) * (1 - f(x))`` for ``f(x)=1/(exp(x)+1)``."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    out = np.empty_like(ax)
    mid = ax <= 40.0
    out[mid] = 0.25 / np.cosh(0.5 * x[mid]) ** 2
    out[~mid] = np.exp(-ax[~mid])
    return out


def _finite_temperature_dn_dmu(beta, mu, a, tail_cut):
    """Return ``dn/dmu`` for the parabolic band at finite temperature."""
    x0 = max(-float(beta) * float(mu), -float(tail_cut))
    x1 = float(tail_cut)
    if x0 >= x1:
        return 0.0

    def integrand(x):
        eps = float(mu) + x / float(beta)
        if eps <= 0.0:
            return 0.0
        return (
            math.sqrt(eps)
            * float(_fermi_window(np.array(x)))
            / (2.0 * a ** 1.5)
        )

    val, _ = integrate.quad(
        integrand, x0, x1, limit=500, epsabs=1.0e-12, epsrel=1.0e-12)
    return val / (math.pi**2)


def _pi_shape_args(q, v):
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.abs(np.asarray(v, dtype=float))
    if q_arr.ndim == 1 and v_arr.ndim == 1:
        return q_arr, v_arr, (q_arr.size, v_arr.size), True
    q_b, v_b = np.broadcast_arrays(q_arr, v_arr)
    return q_b, v_b, q_b.shape, False


def _get_Pi0_shaped(q, v, k_F, m_eff, tol, outer_grid):
    """Evaluate ``get_Pi0`` while preserving finite-T wrapper shape semantics."""
    if outer_grid:
        return get_Pi0(q, v, k_F, m_eff, tol=tol)
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if q_arr.ndim == 1 and v_arr.ndim == 1:
        return get_Pi0(
            q_arr[:, None], v_arr[:, None], k_F, m_eff, tol=tol
        ).reshape(q_arr.shape)
    return get_Pi0(q_arr, v_arr, k_F, m_eff, tol=tol)


def _maldague_panels(lo, hi, breaks):
    """Split the window ``(lo, hi)`` at the strictly interior break points.

    Callers must ensure ``lo < hi``.  Breaks outside the open window
    (including NaN) are dropped.
    """
    if not lo < hi:
        raise ValueError("Maldague window requires lo < hi.")
    cuts = sorted({float(b) for b in breaks if lo < b < hi})
    edges = [float(lo), *cuts, float(hi)]
    return tuple(zip(edges[:-1], edges[1:]))


def _maldague_nodes(panels, n_quad, beta, mu, a):
    r"""Gauss-Legendre nodes in ``k_F`` with Jacobian and thermal weight.

    The Maldague integral is taken in the variable ``k = k_F(E) = sqrt(E/a)``,
    in which the T=0 kernel is analytic at the ``E = 0`` band edge
    (``Pi0 ~ k**3`` times an even series in ``k``).  Each panel receives the
    full ``n_quad``-point rule; the weight is
    ``dx * f(1-f) = 2*a*beta*k * f(x)*(1-f(x)) dk`` with
    ``x = beta*(a*k**2 - mu)``.
    """
    nodes, weights = _leggauss_cached(n_quad)
    ks = []
    ws = []
    for k_lo, k_hi in panels:
        half = 0.5 * (k_hi - k_lo)
        ks.append(half * nodes + 0.5 * (k_hi + k_lo))
        ws.append(half * weights)
    k = np.concatenate(ks)
    w = np.concatenate(ws)
    return k, 2.0 * a * beta * k * w * _fermi_window(beta * (a * k * k - mu))


def _pi0_maldague_sum(q_flat, v_flat, beta, mu, a, panels, n_quad, tol):
    """Maldague sum of the T=0 Lindhard kernel, elementwise over ``q``/``v``."""
    k, w = _maldague_nodes(panels, n_quad, beta, mu, a)
    m_eff = 1.0 / (2.0 * a)
    Pi = np.zeros(np.shape(q_flat), dtype=float)
    for k_f, weight in zip(k, w):
        Pi += weight * _get_Pi0_shaped(
            q_flat, v_flat, float(k_f), m_eff, tol, False)
    return Pi


def _pi0_maldague_q(q_val, v_flat, beta, mu, a, panels, n_quad, tol):
    r"""Maldague sum for one Kohn-split ``q``: all nodes in one scaled call.

    The Lindhard scaling identity

    .. math::

        \Pi_0(q, \nu; k_F) = \frac{k_F}{k_0}
        \Pi_0\!\left(q \frac{k_0}{k_F}, \nu \frac{k_0^2}{k_F^2}; k_0\right)

    holds because :func:`get_Pi0` depends only on ``q/k_F`` and
    ``nu/(q k_F)`` times a ``k_F`` prefactor.  It folds every panel node into
    a single vectorized evaluation at a common reference ``k0``, instead of
    one small call per node.  ``k0`` is the upper window edge so that
    ``s = k0/k >= 1`` for every node: scaled arguments only grow, and can
    never fall below the absolute ``q``/``v`` tolerance branches inside
    :func:`get_Pi0` (a shrunk ``v*s**2 < tol`` would be misclassified as
    static).
    """
    k, w = _maldague_nodes(panels, n_quad, beta, mu, a)
    k0 = float(panels[-1][1])
    s = k0 / k
    Pi = get_Pi0(
        q_val * s[:, None],
        np.asarray(v_flat, dtype=float)[None, :] * np.square(s)[:, None],
        k0, 1.0 / (2.0 * a), tol=tol,
    )
    return np.dot(w * (k / k0), Pi)


def get_Pi0_finite_T(
    q,
    v,
    beta,
    mu,
    a,
    tol=1e-12,
    n_quad=96,
    tail_cut=40.0,
    finite_T_degenerate_cut=1.0e4,
):
    r"""Finite-temperature 3D Lindhard ``Pi0(q, i*v)``.

    The parabolic band is ``eps(k)=a*k**2`` and ``mu`` is the finite-temperature
    chemical potential.  The sign convention matches :func:`get_Pi0`:
    ``Pi0 <= 0`` and the spin degeneracy is included.

    The implementation uses the exact thermal-smearing identity

    .. math::

        \Pi_\beta(q, i\nu; \mu) =
        \int_0^\infty dE\, [-f'(E-\mu)]\,
        \Pi_{T=0}(q, i\nu; k_F=\sqrt{E/a}).

    This is equivalent to doing the fermionic Matsubara sum analytically and
    keeping the finite-temperature Fermi function in the remaining momentum
    integral.  Reusing :func:`get_Pi0` keeps the ``q=0`` and ``2 k_F`` limits in
    the same stable closed-form path as the zero-temperature Lindhard function.

    The thermal window is integrated with composite ``n_quad``-point
    Gauss-Legendre panels in the variable ``k_F(E) = sqrt(E/a)``, in which the
    ``E = 0`` band edge is analytic.  The window is split at ``k_F(mu)``,
    where the thermal weight is concentrated, and at the Kohn point
    ``k_F = q/2``, where the T=0 kernel has its ``2 k_F`` kink, whenever those
    points lie inside the window.  Each panel receives the full ``n_quad``
    rule, which keeps the fixed-order quadrature accurate both for smooth
    entries and for ``q`` near ``2*k_F`` at zero and small Matsubara
    frequencies.  One-dimensional ``q`` and ``v`` inputs are interpreted as an
    outer ``(nq, nv)`` grid, matching :func:`get_Pi0`.
    If ``finite_T_degenerate_cut`` is not ``None`` and
    ``beta * mu > finite_T_degenerate_cut``, the function returns the
    zero-temperature Lindhard result at the supplied ``mu``.  This avoids
    resolving an extremely narrow thermal window in the strong-degeneracy
    limit.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    n_quad = int(n_quad)
    if n_quad <= 0:
        raise ValueError("n_quad must be positive.")
    tail_cut = float(tail_cut)
    if tail_cut <= 0.0:
        raise ValueError("tail_cut must be positive.")
    if finite_T_degenerate_cut is not None:
        finite_T_degenerate_cut = float(finite_T_degenerate_cut)
        if finite_T_degenerate_cut <= 0.0:
            raise ValueError(
                "finite_T_degenerate_cut must be positive or None.")

    q_eval, v_eval, shape, outer_grid = _pi_shape_args(q, v)
    if np.isinf(beta):
        if mu <= 0.0:
            Pi = np.zeros(shape, dtype=float)
            return float(Pi) if Pi.shape == () else Pi
        return _get_Pi0_shaped(
            q_eval,
            v_eval,
            math.sqrt(float(mu) / a),
            1.0 / (2.0 * a),
            tol,
            outer_grid,
        )
    if (
        finite_T_degenerate_cut is not None
        and beta * float(mu) > finite_T_degenerate_cut
    ):
        return _get_Pi0_shaped(
            q_eval,
            v_eval,
            math.sqrt(float(mu) / a),
            1.0 / (2.0 * a),
            tol,
            outer_grid,
        )

    # Use x = beta * (E - mu), so dE * [-f'(E-mu)] = f(x)*(1-f(x)) dx, then
    # integrate in k = sqrt(E/a), where the E = 0 band edge is analytic.  The
    # omitted Fermi-window tails are bounded by exp(-tail_cut); a feature just
    # outside the window edges therefore carries negligible weight, which is
    # why the strict interior tests below need no margin.
    x0 = max(-beta * float(mu), -tail_cut)
    x1 = tail_cut
    if x0 >= x1:
        Pi = np.zeros(shape, dtype=float)
        return float(Pi) if Pi.shape == () else Pi
    k_lo = math.sqrt(max(float(mu) + x0 / beta, 0.0) / a)
    k_hi = math.sqrt((float(mu) + x1 / beta) / a)

    # Two interior features limit a single fixed-order Gauss-Legendre panel:
    # the thermal weight is concentrated at k_F(mu), and the zero-temperature
    # kernel has a (z-1)*log|z-1| kink (plus a small-|nu| boundary layer) at
    # the Kohn point k = q/2.  The window is split at both points when they
    # lie inside it; entries with an interior Kohn point are evaluated in one
    # scaled call per distinct q.
    if outer_grid:
        q_b, v_b = np.broadcast_arrays(q_eval[:, None], v_eval[None, :])
    else:
        q_b, v_b = np.broadcast_arrays(q_eval, v_eval)
    flat_q = q_b.ravel()
    flat_v = v_b.ravel()
    Pi = np.zeros(shape, dtype=float)
    Pi_flat = Pi.reshape(-1)
    breaks = (math.sqrt(float(mu) / a),) if mu > 0.0 else ()
    half_q = 0.5 * flat_q
    split = (half_q > k_lo) & (half_q < k_hi)
    smooth = ~split
    if np.any(smooth):
        Pi_flat[smooth] = _pi0_maldague_sum(
            flat_q[smooth], flat_v[smooth], beta, mu, a,
            _maldague_panels(k_lo, k_hi, breaks), n_quad, tol)
    for q_val in np.unique(flat_q[split]):
        sel = split & (flat_q == q_val)
        Pi_flat[sel] = _pi0_maldague_q(
            float(q_val), flat_v[sel], beta, mu, a,
            _maldague_panels(k_lo, k_hi, breaks + (0.5 * q_val,)),
            n_quad, tol)
    q0 = np.abs(q_b) <= tol
    if np.any(q0):
        v0 = np.abs(v_b) <= tol
        Pi[q0 & (~v0)] = 0.0
        Pi[q0 & v0] = -_finite_temperature_dn_dmu(beta, mu, a, tail_cut)
    if Pi.shape == ():
        return float(Pi)
    return Pi


def get_Pi(q, v, a, mu, tol=1e-12):
    """Compatibility wrapper for ``Pi0`` using ``eps(k)=a*k**2``."""
    if mu <= 0.0:
        q_arr = np.asarray(q)
        v_arr = np.asarray(v)
        if q_arr.ndim == 1 and v_arr.ndim == 1:
            return np.zeros((q_arr.size, v_arr.size), dtype=float)
        return np.zeros(np.broadcast_shapes(q_arr.shape, v_arr.shape), dtype=float)
    m_eff = 1.0 / (2.0 * a)
    k_F = math.sqrt(mu / a)
    return get_Pi0(q, v, k_F, m_eff, tol=tol)


def get_Vph(v, omega_t, omega_l):
    """Return the phonon dielectric factor ``(v^2+w_T^2)/(v^2+w_L^2)``."""
    v = np.asarray(v, dtype=float)
    return (v * v + omega_t * omega_t) / (v * v + omega_l * omega_l)


def has_multipole_phonon(params: STOParams) -> bool:
    """Return True when ``params`` uses the multipole dielectric model."""
    return bool(params.phonon_to_poles_cm)


def _multipole_freqs_ha(params: STOParams) -> tuple[np.ndarray, np.ndarray]:
    to, lo = _multipole_freqs_cm(params)
    to = to * CM1_TO_EV * EV_TO_HA
    lo = lo * CM1_TO_EV * EV_TO_HA
    return to, lo


def _multipole_to_disp_cm2_a2(params: STOParams) -> np.ndarray:
    nmode = len(params.phonon_to_poles_cm)
    if params.phonon_to_omega2_disp_cm2_a2:
        return np.asarray(params.phonon_to_omega2_disp_cm2_a2, dtype=float)
    return np.zeros(nmode, dtype=float)


def _multipole_to_cap2_cm2(params: STOParams) -> np.ndarray:
    nmode = len(params.phonon_to_poles_cm)
    if params.phonon_to_omega2_cap_cm2:
        return np.asarray(params.phonon_to_omega2_cap_cm2, dtype=float)
    return np.full(nmode, np.inf, dtype=float)


def _multipole_freqs_cm(params: STOParams, q_A=None) -> tuple[np.ndarray, np.ndarray]:
    """Return multipole TO/LO frequencies in cm^-1.

    With ``q_A`` supplied, TO modes use the opt-in squared-frequency
    dispersion.  Finite caps use a smooth rational continuation that preserves
    the small-q slope.  LO zeros stay at their zone-center values.
    """
    to0 = np.asarray(params.phonon_to_poles_cm, dtype=float)
    lo = np.asarray(params.phonon_lo_zeros_cm, dtype=float)
    if q_A is None:
        return to0, lo

    q = np.asarray(q_A, dtype=float).reshape(-1)
    beta = _multipole_to_disp_cm2_a2(params)
    q2 = q[:, None] ** 2
    to02 = to0**2
    to2 = to02[None, :] + q2 * beta[None, :]
    cap2 = _multipole_to_cap2_cm2(params)
    capped = np.isfinite(cap2)
    if np.any(capped):
        delta = cap2[capped] - to02[capped]
        qc2 = delta / beta[capped]
        to2[:, capped] = to02[capped] + delta * q2 / (q2 + qc2)
    if np.any(to2 <= 0.0):
        raise ValueError(
            "multipole TO frequencies must stay positive on this q grid"
        )
    to = np.sqrt(to2)
    if params.phonon_to_omega2_disp_cm2_a2:
        _validate_multipole_q_frequencies(to, lo, params.eps_inf)
    return to, lo


def _validate_multipole_q_frequencies(
    to: np.ndarray, lo: np.ndarray, eps_inf: float
) -> None:
    """Validate the opt-in q-dependent multipole ordering on sampled q rows."""
    if np.any(to >= lo[None, :]):
        raise ValueError(
            "q-dependent multipole TO modes must stay below matching LO zeros"
        )
    if to.shape[1] > 1 and np.any(lo[:-1][None, :] >= to[:, 1:]):
        raise ValueError("q-dependent multipole TO/LO frequencies must stay ordered")
    eps0_q = float(eps_inf) * np.prod((lo[None, :] / to) ** 2, axis=-1)
    if np.any(eps0_q <= float(eps_inf)):
        raise ValueError(
            "q-dependent multipole eps_0(q) must stay greater than eps_inf"
        )


def _multipole_freqs_ha_q(params: STOParams, q_A) -> tuple[np.ndarray, np.ndarray]:
    to, lo = _multipole_freqs_cm(params, q_A=q_A)
    to = to * CM1_TO_EV * EV_TO_HA
    lo = lo * CM1_TO_EV * EV_TO_HA
    return to, lo


def phonon_grid_ha(params: STOParams) -> float:
    """Return the phonon scale used to size driver grids."""
    if has_multipole_phonon(params):
        freqs_cm = params.phonon_to_poles_cm + params.phonon_lo_zeros_cm
        return max(freqs_cm) * CM1_TO_EV * EV_TO_HA
    return params.omega_l_ha


def get_phonon_factor_iw(v, params: STOParams):
    """Return the zone-center lattice factor ``eps_inf / eps(i*nu)``."""
    if not has_multipole_phonon(params):
        return get_Vph(v, params.omega_t_ha, params.omega_l_ha)

    v = np.asarray(v, dtype=float)
    to, lo = _multipole_freqs_ha(params)
    v2 = v[..., None] ** 2
    return np.prod(v2 + to**2, axis=-1) / np.prod(v2 + lo**2, axis=-1)


def _get_multipole_phonon_factor_q_iw(q, v, params: STOParams):
    """Return multipole ``Vph(q, i*nu)`` as an ``(nq, nv)`` table."""
    q_A = np.asarray(q, dtype=float) / BOHR
    v = np.asarray(v, dtype=float)
    to, lo = _multipole_freqs_ha_q(params, q_A)
    v2 = v[None, :, None] ** 2
    num = np.prod(v2 + to[:, None, :] ** 2, axis=-1)
    den = np.prod(v2 + lo[None, None, :] ** 2, axis=-1)
    return num / den


def omega_t_cm(q_A, params):
    """Return the transverse optical mode in cm^-1 for q in Angstrom^-1."""
    q_A = np.asarray(q_A, dtype=float)
    wt = float(params.omega_t0_cm) + float(params.omega_t_disp_cm_a2) * q_A**2
    if np.any(wt <= 0.0):
        raise ValueError("omega_T(q) must be positive.")
    return wt


def eps_0_q(q_A, params):
    """Return the static dielectric constant for the STO phonon model.

    For the dispersive single-pole transverse-mode model this uses
    ``eps_0(q) = eps_0 * (omega_T(0) / omega_T(q))**2``, which keeps the LO
    mode fixed.  The exact no-phonon limit ``eps_0 == eps_inf`` is kept
    constant so ``Vph(q, v)`` is unity even if ``omega_T(q)`` disperses.
    For opt-in multipole q-dependence, the same endpoint is built from the
    product ``eps_inf * prod((LO_j / TO_j(q))**2)``.
    """
    q_A = np.asarray(q_A, dtype=float)
    if has_multipole_phonon(params):
        if not params.phonon_to_omega2_disp_cm2_a2:
            return np.zeros_like(q_A, dtype=float) + float(params.eps_0)
        to, lo = _multipole_freqs_ha_q(params, q_A.reshape(-1))
        eps0 = float(params.eps_inf) * np.prod((lo[None, :] / to) ** 2, axis=-1)
        return eps0.reshape(q_A.shape)

    eps0_base = float(params.eps_0)
    if eps0_base == float(params.eps_inf):
        eps0 = np.zeros_like(q_A, dtype=float) + eps0_base
    else:
        eps0 = eps0_base * (
            float(params.omega_t0_cm) / omega_t_cm(q_A, params)
        ) ** 2
    if np.any(eps0 <= 0.0):
        raise ValueError("eps_0(q) must be positive.")
    if np.any(eps0 < float(params.eps_inf)):
        warnings.warn(
            "eps_0(q) < eps_inf on this q grid: with the fixed-omega_L "
            "convention, omega_T(q) exceeds omega_L(0) there, so "
            "Vph(q, 0) > 1 and the lattice anti-screens the Coulomb "
            "repulsion, which is unphysical for a polar phonon. Reduce "
            "omega_t_disp_cm_a2 or restrict the q domain.",
            RuntimeWarning,
            stacklevel=2,
        )
    return eps0


def get_Vph_q(q, v, params):
    """Return ``Vph(q, v)`` as an ``(nq, nv)`` table.

    ``q`` is in Bohr^-1 and ``v`` is in Hartree, matching the radial GW
    drivers.  For the legacy single-pole model, the transverse-mode dispersion
    coefficient in ``params`` uses Angstrom units, so the unit conversion is
    kept here.  For the multipole model, the returned table is q-independent
    unless ``phonon_to_omega2_disp_cm2_a2`` supplies opt-in TO dispersion.
    """
    if float(params.eps_inf) <= 0.0:
        raise ValueError("eps_inf must be positive.")
    q = np.atleast_1d(np.asarray(q, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    if q.ndim != 1 or v.ndim != 1:
        raise ValueError("q and v must be scalar or one-dimensional.")
    if has_multipole_phonon(params):
        return _get_multipole_phonon_factor_q_iw(q, v, params)

    q_A = q / BOHR
    wt = omega_t_cm(q_A, params) * CM1_TO_EV * EV_TO_HA
    eps0 = eps_0_q(q_A, params)
    wl = wt * np.sqrt(eps0 / float(params.eps_inf))
    return get_Vph(v[None, :], wt[:, None], wl[:, None])


def get_lattice_eps_iw_ev(q_A, omega_eV, params):
    """Return the phonon-only dielectric function on the imaginary axis."""
    q_A, omega_eV = np.broadcast_arrays(
        np.asarray(q_A, dtype=float),
        np.asarray(omega_eV, dtype=float),
    )
    if has_multipole_phonon(params):
        flat_q = q_A.reshape(-1)
        flat_w = omega_eV.reshape(-1) * EV_TO_HA
        to, lo = _multipole_freqs_ha_q(params, flat_q)
        w2 = flat_w[:, None] ** 2
        factor = np.prod(w2 + to**2, axis=-1) / np.prod(w2 + lo**2, axis=-1)
        return (float(params.eps_inf) / factor).reshape(q_A.shape)

    wt = omega_t_cm(q_A, params) * CM1_TO_EV
    eps0 = eps_0_q(q_A, params)
    phonon = (eps0 - float(params.eps_inf)) * wt**2 / (wt**2 + omega_eV**2)
    return float(params.eps_inf) + phonon


def default_nelec_tol(n0, rtol=1e-4, atol=1e-14, cap=1e-10):
    """Return an absolute particle-number tolerance for a target density."""
    if n0 <= 0.0:
        raise ValueError("target density must be positive.")
    return min(float(cap), max(float(atol), float(rtol) * abs(float(n0))))


def fermi_quantities_ev(n_cm3: float, m_eff: float) -> tuple[float, float]:
    """Return ``k_F`` in Angstrom^-1 and ``eps_F`` in eV."""
    n_a3 = float(n_cm3) * 1.0e-24
    k_F = (3.0 * math.pi**2 * n_a3) ** (1.0 / 3.0)
    alpha = HBAR2_OVER_2ME_EV_A2 / float(m_eff)
    return k_F, alpha * k_F**2
