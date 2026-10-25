"""Build and validate sparse-IR HDF5 cache files.

The production Migdal grids use dense matrices sampled from sparse-IR basis
functions.  This module keeps the basis construction on the modern
``sparse_ir`` backend, but avoids the current default Matsubara sampling
regression at very large ``lambda`` by using a v1.1.7-compatible high-frequency
search grid.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy import linalg as la


# sparse-ir 2.1.1 / pylibsparseir 0.8.4 can panic in basis_new when
# physical beta or wmax crosses this power-of-two-scale boundary.  Build the
# same-lambda basis at a safer reference scale and rescale arrays afterward.
SPARSE_IR_BASIS_SCALE_LIMIT = 2.0**19

IR_DATASETS = (
    "F_t_f",
    "F_t_b",
    "F_w_f",
    "F_w_b",
    "F_t_beta",
    "F_t_f2b",
    "F_t_b2f",
    "ts_f",
    "ts_b",
    "ws",
    "vs",
    "idx_v0",
    "idx_w0",
)


@dataclass
class SparseIRCacheData:
    """Sparse-IR arrays ready to attach to :class:`migdal.freq.FreqGrid`.

    Parameters
    ----------
    beta : float
        Inverse temperature in Hartree^-1.
    lamb : float
        Dimensionless cutoff ``beta * wmax``.
    wmax : float
        Real-frequency cutoff in Hartree.
    tol : float
        Requested sparse-IR truncation accuracy.
    tau_beta_rtol : float
        Relative offset from ``tau=beta`` used for ``F_t_beta``.
    arrays : dict
        HDF5-compatible arrays with keys from ``IR_DATASETS``.
    metadata : dict
        Nonessential provenance and timing metadata.
    """

    beta: float
    lamb: float
    wmax: float
    tol: float
    tau_beta_rtol: float
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _basis_scale_limit(value=None):
    """Return the sparse-ir physical-axis scale limit used for basis builds."""
    if value is None:
        value = os.environ.get("MIGDAL_SPARSE_IR_BASIS_SCALE_LIMIT")
    if value is None:
        return SPARSE_IR_BASIS_SCALE_LIMIT
    limit = float(value)
    if limit <= 0.0:
        raise ValueError("basis_scale_limit must be positive.")
    return limit


def _select_basis_beta(beta, lamb, limit):
    """Choose a numerically safe beta for sparse-ir basis construction."""
    wmax = lamb / beta
    if beta <= limit and wmax <= limit:
        return beta, "direct"

    lower = lamb / limit
    upper = limit
    if lower <= upper:
        return min(max(beta, lower), upper), "scaled_axis_limit"

    # No beta can make both beta_ref and wmax_ref=lambda/beta_ref fit under the
    # observed sparse-ir physical-axis limit.  Let the caller try the requested
    # scale and report the backend error if it fails.
    return beta, "direct_no_safe_scaled_beta"


def _scale_tau_arrays(arrays, beta_from, beta_to):
    """Scale dense sparse-IR arrays from one beta to another at fixed lambda."""
    beta_from = float(beta_from)
    beta_to = float(beta_to)
    if beta_from <= 0.0 or beta_to <= 0.0:
        raise ValueError("beta_from and beta_to must be positive.")

    scale = beta_to / beta_from
    if scale == 1.0:
        return arrays

    scaled = dict(arrays)
    tau_scale = scale
    freq_scale = 1.0 / scale
    tau_matrix_scale = math.sqrt(beta_from / beta_to)
    freq_matrix_scale = math.sqrt(beta_to / beta_from)

    for key in ("ts_f", "ts_b"):
        scaled[key] = np.asarray(arrays[key]) * tau_scale
    for key in ("ws", "vs"):
        scaled[key] = np.asarray(arrays[key]) * freq_scale
    for key in ("F_t_f", "F_t_b", "F_t_beta", "F_t_f2b", "F_t_b2f"):
        scaled[key] = np.asarray(arrays[key]) * tau_matrix_scale
    for key in ("F_w_f", "F_w_b"):
        scaled[key] = np.asarray(arrays[key]) * freq_matrix_scale
    return scaled


def _matsubara_search_grid(max_power=34):
    """Return the v1.1.7-style integer search grid for Matsubara roots."""
    if max_power < 6:
        raise ValueError("max_power must be at least 6.")
    tail = (2 ** np.linspace(6, max_power, 16 * (max_power - 6) + 1)).astype(
        np.int64
    )
    return np.hstack((np.arange(2**6, dtype=np.int64), tail))


def _symmetrize_matsubara(n):
    """Mirror non-negative reduced Matsubara indices to negative indices."""
    n = np.asarray(n, dtype=np.int64)
    if n.size == 0:
        return n
    if np.any(n[1:] < n[:-1]):
        raise ValueError("Matsubara points must be sorted before symmetrization.")
    if n[0] < 0:
        raise ValueError("Matsubara points must be non-negative.")
    if n[0] == 0:
        return np.hstack((-n[::-1], n[1:]))
    return np.hstack((-n[::-1], n))


def _part_values(func, x, zeta, part):
    vals = func(2 * np.asarray(x, dtype=np.int64) + int(zeta))
    if part == "real":
        return np.asarray(vals.real, dtype=float)
    if part == "imag":
        return np.asarray(vals.imag, dtype=float)
    raise ValueError("part must be 'real' or 'imag'.")


def _bisect_discrete_roots(func, a, b, fa, zeta, part):
    """Vectorized integer bisection matching sparse-ir v1.1.7."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    fa = np.asarray(fa, dtype=float)

    while True:
        mid = (a + b) // 2
        found = a == mid
        if found.any():
            break

        fmid = _part_values(func, mid, zeta, part)
        towards_a = np.signbit(fa) != np.signbit(fmid)
        a = np.where(towards_a, a, mid)
        fa = np.where(towards_a, fa, fmid)
        b = np.where(towards_a, mid, b)

    roots = mid[found]
    if found.all():
        return roots
    more = _bisect_discrete_roots(
        func, a[~found], b[~found], fa[~found], zeta, part
    )
    return np.hstack((roots, more))


def _find_sign_changes(func, xgrid, zeta, part):
    """Find sign changes of one Matsubara basis function on ``xgrid``."""
    fx = _part_values(func, xgrid, zeta, part)
    hit = fx == 0.0
    x_hit = xgrid[hit]

    sign_change = np.signbit(fx[:-1]) != np.signbit(fx[1:])
    sign_change &= ~hit[:-1] & ~hit[1:]
    if not sign_change.any():
        return x_hit.astype(np.int64)

    where_a = np.hstack((sign_change, False))
    where_b = np.hstack((False, sign_change))
    roots = _bisect_discrete_roots(
        func, xgrid[where_a], xgrid[where_b], fx[where_a], zeta, part
    )
    return np.sort(np.hstack((x_hit, roots))).astype(np.int64)


def _bisect_discrete_extremum(func, a, b, absf_a, absf_b, zeta, part):
    """Integer extremum bisection matching sparse-ir v1.1.7."""
    d = int(b - a)
    if d <= 1:
        return int(a if absf_a > absf_b else b)
    if d == 2:
        return int(a + 1)

    m = (a + b) // 2
    n = m + 1
    absf_m = abs(_part_values(func, np.asarray([m]), zeta, part)[0])
    absf_n = abs(_part_values(func, np.asarray([n]), zeta, part)[0])
    if absf_m > absf_n:
        return _bisect_discrete_extremum(func, a, n, absf_a, absf_n, zeta, part)
    return _bisect_discrete_extremum(func, m, b, absf_m, absf_b, zeta, part)


def _find_discrete_extrema(func, xgrid, zeta, part):
    """Find discrete extrema of one Matsubara basis function."""
    fx = _part_values(func, xgrid, zeta, part)
    absfx = np.abs(fx)
    gx = fx[1:] - fx[:-1]
    sgx = np.signbit(gx)
    where = sgx[:-1] != sgx[1:]
    where_a = np.hstack((where, False, False))
    where_b = np.hstack((False, False, where))

    res = [
        _bisect_discrete_extremum(func, a, b, aa, ab, zeta, part)
        for a, b, aa, ab in zip(
            xgrid[where_a], xgrid[where_b], absfx[where_a], absfx[where_b]
        )
    ]

    sfx = np.signbit(fx)
    if absfx[0] > absfx[1] or sfx[0] != sfx[1]:
        res.insert(0, int(xgrid[0]))
    if absfx[-1] > absfx[-2] or sfx[-1] != sfx[-2]:
        res.append(int(xgrid[-1]))
    return np.asarray(res, dtype=np.int64)


def _get_uhat_full(basis):
    """Return current sparse-ir's full Matsubara basis function vector."""
    from ctypes import byref, c_int

    from pylibsparseir.core import _lib
    from sparse_ir.poly import FunctionSetFT, PiecewiseLegendrePolyFTVector

    status = c_int()
    ptr = _lib.spir_basis_get_uhat_full(basis._ptr, byref(status))
    if status.value != 0:
        raise RuntimeError("failed to get sparse-ir uhat_full: %s" % status.value)
    return PiecewiseLegendrePolyFTVector(FunctionSetFT(ptr))


def _candidate_matsubara_points(func, xgrid, zeta, expected, positive_only):
    """Return candidate point sets for real and imaginary components."""
    candidates = []
    for part in ("real", "imag"):
        x0 = _find_sign_changes(func, xgrid, zeta, part)
        wn = 2 * x0 + int(zeta)
        if not positive_only:
            wn = _symmetrize_matsubara(wn)
        candidates.append((wn.astype(np.int64), part, "sign_changes"))

    valid = [c for c in candidates if c[0].size >= expected]
    if valid:
        return min(valid, key=lambda c: c[0].size)
    return max(candidates, key=lambda c: c[0].size)


def select_matsubara_points_v117(
    basis,
    statistics,
    *,
    max_power=34,
    positive_only=False,
):
    """Return v1.1.7-compatible Matsubara sampling indices.

    Parameters
    ----------
    basis : sparse_ir.FiniteTempBasis
        Current sparse-ir finite-temperature basis.
    statistics : {"F", "B"}
        Fermionic or bosonic statistics.
    max_power : int
        Upper search-grid exponent.  ``2**34`` gives the reference production
        grid with the current sparse-ir backend; higher values can pick extra
        ultra-high-frequency sign changes for fermions.
    positive_only : bool
        Return only non-negative frequencies when true.

    Returns
    -------
    wn : (nw,) ndarray of int64
        Reduced Matsubara indices.  Physical frequencies are
        ``wn * pi / beta``.
    info : dict
        Selection metadata.

    Notes
    -----
    The current sparse-ir wrapper does not expose the parity label used by
    v1.1.7 to choose real vs imaginary sign changes.  We therefore evaluate
    both components and pick the smallest candidate that is not underdetermined.
    This preserves the sparse-IR sampling contract without silently truncating
    high-frequency roots.
    """
    if statistics not in ("F", "B"):
        raise ValueError("statistics must be 'F' or 'B'.")
    if basis.size <= 0:
        raise ValueError("basis size must be positive.")

    zeta = 1 if statistics == "F" else 0
    requested = int(basis.size)
    if statistics == "F" and requested % 2 == 1:
        requested += 1
    elif statistics == "B" and requested % 2 == 0:
        requested += 1
    expected = (requested + 1) // 2 if positive_only else requested

    uhat_full = _get_uhat_full(basis)
    xgrid = _matsubara_search_grid(max_power=max_power)
    if requested < uhat_full._funcs.size():
        func = uhat_full[requested]
        wn, part, source = _candidate_matsubara_points(
            func, xgrid, zeta, expected, positive_only
        )
    else:
        func = uhat_full[-1]
        candidates = []
        for part in ("real", "imag"):
            x0 = _find_discrete_extrema(func, xgrid, zeta, part)
            wn_part = 2 * x0 + zeta
            if not positive_only:
                wn_part = _symmetrize_matsubara(wn_part)
            candidates.append((wn_part.astype(np.int64), part, "extrema"))
        valid = [c for c in candidates if c[0].size >= expected]
        wn, part, source = (
            min(valid, key=lambda c: c[0].size)
            if valid
            else max(candidates, key=lambda c: c[0].size)
        )

    if statistics == "B" and not positive_only:
        wn = np.unique(np.hstack((np.asarray([0], dtype=np.int64), wn))).astype(
            np.int64
        )
    if wn.size < expected:
        raise ValueError(
            "v1.1.7-compatible Matsubara sampling is underdetermined: "
            "statistics=%s requested=%s expected=%s got=%s"
            % (statistics, requested, expected, wn.size)
        )

    return wn, {
        "part": part,
        "source": source,
        "requested": requested,
        "expected": expected,
        "max_power": max_power,
        "n_points": int(wn.size),
        "n_min": int(wn[0]) if wn.size else None,
        "n_max": int(wn[-1]) if wn.size else None,
    }


def _load_template_sampling(path, beta, lamb):
    """Load sampling points from an existing sparse-IR HDF5 file."""
    import h5py

    path = Path(path)
    required = ("_beta", "lamb", "ts_f", "ts_b", "ws", "vs")
    with h5py.File(path, "r") as f:
        missing = [key for key in required if key not in f]
        if missing:
            raise ValueError("template IR file %s is missing keys %s" % (path, missing))
        template_beta = float(np.asarray(f["_beta"]))
        template_lamb = float(np.asarray(f["lamb"]))
        if abs(template_lamb - lamb) > max(1.0e-3, 1.0e-12 * abs(lamb)):
            raise ValueError(
                "template IR file %s lambda mismatch: file=%s requested=%s"
                % (path, template_lamb, lamb)
            )
        wn_f = np.rint(np.asarray(f["ws"]) * template_beta / np.pi).astype(np.int64)
        wn_b = np.rint(np.asarray(f["vs"]) * template_beta / np.pi).astype(np.int64)
        ts_f = np.asarray(f["ts_f"], dtype=float) * (beta / template_beta)
        ts_b = np.asarray(f["ts_b"], dtype=float) * (beta / template_beta)

    return {
        "ts_f": ts_f,
        "ts_b": ts_b,
        "wn_f": wn_f,
        "wn_b": wn_b,
        "template_beta": template_beta,
        "template_lamb": template_lamb,
        "template": str(path),
    }


def _default_tau_points(basis_f, basis_b):
    import sparse_ir

    return {
        "ts_f": np.asarray(sparse_ir.TauSampling(basis_f).tau, dtype=float),
        "ts_b": np.asarray(sparse_ir.TauSampling(basis_b).tau, dtype=float),
    }


def _builtin_or_v117_matsubara_points(
    basis,
    statistics,
    max_power,
    builtin_lambda_max,
):
    import sparse_ir

    lamb = float(basis.lambda_)
    if lamb > builtin_lambda_max:
        wn, info = select_matsubara_points_v117(
            basis, statistics, max_power=max_power
        )
        info = {
            **info,
            "algorithm": "v1_1_7_compatible",
            "fallback_reason": "lambda_exceeds_sparse_ir_builtin_window",
            "builtin_lambda_max": float(builtin_lambda_max),
        }
        return wn, info

    wn_builtin = np.asarray(sparse_ir.MatsubaraSampling(basis).wn, dtype=np.int64)
    if wn_builtin.size >= basis.size:
        return wn_builtin, {
            "algorithm": "sparse_ir_builtin",
            "requested": int(basis.size),
            "n_points": int(wn_builtin.size),
            "n_min": int(wn_builtin[0]) if wn_builtin.size else None,
            "n_max": int(wn_builtin[-1]) if wn_builtin.size else None,
        }

    wn, info = select_matsubara_points_v117(
        basis, statistics, max_power=max_power
    )
    info = {
        **info,
        "algorithm": "v1_1_7_compatible",
        "fallback_reason": "sparse_ir_builtin_underdetermined",
        "builtin_n_points": int(wn_builtin.size),
    }
    return wn, info


def _default_matsubara_points(basis_f, basis_b, max_power, builtin_lambda_max):
    wn_f, info_f = _builtin_or_v117_matsubara_points(
        basis_f, "F", max_power, builtin_lambda_max
    )
    wn_b, info_b = _builtin_or_v117_matsubara_points(
        basis_b, "B", max_power, builtin_lambda_max
    )
    return {"wn_f": wn_f, "wn_b": wn_b, "info_f": info_f, "info_b": info_b}


def _matsubara_matrices(basis_f, basis_b, wn_f, wn_b, *, parallel=True):
    if not parallel:
        return basis_f.uhat(wn_f).T, basis_b.uhat(wn_b).T, "serial"

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_f = pool.submit(lambda: basis_f.uhat(wn_f).T)
        future_b = pool.submit(lambda: basis_b.uhat(wn_b).T)
        return future_f.result(), future_b.result(), "threaded"


def _condition_or_inf(mat):
    if mat.shape[0] < mat.shape[1]:
        return np.inf
    try:
        return float(np.linalg.cond(mat))
    except np.linalg.LinAlgError:
        return np.inf


def validate_sparse_ir_arrays(data, *, max_cond=1.0e12):
    """Validate shapes, finiteness, zero frequency, and conditioning."""
    a = data.arrays
    for key in IR_DATASETS:
        if key not in a:
            raise ValueError("missing sparse-IR array %s" % key)
        if not np.all(np.isfinite(np.asarray(a[key]))):
            raise ValueError("sparse-IR array %s contains non-finite values" % key)

    nbf = a["F_t_f"].shape[1]
    nbb = a["F_t_b"].shape[1]
    expected = {
        "F_t_f": (a["ts_f"].size, nbf),
        "F_t_b": (a["ts_b"].size, nbb),
        "F_t_f2b": (a["ts_b"].size, nbf),
        "F_t_b2f": (a["ts_f"].size, nbb),
        "F_t_beta": (1, nbf),
        "F_w_f": (a["ws"].size, nbf),
        "F_w_b": (a["vs"].size, nbb),
    }
    for key, shape in expected.items():
        if a[key].shape != shape:
            raise ValueError("%s has shape %s, expected %s" % (key, a[key].shape, shape))

    if a["F_w_f"].shape[0] < a["F_w_f"].shape[1]:
        raise ValueError("F_w_f is underdetermined: shape=%s" % (a["F_w_f"].shape,))
    if a["F_w_b"].shape[0] < a["F_w_b"].shape[1]:
        raise ValueError("F_w_b is underdetermined: shape=%s" % (a["F_w_b"].shape,))

    idx_v0 = np.atleast_1d(a["idx_v0"]).astype(int)
    if idx_v0.size == 0 or not np.isclose(a["vs"][idx_v0[0]], 0.0, atol=1.0e-14):
        raise ValueError("bosonic Matsubara grid does not contain zero frequency")

    conds = {
        "cond_F_t_f": _condition_or_inf(a["F_t_f"]),
        "cond_F_t_b": _condition_or_inf(a["F_t_b"]),
        "cond_F_w_f": _condition_or_inf(a["F_w_f"]),
        "cond_F_w_b": _condition_or_inf(a["F_w_b"]),
    }
    for key, val in conds.items():
        if not np.isfinite(val) or val > max_cond:
            raise ValueError("%s=%s exceeds max_cond=%s" % (key, val, max_cond))
    data.metadata.update(conds)
    return data


def build_sparse_ir_arrays(
    beta,
    lamb,
    *,
    tol=1.0e-14,
    tau_beta_rtol=0.0,
    template=None,
    max_power=34,
    builtin_lambda_max=1.0e7,
    basis_scale_limit=None,
    validate=True,
    parallel_matsubara=True,
):
    """Build sparse-IR matrices with the official sparse-ir backend.

    The returned arrays use Migdal's dense transform convention: matrices have
    shape ``(n_points, n_basis)`` and Matsubara frequencies are stored as
    physical Hartree values ``wn * pi / beta``.
    """
    beta = float(beta)
    lamb = float(lamb)
    tol = float(tol)
    tau_beta_rtol = float(tau_beta_rtol)
    builtin_lambda_max = float(builtin_lambda_max)
    basis_scale_limit = _basis_scale_limit(basis_scale_limit)
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if lamb <= 0.0:
        raise ValueError("lamb must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if not 0.0 <= tau_beta_rtol < 1.0:
        raise ValueError("tau_beta_rtol must be in [0, 1).")
    if builtin_lambda_max <= 0.0:
        raise ValueError("builtin_lambda_max must be positive.")

    timings = {}
    t0 = time.perf_counter()
    import sparse_ir

    wmax = lamb / beta
    basis_beta, basis_scale_reason = _select_basis_beta(
        beta, lamb, basis_scale_limit
    )
    basis_wmax = lamb / basis_beta
    basis_scaled = basis_beta != beta
    kernel = sparse_ir.LogisticKernel(lamb)
    timings["kernel_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sve = sparse_ir.compute_sve(kernel, eps=tol)
    timings["sve_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        basis_f = sparse_ir.FiniteTempBasis(
            "F", basis_beta, basis_wmax, eps=tol, kernel=kernel, sve_result=sve
        )
        basis_b = sparse_ir.FiniteTempBasis(
            "B", basis_beta, basis_wmax, eps=tol, kernel=kernel, sve_result=sve
        )
    except RuntimeError as err:
        if basis_scaled or math.sqrt(lamb) > basis_scale_limit:
            raise RuntimeError(
                "sparse-ir failed to construct a finite-temperature basis for "
                "beta=%s, wmax=%s, lambda=%s; attempted basis_beta=%s and "
                "basis_wmax=%s.  Try reducing lambda, using a same-lambda IR "
                "template, or increasing MIGDAL_SPARSE_IR_BASIS_SCALE_LIMIT "
                "if the installed sparse-ir backend supports larger axes."
                % (beta, wmax, lamb, basis_beta, basis_wmax)
            ) from err
        basis_beta = math.sqrt(lamb)
        basis_wmax = lamb / basis_beta
        basis_scaled = basis_beta != beta
        basis_scale_reason = "retry_after_basis_failure"
        basis_f = sparse_ir.FiniteTempBasis(
            "F", basis_beta, basis_wmax, eps=tol, kernel=kernel, sve_result=sve
        )
        basis_b = sparse_ir.FiniteTempBasis(
            "B", basis_beta, basis_wmax, eps=tol, kernel=kernel, sve_result=sve
        )
    timings["basis_sec"] = time.perf_counter() - t0

    if template is None:
        t0 = time.perf_counter()
        tau = _default_tau_points(basis_f, basis_b)
        timings["tau_sampling_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mats = _default_matsubara_points(
            basis_f, basis_b, max_power, builtin_lambda_max
        )
        timings["matsubara_sampling_sec"] = time.perf_counter() - t0
        algorithms = {
            mats["info_f"]["algorithm"],
            mats["info_b"]["algorithm"],
        }
        sampling_algorithm = (
            "sparse_ir_builtin"
            if algorithms == {"sparse_ir_builtin"}
            else "sparse_ir_builtin_with_v117_fallback"
        )
        sampling_metadata = {
            "sampling_algorithm": sampling_algorithm,
            "matsubara_grid_max_power": int(max_power),
            "sparse_ir_builtin_lambda_max": float(builtin_lambda_max),
            "fermion_sampling": mats["info_f"],
            "boson_sampling": mats["info_b"],
        }
    else:
        t0 = time.perf_counter()
        sampling = _load_template_sampling(template, basis_beta, lamb)
        timings["template_load_sec"] = time.perf_counter() - t0
        tau = {"ts_f": sampling["ts_f"], "ts_b": sampling["ts_b"]}
        mats = {"wn_f": sampling["wn_f"], "wn_b": sampling["wn_b"]}
        sampling_metadata = {
            "sampling_algorithm": "template",
            "template": sampling["template"],
            "template_beta": sampling["template_beta"],
            "template_lamb": sampling["template_lamb"],
        }

    ts_f = np.asarray(tau["ts_f"], dtype=float)
    ts_b = np.asarray(tau["ts_b"], dtype=float)
    wn_f = np.asarray(mats["wn_f"], dtype=np.int64)
    wn_b = np.asarray(mats["wn_b"], dtype=np.int64)

    t0 = time.perf_counter()
    F_t_f = basis_f.u(ts_f).T
    F_t_b = basis_b.u(ts_b).T
    F_t_f2b = basis_f.u(ts_b).T
    F_t_b2f = basis_b.u(ts_f).T
    tau_beta = basis_beta * (1.0 - tau_beta_rtol)
    F_t_beta = np.atleast_2d(basis_f.u([tau_beta]).T)
    timings["tau_matrix_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    F_w_f, F_w_b, matsubara_matrix_mode = _matsubara_matrices(
        basis_f, basis_b, wn_f, wn_b, parallel=parallel_matsubara
    )
    timings["matsubara_matrix_sec"] = time.perf_counter() - t0

    ws = wn_f * (np.pi / basis_beta)
    vs = wn_b * (np.pi / basis_beta)
    arrays = {
        "F_t_f": np.asarray(F_t_f),
        "F_t_b": np.asarray(F_t_b),
        "F_w_f": np.asarray(F_w_f),
        "F_w_b": np.asarray(F_w_b),
        "F_t_beta": np.asarray(F_t_beta),
        "F_t_f2b": np.asarray(F_t_f2b),
        "F_t_b2f": np.asarray(F_t_b2f),
        "ts_f": ts_f,
        "ts_b": ts_b,
        "ws": np.asarray(ws, dtype=float),
        "vs": np.asarray(vs, dtype=float),
        "idx_v0": np.where(vs == 0.0)[0].astype(int),
        "idx_w0": np.asarray(int(np.argmin(np.abs(ws))), dtype=int),
    }
    arrays = _scale_tau_arrays(arrays, basis_beta, beta)
    arrays["idx_v0"] = np.where(arrays["vs"] == 0.0)[0].astype(int)
    arrays["idx_w0"] = np.asarray(
        int(np.argmin(np.abs(arrays["ws"]))), dtype=int
    )
    ws = arrays["ws"]
    vs = arrays["vs"]
    metadata_dict = {
        "generator": "migdal.ir_cache",
        "backend": "official",
        "mpi_mode": "serial",
        "sparse_ir_version": _package_version("sparse-ir"),
        "pylibsparseir_version": _package_version("pylibsparseir"),
        "basis_f_size": int(basis_f.size),
        "basis_b_size": int(basis_b.size),
        "basis_build_beta": float(basis_beta),
        "basis_build_wmax": float(basis_wmax),
        "basis_scaled_to_target_beta": bool(basis_scaled),
        "basis_scale_factor": float(beta / basis_beta),
        "basis_scale_limit": float(basis_scale_limit),
        "basis_scale_reason": basis_scale_reason,
        "nw": int(ws.size),
        "nv": int(vs.size),
        "matsubara_matrix_mode": matsubara_matrix_mode,
        **sampling_metadata,
        **timings,
    }
    data = SparseIRCacheData(
        beta=beta,
        lamb=lamb,
        wmax=wmax,
        tol=tol,
        tau_beta_rtol=tau_beta_rtol,
        arrays=arrays,
        metadata=metadata_dict,
    )
    if validate:
        validate_sparse_ir_arrays(data)
    return data


def compare_to_template(data, template, *, rtol=1.0e-12, atol=1.0e-12):
    """Compare generated core matrices to an existing HDF5 template."""
    import h5py

    keys = ("F_t_f", "F_t_b", "F_w_f", "F_w_b", "F_t_beta", "F_t_f2b", "F_t_b2f")
    with h5py.File(template, "r") as f:
        for key in keys:
            if key not in f:
                raise ValueError("template IR file is missing key %s" % key)
            ref = np.asarray(f[key])
            val = np.asarray(data.arrays[key])
            if val.shape != ref.shape:
                raise ValueError(
                    "template comparison shape mismatch for %s: %s != %s"
                    % (key, val.shape, ref.shape)
                )
            np.testing.assert_allclose(val, ref, rtol=rtol, atol=atol)


def write_scaled_sparse_ir_h5(template, path, beta, *, overwrite=False):
    """Write a same-``lambda`` sparse-IR cache scaled to a new ``beta``.

    For fixed ``lambda = beta * wmax``, the IR basis is unchanged in reduced
    coordinates.  A temperature change therefore rescales physical tau and
    Matsubara points plus the dense transform matrices' normalization factors,
    without changing the represented kernel.
    """
    import h5py

    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    template = Path(template)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass overwrite=True" % path)

    tau_mats = ("F_t_f", "F_t_b", "F_t_beta", "F_t_f2b", "F_t_b2f")
    freq_mats = ("F_w_f", "F_w_b")
    tau_invs = ("F_t_f_inv", "F_t_b_inv")
    freq_invs = ("F_w_f_inv", "F_w_b_inv")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name("%s.tmp.%d" % (path.name, os.getpid()))
    with h5py.File(template, "r") as src, h5py.File(tmp_path, "w") as dst:
        beta0 = float(np.asarray(src["_beta"]))
        if beta0 <= 0.0:
            raise ValueError("template beta must be positive.")
        lamb = float(np.asarray(src["lamb"]))
        scale_tau = beta / beta0
        scale_freq = beta0 / beta
        scale_tau_mat = math.sqrt(beta0 / beta)
        scale_freq_mat = math.sqrt(beta / beta0)

        for key, value in src.attrs.items():
            dst.attrs[key] = value
        basis_beta = float(src.attrs.get("basis_build_beta", beta0))
        if basis_beta <= 0.0:
            raise ValueError("template basis_build_beta must be positive.")
        basis_wmax = lamb / basis_beta
        basis_scaled = abs(beta - basis_beta) > max(1.0e-12, 1.0e-12 * abs(beta))
        dst.attrs["scaled_from_template"] = str(template)
        dst.attrs["template_beta"] = beta0
        dst.attrs["template_lamb"] = lamb
        dst.attrs["basis_build_beta"] = basis_beta
        dst.attrs["basis_build_wmax"] = basis_wmax
        dst.attrs["basis_scaled_to_target_beta"] = bool(basis_scaled)
        dst.attrs["basis_scale_factor"] = float(beta / basis_beta)
        dst.attrs["basis_scale_reason"] = "scaled_h5_template"

        for key in src.keys():
            val = np.asarray(src[key])
            if key == "_beta":
                val = beta
            elif key == "wmax":
                val = lamb / beta
            elif key in ("ts_f", "ts_b"):
                val = val * scale_tau
            elif key in ("ws", "vs"):
                val = val * scale_freq
            elif key in tau_mats:
                val = val * scale_tau_mat
            elif key in freq_mats:
                val = val * scale_freq_mat
            elif key in tau_invs:
                val = val / scale_tau_mat
            elif key in freq_invs:
                val = val / scale_freq_mat
            dst[key] = val
    tmp_path.replace(path)


def write_sparse_ir_h5(data, path, *, overwrite=False, include_pinv=True):
    """Write sparse-IR arrays to the HDF5 schema used by :class:`FreqGrid`."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass overwrite=True" % path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import h5py

    arrays = data.arrays
    with h5py.File(path, "w") as f:
        for key in IR_DATASETS:
            f[key] = np.asarray(arrays[key])
        f["_beta"] = data.beta
        f["lamb"] = data.lamb
        f["wmax"] = data.wmax
        f["tol"] = data.tol
        f["tau_beta_rtol"] = data.tau_beta_rtol
        if include_pinv:
            f["F_t_f_inv"] = la.pinv(arrays["F_t_f"])
            f["F_t_b_inv"] = la.pinv(arrays["F_t_b"])
            f["F_w_f_inv"] = la.pinv(arrays["F_w_f"])
            f["F_w_b_inv"] = la.pinv(arrays["F_w_b"])
        for key, value in data.metadata.items():
            try:
                f.attrs[key] = value
            except TypeError:
                f.attrs[key] = repr(value)


def _mpi_info(enabled):
    if not enabled:
        return None, 0, 1
    try:
        from mpi4py import MPI
    except ImportError as err:
        raise RuntimeError("--mpi requires mpi4py") from err
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--lamb", type=float, default=1.0e9)
    parser.add_argument("--tol", type=float, default=1.0e-14)
    parser.add_argument("--tau-beta-rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "Use only MPI rank 0 to build/write the cache, then synchronize and "
            "check file visibility on all ranks. This is not a performance path."
        ),
    )
    parser.add_argument("--max-power", type=int, default=34)
    parser.add_argument("--builtin-lambda-max", type=float, default=1.0e7)
    parser.add_argument(
        "--basis-scale-limit",
        type=float,
        default=None,
        help=(
            "Physical-axis magnitude limit for direct sparse-ir basis builds. "
            "When beta or wmax exceeds this limit, Migdal builds the same "
            "lambda basis at a balanced reference scale and rescales arrays."
        ),
    )
    return parser


def main(argv=None):
    """Command-line entry point for sparse-IR cache generation and validation."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is set")
    comm, rank, size = _mpi_info(args.mpi)

    ok = True
    message = ""
    if rank == 0:
        try:
            data = build_sparse_ir_arrays(
                args.beta,
                args.lamb,
                tol=args.tol,
                tau_beta_rtol=args.tau_beta_rtol,
                template=args.template,
                max_power=args.max_power,
                builtin_lambda_max=args.builtin_lambda_max,
                basis_scale_limit=args.basis_scale_limit,
            )
            if comm is not None:
                data.metadata["cli_mpi_size"] = int(size)
                data.metadata["cli_mpi_mode"] = "rank0"
            if args.template is not None:
                compare_to_template(data, args.template)
            if not args.validate_only:
                write_sparse_ir_h5(data, args.output, overwrite=args.overwrite)
            timings = {
                key: value
                for key, value in data.metadata.items()
                if "_sec" in key
            }
            message = (
                "sparse-IR cache %s: nw=%s nv=%s timings=%s"
                % (
                    "validated" if args.validate_only else "written",
                    data.arrays["ws"].size,
                    data.arrays["vs"].size,
                    timings,
                )
            )
        except Exception as err:  # CLI boundary: report and broadcast failure.
            ok = False
            message = "%s: %s" % (type(err).__name__, err)

    if comm is not None:
        states = comm.allgather((ok, message))
        failures = [
            "rank %d: %s" % (i, msg)
            for i, (rank_ok, msg) in enumerate(states)
            if not rank_ok
        ]
        if failures:
            raise RuntimeError("; ".join(failures))
        if not args.validate_only:
            comm.Barrier()
            if not args.output.exists():
                raise RuntimeError("rank %s cannot see output %s" % (rank, args.output))
    elif not ok:
        raise RuntimeError(message)

    if rank == 0:
        print(message, flush=True)


if __name__ == "__main__":
    main()
