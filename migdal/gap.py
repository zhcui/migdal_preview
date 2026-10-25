"""Shared radial gap eigenproblem for G0W0, GW0, and scGW.

The normal-state drivers provide a radial Green's function and screened
interaction.  This module builds the linearized pairing matvec and delegates
the eigenproblem to PySCF's Davidson solver.  The packed vector stores the
scaled gap variable ``phi_tilde = k * Delta`` on the active Matsubara
frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np
from pyscf import lib

from migdal import memory, log as mlog, radial


@dataclass
class GapConfig:
    """Controls for the radial Matsubara gap eigenproblem.

    Parameters mirror the driver keyword arguments.  ``truncate`` selects an
    active Matsubara window, ``precond`` chooses the Davidson preconditioner,
    and ``cache`` controls whether the radial convolution geometry is
    precomputed for repeated matvecs.
    """

    truncate: bool = False
    wcut: float | None = None
    precond: str = "even_diag"
    cache: str = "auto"
    max_memory: float = memory.DEFAULT_MAX_MEMORY
    davidson_tol: float = 1.0e-6
    davidson_max_cycle: int = 300
    davidson_max_space: int = 20
    nroots: int = 1
    check_precond: bool = False
    check_precond_nsamples: int = 12
    verbose: int = 0
    stdout: object = sys.stdout
    return_gap_function: bool = False


@dataclass
class GapResult:
    """Result of a radial gap solve.

    ``lam`` is the reported pairing eigenvalue ``-eig.real`` for the selected
    Davidson root.  ``phi_tilde`` is returned only when requested through
    ``GapConfig(return_gap_function=True)`` and has the full ``(nk, nw)``
    frequency shape with inactive frequencies filled by zero.
    """

    lam: float
    converged: bool
    root: int
    eig: complex
    n_active_w: int
    used_min_pair: bool
    matvecs: int
    davidson_cycles: int
    phi_tilde: np.ndarray | None = None


def _format_mb(value):
    if value is None:
        return None
    try:
        return "%.6g MB" % float(value)
    except (TypeError, ValueError):
        return None


def get_active_w(ws, wcut, truncate):
    """Return active Matsubara mask, keeping at least one +/- pair."""
    ws = np.asarray(ws, dtype=float)
    if not truncate:
        return np.ones(len(ws), dtype=bool), False
    if wcut is None or not np.isfinite(wcut) or wcut < 0.0:
        raise ValueError("wcut must be a non-negative finite value.")
    active = np.abs(ws) <= float(wcut)
    used_min_pair = False
    if not np.any(active):
        idx = int(np.argmin(np.abs(ws)))
        pair = len(ws) - 1 - idx
        active[idx] = True
        active[pair] = True
        used_min_pair = True
    if not np.array_equal(active, active[::-1]):
        raise ValueError("active_w must be symmetric under omega -> -omega.")
    return active, used_min_pair


def symm_gap(ctx, phi):
    """Project a gap-like array onto reverse-frequency conjugate symmetry."""
    return radial.symm_sigma(ctx, phi)


def pair_G(ctx, G, imag_atol=1e-8, imag_rtol=1e-8):
    """Return the Cooper product ``G(k, iw) G(k, -iw)``."""
    radial.check_w_pair(ctx)
    pair = np.arange(G.shape[1] - 1, -1, -1)
    prod = np.asarray(G) * np.asarray(G)[:, pair]
    imag = radial.max_abs(prod.imag)
    scale = max(radial.max_abs(prod.real), imag, 1.0e-300)
    if imag > imag_atol and imag / scale > imag_rtol:
        raise FloatingPointError(
            "G(iw)G(-iw) has an unexpectedly large imaginary component."
        )
    if not np.all(np.isfinite(prod)):
        raise FloatingPointError("non-finite entries in G(iw)G(-iw).")
    return prod.real


def get_v0(ws_active, nk, nroots, w0):
    """Build simple real-even Davidson initial vectors."""
    ws_active = np.asarray(ws_active, dtype=float)
    if not np.allclose(ws_active, -ws_active[::-1], rtol=0.0, atol=1.0e-10):
        raise ValueError("active Matsubara frequencies are not reverse paired.")
    if nroots < 1:
        raise ValueError("nroots must be positive.")
    w0 = max(float(w0), np.finfo(float).tiny)
    abs_w = np.abs(ws_active)
    scales = w0 * np.asarray([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    shapes = [np.ones_like(ws_active)]
    for scale in scales:
        x2 = (abs_w / scale) ** 2
        shapes.append(1.0 / (1.0 + x2))
    for scale in scales:
        x2 = (abs_w / scale) ** 2
        shapes.append(x2 / (1.0 + x2))

    v0 = []
    q_prev = []
    for shape in shapes:
        vec = np.tile(shape[None, :], (nk, 1)).ravel().astype(complex)
        for q in q_prev:
            vec -= np.vdot(q, vec) * q
        norm = np.linalg.norm(vec)
        if norm <= 1.0e-12:
            continue
        vec /= norm
        v0.append(vec)
        q_prev.append(vec)
        if len(v0) == nroots:
            return v0
    raise ValueError("could not build independent initial gap vectors.")


def _check_precond_diag(precond_diag, active_w, matvec, pack, unpack,
                        nsamples=12, seed=0, tol=1.0e-6):
    active_w = np.asarray(active_w, dtype=bool)
    nk, nw = precond_diag.shape
    pair = np.arange(nw - 1, -1, -1)
    active_half = np.where(active_w[: nw // 2])[0]
    specs = [(ik, iw) for ik in range(nk) for iw in active_half]
    rng = np.random.default_rng(seed)
    nsamples = min(int(nsamples), len(specs))
    if nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    specs = [specs[i] for i in rng.choice(len(specs), size=nsamples, replace=False)]
    diag_probe = []
    max_imag = 0.0
    for ik, iw in specs:
        phi = np.zeros((nk, nw), dtype=complex)
        phi[ik, iw] = 1.0
        phi[ik, pair[iw]] = 1.0
        y = unpack(matvec(pack(phi)))
        max_imag = max(max_imag, radial.max_abs(y.imag))
        diag_probe.append(y[ik, iw].real)
    diag_probe = np.asarray(diag_probe)
    diag_formula = np.asarray([precond_diag[ik, iw] for ik, iw in specs])
    err = radial.max_abs(diag_probe - diag_formula)
    rel = err / max(radial.max_abs(diag_probe), radial.max_abs(diag_formula), 1.0e-300)
    if rel > tol:
        raise AssertionError("even gap preconditioner diagonal check failed.")
    return rel, max_imag


def solve_gap(ctx, k_grid, k_wts, G_pair, W, config=None,
              diag_cells=None, mu_scale=1.0):
    r"""Solve the radial Matsubara gap eigenproblem.

    The packed Davidson vector stores
    ``phi_tilde(k, iw) = k * Delta(k, iw)`` on the active Matsubara
    frequencies.  The radial convolution is requested with ``S_has_k=True`` and
    ``return_scaled=True`` so the matvec returns the same variable.  In compact
    notation the source is

    .. math::

        S(k, i\omega_n) =
        G(k, i\omega_n) G(k, -i\omega_n) \phi_\mathrm{tilde}(k, i\omega_n),

    and the matvec returns ``-Sigma[S]``.  Migdal reports
    ``lambda = -eig.real`` for the selected Davidson root.

    If ``W`` exposes its radial q grid, that grid must cover the active pairing
    domain.  Diagonal cell averages can sample momenta inside a cell, so their
    upper bounds are included in this check.  This keeps extended Pi tails out
    of the gap basis unless the screened interaction was built on a matching q
    domain.
    """
    config = GapConfig() if config is None else config
    log = mlog.new_logger(config)
    k_grid = np.asarray(k_grid, dtype=float)
    k_wts = np.asarray(k_wts, dtype=float)
    if (
        k_grid.ndim != 1
        or k_grid.size == 0
        or k_wts.shape != k_grid.shape
        or not np.all(np.isfinite(k_grid))
        or not np.all(np.isfinite(k_wts))
    ):
        raise ValueError(
            "k_grid and k_wts must be finite one-dimensional arrays "
            "with matching nonzero shape."
        )
    if hasattr(W, "q"):
        q = np.asarray(W.q, dtype=float)
        if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
            raise ValueError("W.q must be a finite one-dimensional q grid.")
        q_max = float(np.max(q))
        required = 2.0 * float(np.max(k_grid))
        if diag_cells is not None:
            _, _, bounds = diag_cells
            bounds = np.asarray(bounds, dtype=float)
            if (
                bounds.shape != (k_grid.size, 2)
                or not np.all(np.isfinite(bounds))
            ):
                raise ValueError("diag_cells bounds are inconsistent with k_grid.")
            required = max(required, float(np.max(k_grid + bounds[:, 1])))
        if q_max < required * (1.0 - 1.0e-10):
            raise ValueError(
                "Gap W q grid does not cover radial convolution: "
                "q_max=%s required=%s" % (q_max, required)
            )
    G_pair_sq = pair_G(ctx, G_pair)
    active_w, used_min_pair = get_active_w(ctx.ws, config.wcut, config.truncate)
    n_active_w = int(np.count_nonzero(active_w))
    nk = len(k_grid)
    nw = ctx.nw

    def unpack(phi_flat):
        phi = np.zeros((nk, nw), dtype=complex)
        phi[:, active_w] = np.asarray(phi_flat).reshape(nk, n_active_w)
        return phi

    def pack(phi):
        return np.asarray(phi[:, active_w]).ravel()

    if config.cache not in ("auto", "off"):
        raise ValueError("cache mode must be 'auto' or 'off'.")
    cache_key = None
    cache_info = {
        "enabled": False,
        "reason": "off",
        "key": None,
        "required_mb": 0.0,
        "effective_limit_mb": 0.0,
    }
    if config.cache != "off":
        cache_key_trial = "migdal-gap-%d-%d-%d" % (id(W), nk, nw)
        cache_info = radial.prepare_cache(
            None, cache_key_trial, W, k_grid, diag_cells=diag_cells,
            mode=config.cache, max_memory=config.max_memory,
        )
        if cache_info["enabled"]:
            cache_key = cache_key_trial
    mlog.kv_table(
        log,
        "Gap cache",
        (
            ("mode", config.cache),
            ("enabled", mlog.format_bool(cache_info["enabled"])),
            ("reason", cache_info.get("reason")),
            ("required/rank", _format_mb(cache_info.get("required_mb"))),
            ("effective limit", _format_mb(cache_info.get("effective_limit_mb"))),
            ("key", cache_key),
        ),
        level=mlog.DEBUG,
    )
    mlog.kv_table(
        log,
        "Gap cache memory detail",
        (
            ("current RSS", _format_mb(cache_info.get("current_rss_mb"))),
            ("max memory", _format_mb(cache_info.get("max_memory_mb"))),
            ("safety factor", cache_info.get("safety")),
            ("user available", _format_mb(cache_info.get("user_available_mb"))),
            ("rank effective limit", _format_mb(
                cache_info.get("rank_effective_limit_mb"))),
            ("runtime source", cache_info.get("runtime_source")),
            ("runtime kind", cache_info.get("runtime_kind")),
            ("runtime value", _format_mb(cache_info.get("runtime_value_mb"))),
            ("runtime available", _format_mb(cache_info.get("runtime_available_mb"))),
            ("local ranks", cache_info.get("local_ranks")),
            ("limiting source", cache_info.get("limiting_source")),
            ("total cache", _format_mb(cache_info.get("total_mb"))),
        ),
        level=mlog.DEBUG1,
    )

    matvec_count = {"n": 0}

    def matvec(phi_flat):
        matvec_count["n"] += 1
        phi = symm_gap(ctx, unpack(phi_flat))
        S = G_pair_sq * phi
        if cache_key is not None:
            out = -radial.get_sigma_cache(
                None, ctx, k_grid, k_wts, S, cache_key,
                S_has_k=True, return_scaled=True,
            )
        elif diag_cells is None:
            out = -radial.get_sigma(
                None, ctx, k_grid, k_wts, S, W,
                S_has_k=True, return_scaled=True, tol=1.0e-10,
            )
        else:
            out = -radial.get_sigma_diag(
                None, ctx, k_grid, k_wts, S, W, diag_cells,
                S_has_k=True, return_scaled=True, tol=1.0e-10,
            )
        return pack(symm_gap(ctx, out))

    dim = nk * n_active_w
    x0 = np.ones(dim, dtype=complex) if config.nroots == 1 else get_v0(
        ctx.ws[active_w], nk, config.nroots, max(float(mu_scale), 1.0 / ctx.beta)
    )

    if config.precond == "ones":
        precond = np.ones(dim, dtype=complex)
    elif config.precond == "even_diag":
        precond_diag = radial.get_precond(
            None, ctx, k_grid, k_wts, G_pair_sq, W, diag_cells=diag_cells)
        precond = precond_diag[:, active_w].ravel()
        if config.check_precond:
            _check_precond_diag(
                precond_diag, active_w, matvec, pack, unpack,
                nsamples=config.check_precond_nsamples,
            )
            matvec_count["n"] = 0
    else:
        raise ValueError("unknown gap preconditioner %r" % config.precond)

    mlog.kv_table(
        log,
        "Gap Davidson setup",
        (
            ("nk / nw / active nw", "%d / %d / %d" % (nk, nw, n_active_w)),
            ("linear dimension", "%d" % dim),
            ("truncate frequencies", mlog.format_bool(config.truncate)),
            ("preconditioner", config.precond),
            ("tolerance", "%.3e" % config.davidson_tol),
            ("max cycle / space", "%d / %d" % (
                config.davidson_max_cycle,
                config.davidson_max_space,
            )),
            ("nroots", "%d" % config.nroots),
        ),
        level=mlog.INFO,
    )
    cycle_fmt = "  %5s %7s %11s %18s %11s %11s %8s"
    mlog.table_header(
        log,
        "Gap Davidson cycles",
        cycle_fmt,
        ("cycle", "space", "|r|", "eig", "max|de|", "lindep", "matvecs"),
        level=mlog.DEBUG,
    )

    callback_state = {"calls": 0, "last_cycle": -1}

    def callback(envs):
        callback_state["calls"] += 1
        callback_state["last_cycle"] = envs.get(
            "icyc", callback_state["calls"] - 1
        )
        e = envs.get("e")
        dx_norm = envs.get("dx_norm")
        de = envs.get("de")
        norm_min = envs.get("norm_min", np.nan)
        if e is None or dx_norm is None or de is None:
            mlog.table_row(
                log,
                "  %5d %7s %11s %18s %11s %11s %8d",
                (
                    callback_state["last_cycle"] + 1,
                    "?",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    matvec_count["n"],
                ),
                level=mlog.DEBUG,
            )
            return
        dx_norm = np.asarray(dx_norm)
        de = np.asarray(de)
        ide = int(envs.get("ide", np.argmax(np.abs(de))))
        eig = np.atleast_1d(e)
        eig_text = "%.8e" % eig[0].real if eig.size else "n/a"
        mlog.table_row(
            log,
            "  %5d %7s %11.3e %18s %11.3e %11.3e %8d",
            (
                callback_state["last_cycle"] + 1,
                envs.get("space", "?"),
                float(np.max(dx_norm)) if dx_norm.size else np.nan,
                eig_text,
                abs(de[ide]) if de.size else np.nan,
                norm_min,
                matvec_count["n"],
            ),
            level=mlog.DEBUG,
        )

    try:
        conv, ew, xroots = lib.davidson_nosym1(
            lambda xs: [matvec(x) for x in xs],
            x0=x0,
            precond=precond,
            tol=config.davidson_tol,
            max_cycle=config.davidson_max_cycle,
            max_space=config.davidson_max_space,
            verbose=mlog.new_logger(
                config, verbose=min(int(config.verbose), mlog.WARN)
            ),
            nroots=config.nroots,
            callback=callback,
        )
    finally:
        if cache_key is not None:
            radial.clear_cache(None, cache_key)

    ew = np.atleast_1d(ew)
    conv = np.atleast_1d(conv)
    root = int(np.argmin(ew.real))
    lam = float((-ew[root]).real)
    phi_tilde = None
    if config.return_gap_function and xroots is not None:
        phi_tilde = symm_gap(ctx, unpack(xroots[root]))
    mlog.kv_table(
        log,
        "Gap Davidson result",
        (
            ("converged", mlog.format_bool(bool(conv[root]))),
            ("lambda", "%.12e" % lam),
            ("selected root", "%d" % root),
            ("eigenvalue", "%.12e%+.12ej" % (
                complex(ew[root]).real,
                complex(ew[root]).imag,
            )),
            ("cycles", "%d" % (callback_state["last_cycle"] + 1)),
            ("matvecs", "%d" % matvec_count["n"]),
            ("active frequencies", "%d" % n_active_w),
        ),
        level=mlog.INFO,
    )
    return GapResult(
        lam=lam,
        converged=bool(conv[root]),
        root=root,
        eig=complex(ew[root]),
        n_active_w=n_active_w,
        used_min_pair=used_min_pair,
        matvecs=matvec_count["n"],
        davidson_cycles=callback_state["last_cycle"] + 1,
        phi_tilde=phi_tilde,
    )


def solve_dense(H):
    """Return the leading eigenvalue of a dense symmetric gap matrix."""
    vals = np.linalg.eigvalsh(np.asarray(H))
    return float(vals[-1])
