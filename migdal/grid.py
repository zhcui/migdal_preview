"""Radial k and q grids for the one-band STO model.

The drivers use separate radial grids because the integrands have different
features: the q grid must resolve the Coulomb/phonon interaction and the
``2*k_F`` Lindhard structure, while the k grid must resolve a thermal shell
around the Fermi surface and optionally extend into a high-k tail for scGW
polarization bubbles.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def _q_trapezoid_weights(q_grid):
    w = np.empty_like(q_grid)
    dq = np.diff(q_grid)
    w[1:-1] = 0.5 * (dq[:-1] + dq[1:])
    w[0] = 0.5 * dq[0]
    w[-1] = 0.5 * dq[-1]
    return w


def _positive_q_min(qmax):
    qeps = qmax * 1.0e-8
    if qeps <= 0.0:
        qeps = np.nextafter(0.0, qmax)
    if 1.0e-12 < qmax:
        qeps = max(qeps, 1.0e-12)
    return min(qeps, np.nextafter(qmax, 0.0))


def _allocate_counts(segments, total):
    mins = np.asarray([seg["min"] for seg in segments], dtype=int)
    weights = np.asarray([seg["weight"] for seg in segments], dtype=float)
    total = max(int(total), int(np.sum(mins)))
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    raw = total * weights / np.sum(weights)
    counts = np.maximum(mins, np.floor(raw).astype(int))
    while int(np.sum(counts)) > total:
        candidates = np.where(counts > mins)[0]
        if candidates.size == 0:
            break
        idx = candidates[int(np.argmax(counts[candidates] - raw[candidates]))]
        counts[idx] -= 1
    while int(np.sum(counts)) < total:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1
    return counts


def _cluster_around(left, center, right, n, power=1.6):
    """Cluster points around center on [left, right), excluding center itself."""
    n = int(n)
    if n <= 0:
        return np.empty(0)
    n_left = (n + 1) // 2
    n_right = n - n_left
    left_u = np.linspace(1.0, 0.0, n_left, endpoint=False)
    q_left = center - (center - left) * left_u**power
    if n_right:
        right_u = np.linspace(0.0, 1.0, n_right + 1, endpoint=False)[1:]
        q_right = center + (right - center) * right_u**power
        return np.concatenate([q_left, q_right])
    return q_left


def _optimized_q_grid(qmax, qeps, k_F, nq, frac_log, n_2kf_min):
    """Return a dimensionless piecewise q grid tuned for radial W primitives."""
    target = int(nq)
    k_F = float(k_F)
    q_star = 2.0 * k_F

    low_log_weight = min(0.26, max(0.12, 0.55 * float(frac_log)))
    weights = {
        "small": 0.50 * low_log_weight,
        "low": 0.50 * low_log_weight,
        "mid": 0.18,
        "pre_kohn": 0.05,
        "kohn": 0.26,
        "post_kohn": 0.05,
        "tail": 0.26,
    }
    kohn_min = min(int(n_2kf_min), max(4, int(round(0.26 * target))))
    mins = {
        "small": 8,
        "low": 8,
        "mid": 12,
        "pre_kohn": 4,
        "kohn": kohn_min,
        "post_kohn": 4,
        "tail": 12,
    }

    segments = []

    def add_segment(name, left, right, kind):
        left = float(max(left, qeps))
        right = float(min(right, qmax))
        if right > left:
            segments.append(
                {
                    "name": name,
                    "left": left,
                    "right": right,
                    "kind": kind,
                    "weight": weights[name],
                    "min": mins[name],
                }
            )

    current = qeps
    for name, ratio, kind in (
        ("small", 0.02, "log"),
        ("low", 0.25, "log"),
        ("mid", 1.65, "linear"),
        ("pre_kohn", 1.80, "linear"),
    ):
        right = min(ratio * k_F, qmax)
        add_segment(name, current, right, kind)
        current = max(current, right)

    has_kohn = qeps < q_star < qmax and qmax > 1.8 * k_F
    if has_kohn:
        left = max(current, 1.8 * k_F)
        right = min(2.2 * k_F, qmax)
        add_segment("kohn", left, right, "cluster")
        current = max(current, right)

    right = min(2.7 * k_F, qmax)
    add_segment("post_kohn", current, right, "linear")
    current = max(current, right)
    add_segment("tail", current, qmax, "log")

    if not segments:
        return np.geomspace(qeps, qmax, max(target, 2))

    # One final point is appended at qmax below; q_star is appended when present.
    protected_extra = 1 if has_kohn else 0
    total_segments = max(target - 1 - protected_extra, 2)
    counts = _allocate_counts(segments, total_segments)
    parts = []
    for seg, count in zip(segments, counts):
        if seg["kind"] == "log":
            parts.append(np.geomspace(seg["left"], seg["right"], int(count), endpoint=False))
        elif seg["kind"] == "cluster":
            parts.append(_cluster_around(seg["left"], q_star, seg["right"], int(count)))
        else:
            parts.append(np.linspace(seg["left"], seg["right"], int(count), endpoint=False))
    if has_kohn:
        parts.append(np.array([q_star]))
    parts.append(np.array([qmax]))
    return np.unique(np.concatenate(parts))


def get_q_grid(
    k_grid_or_max,
    k_F=None,
    nq=1000,
    frac_log=0.35,
    n_2kf_min=101,
):
    """Return a positive q grid and trapezoid weights.

    When ``k_F`` is supplied, the grid uses dimensionless pieces in ``q/k_F``:
    logarithmic small-q panels, linear middle panels, a clustered ``2*k_F``
    panel, and a logarithmic high-q tail.  This keeps fixed point budgets useful
    from dense to dilute densities.  Without ``k_F`` a two-segment log/linear
    split is used.  The returned grid intentionally does not contain q=0 because the
    Coulomb primitive treats that limit analytically.  The ``2*k_F`` refinement
    uses at least ``n_2kf_min`` points when the Kohn feature lies inside the
    domain, capped by the requested budget for very small ``nq`` values.
    """
    k_arr = np.asarray(k_grid_or_max, dtype=float)
    if k_arr.size == 0 or not np.all(np.isfinite(k_arr)):
        raise ValueError("k_grid_or_max must contain finite values.")
    kmax = float(np.max(k_arr))
    if kmax <= 0.0:
        raise ValueError("kmax must be positive.")
    nq = int(nq)
    if nq < 2:
        raise ValueError("nq must be at least 2.")
    frac_log = float(frac_log)
    if not np.isfinite(frac_log) or frac_log < 0.0 or frac_log > 1.0:
        raise ValueError("frac_log must be in [0, 1].")
    n_2kf_min = int(n_2kf_min)
    if n_2kf_min < 0:
        raise ValueError("n_2kf_min must be non-negative.")
    if k_F is not None:
        k_F = float(k_F)
        if not np.isfinite(k_F) or k_F < 0.0:
            raise ValueError("k_F must be non-negative and finite.")

    qmax = (2.0 + 1.0e-10) * kmax
    qeps = _positive_q_min(qmax)

    if k_F is not None and k_F > 0.0:
        q_grid = _optimized_q_grid(qmax, qeps, k_F, nq, frac_log, n_2kf_min)
    else:
        n_log = int(nq * frac_log)
        n_lin = int(nq) - n_log
        q_switch = 0.02 * qmax
        q_switch = min(max(q_switch, min(qeps * 1.0001, qmax)), qmax)

        qlog = np.geomspace(qeps, q_switch, max(n_log, 8), endpoint=False)
        qlin = np.linspace(q_switch, qmax, max(n_lin, 16))
        q_grid = np.unique(np.concatenate([qlog, qlin]))

    q_grid = q_grid[(q_grid > 0.0) & (q_grid <= qmax)]
    if q_grid.size < 2 or not np.all(np.diff(q_grid) > 0.0):
        raise RuntimeError("internal error: q grid is not strictly increasing.")

    return q_grid, _q_trapezoid_weights(q_grid)


def _gl_segment(a, b, n):
    x, w = leggauss(int(n))
    k = 0.5 * (b - a) * (x + 1.0) + a
    wk = 0.5 * (b - a) * w
    return k, wk


def _estimate_vF(k_F, eps, h=1e-6):
    h = max(h, 1.0e-6 * max(1.0, float(k_F)))
    k_F = float(k_F)
    if k_F - h < 0.0:
        return (eps(k_F + h) - eps(k_F)) / h
    return (eps(k_F + h) - eps(k_F - h)) / (2.0 * h)


def get_k_grid(
    k_F,
    T,
    q_grid,
    kmax_sigma,
    eps,
    L=8.0,
    eta=0.10,
    pad_factor=1.5,
    n_pad=20,
    n_left=200,
    n_right=200,
    n_tail=100,
    n_shell_max=2000,
    n_shell_min=300,
):
    """Return a piecewise Gauss-Legendre radial k grid.

    The central shell spans roughly ``k_F +/- L*T/v_F`` and is capped by
    ``n_shell_min``/``n_shell_max``.  Points outside the shell are split into
    low-k, high-k, and optional tail segments.  If ``n_tail > 0``, the grid
    extends to ``kmax_sigma + max(q_grid)`` so that the Pi convolution has
    high-k coverage.  The passive extension receives at least ``n_tail`` Gauss
    points; thermal-shell overlap can add more.  If ``n_tail == 0``, the domain
    is explicitly truncated at ``kmax_sigma``.

    Returns
    -------
    k_all, w_all : ndarray
        Strictly increasing radial momenta and their interval weights.
    info : dict
        Diagnostic quantities such as estimated ``vF``, shell bounds, and
        final grid size.  ``info["nk_active"]`` marks the prefix covering
        exactly ``[0, kmax_sigma]``; any remaining points are passive Pi-tail
        points.  These diagnostics are reported by verbose drivers and
        examples.
    """
    q_grid = np.asarray(q_grid, dtype=float)
    if (
        q_grid.ndim != 1
        or q_grid.size == 0
        or not np.all(np.isfinite(q_grid))
        or np.any(q_grid < 0.0)
    ):
        raise ValueError("q_grid must be a one-dimensional finite non-negative array.")
    q_pos = q_grid[q_grid > 0.0]
    if q_pos.size == 0:
        raise ValueError("q_grid must contain at least one positive point.")
    qmin = float(q_pos.min())
    qmax = float(np.max(q_grid))

    k_F = float(k_F)
    T = float(T)
    kmax_sigma = float(kmax_sigma)
    L = float(L)
    eta = float(eta)
    pad_factor = float(pad_factor)
    if not np.isfinite(k_F) or k_F < 0.0:
        raise ValueError("k_F must be non-negative and finite.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be positive and finite.")
    if not np.isfinite(kmax_sigma) or kmax_sigma <= 0.0:
        raise ValueError("kmax_sigma must be positive and finite.")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be positive and finite.")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be positive and finite.")
    if not np.isfinite(pad_factor) or pad_factor < 0.0:
        raise ValueError("pad_factor must be non-negative and finite.")
    n_tail = int(n_tail)
    if n_tail < 0:
        raise ValueError("n_tail must be non-negative.")
    n_pad = int(n_pad)
    n_left = int(n_left)
    n_right = int(n_right)
    if n_pad < 0 or n_left < 0 or n_right < 0:
        raise ValueError("n_pad, n_left, and n_right must be non-negative.")
    n_shell_min = int(n_shell_min)
    n_shell_max = int(n_shell_max)
    if n_shell_min < 1 or n_shell_max < 1 or n_shell_min > n_shell_max:
        raise ValueError("n_shell_min/max must be positive with min <= max.")

    kmax_pi = kmax_sigma + qmax if n_tail > 0 else kmax_sigma

    vF = float(_estimate_vF(k_F, eps))
    if not np.isfinite(vF) or vF <= 0.0:
        raise ValueError("estimated vF <= 0; check eps(k).")

    w_shell = L * T / vF
    dk_shell = max(eta * T / vF, 1.0e-12)
    k1 = max(0.0, k_F - w_shell)
    k2 = k_F + w_shell
    k_pad = max(pad_factor * qmin, 0.0)

    def add_segment(nodes, weights, a, b, n):
        a = float(a)
        b = float(b)
        if b > a:
            k_seg, w_seg = _gl_segment(a, b, max(int(n), 1))
            nodes.append(k_seg)
            weights.append(w_seg)

    def shell_count(a, b, use_min=True):
        n = int(np.ceil((float(b) - float(a)) / dk_shell))
        if use_min:
            n = max(n, n_shell_min)
        return min(n_shell_max, max(n, 1))

    def build_active_grid():
        """Build the Sigma/gap domain before any passive Pi-tail points."""
        nodes = []
        weights = []
        shell_l_active = min(max(k1, 0.0), kmax_sigma)
        shell_r_active = min(max(k2, 0.0), kmax_sigma)
        has_shell_active = shell_r_active > shell_l_active

        pre_shell_r = shell_l_active if has_shell_active else kmax_sigma
        pad_r = min(k_pad, pre_shell_r)
        add_segment(nodes, weights, 0.0, pad_r, max(n_pad, 4))
        add_segment(nodes, weights, pad_r, pre_shell_r, max(n_left, 8))

        if has_shell_active:
            add_segment(
                nodes,
                weights,
                shell_l_active,
                shell_r_active,
                shell_count(shell_l_active, shell_r_active, use_min=True),
            )

        post_shell_l = shell_r_active if has_shell_active else pre_shell_r
        add_segment(nodes, weights, post_shell_l, kmax_sigma, max(n_right, 8))
        return nodes, weights

    def allocate_counts(total, lengths):
        """Distribute a minimum point budget across positive-length segments."""
        lengths = np.asarray(lengths, dtype=float)
        counts = np.zeros(len(lengths), dtype=int)
        active = np.where(lengths > 0.0)[0]
        if active.size == 0:
            return counts
        total = max(int(total), int(active.size))
        raw = total * lengths[active] / np.sum(lengths[active])
        base = np.maximum(1, np.floor(raw).astype(int))
        while int(np.sum(base)) > total:
            candidates = np.where(base > 1)[0]
            if candidates.size == 0:
                break
            j = candidates[int(np.argmax(base[candidates] - raw[candidates]))]
            base[j] -= 1
        while int(np.sum(base)) < total:
            j = int(np.argmax(raw - base))
            base[j] += 1
        counts[active] = base
        return counts

    k_nodes, w_nodes = build_active_grid()

    if n_tail > 0:
        passive_shell_l = max(kmax_sigma, k1)
        passive_shell_r = min(kmax_pi, k2)
        has_passive_shell = passive_shell_r > passive_shell_l
        if has_passive_shell:
            passive_segments = (
                (kmax_sigma, passive_shell_l, False),
                (passive_shell_l, passive_shell_r, True),
                (passive_shell_r, kmax_pi, False),
            )
            counts = allocate_counts(
                n_tail, [right - left for left, right, _ in passive_segments]
            )
            for idx, (left, right, is_shell) in enumerate(passive_segments):
                n_seg = counts[idx]
                if is_shell:
                    n_seg = max(
                        n_seg,
                        shell_count(left, right, use_min=False),
                    )
                add_segment(k_nodes, w_nodes, left, right, n_seg)
        else:
            add_segment(k_nodes, w_nodes, kmax_sigma, kmax_pi, n_tail)

    k_all = np.concatenate(k_nodes) if k_nodes else np.array([0.0])
    w_all = np.concatenate(w_nodes) if w_nodes else np.array([0.0])
    if k_all.size > 1 and not np.all(np.diff(k_all) > 0.0):
        raise RuntimeError("internal error: k grid is not strictly increasing.")
    atol = 1.0e-12 * max(1.0, abs(kmax_pi))
    if not np.isclose(w_all.sum(), kmax_pi, rtol=1.0e-10, atol=atol):
        raise RuntimeError("internal error: k weights do not cover the domain.")
    n_active = int(np.count_nonzero(k_all <= kmax_sigma))
    if n_active == 0 or not np.all(k_all[:n_active] <= kmax_sigma):
        raise RuntimeError("internal error: active k grid is not a leading block.")
    if np.any(k_all[n_active:] <= kmax_sigma):
        raise RuntimeError("internal error: passive k grid overlaps active domain.")
    if not np.isclose(w_all[:n_active].sum(), kmax_sigma, rtol=1.0e-10, atol=atol):
        raise RuntimeError("internal error: active k weights do not cover the domain.")

    shell_l = min(max(k1, 0.0), kmax_pi)
    shell_r = min(max(k2, 0.0), kmax_pi)
    has_shell = shell_r > shell_l
    if has_shell:
        k_shell = k_all[(k_all >= shell_l) & (k_all <= shell_r)]
        dk_min_shell = np.min(np.diff(k_shell)) if k_shell.size > 1 else np.nan
    else:
        dk_min_shell = np.nan
    info = {
        "vF": vF,
        "k_shell": (k1, k2),
        "k_shell_used": (shell_l, shell_r),
        "dk_shell_target": dk_shell,
        "dk_shell_min": dk_min_shell,
        "k_pad": k_pad,
        "kmax_pi": kmax_pi,
        "kmax_sigma": kmax_sigma,
        "nk": k_all.size,
        "nk_active": n_active,
    }
    return k_all, w_all, info
