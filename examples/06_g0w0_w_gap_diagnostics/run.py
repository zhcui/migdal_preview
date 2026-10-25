#!/usr/bin/env python

"""Plot G0W0 q-averaged interactions and gap-function diagnostics.

The plotted interaction is the fixed G0W0 RPA table

    W0(q, i nu) = Vc(q) Vph(q, i nu) /
                  (1 - Pi0(q, i nu) Vc(q) Vph(q, i nu)).

The corresponding unscreened interaction is
``V_tot(q, i nu)=Vc(q)Vph(q, i nu)``.
The default q average uses the trapezoid weights returned by the current radial
q grid, i.e. ``sum_q w_q X(q, i nu) / sum_q w_q``.  A 3D volume average is
available as an explicit comparison option.  The gap-function diagnostics can
write both no-truncation and truncated Matsubara-window views from one script.

Run from the repository root:

    python examples/06_g0w0_w_gap_diagnostics/run.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from pyscf.data.nist import BOHR, HARTREE2EV


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402
from migdal import grid, memory, model  # noqa: E402
from migdal.freq import FreqGrid  # noqa: E402


DENSITIES_CM3 = (1.0e17, 4.0e20, 1.0e23)
GAP_MODE_ALIASES = {
    "no_trunc": "no_trunc",
    "notrunc": "no_trunc",
    "no-trunc": "no_trunc",
    "full": "no_trunc",
    "truncated": "truncated",
    "truncate": "truncated",
    "trunc": "truncated",
}
HA_TO_MEV = HARTREE2EV * 1.0e3
W_AU_TO_EV_A3 = HARTREE2EV * BOHR**3
BOHR_INV_TO_A_INV = 1.0 / BOHR
ENERGY_SCALE_KEYS = ("E_F", "phonon_scale", "E_plasmon", "E_TF")
ENERGY_SCALE_STYLES = {
    "E_F": (r"$E_{\mathrm{F}}$", "#1f77b4", (0, (4.0, 1.8))),
    "phonon_scale": (r"$\omega_{\mathrm{ph,max}}$", "#d62728", (0, (2.0, 1.4))),
    "E_plasmon": (r"$E_{\mathrm{pl}}$", "#2ca02c", (0, (6.0, 2.0))),
    "E_TF": (r"$E_{\mathrm{TF}}$", "#9467bd", (0, (1.0, 1.4))),
}


def default_ir_file():
    """Return the preferred sparse-IR cache path for this plotting example."""
    candidates = (
        SCRIPT_DIR / "ir.h5",
        SCRIPT_DIR.parent / "04_g0w0_gw0_scgw_curve" / "ir.5",
        SCRIPT_DIR.parent / "ir.h5",
        REPO_ROOT / "examples" / "ir.h5",
    )
    for path in candidates:
        if path.exists():
            return path
    return SCRIPT_DIR / "ir.h5"


def parse_gap_modes(text):
    """Parse and de-duplicate requested G0W0 gap-function views."""
    modes = []
    for part in str(text).replace(";", ",").split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in GAP_MODE_ALIASES:
            raise ValueError("unknown gap mode %r; choose no_trunc or truncated" % part)
        mode = GAP_MODE_ALIASES[key]
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("--gap-modes must name at least one mode")
    return tuple(modes)


def add_run_args(parser):
    """Add user-facing run controls."""
    group = parser.add_argument_group("run")
    group.add_argument(
        "--densities",
        default=",".join("%.12g" % n for n in DENSITIES_CM3),
        help="Comma-separated densities in cm^-3.",
    )
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--lamb", type=float, default=1.0e9)
    group.add_argument("--ir-file", default=str(default_ir_file()))
    group.add_argument("--save-ir", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument(
        "--gap-modes",
        default="no_trunc,truncated",
        help="Comma-separated gap views to write: no_trunc, truncated.",
    )
    group.add_argument(
        "--gap-truncate",
        action="store_true",
        help="Shortcut for --gap-modes truncated.",
    )
    common.add_output_arg(group, __file__)


def add_radial_grid_args(parser):
    """Add radial momentum-grid controls used by W and gap diagnostics."""
    group = parser.add_argument_group("radial grid")
    group.add_argument("--nq", type=int, default=500)
    group.add_argument("--kmax-factor", type=float, default=4.0)
    group.add_argument("--kmax-lo-factor", type=float, default=2.0)
    group.add_argument("--qmax-factor", type=float, default=2.0)
    group.add_argument("--k-pad", type=int, default=5)
    group.add_argument("--k-left", type=int, default=200)
    group.add_argument("--k-right", type=int, default=200)
    group.add_argument("--k-shell-min", type=int, default=50)
    group.add_argument("--k-shell-max", type=int, default=5000)
    group.add_argument("--gap-diag-ncell", type=int, default=12)


def add_gap_solver_args(parser):
    """Add Davidson solver controls for the leading gap eigenfunction."""
    group = parser.add_argument_group("gap solver")
    group.add_argument("--davidson-tol", type=float, default=1.0e-6)
    group.add_argument("--davidson-max-cycle", type=int, default=300)
    group.add_argument("--davidson-max-space", type=int, default=20)
    group.add_argument("--precond", choices=("even_diag", "ones"), default="even_diag")
    group.add_argument("--cache", choices=("auto", "off"), default="auto")
    group.add_argument(
        "--max-memory",
        type=float,
        default=memory.DEFAULT_MAX_MEMORY,
        help="PySCF-style per-rank memory limit in MB.",
    )
    group.add_argument("--gap-davidson-tol", dest="davidson_tol", type=float, help=argparse.SUPPRESS)
    group.add_argument(
        "--gap-davidson-max-cycle",
        dest="davidson_max_cycle",
        type=int,
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--gap-davidson-max-space",
        dest="davidson_max_space",
        type=int,
        help=argparse.SUPPRESS,
    )
    group.add_argument("--gap-precond", dest="precond", choices=("even_diag", "ones"), help=argparse.SUPPRESS)
    group.add_argument("--gap-cache", dest="cache", choices=("auto", "off"), help=argparse.SUPPRESS)


def add_plot_args(parser):
    """Add plotting and averaging controls."""
    group = parser.add_argument_group("plot")
    group.add_argument("--xmax-scale", type=float, default=4.0)
    group.add_argument(
        "--average-measure",
        choices=("volume", "dq"),
        default="dq",
        help="Use q-grid trapezoid weights by default; volume uses q^2 w_q.",
    )
    group.add_argument(
        "--gap-average-measure",
        choices=("dk", "volume"),
        default="dk",
        help="Use k-grid weights for the gap average by default; volume uses k^2 w_k.",
    )


def parse_args(argv=None):
    """Parse q-weighted interaction and gap-function plot arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_args(parser)
    add_radial_grid_args(parser)
    add_gap_solver_args(parser)
    add_plot_args(parser)
    args = parser.parse_args(argv)
    try:
        args.gap_modes = ("truncated",) if args.gap_truncate else parse_gap_modes(args.gap_modes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def density_domain(ctx, params, n_cm3, args):
    """Return density-dependent scales and radial cutoffs in atomic units."""
    a = params.band_a_ha_bohr2
    n_bohr3 = model.n_cm3_to_bohr3(float(n_cm3))
    mu0 = model.get_mu(n_bohr3, beta=np.inf, a=a)
    mu = model.get_mu(n_bohr3, beta=ctx.beta, a=a)
    k_F = math.sqrt(mu0 / a)
    k_ph = math.sqrt((max(mu0, 0.0) + model.phonon_grid_ha(params)) / a)
    kmax = max(args.kmax_factor * k_F, args.kmax_lo_factor * k_ph)
    return {
        "a": a,
        "n_cm3": float(n_cm3),
        "n_bohr3": n_bohr3,
        "mu": mu,
        "mu0": mu0,
        "k_F": k_F,
        "kmax": kmax,
        "qmax_target": args.qmax_factor * kmax,
    }


def make_q_grid(domain, args):
    """Build the diagnostic q grid from the density-dependent cutoff."""
    return grid.get_q_grid(
        0.5 * domain["qmax_target"],
        k_F=domain["k_F"],
        nq=args.nq,
        frac_log=0.35,
    )


def q_weighted_w(ctx, params, n_cm3, args):
    """Return q-weighted screened W0, unscreened Vtot, and diagnostics."""
    domain = density_domain(ctx, params, n_cm3, args)
    q_grid, q_wts = make_q_grid(domain, args)

    Vc = model.get_Vc(q_grid, params.eps_inf, q0_treat=False, tol=1.0e-10)
    Vph = model.get_Vph_q(q_grid, ctx.vs, params)
    Vtot = Vc[:, None] * Vph
    Pi0 = model.get_Pi(q_grid, ctx.vs, a=domain["a"], mu=domain["mu0"], tol=1.0e-10)
    denom = 1.0 - Pi0 * Vtot
    if not np.all(np.isfinite(denom)):
        raise FloatingPointError("non-finite entries in the G0W0 RPA denominator.")
    W = Vtot / denom
    if not np.all(np.isfinite(W)):
        raise FloatingPointError("non-finite entries in the G0W0 W table.")

    avg_wts = q_wts
    if args.average_measure == "volume":
        avg_wts = avg_wts * q_grid**2
    q_wt_norm = np.sum(avg_wts)
    W_avg = np.tensordot(avg_wts, W, axes=(0, 0)) / q_wt_norm
    Vtot_avg = np.tensordot(avg_wts, Vtot, axes=(0, 0)) / q_wt_norm
    scales = energy_scales(params, domain["n_bohr3"], domain["k_F"], domain["mu0"])
    diag = {
        "n_cm3": domain["n_cm3"],
        "n_bohr3": domain["n_bohr3"],
        "mu": domain["mu"],
        "mu0": domain["mu0"],
        "k_F": domain["k_F"],
        "qmax": float(np.max(q_grid)),
        "nq": len(q_grid),
        "average_measure": args.average_measure,
        **scales,
    }
    return np.asarray(W_avg.real), np.asarray(Vtot_avg.real), diag


def g0w0_gap_function(ctx, params, n_cm3, args):
    """Return the normalized k-averaged G0W0 leading gap eigenfunction."""
    from migdal import gap, radial

    domain = density_domain(ctx, params, n_cm3, args)
    q_grid, _ = make_q_grid(domain, args)
    k_grid, k_wts, _ = grid.get_k_grid(
        domain["k_F"],
        1.0 / ctx.beta,
        q_grid,
        domain["kmax"],
        lambda k: domain["a"] * k**2 - domain["mu0"],
        n_pad=args.k_pad,
        n_left=args.k_left,
        n_right=args.k_right,
        n_tail=0,
        n_shell_min=args.k_shell_min,
        n_shell_max=args.k_shell_max,
    )
    e_k = domain["a"] * k_grid**2
    G0 = ctx.get_G0(e_k=e_k, mu=domain["mu"])
    Vc = model.get_Vc(q_grid, params.eps_inf, q0_treat=False, tol=1.0e-10)
    Vph = model.get_Vph_q(q_grid, ctx.vs, params)
    Vtot = Vc[:, None] * Vph
    Pi0 = model.get_Pi(q_grid, ctx.vs, a=domain["a"], mu=domain["mu0"], tol=1.0e-10)
    W, denom = radial.get_W_from_Pi(
        ctx,
        Pi0,
        q_grid,
        Vc,
        Vtot,
        denom_policy="legacy_clip",
        denom_min=1.0e-10,
        return_metrics=True,
    )
    diag_cells = radial.get_diag_cells(
        None, W, k_grid, k_wts, domain["kmax"], args.gap_diag_ncell
    )
    wcut = max(domain["mu0"], domain["mu"])
    gres = gap.solve_gap(
        ctx,
        k_grid,
        k_wts,
        G0,
        W,
        config=gap.GapConfig(
            truncate=bool(args.gap_truncate),
            wcut=wcut,
            precond=args.precond,
            cache=args.cache,
            max_memory=args.max_memory,
            davidson_tol=args.davidson_tol,
            davidson_max_cycle=args.davidson_max_cycle,
            davidson_max_space=args.davidson_max_space,
            return_gap_function=True,
        ),
        diag_cells=diag_cells,
        mu_scale=domain["mu0"],
    )
    if gres.phi_tilde is None:
        raise RuntimeError("gap solver did not return a gap eigenfunction.")

    delta = gres.phi_tilde / k_grid[:, None]
    avg_wts = k_wts
    if args.gap_average_measure == "volume":
        avg_wts = avg_wts * k_grid**2
    delta_avg = np.tensordot(avg_wts, delta, axes=(0, 0)) / np.sum(avg_wts)
    pos = np.asarray(ctx.ws) > 0.0
    omega_mev = np.asarray(ctx.ws)[pos] * HA_TO_MEV
    gap_vals = np.asarray(delta_avg[pos])
    ref = int(np.argmax(np.abs(gap_vals)))
    if abs(gap_vals[ref]) > 0.0:
        gap_vals = gap_vals * np.exp(-1j * np.angle(gap_vals[ref]))
        if gap_vals[ref].real < 0.0:
            gap_vals = -gap_vals
    scale = max(float(np.max(np.abs(gap_vals.real))), 1.0e-300)
    gap_norm = gap_vals.real / scale
    gap_imag_norm = gap_vals.imag / scale
    diag = {
        "n_cm3": domain["n_cm3"],
        "k_F": domain["k_F"],
        "kmax": domain["kmax"],
        "qmax": float(np.max(q_grid)),
        "nk": len(k_grid),
        "nq": len(q_grid),
        "lambda": gres.lam,
        "gap_converged": gres.converged,
        "gap_cycles": gres.davidson_cycles,
        "gap_truncate": bool(args.gap_truncate),
        "gap_n_active_w": gres.n_active_w,
        "gap_used_min_pair": gres.used_min_pair,
        "gap_average_measure": args.gap_average_measure,
        "gap_imag_max": float(np.max(np.abs(gap_imag_norm))),
        "denom_min_abs": denom.min_abs,
        **energy_scales(params, domain["n_bohr3"], domain["k_F"], domain["mu0"]),
    }
    return omega_mev, gap_norm, gap_imag_norm, diag


def energy_scales(params, n_bohr3, k_F, e_F):
    """Return selected scales in Hartree using eps_inf screening conventions."""
    m_eff = float(params.m_eff)
    q_tf2 = 4.0 * m_eff * float(k_F) / (math.pi * float(params.eps_inf))
    return {
        "E_F": float(e_F),
        "phonon_scale": model.phonon_grid_ha(params),
        "E_plasmon": math.sqrt(4.0 * math.pi * float(n_bohr3) / (m_eff * params.eps_inf)),
        "E_TF": q_tf2 / (2.0 * m_eff),
    }


def positive_frequency_view(vs, W_avg):
    """Return non-negative bosonic frequencies sorted from zero upward."""
    vs = np.asarray(vs, dtype=float)
    W_avg = np.asarray(W_avg, dtype=float)
    mask = vs >= -1.0e-14
    order = np.argsort(vs[mask])
    return vs[mask][order] * HA_TO_MEV, W_avg[mask][order] * W_AU_TO_EV_A3


def density_label(n_cm3):
    """Return a compact plot label for a carrier density."""
    exp = int(math.floor(math.log10(float(n_cm3))))
    mant = float(n_cm3) / 10.0**exp
    if abs(mant - round(mant)) < 1.0e-12:
        mant_text = "%d" % round(mant)
    else:
        mant_text = "%.3g" % mant
    return r"$n=%s\times10^{%d}\,\mathrm{cm}^{-3}$" % (mant_text, exp)


def math_number(value, precision=3):
    """Format a number for Matplotlib mathtext."""
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    sign = r"{-}" if value < 0.0 else ""
    if 1.0e-2 <= abs_value < 1.0e3:
        return "%s%.*g" % (sign, precision, abs_value)
    exp = int(math.floor(math.log10(abs_value)))
    mant = abs_value / 10.0**exp
    return r"%s%.*g\times10^{%d}" % (sign, precision, mant, exp)


def mev_text(value):
    """Return a mathtext-formatted meV value."""
    return r"%s\,\mathrm{meV}" % math_number(value)


def ainv_text(value):
    """Return a mathtext-formatted inverse-Angstrom value."""
    return r"%s\,\AA^{-1}" % math_number(value, precision=4)


def average_label(measure, coordinate):
    """Return the TeX label for the diagnostic averaging measure."""
    if measure == "volume":
        return r"%s^2 w_%s" % (coordinate, coordinate)
    return r"w_%s" % coordinate


def energy_scale_text(diag):
    """Return a compact TeX row with the energy scales in meV."""
    return (
        r"$E_{\mathrm{F}}=%s,\ \omega_{\mathrm{ph,max}}=%s,\ "
        r"E_{\mathrm{pl}}=%s,\ E_{\mathrm{TF}}=%s$"
        % tuple(mev_text(diag[key] * HA_TO_MEV) for key in ENERGY_SCALE_KEYS)
    )


def interaction_note_text(diag):
    """Return the below-panel interaction diagnostic text."""
    return "\n".join(
        [
            energy_scale_text(diag),
            (
                r"$k_{\mathrm{F}}=%s,\ q_{\max}=%s,\ N_q=%d,\ "
                r"\langle\cdot\rangle_q:\ %s$"
                % (
                    ainv_text(diag["k_F"] * BOHR_INV_TO_A_INV),
                    ainv_text(diag["qmax"] * BOHR_INV_TO_A_INV),
                    diag["nq"],
                    average_label(diag["average_measure"], "q"),
                )
            ),
        ]
    )


def gap_note_text(diag):
    """Return the below-panel gap-function diagnostic text."""
    return "\n".join(
        [
            energy_scale_text(diag),
            (
                r"$\lambda=%s,\ \mathrm{converged}=%s,\ N_k=%d,\ "
                r"\langle\cdot\rangle_k:\ %s$"
                % (
                    math_number(diag["lambda"], precision=4),
                    r"\mathrm{yes}" if diag["gap_converged"] else r"\mathrm{no}",
                    diag["nk"],
                    average_label(diag["gap_average_measure"], "k"),
                )
            ),
            (
                r"$k_{\mathrm{F}}=%s,\ k_{\max}=%s$"
                % (
                    ainv_text(diag["k_F"] * BOHR_INV_TO_A_INV),
                    ainv_text(diag["kmax"] * BOHR_INV_TO_A_INV),
                )
            ),
        ]
    )


def make_diagnostic_figure(plt, n_panels, title):
    """Create plot axes plus a separate text row for panel diagnostics."""
    fig_width = max(5.2 * n_panels, 6.0)
    fig = plt.figure(figsize=(fig_width, 6.0), constrained_layout=True)
    grid_spec = fig.add_gridspec(
        3,
        n_panels,
        height_ratios=(0.09, 1.0, 0.2),
    )
    legend_ax = fig.add_subplot(grid_spec[0, :])
    axes = np.array([fig.add_subplot(grid_spec[1, idx]) for idx in range(n_panels)])
    note_axes = np.array(
        [fig.add_subplot(grid_spec[2, idx]) for idx in range(n_panels)]
    )
    fig.set_constrained_layout_pads(h_pad=0.04, w_pad=0.04, hspace=0.02, wspace=0.08)
    fig.suptitle(title, fontsize=13)
    legend_ax.axis("off")
    for note_ax in note_axes:
        note_ax.axis("off")
    return fig, legend_ax, axes, note_axes


def add_energy_scale_legend(legend_ax):
    """Add one legend for all energy-scale guide lines."""
    from matplotlib.lines import Line2D

    handles = []
    for key in ENERGY_SCALE_KEYS:
        label, color, linestyle = ENERGY_SCALE_STYLES[key]
        handles.append(Line2D([0], [0], color=color, lw=1.1, linestyle=linestyle))
    labels = [ENERGY_SCALE_STYLES[key][0] for key in ENERGY_SCALE_KEYS]
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=2.3,
        columnspacing=1.7,
    )


def draw_energy_scale_lines(ax, diag):
    """Draw non-overlapping energy-scale guide lines in one panel."""
    x0, x1 = ax.get_xlim()
    for key in ENERGY_SCALE_KEYS:
        _, color, linestyle = ENERGY_SCALE_STYLES[key]
        val = diag[key] * HA_TO_MEV
        if x0 <= val <= x1:
            ax.axvline(
                val,
                color=color,
                lw=1.0,
                linestyle=linestyle,
                alpha=0.85,
                zorder=0,
            )


def write_panel_note(note_ax, text):
    """Write diagnostic text outside the data region."""
    note_ax.text(
        0.5,
        0.98,
        text,
        transform=note_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        linespacing=1.35,
    )


def common_limits(datasets, y_keys):
    """Return shared x/y limits across all requested interaction curves."""
    xmax = 10.0
    for item in datasets:
        diag = item["diag"]
        scale_vals_mev = np.array([diag[k] for k in ENERGY_SCALE_KEYS]) * HA_TO_MEV
        xmax = max(xmax, item["xmax_scale"] * float(np.max(scale_vals_mev)))
    y_blocks = []
    for item in datasets:
        visible = np.asarray(item["nu_mev"]) <= xmax
        for key in y_keys:
            vals = np.asarray(item[key])[visible]
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size:
                y_blocks.append(vals)
    if not y_blocks:
        raise ValueError("no positive finite interaction values to plot.")
    all_y = np.concatenate(y_blocks)
    ymin = 10.0 ** np.floor(np.log10(np.min(all_y)))
    ymax = 10.0 ** np.ceil(np.log10(np.max(all_y)))
    if ymin == ymax:
        ymin *= 0.1
        ymax *= 10.0
    return (0.0, xmax), (ymin, ymax)


def plot_interaction(plt, datasets, y_key, ylabel, title, out_dir, stem, limits):
    """Plot one q-averaged interaction family and return CSV rows."""
    fig, legend_ax, axes, note_axes = make_diagnostic_figure(plt, len(datasets), title)
    add_energy_scale_legend(legend_ax)
    xlim, ylim = limits
    csv_rows = []
    for ax, note_ax, item in zip(axes, note_axes, datasets):
        diag = item["diag"]
        x_mev = item["nu_mev"]
        y_ev_a3 = item[y_key]
        visible = x_mev <= xlim[1]
        ax.plot(
            x_mev[visible],
            y_ev_a3[visible],
            marker="o",
            ms=3.0,
            lw=1.2,
            color="#222222",
        )
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\nu_n$ [meV]")
        ax.set_title(density_label(diag["n_cm3"]))
        ax.grid(True, which="both", alpha=0.25)
        draw_energy_scale_lines(ax, diag)
        write_panel_note(note_ax, interaction_note_text(diag))
        for nu, yval in zip(x_mev, y_ev_a3):
            csv_rows.append((diag["n_cm3"], nu, yval))
    axes[0].set_ylabel(ylabel)
    fig.savefig(out_dir / ("%s.png" % stem), dpi=220)
    fig.savefig(out_dir / ("%s.pdf" % stem))
    plt.close(fig)
    return csv_rows


def plot_gap_function(
    plt,
    datasets,
    out_dir,
    stem="g0w0_gap_function",
    title=r"$G_0W_0$ k-averaged leading gap function",
):
    """Plot normalized k-averaged leading G0W0 gap functions."""
    fig, legend_ax, axes, note_axes = make_diagnostic_figure(plt, len(datasets), title)
    add_energy_scale_legend(legend_ax)
    xmax = 10.0
    for item in datasets:
        diag = item["gap_diag"]
        scale_vals_mev = np.array([diag[k] for k in ENERGY_SCALE_KEYS]) * HA_TO_MEV
        xmax = max(xmax, item["xmax_scale"] * float(np.max(scale_vals_mev)))
    ymin = min(-1.05, min(float(np.min(item["gap_norm"])) for item in datasets) * 1.05)
    ymax = max(1.05, max(float(np.max(item["gap_norm"])) for item in datasets) * 1.05)
    rows = []
    for ax, note_ax, item in zip(axes, note_axes, datasets):
        diag = item["gap_diag"]
        x_mev = item["gap_omega_mev"]
        y = item["gap_norm"]
        visible = x_mev <= xmax
        ax.plot(
            x_mev[visible],
            y[visible],
            marker="o",
            ms=3.0,
            lw=1.2,
            color="#222222",
        )
        ax.axhline(0.0, color="0.55", lw=0.8)
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$\omega_n$ [meV]")
        ax.set_title(density_label(diag["n_cm3"]))
        ax.grid(True, which="both", alpha=0.25)
        draw_energy_scale_lines(ax, diag)
        write_panel_note(note_ax, gap_note_text(diag))
        for omega, val, imag_val in zip(x_mev, y, item["gap_imag_norm"]):
            rows.append((diag["n_cm3"], omega, val, imag_val))
    axes[0].set_ylabel(r"normalized $\langle \Delta(k,\mathrm{i}\omega_n)\rangle_k$")
    fig.savefig(out_dir / ("%s.png" % stem), dpi=220)
    fig.savefig(out_dir / ("%s.pdf" % stem))
    plt.close(fig)
    return rows


def write_outputs(out_dir, w_rows, vtot_rows, scales):
    """Write screened/unscreened interaction CSV files and summary text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_dir / "g0w0_qavg_w.csv",
        np.asarray(w_rows, dtype=float),
        delimiter=",",
        header="n_cm3,nu_meV,W_qavg_eV_A3",
        comments="",
    )
    np.savetxt(
        out_dir / "g0w0_qavg_vtot.csv",
        np.asarray(vtot_rows, dtype=float),
        delimiter=",",
        header="n_cm3,nu_meV,Vtot_qavg_eV_A3",
        comments="",
    )
    scale_data = np.asarray(scales, dtype=float)
    np.savetxt(
        out_dir / "g0w0_qavg_w_scales.csv",
        scale_data,
        delimiter=",",
        header="n_cm3,k_F_Ainv,qmax_Ainv,E_F_meV,phonon_scale_meV,E_plasmon_meV,E_TF_meV",
        comments="",
    )
    common.write_lines(
        out_dir / "g0w0_w_gap_diagnostics.txt",
        [
            r"Example 06: $G_0W_0$ $W(i\nu_n)$ and gap-function diagnostics.",
            r"Screened interaction: $W_0=V_{\mathrm{tot}}/(1-\Pi_0 V_{\mathrm{tot}})$.",
            (
                r"Unscreened interaction: "
                r"$V_{\mathrm{tot}}(q,i\nu_n)=V_c(q)V_{\mathrm{ph}}(q,i\nu_n)$."
            ),
            (
                r"Default q average: "
                r"$\sum_q w_q X(q,i\nu_n) / \sum_q w_q$."
            ),
            (
                r"Use --average-measure volume for "
                r"$\sum_q q^2w_q X(q,i\nu_n) / \sum_q q^2w_q$."
            ),
            r"Interaction units in the plots and CSVs: $\mathrm{eV}\,\AA^3$.",
            r"The $W_0$ and $V_{\mathrm{tot}}$ plots share x and y limits.",
            (
                r"Gap-function plots use the normalized k-averaged leading "
                r"Davidson eigenvector."
            ),
            r"The stored Davidson variable is $\tilde\phi=k\Delta$; the plot averages $\Delta$.",
            r"Use --gap-modes to choose no_trunc, truncated, or both.",
            r"Energy-scale formulas use atomic units internally:",
            r"$E_F=k_F^2/(2m^*)$.",
            (
                r"$\omega_{\mathrm{ph,max}}$ is the maximum phonon feature "
                r"used for radial-grid sizing."
            ),
            (
                r"For legacy single-pole parameters, "
                r"$\omega_{\mathrm{ph,max}}=\omega_{\mathrm{LO}}(0)$."
            ),
            r"$E_{\mathrm{pl}}=\sqrt{4\pi n/(m^*\epsilon_\infty)}$.",
            (
                r"$E_{\mathrm{TF}}=q_{\mathrm{TF}}^2/(2m^*)$, "
                r"$q_{\mathrm{TF}}^2=4m^*k_F/(\pi\epsilon_\infty)$."
            ),
        ],
    )


def write_gap_outputs(out_dir, gap_rows, stem="g0w0_gap_function"):
    """Write one k-averaged leading gap-function CSV file."""
    np.savetxt(
        out_dir / ("%s.csv" % stem),
        np.asarray(gap_rows, dtype=float),
        delimiter=",",
        header="n_cm3,omega_meV,gap_kavg_norm,gap_kavg_imag_norm",
        comments="",
    )


def write_gap_scale_outputs(out_dir, stem, scale_rows):
    """Write per-density gap solver scales and convergence flags."""
    np.savetxt(
        out_dir / ("%s_scales.csv" % stem),
        np.asarray(scale_rows, dtype=float),
        delimiter=",",
        header=(
            "n_cm3,k_F_Ainv,kmax_Ainv,qmax_Ainv,E_F_meV,phonon_scale_meV,"
            "E_plasmon_meV,E_TF_meV,lambda,gap_converged,gap_cycles,"
            "nk,nq,n_active_w,used_min_pair"
        ),
        comments="",
    )


def gap_mode_details(mode):
    """Return ``(truncate, output stem, plot title)`` for one gap mode."""
    if mode == "no_trunc":
        return (
            False,
            "g0w0_gap_function",
            r"$G_0W_0$ no-truncation k-averaged leading gap function",
        )
    if mode == "truncated":
        return (
            True,
            "g0w0_truncated_gap_function",
            r"$G_0W_0$ truncated k-averaged leading gap function",
        )
    raise ValueError("unknown gap mode %r" % mode)


def run_gap_mode(plt, ctx, params, densities, args, mode, out_dir):
    """Run and write one G0W0 gap-function diagnostic mode."""
    truncate, stem, title = gap_mode_details(mode)
    mode_args = argparse.Namespace(**vars(args))
    mode_args.gap_truncate = truncate
    datasets = []
    scale_rows = []
    for n_cm3 in densities:
        label = "truncated" if truncate else "no-trunc"
        print("[G0W0 gap %s] start n=%.6e cm^-3" % (label, float(n_cm3)), flush=True)
        omega_mev, gap_norm, gap_imag_norm, diag = g0w0_gap_function(
            ctx, params, n_cm3, mode_args
        )
        print(
            "[G0W0 gap %s] n=%.6e lambda=%.6g conv=%s cycles=%d nk=%d n_active_w=%d"
            % (
                label,
                float(n_cm3),
                diag["lambda"],
                diag["gap_converged"],
                diag["gap_cycles"],
                diag["nk"],
                diag["gap_n_active_w"],
            ),
            flush=True,
        )
        datasets.append(
            {
                "gap_diag": diag,
                "gap_omega_mev": omega_mev,
                "gap_norm": gap_norm,
                "gap_imag_norm": gap_imag_norm,
                "xmax_scale": args.xmax_scale,
            }
        )
        scale_rows.append(
            (
                diag["n_cm3"],
                diag["k_F"] * BOHR_INV_TO_A_INV,
                diag["kmax"] * BOHR_INV_TO_A_INV,
                diag["qmax"] * BOHR_INV_TO_A_INV,
                diag["E_F"] * HA_TO_MEV,
                diag["phonon_scale"] * HA_TO_MEV,
                diag["E_plasmon"] * HA_TO_MEV,
                diag["E_TF"] * HA_TO_MEV,
                diag["lambda"],
                int(diag["gap_converged"]),
                diag["gap_cycles"],
                diag["nk"],
                diag["nq"],
                diag["gap_n_active_w"],
                int(diag["gap_used_min_pair"]),
            )
        )
    gap_rows = plot_gap_function(plt, datasets, out_dir, stem=stem, title=title)
    write_gap_outputs(out_dir, gap_rows, stem=stem)
    write_gap_scale_outputs(out_dir, stem, scale_rows)
    return gap_rows


def main(argv=None):
    """Run the G0W0 interaction and gap-function diagnostics."""
    args = parse_args(argv)
    out_dir = common.prepare_out_dir(args.out_dir)
    params = model.STOParams()
    ctx = FreqGrid(
        params.beta_ha(args.temperature),
        lamb=args.lamb,
        ir_file=args.ir_file,
        save_ir=args.save_ir,
        verbose=0,
    )
    densities = model.density_grid(args.densities)
    plt = common.pyplot(out_dir)

    datasets = []
    scale_rows = []
    for n_cm3 in densities:
        W_avg, Vtot_avg, diag = q_weighted_w(ctx, params, n_cm3, args)
        x_mev, w_ev_a3 = positive_frequency_view(ctx.vs, W_avg)
        _, vtot_ev_a3 = positive_frequency_view(ctx.vs, Vtot_avg)
        datasets.append(
            {
                "diag": diag,
                "nu_mev": x_mev,
                "W": w_ev_a3,
                "Vtot": vtot_ev_a3,
                "xmax_scale": args.xmax_scale,
            }
        )
        scale_rows.append(
            (
                diag["n_cm3"],
                diag["k_F"] * BOHR_INV_TO_A_INV,
                diag["qmax"] * BOHR_INV_TO_A_INV,
                diag["E_F"] * HA_TO_MEV,
                diag["phonon_scale"] * HA_TO_MEV,
                diag["E_plasmon"] * HA_TO_MEV,
                diag["E_TF"] * HA_TO_MEV,
            )
        )
    plot_limits = common_limits(datasets, ("W", "Vtot"))
    w_rows = plot_interaction(
        plt,
        datasets,
        "W",
        r"$\langle W_0(\mathrm{i}\nu_n)\rangle_q$ [eV $\AA^3$]",
        r"$G_0W_0$ q-weighted screened interaction",
        out_dir,
        "g0w0_qavg_w",
        plot_limits,
    )
    vtot_rows = plot_interaction(
        plt,
        datasets,
        "Vtot",
        r"$\langle V_{\mathrm{tot}}(\mathrm{i}\nu_n)\rangle_q$ [eV $\AA^3$]",
        r"$G_0W_0$ q-weighted unscreened interaction",
        out_dir,
        "g0w0_qavg_vtot",
        plot_limits,
    )
    write_outputs(out_dir, w_rows, vtot_rows, scale_rows)
    for mode in args.gap_modes:
        run_gap_mode(plt, ctx, params, densities, args, mode, out_dir)
    print("wrote %s" % out_dir, flush=True)


if __name__ == "__main__":
    main()
