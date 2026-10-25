#!/usr/bin/env python

"""Plot G0W0/GW0/scGW q-averaged W(i nu_n) and gap diagnostics.

This example mirrors the plotting style of example 06, but the screened
interaction is reconstructed from each method's final radial W table.
G0W0 uses the fixed RPA table, while GW0 and scGW use their converged normal
states.  The default SCF settings follow the tuned example-04 GW0/scGW
convergence parameters.

Run from the repository root, preferably under MPI:

    mpirun -n 2 python examples/07_gw0_scgw_w_gap_diagnostics/run.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402


G0W0_DIAGNOSTICS_RUN = SCRIPT_DIR.parent / "06_g0w0_w_gap_diagnostics" / "run.py"
DENSITIES_CM3 = (1.0e17, 4.0e20, 1.0e23)
METHODS = ("G0W0", "GW0", "scGW")
METHOD_STYLES = {
    "G0W0": ("#222222", "o"),
    "GW0": ("#1f77b4", "s"),
    "scGW": ("#d62728", "^"),
}
LINE_ONLY_METHOD_STYLES = {
    "G0W0": ("#222222", (0, (5.0, 2.0))),
    "GW0": ("#1f77b4", (0, (3.0, 1.6))),
    "scGW": ("#d62728", (0, (1.4, 1.4))),
}


def load_g0w0_diagnostics():
    """Load example 06 helper functions for shared plotting conventions."""
    spec = importlib.util.spec_from_file_location(
        "example06_g0w0_w_gap_diagnostics_run", G0W0_DIAGNOSTICS_RUN
    )
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("could not load %s" % G0W0_DIAGNOSTICS_RUN)
    spec.loader.exec_module(mod)
    return mod


def default_ir_file():
    """Return the preferred sparse-IR cache path for this example."""
    candidates = (
        SCRIPT_DIR / "ir.h5",
        SCRIPT_DIR.parent / "04_g0w0_gw0_scgw_curve" / "ir.h5",
        SCRIPT_DIR.parent / "04_g0w0_gw0_scgw_curve" / "ir.5",
        SCRIPT_DIR.parent / "ir.h5",
        REPO_ROOT / "examples" / "ir.h5",
    )
    for path in candidates:
        if path.exists():
            return path
    return SCRIPT_DIR / "ir.h5"


def parse_args(argv=None):
    """Parse G0W0/GW0/scGW interaction diagnostic arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--densities",
        default=",".join("%.12g" % n for n in DENSITIES_CM3),
        help="Comma-separated densities in cm^-3.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    common.add_radial_args(
        parser,
        default_ir_file=default_ir_file(),
        ir_help=(
            "Sparse-IR h5 cache file. Defaults to this example's ir.h5, then "
            "example 04's ir.h5/ir.5."
        ),
    )
    parser.add_argument("--save-ir", dest="save_ir", action="store_true", default=True)
    parser.add_argument("--no-save-ir", dest="save_ir", action="store_false")
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument("--kmax-factor", type=float, default=4.0)
    parser.add_argument("--kmax-lo-factor", type=float, default=2.0)
    parser.add_argument("--qmax-factor", type=float, default=2.0)
    parser.add_argument("--diag-ncell", type=int, default=12)
    parser.add_argument("--xmax-scale", type=float, default=4.0)
    parser.add_argument(
        "--average-measure",
        choices=("dq", "volume"),
        default="dq",
        help="Use q-grid trapezoid weights by default; volume uses q^2 w_q.",
    )
    parser.add_argument(
        "--gap-average-measure",
        choices=("dk", "volume"),
        default="dk",
        help="Use k-grid weights for gap averages by default; volume uses k^2 w_k.",
    )
    parser.add_argument("--gw0-scf-max-cycle", type=int, default=300)
    parser.add_argument("--gw0-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--scgw-scf-max-cycle", type=int, default=300)
    parser.add_argument("--scgw-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--scgw-scf-damp", type=float, default=0.98)
    parser.add_argument("--scgw-scf-diis-space", type=int, default=8)
    parser.add_argument("--scgw-scf-diis-start-cycle", type=int, default=0)
    parser.add_argument(
        "--scf-fs-pin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-scf-fs-pin to run the unpinned SCF gauge.",
    )
    parser.add_argument("--allow-no-pi-tail", action="store_true")
    parser.add_argument(
        "--fail-unconverged",
        action="store_true",
        help="Raise if GW0/scGW SCF does not converge; default writes flags and continues.",
    )
    parser.add_argument("--scgw-n-tail", type=int, default=100)
    parser.add_argument("--pi-imag-tol", type=float, default=1.0e-5)
    parser.add_argument("--pi-imag-policy", choices=("warn", "fail"), default="warn")
    parser.add_argument(
        "--sigma-tail-policy",
        choices=("exchange_static", "exchange_t0", "zero"),
        default="exchange_static",
    )
    common.add_output_arg(parser, __file__)
    return parser.parse_args(argv)


def radial_kwargs(args):
    """Return common radial driver kwargs for one diagnostic run."""
    kwargs = common.radial_kwargs(args)
    kwargs.update(
        temperature=args.temperature,
        save_ir=args.save_ir,
        verbose=args.verbose,
        truncate=False,
        diag_average=1,
        diag_ncell=args.diag_ncell,
        kmax_factor=args.kmax_factor,
        kmax_lo_factor=args.kmax_lo_factor,
        qmax_factor=args.qmax_factor,
        return_gap_function=True,
    )
    return kwargs


def make_driver(method, args, densities):
    """Construct the requested G0W0, GW0, or scGW driver."""
    from migdal.g0w0 import G0W0
    from migdal.gw0 import GW0
    from migdal.scgw import ScGW

    kwargs = radial_kwargs(args)
    kwargs["densities"] = densities
    if method == "G0W0":
        return G0W0(**kwargs)
    if method == "GW0":
        gw0_kwargs = common.changed_driver_kwargs(args, "gw0")
        if not args.fail_unconverged:
            gw0_kwargs["allow_unconverged"] = True
        return GW0(**kwargs, **gw0_kwargs)
    if method == "scGW":
        scgw_kwargs = common.changed_driver_kwargs(args, "scgw")
        if not args.fail_unconverged:
            scgw_kwargs["allow_unconverged"] = True
        return ScGW(**kwargs, **scgw_kwargs)
    raise ValueError("unknown method %r" % method)


def trapezoid_weights(x):
    """Return trapezoid integration weights on a strictly increasing grid."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points.")
    dx = np.diff(x)
    if not np.all(dx > 0.0):
        raise ValueError("x grid must be strictly increasing.")
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w


def wprecomp_frequency_table(ctx, W):
    """Reconstruct ``W(q, i nu_n)`` from a radial W precomputation object."""
    if hasattr(W, "frequency_table"):
        return W.frequency_table(ctx)
    inv = getattr(ctx, "_example07_F_t_b2f_inv", None)
    if inv is None:
        inv = np.linalg.pinv(np.asarray(ctx.F_t_b2f))
        setattr(ctx, "_example07_F_t_b2f_inv", inv)
    dW = ctx.trans(np.asarray(W.yD), inv, ctx.F_w_b)
    return dW + np.asarray(W.yS)[:, None]


def q_average(q_grid, table, measure):
    """Average a q-dependent frequency table using dq or volume weights."""
    weights = trapezoid_weights(q_grid)
    if measure == "volume":
        weights = weights * q_grid**2
    norm = np.sum(weights)
    if norm <= 0.0:
        raise ValueError("non-positive q-weight normalization.")
    return np.tensordot(weights, table, axes=(0, 0)) / norm


def positive_frequency_complex(base, vs, vals, unit):
    """Return positive-frequency real and imaginary components in target units."""
    vs = np.asarray(vs, dtype=float)
    vals = np.asarray(vals)
    mask = vs >= -1.0e-14
    order = np.argsort(vs[mask])
    x = vs[mask][order] * base.HA_TO_MEV
    y = vals[mask][order] * unit
    return x, y.real, y.imag


def bare_vtot_table(base, ctx, params, q_grid):
    """Return the bare ``Vtot(q, nu) = Vc(q) * Vph(q, nu)`` table."""
    Vc = base.model.get_Vc(
        q_grid, params.eps_inf, q0_treat=False, tol=1.0e-10
    )
    Vph = base.model.get_Vph_q(q_grid, ctx.vs, params)
    return Vc[:, None] * Vph


def gap_view(base, row, drv, args):
    """Return a normalized k-averaged gap-function view for one snapshot."""
    phi = row.gap.phi_tilde
    if phi is None:
        raise RuntimeError("gap solver did not return a gap eigenfunction.")
    k_grid = np.asarray(drv._last_k_grid, dtype=float)
    k_wts = np.asarray(drv._last_k_wts, dtype=float)
    delta = np.asarray(phi) / k_grid[:, None]
    avg_wts = k_wts
    if args.gap_average_measure == "volume":
        avg_wts = avg_wts * k_grid**2
    delta_avg = np.tensordot(avg_wts, delta, axes=(0, 0)) / np.sum(avg_wts)
    pos = np.asarray(drv.ctx.ws) > 0.0
    omega_mev = np.asarray(drv.ctx.ws)[pos] * base.HA_TO_MEV
    gap_vals = np.asarray(delta_avg[pos])
    ref = int(np.argmax(np.abs(gap_vals)))
    if abs(gap_vals[ref]) > 0.0:
        gap_vals = gap_vals * np.exp(-1j * np.angle(gap_vals[ref]))
        if gap_vals[ref].real < 0.0:
            gap_vals = -gap_vals
    scale = max(float(np.max(np.abs(gap_vals.real))), 1.0e-300)
    return omega_mev, gap_vals.real / scale, gap_vals.imag / scale


def make_snapshot(base, method, row, drv, args):
    """Collect W, Vtot, gap, scales, and convergence diagnostics."""
    data = drv._last_point_data
    W = drv._last_W
    q_grid = np.asarray(W.q, dtype=float)
    W_freq = wprecomp_frequency_table(drv.ctx, W)
    W_avg = q_average(q_grid, W_freq, args.average_measure)
    Vtot = bare_vtot_table(base, drv.ctx, drv.params, q_grid)
    Vtot_avg = q_average(q_grid, Vtot, args.average_measure)
    nu_mev, w_ev_a3, w_imag_ev_a3 = positive_frequency_complex(
        base, drv.ctx.vs, W_avg, base.W_AU_TO_EV_A3
    )
    _, vtot_ev_a3, vtot_imag_ev_a3 = positive_frequency_complex(
        base, drv.ctx.vs, Vtot_avg, base.W_AU_TO_EV_A3
    )
    omega_mev, gap_norm, gap_imag_norm = gap_view(base, row, drv, args)
    scales = base.energy_scales(drv.params, row.n_bohr3, data["k_F"], row.mu0)
    diag = {
        "method": method,
        "n_cm3": float(row.n_cm3),
        "n_bohr3": float(row.n_bohr3),
        "mu": float(row.mu),
        "mu0": float(row.mu0),
        "k_F": float(data["k_F"]),
        "kmax": float(data["kmax"]),
        "qmax": float(np.max(q_grid)),
        "nk": int(row.nk),
        "nq": int(len(q_grid)),
        "average_measure": args.average_measure,
        "gap_average_measure": args.gap_average_measure,
        "lambda": float(row.lam),
        "gap_converged": bool(row.gap_converged),
        "gap_cycles": int(row.gap.davidson_cycles),
        "gap_n_active_w": int(row.gap.n_active_w),
        "gap_used_min_pair": bool(row.gap.used_min_pair),
        "scf_converged": row.scf_converged,
        "scf_cycles": row.scf_cycles,
        "scf_res_sigma": row.scf_res_sigma,
        "scf_dn": row.scf_dn,
        **scales,
    }
    return {
        "method": method,
        "diag": diag,
        "nu_mev": nu_mev,
        "W": w_ev_a3,
        "W_imag": w_imag_ev_a3,
        "Vtot": vtot_ev_a3,
        "Vtot_imag": vtot_imag_ev_a3,
        "gap_omega_mev": omega_mev,
        "gap_norm": gap_norm,
        "gap_imag_norm": gap_imag_norm,
        "xmax_scale": args.xmax_scale,
    }


def run_driver_points(base, method, drv, densities, args):
    """Run the requested density points independently."""
    snapshots = []
    for n_cm3 in densities:
        row = drv.kernel(float(n_cm3))
        snapshots.append(make_snapshot(base, method, row, drv, args))
    return snapshots


def run_method_points(base, method, drv, densities, args):
    """Dispatch the method-specific point runner."""
    return run_driver_points(base, method, drv, densities, args)


def group_snapshots(snapshots, densities, methods):
    """Group snapshots into method/density slots for panel plotting."""
    groups = []
    for n_cm3 in densities:
        by_method = {}
        for item in snapshots:
            if np.isclose(item["diag"]["n_cm3"], float(n_cm3), rtol=1.0e-12, atol=0.0):
                by_method[item["method"]] = item
        group = [by_method[m] for m in methods if m in by_method]
        if group:
            groups.append((float(n_cm3), group))
    return groups


def check_matching_grids(snapshots, densities, methods):
    """Require all plotted methods to use the same nk and nq at each density."""
    for density, group in group_snapshots(snapshots, densities, methods):
        if len(group) < 2:
            continue
        nk0 = group[0]["diag"]["nk"]
        nq0 = group[0]["diag"]["nq"]
        for item in group[1:]:
            diag = item["diag"]
            if diag["nk"] != nk0 or diag["nq"] != nq0:
                raise RuntimeError(
                    "grid mismatch at n=%.12g cm^-3: %s has nk/nq=%d/%d, "
                    "expected %d/%d"
                    % (
                        density,
                        diag["method"],
                        diag["nk"],
                        diag["nq"],
                        nk0,
                        nq0,
                    )
                )


def interaction_limits(base, snapshots, y_keys):
    """Return shared x/y limits for interaction figures."""
    xmax = 10.0
    for item in snapshots:
        diag = item["diag"]
        scales = np.array([diag[k] for k in ("E_F", "phonon_scale", "E_plasmon", "E_TF")])
        xmax = max(xmax, item["xmax_scale"] * float(np.max(scales * base.HA_TO_MEV)))
    y_blocks = []
    for item in snapshots:
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


def make_diagnostic_figure(plt, n_panels, title):
    """Create example-06 style axes plus legend and diagnostic text rows."""
    fig_width = max(5.2 * n_panels, 6.0)
    fig = plt.figure(figsize=(fig_width, 6.35), constrained_layout=True)
    grid_spec = fig.add_gridspec(3, n_panels, height_ratios=(0.09, 1.0, 0.30))
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


def scf_state_text(value):
    """Format SCF convergence state for compact notes."""
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def density_note_text(base, group, gap=False):
    """Return below-panel diagnostic text for one density group."""
    diag = group[0]["diag"]
    grid_text = ", ".join(
        "%s=%d/%d" % (item["method"], item["diag"]["nk"], item["diag"]["nq"])
        for item in group
    )
    lines = [
        base.energy_scale_text(diag),
    ]
    if gap:
        lines.append(
            (
                r"$k_{\mathrm{F}}=%s,\ k_{\max}=%s,\ \langle\cdot\rangle_k:\ %s$"
                % (
                    base.ainv_text(diag["k_F"] * base.BOHR_INV_TO_A_INV),
                    base.ainv_text(diag["kmax"] * base.BOHR_INV_TO_A_INV),
                    base.average_label(diag["gap_average_measure"], "k"),
                )
            )
        )
        lambdas = ", ".join(
            "%s=%s" % (item["method"], base.math_number(item["diag"]["lambda"], 3))
            for item in group
        )
        lines.append(r"$\lambda:\ %s$" % lambdas)
    else:
        lines.append(
            (
                r"$k_{\mathrm{F}}=%s,\ q_{\max}=%s,\ \langle\cdot\rangle_q:\ %s$"
                % (
                    base.ainv_text(diag["k_F"] * base.BOHR_INV_TO_A_INV),
                    base.ainv_text(diag["qmax"] * base.BOHR_INV_TO_A_INV),
                    base.average_label(diag["average_measure"], "q"),
                )
            )
        )
    lines.append(r"$N_k/N_q:\ %s$" % grid_text)
    if not gap:
        scf_text = ", ".join(
            "%s=%s" % (item["method"], scf_state_text(item["diag"]["scf_converged"]))
            for item in group
        )
        lines.append(r"$\mathrm{SCF}:\ %s$" % scf_text)
    return "\n".join(lines)


def plot_interaction(base, plt, snapshots, densities, methods, y_key, imag_key,
                     ylabel, title, out_dir, stem, limits):
    """Plot q-averaged interaction diagnostics for the given methods."""
    groups = group_snapshots(snapshots, densities, methods)
    fig, legend_ax, axes, note_axes = make_diagnostic_figure(plt, len(groups), title)
    base.add_energy_scale_legend(legend_ax)
    xlim, ylim = limits
    rows = []
    for ax, note_ax, (_, group) in zip(axes, note_axes, groups):
        for item in group:
            method = item["method"]
            color, marker = METHOD_STYLES[method]
            x = item["nu_mev"]
            y = item[y_key]
            visible = x <= xlim[1]
            ax.plot(
                x[visible],
                y[visible],
                marker=marker,
                ms=3.0,
                lw=1.2,
                color=color,
                label=method,
            )
            for nu, real_val, imag_val in zip(x, y, item[imag_key]):
                rows.append((method, item["diag"]["n_cm3"], nu, real_val, imag_val))
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\nu_n$ [meV]")
        ax.set_title(base.density_label(group[0]["diag"]["n_cm3"]))
        ax.grid(True, which="both", alpha=0.25)
        base.draw_energy_scale_lines(ax, group[0]["diag"])
        base.write_panel_note(note_ax, density_note_text(base, group, gap=False))
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel(ylabel)
    fig.savefig(out_dir / ("%s.png" % stem), dpi=220)
    fig.savefig(out_dir / ("%s.pdf" % stem))
    plt.close(fig)
    return rows


def gap_limits(base, snapshots):
    """Return shared x/y limits for gap-function plots."""
    xmax = 10.0
    for item in snapshots:
        diag = item["diag"]
        scales = np.array([diag[k] for k in ("E_F", "phonon_scale", "E_plasmon", "E_TF")])
        xmax = max(xmax, item["xmax_scale"] * float(np.max(scales * base.HA_TO_MEV)))
    ymin = min(-1.05, min(float(np.min(item["gap_norm"])) for item in snapshots) * 1.05)
    ymax = max(1.05, max(float(np.max(item["gap_norm"])) for item in snapshots) * 1.05)
    return xmax, ymin, ymax


def plot_gap_function(
    base,
    plt,
    snapshots,
    densities,
    methods,
    out_dir,
    stem="gw0_scgw_gap_function",
    title="G0W0/GW0/scGW k-averaged leading gap function",
):
    """Plot k-averaged leading gap functions for the given methods."""
    groups = group_snapshots(snapshots, densities, methods)
    fig, legend_ax, axes, note_axes = make_diagnostic_figure(plt, len(groups), title)
    base.add_energy_scale_legend(legend_ax)
    xmax, ymin, ymax = gap_limits(base, snapshots)
    rows = []
    for ax, note_ax, (_, group) in zip(axes, note_axes, groups):
        for item in group:
            method = item["method"]
            color, marker = METHOD_STYLES[method]
            x = item["gap_omega_mev"]
            y = item["gap_norm"]
            visible = x <= xmax
            ax.plot(
                x[visible],
                y[visible],
                marker=marker,
                ms=3.0,
                lw=1.2,
                color=color,
                label="%s lambda=%.3g" % (method, item["diag"]["lambda"]),
            )
            for omega, real_val, imag_val in zip(x, y, item["gap_imag_norm"]):
                rows.append((method, item["diag"]["n_cm3"], omega, real_val, imag_val))
        ax.axhline(0.0, color="0.55", lw=0.8)
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$\omega_n$ [meV]")
        ax.set_title(base.density_label(group[0]["diag"]["n_cm3"]))
        ax.grid(True, which="both", alpha=0.25)
        base.draw_energy_scale_lines(ax, group[0]["diag"])
        base.write_panel_note(note_ax, density_note_text(base, group, gap=True))
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"normalized $\langle \Delta(k,\mathrm{i}\omega_n)\rangle_k$")
    fig.savefig(out_dir / ("%s.png" % stem), dpi=220)
    fig.savefig(out_dir / ("%s.pdf" % stem))
    plt.close(fig)
    return rows


def write_mixed_csv(path, header, rows):
    """Write rows with a header to CSV, allowing mixed value types."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(base, out_dir, snapshots, w_rows, vtot_rows, gap_rows, args):
    """Write combined CSV files and the main text summary."""
    write_mixed_csv(
        out_dir / "gw0_scgw_qavg_w.csv",
        ["method", "n_cm3", "nu_meV", "W_qavg_real_eV_A3", "W_qavg_imag_eV_A3"],
        w_rows,
    )
    write_mixed_csv(
        out_dir / "gw0_scgw_qavg_vtot.csv",
        ["method", "n_cm3", "nu_meV", "Vtot_qavg_real_eV_A3", "Vtot_qavg_imag_eV_A3"],
        vtot_rows,
    )
    write_mixed_csv(
        out_dir / "gw0_scgw_gap_function.csv",
        ["method", "n_cm3", "omega_meV", "gap_kavg_norm", "gap_kavg_imag_norm"],
        gap_rows,
    )
    scale_rows = []
    for item in snapshots:
        d = item["diag"]
        scale_rows.append(
            (
                d["method"],
                d["n_cm3"],
                d["k_F"] * base.BOHR_INV_TO_A_INV,
                d["kmax"] * base.BOHR_INV_TO_A_INV,
                d["qmax"] * base.BOHR_INV_TO_A_INV,
                d["E_F"] * base.HA_TO_MEV,
                d["phonon_scale"] * base.HA_TO_MEV,
                d["E_plasmon"] * base.HA_TO_MEV,
                d["E_TF"] * base.HA_TO_MEV,
                d["lambda"],
                int(d["gap_converged"]),
                d["gap_cycles"],
                -1 if d["scf_converged"] is None else int(d["scf_converged"]),
                -1 if d["scf_cycles"] is None else int(d["scf_cycles"]),
                np.nan if d["scf_res_sigma"] is None else d["scf_res_sigma"],
                np.nan if d["scf_dn"] is None else d["scf_dn"],
                d["nk"],
                d["nq"],
                d["gap_n_active_w"],
                int(d["gap_used_min_pair"]),
            )
        )
    write_mixed_csv(
        out_dir / "gw0_scgw_qavg_w_scales.csv",
        [
            "method",
            "n_cm3",
            "k_F_Ainv",
            "kmax_Ainv",
            "qmax_Ainv",
            "E_F_meV",
            "phonon_scale_meV",
            "E_plasmon_meV",
            "E_TF_meV",
            "lambda",
            "gap_converged",
            "gap_cycles",
            "scf_converged",
            "scf_cycles",
            "scf_res_sigma",
            "scf_dn",
            "nk",
            "nq",
            "n_active_w",
            "used_min_pair",
        ],
        scale_rows,
    )
    common.write_lines(
        out_dir / "gw0_scgw_qavg_w.txt",
        [
            "G0W0/GW0/scGW q-weighted W(i nu_n) diagnostic.",
            "Screened interactions are reconstructed from each method's final radial W table.",
            "Default q average: sum_q w_q X(q, i nu_n) / sum_q w_q.",
            "Use --average-measure volume for sum_q q^2 w_q X / sum_q q^2 w_q.",
            "Interaction units in the plots and CSVs: eV Angstrom^3.",
            "The gap-function plot uses the normalized k-averaged leading Davidson eigenvector.",
            "The stored Davidson variable is phi_tilde=k Delta; the plot averages Delta.",
            "GW0/scGW SCF defaults follow example 04: damp=%s, "
            "DIIS space=%d, DIIS start=%d, max_cycle=%d."
            % (
                args.scgw_scf_damp,
                args.scgw_scf_diis_space,
                args.scgw_scf_diis_start_cycle,
                args.scgw_scf_max_cycle,
            ),
            "Convergence flags and SCF cycles are stored in gw0_scgw_qavg_w_scales.csv.",
            "The example writes unconverged rows by default; inspect flags before using lambda.",
            "Extra no-marker line overlays including G0W0 are written as "
            "g0w0_gw0_scgw_qavg_w_lines.* and "
            "g0w0_gw0_scgw_gap_function_lines.*.",
        ],
    )


def _density_key(densities, value):
    value = float(value)
    for density in densities:
        if np.isclose(value, float(density), rtol=1.0e-10, atol=0.0):
            return float(density)
    return None


def _append_overlay_point(store, densities, method, n_cm3, x, y, imag=0.0):
    key = _density_key(densities, n_cm3)
    if key is None:
        return
    rec = store.setdefault(method, {}).setdefault(
        key, {"x": [], "y": [], "imag": []}
    )
    rec["x"].append(float(x))
    rec["y"].append(float(y))
    rec["imag"].append(float(imag))


def _finalize_overlay_store(store):
    for by_density in store.values():
        for rec in by_density.values():
            order = np.argsort(np.asarray(rec["x"], dtype=float))
            for name in ("x", "y", "imag"):
                rec[name] = np.asarray(rec[name], dtype=float)[order]


def _read_csv_dicts(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_overlay_data(out_dir, densities):
    w_store = {}
    gap_store = {}
    lambda_by_key = {}
    scales_by_density = {}

    for row in _read_csv_dicts(out_dir / "gw0_scgw_qavg_w.csv"):
        method = row.get("method", "")
        if method in METHODS:
            _append_overlay_point(
                w_store,
                densities,
                method,
                row["n_cm3"],
                row["nu_meV"],
                row["W_qavg_real_eV_A3"],
                row.get("W_qavg_imag_eV_A3", 0.0),
            )
    for row in _read_csv_dicts(out_dir / "gw0_scgw_gap_function.csv"):
        method = row.get("method", "")
        if method in METHODS:
            _append_overlay_point(
                gap_store,
                densities,
                method,
                row["n_cm3"],
                row["omega_meV"],
                row["gap_kavg_norm"],
                row.get("gap_kavg_imag_norm", 0.0),
            )

    for row in _read_csv_dicts(out_dir / "gw0_scgw_qavg_w_scales.csv"):
        key = _density_key(densities, row["n_cm3"])
        if key is None:
            continue
        method = row.get("method", "")
        if method in METHODS and row.get("lambda", "") != "":
            lambda_by_key[(method, key)] = float(row["lambda"])
        scales_by_density.setdefault(
            key,
            {
                "E_F_meV": float(row["E_F_meV"]),
                "phonon_scale_meV": float(row["phonon_scale_meV"]),
                "E_plasmon_meV": float(row["E_plasmon_meV"]),
                "E_TF_meV": float(row["E_TF_meV"]),
            },
        )

    _finalize_overlay_store(w_store)
    _finalize_overlay_store(gap_store)
    return w_store, gap_store, lambda_by_key, scales_by_density


def _overlay_xmax(stores, scales_by_density, xmax_scale):
    xmax = 10.0
    for scales in scales_by_density.values():
        vals = [
            scales["E_F_meV"],
            scales["phonon_scale_meV"],
            scales["E_plasmon_meV"],
            scales["E_TF_meV"],
        ]
        xmax = max(xmax, xmax_scale * max(vals))
    if xmax > 10.0:
        return xmax
    for store in stores:
        for by_density in store.values():
            for rec in by_density.values():
                if len(rec["x"]):
                    xmax = max(xmax, float(np.max(rec["x"])))
    return xmax


def _overlay_w_limits(w_store, xmax):
    vals = []
    for by_density in w_store.values():
        for rec in by_density.values():
            visible = rec["x"] <= xmax
            y = rec["y"][visible]
            y = y[np.isfinite(y) & (y > 0.0)]
            if y.size:
                vals.append(y)
    if not vals:
        return 1.0e-3, 1.0
    all_y = np.concatenate(vals)
    ymin = 10.0 ** np.floor(np.log10(np.min(all_y)))
    ymax = 10.0 ** np.ceil(np.log10(np.max(all_y)))
    if ymin == ymax:
        ymin *= 0.1
        ymax *= 10.0
    return ymin, ymax


def _aligned_gap(rec):
    y = np.array(rec["y"], dtype=float, copy=True)
    imag = np.array(rec["imag"], dtype=float, copy=True)
    if y.size:
        ref = int(np.argmax(np.abs(y)))
        if y[ref] < 0.0:
            y = -y
            imag = -imag
    return y, imag


def _overlay_gap_limits(gap_store, xmax):
    vals = []
    for by_density in gap_store.values():
        for rec in by_density.values():
            y, _ = _aligned_gap(rec)
            visible = rec["x"] <= xmax
            y = y[visible]
            if y.size:
                vals.append(y)
    if not vals:
        return -1.05, 1.05
    all_y = np.concatenate(vals)
    return min(-1.05, float(np.min(all_y)) * 1.05), max(
        1.05, float(np.max(all_y)) * 1.05
    )


def plot_g0w0_line_overlays(base, plt, out_dir, densities, args):
    """Write no-marker G0W0/GW0/scGW line overlays from current CSV data."""
    w_store, gap_store, lambda_by_key, scales_by_density = _load_overlay_data(
        out_dir, densities
    )
    if not w_store or not gap_store:
        return

    xmax = _overlay_xmax((w_store, gap_store), scales_by_density, args.xmax_scale)
    w_ylim = _overlay_w_limits(w_store, xmax)
    gap_ylim = _overlay_gap_limits(gap_store, xmax)

    fig, axes = plt.subplots(
        1, len(densities), figsize=(5.1 * len(densities), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    w_rows = []
    for ax, density in zip(axes, densities):
        for method in METHODS:
            rec = w_store.get(method, {}).get(float(density))
            if rec is None:
                continue
            color, linestyle = LINE_ONLY_METHOD_STYLES[method]
            visible = rec["x"] <= xmax
            ax.plot(
                rec["x"][visible],
                rec["y"][visible],
                lw=1.6,
                color=color,
                linestyle=linestyle,
                label=method,
            )
            for nu, real_val, imag_val in zip(rec["x"], rec["y"], rec["imag"]):
                w_rows.append((method, density, nu, real_val, imag_val))
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_yscale("log")
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(*w_ylim)
        ax.set_xlabel(r"$\nu_n$ [meV]")
        ax.set_title(base.density_label(float(density)))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel(r"$\langle W(\mathrm{i}\nu_n)\rangle_q$ [eV A$^3$]")
    fig.suptitle("G0W0/GW0/scGW q-weighted screened interaction", fontsize=13)
    fig.savefig(out_dir / "g0w0_gw0_scgw_qavg_w_lines.png", dpi=220)
    fig.savefig(out_dir / "g0w0_gw0_scgw_qavg_w_lines.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(densities), figsize=(5.1 * len(densities), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    gap_rows = []
    for ax, density in zip(axes, densities):
        for method in METHODS:
            rec = gap_store.get(method, {}).get(float(density))
            if rec is None:
                continue
            color, linestyle = LINE_ONLY_METHOD_STYLES[method]
            y, imag = _aligned_gap(rec)
            visible = rec["x"] <= xmax
            lam = lambda_by_key.get((method, float(density)))
            label = method if lam is None else "%s lambda=%.3g" % (method, lam)
            ax.plot(
                rec["x"][visible],
                y[visible],
                lw=1.6,
                color=color,
                linestyle=linestyle,
                label=label,
            )
            for omega, real_val, imag_val in zip(rec["x"], y, imag):
                gap_rows.append((method, density, omega, real_val, imag_val))
        ax.axhline(0.0, color="0.55", lw=0.8)
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(*gap_ylim)
        ax.set_xlabel(r"$\omega_n$ [meV]")
        ax.set_title(base.density_label(float(density)))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"normalized $\langle \Delta(k,\mathrm{i}\omega_n)\rangle_k$")
    fig.suptitle("G0W0/GW0/scGW k-averaged leading gap function", fontsize=13)
    fig.savefig(out_dir / "g0w0_gw0_scgw_gap_function_lines.png", dpi=220)
    fig.savefig(out_dir / "g0w0_gw0_scgw_gap_function_lines.pdf")
    plt.close(fig)

    write_mixed_csv(
        out_dir / "g0w0_gw0_scgw_qavg_w_lines.csv",
        ["method", "n_cm3", "nu_meV", "W_qavg_real_eV_A3", "W_qavg_imag_eV_A3"],
        w_rows,
    )
    write_mixed_csv(
        out_dir / "g0w0_gw0_scgw_gap_function_lines.csv",
        [
            "method",
            "n_cm3",
            "omega_meV",
            "gap_kavg_norm_aligned",
            "gap_kavg_imag_norm_aligned",
        ],
        gap_rows,
    )


def main(argv=None):
    """Run G0W0, GW0, and scGW diagnostics."""
    args = parse_args(argv)
    from mpi4pyscf.lib import logger  # noqa: F401
    from mpi4pyscf.tools import mpi

    base = load_g0w0_diagnostics()
    densities = base.model.density_grid(args.densities)
    out_dir = common.prepare_out_dir(args.out_dir)
    snapshots = []
    common.rank0_print(
        mpi.rank,
        "example 7 start methods=%s densities=%d out_dir=%s ir_file=%s"
        % (",".join(METHODS), len(densities), out_dir, args.ir_file),
    )
    for method in METHODS:
        common.rank0_print(mpi.rank, "running %s" % method)
        drv = make_driver(method, args, densities)
        snapshots.extend(run_method_points(base, method, drv, densities, args))
    check_matching_grids(snapshots, densities, METHODS)

    if mpi.rank == 0:
        plt = common.pyplot(out_dir)
        limits = interaction_limits(base, snapshots, ("W", "Vtot"))
        w_rows = plot_interaction(
            base,
            plt,
            snapshots,
            densities,
            METHODS,
            "W",
            "W_imag",
            r"$\langle W(\mathrm{i}\nu_n)\rangle_q$ [eV A$^3$]",
            "G0W0/GW0/scGW q-weighted screened interaction",
            out_dir,
            "gw0_scgw_qavg_w",
            limits,
        )
        vtot_rows = plot_interaction(
            base,
            plt,
            snapshots,
            densities,
            METHODS,
            "Vtot",
            "Vtot_imag",
            r"$\langle V_{\mathrm{tot}}(\mathrm{i}\nu_n)\rangle_q$ [eV A$^3$]",
            "G0W0/GW0/scGW q-weighted unscreened interaction",
            out_dir,
            "gw0_scgw_qavg_vtot",
            limits,
        )
        gap_rows = plot_gap_function(
            base, plt, snapshots, densities, METHODS, out_dir
        )
        write_outputs(base, out_dir, snapshots, w_rows, vtot_rows, gap_rows, args)
        plot_g0w0_line_overlays(base, plt, out_dir, densities, args)
        print("wrote %s" % out_dir, flush=True)


if __name__ == "__main__":
    main()
