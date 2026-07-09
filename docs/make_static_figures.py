"""Generate documentation-only static figures.

These figures are explanatory schematics used by the Sphinx pages. They are
kept separate from example-output figures so docs diagrams can have a consistent
publication style without changing runnable example artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/migdal-matplotlib")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from migdal import grid, model


DOCS_DIR = Path(__file__).resolve().parent
STATIC_DIR = DOCS_DIR / "_static"


COLORS = {
    "ink": "#1b2433",
    "muted": "#5f6b7a",
    "blue": "#2b7ec7",
    "blue_fill": "#e9f3ff",
    "teal": "#159d8e",
    "teal_fill": "#e7f6f1",
    "orange": "#d97706",
    "orange_fill": "#fff7ea",
    "red": "#c81e1e",
    "rose_fill": "#fff1f1",
    "gray_fill": "#f7f8fa",
    "line": "#1f2937",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "axes.formatter.use_mathtext": True,
            "axes.edgecolor": "#c8d3df",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, name: str, dpi: int = 220) -> None:
    fig.savefig(STATIC_DIR / name, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def rounded_box(ax, xy, width, height, facecolor, edgecolor=COLORS["line"]):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, *, color=COLORS["line"], ms=16, lw=1.8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_w_construction_chain() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.35))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    rounded_box(ax, (0.035, 0.615), 0.29, 0.225, COLORS["blue_fill"])
    rounded_box(ax, (0.035, 0.305), 0.29, 0.235, COLORS["teal_fill"])
    rounded_box(ax, (0.405, 0.535), 0.30, 0.295, COLORS["gray_fill"])
    rounded_box(ax, (0.405, 0.155), 0.30, 0.265, COLORS["orange_fill"])
    rounded_box(ax, (0.795, 0.435), 0.185, 0.295, COLORS["rose_fill"])

    ax.text(0.18, 0.790, "Coulomb head", ha="center", va="center", fontsize=13.8, weight="bold")
    ax.text(0.18, 0.713, r"$V_{\mathrm{c}}(q)=4\pi/(\epsilon_\infty q^2)$", ha="center", va="center", fontsize=15.2)
    ax.text(0.18, 0.640, "long-range part", ha="center", va="center", fontsize=10.7, color=COLORS["muted"])

    ax.text(0.18, 0.490, "polar phonon factor", ha="center", va="center", fontsize=13.8, weight="bold")
    ax.text(0.18, 0.414, r"$V_{\mathrm{ph}}(q,\mathrm{i}\nu)$", ha="center", va="center", fontsize=15.2)
    ax.text(
        0.18,
        0.335,
        r"from $\omega_{\mathrm{TO}}$ and $\omega_{\mathrm{LO}}$",
        ha="center",
        va="center",
        fontsize=10.4,
        color=COLORS["muted"],
    )

    ax.text(0.555, 0.762, "bare interaction table", ha="center", va="center", fontsize=13.8, weight="bold")
    ax.text(0.555, 0.660, r"$V_{\mathrm{tot}}=V_{\mathrm{c}}V_{\mathrm{ph}}$", ha="center", va="center", fontsize=15.8)
    ax.text(0.555, 0.574, r"on the $(q,\mathrm{i}\nu)$ grid", ha="center", va="center", fontsize=10.7, color=COLORS["muted"])

    ax.text(0.555, 0.365, "electronic polarization", ha="center", va="center", fontsize=13.0, weight="bold")
    ax.text(0.555, 0.285, r"$\Pi(q,\mathrm{i}\nu)\leqslant 0$", ha="center", va="center", fontsize=14.8)
    ax.text(0.555, 0.185, "Lindhard or GG bubble", ha="center", va="center", fontsize=10.4, color=COLORS["muted"])

    ax.text(0.887, 0.655, "screened interaction", ha="center", va="center", fontsize=11.6, weight="bold")
    ax.text(
        0.887,
        0.535,
        r"$W=\dfrac{V_{\mathrm{tot}}}{1-\Pi V_{\mathrm{tot}}}$",
        ha="center",
        va="center",
        fontsize=16.6,
    )

    ax.text(
        0.50,
        0.070,
        r"Convention: $\Pi\leqslant 0$, so $1-\Pi V_{\mathrm{tot}}=1+|\Pi|V_{\mathrm{tot}}$ for static screening.",
        ha="center",
        va="center",
        fontsize=11.0,
        color=COLORS["muted"],
    )

    arrow(ax, (0.347, 0.733), (0.378, 0.675), ms=14, lw=1.7)
    arrow(ax, (0.347, 0.427), (0.378, 0.635), ms=14, lw=1.7)
    arrow(ax, (0.730, 0.675), (0.765, 0.600), ms=14, lw=1.7)
    arrow(ax, (0.730, 0.300), (0.765, 0.505), ms=14, lw=1.7)

    save(fig, "w_construction_chain.png")


def draw_radial_convolution_schematic() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.text(0.22, 0.88, "Angular geometry", ha="center", va="center", fontsize=16, weight="bold")
    origin = np.array([0.08, 0.25])
    k_end = np.array([0.35, 0.25])
    p_end = np.array([0.27, 0.62])
    arrow(ax, origin, k_end, color=COLORS["blue"], ms=19, lw=2.8)
    arrow(ax, origin, p_end, color=COLORS["orange"], ms=19, lw=2.8)
    arrow(ax, p_end, k_end, color=COLORS["teal"], ms=19, lw=2.8)
    ax.plot(origin[0], origin[1], "o", color=COLORS["ink"], ms=10)
    ax.text(0.215, 0.155, r"target $k$", color=COLORS["blue"], fontsize=14, ha="center", va="center")
    ax.text(
        0.163,
        0.535,
        r"source $p$",
        color=COLORS["orange"],
        fontsize=13.5,
        rotation=43,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )
    ax.text(
        0.345,
        0.510,
        r"$q=|k-p|$",
        color=COLORS["teal"],
        fontsize=13.5,
        rotation=-69,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )
    ax.add_patch(Arc(origin, 0.12, 0.12, theta1=0, theta2=42, lw=1.1, color=COLORS["muted"]))
    ax.text(0.142, 0.320, r"$\theta$", color=COLORS["muted"], fontsize=12.5, ha="center")
    ax.text(
        0.22,
        0.055,
        "Integrating over angle sweeps q through an interval.",
        ha="center",
        va="center",
        fontsize=11.8,
        color=COLORS["muted"],
    )

    ax.text(0.74, 0.88, "Radial interval", ha="center", va="center", fontsize=16, weight="bold")
    y = 0.52
    x0, x1, x2 = 0.61, 0.75, 0.89
    ax.plot([0.57, 0.94], [y, y], color=COLORS["ink"], lw=1.7)
    for x in (x0, x1, x2):
        ax.plot([x, x], [y - 0.045, y + 0.045], color=COLORS["ink"], lw=1.7)
    arrow(ax, (x0, y + 0.13), (x2, y + 0.13), color=COLORS["teal"], ms=15, lw=2.1, style="<|-|>")
    ax.text(
        0.76,
        y + 0.215,
        r"$\int_{|k-p|}^{k+p} \mathrm{d}q\,qW(q)$",
        color=COLORS["teal"],
        fontsize=15.5,
        ha="center",
        va="center",
    )
    ax.text(x0, y - 0.105, r"$|k-p|$", fontsize=13.5, ha="center", va="center")
    ax.text(x1, y - 0.105, r"$q$", fontsize=13.5, ha="center", va="center")
    ax.text(x2, y - 0.105, r"$k+p$", fontsize=13.5, ha="center", va="center")

    cell_y = 0.195
    cell_h = 0.13
    cell = Rectangle((0.59, cell_y), 0.32, cell_h, linewidth=1.5, edgecolor=COLORS["orange"], facecolor="#fff8ef")
    ax.add_patch(cell)
    ax.plot([0.59, 0.91], [cell_y + 0.5 * cell_h, cell_y + 0.5 * cell_h], color=COLORS["orange"], lw=1.1)
    ax.plot([x1, x1], [cell_y, cell_y + cell_h], color=COLORS["red"], lw=2.2)
    ax.text(
        x1 + 0.035,
        cell_y + cell_h + 0.040,
        r"$k=p$",
        color=COLORS["red"],
        fontsize=12.8,
        ha="left",
        va="center",
    )
    ax.text(
        0.75,
        0.080,
        "diagonal cell average replaces the singular point block",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    save(fig, "radial_convolution_schematic.png")


def _representative_grids():
    params = model.STOParams()
    n_bohr3 = model.n_cm3_to_bohr3(1.0e20)
    a = params.band_a_ha_bohr2
    mu0 = model.get_mu(n_bohr3, beta=np.inf, a=a)
    k_f = np.sqrt(mu0 / a)
    k_ph = np.sqrt((max(mu0, 0.0) + model.phonon_grid_ha(params)) / a)
    kmax = max(4.0 * k_f, 2.0 * k_ph)
    qmax = 2.0 * kmax
    q_grid, _ = grid.get_q_grid(
        0.5 * qmax,
        k_F=k_f,
        nq=500,
        frac_log=0.35,
        n_2kf_min=101,
    )
    k_grid, _, info = grid.get_k_grid(
        k_f,
        params.beta_ha(1.0) ** -1,
        q_grid,
        kmax,
        lambda k: model.e(k, a) - mu0,
        n_pad=5,
        n_left=200,
        n_right=200,
        n_tail=0,
        n_shell_min=50,
        n_shell_max=5000,
    )
    return k_grid / k_f, q_grid / k_f, np.asarray(info["k_shell_used"]) / k_f


def _style_grid_axis(ax, xlim, label):
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#c8d3df")
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", color="#dce4ee", lw=0.8, alpha=0.9)
    ax.text(0.985, 0.12, label, transform=ax.transAxes, ha="right", va="center", fontsize=17, color=COLORS["muted"])


def draw_kq_grid_schematic() -> None:
    set_style()
    k_grid, q_grid, shell = _representative_grids()
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 4.95), gridspec_kw={"hspace": 0.48})

    ax = axes[0]
    _style_grid_axis(ax, (-0.05, 4.15), r"$k/k_{\mathrm{F}}$")
    ax.text(0.02, 0.91, "active k grid", transform=ax.transAxes, fontsize=16, weight="bold", color=COLORS["ink"])
    ax.hlines(0.36, 0.0, 4.0, color=COLORS["ink"], lw=1.4)
    ax.vlines(k_grid, 0.29, 0.43, color=COLORS["ink"], alpha=0.36, lw=0.55)
    ax.axvspan(shell[0], shell[1], ymin=0.23, ymax=0.67, facecolor=COLORS["blue_fill"], edgecolor=COLORS["blue"], lw=1.2)
    ax.vlines(1.0, 0.22, 0.66, color=COLORS["red"], lw=2.0)
    ax.text(0.955, 0.70, r"$k_{\mathrm{F}}$", color=COLORS["red"], fontsize=15, ha="left")
    ax.annotate(
        "Fermi-shell\nGauss points",
        xy=(1.03, 0.51),
        xytext=(1.55, 0.76),
        fontsize=12.8,
        color=COLORS["blue"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["blue"], "lw": 1.4},
        ha="left",
        va="center",
    )
    inset = ax.inset_axes([0.62, 0.64, 0.31, 0.23])
    inset.set_xlim(0.988, 1.012)
    inset.set_ylim(0.0, 1.0)
    inset.set_facecolor("#f8fafc")
    inset.axvspan(shell[0], shell[1], ymin=0.17, ymax=0.84, facecolor="#d8e8fb", edgecolor=COLORS["blue"], lw=1.0)
    inset.hlines(0.45, 0.988, 1.012, color=COLORS["ink"], lw=1.0)
    in_shell = k_grid[(k_grid >= 0.988) & (k_grid <= 1.012)]
    inset.vlines(in_shell, 0.26, 0.60, color=COLORS["blue"], alpha=0.75, lw=0.55)
    inset.vlines(1.0, 0.18, 0.84, color=COLORS["red"], lw=1.5)
    inset.set_xticks([0.99, 1.00, 1.01])
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=8.8, pad=1)
    for spine in inset.spines.values():
        spine.set_color("#9bbce0")
    inset.text(
        0.0,
        1.10,
        "Fermi-shell zoom",
        transform=inset.transAxes,
        fontsize=9.6,
        color=COLORS["ink"],
        ha="left",
        va="bottom",
        clip_on=False,
    )

    ax = axes[1]
    _style_grid_axis(ax, (-0.10, 8.35), r"$q/k_{\mathrm{F}}$")
    ax.text(0.02, 0.92, r"$q$ grid for $W(q)$", transform=ax.transAxes, fontsize=16, weight="bold", color=COLORS["ink"])
    ax.hlines(0.34, 0.0, 8.0, color=COLORS["ink"], lw=1.4)
    ax.vlines(q_grid, 0.27, 0.41, color=COLORS["ink"], alpha=0.38, lw=0.55)
    ax.axvspan(1.8, 2.2, ymin=0.17, ymax=0.58, facecolor="#dff5ef", edgecolor=COLORS["teal"], lw=1.2)
    ax.vlines(2.0, 0.10, 0.66, color=COLORS["orange"], lw=2.0)
    ax.text(2.05, 0.665, r"$2k_{\mathrm{F}}$", color=COLORS["orange"], fontsize=14.5, ha="center")
    ax.annotate(
        "$2k_{\\mathrm{F}}$ block\nminimum",
        xy=(1.80, 0.55),
        xytext=(1.08, 0.76),
        fontsize=11.4,
        color=COLORS["teal"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["teal"], "lw": 1.4},
        ha="left",
        va="center",
    )
    ax.annotate(
        "clustered near the\nLindhard $2k_{\\mathrm{F}}$ feature",
        xy=(2.18, 0.47),
        xytext=(3.05, 0.74),
        fontsize=12.8,
        color=COLORS["teal"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["teal"], "lw": 1.4},
        ha="left",
        va="center",
    )

    save(fig, "kq_grid_schematic.png", dpi=220)


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    draw_w_construction_chain()
    draw_radial_convolution_schematic()
    draw_kq_grid_schematic()


if __name__ == "__main__":
    main()
