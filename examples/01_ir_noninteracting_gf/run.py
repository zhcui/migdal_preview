#!/usr/bin/env python

"""Tutorial: sparse-IR round trip for a noninteracting Green's function.

Run from the repository root:

    python examples/01_ir_noninteracting_gf/run.py

This example starts with a known Green's function,

    G0(k, iw) = 1 / (iw - (e_k - mu)),

transforms it from Matsubara frequency to imaginary time, and transforms it
back.  If the sparse-IR grid is set up correctly, the reconstructed G0 should
match the original one to nearly machine precision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402
from migdal.freq import FreqGrid  # noqa: E402


def parse_args(argv=None):
    """Parse sparse-IR tutorial arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=2500.0)
    parser.add_argument("--lamb", type=float, default=1.0e3)
    parser.add_argument("--tol", type=float, default=1.0e-12)
    parser.add_argument("--assert-atol", type=float, default=1.0e-9)
    common.add_output_arg(parser, __file__)
    return parser.parse_args(argv)


def main(argv=None):
    """Run the noninteracting sparse-IR round-trip tutorial."""
    args = parse_args(argv)
    out_dir = common.prepare_out_dir(args.out_dir)

    # 1. Build the sparse-IR context: sampled fermion frequencies, sampled
    #    imaginary times, and the transforms between them.
    ctx = FreqGrid(beta=args.beta, lamb=args.lamb, tol=args.tol, verbose=0)

    # 2. Define a tiny one-band toy spectrum in Hartree.
    e_k = np.array([-0.30, -0.08, 0.0, 0.07, 0.25])
    mu = 0.015

    # 3. Evaluate G0 on the sparse Matsubara frequency samples, move to tau,
    #    and come back to frequency space.
    G_w = ctx.get_G0(e_k=e_k, mu=mu)
    G_t = ctx.w2t_f(G_w)
    G_re = ctx.t2w_f(G_t)

    # 4. The round-trip error is the whole point of the example.
    err = np.abs(G_re - G_w)
    max_abs = float(np.max(err))
    max_rel = float(max_abs / max(np.max(np.abs(G_w)), 1.0e-300))
    if max_abs > args.assert_atol:
        raise AssertionError(
            "IR round-trip error %.6e exceeds %.6e" % (max_abs, args.assert_atol)
        )

    csv_path = out_dir / "ir_roundtrip_error.csv"
    summary_path = out_dir / "ir_roundtrip_summary.txt"
    fig_path = out_dir / "ir_roundtrip_error.png"

    np.savetxt(
        csv_path,
        np.column_stack([ctx.ws, np.max(err, axis=0)]),
        header="omega_Ha max_abs_error",
    )
    report = [
        "Example 01: sparse-IR round trip for G0",
        "",
        "Model:",
        "  G0(k, iw) = 1 / (iw - (e_k - mu))",
        "  e_k [Ha] = %s" % np.array2string(e_k, precision=3),
        "  mu  [Ha] = %.6g" % mu,
        "",
        "Sparse-IR grid:",
        "  beta = %.6g Ha^-1, lambda = %.6g, tol = %.1e"
        % (args.beta, args.lamb, args.tol),
        "  %d fermion frequencies, %d fermion tau points" % (ctx.nw, ctx.nt_f),
        "",
        "Round trip:",
        "  G0(iw) -> G0(tau) -> G0(iw)",
        "  max absolute error = %.6e" % max_abs,
        "  max relative error = %.6e" % max_rel,
        "",
        "Files:",
        "  ir_roundtrip_summary.txt",
        "  ir_roundtrip_error.csv",
        "  ir_roundtrip_error.png",
    ]
    common.write_lines(summary_path, report)

    plt = common.pyplot(out_dir)
    per_freq_err = np.max(err, axis=0)
    omega_abs = np.abs(ctx.ws)
    order = np.argsort(omega_abs)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(
        omega_abs[order],
        per_freq_err[order],
        s=28,
        color="#2f6f8f",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=r"sparse-IR samples",
    )
    ax.axhline(
        args.assert_atol,
        color="#b65f5f",
        linestyle="--",
        linewidth=1.1,
        label=r"check tolerance",
    )
    ax.set_xlabel(r"fermionic Matsubara frequency $|\omega_n|$ [Ha]")
    ax.set_ylabel(r"$\max_k |\tilde{G}_0(k,\mathrm{i}\omega_n)-G_0(k,\mathrm{i}\omega_n)|$")
    ax.set_title(r"Sparse-IR round-trip error for $G_0$")
    ax.grid(True, which="major", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#e4e7ec", linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False)
    ax.text(
        0.03,
        0.9,
        "\n".join(
            [
                r"$G_0(k,\mathrm{i}\omega_n)=\frac{1}{\mathrm{i}\omega_n-(\varepsilon_k-\mu)}$",
                r"$\tilde{G}_0=\mathcal{T}_{\tau\to\omega}\mathcal{T}_{\omega\to\tau}G_0$",
                r"$\max|\tilde{G}_0-G_0|=%.2e$" % max_abs,
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d0d5dd",
            "alpha": 0.96,
        },
    )
    fig.savefig(fig_path, dpi=200)

    print("\n".join(report + ["", "Output directory:", "  %s" % out_dir]), flush=True)


if __name__ == "__main__":
    main()
