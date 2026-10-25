"""Small shared helpers for Migdal examples."""

from __future__ import annotations

from pathlib import Path
import inspect
import os

import numpy as np

from migdal import memory, model


GW0_KWARG_NAMES = {
    "gw0_scf_max_cycle": "scf_max_cycle",
    "gw0_scf_conv_tol": "scf_conv_tol",
    "gw0_scf_damp": "scf_damp",
    "gw0_scf_diis_space": "scf_diis_space",
    "gw0_scf_diis_start_cycle": "scf_diis_start_cycle",
    "gw0_continuation_max_dex": "scf_continuation_max_dex",
    "scf_fs_pin": "scf_fs_pin",
}
SCGW_KWARG_NAMES = {
    "scgw_n_tail": "n_tail",
    "scgw_scf_max_cycle": "scf_max_cycle",
    "scgw_scf_conv_tol": "scf_conv_tol",
    "scgw_scf_damp": "scf_damp",
    "scgw_scf_diis_space": "scf_diis_space",
    "scgw_scf_diis_start_cycle": "scf_diis_start_cycle",
    "scgw_continuation_max_dex": "scf_continuation_max_dex",
    "scf_fs_pin": "scf_fs_pin",
    "sigma_tail_policy": "sigma_tail_policy",
    "pi_imag_tol": "pi_imag_tol",
    "pi_imag_policy": "pi_imag_policy",
    "allow_no_pi_tail": "allow_no_pi_tail",
}


def add_density_args(parser, default_n=6):
    """Add the shared density-scan arguments to an example parser."""
    parser.add_argument(
        "--densities",
        default=None,
        help="Comma-separated densities in cm^-3. Overrides --n-density.",
    )
    parser.add_argument(
        "--n-density",
        type=int,
        default=default_n,
        help="Number of log-spaced density points from 1e23 to 1e16 cm^-3.",
    )


def density_points(args):
    """Return example density points in cm^-3 from parsed arguments."""
    if args.densities:
        return model.density_grid(args.densities)
    if args.n_density < 2:
        raise ValueError("--n-density must be at least 2.")
    return np.geomspace(1.0e23, 1.0e16, int(args.n_density))


def add_output_arg(parser, script_file):
    """Add a shared ``--out-dir`` argument defaulting beside ``script_file``."""
    default = Path(script_file).resolve().parent
    parser.add_argument(
        "--out-dir",
        default=str(default),
        help="Directory for CSV, text summaries, and figures.",
    )


def prepare_out_dir(path):
    """Create and return an absolute output directory path."""
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pyplot(out_dir):
    """Return matplotlib pyplot configured for noninteractive file output."""
    mpl_dir = Path(os.environ.get("MIGDAL_MPLCONFIGDIR", "/tmp/migdal_mplconfig"))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def add_radial_args(parser, default_ir_file=None, ir_help=None):
    """Add shared sparse-IR, radial-grid, and gap-solver CLI options."""
    if default_ir_file is None:
        default_ir_file = legacy_ir_file()
    if ir_help is None:
        ir_help = (
            "Sparse-IR h5 file. Use an existing examples/ir.h5 cache to avoid "
            "regeneration."
        )
    parser.add_argument("--lamb", type=float, default=1.0e9)
    parser.add_argument(
        "--ir-file",
        default=str(default_ir_file),
        help=ir_help,
    )
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--k-pad", type=int, default=5)
    parser.add_argument("--k-left", type=int, default=200)
    parser.add_argument("--k-right", type=int, default=200)
    parser.add_argument("--k-shell-min", type=int, default=50)
    parser.add_argument("--k-shell-max", type=int, default=5000)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-6)
    parser.add_argument("--davidson-max-cycle", type=int, default=300)
    parser.add_argument("--davidson-max-space", type=int, default=20)
    parser.add_argument("--n0-check-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--q-interp-mode",
        choices=("legacy", "q_singular"),
        default="q_singular",
        help=(
            "Radial W primitive mode. Defaults to q_singular, which handles "
            "Coulomb tails and needs diagonal cell averages."
        ),
    )
    parser.add_argument(
        "--q-singular-head-policy",
        choices=("pi_head", "fit_w"),
        default="pi_head",
        help=(
            "Source for q_singular screened Coulomb head; fit_w fits q^2 W "
            "directly from sampled low-q rows."
        ),
    )
    parser.add_argument(
        "--diag-average-mode",
        choices=("auto", "gauss", "analytic"),
        default="auto",
        help=(
            "Diagonal cell averaging rule; auto uses analytic for q_singular "
            "and gauss otherwise."
        ),
    )
    parser.add_argument(
        "--precond",
        choices=("even_diag", "ones"),
        default="even_diag",
    )
    parser.add_argument("--cache", choices=("auto", "off"), default="auto")
    parser.add_argument(
        "--max-memory",
        type=float,
        default=memory.DEFAULT_MAX_MEMORY,
        help="PySCF-style per-rank memory limit in MB.",
    )


def radial_kwargs(args):
    """Translate shared parsed radial options to driver keyword arguments."""
    return {
        "lamb": args.lamb,
        "ir_file": args.ir_file,
        "nq": args.nq,
        "kgrid_n_pad": args.k_pad,
        "kgrid_n_left": args.k_left,
        "kgrid_n_right": args.k_right,
        "kgrid_n_shell_min": args.k_shell_min,
        "kgrid_n_shell_max": args.k_shell_max,
        "davidson_tol": args.davidson_tol,
        "davidson_max_cycle": args.davidson_max_cycle,
        "davidson_max_space": args.davidson_max_space,
        "precond": args.precond,
        "cache": args.cache,
        "max_memory": args.max_memory,
        "n0_check_atol": args.n0_check_atol,
        "q_interp_mode": args.q_interp_mode,
        "q_singular_head_policy": args.q_singular_head_policy,
        "diag_average_mode": args.diag_average_mode,
        "verbose": 0,
    }


def legacy_ir_file():
    """Return the repository-level sparse-IR cache path used by examples."""
    return Path(__file__).resolve().parent / "ir.h5"


def rows_to_array(rows):
    """Convert driver ``PointResult`` rows to a compact numeric table."""
    return np.asarray(
        [
            (
                row.n_cm3,
                row.lam,
                int(row.gap_converged),
                -1 if row.scf_converged is None else int(row.scf_converged),
                -1 if row.scf_cycles is None else int(row.scf_cycles),
                row.nk,
                row.nq,
            )
            for row in rows
        ],
        dtype=float,
    )


def save_curve(path, curves):
    """Write one or more density curves with shared density column."""
    cols = []
    names = []
    fmts = []
    for name, rows in curves:
        arr = rows_to_array(rows)
        if not cols:
            cols.append(arr[:, 0])
            names.append("n_cm3")
            fmts.append("%.18e")
        cols.extend(
            [arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]]
        )
        names.extend(
            [
                f"{name}_lambda",
                f"{name}_gap_converged",
                f"{name}_scf_converged",
                f"{name}_scf_cycles",
                f"{name}_nk",
                f"{name}_nq",
            ]
        )
        fmts.extend(["%.18e", "%d", "%d", "%d", "%d", "%d"])
    save_csv(path, np.column_stack(cols), names, fmt=fmts)


def save_csv(path, data, names, fmt="%.18e"):
    """Write a small numeric table as a comma-separated CSV file."""
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="", fmt=fmt)


def write_lines(path, lines):
    """Write text lines with a trailing newline."""
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank0_print(rank, msg):
    """Print ``msg`` only from MPI rank 0."""
    if rank == 0:
        print(msg, flush=True)


def changed_kwargs(args, defaults, names):
    """Return driver kwargs whose parsed values differ from driver defaults."""
    kwargs = {}
    for arg_name, default in defaults.items():
        if not hasattr(args, arg_name):
            continue
        value = getattr(args, arg_name)
        if value != default:
            kwargs[names[arg_name]] = value
    return kwargs


def driver_defaults(cls, names):
    """Return parser-argument defaults from a driver constructor signature."""
    params = inspect.signature(cls.__init__).parameters
    defaults = {}
    for arg_name, kw_name in names.items():
        param = params.get(kw_name)
        if param is None or param.default is inspect.Parameter.empty:
            raise KeyError("%s is not a defaulted constructor argument" % kw_name)
        defaults[arg_name] = param.default
    return defaults


def changed_driver_kwargs(args, driver):
    """Return parsed kwargs that differ from the named driver's defaults."""
    if driver == "gw0":
        from migdal.gw0 import GW0

        names = GW0_KWARG_NAMES
        defaults = driver_defaults(GW0, names)
    elif driver == "scgw":
        from migdal.scgw import ScGW

        names = SCGW_KWARG_NAMES
        defaults = driver_defaults(ScGW, names)
    else:
        raise ValueError("unknown driver %r" % driver)
    return changed_kwargs(args, defaults, names)
