#!/usr/bin/env python

"""Compare screening and phonon controls for the density curves.

This example starts from the same no-truncation radial settings as example 04,
then adds compact controls that isolate the roles of RPA screening and the
phonon-mediated part of the interaction.

Run from the repository root, preferably under MPI:

    mpirun -n 2 python examples/05_screening_phonon_controls/run.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402
from migdal import model  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
BASELINE_DIR = SCRIPT_DIR.parent / "04_g0w0_gw0_scgw_curve"
BASELINE_CSV = BASELINE_DIR / "g0w0_gw0_scgw_curve.csv"
BASELINE_METHODS = ("G0W0", "GW0", "scGW")
LAMBDA_CSV = "screening_phonon_controls.csv"
Z_CSV = "screening_phonon_controls_z.csv"
FIXED_PI_VALUE = 0.0
Z_COLUMNS = (
    "method",
    "n_cm3",
    "Z_kF_i0",
    "Z_table_min",
    "Z_table_max",
    "sigma_symm_err",
    "sigma_source",
    "gap_converged",
    "scf_converged",
    "scf_cycles",
    "nk",
    "nq",
)
METHOD_LABELS = {
    "G0W0": r"$G_0W_0$",
    "GW0": r"$GW_0$",
    "scGW": "scGW",
    "G0V": r"$G_0V$",
    "GV": r"$GV$",
    "G0W0_no_phonon": r"$G_0W_0$, no phonon",
    "G0V_no_phonon": r"$G_0V$, no phonon",
    "GW0_no_phonon": r"$GW_0$, no phonon",
    "GV_no_phonon": r"$GV$, no phonon",
    "scGW_no_phonon": "scGW, no phonon",
}
METHOD_NOTES = {
    "G0W0": "fixed G0 and fixed RPA W0 from example 04",
    "GW0": "self-consistent normal-state G with fixed RPA W0",
    "scGW": "self-consistent normal-state G and W",
    "G0V": "one-shot G0 run with Pi fixed to 0",
    "GV": "control-method run with Pi fixed to 0",
    "G0W0_no_phonon": "one-shot run with Vph(q, i nu)=1",
    "G0V_no_phonon": "one-shot run with Pi=0 and Vph(q, i nu)=1",
    "GW0_no_phonon": "GW0 run with Vph(q, i nu)=1",
    "GV_no_phonon": "control-method run with Pi=0 and Vph(q, i nu)=1",
    "scGW_no_phonon": "scGW run with Vph(q, i nu)=1",
}
METHOD_STYLES = {
    "G0W0": ("#2f6f8f", "o", "-"),
    "GW0": ("#5f8d4e", "s", "-"),
    "scGW": ("#b65f5f", "^", "-"),
    "G0V": ("#d18f2f", "*", ":"),
    "GV": ("#7a4d9b", "D", "--"),
    "G0W0_no_phonon": ("#777777", "x", ":"),
    "G0V_no_phonon": ("#a8892c", "*", "-."),
    "GW0_no_phonon": ("#4d8f8b", "v", "--"),
    "GV_no_phonon": ("#b45a8a", "D", "-."),
    "scGW_no_phonon": ("#4b7fb9", "P", "--"),
}
def default_ir_file():
    """Return the preferred sparse-IR cache path for this example."""
    candidates = (
        BASELINE_DIR / "ir.h5",
        BASELINE_DIR / "ir.5",
        SCRIPT_DIR / "ir.h5",
        SCRIPT_DIR.parent / "ir.h5",
    )
    for path in candidates:
        if path.exists():
            return path
    return SCRIPT_DIR / "ir.h5"


def no_phonon_params(params=None):
    """Return STO parameters with ``Vph(q, nu) == 1``."""
    params = model.STOParams() if params is None else params
    return replace(params, eps_0=params.eps_inf)


def parse_args(argv=None):
    """Parse screening/phonon control-curve arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_density_args(parser, default_n=6)
    common.add_radial_args(
        parser,
        default_ir_file=default_ir_file(),
        ir_help=(
            "Sparse-IR h5 cache file. Defaults to example 04's ir.h5/ir.5, "
            "then this example's ir.h5."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--diag-ncell", type=int, default=12)
    parser.add_argument(
        "--control-method",
        choices=("G0W0", "GW0", "scGW"),
        default="GW0",
        help="Method used for the GV and no-phonon control curves.",
    )
    parser.add_argument(
        "--all-controls",
        action="store_true",
        help=(
            "Also write the extra one-shot G0V and no-phonon/fixed-Pi "
            "combinations. The default keeps the plot compact."
        ),
    )
    parser.add_argument(
        "--gw0-scf-max-cycle",
        type=int,
        default=300,
        help="GW0 control SCF cycle limit.",
    )
    parser.add_argument("--gw0-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--gw0-continuation-max-dex",
        type=float,
        default=1.0,
        help=(
            "Maximum density step, in decades, for inserted GW0 continuation "
            "points when --restart-scf is enabled."
        ),
    )
    parser.add_argument(
        "--scgw-nq",
        type=int,
        default=500,
        help="scGW q-grid size; defaults to the example-04 baseline.",
    )
    parser.add_argument("--scgw-scf-max-cycle", type=int, default=300)
    parser.add_argument("--scgw-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--scgw-scf-damp", type=float, default=0.98)
    parser.add_argument("--scgw-scf-diis-space", type=int, default=8)
    parser.add_argument("--scgw-scf-diis-start-cycle", type=int, default=0)
    parser.add_argument(
        "--scgw-continuation-max-dex",
        type=float,
        default=1.0,
        help=(
            "Maximum density step, in decades, for inserted scGW continuation "
            "points when --restart-scf is enabled."
        ),
    )
    parser.add_argument(
        "--restart-scf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Warm-start GW0/scGW density scans from the previous density point.",
    )
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


def radial_kwargs(args, nq=None):
    """Return radial driver kwargs using the example-04 baseline defaults."""
    kwargs = common.radial_kwargs(args)
    kwargs.update(
        save_ir=True,
        verbose=5,
        diag_ncell=args.diag_ncell,
    )
    if args.temperature != 1.0:
        kwargs["temperature"] = args.temperature
    if nq is not None:
        kwargs["nq"] = nq
    return kwargs


def display_path(path):
    """Return a readable path without machine-specific prefixes."""
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, Path(__file__).resolve().parents[2])


def density_summary(densities):
    """Return one readable line describing the density grid."""
    densities = np.asarray(densities, dtype=float)
    return "%d points from %.3e down to %.3e cm^-3" % (
        len(densities),
        float(np.max(densities)),
        float(np.min(densities)),
    )


def uses_scf_restart(restart_scf):
    """Return whether GW0/scGW density scans should warm-start SCF."""
    return bool(restart_scf)


def gw0_kwargs(args):
    """Return GW0 controls, omitting values already equal to driver defaults."""
    kwargs = common.changed_driver_kwargs(args, "gw0")
    if not args.fail_unconverged:
        kwargs["allow_unconverged"] = True
    return kwargs


def scgw_kwargs(args):
    """Return scGW controls, omitting values already equal to driver defaults."""
    kwargs = common.changed_driver_kwargs(args, "scgw")
    if not args.fail_unconverged:
        kwargs["allow_unconverged"] = True
    return kwargs


def default_control_names(control_method):
    """Return the compact default control-curve names in plotting order."""
    names = ["GV", "G0W0_no_phonon"]
    control_no_phonon = "%s_no_phonon" % control_method
    if control_no_phonon not in names:
        names.append(control_no_phonon)
    return names


def _flag_from_csv(value):
    """Return a bool/None convergence flag from an example CSV field."""
    ival = int(float(value))
    if ival < 0:
        return None
    return bool(ival)


def _loaded_row(row, method):
    """Build the small row object consumed by shared example output helpers."""
    prefix = "%s_" % method
    return SimpleNamespace(
        n_cm3=float(row["n_cm3"]),
        lam=float(row[prefix + "lambda"]),
        gap_converged=bool(int(float(row[prefix + "gap_converged"]))),
        scf_converged=_flag_from_csv(row[prefix + "scf_converged"]),
        scf_cycles=None
        if int(float(row[prefix + "scf_cycles"])) < 0
        else int(float(row[prefix + "scf_cycles"])),
        nk=int(float(row[prefix + "nk"])),
        nq=int(float(row[prefix + "nq"])),
    )


def _match_density_rows(rows, densities):
    """Return baseline CSV rows matching ``densities`` in requested order."""
    out = []
    used = set()
    for density in np.asarray(densities, dtype=float):
        match = None
        for idx, row in enumerate(rows):
            if idx in used:
                continue
            if np.isclose(float(row["n_cm3"]), density, rtol=1.0e-10, atol=0.0):
                match = idx
                break
        if match is None:
            return None
        used.add(match)
        out.append(rows[match])
    return out


def load_example4_baseline(densities, path=BASELINE_CSV):
    """Load G0W0/GW0/scGW rows from example 04 when its CSV is available."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    required = ["n_cm3"]
    for method in BASELINE_METHODS:
        required.extend(
            [
                "%s_lambda" % method,
                "%s_gap_converged" % method,
                "%s_scf_converged" % method,
                "%s_scf_cycles" % method,
                "%s_nk" % method,
                "%s_nq" % method,
            ]
        )
    if any(name not in rows[0] for name in required):
        return None
    matched = _match_density_rows(rows, densities)
    if matched is None:
        return None
    return [
        (method, [_loaded_row(row, method) for row in matched])
        for method in BASELINE_METHODS
    ]


def fixed_pi_w(ctx, params, q_grid, pi_value, denom_policy, denom_min, *,
               q_interp_mode="legacy", q_singular_head_policy="pi_head"):
    """Build a fixed-polarization W table for the GV control.

    With the repository convention ``W = Vtot / (1 - Pi Vtot)``,
    ``Pi = 0`` gives the unscreened interaction ``W = Vtot``.
    """
    from migdal import radial

    q_grid = np.asarray(q_grid, dtype=float)
    q_singular = q_interp_mode == "q_singular"
    vc_tol = np.nextafter(0.0, 1.0) if q_singular else 1.0e-10
    Vc = model.get_Vc(q_grid, params.eps_inf, q0_treat=False, tol=vc_tol)
    Vph = model.get_Vph_q(q_grid, ctx.vs, params)
    Vtot = Vc[:, None] * Vph
    Pi = np.full_like(Vtot, float(pi_value), dtype=float)
    static_head = 4.0 * np.pi / float(params.eps_inf) if q_singular else None
    vtot_head = None
    if q_singular:
        vph0 = model.get_Vph_q(np.array([0.0]), ctx.vs, params)[0]
        vtot_head = static_head * vph0
    return radial.get_W_from_Pi(
        ctx,
        Pi,
        q_grid,
        Vc,
        Vtot,
        denom_policy=denom_policy,
        denom_min=denom_min,
        q_interp_mode=q_interp_mode,
        q_singular_head_policy=q_singular_head_policy,
        static_singular_coeff=static_head,
        vtot_head=vtot_head,
        return_metrics=True,
    )


class FixedPiG0W0Mixin:
    """Replace the G0W0 W table by one built from fixed Pi."""

    def __init__(self, *args, fixed_pi_value=FIXED_PI_VALUE, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_pi_value = float(fixed_pi_value)

    def _extra_flag_rows(self):
        rows = list(super()._extra_flag_rows())
        rows.append(("fixed Pi value", "%.6g" % self.fixed_pi_value))
        return rows

    def _setup_point(self, n_cm3):
        from migdal import radial

        data = super()._setup_point(n_cm3)
        W, denom = fixed_pi_w(
            self.ctx,
            self.params,
            data["q_grid"],
            self.fixed_pi_value,
            self.denom_policy,
            self.denom_min,
            q_interp_mode=self.q_interp_mode,
            q_singular_head_policy=self.q_singular_head_policy,
        )
        data["W"] = W
        data["denom"] = denom
        diag_average = model.bool_auto(self.diag_average, default=not self.truncate)
        data["diag_cells"] = None
        if diag_average:
            data["diag_cells"] = radial.get_diag_cells(
                None,
                W,
                data["k_grid"],
                data["k_wts"],
                data["kmax"],
                self.diag_ncell,
                cell_mode=self.diag_average_mode,
            )
        return data


class FixedPiScGWMixin:
    """Replace each scGW W update by one built from fixed Pi."""

    def __init__(self, *args, fixed_pi_value=FIXED_PI_VALUE, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_pi_value = float(fixed_pi_value)

    def _extra_flag_rows(self):
        rows = list(super()._extra_flag_rows())
        rows.append(("fixed Pi value", "%.6g" % self.fixed_pi_value))
        return rows

    def _update_W(
        self, data, sigma, mu, sigma_tail, fs_pin_shift=0.0,
        collect_pi_metrics=True, emit_pi_warnings=True,
    ):
        from migdal import radial

        W, denom_metrics = fixed_pi_w(
            self.ctx,
            self.params,
            data["q_grid"],
            self.fixed_pi_value,
            self.denom_policy,
            self.denom_min,
            q_interp_mode=self.q_interp_mode,
            q_singular_head_policy=self.q_singular_head_policy,
        )
        diag_average = data.get(
            "diag_average", model.bool_auto(self.diag_average, default=True)
        )
        diag_cells = None
        if diag_average:
            diag_helper = getattr(self, "_diag_cells_for_W_update", None)
            if diag_helper is not None:
                diag_cells = diag_helper(data, W)
            else:
                diag_cells = radial.get_diag_cells(
                    None,
                    W,
                    data["k_scf"],
                    data["k_wts_scf"],
                    data["kmax"],
                    self.diag_ncell,
                    geometry=data.get("diag_geometry"),
                    cell_mode=self.diag_average_mode,
                )
        from migdal.scgw import PiMetrics

        return W, diag_cells, PiMetrics(), denom_metrics


def make_g0w0_class():
    """Return a G0W0 subclass that uses the fixed-Pi control interaction."""
    from migdal.g0w0 import G0W0

    class FixedPiG0W0(FixedPiG0W0Mixin, G0W0):
        pass

    return FixedPiG0W0


def make_gw0_class():
    """Return a GW0 subclass that uses the fixed-Pi control interaction."""
    from migdal.gw0 import GW0

    class FixedPiGW0(FixedPiG0W0Mixin, GW0):
        pass

    return FixedPiGW0


def make_scgw_class():
    """Return an scGW subclass that uses the fixed-Pi control interaction."""
    from migdal.scgw import ScGW

    class FixedPiScGW(FixedPiScGWMixin, ScGW):
        pass

    return FixedPiScGW


def make_driver(method, args, densities, params=None, fixed_pi=False):
    """Construct a physical or control driver for one method."""
    from migdal.g0w0 import G0W0
    from migdal.scgw import ScGW

    kwargs = radial_kwargs(args, nq=args.scgw_nq if method == "scGW" else None)
    kwargs["densities"] = densities
    if params is not None:
        kwargs["params"] = params
    if fixed_pi:
        kwargs["fixed_pi_value"] = FIXED_PI_VALUE

    if method == "G0W0":
        cls = make_g0w0_class() if fixed_pi else G0W0
        return cls(**kwargs)
    if method == "GW0":
        from migdal.gw0 import GW0

        cls = make_gw0_class() if fixed_pi else GW0
        return cls(**kwargs, **gw0_kwargs(args))
    if method == "scGW":
        cls = make_scgw_class() if fixed_pi else ScGW
        return cls(**kwargs, **scgw_kwargs(args))
    raise ValueError("unknown method %r" % method)


def run_driver(name, drv, restart_scf=False):
    """Run one driver while respecting each class's scan signature."""
    params = inspect.signature(drv.scan).parameters
    if "return_intermediate" in params:
        return drv.scan(restart=restart_scf, return_intermediate=False)
    if "restart" in params:
        return drv.scan(restart=False)
    return drv.scan()


def _point_k_grid(data):
    """Return the k grid used by the Sigma table stored for a finished point."""
    if "k_scf" in data:
        return data["k_scf"], data["k_wts_scf"]
    return data["k_grid"], data["k_wts"]


def point_z_row(name, drv, row):
    r"""Return the Example-05 scalar Z diagnostic for one completed point.

    ``Z_kF_i0`` is evaluated from the Matsubara-axis normal self-energy
    convention implemented by :func:`migdal.scf.get_Z_matsubara`.  The function
    uses the non-uniform radial ``k`` grid stored by the driver and fits the
    lowest sparse-IR Matsubara pairs to obtain the low-frequency value at
    ``k_F``.
    """
    from migdal import radial, scf

    data = getattr(drv, "_last_point_data", None)
    if data is None:
        raise RuntimeError("driver has no completed point data for Z.")
    k_grid, k_wts = _point_k_grid(data)
    sres = getattr(drv, "_last_scf", None)
    if sres is not None and getattr(sres, "sigma", None) is not None:
        sigma = sres.sigma
        sigma_source = "scf_sigma"
    else:
        G0 = data.get("G0")
        if G0 is None:
            raise RuntimeError("one-shot Z requires stored G0 for the point.")
        sigma = scf.get_sigma(
            drv.ctx,
            k_grid,
            k_wts,
            G0,
            data["W"],
            diag_cells=data["diag_cells"],
        )
        sigma_source = "g0_sigma"
    sigma_symm_err = radial.check_sigma_symm(drv.ctx, sigma)
    Z = scf.get_Z_matsubara(drv.ctx, sigma)
    return {
        "method": name,
        "n_cm3": float(row.n_cm3),
        "Z_kF_i0": scf.low_freq_real_at_k(
            drv.ctx, Z, k_grid, data["k_F"]
        ),
        "Z_table_min": float(np.min(Z)),
        "Z_table_max": float(np.max(Z)),
        "sigma_symm_err": float(sigma_symm_err),
        "sigma_source": sigma_source,
        "gap_converged": int(bool(row.gap_converged)),
        "scf_converged": (
            -1 if row.scf_converged is None else int(bool(row.scf_converged))
        ),
        "scf_cycles": -1 if row.scf_cycles is None else int(row.scf_cycles),
        "nk": int(row.nk),
        "nq": int(row.nq),
    }


def _run_one_shot_driver_with_z(name, drv):
    """Run a G0W0-like scan and collect Z after each requested point."""
    if drv.ctx is None:
        drv.build()
    rows = []
    z_rows = []
    for n_cm3 in drv._density_values(drv.densities):
        row = drv.kernel(float(n_cm3))
        rows.append(row)
        z_rows.append(point_z_row(name, drv, row))
    drv.results = rows
    return rows, z_rows


def _run_scf_driver_with_z(name, drv, restart_scf=False):
    """Run a GW0/scGW-like scan and collect Z on requested density points."""
    if drv.ctx is None:
        drv.build()
    densities = drv.densities
    max_dex = drv.scf_continuation_max_dex if restart_scf else 0.0
    path, keep = drv._scan_path(densities, max_step_dex=max_dex)
    rows = []
    kept_rows = []
    z_rows = []
    sigma0 = None
    mu0 = None
    fs_pin_shift0 = None
    old_k = None
    for n_cm3, keep_row in zip(path, keep):
        row = drv.kernel(
            float(n_cm3),
            sigma0=sigma0 if restart_scf else None,
            mu0=mu0 if restart_scf else None,
            sigma0_k=old_k if restart_scf else None,
            fs_pin_shift0=fs_pin_shift0 if restart_scf else None,
        )
        rows.append(row)
        if keep_row:
            kept_rows.append(row)
            z_rows.append(point_z_row(name, drv, row))
        sres = drv._last_scf
        if restart_scf and sres.converged:
            sigma0 = sres.sigma
            mu0 = sres.mu
            fs_pin_shift0 = sres.extra.get("fs_pin_shift")
            old_k = drv._last_k_grid
        elif restart_scf:
            sigma0 = None
            mu0 = None
            fs_pin_shift0 = None
            old_k = None
    drv.results = kept_rows
    return kept_rows, z_rows


def run_driver_with_z(name, drv, restart_scf=False):
    """Run one driver and return ``(lambda_rows, z_rows)``."""
    params = inspect.signature(drv.scan).parameters
    if "return_intermediate" in params:
        return _run_scf_driver_with_z(name, drv, restart_scf=restart_scf)
    return _run_one_shot_driver_with_z(name, drv)


def _format_z_value(name, value):
    """Return one CSV field for a Z diagnostic row."""
    if name in {"method", "sigma_source"}:
        return str(value)
    if name in {
        "gap_converged",
        "scf_converged",
        "scf_cycles",
        "nk",
        "nq",
    }:
        return "%d" % int(value)
    return "%.18e" % float(value)


def write_z_csv(path, rows):
    """Write long-form Z diagnostics for all Example-05 curves."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=Z_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: _format_z_value(name, row[name]) for name in Z_COLUMNS}
            )


def plot_z_curves(out_dir, z_rows, method_order=None):
    """Plot the saved scalar Z diagnostic using the standalone plot helper."""
    import plot_z

    return plot_z.plot_z_rows(
        z_rows,
        out_dir,
        method_order=method_order,
        method_labels=METHOD_LABELS,
        method_styles=METHOD_STYLES,
    )


def build_jobs(args, densities, include_baseline=True):
    """Return the compact default controls plus optional extra controls."""
    no_ph_params = no_phonon_params()
    control_no_phonon = "%s_no_phonon" % args.control_method
    jobs = []
    if include_baseline:
        jobs.extend(
            [
                ("G0W0", make_driver("G0W0", args, densities)),
                ("GW0", make_driver("GW0", args, densities)),
                ("scGW", make_driver("scGW", args, densities)),
            ]
        )
    jobs.extend(
        [
            ("GV", make_driver(args.control_method, args, densities, fixed_pi=True)),
            (
                "G0W0_no_phonon",
                make_driver("G0W0", args, densities, params=no_ph_params),
            ),
        ]
    )
    if control_no_phonon != "G0W0_no_phonon":
        jobs.append(
            (
                control_no_phonon,
                make_driver(args.control_method, args, densities, params=no_ph_params),
            ),
        )
    if args.all_controls:
        if args.control_method != "G0W0":
            jobs.append(("G0V", make_driver("G0W0", args, densities, fixed_pi=True)))
        jobs.append(
            (
                "G0V_no_phonon",
                make_driver(
                    "G0W0",
                    args,
                    densities,
                    params=no_ph_params,
                    fixed_pi=True,
                ),
            )
        )
        if args.control_method != "G0W0":
            jobs.append(
                (
                    "GV_no_phonon",
                    make_driver(
                        args.control_method,
                        args,
                        densities,
                        params=no_ph_params,
                        fixed_pi=True,
                    ),
                )
            )

    unique = []
    seen = set()
    for name, drv in jobs:
        if name in seen:
            continue
        unique.append((name, drv))
        seen.add(name)
    return unique


def lambda_range(rows):
    """Return a compact min/max lambda string for one method."""
    vals = np.asarray([row.lam for row in rows], dtype=float)
    return "%.6g to %.6g" % (float(np.min(vals)), float(np.max(vals)))


def convergence_line(name, rows):
    """Summarize gap and SCF convergence for one method."""
    gap_ok = sum(int(row.gap_converged) for row in rows)
    scf_rows = [row for row in rows if row.scf_converged is not None]
    if not scf_rows:
        return "  %s: gap %d/%d, no normal-state SCF loop" % (
            name,
            gap_ok,
            len(rows),
        )
    scf_ok = sum(int(row.scf_converged) for row in scf_rows)
    cycles = [int(row.scf_cycles) for row in scf_rows]
    return "  %s: gap %d/%d, SCF %d/%d, cycles %d-%d" % (
        name,
        gap_ok,
        len(rows),
        scf_ok,
        len(scf_rows),
        min(cycles),
        max(cycles),
    )


def plot_curves(out_dir, curves, args):
    """Plot physical and control lambda density curves."""
    plt = common.pyplot(out_dir)
    fig, ax_lam = plt.subplots(figsize=(6.5, 6.3))
    fig.patch.set_facecolor("white")
    all_lam = []

    for name, rows in curves:
        arr = common.rows_to_array(rows)
        order = np.argsort(arr[:, 0])
        density = arr[order, 0]
        lam = arr[order, 1]
        all_lam.extend(lam.tolist())
        color, marker, linestyle = METHOD_STYLES.get(name, (None, "o", "-"))
        label = METHOD_LABELS.get(name, name)
        ax_lam.plot(
            density,
            lam,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=1.8,
            markersize=5,
            label=label,
        )

    ax_lam.set_xscale("log")
    ax_lam.set_xlabel(r"carrier density $n$ [cm$^{-3}$]")
    ax_lam.set_ylabel(r"leading eigenvalue $\lambda$")
    ax_lam.set_title(
        "Example 05: RPA screening vs phonon enhancement",
        fontsize=14,
    )
    ax_lam.grid(True, which="both", color="#d0d5dd", linewidth=0.8, alpha=0.8)
    ax_lam.axhline(1.0, color="#98a2b3", linestyle=":", linewidth=0.9, zorder=0)
    ax_lam.set_ylim(bottom=0.0)
    if all_lam:
        ax_lam.set_ylim(0.0, max(all_lam) * 1.08)
    ax_lam.legend(frameon=False, loc="upper right", ncol=2, fontsize=9.0)
    fig.text(
        0.12,
        0.045,
        (
            r"$W=V_{\mathrm{tot}}/(1-\Pi V_{\mathrm{tot}})$, "
            r"$V_{\mathrm{tot}}=V_{\mathrm{c}}\,V_{\mathrm{ph}}$; GV fixes $\Pi=0$."
            "\n"
            r"No-phonon sets $\epsilon_0=\epsilon_\infty$, "
            r"so $V_{\mathrm{ph}}=1$ while electronic RPA screening remains."
        ),
        va="bottom",
        ha="left",
        fontsize=8.6,
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.91,
        bottom=0.24,
    )
    fig.savefig(out_dir / "screening_phonon_controls.png", dpi=200)
    fig.savefig(out_dir / "screening_phonon_controls.pdf")


def tutorial_report(args, densities, curves, baseline_source=None):
    """Build the tutorial text printed to terminal and saved beside outputs."""
    names = [name for name, _ in curves]
    control_name = "%s_no_phonon" % args.control_method
    default_names = default_control_names(args.control_method)
    lines = [
        "Example 05: screening and phonon-control tutorial",
        "",
        "Question:",
        "  Starting from the example 04 G0W0/GW0/scGW density curves, how much",
        "  of lambda(n) comes from electronic RPA screening, and how much comes",
        "  from the soft-phonon dielectric factor?",
        "",
        "Model formulas:",
        "  Vtot(q, i nu) = Vc(q) Vph(q, i nu), with Vc(q)=4*pi/(eps_inf q^2) for q>0.",
        "  W(q, i nu) = Vtot(q, i nu) / [1 - Pi(q, i nu) Vtot(q, i nu)].",
        "  Vph(q, i nu) = [nu^2 + omega_T(q)^2] / [nu^2 + omega_L(q)^2].",
        "  The code convention has Pi <= 0; fixing Pi=0 removes electronic RPA screening.",
        "",
        "Density scan:",
        "  %s" % density_summary(densities),
        "",
        "Curves:",
    ]
    for name, _ in curves:
        lines.append("  %s: %s." % (name, METHOD_NOTES.get(name, "control curve")))
    lines.extend(
        [
            "",
            "Default controls:",
            "  GV: %s-style run with Pi fixed to 0, so W=Vtot."
            % args.control_method,
            "  G0W0_no_phonon: one-shot no-phonon reference with Vph=1.",
        ]
    )
    if control_name != "G0W0_no_phonon":
        lines.append(
            "  %s: %s-style no-phonon run with RPA screening retained."
            % (control_name, args.control_method)
        )
    lines.extend(
        [
            "  In every no-phonon curve eps_0 is set to eps_inf, so omega_L=omega_T and Vph=1.",
            "  GV removes electronic RPA screening while retaining the phonon factor.",
            "  The no-phonon control removes phonon enhancement while retaining RPA screening.",
            "",
            "Z diagnostic:",
            "  Z_kF_i0 is an imaginary-axis diagnostic from the odd part of Sigma:",
            "  Z(k, iwn)=1 - Sigma_odd(k, iwn)/(i wn), then a low-frequency",
            "  fit is evaluated at k_F on the same non-uniform radial k grid.",
            "  It is not the real-axis quasiparticle residue.",
        ]
    )
    if baseline_source is not None:
        lines.extend(
            [
                "",
                "Baseline source:",
                "  Loaded G0W0/GW0/scGW from %s." % display_path(baseline_source),
                "  Only the control curves are recomputed in this run.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Baseline source:",
                "  Computed G0W0/GW0/scGW in this run with example-04-style settings.",
            ]
        )
    if args.all_controls:
        baseline_default = tuple(BASELINE_METHODS) + tuple(default_names)
        extras = [name for name in names if name not in baseline_default]
        lines.extend(
            [
                "",
                "Extra controls enabled by --all-controls:",
                "  %s." % ", ".join(extras),
                "  G0V is the one-shot fixed-Pi control.",
                "  *_no_phonon additionally sets Vph(q, i nu)=1.",
            ]
        )
    lines.extend(
        [
            "",
            "Driver settings used:",
            "  No truncation; diag_average='auto' selects diagonal cell averaging.",
            "  q_interp_mode=%s, diag_average_mode=%s."
            % (args.q_interp_mode, args.diag_average_mode),
            "  sparse-IR file=%s" % display_path(args.ir_file),
            "  default control method=%s; default control curves=%s."
            % (args.control_method, ", ".join(default_names)),
            "  scGW q grid=%d when scGW curves are computed."
            % args.scgw_nq,
            "",
            "SCF settings:",
            "  GW0 inserted-continuation max_dex=%s, restart=%s, max_cycle=%s."
            % (
                args.gw0_continuation_max_dex,
                "on" if uses_scf_restart(args.restart_scf) else "off",
                args.gw0_scf_max_cycle,
            ),
            "  scGW inserted-continuation max_dex=%s, restart=%s."
            % (
                args.scgw_continuation_max_dex,
                "on" if uses_scf_restart(args.restart_scf) else "off",
            ),
            "  scGW damp=%s, DIIS space=%d, DIIS start=%d, max_cycle=%d."
            % (
                args.scgw_scf_damp,
                args.scgw_scf_diis_space,
                args.scgw_scf_diis_start_cycle,
                args.scgw_scf_max_cycle,
            ),
            "  Fermi-surface pin: %s; unconverged rows are %s."
            % (
                "on" if args.scf_fs_pin else "off",
                "rejected" if args.fail_unconverged else "recorded with flags",
            ),
            "",
            "Result snapshot:",
        ]
    )
    for name, rows in curves:
        lines.append("  %s lambda range: %s" % (name, lambda_range(rows)))
    lines.extend(["", "Convergence summary:"])
    for name, rows in curves:
        lines.append(convergence_line(name, rows))
    lines.extend(
        [
            "",
            "Files:",
            "  %s" % LAMBDA_CSV,
            "  %s" % Z_CSV,
            "  screening_phonon_controls.png",
            "  screening_phonon_controls.pdf",
            "  screening_phonon_controls_z.png",
            "  screening_phonon_controls_z.pdf",
            "  screening_phonon_controls.txt",
        ]
    )
    return lines


def main(argv=None):
    """Run physical curves plus screening and phonon controls."""
    args = parse_args(argv)
    from mpi4pyscf.lib import logger  # noqa: F401
    from mpi4pyscf.tools import mpi

    densities = common.density_points(args)
    out_dir = common.prepare_out_dir(args.out_dir)
    baseline_source = None
    # Z needs the same-run normal self-energy, so do not reuse example-04 CSV
    # rows even when they exist for the lambda-only comparison.
    jobs = build_jobs(args, densities, include_baseline=True)
    common.rank0_print(
        mpi.rank,
        "Example 05: %s" % density_summary(densities),
    )
    common.rank0_print(
        mpi.rank,
        "Control method=%s; jobs to compute=%d; output=%s"
        % (args.control_method, len(jobs), out_dir),
    )
    common.rank0_print(
        mpi.rank,
        "Computing G0W0/GW0/scGW in this run so Z uses the same self-energy.",
    )
    curves = []
    z_rows = []
    for idx, (name, drv) in enumerate(jobs, start=1):
        common.rank0_print(
            mpi.rank,
            "step %d/%d: %s - %s"
            % (idx, len(jobs), name, METHOD_NOTES.get(name, "control curve")),
        )
        rows, rows_z = run_driver_with_z(
            name, drv, restart_scf=args.restart_scf)
        curves.append((name, rows))
        z_rows.extend(rows_z)

    if mpi.rank == 0:
        method_order = [name for name, _ in curves]
        common.save_curve(out_dir / LAMBDA_CSV, curves)
        write_z_csv(out_dir / Z_CSV, z_rows)
        plot_curves(out_dir, curves, args)
        plot_z_curves(out_dir, z_rows, method_order=method_order)
        report = tutorial_report(
            args,
            densities,
            curves,
            baseline_source=baseline_source,
        )
        common.write_lines(
            out_dir / "screening_phonon_controls.txt",
            report,
        )
        print(
            "\n".join(report + ["", "Output directory:", "  %s" % out_dir]),
            flush=True,
        )


if __name__ == "__main__":
    main()
