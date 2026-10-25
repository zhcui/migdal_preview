#!/usr/bin/env python

"""Estimate Tc(n) with fixed-density cooling temperature restarts.

For each method and density this example runs one temperature chain from high
temperature to low temperature.  GW0 and scGW reuse only the previous converged
state from the same fixed-density chain; different densities are never mixed in
one driver scan.

Run from the repository root, preferably under MPI:

    mpirun -n 2 python examples/08_tc_vs_density/run.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot as plot08  # noqa: E402
import tc_common as tc  # noqa: E402

common = tc.common


COOLING_TEMPERATURES = tuple(
    float("%.12g" % t)
    for t in np.logspace(np.log10(150.0), np.log10(0.1), 20)
)
# Common radial defaults such as nq=500 are left alone; continuation defaults
# are adjusted here because this example does temperature continuation, not
# density continuation.
EXAMPLE08_DEFAULTS = {
    "temperatures": ",".join("%.12g" % t for t in COOLING_TEMPERATURES),
    "gw0_scf_max_cycle": 300,
    "scgw_scf_max_cycle": 300,
    "scgw_nq": 500,
    "gw0_continuation_max_dex": 0.0,
    "scgw_continuation_max_dex": 0.0,
}
COOLING_LOG_COLUMNS = (
    "method",
    "n_cm3",
    "temperature_K",
    "restart_used",
    "restart_retry_without_restart",
    "restart_updated",
    "restart_source_temperature_K",
    "lambda",
    "gap_converged",
    "scf_converged",
    "scf_cycles",
)


def default_ir_template():
    """Return a same-lambda sparse-IR template for per-temperature caches."""
    candidates = (
        SCRIPT_DIR / "cache" / "ir_T1K.h5",
        SCRIPT_DIR.parent / "04_g0w0_gw0_scgw_curve" / "ir.5",
        SCRIPT_DIR.parent / "ir.h5",
    )
    for path in candidates:
        if path.exists():
            return path
    return ""


def set_parser_default(parser, dest, value):
    """Update an argparse action default in-place."""
    parser.set_defaults(**{dest: value})
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            break


def build_parser(description=None, script_file=__file__):
    """Return the fixed-density cooling-restart Tc parser."""
    parser = tc.build_parser(
        description=description or __doc__,
        script_file=script_file,
        include_restart_scf=False,
    )
    for dest, value in EXAMPLE08_DEFAULTS.items():
        set_parser_default(parser, dest, value)
    parser.add_argument(
        "--ir-template",
        default=str(default_ir_template()),
        help=(
            "Sparse-IR h5 template scaled to missing per-temperature caches. "
            "Use an empty string to build each missing IR cache directly."
        ),
    )
    parser.add_argument(
        "--restart-temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the previous converged same-density GW0/scGW SCF state while cooling.",
    )
    parser.add_argument(
        "--stop-after-crossing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop each method/density chain as soon as a usable lambda=1 bracket is found.",
    )
    return parser


def parse_args(argv=None):
    """Parse arguments and normalize method and cooling-temperature order."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.temperatures = cooling_temperatures(
        tc.parse_float_list(args.temperatures, "temperatures")
    )
    args.methods = tc.parse_methods(args.methods)
    return args


def cooling_temperatures(temperatures):
    """Return unique positive temperatures sorted from high to low."""
    vals = sorted({float("%.12g" % float(t)) for t in temperatures}, reverse=True)
    if not vals or any(t <= 0.0 for t in vals):
        raise ValueError("temperatures must be positive.")
    return vals


def density_key(n_cm3):
    """Return a stable density key for row grouping."""
    return "%.12e" % float(n_cm3)


def same_density(a, b):
    """Return whether two density labels represent the same requested point."""
    return density_key(a) == density_key(b)


def lambda_rows_for(lambda_rows, method, n_cm3):
    """Return rows for one method/density sorted by increasing temperature."""
    return sorted(
        [
            row
            for row in lambda_rows
            if row.get("method") == method and same_density(row["n_cm3"], n_cm3)
        ],
        key=lambda row: tc.as_float(row, "temperature_K"),
    )


def drop_chain_rows(lambda_rows, method, n_cm3):
    """Remove rows belonging to one method/density chain."""
    return [
        row
        for row in lambda_rows
        if not (row.get("method") == method and same_density(row["n_cm3"], n_cm3))
    ]


def drop_cooling_log_rows(cooling_log, method, n_cm3):
    """Remove restart-log rows belonging to one method/density chain."""
    return [
        row
        for row in cooling_log
        if not (row.get("method") == method and same_density(row["n_cm3"], n_cm3))
    ]


def summarize_chain(lambda_rows, method, n_cm3, args):
    """Return the Tc summary row for one method/density chain."""
    rows = lambda_rows_for(lambda_rows, method, n_cm3)
    if not rows:
        return None
    tc_rows = tc.summarize_tc(
        rows,
        allow_unconverged=args.tc_allow_unconverged,
    )
    if not tc_rows:
        return None
    return tc_rows[0]


def chain_reached_lowest_temperature(lambda_rows, method, n_cm3, args):
    """Return whether a chain already contains the requested low-T floor."""
    rows = lambda_rows_for(lambda_rows, method, n_cm3)
    if not rows:
        return False
    t_min = min(tc.as_float(row, "temperature_K") for row in rows)
    return t_min <= min(args.temperatures) * (1.0 + 1.0e-12)


def expected_nq(method, args):
    """Return the q-grid size expected in output rows for one method."""
    return int(args.scgw_nq if method == "scGW" else args.nq)


def chain_rows_match_settings(lambda_rows, method, n_cm3, args):
    """Return whether resume rows match the current q and temperature grids."""
    rows = lambda_rows_for(lambda_rows, method, n_cm3)
    if not rows:
        return False
    return (
        all(tc.as_int(row, "nq") == expected_nq(method, args) for row in rows)
        and chain_temperatures_match_settings(lambda_rows, method, n_cm3, args)
    )


def chain_complete(lambda_rows, method, n_cm3, args):
    """Return whether resume can skip a method/density chain."""
    if not chain_rows_match_settings(lambda_rows, method, n_cm3, args):
        return False
    row = summarize_chain(lambda_rows, method, n_cm3, args)
    if row is None:
        return False
    if row["status"] == "crossed":
        return True
    if row["status"] == "above_window":
        return True
    return chain_reached_lowest_temperature(lambda_rows, method, n_cm3, args)


def chain_stop_reason(summary, args):
    """Return a short reason to stop a cooling chain, or ``None`` to continue."""
    if summary is None:
        return None
    if args.stop_after_crossing and summary["status"] == "crossed":
        return "lambda=1 bracket found"
    if summary["status"] == "above_window":
        return "lambda already exceeds 1 at highest sampled T"
    return None


def canonical_temperature(temperature):
    """Round a temperature to the CSV/cache precision."""
    return float("%.12g" % float(temperature))


def temperature_key(temperature):
    """Return the normalized temperature label used for resume validation."""
    return "%.12g" % float(temperature)


def chain_temperatures_match_settings(lambda_rows, method, n_cm3, args):
    """Return whether resume rows belong to the current temperature grid."""
    rows = lambda_rows_for(lambda_rows, method, n_cm3)
    if not rows:
        return False
    expected = {temperature_key(t) for t in args.temperatures}
    return all(temperature_key(row["temperature_K"]) in expected for row in rows)


def maybe_write_scaled_ir_cache(args, temperature, ir_file, rank):
    """Create a target-temperature IR cache by scaling a same-lambda template."""
    ir_file = Path(ir_file)
    if ir_file.exists() or not str(args.ir_template).strip():
        return
    template = Path(args.ir_template)
    if not template.exists():
        common.rank0_print(rank, "IR template not found: %s" % template)
        return

    import h5py
    from migdal import ir_cache

    with h5py.File(template, "r") as fh:
        template_lamb = float(np.asarray(fh["lamb"]))
    if abs(template_lamb - float(args.lamb)) > max(1.0e-3, 1.0e-12 * abs(args.lamb)):
        common.rank0_print(
            rank,
            "skip scaled IR cache because template lambda=%s requested=%s"
            % (template_lamb, args.lamb),
        )
        return

    beta = common.model.STOParams().beta_ha(float(temperature))
    common.rank0_print(
        rank,
        "writing scaled IR cache T=%.6g K from %s" % (float(temperature), template),
    )
    ir_cache.write_scaled_sparse_ir_h5(template, ir_file, beta)


def make_driver(method, n_cm3, temperature, args, ir_file):
    """Construct one single-density driver for a single temperature."""
    from migdal.g0w0 import G0W0
    from migdal.gw0 import GW0
    from migdal.scgw import ScGW

    settings = dict(
        densities=[float(n_cm3)],
        temperature=float(temperature),
        truncate=False,
        diag_average=1,
        diag_ncell=args.diag_ncell,
        **tc.radial_kwargs(args, ir_file),
    )
    if method == "G0W0":
        return G0W0(**settings)
    if method == "GW0":
        gw0_kwargs = common.changed_driver_kwargs(args, "gw0")
        if not args.fail_unconverged:
            gw0_kwargs["allow_unconverged"] = True
        return GW0(**settings, **gw0_kwargs)
    if method == "scGW":
        scgw_settings = dict(settings)
        scgw_settings["nq"] = args.scgw_nq
        scgw_kwargs = common.changed_driver_kwargs(args, "scgw")
        if not args.fail_unconverged:
            scgw_kwargs["allow_unconverged"] = True
        return ScGW(**scgw_settings, **scgw_kwargs)
    raise AssertionError(method)


def cooling_log_row(method, event):
    """Return one transparent restart bookkeeping row."""
    return {
        "method": method,
        "n_cm3": float(event["n_cm3"]),
        "temperature_K": float(event["temperature_K"]),
        "restart_used": int(event["restart_used"]),
        "restart_retry_without_restart": int(event["restart_retry_without_restart"]),
        "restart_updated": int(event["restart_updated"]),
        "restart_source_temperature_K": event["restart_source_temperature_K"],
        "lambda": event["lambda"],
        "gap_converged": int(event["gap_converged"]),
        "scf_converged": int(event["scf_converged"]),
        "scf_cycles": int(event["scf_cycles"]),
    }


def write_cooling_log(path, rows):
    """Write restart bookkeeping rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COOLING_LOG_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_cooling_log(path):
    """Load restart bookkeeping rows, returning an empty list when absent."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def int_flag(row, key):
    """Return an integer CSV flag, accepting either numeric or boolean values."""
    value = row.get(key, 0)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none"):
            return 0
        if text in ("true", "yes"):
            return 1
        if text in ("false", "no"):
            return 0
    return int(float(value))


def scf_cycle_max(lambda_rows, method):
    """Return the maximum SCF cycle count recorded for a method, or ``None``."""
    cycles = []
    for row in lambda_rows:
        if row.get("method") != method or str(row.get("scf_cycles", "")).strip() == "":
            continue
        cycle = int(float(row["scf_cycles"]))
        if cycle >= 0:
            cycles.append(cycle)
    return max(cycles) if cycles else None


def tutorial_report(args, lambda_rows, tc_rows, cooling_log):
    """Return tutorial-style lines explaining the example-08 workflow and outputs."""
    status_counts = Counter(row["status"] for row in tc_rows)
    status_text = ", ".join(
        "%s=%d" % (status, status_counts[status])
        for status in sorted(status_counts)
    ) or "none"
    restart_rows = [row for row in cooling_log if row.get("method") in ("GW0", "scGW")]
    restart_used = sum(int_flag(row, "restart_used") for row in restart_rows)
    restart_retries = sum(
        int_flag(row, "restart_retry_without_restart") for row in restart_rows
    )
    restart_updates = sum(int_flag(row, "restart_updated") for row in restart_rows)
    gap_failures = sum(1 for row in lambda_rows if int_flag(row, "gap_converged") == 0)
    scf_failures = sum(
        1
        for row in lambda_rows
        if row.get("method") in ("GW0", "scGW")
        and int_flag(row, "scf_converged") == 0
    )
    max_cycles = []
    for method in ("GW0", "scGW"):
        cycles = scf_cycle_max(lambda_rows, method)
        if cycles is not None:
            max_cycles.append("%s=%d" % (method, cycles))

    return [
        "",
        "Example 08 tutorial:",
        "  Goal: find Tc(n) from lambda(T)=1 while changing temperature at fixed density.",
        "  Workflow: GW0/scGW cool one fixed-density chain and reuse the previous",
        "    converged normal state only within that same method and density.",
        "  Adaptive note: this example does not extend the requested temperature window;",
        "    the default window is the 20-point log grid %.6g K -> %.6g K."
        % (float(args.temperatures[0]), float(args.temperatures[-1])),
        "  Restart rule: G0W0 has no SCF state; GW0/scGW use restart=True after the first",
        "    temperature in a chain. If a restarted SCF does not converge, the same",
        "    temperature is retried once from a cold start; if that also fails, the last",
        "    converged restart state is kept for the next temperature.",
        "  Stop rule: a chain stops after the first usable lambda=1 bracket, or after the",
        "    %.6g K floor if lambda remains below one." % float(args.temperatures[-1]),
        (
            "  Main defaults: G0W0/GW0 nq=%d, scGW nq=%d, "
            "GW0/scGW max_cycle=%s/%s, density continuation disabled for both "
            "GW0 and scGW."
        )
        % (
            args.nq,
            args.scgw_nq,
            args.gw0_scf_max_cycle,
            args.scgw_scf_max_cycle,
        ),
        "  Outputs:",
        "    lambda_vs_temperature.csv/png: raw lambda(T) points and crossing plot.",
        "    tc_vs_density.csv/png: interpolated Tc values; below_window means Tc < floor.",
        "    cooling_restart_log.csv: restart provenance for every temperature point.",
        "    plot.py: redraws figures from saved CSV data without rerunning GW/SCF.",
        "    production_run.log: full verbose stdout from the production MPI run, if saved.",
        "  Current summary: %d lambda rows, %d Tc rows (%s)."
        % (len(lambda_rows), len(tc_rows), status_text),
        "  Convergence summary: gap failures=%d, GW0/scGW SCF failures=%d, restart retries=%d."
        % (gap_failures, scf_failures, restart_retries),
        "  Restart bookkeeping: restart-used rows=%d, updated states=%d%s."
        % (
            restart_used,
            restart_updates,
            "; max SCF cycles " + ", ".join(max_cycles) if max_cycles else "",
        ),
    ]


def append_example08_summary_note(out_dir, args, lambda_rows, tc_rows, cooling_log):
    """Append fixed-density cooling-restart tutorial details to the text summary."""
    path = Path(out_dir) / "tc_vs_density.txt"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    note = tutorial_report(args, lambda_rows, tc_rows, cooling_log)
    path.write_text(text.rstrip() + "\n" + "\n".join(note) + "\n", encoding="utf-8")


def write_outputs(out_dir, args, lambda_rows, cooling_log):
    """Regenerate CSV, figure, summary, and restart bookkeeping outputs."""
    tc_rows = tc.summarize_tc(
        lambda_rows,
        allow_unconverged=args.tc_allow_unconverged,
    )
    tc.write_lambda_csv(out_dir / "lambda_vs_temperature.csv", lambda_rows)
    tc.write_tc_csv(out_dir / "tc_vs_density.csv", tc_rows)
    plot08.plot_outputs(
        out_dir,
        lambda_rows,
        tc_rows,
        allow_unconverged=args.tc_allow_unconverged,
    )
    summary_args = argparse.Namespace(**vars(args))
    summary_args.temperatures = sorted(
        {tc.as_float(row, "temperature_K") for row in lambda_rows}
    )
    tc.write_summary(out_dir, summary_args, lambda_rows, tc_rows)
    append_example08_summary_note(out_dir, args, lambda_rows, tc_rows, cooling_log)
    write_cooling_log(out_dir / "cooling_restart_log.csv", cooling_log)
    return tc_rows


def run_cooling_chain(lambda_rows, cooling_log, method, n_cm3, args, out_dir, rank):
    """Run one fixed-density cooling chain and update outputs incrementally."""
    temperatures = [canonical_temperature(t) for t in args.temperatures]

    def ir_file_for_temperature(temperature):
        ir_file = tc.ir_file_for_temperature(args, temperature)
        maybe_write_scaled_ir_cache(args, temperature, ir_file, rank)
        return ir_file

    def record_point(point, event):
        nonlocal lambda_rows, cooling_log
        temperature = canonical_temperature(event["temperature_K"])
        ir_file = event["ir_file"] or tc.ir_file_for_temperature(args, temperature)
        new_row = tc.point_to_lambda_row(method, temperature, ir_file, point)
        lambda_rows = tc.replace_lambda_rows(lambda_rows, [new_row])
        cooling_log.append(cooling_log_row(method, event))
        if rank == 0:
            write_outputs(out_dir, args, lambda_rows, cooling_log)

        summary = summarize_chain(lambda_rows, method, n_cm3, args)
        stop_reason = chain_stop_reason(summary, args)
        if stop_reason is not None:
            common.rank0_print(
                rank,
                "stop %s n=%.6e: %s" % (method, float(n_cm3), stop_reason),
            )
            return True
        return False

    first_temperature = temperatures[0]
    first_ir_file = tc.ir_file_for_temperature(args, first_temperature)
    drv = make_driver(method, n_cm3, first_temperature, args, first_ir_file)
    drv.scan_temp(
        temperatures,
        n_cm3=float(n_cm3),
        restart=args.restart_temperature,
        retry_without_restart=True,
        ir_file_for_temperature=ir_file_for_temperature,
        stop_callback=record_point,
    )
    return lambda_rows, cooling_log


def density_summary(densities):
    """Return a compact density-grid description."""
    densities = np.asarray(densities, dtype=float)
    return "%d points from %.3e down to %.3e cm^-3" % (
        len(densities),
        float(np.max(densities)),
        float(np.min(densities)),
    )


def temperature_summary(temperatures):
    """Return a compact high-to-low temperature-grid description."""
    temperatures = np.asarray(temperatures, dtype=float)
    return "%d points from %.6g down to %.6g K" % (
        len(temperatures),
        float(np.max(temperatures)),
        float(np.min(temperatures)),
    )


def run_plan_lines(args, densities, out_dir):
    """Return the tutorial-style run plan printed at startup."""
    return [
        "example 08 start",
        "  workflow: fixed-density cooling chains; GW0/scGW restart only within",
        "    the same method and density",
        "  methods: %s" % ", ".join(args.methods),
        "  densities: %s" % density_summary(densities),
        "  temperatures: %s" % temperature_summary(args.temperatures),
        "  grids: G0W0/GW0 nq=%d; scGW nq=%d; diag_ncell=%d"
        % (args.nq, args.scgw_nq, args.diag_ncell),
        "  outputs: %s" % out_dir,
    ]


def main(argv=None):
    """Run fixed-density cooling chains and write Tc(n) outputs."""
    args = parse_args(argv)
    from mpi4pyscf.tools import mpi

    densities = common.density_points(args)
    out_dir = common.prepare_out_dir(args.out_dir)
    lambda_csv = out_dir / "lambda_vs_temperature.csv"
    lambda_rows = tc.load_lambda_rows(lambda_csv) if args.resume else []
    cooling_log = (
        load_cooling_log(out_dir / "cooling_restart_log.csv") if args.resume else []
    )

    for line in run_plan_lines(args, densities, out_dir):
        common.rank0_print(mpi.rank, line)

    for n_cm3 in map(float, densities):
        for method in args.methods:
            if args.resume and chain_complete(lambda_rows, method, n_cm3, args):
                common.rank0_print(
                    mpi.rank,
                    "skip %s n=%.6e from existing CSV" % (method, n_cm3),
                )
                continue
            lambda_rows = drop_chain_rows(lambda_rows, method, n_cm3)
            cooling_log = drop_cooling_log_rows(cooling_log, method, n_cm3)
            lambda_rows, cooling_log = run_cooling_chain(
                lambda_rows,
                cooling_log,
                method,
                n_cm3,
                args,
                out_dir,
                mpi.rank,
            )

    if mpi.rank == 0:
        tc_rows = write_outputs(out_dir, args, lambda_rows, cooling_log)
        remaining = [row for row in tc_rows if row["status"] != "crossed"]
        if remaining:
            print(
                "wrote %s with %d bounded or incomplete Tc rows remaining"
                % (out_dir, len(remaining)),
                flush=True,
            )
        else:
            print("wrote %s with all Tc rows crossed" % out_dir, flush=True)


if __name__ == "__main__":
    main()
