"""Local Tc parsing, CSV, and interpolation helpers for example 08."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import _common as common  # noqa: E402


DEFAULT_TEMPERATURES = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    35.0,
    40.0,
    45.0,
    60.0,
    80.0,
    150.0,
)
METHODS = ("G0W0", "GW0", "scGW")
LAMBDA_COLUMNS = (
    "method",
    "n_cm3",
    "temperature_K",
    "lambda",
    "gap_converged",
    "gap_cycles",
    "scf_converged",
    "scf_cycles",
    "scf_res_sigma",
    "scf_dn",
    "nk",
    "nq",
    "mu_Ha",
    "mu0_Ha",
    "ir_file",
)
TC_COLUMNS = (
    "method",
    "n_cm3",
    "tc_K",
    "tc_lower_bound_K",
    "tc_upper_bound_K",
    "status",
    "usable_points",
    "total_points",
    "excluded_points",
    "bracket_low_T_K",
    "bracket_low_lambda",
    "bracket_high_T_K",
    "bracket_high_lambda",
)


def default_ir_template():
    """Return the default per-temperature sparse-IR cache template."""
    return "cache/ir_{temperature_slug}.h5"


def parse_float_list(text, name):
    """Parse a comma-separated list of positive floats."""
    vals = np.asarray([float(x) for x in str(text).split(",") if x.strip()], dtype=float)
    if vals.ndim != 1 or vals.size == 0 or np.any(vals <= 0.0):
        raise ValueError("%s must be a comma-separated list of positive numbers." % name)
    return vals


def parse_methods(text):
    """Parse and de-duplicate requested method names."""
    aliases = {name.lower(): name for name in METHODS}
    vals = []
    for raw in str(text).split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError("--methods entries must be one of: g0w0, gw0, scgw")
        vals.append(aliases[key])
    if not vals:
        raise ValueError("--methods must contain at least one method.")
    return tuple(dict.fromkeys(vals))


def build_parser(description=None, script_file=__file__, include_restart_scf=True):
    """Return the Tc scan argument parser used by this example."""
    parser = argparse.ArgumentParser(description=description or __doc__)
    common.add_density_args(parser, default_n=6)
    parser.add_argument(
        "--temperatures",
        default=",".join("%.12g" % t for t in DEFAULT_TEMPERATURES),
        help="Comma-separated temperatures in K used to bracket lambda(T)=1.",
    )
    parser.add_argument(
        "--methods",
        default="G0W0,GW0,scGW",
        help="Comma-separated subset of G0W0,GW0,scGW.",
    )
    common.add_radial_args(
        parser,
        default_ir_file=default_ir_template(),
        ir_help=(
            "Sparse-IR cache filename template, relative to --out-dir unless "
            "absolute. Use {temperature} or {temperature_slug}; without a "
            "template token, a temperature suffix is added automatically when "
            "scanning multiple temperatures."
        ),
    )
    parser.add_argument("--diag-ncell", type=int, default=12)
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--gw0-scf-max-cycle", type=int, default=300)
    parser.add_argument("--gw0-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--gw0-scf-damp", type=float, default=0.98)
    parser.add_argument("--gw0-scf-diis-space", type=int, default=8)
    parser.add_argument("--gw0-scf-diis-start-cycle", type=int, default=0)
    parser.add_argument("--gw0-continuation-max-dex", type=float, default=1.0)
    parser.add_argument("--scgw-nq", type=int, default=80)
    parser.add_argument("--scgw-scf-max-cycle", type=int, default=300)
    parser.add_argument("--scgw-scf-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--scgw-scf-damp", type=float, default=0.98)
    parser.add_argument("--scgw-scf-diis-space", type=int, default=8)
    parser.add_argument("--scgw-scf-diis-start-cycle", type=int, default=0)
    parser.add_argument("--scgw-continuation-max-dex", type=float, default=1.0)
    if include_restart_scf:
        parser.add_argument(
            "--restart-scf",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Warm-start GW0/scGW density scans from the previous density point.",
        )
    parser.add_argument("--scgw-n-tail", type=int, default=100)
    parser.add_argument("--pi-imag-tol", type=float, default=1.0e-5)
    parser.add_argument("--pi-imag-policy", choices=("warn", "fail"), default="warn")
    parser.add_argument(
        "--sigma-tail-policy",
        choices=("exchange_static", "exchange_t0", "zero"),
        default="exchange_static",
    )
    parser.add_argument("--allow-no-pi-tail", action="store_true")
    parser.add_argument(
        "--scf-fs-pin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-scf-fs-pin to run the unpinned SCF gauge.",
    )
    parser.add_argument(
        "--fail-unconverged",
        action="store_true",
        help="Raise if GW0/scGW SCF does not converge; default records flags and continues.",
    )
    parser.add_argument(
        "--tc-allow-unconverged",
        action="store_true",
        help="Use unconverged rows when interpolating Tc.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete method/temperature/density rows already in the CSV.",
    )
    common.add_output_arg(parser, script_file)
    return parser


def parse_args(argv=None):
    """Parse Tc scan arguments and normalize temperatures and methods."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.temperatures = parse_float_list(args.temperatures, "temperatures")
    args.methods = parse_methods(args.methods)
    return args


def temperature_slug(temperature):
    """Return a filesystem-safe label such as ``T35K`` or ``T2p5K``."""
    text = "%.12g" % float(temperature)
    return "T" + text.replace("-", "m").replace("+", "").replace(".", "p") + "K"


def ir_file_for_temperature(args, temperature):
    """Return the sparse-IR cache path assigned to one temperature."""
    raw = str(args.ir_file)
    mapping = {
        "temperature": float(temperature),
        "temperature_slug": temperature_slug(temperature),
    }
    if "{" in raw:
        path = Path(raw.format(**mapping))
    else:
        path = Path(raw)
        if len(args.temperatures) > 1:
            if path.suffix:
                path = path.with_name(
                    "%s_%s%s" % (path.stem, mapping["temperature_slug"], path.suffix)
                )
            else:
                path = path / ("ir_%s.h5" % mapping["temperature_slug"])
    if not path.is_absolute():
        path = Path(args.out_dir).resolve() / path
    return path.resolve()


def radial_kwargs(args, ir_file):
    """Return shared radial driver kwargs with the temperature-specific IR file."""
    kwargs = common.radial_kwargs(args)
    kwargs["ir_file"] = str(ir_file)
    kwargs["save_ir"] = True
    kwargs["verbose"] = args.verbose
    return kwargs


def lambda_key(row):
    """Return the unique CSV merge key for one lambda row."""
    return (
        str(row["method"]),
        "%.12e" % float(row["temperature_K"]),
        "%.12e" % float(row["n_cm3"]),
    )


def load_lambda_rows(path):
    """Load existing lambda scan rows, returning an empty list if absent."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def replace_lambda_rows(rows, new_rows):
    """Merge new lambda rows into existing rows by method/T/density key."""
    merged = {lambda_key(row): dict(row) for row in rows}
    for row in new_rows:
        merged[lambda_key(row)] = dict(row)
    return sorted(
        merged.values(),
        key=lambda r: (
            METHODS.index(r["method"]) if r["method"] in METHODS else 99,
            -float(r["n_cm3"]),
            float(r["temperature_K"]),
        ),
    )


def write_lambda_csv(path, rows):
    """Write raw lambda(T) rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LAMBDA_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LAMBDA_COLUMNS})


def as_float(row, key, default=np.nan):
    """Read a CSV field as float with a default for blanks."""
    val = row.get(key, "")
    if val in ("", None):
        return default
    return float(val)


def as_int(row, key, default=-1):
    """Read a CSV field as int with a default for blanks."""
    val = row.get(key, "")
    if val in ("", None):
        return default
    return int(float(val))


def usable_for_tc(row, allow_unconverged=False):
    """Return whether a lambda row can be used for Tc interpolation."""
    lam = as_float(row, "lambda")
    if not (np.isfinite(lam) and lam > 0.0):
        return False
    if as_int(row, "gap_converged", 0) != 1 and not allow_unconverged:
        return False
    if row.get("method") in ("GW0", "scGW") and as_int(row, "scf_converged", 0) != 1:
        return bool(allow_unconverged)
    return True


def exclusion_reason(row, allow_unconverged=False):
    """Return a compact reason string when a lambda row is excluded from Tc."""
    reasons = []
    lam = as_float(row, "lambda")
    if not np.isfinite(lam):
        reasons.append("nonfinite_lambda")
    elif lam <= 0.0:
        reasons.append("nonpositive_lambda")
    if as_int(row, "gap_converged", 0) != 1 and not allow_unconverged:
        reasons.append("gap_not_converged")
    if (
        row.get("method") in ("GW0", "scGW")
        and as_int(row, "scf_converged", 0) != 1
        and not allow_unconverged
    ):
        reasons.append("scf_not_converged")
    return "+".join(reasons) if reasons else ""


def interpolate_log_temperature(t1, l1, t2, l2):
    """Interpolate the lambda=1 crossing linearly in log temperature."""
    if l1 == l2:
        return math.sqrt(t1 * t2)
    x1 = math.log(t1)
    x2 = math.log(t2)
    frac = (1.0 - l1) / (l2 - l1)
    return math.exp(x1 + frac * (x2 - x1))


def summarize_tc(lambda_rows, allow_unconverged=False):
    """Summarize Tc or window bounds from raw lambda(T) rows."""
    grouped = {}
    total_counts = Counter()
    excluded_counts = Counter()
    for row in lambda_rows:
        key = (row["method"], "%.12e" % float(row["n_cm3"]))
        total_counts[key] += 1
        if usable_for_tc(row, allow_unconverged=allow_unconverged):
            grouped.setdefault(key, []).append(row)
        else:
            excluded_counts[key] += 1

    out = []
    for method, n_key in sorted(
        total_counts,
        key=lambda item: (
            METHODS.index(item[0]) if item[0] in METHODS else 99,
            -float(item[1]),
        ),
    ):
        rows = grouped.get((method, n_key), [])
        points = sorted(
            [(as_float(r, "temperature_K"), as_float(r, "lambda")) for r in rows],
            key=lambda item: item[0],
        )
        n_cm3 = float(n_key)
        crossings = []
        for (t1, l1), (t2, l2) in zip(points[:-1], points[1:]):
            d1 = l1 - 1.0
            d2 = l2 - 1.0
            if d1 == 0.0:
                crossings.append((t1, t1, l1, t1, l1))
            elif d2 == 0.0:
                crossings.append((t2, t2, l2, t2, l2))
            elif d1 * d2 < 0.0:
                tc = interpolate_log_temperature(t1, l1, t2, l2)
                crossings.append((tc, t1, l1, t2, l2))

        row = {
            "method": method,
            "n_cm3": n_cm3,
            "tc_K": np.nan,
            "tc_lower_bound_K": np.nan,
            "tc_upper_bound_K": np.nan,
            "status": "",
            "usable_points": len(points),
            "total_points": total_counts[(method, n_key)],
            "excluded_points": excluded_counts[(method, n_key)],
            "bracket_low_T_K": np.nan,
            "bracket_low_lambda": np.nan,
            "bracket_high_T_K": np.nan,
            "bracket_high_lambda": np.nan,
        }
        if not points:
            row["status"] = "no_usable"
        elif crossings:
            tc, t1, l1, t2, l2 = max(crossings, key=lambda item: item[0])
            lo_t, lo_l, hi_t, hi_l = (t1, l1, t2, l2) if t1 <= t2 else (t2, l2, t1, l1)
            row.update(
                {
                    "tc_K": tc,
                    "tc_lower_bound_K": tc,
                    "tc_upper_bound_K": tc,
                    "status": "crossed",
                    "bracket_low_T_K": lo_t,
                    "bracket_low_lambda": lo_l,
                    "bracket_high_T_K": hi_t,
                    "bracket_high_lambda": hi_l,
                }
            )
        else:
            lambdas = [lam for _, lam in points]
            temps = [temp for temp, _ in points]
            if all(lam > 1.0 for lam in lambdas):
                row.update(
                    {
                        "tc_lower_bound_K": max(temps),
                        "status": "above_window",
                    }
                )
            elif all(lam < 1.0 for lam in lambdas):
                row.update(
                    {
                        "tc_upper_bound_K": min(temps),
                        "status": "below_window",
                    }
                )
            else:
                row["status"] = "no_bracket"
        out.append(row)
    return out


def write_tc_csv(path, rows):
    """Write Tc summary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in TC_COLUMNS})


def csv_value(value):
    """Return a CSV-friendly scalar, leaving unavailable floating values blank."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return value


def point_to_lambda_row(method, temperature, ir_file, row):
    """Convert a driver ``PointResult`` into a lambda CSV row."""
    return {
        "method": method,
        "n_cm3": row.n_cm3,
        "temperature_K": float(temperature),
        "lambda": row.lam,
        "gap_converged": int(row.gap_converged),
        "gap_cycles": row.gap.davidson_cycles,
        "scf_converged": -1 if row.scf_converged is None else int(row.scf_converged),
        "scf_cycles": -1 if row.scf_cycles is None else int(row.scf_cycles),
        "scf_res_sigma": "" if row.scf_res_sigma is None else row.scf_res_sigma,
        "scf_dn": "" if row.scf_dn is None else row.scf_dn,
        "nk": row.nk,
        "nq": row.nq,
        "mu_Ha": row.mu,
        "mu0_Ha": row.mu0,
        "ir_file": str(ir_file),
    }


def write_summary(out_dir, args, lambda_rows, tc_rows):
    """Write the human-readable Tc scan summary."""
    methods = sorted(
        {row["method"] for row in tc_rows},
        key=lambda method: METHODS.index(method) if method in METHODS else 99,
    )
    excluded = {}
    total_by_key = Counter()
    for row in lambda_rows:
        key = (row["method"], "%.12e" % float(row["n_cm3"]))
        total_by_key[key] += 1
        reason = exclusion_reason(row, allow_unconverged=args.tc_allow_unconverged)
        if not reason:
            continue
        excluded.setdefault(key, []).append((row, reason))

    if args.tc_allow_unconverged:
        tc_policy_line = (
            "Tc excludes only nonfinite or nonpositive lambda; unconverged "
            "gap and GW0/scGW SCF rows are included."
        )
    else:
        tc_policy_line = (
            "Tc excludes nonpositive lambda, failed gap solves, and "
            "unconverged GW0/scGW rows."
        )

    lines = [
        "Tc(n) from temperature scans of lambda(T).",
        "Methods: %s." % ", ".join(methods),
        "Temperatures [K]: %s." % ", ".join("%.6g" % t for t in args.temperatures),
        "Tc interpolation: lambda vs log(T), highest-temperature lambda=1 crossing.",
        "Rows marked above_window/below_window are bounds, not interpolated Tc values.",
        tc_policy_line,
        "The lambda(T) plot marks excluded positive rows with x symbols.",
        "No-truncation diagonal-cell-average settings follow example 04.",
        "",
        "method n_cm^-3 status Tc_or_bound_K usable/total excluded bracket_K",
    ]
    for row in tc_rows:
        if row["status"] == "crossed":
            val = "%.6g" % float(row["tc_K"])
            bracket = "%.6g-%.6g" % (
                float(row["bracket_low_T_K"]),
                float(row["bracket_high_T_K"]),
            )
        elif row["status"] == "suspect_crossed":
            val = "~ %.6g" % float(row["tc_K"])
            bracket = "%.6g-%.6g" % (
                float(row["bracket_low_T_K"]),
                float(row["bracket_high_T_K"]),
            )
        elif row["status"] == "above_window":
            val = "> %.6g" % float(row["tc_lower_bound_K"])
            bracket = "n/a"
        elif row["status"] == "below_window":
            val = "< %.6g" % float(row["tc_upper_bound_K"])
            bracket = "n/a"
        else:
            val = "n/a"
            bracket = "n/a"
        lines.append(
            "%s %.6e %s %s %d/%d %d %s"
            % (
                row["method"],
                float(row["n_cm3"]),
                row["status"],
                val,
                int(row["usable_points"]),
                int(row["total_points"]),
                int(row["excluded_points"]),
                bracket,
            )
        )
    if excluded:
        lines.extend(["", "Excluded lambda(T) rows:"])
        for method, n_key in sorted(
            excluded,
            key=lambda item: (
                METHODS.index(item[0]) if item[0] in METHODS else 99,
                -float(item[1]),
            ),
        ):
            vals = excluded[(method, n_key)]
            temps = ", ".join("%.6g" % as_float(row, "temperature_K") for row, _ in vals)
            reason_counts = Counter(reason for _, reason in vals)
            reason_text = ", ".join(
                "%s:%d" % (reason, count)
                for reason, count in sorted(reason_counts.items())
            )
            lines.append(
                "%s %.6e excluded %d/%d T[K]=%s reasons=%s"
                % (
                    method,
                    float(n_key),
                    len(vals),
                    total_by_key[(method, n_key)],
                    temps,
                    reason_text,
                )
            )
    common.write_lines(out_dir / "tc_vs_density.txt", lines)
