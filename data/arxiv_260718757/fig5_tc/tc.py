"""Tc interpolation and the low-frequency Fermi-energy line for Fig. 5."""

import csv
import math
from collections import defaultdict

import numpy as np


METHODS = ("G0W0", "GW0", "scGW")
TC_COLUMNS = (
    "method", "n_cm3", "tc_K", "tc_lower_bound_K", "tc_upper_bound_K",
    "status", "usable_points", "total_points", "excluded_points",
    "bracket_low_T_K", "bracket_low_lambda", "bracket_high_T_K",
    "bracket_high_lambda",
)
EFSTAR_COLUMNS = (
    "n_cm3", "tc_status", "tc_K", "line_temperature_K", "Z_kF_i0",
    "E_F_K", "E_F_star_K", "source",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def usable(row):
    lam = float(row["lambda"])
    return (
        math.isfinite(lam) and lam > 0
        and int(row["gap_converged"]) == 1
        and (row["method"] == "G0W0" or int(row["scf_converged"]) == 1)
    )


def summarize(rows):
    """Use the highest lambda=1 crossing, interpolated linearly in log(T)."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], float(row["n_cm3"])].append(row)
    results = []
    for (method, density), group in sorted(
        groups.items(), key=lambda item: (METHODS.index(item[0][0]), -item[0][1])
    ):
        points = sorted(
            (float(r["temperature_K"]), float(r["lambda"]))
            for r in group if usable(r)
        )
        result = dict.fromkeys(TC_COLUMNS, "")
        result.update(
            method=method, n_cm3=density, usable_points=len(points),
            total_points=len(group), excluded_points=len(group) - len(points),
        )
        crossings = [(t, t, lam, t, lam) for t, lam in points if lam == 1]
        for (t0, l0), (t1, l1) in zip(points, points[1:]):
            if (l0 - 1) * (l1 - 1) < 0:
                tc = math.exp(math.log(t0) + (1 - l0) / (l1 - l0) * math.log(t1 / t0))
                crossings.append((tc, t0, l0, t1, l1))
        if crossings:
            tc, t0, l0, t1, l1 = max(crossings)
            result.update(
                status="crossed", tc_K=tc, tc_lower_bound_K=tc, tc_upper_bound_K=tc,
                bracket_low_T_K=t0, bracket_low_lambda=l0,
                bracket_high_T_K=t1, bracket_high_lambda=l1,
            )
        elif not points:
            result["status"] = "no_usable"
        elif all(lam < 1 for _, lam in points):
            result.update(status="below_window", tc_upper_bound_K=points[0][0])
        elif all(lam > 1 for _, lam in points):
            result.update(status="above_window", tc_lower_bound_K=points[-1][0])
        else:
            result["status"] = "no_bracket"
        results.append(result)
    return results


def fermi_line(tc_rows, z_rows):
    """Evaluate E_F/Z(i0) at Tc, or at the sampled temperature bound."""
    results = []
    for row in tc_rows:
        if row["method"] != "scGW" or row["status"] in ("no_usable", "no_bracket"):
            continue
        points = sorted(
            (float(z["temperature_K"]), z) for z in z_rows
            if float(z["n_cm3"]) == float(row["n_cm3"])
            and int(z["scf_converged"]) == 1
        )
        if not points:
            continue
        status = row["status"]
        if status == "crossed":
            temperature = float(row["tc_K"])
            source = "tc_bracket_logT"
            bracket = (float(row["bracket_low_T_K"]), float(row["bracket_high_T_K"]))
            z_points = [p for p in points if p[0] in bracket]
        elif status == "below_window":
            temperature = float(row["tc_upper_bound_K"])
            source = "lowest_temperature_bound"
            z_points = [p for p in points if p[0] == temperature]
        else:
            temperature = float(row["tc_lower_bound_K"])
            source = "highest_temperature_bound"
            z_points = [p for p in points if p[0] == temperature]
        z = np.interp(
            np.log(temperature), np.log([p[0] for p in z_points]),
            [float(p[1]["Z_kF_i0"]) for p in z_points],
        )
        ef = float(points[0][1]["E_F_K"])
        results.append(dict(
            n_cm3=row["n_cm3"], tc_status=status, tc_K=row["tc_K"],
            line_temperature_K=temperature, Z_kF_i0=z, E_F_K=ef,
            E_F_star_K=ef / z, source=source,
        ))
    return results
