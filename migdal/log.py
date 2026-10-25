"""Formatted logger helpers for Migdal radial drivers.

This module intentionally uses PySCF's logger without importing
``mpi4pyscf.lib.logger``.  Importing the mpi4pyscf logger initializes MPI, which
would make non-MPI helpers such as ``FreqGrid`` and example ``--help`` paths
fail before argument parsing.  MPI kernels import mpi4pyscf explicitly where
parallel execution is required.
"""

from __future__ import annotations

import sys
import textwrap

from pyscf.lib import logger


QUIET = logger.QUIET
WARN = logger.WARN
NOTE = logger.NOTE
INFO = logger.INFO
DEBUG = logger.DEBUG
DEBUG1 = getattr(logger, "DEBUG1", DEBUG + 1)
DEBUG2 = getattr(logger, "DEBUG2", DEBUG1 + 1)

wall_clock = logger.perf_counter

_TIMING_ORDER = (
    "initialization",
    "screening",
    "self_energy",
    "gap_solver",
)

LINE = "=" * 78
SUBLINE = "-" * 78
ALERT_WIDTH = 77

_ALERT_ART = {
    "WARNING": (
        "W   W   AAAAA   RRRR    N   N   III   N   N    GGGG      !!!",
        "W   W   A   A   R   R   NN  N    I    NN  N   G          !!!",
        "W W W   AAAAA   RRRR    N N N    I    N N N   G  GG       ! ",
        "WW WW   A   A   R  R    N  NN    I    N  NN   G   G         ",
        "W   W   A   A   R   R   N   N   III   N   N    GGGG      !!!",
    ),
    "ERROR": (
        "EEEEEEE   RRRR    RRRR     OOOO    RRRR       !!!",
        "E         R   R   R   R   O    O   R   R      !!!",
        "EEEEE     RRRR    RRRR    O    O   RRRR        ! ",
        "E         R  R    R  R    O    O   R  R          ",
        "EEEEEEE   R   R   R   R    OOOO    R   R      !!!",
    ),
}


class _LogRecord:
    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        self.stdout = stdout
        self.verbose = verbose


def new_logger(rec=None, verbose=None, stdout=None):
    """Return a PySCF-style logger for ``rec`` or an ad-hoc stream record."""
    if isinstance(verbose, logger.Logger):
        return verbose
    if rec is None:
        rec = _LogRecord(
            sys.stdout if stdout is None else stdout,
            NOTE if verbose is None else verbose,
        )
        return logger.new_logger(rec, verbose)
    if stdout is None:
        return logger.new_logger(rec, verbose)
    tmp = _LogRecord(stdout, getattr(rec, "verbose", NOTE))
    return logger.new_logger(tmp, verbose)


def enabled(log, level):
    """Return whether ``log`` would emit messages at ``level``."""
    return int(getattr(log, "verbose", QUIET)) >= int(level)


def new_timings(log):
    """Return a wall-time accumulator when DEBUG1 logging is enabled.

    Returning ``None`` below DEBUG1 keeps ordinary calculations off the timing
    path: they do not read the clock or update dictionaries inside SCF loops.
    """
    return {} if enabled(log, DEBUG1) else None


def record_wall(timings, stage, start):
    """Accumulate one rank-0 wall-time interval in ``timings``."""
    if timings is None:
        return
    elapsed = float(wall_clock() - start)
    stats = timings.setdefault(stage, {"calls": 0, "total_s": 0.0})
    stats["calls"] += 1
    stats["total_s"] += elapsed


def snapshot_timings(timings):
    """Return a detached, JSON-serializable timing dictionary."""
    if timings is None:
        return {}
    return {
        str(stage): {
            "calls": int(stats["calls"]),
            "total_s": float(stats["total_s"]),
        }
        for stage, stats in timings.items()
    }


def timing_summary(log, timings):
    """Print the accumulated rank-0 wall times at DEBUG1."""
    if not timings or not enabled(log, DEBUG1):
        return
    fmt = "  %-18s %8s %14s %14s"
    table_header(
        log,
        "Timing summary (rank-0 wall; nested stages are not additive)",
        fmt,
        ("stage", "calls", "total/s", "mean/s"),
        level=DEBUG1,
    )
    ordered = [stage for stage in _TIMING_ORDER if stage in timings]
    ordered.extend(stage for stage in timings if stage not in _TIMING_ORDER)
    for stage in ordered:
        stats = timings[stage]
        calls = int(stats["calls"])
        total = float(stats["total_s"])
        mean = total / calls if calls else 0.0
        table_row(
            log,
            "  %-18s %8d %14.6f %14.6f",
            (stage, calls, total, mean),
            level=DEBUG1,
        )


def emit(log, level, msg="", *args):
    """Emit one formatted line at the requested PySCF logger level."""
    if not enabled(log, level):
        return
    if args:
        msg = msg % args
    if level >= DEBUG:
        log.debug("%s", msg)
    elif level >= INFO:
        log.info("%s", msg)
    elif level >= NOTE:
        log.note("%s", msg)
    elif level >= WARN:
        log.warn("%s", msg)
    else:
        log.log("%s", msg)


def _alert_inner_width(width):
    width = int(width)
    if width < 40:
        raise ValueError("alert box width must be at least 40.")
    return width


def _alert_border(inner_width):
    return " " + "-" * (inner_width + 2)


def _alert_line(text="", inner_width=ALERT_WIDTH):
    return "| " + str(text)[:inner_width].ljust(inner_width) + " |"


def _wrap_alert_text(text, inner_width):
    wrap_width = max(20, inner_width - 6)
    wrapped = []
    for raw in str(text).splitlines() or [""]:
        if not raw:
            wrapped.append("")
            continue
        parts = textwrap.wrap(
            raw,
            width=wrap_width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(parts or [""])
    return wrapped


def format_alert_box(kind, message, width=ALERT_WIDTH, footer=None):
    """Return a VASP-style boxed warning/error message."""
    inner_width = _alert_inner_width(width)
    kind = str(kind).upper()
    art = _ALERT_ART.get(kind, (kind,))
    lines = [_alert_border(inner_width), _alert_line(inner_width=inner_width)]
    for art_line in art:
        lines.append(_alert_line(art_line.center(inner_width), inner_width))
    lines.append(_alert_line(inner_width=inner_width))
    for msg_line in _wrap_alert_text(message, inner_width):
        lines.append(_alert_line(("    " + msg_line) if msg_line else "", inner_width))
    if footer:
        lines.append(_alert_line(inner_width=inner_width))
        for msg_line in _wrap_alert_text(footer, inner_width):
            lines.append(
                _alert_line(("    " + msg_line) if msg_line else "", inner_width)
            )
    lines.append(_alert_line(inner_width=inner_width))
    lines.append(_alert_border(inner_width))
    return "\n".join(lines)


def alert_box(log, kind, message, level=WARN, footer=None, width=ALERT_WIDTH):
    """Emit a VASP-style boxed warning/error message."""
    kind = str(kind).upper()
    force = kind == "ERROR"
    if not force and not enabled(log, level):
        return
    stdout = getattr(log, "stdout", None)
    if stdout is None:
        for line in format_alert_box(
            kind, message, width=width, footer=footer
        ).splitlines():
            if force:
                sys.stderr.write(line + "\n")
            else:
                emit(log, level, line)
        if force:
            sys.stderr.flush()
        return
    for line in format_alert_box(kind, message, width=width, footer=footer).splitlines():
        stdout.write(line + "\n")
    flush = getattr(stdout, "flush", None)
    if flush is not None:
        flush()


def warning(log, message, footer=None, width=ALERT_WIDTH):
    """Emit a logger-controlled VASP-style warning box."""
    if log is None:
        log = new_logger(verbose=WARN)
    alert_box(log, "WARNING", message, level=WARN, footer=footer, width=width)


def error(log, message, footer=None, width=ALERT_WIDTH):
    """Emit an unconditional VASP-style error box."""
    if log is None:
        log = new_logger(verbose=QUIET, stdout=sys.stderr)
    alert_box(log, "ERROR", message, level=QUIET, footer=footer, width=width)


def blank(log, level=INFO):
    """Emit a blank line when ``level`` is enabled."""
    emit(log, level, "")


def banner(log, title, level=INFO):
    """Emit a top-level section banner."""
    blank(log, level)
    emit(log, level, LINE)
    emit(log, level, title)
    emit(log, level, LINE)


def section(log, title, level=INFO):
    """Emit a subsection header."""
    blank(log, level)
    emit(log, level, SUBLINE)
    emit(log, level, title)
    emit(log, level, SUBLINE)


def kv_table(log, title, rows, level=INFO):
    """Emit a small key/value table with aligned keys."""
    rows = [(str(key), str(val)) for key, val in rows if val is not None]
    if not rows:
        return
    if title:
        section(log, title, level=level)
    width = max(len(key) for key, _ in rows)
    for key, val in rows:
        emit(log, level, "  %-*s : %s", width, key, val)


def table_header(log, title, fmt, labels, level=DEBUG):
    """Emit an optional table title followed by a formatted header row."""
    if title:
        section(log, title, level=level)
    emit(log, level, fmt, *labels)


def table_row(log, fmt, values, level=DEBUG):
    """Emit one formatted table row."""
    emit(log, level, fmt, *values)


def format_bool(value):
    """Return a human-readable yes/no/n/a string."""
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def format_optional(value, fmt="%s"):
    """Format optional values with ``fmt`` or return ``"n/a"``."""
    if value is None:
        return "n/a"
    return fmt % value
