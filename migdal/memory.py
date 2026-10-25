"""Memory-budget helpers for Migdal drivers.

The public ``max_memory`` convention follows PySCF: values are in MB and apply
to one rank/process.  Runtime limits detected from cgroups, Slurm, or
``/proc/meminfo`` are treated as an additional safety guard.
"""

from __future__ import annotations

import math
import os
import socket

from pyscf import lib


MB = 1024.0 * 1024.0
DEFAULT_MAX_MEMORY = 100.0 * 1024.0
DEFAULT_MEMORY_SAFETY = 0.9


def current_rss_mb():
    """Return current resident set size in MB."""
    return float(lib.current_memory()[0])


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _bytes_to_mb(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    # Ignore sentinel values commonly used for "unlimited".
    if value >= 1 << 60:
        return None
    return value / MB


def _cgroup_limit_mb():
    """Return a finite cgroup memory limit in MB, or ``None``."""
    text = _read_text("/sys/fs/cgroup/memory.max")
    if text and text != "max":
        val = _bytes_to_mb(text)
        if val is not None:
            return val, "cgroup"
    text = _read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    val = _bytes_to_mb(text)
    if val is not None:
        return val, "cgroup"
    return None, None


def _env_float(name):
    text = os.environ.get(name)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _slurm_limit_mb():
    """Return a Slurm node/job memory allocation in MB, or ``None``."""
    per_node = _env_float("SLURM_MEM_PER_NODE")
    if per_node and per_node > 0.0:
        return per_node, "slurm"
    per_cpu = _env_float("SLURM_MEM_PER_CPU")
    if per_cpu and per_cpu > 0.0:
        cpus = _env_float("SLURM_CPUS_ON_NODE")
        if cpus is None or cpus <= 0.0:
            cpus = _env_float("SLURM_CPUS_PER_TASK")
        if cpus is None or cpus <= 0.0:
            cpus = 1.0
        return per_cpu * cpus, "slurm"
    return None, None


def _mem_available_mb():
    text = _read_text("/proc/meminfo")
    if not text:
        return None, None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return float(fields[1]) / 1024.0, "meminfo"
                except ValueError:
                    return None, None
    return None, None


def runtime_memory_mb():
    """Return ``(value_mb, source, kind)`` for a runtime memory guard.

    ``kind`` is ``"limit"`` for hard/node allocation limits and
    ``"available"`` for currently available memory estimates.
    """
    val, source = _cgroup_limit_mb()
    if val is not None:
        return val, source, "limit"
    val, source = _slurm_limit_mb()
    if val is not None:
        return val, source, "limit"
    val, source = _mem_available_mb()
    if val is not None:
        return val, source, "available"
    return None, None, None


def local_rank_count(comm=None):
    """Return the number of MPI ranks on this host, or 1 if unavailable."""
    if comm is None:
        return 1
    try:
        host = socket.gethostname()
        hosts = comm.allgather(host)
        return max(1, sum(1 for item in hosts if item == host))
    except Exception:
        return 1


def available_memory_info(max_memory, *, comm=None, safety=DEFAULT_MEMORY_SAFETY):
    """Return a per-rank memory budget summary in MB.

    ``max_memory`` is the user limit per rank.  Runtime limits are divided by
    the number of local ranks when they describe node/job memory.
    """
    max_memory = float(max_memory)
    if not math.isfinite(max_memory) or max_memory < 0.0:
        raise ValueError("max_memory must be a non-negative finite MB value.")
    safety = float(safety)
    if not math.isfinite(safety) or safety <= 0.0 or safety > 1.0:
        raise ValueError("memory safety factor must be in (0, 1].")

    rss = current_rss_mb()
    user_available = max(0.0, max_memory * safety - rss)
    ranks = local_rank_count(comm)
    runtime_value, runtime_source, runtime_kind = runtime_memory_mb()
    runtime_available = None
    if runtime_value is not None:
        per_rank = runtime_value / max(1, ranks)
        if runtime_kind == "limit":
            runtime_available = max(0.0, per_rank * safety - rss)
        else:
            runtime_available = max(0.0, per_rank * safety)

    candidates = [user_available]
    if runtime_available is not None:
        candidates.append(runtime_available)
    effective = min(candidates)
    if effective == user_available:
        limiting = "max_memory"
    else:
        limiting = runtime_source

    return {
        "max_memory_mb": max_memory,
        "safety": safety,
        "current_rss_mb": rss,
        "user_available_mb": user_available,
        "runtime_value_mb": runtime_value,
        "runtime_source": runtime_source,
        "runtime_kind": runtime_kind,
        "runtime_available_mb": runtime_available,
        "local_ranks": ranks,
        "effective_limit_mb": effective,
        "limiting_source": limiting,
    }
