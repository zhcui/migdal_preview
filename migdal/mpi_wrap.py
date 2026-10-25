"""Thin compatibility layer for ``mpi4pyscf.tools.mpi``."""

from __future__ import annotations

from functools import wraps
import os

import numpy as np


_mpi_logger = None
_mpi_module = None


def get_mpi(import_logger=False):
    """Import mpi4pyscf only when an MPI path is actually entered."""
    global _mpi_logger, _mpi_module
    if import_logger and _mpi_logger is None:
        from mpi4pyscf.lib import logger as logger_mod  # noqa: F401

        _mpi_logger = logger_mod
    if _mpi_module is None:
        from mpi4pyscf.tools import mpi as mpi_mod

        _mpi_module = mpi_mod
    return _mpi_module


def loaded():
    return _mpi_module is not None


def world_size_hint():
    """Return an MPI world-size hint from launcher env vars, or 1 if absent."""
    for key in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
    ):
        val = os.environ.get(key)
        if val:
            try:
                return int(val)
            except ValueError:
                continue
    return 1


class _MPIProxy:
    def __getattr__(self, name):
        return getattr(get_mpi(), name)


mpi = _MPIProxy()


def parallel_call(fn=None, skip_args=None, skip_kwargs=None):
    def mpi_map(f):
        parallel_fn = None

        @wraps(f)
        def with_mpi(dev, *args, **kwargs):
            nonlocal parallel_fn
            mpi_mod = get_mpi(import_logger=True)
            if mpi_mod.pool.size <= 1 or mpi_mod.pool.worker_status == "R":
                return f(dev, *args, **kwargs)
            if parallel_fn is None:
                parallel_fn = mpi_mod.parallel_call(
                    f, skip_args=skip_args, skip_kwargs=skip_kwargs)
            return parallel_fn(dev, *args, **kwargs)

        return with_mpi

    if fn is None:
        return mpi_map
    return mpi_map(fn)


def gather(sendbuf, root=0, split_recvbuf=False):
    mpi_mod = get_mpi()
    if mpi_mod.pool.size <= 1:
        arr = np.asarray(sendbuf)
        if split_recvbuf:
            return [arr]
        return arr
    gather_fn = getattr(mpi_mod, "gather_new", mpi_mod.gather)
    return gather_fn(sendbuf, root=root, split_recvbuf=split_recvbuf)


def reduce(sendbuf, op=None, root=0):
    mpi_mod = get_mpi()
    arr = np.ascontiguousarray(sendbuf)
    if op is None:
        op = mpi_mod.MPI.SUM
    if mpi_mod.pool.size <= 1:
        return arr
    reduce_inplace = getattr(mpi_mod, "reduce_inplace", None)
    if reduce_inplace is not None:
        return reduce_inplace(arr, op=op, root=root)
    return mpi_mod.reduce(arr, op=op, root=root)
