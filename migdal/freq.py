r"""Sparse-IR Matsubara grids and one-band Green's-function helpers.

Migdal stores fermionic and bosonic sparse-IR transforms as dense sampling
matrices.  Energies are in Hartree, imaginary times are in Hartree^-1, and the
fermionic frequency array ``ws`` is reverse-index paired:
``ws[i] == -ws[-1-i]``.  All Green's functions in this module use the normal
one-band convention

.. math::

    G(k, i\omega_n) =
    \frac{1}{i\omega_n - (\epsilon_k - \mu) - \Sigma(k, i\omega_n)}.

The sampled Matsubara or tau axis is always axis 1.  This keeps the transform
helpers compatible with both dense radial arrays ``(nk, nw)`` and rank-local
MPI blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg as la

from migdal import log as mlog, model


def get_G_inv(ws, e_k, mu, sigma=None):
    """Return ``G^-1(k, iw) = iw - (e_k - mu) - Sigma(k, iw)``.

    Parameters
    ----------
    ws : (nw,) array_like
        Fermionic Matsubara frequencies in Hartree.
    e_k : (nk,) array_like
        One-band dispersion in Hartree.
    mu : float
        Chemical potential in Hartree.
    sigma : (nk, nw) array_like, optional
        Normal self-energy in Hartree.
    """
    e_k = np.asarray(e_k, dtype=float) - float(mu)
    G_inv = 1j * np.asarray(ws, dtype=float)[None, :] - e_k[:, None]
    if sigma is not None:
        G_inv = G_inv - sigma
    return G_inv


def get_G(ws, e_k, mu, sigma=None):
    """Return the normal Green's function ``G(k, iw)``."""
    return 1.0 / get_G_inv(ws, e_k, mu, sigma=sigma)


def get_G0(ws, e_k, mu):
    """Return the noninteracting Green's function."""
    return get_G(ws, e_k, mu, sigma=None)


def _env_int(names, default=None):
    for name in names:
        val = os.environ.get(name)
        if val is None:
            continue
        try:
            return int(val)
        except ValueError:
            continue
    return default


def _active_mpi_context():
    """Return ``(rank, size, comm)`` without importing MPI.

    Migdal's main MPI drivers use mpi4pyscf's master/worker execution model,
    where ordinary Python construction code runs only on the master.  Therefore
    this helper intentionally relies only on launcher environment variables and
    never calls ``mpi4pyscf.tools.mpi.comm.Barrier()`` from normal constructors.
    """
    rank = _env_int(("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"), 0)
    size = _env_int(("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"), 1)
    return rank, size, None


def _wait_for_file(path, *, timeout=600.0, interval=0.1):
    start = time.monotonic()
    while not path.exists():
        if time.monotonic() - start > timeout:
            raise TimeoutError("timed out waiting for IR file %s" % path)
        time.sleep(interval)


@dataclass
class FreqGrid:
    """Sparse-IR frequency transform context.

    Parameters
    ----------
    beta : float
        Inverse temperature in Hartree^-1.
    lamb : float
        Dimensionless sparse-IR cutoff ``beta*wmax``.
    tol : float
        Sparse-IR accuracy parameter.
    cache_transforms : bool, optional
        If true, precontract default sparse-IR sampling transforms such as
        ``F_t_f @ F_w_f_inv`` and apply them as one matrix multiplication.
        Custom :meth:`trans` calls still use the explicit two-step projection.

    Notes
    -----
    The class intentionally does not know about k-point maps, orbitals, or
    lattice objects.  It only carries Matsubara/tau transforms and the
    one-band Green's-function formulas used by the radial drivers.

    The density matrix and static self-energy terms use the fermionic endpoint
    ``tau = beta`` by default.  A positive ``tau_beta_rtol`` can be supplied
    explicitly to use ``beta * (1 - tau_beta_rtol)`` instead, but that changes
    the numerical convention and is not the production default.
    """

    beta: float
    lamb: float = 1.0e9
    tol: float = 1.0e-14
    tau_beta_rtol: float = 0.0
    ir_file: str | Path | None = None
    save_ir: bool = False
    verbose: int = 4
    stdout: object = sys.stdout
    cache_transforms: bool = True

    def __post_init__(self):
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if self.lamb <= 0.0:
            raise ValueError("lamb must be positive.")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive.")
        if not 0.0 <= self.tau_beta_rtol < 1.0:
            raise ValueError("tau_beta_rtol must be in [0, 1).")

        self.wmax = float(self.lamb) / float(self.beta)
        if self._try_load_ir():
            self._init_runtime_state()
            return

        rank, size, _ = _active_mpi_context()
        cache_path = self._generation_cache_path()
        if size > 1 and cache_path is not None:
            if rank == 0:
                self._build_ir()
                self._save_ir(cache_path)
            if rank != 0:
                _wait_for_file(cache_path)
                if not self._try_load_ir(cache_path):
                    raise RuntimeError("rank %s cannot load IR file %s" % (rank, cache_path))
            self._init_runtime_state()
            return

        self._build_ir()
        self._save_ir_if_requested()
        self._init_runtime_state()

    def _build_ir(self):
        from migdal.ir_cache import build_sparse_ir_arrays

        start = time.monotonic()
        mlog.emit(
            mlog.new_logger(self),
            mlog.DEBUG,
            "FreqGrid: building sparse-IR arrays beta=%.12g lambda=%.12g tol=%.3g"
            % (self.beta, self.lamb, self.tol),
        )
        cache = build_sparse_ir_arrays(
            self.beta,
            self.lamb,
            tol=self.tol,
            tau_beta_rtol=self.tau_beta_rtol,
        )
        self.beta = cache.beta
        self.lamb = cache.lamb
        self.wmax = cache.wmax
        for key, val in cache.arrays.items():
            setattr(self, key, np.asarray(val))
        self._sparse_ir_metadata = cache.metadata

        self.F_t_f_inv = la.pinv(self.F_t_f)
        self.F_t_b_inv = la.pinv(self.F_t_b)
        self.F_w_f_inv = la.pinv(self.F_w_f)
        self.F_w_b_inv = la.pinv(self.F_w_b)
        self.idx_v0 = np.atleast_1d(self.idx_v0).astype(int)
        self.idx_w0 = int(np.argmin(np.abs(self.ws)))
        self.tau_beta = self.beta * (1.0 - self.tau_beta_rtol)
        mlog.emit(
            mlog.new_logger(self),
            mlog.DEBUG,
            "FreqGrid: built sparse-IR arrays nw=%d nv=%d nt_f=%d nt_b=%d in %.2f s"
            % (
                len(self.ws),
                len(self.vs),
                len(self.ts_f),
                len(self.ts_b),
                time.monotonic() - start,
            ),
        )

    def _init_runtime_state(self):
        self.F_t_beta = np.atleast_2d(self.F_t_beta)
        self._init_transform_cache()
        self.symm = False
        self.conv = False
        self.mu = None
        self.n = None
        self.sigma = None

    def _init_transform_cache(self):
        self._F_w2t_f = None
        self._F_t2w_f = None
        self._F_w2t_b = None
        self._F_t2w_b = None
        self._F_wf2tb = None
        self._F_wb2tf = None
        self._F_w2beta_f = None
        if not self.cache_transforms:
            return
        self._F_w2t_f = np.dot(self.F_t_f, self.F_w_f_inv)
        self._F_t2w_f = np.dot(self.F_w_f, self.F_t_f_inv)
        self._F_w2t_b = np.dot(self.F_t_b, self.F_w_b_inv)
        self._F_t2w_b = np.dot(self.F_w_b, self.F_t_b_inv)
        self._F_wf2tb = np.dot(self.F_t_f2b, self.F_w_f_inv)
        self._F_wb2tf = np.dot(self.F_t_b2f, self.F_w_b_inv)
        self._F_w2beta_f = np.dot(self.F_t_beta, self.F_w_f_inv)[0]

    def _transform_or_trans(self, A, matrix_name, F_inv, F):
        if not self.cache_transforms:
            return self.trans(A, F_inv, F)
        matrix = getattr(self, matrix_name, None)
        if matrix is None:
            self._init_transform_cache()
            matrix = getattr(self, matrix_name)
        A = np.asarray(A)
        out = np.tensordot(A, matrix, axes=((1,), (1,)))
        return np.moveaxis(out, -1, 1)

    def _try_load_ir(self, path=None):
        path = self._ir_path() if path is None else Path(path)
        if path is None:
            return False
        if not path.exists():
            mlog.emit(
                mlog.new_logger(self),
                mlog.DEBUG,
                "FreqGrid: IR cache %s not found; building a new one" % path,
            )
            return False
        mlog.emit(
            mlog.new_logger(self),
            mlog.DEBUG,
            "FreqGrid: loading IR cache %s" % path,
        )
        import h5py

        required = (
            "F_t_f",
            "F_t_b",
            "F_w_f",
            "F_w_b",
            "F_t_beta",
            "F_t_f2b",
            "F_t_b2f",
            "ts_f",
            "ts_b",
            "ws",
            "vs",
            "_beta",
            "lamb",
            "wmax",
            "idx_v0",
            "idx_w0",
        )
        inv_keys = {
            "F_t_f_inv": "F_t_f",
            "F_t_b_inv": "F_t_b",
            "F_w_f_inv": "F_w_f",
            "F_w_b_inv": "F_w_b",
        }
        loaded_inv = {}
        with h5py.File(path, "r") as f:
            missing = [key for key in required if key not in f]
            if missing:
                raise ValueError("IR file %s is missing keys %s" % (path, missing))
            beta = float(np.asarray(f["_beta"]))
            lamb = float(np.asarray(f["lamb"]))
            if abs(beta - self.beta) > 1.0e-3:
                raise ValueError(
                    "IR file %s beta mismatch: file=%s requested=%s"
                    % (path, beta, self.beta)
                )
            if abs(lamb - self.lamb) > 1.0e-3:
                raise ValueError(
                    "IR file %s lambda mismatch: file=%s requested=%s"
                    % (path, lamb, self.lamb)
                )
            if "tol" in f:
                file_tol = float(np.asarray(f["tol"]))
                if file_tol <= 0.0:
                    raise ValueError(
                        "IR file %s stores non-positive tol=%s" % (path, file_tol)
                    )
                if file_tol > float(self.tol) * (1.0 + 1.0e-8):
                    raise ValueError(
                        "IR file %s tol is looser than requested: file=%s requested=%s"
                        % (path, file_tol, self.tol)
                    )
            else:
                mlog.warning(
                    mlog.new_logger(self),
                    "IR file %s does not store sparse-IR tol; requested tol=%s "
                    "cannot be verified." % (path, self.tol),
                )
            if "tau_beta_rtol" in f:
                file_tau_beta_rtol = float(np.asarray(f["tau_beta_rtol"]))
                if abs(file_tau_beta_rtol - self.tau_beta_rtol) > 1.0e-15:
                    raise ValueError(
                        "IR file %s tau_beta_rtol mismatch: file=%s requested=%s"
                        % (path, file_tau_beta_rtol, self.tau_beta_rtol)
                    )
            elif self.tau_beta_rtol != 0.0:
                raise ValueError(
                    "loading an IR file without tau_beta_rtol metadata "
                    "requires tau_beta_rtol=0.0"
                )
            for key in required:
                setattr(self, key, np.asarray(f[key]))
            present_inv = [key for key in inv_keys if key in f]
            if present_inv and len(present_inv) != len(inv_keys):
                raise ValueError(
                    "IR file %s stores an incomplete inverse-matrix cache: %s"
                    % (path, present_inv)
                )
            for inv_key, mat_key in inv_keys.items():
                if inv_key not in f:
                    continue
                inv = np.asarray(f[inv_key])
                mat = np.asarray(getattr(self, mat_key))
                expected_shape = (mat.shape[1], mat.shape[0])
                if inv.shape != expected_shape:
                    raise ValueError(
                        "IR file %s inverse %s has shape %s, expected %s"
                        % (path, inv_key, inv.shape, expected_shape)
                    )
                if not np.all(np.isfinite(inv)):
                    raise ValueError("IR file %s inverse %s is non-finite" % (path, inv_key))
                loaded_inv[inv_key] = inv
        self.beta = beta
        self.lamb = lamb
        self.wmax = float(np.asarray(self.wmax))
        self.F_t_beta = np.atleast_2d(self.F_t_beta)
        self.idx_v0 = np.atleast_1d(self.idx_v0).astype(int)
        self.idx_w0 = int(np.asarray(self.idx_w0))
        self.F_t_f_inv = (
            loaded_inv["F_t_f_inv"] if "F_t_f_inv" in loaded_inv else la.pinv(self.F_t_f)
        )
        self.F_t_b_inv = (
            loaded_inv["F_t_b_inv"] if "F_t_b_inv" in loaded_inv else la.pinv(self.F_t_b)
        )
        self.F_w_f_inv = (
            loaded_inv["F_w_f_inv"] if "F_w_f_inv" in loaded_inv else la.pinv(self.F_w_f)
        )
        self.F_w_b_inv = (
            loaded_inv["F_w_b_inv"] if "F_w_b_inv" in loaded_inv else la.pinv(self.F_w_b)
        )
        self.tau_beta = self.beta * (1.0 - self.tau_beta_rtol)
        mlog.emit(
            mlog.new_logger(self),
            mlog.DEBUG,
            "FreqGrid: loaded IR cache %s nw=%d nv=%d nt_f=%d nt_b=%d"
            % (path, len(self.ws), len(self.vs), len(self.ts_f), len(self.ts_b)),
        )
        return True

    def _ir_path(self):
        if self.ir_file is None:
            return None
        if str(self.ir_file).lower() == "auto":
            path = Path("ir.h5")
        else:
            path = Path(self.ir_file)
        return path

    def _generation_cache_path(self):
        path = self._ir_path()
        if path is not None:
            return path
        if self.save_ir:
            return Path("ir.h5")
        return None

    def _save_ir_if_requested(self):
        if not self.save_ir:
            return
        self._save_ir(self._generation_cache_path())

    def _save_ir(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mlog.emit(
            mlog.new_logger(self),
            mlog.DEBUG,
            "FreqGrid: saving IR cache %s" % path,
        )
        import h5py

        tmp_path = path.with_name("%s.tmp.%d" % (path.name, os.getpid()))
        with h5py.File(tmp_path, "w") as f:
            for key in (
                "F_t_f",
                "F_t_b",
                "F_w_f",
                "F_w_b",
                "F_t_beta",
                "F_t_f2b",
                "F_t_b2f",
                "F_t_f_inv",
                "F_t_b_inv",
                "F_w_f_inv",
                "F_w_b_inv",
                "ts_f",
                "ts_b",
                "ws",
                "vs",
                "idx_v0",
                "idx_w0",
            ):
                f[key] = np.asarray(getattr(self, key))
            f["_beta"] = self.beta
            f["lamb"] = self.lamb
            f["wmax"] = self.wmax
            f["tol"] = self.tol
            f["tau_beta_rtol"] = self.tau_beta_rtol
            for key, value in getattr(self, "_sparse_ir_metadata", {}).items():
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = repr(value)
        tmp_path.replace(path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["stdout"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "cache_transforms"):
            self.cache_transforms = True
        if not hasattr(self, "_F_w2t_f"):
            self._init_transform_cache()
        if self.stdout is None:
            self.stdout = sys.stdout

    @property
    def nw(self):
        """Number of sampled fermionic Matsubara frequencies."""
        return len(self.ws)

    @property
    def nv(self):
        """Number of sampled bosonic Matsubara frequencies."""
        return len(self.vs)

    @property
    def nt_f(self):
        """Number of sampled fermionic imaginary-time points."""
        return len(self.ts_f)

    @property
    def nt_b(self):
        """Number of sampled bosonic imaginary-time points."""
        return len(self.ts_b)

    def trans(self, A_t, F_inv, F):
        """Transform between tau and Matsubara representations.

        The sampled axis is axis 1.  For an input ``A_t`` with shape
        ``(nrow, nt)``, ``F_inv`` maps sampled tau values to IR coefficients and
        ``F`` maps the coefficients to sampled Matsubara frequencies.
        """
        A_l = np.tensordot(A_t, F_inv, axes=((1,), (1,)))
        A_w = np.tensordot(A_l, F, axes=((-1,), (1,)))
        return np.moveaxis(A_w, -1, 1)

    def t2w_f(self, G_t, F_inv=None, F=None):
        """Fermion tau-to-frequency transform."""
        if F_inv is None and F is None:
            return self._transform_or_trans(
                G_t, "_F_t2w_f", self.F_t_f_inv, self.F_w_f
            )
        return self.trans(
            G_t,
            self.F_t_f_inv if F_inv is None else F_inv,
            self.F_w_f if F is None else F,
        )

    def w2t_f(self, G_w, F_inv=None, F=None):
        """Fermion frequency-to-tau transform."""
        if F_inv is None and F is None:
            return self._transform_or_trans(
                G_w, "_F_w2t_f", self.F_w_f_inv, self.F_t_f
            )
        if F_inv is None and F is self.F_t_f2b:
            return self.w2t_f2b(G_w)
        return self.trans(
            G_w,
            self.F_w_f_inv if F_inv is None else F_inv,
            self.F_t_f if F is None else F,
        )

    def w2t_f2b(self, G_w):
        """Fermion frequency-to-boson-tau cross transform."""
        return self._transform_or_trans(
            G_w, "_F_wf2tb", self.F_w_f_inv, self.F_t_f2b
        )

    def w2beta_f(self, G_w):
        r"""Return the fermionic ``tau=beta^-`` endpoint.

        This is algebraically identical to
        ``w2t_f(G_w, F=F_t_beta)[:, 0]`` for the default fermionic frequency
        samples, but it applies the precontracted endpoint row

        .. math::

            B_j = \sum_l F_{\beta l} (F_{\omega}^{-1})_{l j}

        directly as ``sum_j G(iw_j) B_j``.  The sampled Matsubara axis is
        axis 1, matching :meth:`w2t_f`.
        """
        G_w = np.asarray(G_w)
        if G_w.ndim < 2 or G_w.shape[1] != self.nw:
            raise ValueError(
                "G_w must have fermionic Matsubara axis 1 with length %d"
                % self.nw
            )
        if not self.cache_transforms:
            return np.take(self.w2t_f(G_w, F=self.F_t_beta), 0, axis=1)
        return np.tensordot(G_w, self.w2beta_row_f(), axes=((1,), (0,)))

    def w2beta_row_f(self):
        """Return the row mapping fermion frequencies to ``tau=beta^-``."""
        if not self.cache_transforms:
            return np.dot(self.F_t_beta, self.F_w_f_inv)[0]
        if self._F_w2beta_f is None:
            self._init_transform_cache()
        return self._F_w2beta_f

    def t2w_b(self, D_t, F_inv=None, F=None):
        """Boson tau-to-frequency transform."""
        if F_inv is None and F is None:
            return self._transform_or_trans(
                D_t, "_F_t2w_b", self.F_t_b_inv, self.F_w_b
            )
        return self.trans(
            D_t,
            self.F_t_b_inv if F_inv is None else F_inv,
            self.F_w_b if F is None else F,
        )

    def w2t_b(self, D_w, F_inv=None, F=None):
        """Boson frequency-to-tau transform."""
        if F_inv is None and F is None:
            return self._transform_or_trans(
                D_w, "_F_w2t_b", self.F_w_b_inv, self.F_t_b
            )
        if F_inv is None and F is self.F_t_b2f:
            return self.w2t_b2f(D_w)
        return self.trans(
            D_w,
            self.F_w_b_inv if F_inv is None else F_inv,
            self.F_t_b if F is None else F,
        )

    def w2t_b2f(self, D_w):
        """Boson frequency-to-fermion-tau cross transform."""
        return self._transform_or_trans(
            D_w, "_F_wb2tf", self.F_w_b_inv, self.F_t_b2f
        )

    def get_G_inv(self, ws=None, e_k=None, mu=None, sigma=None):
        """Return ``G^-1`` on this context's default ``ws`` if omitted."""
        return get_G_inv(self.ws if ws is None else ws, e_k, mu, sigma=sigma)

    def get_G(self, ws=None, e_k=None, mu=None, sigma=None):
        """Return ``G`` on this context's default ``ws`` if omitted."""
        return get_G(self.ws if ws is None else ws, e_k, mu, sigma=sigma)

    def get_G0(self, ws=None, e_k=None, mu=None):
        """Return noninteracting ``G0`` on this context's default ``ws``."""
        return self.get_G(ws=ws, e_k=e_k, mu=mu, sigma=None)

    def make_rdm1(self, G=None, ws=None, e_k=None, mu=None, sigma=None):
        r"""Return spin-summed one-band density matrix on the radial grid.

        If ``e_k`` and ``mu`` are supplied, the density is evaluated as

        .. math::

            n_k = 2 f(e_k-\mu) - 2 [G - G_0](\beta^-).

        This high-frequency correction avoids reconstructing the full
        discontinuous ``G(beta^-)`` endpoint from sparse-IR data.
        Without the correction, the direct endpoint formula is
        ``n_k = -2 G_k(tau=beta^-)``.  Both formulas return spin-summed
        occupations for a one-band radial grid.
        """
        if G is None:
            G = self.get_G(ws=ws, e_k=e_k, mu=mu, sigma=sigma)
        if e_k is not None and mu is not None:
            G0 = self.get_G0(ws=ws, e_k=e_k, mu=mu)
            dG_beta = self.w2beta_f(G - G0)
            occ0 = model.f_occ(np.asarray(e_k, dtype=float) - float(mu), self.beta)
            return 2.0 * occ0 - (2.0 * dG_beta)
        G_beta = self.w2beta_f(G)
        return -2.0 * G_beta

    def check_w_pair(self, atol=1e-10):
        """Validate reverse-index fermionic frequency pairing."""
        if self.nw % 2:
            raise ValueError("fermion frequency grid must contain pairs.")
        err = np.max(np.abs(self.ws + self.ws[::-1]))
        if err > atol:
            raise ValueError("fermion frequencies are not reverse paired.")
        return err
