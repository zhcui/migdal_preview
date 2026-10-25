# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython kernels for radial convolution hot paths."""

import numpy as np
from cython.parallel cimport prange, threadid
from libc.math cimport INFINITY, fabs, nextafter
from openmp cimport omp_get_max_threads


RADIAL_KERNEL_API_VERSION = 3


cdef struct _PrimitiveCoeff:
    Py_ssize_t m
    double tmp
    double coef


def openmp_max_threads():
    """Return the OpenMP runtime's current maximum thread count."""
    return omp_get_max_threads()


cdef inline _PrimitiveCoeff _primitive_coeff_value(
    double x,
    double[::1] grid,
    double[::1] grid2,
    double[::1] grid3,
    Py_ssize_t n,
) noexcept nogil:
    cdef double upper = nextafter(grid[n - 1], -INFINITY)
    cdef Py_ssize_t lo
    cdef Py_ssize_t hi
    cdef Py_ssize_t mid
    cdef Py_ssize_t m
    cdef double x2
    cdef double x3
    cdef double tmp
    cdef _PrimitiveCoeff out

    if x < grid[0]:
        x = grid[0]
    elif x > upper:
        x = upper

    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if x < grid[mid]:
            hi = mid
        else:
            lo = mid + 1
    m = lo - 1
    if m < 0:
        m = 0
    elif m > n - 2:
        m = n - 2

    x2 = x * x
    x3 = x2 * x
    tmp = 0.5 * (x2 - grid2[m])
    out.m = m
    out.tmp = tmp
    out.coef = (x3 - grid3[m]) / 3.0 - grid[m] * tmp
    return out


cdef inline void _primitive_coeff(
    double x,
    double[::1] grid,
    double[::1] grid2,
    double[::1] grid3,
    Py_ssize_t n,
    Py_ssize_t* m_out,
    double* tmp_out,
    double* coef_out,
) noexcept nogil:
    cdef _PrimitiveCoeff out = _primitive_coeff_value(x, grid, grid2, grid3, n)
    m_out[0] = out.m
    tmp_out[0] = out.tmp
    coef_out[0] = out.coef


cdef inline double complex _primitive_sta_from_coeff(
    Py_ssize_t m,
    double tmp,
    double coef,
    double complex[::1] y,
    double complex[::1] s,
    double complex[::1] P,
) noexcept nogil:
    return tmp * y[m] + coef * s[m] + P[m]


cdef inline void _store_sweep_coeff(
    double x,
    double[::1] grid,
    double[::1] grid2,
    double[::1] grid3,
    Py_ssize_t n,
    double upper,
    Py_ssize_t* m,
    int[:, ::1] m_out,
    double[:, ::1] tmp_out,
    double[:, ::1] coef_out,
    Py_ssize_t row,
    Py_ssize_t col,
) noexcept nogil:
    cdef double x2
    cdef double x3
    cdef double tmp

    if x < grid[0]:
        x = grid[0]
        m[0] = 0
    elif x > upper:
        x = upper
        m[0] = n - 2
    else:
        while m[0] < n - 2 and x >= grid[m[0] + 1]:
            m[0] += 1
        while m[0] > 0 and x < grid[m[0]]:
            m[0] -= 1

    x2 = x * x
    x3 = x2 * x
    tmp = 0.5 * (x2 - grid2[m[0]])
    m_out[row, col] = <int>m[0]
    tmp_out[row, col] = tmp
    coef_out[row, col] = (x3 - grid3[m[0]]) / 3.0 - grid[m[0]] * tmp


def build_sigma_pair_geometry(
    double[::1] k_grid,
    double[::1] q,
    double[::1] q2,
    double[::1] q3,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    Py_ssize_t num_threads=1,
):
    """Precompute rank-local pair primitive intervals for sigma kernels."""
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nq = q.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object plus_m_arr
    cdef object minus_m_arr
    cdef object plus_tmp_arr
    cdef object minus_tmp_arr
    cdef object plus_coef_arr
    cdef object minus_coef_arr
    cdef int[:, ::1] plus_m
    cdef int[:, ::1] minus_m
    cdef double[:, ::1] plus_tmp
    cdef double[:, ::1] minus_tmp
    cdef double[:, ::1] plus_coef
    cdef double[:, ::1] minus_coef
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ik
    cdef Py_ssize_t m
    cdef double kp
    cdef double upper

    if nq < 2:
        raise ValueError("q grid must contain at least two points.")
    if q2.shape[0] != nq or q3.shape[0] != nq:
        raise ValueError("q power arrays are inconsistent with q grid.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")

    plus_m_arr = np.empty((nloc, nk), dtype=np.intc)
    minus_m_arr = np.empty((nloc, nk), dtype=np.intc)
    plus_tmp_arr = np.empty((nloc, nk), dtype=np.float64)
    minus_tmp_arr = np.empty((nloc, nk), dtype=np.float64)
    plus_coef_arr = np.empty((nloc, nk), dtype=np.float64)
    minus_coef_arr = np.empty((nloc, nk), dtype=np.float64)
    plus_m = plus_m_arr
    minus_m = minus_m_arr
    plus_tmp = plus_tmp_arr
    minus_tmp = minus_tmp_arr
    plus_coef = plus_coef_arr
    minus_coef = minus_coef_arr
    upper = nextafter(q[nq - 1], -INFINITY)

    with nogil:
        for local_j in prange(
            nloc,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            j = kloc0 + local_j
            kp = k_grid[j]

            m = 0
            for ik in range(nk):
                _store_sweep_coeff(
                    k_grid[ik] + kp, q, q2, q3, nq, upper, &m,
                    plus_m, plus_tmp, plus_coef, local_j, ik,
                )

            m = 0
            for ik in range(j, -1, -1):
                _store_sweep_coeff(
                    kp - k_grid[ik], q, q2, q3, nq, upper, &m,
                    minus_m, minus_tmp, minus_coef, local_j, ik,
                )

            m = 0
            for ik in range(j + 1, nk):
                _store_sweep_coeff(
                    k_grid[ik] - kp, q, q2, q3, nq, upper, &m,
                    minus_m, minus_tmp, minus_coef, local_j, ik,
                )

    return (
        plus_m_arr,
        minus_m_arr,
        plus_tmp_arr,
        minus_tmp_arr,
        plus_coef_arr,
        minus_coef_arr,
    )


def build_pi_pair_geometry(
    double[::1] source_k,
    double[::1] q_grid,
    double[::1] primitive_k,
    double[::1] primitive_k2,
    double[::1] primitive_k3,
    Py_ssize_t qloc0,
    Py_ssize_t qloc1,
    Py_ssize_t num_threads=1,
):
    """Precompute rank-local pair primitive intervals for Pi kernels."""
    cdef Py_ssize_t nk = source_k.shape[0]
    cdef Py_ssize_t nq = q_grid.shape[0]
    cdef Py_ssize_t nprim = primitive_k.shape[0]
    cdef Py_ssize_t nloc = qloc1 - qloc0
    cdef object plus_m_arr
    cdef object minus_m_arr
    cdef object plus_tmp_arr
    cdef object minus_tmp_arr
    cdef object plus_coef_arr
    cdef object minus_coef_arr
    cdef int[:, ::1] plus_m
    cdef int[:, ::1] minus_m
    cdef double[:, ::1] plus_tmp
    cdef double[:, ::1] minus_tmp
    cdef double[:, ::1] plus_coef
    cdef double[:, ::1] minus_coef
    cdef Py_ssize_t local_i
    cdef Py_ssize_t iq
    cdef Py_ssize_t ik
    cdef Py_ssize_t pivot
    cdef Py_ssize_t lo
    cdef Py_ssize_t hi
    cdef Py_ssize_t mid
    cdef Py_ssize_t m
    cdef double qv
    cdef double upper

    if nprim < 2:
        raise ValueError("primitive k grid must contain at least two points.")
    if primitive_k2.shape[0] != nprim or primitive_k3.shape[0] != nprim:
        raise ValueError("primitive k power arrays are inconsistent.")
    if qloc0 < 0 or qloc1 < qloc0 or qloc1 > nq:
        raise ValueError("invalid local q range.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")

    plus_m_arr = np.empty((nloc, nk), dtype=np.intc)
    minus_m_arr = np.empty((nloc, nk), dtype=np.intc)
    plus_tmp_arr = np.empty((nloc, nk), dtype=np.float64)
    minus_tmp_arr = np.empty((nloc, nk), dtype=np.float64)
    plus_coef_arr = np.empty((nloc, nk), dtype=np.float64)
    minus_coef_arr = np.empty((nloc, nk), dtype=np.float64)
    plus_m = plus_m_arr
    minus_m = minus_m_arr
    plus_tmp = plus_tmp_arr
    minus_tmp = minus_tmp_arr
    plus_coef = plus_coef_arr
    minus_coef = minus_coef_arr
    upper = nextafter(primitive_k[nprim - 1], -INFINITY)

    with nogil:
        for local_i in prange(
            nloc,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            iq = qloc0 + local_i
            qv = q_grid[iq]

            m = 0
            for ik in range(nk):
                _store_sweep_coeff(
                    source_k[ik] + qv,
                    primitive_k,
                    primitive_k2,
                    primitive_k3,
                    nprim,
                    upper,
                    &m,
                    plus_m,
                    plus_tmp,
                    plus_coef,
                    local_i,
                    ik,
                )

            lo = 0
            hi = nk
            while lo < hi:
                mid = (lo + hi) // 2
                if source_k[mid] <= qv:
                    lo = mid + 1
                else:
                    hi = mid
            pivot = lo - 1

            m = 0
            for ik in range(pivot, -1, -1):
                _store_sweep_coeff(
                    qv - source_k[ik],
                    primitive_k,
                    primitive_k2,
                    primitive_k3,
                    nprim,
                    upper,
                    &m,
                    minus_m,
                    minus_tmp,
                    minus_coef,
                    local_i,
                    ik,
                )

            m = 0
            for ik in range(pivot + 1, nk):
                _store_sweep_coeff(
                    source_k[ik] - qv,
                    primitive_k,
                    primitive_k2,
                    primitive_k3,
                    nprim,
                    upper,
                    &m,
                    minus_m,
                    minus_tmp,
                    minus_coef,
                    local_i,
                    ik,
                )

    return (
        plus_m_arr,
        minus_m_arr,
        plus_tmp_arr,
        minus_tmp_arr,
        plus_coef_arr,
        minus_coef_arr,
    )


def sigma_kernel(
    double[::1] k_grid,
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double[::1] q,
    double[::1] q2,
    double[::1] q3,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    double complex[::1] yS,
    double complex[::1] sS,
    double complex[::1] PS,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    Py_ssize_t num_threads=1,
):
    """Accumulate rank-local Sigma_t and Sigma0 without materializing dI blocks."""
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
    cdef Py_ssize_t nq = q.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object Sigma_t_arr
    cdef object Sigma0_arr
    cdef double complex[:, ::1] Sigma_t
    cdef double complex[::1] Sigma0
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef double kp
    cdef double complex sta_val
    cdef double complex dyn_val
    cdef _PrimitiveCoeff plus_coeff
    cdef _PrimitiveCoeff minus_coeff

    if nq < 2:
        raise ValueError("q grid must contain at least two points.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if weighted_S_t.shape[0] != nk or weighted_S_beta.shape[0] != nk:
        raise ValueError("weighted source shapes are inconsistent with k_grid.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    if yD.shape[0] != nq or yD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if yS.shape[0] != nq or sS.shape[0] != nq - 1 or PS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    if use_diag:
        if dyn_avg.shape[0] != nk or dyn_avg.shape[1] != nt:
            raise ValueError("diagonal dynamic average shape mismatch.")
        if sta_avg.shape[0] != nk:
            raise ValueError("diagonal static average shape mismatch.")

    Sigma_t_arr = np.zeros((nloc, nt), dtype=np.complex128)
    Sigma0_arr = np.zeros(nloc, dtype=np.complex128)
    Sigma_t = Sigma_t_arr
    Sigma0 = Sigma0_arr

    with nogil:
        for local_j in prange(
            nloc,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            j = kloc0 + local_j
            kp = k_grid[j]
            for ik in range(nk):
                if use_diag and ik == j:
                    sta_val = sta_avg[j]
                    for it in range(nt):
                        Sigma_t[local_j, it] += weighted_S_t[ik, it] * dyn_avg[j, it]
                    Sigma0[local_j] += weighted_S_beta[ik] * sta_val
                    continue

                plus_coeff = _primitive_coeff_value(
                    k_grid[ik] + kp, q, q2, q3, nq
                )
                minus_coeff = _primitive_coeff_value(
                    fabs(k_grid[ik] - kp), q, q2, q3, nq
                )
                sta_val = (
                    _primitive_sta_from_coeff(
                        plus_coeff.m, plus_coeff.tmp, plus_coeff.coef, yS, sS, PS
                    )
                    - _primitive_sta_from_coeff(
                        minus_coeff.m, minus_coeff.tmp, minus_coeff.coef,
                        yS, sS, PS,
                    )
                )
                for it in range(nt):
                    dyn_val = (
                        plus_coeff.tmp * yD[plus_coeff.m, it]
                        + plus_coeff.coef * sD[plus_coeff.m, it]
                        + PD[plus_coeff.m, it]
                        - minus_coeff.tmp * yD[minus_coeff.m, it]
                        - minus_coeff.coef * sD[minus_coeff.m, it]
                        - PD[minus_coeff.m, it]
                    )
                    Sigma_t[local_j, it] += weighted_S_t[ik, it] * dyn_val
                Sigma0[local_j] += weighted_S_beta[ik] * sta_val

    return Sigma_t_arr, Sigma0_arr


def sigma_kernel_geometry(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    double complex[::1] yS,
    double complex[::1] sS,
    double complex[::1] PS,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_tmp,
    double[:, ::1] minus_tmp,
    double[:, ::1] plus_coef,
    double[:, ::1] minus_coef,
    Py_ssize_t kloc0,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    Py_ssize_t num_threads=1,
):
    """Accumulate Sigma with precomputed pair primitive intervals."""
    cdef Py_ssize_t nloc = plus_m.shape[0]
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
    cdef Py_ssize_t nq = yD.shape[0]
    cdef object Sigma_t_arr
    cdef object Sigma0_arr
    cdef double complex[:, ::1] Sigma_t
    cdef double complex[::1] Sigma0
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double pt
    cdef double mt
    cdef double pc
    cdef double mc
    cdef double complex sta_val
    cdef double complex dyn_val
    cdef double complex* outrow
    cdef double complex* Srow
    cdef double complex* diagrow
    cdef double complex* yP
    cdef double complex* sP
    cdef double complex* PP
    cdef double complex* yM
    cdef double complex* sM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* ptrow
    cdef double* mtrow
    cdef double* pcrow
    cdef double* mcrow

    if weighted_S_beta.shape[0] != nk:
        raise ValueError("weighted source shapes are inconsistent.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    if yD.shape[1] != nt:
        raise ValueError("dynamic primitive tau dimension mismatch.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if yS.shape[0] != nq or sS.shape[0] != nq - 1 or PS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    if (
        plus_m.shape[1] != nk
        or minus_m.shape[0] != nloc
        or minus_m.shape[1] != nk
        or plus_tmp.shape[0] != nloc
        or plus_tmp.shape[1] != nk
        or minus_tmp.shape[0] != nloc
        or minus_tmp.shape[1] != nk
        or plus_coef.shape[0] != nloc
        or plus_coef.shape[1] != nk
        or minus_coef.shape[0] != nloc
        or minus_coef.shape[1] != nk
    ):
        raise ValueError("pair geometry shape mismatch.")
    if use_diag:
        if dyn_avg.shape[0] != nk or dyn_avg.shape[1] != nt:
            raise ValueError("diagonal dynamic average shape mismatch.")
        if sta_avg.shape[0] != nk:
            raise ValueError("diagonal static average shape mismatch.")

    Sigma_t_arr = np.zeros((nloc, nt), dtype=np.complex128)
    Sigma0_arr = np.zeros(nloc, dtype=np.complex128)
    Sigma_t = Sigma_t_arr
    Sigma0 = Sigma0_arr

    with nogil:
        for local_j in prange(
            nloc,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            j = kloc0 + local_j
            outrow = &Sigma_t[local_j, 0]
            pmrow = &plus_m[local_j, 0]
            mmrow = &minus_m[local_j, 0]
            ptrow = &plus_tmp[local_j, 0]
            mtrow = &minus_tmp[local_j, 0]
            pcrow = &plus_coef[local_j, 0]
            mcrow = &minus_coef[local_j, 0]
            for ik in range(nk):
                Srow = &weighted_S_t[ik, 0]
                if use_diag and ik == j:
                    sta_val = sta_avg[j]
                    diagrow = &dyn_avg[j, 0]
                    for it in range(nt):
                        outrow[it] += Srow[it] * diagrow[it]
                    Sigma0[local_j] += weighted_S_beta[ik] * sta_val
                    continue

                pm = <Py_ssize_t>pmrow[ik]
                mm = <Py_ssize_t>mmrow[ik]
                pt = ptrow[ik]
                mt = mtrow[ik]
                pc = pcrow[ik]
                mc = mcrow[ik]
                sta_val = (
                    _primitive_sta_from_coeff(pm, pt, pc, yS, sS, PS)
                    - _primitive_sta_from_coeff(mm, mt, mc, yS, sS, PS)
                )
                yP = &yD[pm, 0]
                sP = &sD[pm, 0]
                PP = &PD[pm, 0]
                yM = &yD[mm, 0]
                sM = &sD[mm, 0]
                PM = &PD[mm, 0]
                for it in range(nt):
                    dyn_val = (
                        pt * yP[it]
                        + pc * sP[it]
                        + PP[it]
                        - mt * yM[it]
                        - mc * sM[it]
                        - PM[it]
                    )
                    outrow[it] += Srow[it] * dyn_val
                Sigma0[local_j] += weighted_S_beta[ik] * sta_val

    return Sigma_t_arr, Sigma0_arr


def sigma_kernel_geometry_triangular(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    double complex[::1] yS,
    double complex[::1] sS,
    double complex[::1] PS,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_tmp,
    double[:, ::1] minus_tmp,
    double[:, ::1] plus_coef,
    double[:, ::1] minus_coef,
    Py_ssize_t rank,
    Py_ssize_t nranks,
    Py_ssize_t tile_size,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
):
    """Accumulate a rank-owned subset of symmetric Sigma pairs.

    The returned arrays have full ``nk`` rows and contain this rank's partial
    triangular-pair contribution.  The Python wrapper reduces them over MPI.
    """
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
    cdef Py_ssize_t nq = yD.shape[0]
    cdef object Sigma_t_arr
    cdef object Sigma0_arr
    cdef double complex[:, ::1] Sigma_t
    cdef double complex[::1] Sigma0
    cdef Py_ssize_t tile_id
    cdef Py_ssize_t jb
    cdef Py_ssize_t kb
    cdef Py_ssize_t jend
    cdef Py_ssize_t kend
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t k0
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double pt
    cdef double mt
    cdef double pc
    cdef double mc
    cdef double complex sta_val
    cdef double complex dyn_val
    cdef double complex* outj
    cdef double complex* outk
    cdef double complex* Srowj
    cdef double complex* Srowk
    cdef double complex* diagrow
    cdef double complex* yP
    cdef double complex* sP
    cdef double complex* PP
    cdef double complex* yM
    cdef double complex* sM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* ptrow
    cdef double* mtrow
    cdef double* pcrow
    cdef double* mcrow

    if nranks < 1:
        raise ValueError("nranks must be positive.")
    if rank < 0 or rank >= nranks:
        raise ValueError("invalid rank for triangular sigma kernel.")
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    if weighted_S_beta.shape[0] != nk:
        raise ValueError("weighted source shapes are inconsistent.")
    if yD.shape[1] != nt:
        raise ValueError("dynamic primitive tau dimension mismatch.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if yS.shape[0] != nq or sS.shape[0] != nq - 1 or PS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    if (
        plus_m.shape[0] != nk
        or plus_m.shape[1] != nk
        or minus_m.shape[0] != nk
        or minus_m.shape[1] != nk
        or plus_tmp.shape[0] != nk
        or plus_tmp.shape[1] != nk
        or minus_tmp.shape[0] != nk
        or minus_tmp.shape[1] != nk
        or plus_coef.shape[0] != nk
        or plus_coef.shape[1] != nk
        or minus_coef.shape[0] != nk
        or minus_coef.shape[1] != nk
    ):
        raise ValueError("full pair geometry shape mismatch.")
    if use_diag:
        if dyn_avg.shape[0] != nk or dyn_avg.shape[1] != nt:
            raise ValueError("diagonal dynamic average shape mismatch.")
        if sta_avg.shape[0] != nk:
            raise ValueError("diagonal static average shape mismatch.")

    Sigma_t_arr = np.zeros((nk, nt), dtype=np.complex128)
    Sigma0_arr = np.zeros(nk, dtype=np.complex128)
    Sigma_t = Sigma_t_arr
    Sigma0 = Sigma0_arr

    with nogil:
        tile_id = 0
        jb = 0
        while jb < nk:
            jend = jb + tile_size
            if jend > nk:
                jend = nk
            kb = jb
            while kb < nk:
                kend = kb + tile_size
                if kend > nk:
                    kend = nk
                if tile_id % nranks != rank:
                    tile_id += 1
                    kb += tile_size
                    continue

                for j in range(jb, jend):
                    pmrow = &plus_m[j, 0]
                    mmrow = &minus_m[j, 0]
                    ptrow = &plus_tmp[j, 0]
                    mtrow = &minus_tmp[j, 0]
                    pcrow = &plus_coef[j, 0]
                    mcrow = &minus_coef[j, 0]
                    outj = &Sigma_t[j, 0]
                    Srowj = &weighted_S_t[j, 0]
                    k0 = kb
                    if kb == jb:
                        k0 = j
                    for k in range(k0, kend):
                        Srowk = &weighted_S_t[k, 0]
                        if use_diag and k == j:
                            sta_val = sta_avg[j]
                            diagrow = &dyn_avg[j, 0]
                            for it in range(nt):
                                outj[it] += Srowk[it] * diagrow[it]
                            Sigma0[j] += weighted_S_beta[k] * sta_val
                            continue

                        pm = <Py_ssize_t>pmrow[k]
                        mm = <Py_ssize_t>mmrow[k]
                        pt = ptrow[k]
                        mt = mtrow[k]
                        pc = pcrow[k]
                        mc = mcrow[k]
                        sta_val = (
                            _primitive_sta_from_coeff(pm, pt, pc, yS, sS, PS)
                            - _primitive_sta_from_coeff(mm, mt, mc, yS, sS, PS)
                        )
                        yP = &yD[pm, 0]
                        sP = &sD[pm, 0]
                        PP = &PD[pm, 0]
                        yM = &yD[mm, 0]
                        sM = &sD[mm, 0]
                        PM = &PD[mm, 0]
                        Sigma0[j] += weighted_S_beta[k] * sta_val

                        if k != j:
                            outk = &Sigma_t[k, 0]
                            for it in range(nt):
                                dyn_val = (
                                    pt * yP[it]
                                    + pc * sP[it]
                                    + PP[it]
                                    - mt * yM[it]
                                    - mc * sM[it]
                                    - PM[it]
                                )
                                outj[it] += Srowk[it] * dyn_val
                                outk[it] += Srowj[it] * dyn_val
                            Sigma0[k] += weighted_S_beta[j] * sta_val
                        else:
                            for it in range(nt):
                                dyn_val = (
                                    pt * yP[it]
                                    + pc * sP[it]
                                    + PP[it]
                                    - mt * yM[it]
                                    - mc * sM[it]
                                    - PM[it]
                                )
                                outj[it] += Srowk[it] * dyn_val

                tile_id += 1
                kb += tile_size
            jb += tile_size

    return Sigma_t_arr, Sigma0_arr


def sigma_kernel_geometry_triangular_omp(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    double complex[::1] yS,
    double complex[::1] sS,
    double complex[::1] PS,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_tmp,
    double[:, ::1] minus_tmp,
    double[:, ::1] plus_coef,
    double[:, ::1] minus_coef,
    Py_ssize_t rank,
    Py_ssize_t nranks,
    Py_ssize_t tile_size,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    Py_ssize_t num_threads,
):
    """OpenMP triangular Sigma accumulation with per-thread private outputs."""
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
    cdef Py_ssize_t nq = yD.shape[0]
    cdef object Sigma_t_arr
    cdef object Sigma0_arr
    cdef object Sigma_t_priv_arr
    cdef object Sigma0_priv_arr
    cdef object tile_j_arr
    cdef object tile_k_arr
    cdef double complex[:, ::1] Sigma_t
    cdef double complex[::1] Sigma0
    cdef double complex[:, :, ::1] Sigma_t_priv
    cdef double complex[:, ::1] Sigma0_priv
    cdef Py_ssize_t[::1] tile_j
    cdef Py_ssize_t[::1] tile_k
    cdef Py_ssize_t tile_id
    cdef Py_ssize_t tile_count
    cdef Py_ssize_t tile_pos
    cdef Py_ssize_t tid
    cdef Py_ssize_t jb
    cdef Py_ssize_t kb
    cdef Py_ssize_t jend
    cdef Py_ssize_t kend
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t k0
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double pt
    cdef double mt
    cdef double pc
    cdef double mc
    cdef double complex sta_val
    cdef double complex dyn_val
    cdef double complex* outj
    cdef double complex* outk
    cdef double complex* Srowj
    cdef double complex* Srowk
    cdef double complex* diagrow
    cdef double complex* yP
    cdef double complex* sP
    cdef double complex* PP
    cdef double complex* yM
    cdef double complex* sM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* ptrow
    cdef double* mtrow
    cdef double* pcrow
    cdef double* mcrow

    if num_threads <= 1:
        return sigma_kernel_geometry_triangular(
            weighted_S_t,
            weighted_S_beta,
            yD,
            sD,
            PD,
            yS,
            sS,
            PS,
            plus_m,
            minus_m,
            plus_tmp,
            minus_tmp,
            plus_coef,
            minus_coef,
            rank,
            nranks,
            tile_size,
            use_diag,
            dyn_avg,
            sta_avg,
        )
    if nranks < 1:
        raise ValueError("nranks must be positive.")
    if rank < 0 or rank >= nranks:
        raise ValueError("invalid rank for triangular sigma kernel.")
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    if weighted_S_beta.shape[0] != nk:
        raise ValueError("weighted source shapes are inconsistent.")
    if yD.shape[1] != nt:
        raise ValueError("dynamic primitive tau dimension mismatch.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if yS.shape[0] != nq or sS.shape[0] != nq - 1 or PS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    if (
        plus_m.shape[0] != nk
        or plus_m.shape[1] != nk
        or minus_m.shape[0] != nk
        or minus_m.shape[1] != nk
        or plus_tmp.shape[0] != nk
        or plus_tmp.shape[1] != nk
        or minus_tmp.shape[0] != nk
        or minus_tmp.shape[1] != nk
        or plus_coef.shape[0] != nk
        or plus_coef.shape[1] != nk
        or minus_coef.shape[0] != nk
        or minus_coef.shape[1] != nk
    ):
        raise ValueError("full pair geometry shape mismatch.")
    if use_diag:
        if dyn_avg.shape[0] != nk or dyn_avg.shape[1] != nt:
            raise ValueError("diagonal dynamic average shape mismatch.")
        if sta_avg.shape[0] != nk:
            raise ValueError("diagonal static average shape mismatch.")

    tile_count = 0
    tile_id = 0
    jb = 0
    while jb < nk:
        kb = jb
        while kb < nk:
            if tile_id % nranks == rank:
                tile_count += 1
            tile_id += 1
            kb += tile_size
        jb += tile_size

    tile_j_arr = np.empty(tile_count, dtype=np.intp)
    tile_k_arr = np.empty(tile_count, dtype=np.intp)
    tile_j = tile_j_arr
    tile_k = tile_k_arr
    tile_count = 0
    tile_id = 0
    jb = 0
    while jb < nk:
        kb = jb
        while kb < nk:
            if tile_id % nranks == rank:
                tile_j[tile_count] = jb
                tile_k[tile_count] = kb
                tile_count += 1
            tile_id += 1
            kb += tile_size
        jb += tile_size

    Sigma_t_arr = np.zeros((nk, nt), dtype=np.complex128)
    Sigma0_arr = np.zeros(nk, dtype=np.complex128)
    Sigma_t_priv_arr = np.zeros((num_threads, nk, nt), dtype=np.complex128)
    Sigma0_priv_arr = np.zeros((num_threads, nk), dtype=np.complex128)
    Sigma_t = Sigma_t_arr
    Sigma0 = Sigma0_arr
    Sigma_t_priv = Sigma_t_priv_arr
    Sigma0_priv = Sigma0_priv_arr

    with nogil:
        for tile_pos in prange(
            tile_count, schedule="dynamic", chunksize=1, num_threads=num_threads
        ):
            tid = threadid()
            jb = tile_j[tile_pos]
            kb = tile_k[tile_pos]
            jend = jb + tile_size
            if jend > nk:
                jend = nk
            kend = kb + tile_size
            if kend > nk:
                kend = nk

            for j in range(jb, jend):
                pmrow = &plus_m[j, 0]
                mmrow = &minus_m[j, 0]
                ptrow = &plus_tmp[j, 0]
                mtrow = &minus_tmp[j, 0]
                pcrow = &plus_coef[j, 0]
                mcrow = &minus_coef[j, 0]
                outj = &Sigma_t_priv[tid, j, 0]
                Srowj = &weighted_S_t[j, 0]
                k0 = kb
                if kb == jb:
                    k0 = j
                for k in range(k0, kend):
                    Srowk = &weighted_S_t[k, 0]
                    if use_diag and k == j:
                        sta_val = sta_avg[j]
                        diagrow = &dyn_avg[j, 0]
                        for it in range(nt):
                            outj[it] += Srowk[it] * diagrow[it]
                        Sigma0_priv[tid, j] += weighted_S_beta[k] * sta_val
                        continue

                    pm = <Py_ssize_t>pmrow[k]
                    mm = <Py_ssize_t>mmrow[k]
                    pt = ptrow[k]
                    mt = mtrow[k]
                    pc = pcrow[k]
                    mc = mcrow[k]
                    sta_val = (
                        _primitive_sta_from_coeff(pm, pt, pc, yS, sS, PS)
                        - _primitive_sta_from_coeff(mm, mt, mc, yS, sS, PS)
                    )
                    yP = &yD[pm, 0]
                    sP = &sD[pm, 0]
                    PP = &PD[pm, 0]
                    yM = &yD[mm, 0]
                    sM = &sD[mm, 0]
                    PM = &PD[mm, 0]
                    Sigma0_priv[tid, j] += weighted_S_beta[k] * sta_val

                    if k != j:
                        outk = &Sigma_t_priv[tid, k, 0]
                        for it in range(nt):
                            dyn_val = (
                                pt * yP[it]
                                + pc * sP[it]
                                + PP[it]
                                - mt * yM[it]
                                - mc * sM[it]
                                - PM[it]
                            )
                            outj[it] += Srowk[it] * dyn_val
                            outk[it] += Srowj[it] * dyn_val
                        Sigma0_priv[tid, k] += weighted_S_beta[j] * sta_val
                    else:
                        for it in range(nt):
                            dyn_val = (
                                pt * yP[it]
                                + pc * sP[it]
                                + PP[it]
                                - mt * yM[it]
                                - mc * sM[it]
                                - PM[it]
                            )
                            outj[it] += Srowk[it] * dyn_val

        for j in prange(nk, schedule="static", num_threads=num_threads):
            outj = &Sigma_t[j, 0]
            for tid in range(num_threads):
                Sigma0[j] += Sigma0_priv[tid, j]
                Srowj = &Sigma_t_priv[tid, j, 0]
                for it in range(nt):
                    outj[it] += Srowj[it]

    return Sigma_t_arr, Sigma0_arr


def prepare_cache_kernel(
    double[::1] k_grid,
    double[::1] q,
    double[::1] q2,
    double[::1] q3,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    double complex[::1] yS,
    double complex[::1] sS,
    double complex[::1] PS,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    Py_ssize_t num_threads=1,
):
    """Build rank-local radial blocks for the cached Sigma path."""
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nt = yD.shape[1]
    cdef Py_ssize_t nq = q.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object dyn_arr
    cdef object sta_arr
    cdef double complex[:, :, ::1] dyn
    cdef double complex[:, ::1] sta
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef double kp
    cdef _PrimitiveCoeff plus_coeff
    cdef _PrimitiveCoeff minus_coeff

    if nq < 2:
        raise ValueError("q grid must contain at least two points.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if yS.shape[0] != nq or sS.shape[0] != nq - 1 or PS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    if use_diag:
        if dyn_avg.shape[0] != nk or dyn_avg.shape[1] != nt:
            raise ValueError("diagonal dynamic average shape mismatch.")
        if sta_avg.shape[0] != nk:
            raise ValueError("diagonal static average shape mismatch.")

    dyn_arr = np.empty((nloc, nk, nt), dtype=np.complex128)
    sta_arr = np.empty((nloc, nk), dtype=np.complex128)
    dyn = dyn_arr
    sta = sta_arr

    with nogil:
        for local_j in prange(
            nloc,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            j = kloc0 + local_j
            kp = k_grid[j]
            for ik in range(nk):
                if use_diag and ik == j:
                    sta[local_j, ik] = sta_avg[j]
                    for it in range(nt):
                        dyn[local_j, ik, it] = dyn_avg[j, it]
                    continue

                plus_coeff = _primitive_coeff_value(
                    k_grid[ik] + kp, q, q2, q3, nq
                )
                minus_coeff = _primitive_coeff_value(
                    fabs(k_grid[ik] - kp), q, q2, q3, nq
                )
                sta[local_j, ik] = (
                    _primitive_sta_from_coeff(
                        plus_coeff.m, plus_coeff.tmp, plus_coeff.coef, yS, sS, PS
                    )
                    - _primitive_sta_from_coeff(
                        minus_coeff.m, minus_coeff.tmp, minus_coeff.coef,
                        yS, sS, PS,
                    )
                )
                for it in range(nt):
                    dyn[local_j, ik, it] = (
                        plus_coeff.tmp * yD[plus_coeff.m, it]
                        + plus_coeff.coef * sD[plus_coeff.m, it]
                        + PD[plus_coeff.m, it]
                        - minus_coeff.tmp * yD[minus_coeff.m, it]
                        - minus_coeff.coef * sD[minus_coeff.m, it]
                        - PD[minus_coeff.m, it]
                    )

    return dyn_arr, sta_arr


def diag_cells_kernel(
    double[::1] k_grid,
    double[:, ::1] bounds,
    double[::1] x_nodes,
    double[::1] w_nodes,
    double[::1] q,
    double[::1] q2,
    double[::1] q3,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    double complex[::1] yS,
    double complex[::1] sS,
    double complex[::1] PS,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
):
    """Average diagonal radial blocks over precomputed k cells."""
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nt = yD.shape[1]
    cdef Py_ssize_t nq = q.shape[0]
    cdef Py_ssize_t ncell = x_nodes.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object dyn_arr
    cdef object sta_arr
    cdef double complex[:, ::1] dyn
    cdef double complex[::1] sta
    cdef Py_ssize_t local_ik
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef Py_ssize_t ix
    cdef Py_ssize_t part
    cdef Py_ssize_t plus_m
    cdef Py_ssize_t minus_m
    cdef double kp
    cdef double lo
    cdef double hi
    cdef double a
    cdef double b
    cdef double mid
    cdef double half
    cdef double kg
    cdef double weight
    cdef double inv_width
    cdef double plus_tmp
    cdef double minus_tmp
    cdef double plus_coef
    cdef double minus_coef
    cdef double complex sta_val
    cdef double complex dyn_val

    if nq < 2:
        raise ValueError("q grid must contain at least two points.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if bounds.shape[0] != nk or bounds.shape[1] != 2:
        raise ValueError("cell bounds shape mismatch.")
    if w_nodes.shape[0] != ncell:
        raise ValueError("quadrature node/weight mismatch.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if yS.shape[0] != nq or sS.shape[0] != nq - 1 or PS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    for local_ik in range(nloc):
        ik = kloc0 + local_ik
        if bounds[ik, 1] <= bounds[ik, 0]:
            raise ValueError("invalid empty diagonal cell.")

    dyn_arr = np.empty((nloc, nt), dtype=np.complex128)
    sta_arr = np.empty(nloc, dtype=np.complex128)
    dyn = dyn_arr
    sta = sta_arr

    with nogil:
        for local_ik in range(nloc):
            ik = kloc0 + local_ik
            kp = k_grid[ik]
            lo = bounds[ik, 0]
            hi = bounds[ik, 1]
            sta[local_ik] = 0.0
            for it in range(nt):
                dyn[local_ik, it] = 0.0

            for part in range(2):
                if part == 0:
                    a = lo
                    b = kp if kp < hi else hi
                else:
                    a = kp if kp > lo else lo
                    b = hi
                if b <= a:
                    continue

                mid = 0.5 * (a + b)
                half = 0.5 * (b - a)
                for ix in range(ncell):
                    kg = mid + half * x_nodes[ix]
                    weight = half * w_nodes[ix]
                    _primitive_coeff(
                        kg + kp, q, q2, q3, nq,
                        &plus_m, &plus_tmp, &plus_coef,
                    )
                    _primitive_coeff(
                        fabs(kg - kp), q, q2, q3, nq,
                        &minus_m, &minus_tmp, &minus_coef,
                    )
                    sta_val = (
                        _primitive_sta_from_coeff(plus_m, plus_tmp, plus_coef, yS, sS, PS)
                        - _primitive_sta_from_coeff(minus_m, minus_tmp, minus_coef, yS, sS, PS)
                    )
                    sta[local_ik] += weight * sta_val
                    for it in range(nt):
                        dyn_val = (
                            plus_tmp * yD[plus_m, it]
                            + plus_coef * sD[plus_m, it]
                            + PD[plus_m, it]
                            - minus_tmp * yD[minus_m, it]
                            - minus_coef * sD[minus_m, it]
                            - PD[minus_m, it]
                        )
                        dyn[local_ik, it] += weight * dyn_val

            inv_width = 1.0 / (hi - lo)
            sta[local_ik] *= inv_width
            for it in range(nt):
                dyn[local_ik, it] *= inv_width

    return dyn_arr, sta_arr


def diag_cells_dyn_kernel(
    double[::1] k_grid,
    double[:, ::1] bounds,
    double[::1] x_nodes,
    double[::1] w_nodes,
    double[::1] q,
    double[::1] q2,
    double[::1] q3,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
):
    """Average only dynamic diagonal radial blocks over precomputed k cells."""
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nt = yD.shape[1]
    cdef Py_ssize_t nq = q.shape[0]
    cdef Py_ssize_t ncell = x_nodes.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object dyn_arr
    cdef double complex[:, ::1] dyn
    cdef Py_ssize_t local_ik
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef Py_ssize_t ix
    cdef Py_ssize_t part
    cdef Py_ssize_t plus_m
    cdef Py_ssize_t minus_m
    cdef double kp
    cdef double lo
    cdef double hi
    cdef double a
    cdef double b
    cdef double mid
    cdef double half
    cdef double kg
    cdef double weight
    cdef double inv_width
    cdef double plus_tmp
    cdef double minus_tmp
    cdef double plus_coef
    cdef double minus_coef
    cdef double complex dyn_val

    if nq < 2:
        raise ValueError("q grid must contain at least two points.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if bounds.shape[0] != nk or bounds.shape[1] != 2:
        raise ValueError("cell bounds shape mismatch.")
    if w_nodes.shape[0] != ncell:
        raise ValueError("quadrature node/weight mismatch.")
    if sD.shape[0] != nq - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    for local_ik in range(nloc):
        ik = kloc0 + local_ik
        if bounds[ik, 1] <= bounds[ik, 0]:
            raise ValueError("invalid empty diagonal cell.")

    dyn_arr = np.empty((nloc, nt), dtype=np.complex128)
    dyn = dyn_arr

    with nogil:
        for local_ik in range(nloc):
            ik = kloc0 + local_ik
            kp = k_grid[ik]
            lo = bounds[ik, 0]
            hi = bounds[ik, 1]
            for it in range(nt):
                dyn[local_ik, it] = 0.0

            for part in range(2):
                if part == 0:
                    a = lo
                    b = kp if kp < hi else hi
                else:
                    a = kp if kp > lo else lo
                    b = hi
                if b <= a:
                    continue

                mid = 0.5 * (a + b)
                half = 0.5 * (b - a)
                for ix in range(ncell):
                    kg = mid + half * x_nodes[ix]
                    weight = half * w_nodes[ix]
                    _primitive_coeff(
                        kg + kp, q, q2, q3, nq,
                        &plus_m, &plus_tmp, &plus_coef,
                    )
                    _primitive_coeff(
                        fabs(kg - kp), q, q2, q3, nq,
                        &minus_m, &minus_tmp, &minus_coef,
                    )
                    for it in range(nt):
                        dyn_val = (
                            plus_tmp * yD[plus_m, it]
                            + plus_coef * sD[plus_m, it]
                            + PD[plus_m, it]
                            - minus_tmp * yD[minus_m, it]
                            - minus_coef * sD[minus_m, it]
                            - PD[minus_m, it]
                        )
                        dyn[local_ik, it] += weight * dyn_val

            inv_width = 1.0 / (hi - lo)
            for it in range(nt):
                dyn[local_ik, it] *= inv_width

    return dyn_arr


def pi_kernel_geometry(
    double[::1] q_grid,
    double complex[:, ::1] kG,
    double complex[::1] k2GG_sum,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_tmp,
    double[:, ::1] minus_tmp,
    double[:, ::1] plus_coef,
    double[:, ::1] minus_coef,
    Py_ssize_t qloc0,
    double q_zero_tol,
):
    """Accumulate Pi_tau using precomputed pair primitive intervals."""
    cdef Py_ssize_t nk = kG.shape[0]
    cdef Py_ssize_t nt = kG.shape[1]
    cdef Py_ssize_t nprim = yD.shape[0]
    cdef Py_ssize_t nq = q_grid.shape[0]
    cdef Py_ssize_t nloc = plus_m.shape[0]
    cdef object Pi_arr
    cdef double complex[:, ::1] Pi
    cdef Py_ssize_t local_i
    cdef Py_ssize_t iq
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double qv
    cdef double inv_q
    cdef double pt
    cdef double mt
    cdef double pc
    cdef double mc
    cdef double complex dyn_val
    cdef double complex* outrow
    cdef double complex* Srow
    cdef double complex* yP
    cdef double complex* sP
    cdef double complex* PP
    cdef double complex* yM
    cdef double complex* sM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* ptrow
    cdef double* mtrow
    cdef double* pcrow
    cdef double* mcrow

    if nprim < 2:
        raise ValueError("primitive k grid must contain at least two points.")
    if qloc0 < 0 or qloc0 + nloc > nq:
        raise ValueError("invalid local q range.")
    if k2GG_sum.shape[0] != nt:
        raise ValueError("source shapes are inconsistent.")
    if yD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if sD.shape[0] != nprim - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nprim or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if (
        plus_m.shape[1] != nk
        or minus_m.shape[0] != nloc
        or minus_m.shape[1] != nk
        or plus_tmp.shape[0] != nloc
        or plus_tmp.shape[1] != nk
        or minus_tmp.shape[0] != nloc
        or minus_tmp.shape[1] != nk
        or plus_coef.shape[0] != nloc
        or plus_coef.shape[1] != nk
        or minus_coef.shape[0] != nloc
        or minus_coef.shape[1] != nk
    ):
        raise ValueError("pair geometry shape mismatch.")

    Pi_arr = np.zeros((nloc, nt), dtype=np.complex128)
    Pi = Pi_arr

    with nogil:
        for local_i in range(nloc):
            iq = qloc0 + local_i
            qv = q_grid[iq]
            outrow = &Pi[local_i, 0]
            if fabs(qv) <= q_zero_tol:
                for it in range(nt):
                    outrow[it] = k2GG_sum[it]
                continue

            inv_q = 1.0 / qv
            pmrow = &plus_m[local_i, 0]
            mmrow = &minus_m[local_i, 0]
            ptrow = &plus_tmp[local_i, 0]
            mtrow = &minus_tmp[local_i, 0]
            pcrow = &plus_coef[local_i, 0]
            mcrow = &minus_coef[local_i, 0]
            for ik in range(nk):
                pm = <Py_ssize_t>pmrow[ik]
                mm = <Py_ssize_t>mmrow[ik]
                pt = ptrow[ik]
                mt = mtrow[ik]
                pc = pcrow[ik]
                mc = mcrow[ik]
                Srow = &kG[ik, 0]
                yP = &yD[pm, 0]
                sP = &sD[pm, 0]
                PP = &PD[pm, 0]
                yM = &yD[mm, 0]
                sM = &sD[mm, 0]
                PM = &PD[mm, 0]
                for it in range(nt):
                    dyn_val = (
                        pt * yP[it]
                        + pc * sP[it]
                        + PP[it]
                        - mt * yM[it]
                        - mc * sM[it]
                        - PM[it]
                    ) * inv_q
                    outrow[it] += Srow[it] * dyn_val

    return Pi_arr


def pi_kernel_geometry_omp(
    double[::1] q_grid,
    double complex[:, ::1] kG,
    double complex[::1] k2GG_sum,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_tmp,
    double[:, ::1] minus_tmp,
    double[:, ::1] plus_coef,
    double[:, ::1] minus_coef,
    Py_ssize_t qloc0,
    double q_zero_tol,
    Py_ssize_t num_threads,
):
    """OpenMP row-parallel Pi_tau accumulation with precomputed geometry."""
    cdef Py_ssize_t nk = kG.shape[0]
    cdef Py_ssize_t nt = kG.shape[1]
    cdef Py_ssize_t nprim = yD.shape[0]
    cdef Py_ssize_t nq = q_grid.shape[0]
    cdef Py_ssize_t nloc = plus_m.shape[0]
    cdef object Pi_arr
    cdef double complex[:, ::1] Pi
    cdef Py_ssize_t local_i
    cdef Py_ssize_t iq
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double qv
    cdef double inv_q
    cdef double pt
    cdef double mt
    cdef double pc
    cdef double mc
    cdef double complex dyn_val
    cdef double complex* outrow
    cdef double complex* Srow
    cdef double complex* yP
    cdef double complex* sP
    cdef double complex* PP
    cdef double complex* yM
    cdef double complex* sM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* ptrow
    cdef double* mtrow
    cdef double* pcrow
    cdef double* mcrow

    if num_threads <= 1:
        return pi_kernel_geometry(
            q_grid,
            kG,
            k2GG_sum,
            yD,
            sD,
            PD,
            plus_m,
            minus_m,
            plus_tmp,
            minus_tmp,
            plus_coef,
            minus_coef,
            qloc0,
            q_zero_tol,
        )
    if nprim < 2:
        raise ValueError("primitive k grid must contain at least two points.")
    if qloc0 < 0 or qloc0 + nloc > nq:
        raise ValueError("invalid local q range.")
    if k2GG_sum.shape[0] != nt:
        raise ValueError("source shapes are inconsistent.")
    if yD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if sD.shape[0] != nprim - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nprim or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")
    if (
        plus_m.shape[1] != nk
        or minus_m.shape[0] != nloc
        or minus_m.shape[1] != nk
        or plus_tmp.shape[0] != nloc
        or plus_tmp.shape[1] != nk
        or minus_tmp.shape[0] != nloc
        or minus_tmp.shape[1] != nk
        or plus_coef.shape[0] != nloc
        or plus_coef.shape[1] != nk
        or minus_coef.shape[0] != nloc
        or minus_coef.shape[1] != nk
    ):
        raise ValueError("pair geometry shape mismatch.")

    Pi_arr = np.zeros((nloc, nt), dtype=np.complex128)
    Pi = Pi_arr

    with nogil:
        for local_i in prange(nloc, schedule="static", num_threads=num_threads):
            iq = qloc0 + local_i
            qv = q_grid[iq]
            outrow = &Pi[local_i, 0]
            if fabs(qv) <= q_zero_tol:
                for it in range(nt):
                    outrow[it] = k2GG_sum[it]
                continue

            inv_q = 1.0 / qv
            pmrow = &plus_m[local_i, 0]
            mmrow = &minus_m[local_i, 0]
            ptrow = &plus_tmp[local_i, 0]
            mtrow = &minus_tmp[local_i, 0]
            pcrow = &plus_coef[local_i, 0]
            mcrow = &minus_coef[local_i, 0]
            for ik in range(nk):
                pm = <Py_ssize_t>pmrow[ik]
                mm = <Py_ssize_t>mmrow[ik]
                pt = ptrow[ik]
                mt = mtrow[ik]
                pc = pcrow[ik]
                mc = mcrow[ik]
                Srow = &kG[ik, 0]
                yP = &yD[pm, 0]
                sP = &sD[pm, 0]
                PP = &PD[pm, 0]
                yM = &yD[mm, 0]
                sM = &sD[mm, 0]
                PM = &PD[mm, 0]
                for it in range(nt):
                    dyn_val = (
                        pt * yP[it]
                        + pc * sP[it]
                        + PP[it]
                        - mt * yM[it]
                        - mc * sM[it]
                        - PM[it]
                    ) * inv_q
                    outrow[it] += Srow[it] * dyn_val

    return Pi_arr


def pi_kernel(
    double[::1] source_k,
    double[::1] q_grid,
    double complex[:, ::1] kG,
    double complex[::1] k2GG_sum,
    double[::1] primitive_k,
    double[::1] primitive_k2,
    double[::1] primitive_k3,
    double complex[:, ::1] yD,
    double complex[:, ::1] sD,
    double complex[:, ::1] PD,
    Py_ssize_t qloc0,
    Py_ssize_t qloc1,
    double q_zero_tol,
):
    """Accumulate rank-local Pi_tau without materializing dI blocks."""
    cdef Py_ssize_t nk = source_k.shape[0]
    cdef Py_ssize_t nt = kG.shape[1]
    cdef Py_ssize_t nq_grid = primitive_k.shape[0]
    cdef Py_ssize_t nq = q_grid.shape[0]
    cdef Py_ssize_t nloc = qloc1 - qloc0
    cdef object Pi_arr
    cdef double complex[:, ::1] Pi
    cdef Py_ssize_t local_i
    cdef Py_ssize_t iq
    cdef Py_ssize_t ik
    cdef Py_ssize_t it
    cdef Py_ssize_t plus_m
    cdef Py_ssize_t minus_m
    cdef double qv
    cdef double inv_q
    cdef double plus_tmp
    cdef double minus_tmp
    cdef double plus_coef
    cdef double minus_coef
    cdef double complex dyn_val

    if nq_grid < 2:
        raise ValueError("primitive k grid must contain at least two points.")
    if qloc0 < 0 or qloc1 < qloc0 or qloc1 > nq:
        raise ValueError("invalid local q range.")
    if kG.shape[0] != nk or k2GG_sum.shape[0] != nt:
        raise ValueError("source shapes are inconsistent.")
    if yD.shape[0] != nq_grid or yD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if sD.shape[0] != nq_grid - 1 or sD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PD.shape[0] != nq_grid or PD.shape[1] != nt:
        raise ValueError("dynamic integral shape mismatch.")

    Pi_arr = np.zeros((nloc, nt), dtype=np.complex128)
    Pi = Pi_arr

    with nogil:
        for local_i in range(nloc):
            iq = qloc0 + local_i
            qv = q_grid[iq]
            if fabs(qv) <= q_zero_tol:
                for it in range(nt):
                    Pi[local_i, it] = k2GG_sum[it]
                continue

            inv_q = 1.0 / qv
            for ik in range(nk):
                _primitive_coeff(
                    source_k[ik] + qv, primitive_k, primitive_k2, primitive_k3,
                    nq_grid, &plus_m, &plus_tmp, &plus_coef,
                )
                _primitive_coeff(
                    fabs(source_k[ik] - qv), primitive_k, primitive_k2, primitive_k3,
                    nq_grid, &minus_m, &minus_tmp, &minus_coef,
                )
                for it in range(nt):
                    dyn_val = (
                        plus_tmp * yD[plus_m, it]
                        + plus_coef * sD[plus_m, it]
                        + PD[plus_m, it]
                        - minus_tmp * yD[minus_m, it]
                        - minus_coef * sD[minus_m, it]
                        - PD[minus_m, it]
                    ) * inv_q
                    Pi[local_i, it] += kG[ik, it] * dyn_val

    return Pi_arr
