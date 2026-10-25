# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython kernels for q-singular radial self-energy hot paths."""

import numpy as np
from cython.parallel cimport prange, threadid
from libc.math cimport INFINITY, fabs, log, nextafter
from openmp cimport omp_get_max_threads


QW_RADIAL_KERNEL_API_VERSION = 3


cdef struct _QWCoeff:
    Py_ssize_t m
    double dx
    double dx2half


def openmp_max_threads():
    """Return the OpenMP runtime's current maximum thread count."""
    return omp_get_max_threads()


cdef inline void _store_qw_coeff(
    double x,
    const double[::1] grid,
    Py_ssize_t n,
    double upper,
    Py_ssize_t* m,
    int[:, ::1] m_out,
    double[:, ::1] dx_out,
    double[:, ::1] dx2half_out,
    Py_ssize_t row,
    Py_ssize_t col,
) noexcept nogil:
    cdef double dx

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

    dx = x - grid[m[0]]
    m_out[row, col] = <int>m[0]
    dx_out[row, col] = dx
    dx2half_out[row, col] = 0.5 * dx * dx


cdef inline _QWCoeff _qw_coeff_value(
    double x,
    const double[::1] grid,
    Py_ssize_t n,
    double upper,
) noexcept nogil:
    cdef Py_ssize_t lo
    cdef Py_ssize_t hi
    cdef Py_ssize_t mid
    cdef Py_ssize_t m
    cdef double dx
    cdef _QWCoeff out

    if x < grid[0]:
        x = grid[0]
        m = 0
    elif x > upper:
        x = upper
        m = n - 2
    else:
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

    dx = x - grid[m]
    out.m = m
    out.dx = dx
    out.dx2half = 0.5 * dx * dx
    return out


cdef inline double complex _qw_primitive_sta_from_coeff(
    Py_ssize_t m,
    double dx,
    double dx2half,
    double complex[::1] h,
    double complex[::1] sh,
    double complex[::1] P,
) noexcept nogil:
    return P[m] + dx * h[m] + dx2half * sh[m]


cdef inline double complex _qw_regular_dyn_value(
    Py_ssize_t it,
    double pdx,
    double mdx,
    double pdx2,
    double mdx2,
    double complex* hP,
    double complex* shP,
    double complex* PP,
    double complex* hM,
    double complex* shM,
    double complex* PM,
) noexcept nogil:
    return (
        PP[it] + pdx * hP[it] + pdx2 * shP[it]
        - PM[it] - mdx * hM[it] - mdx2 * shM[it]
    )


cdef inline double complex _qw_block_sta_value(
    Py_ssize_t pm,
    Py_ssize_t mm,
    double pdx,
    double mdx,
    double pdx2,
    double mdx2,
    double logv,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex singular_sta,
    bint has_sta_sing,
) noexcept nogil:
    cdef double complex sta_val = (
        _qw_primitive_sta_from_coeff(pm, pdx, pdx2, hS, shS, PhS)
        - _qw_primitive_sta_from_coeff(mm, mdx, mdx2, hS, shS, PhS)
    )
    if has_sta_sing:
        sta_val += logv * singular_sta
    return sta_val


cdef inline double complex _qw_block_dyn_value(
    Py_ssize_t it,
    double pdx,
    double mdx,
    double pdx2,
    double mdx2,
    double logv,
    double complex* hP,
    double complex* shP,
    double complex* PP,
    double complex* hM,
    double complex* shM,
    double complex* PM,
    double complex[::1] singular_dyn,
    bint has_dyn_sing,
) noexcept nogil:
    cdef double complex dyn_val = _qw_regular_dyn_value(
        it, pdx, mdx, pdx2, mdx2, hP, shP, PP, hM, shM, PM,
    )
    if has_dyn_sing:
        dyn_val += logv * singular_dyn[it]
    return dyn_val


cdef inline void _qw_accumulate_dyn_one(
    Py_ssize_t nt,
    double pdx,
    double mdx,
    double pdx2,
    double mdx2,
    double logv,
    double complex* hP,
    double complex* shP,
    double complex* PP,
    double complex* hM,
    double complex* shM,
    double complex* PM,
    double complex[::1] singular_dyn,
    bint has_dyn_sing,
    double complex* Srow,
    double complex* outrow,
) noexcept nogil:
    cdef Py_ssize_t it
    cdef double complex dyn_val
    for it in range(nt):
        dyn_val = _qw_block_dyn_value(
            it, pdx, mdx, pdx2, mdx2, logv,
            hP, shP, PP, hM, shM, PM, singular_dyn, has_dyn_sing,
        )
        outrow[it] += Srow[it] * dyn_val


cdef inline void _qw_accumulate_dyn_symmetric(
    Py_ssize_t nt,
    double pdx,
    double mdx,
    double pdx2,
    double mdx2,
    double logv,
    double complex* hP,
    double complex* shP,
    double complex* PP,
    double complex* hM,
    double complex* shM,
    double complex* PM,
    double complex[::1] singular_dyn,
    bint has_dyn_sing,
    double complex* Srowj,
    double complex* Srowk,
    double complex* outj,
    double complex* outk,
) noexcept nogil:
    cdef Py_ssize_t it
    cdef double complex dyn_val
    for it in range(nt):
        dyn_val = _qw_block_dyn_value(
            it, pdx, mdx, pdx2, mdx2, logv,
            hP, shP, PP, hM, shM, PM, singular_dyn, has_dyn_sing,
        )
        outj[it] += Srowk[it] * dyn_val
        outk[it] += Srowj[it] * dyn_val


cdef int _validate_qw_sigma_inputs(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex[::1] singular_dyn,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
) except -1:
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
    cdef Py_ssize_t nq = hD.shape[0]

    if weighted_S_beta.shape[0] != nk:
        raise ValueError("weighted source shapes are inconsistent.")
    if shD.shape[0] != nq - 1 or shD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PhD.shape[0] != nq or PhD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if hS.shape[0] != nq or shS.shape[0] != nq - 1 or PhS.shape[0] != nq:
        raise ValueError("static primitive shape mismatch.")
    if singular_dyn.shape[0] != nt:
        raise ValueError("dynamic singular coefficient shape mismatch.")
    if use_diag:
        if dyn_avg.shape[0] != nk or dyn_avg.shape[1] != nt:
            raise ValueError("diagonal dynamic average shape mismatch.")
        if sta_avg.shape[0] != nk:
            raise ValueError("diagonal static average shape mismatch.")
    return 0


cdef int _validate_qw_pair_geometry_shape(
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[:, ::1] log_interval,
    Py_ssize_t nrows,
    Py_ssize_t nk,
    object message,
) except -1:
    if (
        plus_m.shape[0] != nrows
        or plus_m.shape[1] != nk
        or minus_m.shape[0] != nrows
        or minus_m.shape[1] != nk
        or plus_dx.shape[0] != nrows
        or plus_dx.shape[1] != nk
        or minus_dx.shape[0] != nrows
        or minus_dx.shape[1] != nk
        or plus_dx2half.shape[0] != nrows
        or plus_dx2half.shape[1] != nk
        or minus_dx2half.shape[0] != nrows
        or minus_dx2half.shape[1] != nk
        or log_interval.shape[0] != nrows
        or log_interval.shape[1] != nk
    ):
        raise ValueError(message)
    return 0


cdef inline double _safe_z_log_abs_minus_z(double z) noexcept nogil:
    if z == 0.0:
        return 0.0
    return z * log(fabs(z)) - z


cdef inline double _singular_cell_average_factor(
    double p,
    double lo,
    double hi,
) noexcept nogil:
    return (
        _safe_z_log_abs_minus_z(hi + p)
        - _safe_z_log_abs_minus_z(lo + p)
        - _safe_z_log_abs_minus_z(hi - p)
        + _safe_z_log_abs_minus_z(lo - p)
    ) / (hi - lo)


cdef int _validate_qw_diag_geometry_shape(
    int[::1] point_count,
    double[:, ::1] weight,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[::1] singular_factor,
    Py_ssize_t nk,
) except -1:
    cdef Py_ssize_t max_points = weight.shape[1]
    cdef Py_ssize_t j
    if (
        point_count.shape[0] != nk
        or weight.shape[0] != nk
        or plus_m.shape[0] != nk
        or plus_m.shape[1] != max_points
        or minus_m.shape[0] != nk
        or minus_m.shape[1] != max_points
        or plus_dx.shape[0] != nk
        or plus_dx.shape[1] != max_points
        or minus_dx.shape[0] != nk
        or minus_dx.shape[1] != max_points
        or plus_dx2half.shape[0] != nk
        or plus_dx2half.shape[1] != max_points
        or minus_dx2half.shape[0] != nk
        or minus_dx2half.shape[1] != max_points
        or singular_factor.shape[0] != nk
    ):
        raise ValueError("qW diagonal-cell geometry shape mismatch.")
    for j in range(nk):
        if point_count[j] < 0 or point_count[j] > max_points:
            raise ValueError("qW diagonal-cell point count is invalid.")
    return 0


def build_qw_diag_cell_geometry(
    const double[::1] k_grid,
    const double[:, ::1] bounds,
    const double[::1] x_nodes,
    const double[::1] w_nodes,
    const double[::1] qh,
    Py_ssize_t num_threads=1,
):
    """Precompute qW diagonal-cell quadrature geometry.

    The returned arrays encode the current Gauss cell-average definition used by
    ``QWPrecomp.cell_average_*``: each cell is split at ``k = kp`` before
    quadrature, regular primitives use clipped qW primitive coordinates, and
    the coefficient-free singular diagonal factor is stored separately.
    """
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nq = qh.shape[0]
    cdef Py_ssize_t ncell = x_nodes.shape[0]
    cdef Py_ssize_t max_points = 2 * ncell
    cdef object point_count_arr
    cdef object weight_arr
    cdef object plus_m_arr
    cdef object minus_m_arr
    cdef object plus_dx_arr
    cdef object minus_dx_arr
    cdef object plus_dx2half_arr
    cdef object minus_dx2half_arr
    cdef object singular_factor_arr
    cdef int[::1] point_count
    cdef double[:, ::1] weight
    cdef int[:, ::1] plus_m
    cdef int[:, ::1] minus_m
    cdef double[:, ::1] plus_dx
    cdef double[:, ::1] minus_dx
    cdef double[:, ::1] plus_dx2half
    cdef double[:, ::1] minus_dx2half
    cdef double[::1] singular_factor
    cdef Py_ssize_t j
    cdef Py_ssize_t ix
    cdef Py_ssize_t part
    cdef Py_ssize_t ip
    cdef Py_ssize_t m_plus
    cdef Py_ssize_t m_minus
    cdef double kp
    cdef double lo
    cdef double hi
    cdef double a
    cdef double b
    cdef double mid
    cdef double half
    cdef double inv_width
    cdef double kg
    cdef double upper

    if nq < 2:
        raise ValueError("qW primitive grid must contain at least two points.")
    if bounds.shape[0] != nk or bounds.shape[1] != 2:
        raise ValueError("diagonal cell bounds shape mismatch.")
    if w_nodes.shape[0] != ncell:
        raise ValueError("quadrature node/weight mismatch.")
    if ncell < 1:
        raise ValueError("ncell must be at least 1.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    for j in range(nk):
        if bounds[j, 1] <= bounds[j, 0]:
            raise ValueError("invalid empty diagonal cell.")

    point_count_arr = np.empty(nk, dtype=np.intc)
    weight_arr = np.empty((nk, max_points), dtype=np.float64)
    plus_m_arr = np.empty((nk, max_points), dtype=np.intc)
    minus_m_arr = np.empty((nk, max_points), dtype=np.intc)
    plus_dx_arr = np.empty((nk, max_points), dtype=np.float64)
    minus_dx_arr = np.empty((nk, max_points), dtype=np.float64)
    plus_dx2half_arr = np.empty((nk, max_points), dtype=np.float64)
    minus_dx2half_arr = np.empty((nk, max_points), dtype=np.float64)
    singular_factor_arr = np.empty(nk, dtype=np.float64)
    point_count = point_count_arr
    weight = weight_arr
    plus_m = plus_m_arr
    minus_m = minus_m_arr
    plus_dx = plus_dx_arr
    minus_dx = minus_dx_arr
    plus_dx2half = plus_dx2half_arr
    minus_dx2half = minus_dx2half_arr
    singular_factor = singular_factor_arr
    upper = nextafter(qh[nq - 1], -INFINITY)

    with nogil:
        for j in prange(
            nk,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            kp = k_grid[j]
            lo = bounds[j, 0]
            hi = bounds[j, 1]
            inv_width = 1.0 / (hi - lo)
            ip = 0
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
                m_plus = 0
                m_minus = 0
                for ix in range(ncell):
                    kg = mid + half * x_nodes[ix]
                    weight[j, ip] = half * w_nodes[ix] * inv_width
                    _store_qw_coeff(
                        kg + kp, qh, nq, upper, &m_plus,
                        plus_m, plus_dx, plus_dx2half, j, ip,
                    )
                    _store_qw_coeff(
                        fabs(kg - kp), qh, nq, upper, &m_minus,
                        minus_m, minus_dx, minus_dx2half, j, ip,
                    )
                    ip = ip + 1
            point_count[j] = <int>ip
            singular_factor[j] = _singular_cell_average_factor(kp, lo, hi)

    return (
        point_count_arr,
        weight_arr,
        plus_m_arr,
        minus_m_arr,
        plus_dx_arr,
        minus_dx_arr,
        plus_dx2half_arr,
        minus_dx2half_arr,
        singular_factor_arr,
    )


def qw_diag_cells_dyn_kernel(
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] singular_dyn,
    bint has_dyn_sing,
    int[::1] point_count,
    double[:, ::1] weight,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[::1] singular_factor,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    int num_threads=1,
):
    """Evaluate dynamic qW diagonal-cell averages from cached geometry."""
    cdef Py_ssize_t nk = hD.shape[0]
    cdef Py_ssize_t nt = hD.shape[1]
    cdef Py_ssize_t nrows = point_count.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object dyn_arr
    cdef double complex[:, ::1] dyn
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ip
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double wt
    cdef double pdx
    cdef double mdx
    cdef double pdx2
    cdef double mdx2
    cdef double factor
    cdef double complex* hP
    cdef double complex* shP
    cdef double complex* PP
    cdef double complex* hM
    cdef double complex* shM
    cdef double complex* PM

    if shD.shape[0] != nk - 1 or shD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PhD.shape[0] != nk or PhD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if singular_dyn.shape[0] != nt:
        raise ValueError("dynamic singular coefficient shape mismatch.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nrows:
        raise ValueError("invalid local k range.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    _validate_qw_diag_geometry_shape(
        point_count, weight, plus_m, minus_m, plus_dx, minus_dx,
        plus_dx2half, minus_dx2half, singular_factor, nrows,
    )

    dyn_arr = np.zeros((nloc, nt), dtype=np.complex128)
    dyn = dyn_arr

    with nogil:
        for local_j in prange(nloc, schedule="static", num_threads=num_threads):
            j = kloc0 + local_j
            for ip in range(point_count[j]):
                pm = <Py_ssize_t>plus_m[j, ip]
                mm = <Py_ssize_t>minus_m[j, ip]
                wt = weight[j, ip]
                pdx = plus_dx[j, ip]
                mdx = minus_dx[j, ip]
                pdx2 = plus_dx2half[j, ip]
                mdx2 = minus_dx2half[j, ip]
                hP = &hD[pm, 0]
                shP = &shD[pm, 0]
                PP = &PhD[pm, 0]
                hM = &hD[mm, 0]
                shM = &shD[mm, 0]
                PM = &PhD[mm, 0]
                for it in range(nt):
                    dyn[local_j, it] += wt * _qw_regular_dyn_value(
                        it, pdx, mdx, pdx2, mdx2,
                        hP, shP, PP, hM, shM, PM,
                    )
            if has_dyn_sing:
                factor = singular_factor[j]
                for it in range(nt):
                    dyn[local_j, it] += factor * singular_dyn[it]

    return dyn_arr


def qw_diag_cells_kernel(
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex[::1] singular_dyn,
    double complex singular_sta,
    bint has_dyn_sing,
    bint has_sta_sing,
    int[::1] point_count,
    double[:, ::1] weight,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[::1] singular_factor,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    int num_threads=1,
):
    """Evaluate dynamic and static qW diagonal-cell averages from geometry."""
    cdef Py_ssize_t nk = hD.shape[0]
    cdef Py_ssize_t nt = hD.shape[1]
    cdef Py_ssize_t nrows = point_count.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object dyn_arr
    cdef object sta_arr
    cdef double complex[:, ::1] dyn
    cdef double complex[::1] sta
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ip
    cdef Py_ssize_t it
    cdef Py_ssize_t pm
    cdef Py_ssize_t mm
    cdef double wt
    cdef double pdx
    cdef double mdx
    cdef double pdx2
    cdef double mdx2
    cdef double factor
    cdef double complex sta_val
    cdef double complex* hP
    cdef double complex* shP
    cdef double complex* PP
    cdef double complex* hM
    cdef double complex* shM
    cdef double complex* PM

    if shD.shape[0] != nk - 1 or shD.shape[1] != nt:
        raise ValueError("dynamic slope shape mismatch.")
    if PhD.shape[0] != nk or PhD.shape[1] != nt:
        raise ValueError("dynamic primitive shape mismatch.")
    if hS.shape[0] != nk or shS.shape[0] != nk - 1 or PhS.shape[0] != nk:
        raise ValueError("static primitive shape mismatch.")
    if singular_dyn.shape[0] != nt:
        raise ValueError("dynamic singular coefficient shape mismatch.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nrows:
        raise ValueError("invalid local k range.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    _validate_qw_diag_geometry_shape(
        point_count, weight, plus_m, minus_m, plus_dx, minus_dx,
        plus_dx2half, minus_dx2half, singular_factor, nrows,
    )

    dyn_arr = np.zeros((nloc, nt), dtype=np.complex128)
    sta_arr = np.zeros(nloc, dtype=np.complex128)
    dyn = dyn_arr
    sta = sta_arr

    with nogil:
        for local_j in prange(nloc, schedule="static", num_threads=num_threads):
            j = kloc0 + local_j
            for ip in range(point_count[j]):
                pm = <Py_ssize_t>plus_m[j, ip]
                mm = <Py_ssize_t>minus_m[j, ip]
                wt = weight[j, ip]
                pdx = plus_dx[j, ip]
                mdx = minus_dx[j, ip]
                pdx2 = plus_dx2half[j, ip]
                mdx2 = minus_dx2half[j, ip]
                sta_val = (
                    _qw_primitive_sta_from_coeff(pm, pdx, pdx2, hS, shS, PhS)
                    - _qw_primitive_sta_from_coeff(mm, mdx, mdx2, hS, shS, PhS)
                )
                sta[local_j] += wt * sta_val
                hP = &hD[pm, 0]
                shP = &shD[pm, 0]
                PP = &PhD[pm, 0]
                hM = &hD[mm, 0]
                shM = &shD[mm, 0]
                PM = &PhD[mm, 0]
                for it in range(nt):
                    dyn[local_j, it] += wt * _qw_regular_dyn_value(
                        it, pdx, mdx, pdx2, mdx2,
                        hP, shP, PP, hM, shM, PM,
                    )
            factor = singular_factor[j]
            if has_dyn_sing:
                for it in range(nt):
                    dyn[local_j, it] += factor * singular_dyn[it]
            if has_sta_sing:
                sta[local_j] += factor * singular_sta

    return dyn_arr, sta_arr


def build_qw_sigma_pair_geometry(
    double[::1] k_grid,
    double[::1] qh,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    Py_ssize_t num_threads=1,
):
    """Precompute qW pair geometry used by compiled sigma kernels.

    For each target row ``kp`` and source ``k``, the returned arrays store
    primitive segment indices and offsets for ``q_plus = kp + k`` and
    ``q_minus = abs(kp - k)``.  ``log_interval`` stores the off-diagonal
    singular primitive ``log(q_plus) - log(q_minus)``; diagonal entries are set
    to zero because singular self-cells are supplied separately by
    ``dyn_avg``/``sta_avg`` when ``use_diag`` is true.
    """
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nq = qh.shape[0]
    cdef Py_ssize_t nloc = kloc1 - kloc0
    cdef object plus_m_arr
    cdef object minus_m_arr
    cdef object plus_dx_arr
    cdef object minus_dx_arr
    cdef object plus_dx2half_arr
    cdef object minus_dx2half_arr
    cdef object log_interval_arr
    cdef int[:, ::1] plus_m
    cdef int[:, ::1] minus_m
    cdef double[:, ::1] plus_dx
    cdef double[:, ::1] minus_dx
    cdef double[:, ::1] plus_dx2half
    cdef double[:, ::1] minus_dx2half
    cdef double[:, ::1] log_interval
    cdef Py_ssize_t local_j
    cdef Py_ssize_t j
    cdef Py_ssize_t ik
    cdef Py_ssize_t m
    cdef double kp
    cdef double x
    cdef double upper

    if nq < 2:
        raise ValueError("qW primitive grid must contain at least two points.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")

    plus_m_arr = np.empty((nloc, nk), dtype=np.intc)
    minus_m_arr = np.empty((nloc, nk), dtype=np.intc)
    plus_dx_arr = np.empty((nloc, nk), dtype=np.float64)
    minus_dx_arr = np.empty((nloc, nk), dtype=np.float64)
    plus_dx2half_arr = np.empty((nloc, nk), dtype=np.float64)
    minus_dx2half_arr = np.empty((nloc, nk), dtype=np.float64)
    log_interval_arr = np.empty((nloc, nk), dtype=np.float64)
    plus_m = plus_m_arr
    minus_m = minus_m_arr
    plus_dx = plus_dx_arr
    minus_dx = minus_dx_arr
    plus_dx2half = plus_dx2half_arr
    minus_dx2half = minus_dx2half_arr
    log_interval = log_interval_arr
    upper = nextafter(qh[nq - 1], -INFINITY)

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
                x = k_grid[ik] + kp
                _store_qw_coeff(
                    x, qh, nq, upper, &m,
                    plus_m, plus_dx, plus_dx2half, local_j, ik,
                )
                log_interval[local_j, ik] = log(x) if x > 0.0 else 0.0

            m = 0
            for ik in range(j, -1, -1):
                x = kp - k_grid[ik]
                _store_qw_coeff(
                    x, qh, nq, upper, &m,
                    minus_m, minus_dx, minus_dx2half, local_j, ik,
                )
                if x > 0.0:
                    log_interval[local_j, ik] -= log(x)
                else:
                    log_interval[local_j, ik] = 0.0

            m = 0
            for ik in range(j + 1, nk):
                x = k_grid[ik] - kp
                _store_qw_coeff(
                    x, qh, nq, upper, &m,
                    minus_m, minus_dx, minus_dx2half, local_j, ik,
                )
                if x > 0.0:
                    log_interval[local_j, ik] -= log(x)
                else:
                    log_interval[local_j, ik] = 0.0

            log_interval[local_j, j] = 0.0

    return (
        plus_m_arr,
        minus_m_arr,
        plus_dx_arr,
        minus_dx_arr,
        plus_dx2half_arr,
        minus_dx2half_arr,
        log_interval_arr,
    )


def qw_sigma_kernel_geometry(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex[::1] singular_dyn,
    double complex singular_sta,
    bint has_dyn_sing,
    bint has_sta_sing,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[:, ::1] log_interval,
    Py_ssize_t kloc0,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    bint skip_static=False,
    Py_ssize_t num_threads=1,
):
    """Accumulate untransformed qW Sigma for rank-local target rows.

    ``weighted_S_t`` and ``weighted_S_beta`` already include radial weights and
    source scaling.  The returned ``Sigma_t`` and ``Sigma0`` are still before
    the Matsubara transform, static addition, and global prefactor applied by
    ``radial.py``.  When ``use_diag`` is true, the ``j == k`` block is replaced
    by the supplied diagonal cell averages.
    """
    cdef Py_ssize_t nloc = plus_m.shape[0]
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
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
    cdef double pdx
    cdef double mdx
    cdef double pdx2
    cdef double mdx2
    cdef double logv
    cdef double complex sta_val = 0.0
    cdef double complex* outrow
    cdef double complex* Srow
    cdef double complex* diagrow
    cdef double complex* hP
    cdef double complex* shP
    cdef double complex* PP
    cdef double complex* hM
    cdef double complex* shM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* pdxrow
    cdef double* mdxrow
    cdef double* pdx2row
    cdef double* mdx2row
    cdef double* logrow

    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    _validate_qw_sigma_inputs(
        weighted_S_t, weighted_S_beta, hD, shD, PhD, hS, shS, PhS,
        singular_dyn, use_diag, dyn_avg, sta_avg,
    )
    _validate_qw_pair_geometry_shape(
        plus_m, minus_m, plus_dx, minus_dx, plus_dx2half, minus_dx2half,
        log_interval, nloc, nk, "qW pair geometry shape mismatch.",
    )

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
            pdxrow = &plus_dx[local_j, 0]
            mdxrow = &minus_dx[local_j, 0]
            pdx2row = &plus_dx2half[local_j, 0]
            mdx2row = &minus_dx2half[local_j, 0]
            logrow = &log_interval[local_j, 0]
            for ik in range(nk):
                Srow = &weighted_S_t[ik, 0]
                if use_diag and ik == j:
                    diagrow = &dyn_avg[j, 0]
                    for it in range(nt):
                        outrow[it] += Srow[it] * diagrow[it]
                    if not skip_static:
                        sta_val = sta_avg[j]
                        Sigma0[local_j] += weighted_S_beta[ik] * sta_val
                    continue

                pm = <Py_ssize_t>pmrow[ik]
                mm = <Py_ssize_t>mmrow[ik]
                pdx = pdxrow[ik]
                mdx = mdxrow[ik]
                pdx2 = pdx2row[ik]
                mdx2 = mdx2row[ik]
                logv = logrow[ik]
                hP = &hD[pm, 0]
                shP = &shD[pm, 0]
                PP = &PhD[pm, 0]
                hM = &hD[mm, 0]
                shM = &shD[mm, 0]
                PM = &PhD[mm, 0]
                _qw_accumulate_dyn_one(
                    nt, pdx, mdx, pdx2, mdx2, logv,
                    hP, shP, PP, hM, shM, PM, singular_dyn, has_dyn_sing,
                    Srow, outrow,
                )
                if not skip_static:
                    sta_val = _qw_block_sta_value(
                        pm, mm, pdx, mdx, pdx2, mdx2, logv,
                        hS, shS, PhS, singular_sta, has_sta_sing,
                    )
                    Sigma0[local_j] += weighted_S_beta[ik] * sta_val

    return Sigma_t_arr, Sigma0_arr


def qw_sigma_kernel(
    double[::1] k_grid,
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double[::1] qh,
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex[::1] singular_dyn,
    double complex singular_sta,
    bint has_dyn_sing,
    bint has_sta_sing,
    Py_ssize_t kloc0,
    Py_ssize_t kloc1,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    bint skip_static=False,
    Py_ssize_t num_threads=1,
):
    """Accumulate rank-local qW Sigma without materializing pair geometry."""
    cdef Py_ssize_t nk = k_grid.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
    cdef Py_ssize_t nq = qh.shape[0]
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
    cdef double q_plus
    cdef double q_minus
    cdef double logv
    cdef double upper
    cdef double complex sta_val = 0.0
    cdef _QWCoeff plus_coeff
    cdef _QWCoeff minus_coeff
    cdef double complex* outrow
    cdef double complex* Srow
    cdef double complex* diagrow
    cdef double complex* hP
    cdef double complex* shP
    cdef double complex* PP
    cdef double complex* hM
    cdef double complex* shM
    cdef double complex* PM

    if nq < 2:
        raise ValueError("qW primitive grid must contain at least two points.")
    if kloc0 < 0 or kloc1 < kloc0 or kloc1 > nk:
        raise ValueError("invalid local k range.")
    if weighted_S_t.shape[0] != nk or weighted_S_beta.shape[0] != nk:
        raise ValueError("weighted source shapes are inconsistent with k_grid.")
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    _validate_qw_sigma_inputs(
        weighted_S_t, weighted_S_beta, hD, shD, PhD, hS, shS, PhS,
        singular_dyn, use_diag, dyn_avg, sta_avg,
    )
    if qh.shape[0] != hD.shape[0] or hS.shape[0] != nq:
        raise ValueError("qW primitive grid shape mismatch.")

    Sigma_t_arr = np.zeros((nloc, nt), dtype=np.complex128)
    Sigma0_arr = np.zeros(nloc, dtype=np.complex128)
    Sigma_t = Sigma_t_arr
    Sigma0 = Sigma0_arr
    upper = nextafter(qh[nq - 1], -INFINITY)

    with nogil:
        for local_j in prange(
            nloc,
            schedule="static",
            num_threads=num_threads,
            use_threads_if=num_threads > 1,
        ):
            j = kloc0 + local_j
            kp = k_grid[j]
            outrow = &Sigma_t[local_j, 0]
            for ik in range(nk):
                Srow = &weighted_S_t[ik, 0]
                if use_diag and ik == j:
                    diagrow = &dyn_avg[j, 0]
                    for it in range(nt):
                        outrow[it] += Srow[it] * diagrow[it]
                    if not skip_static:
                        sta_val = sta_avg[j]
                        Sigma0[local_j] += weighted_S_beta[ik] * sta_val
                    continue

                q_plus = k_grid[ik] + kp
                q_minus = fabs(k_grid[ik] - kp)
                plus_coeff = _qw_coeff_value(q_plus, qh, nq, upper)
                minus_coeff = _qw_coeff_value(q_minus, qh, nq, upper)
                if q_plus > 0.0:
                    logv = log(q_plus)
                else:
                    logv = 0.0
                if q_minus > 0.0:
                    logv -= log(q_minus)
                else:
                    logv = 0.0
                hP = &hD[plus_coeff.m, 0]
                shP = &shD[plus_coeff.m, 0]
                PP = &PhD[plus_coeff.m, 0]
                hM = &hD[minus_coeff.m, 0]
                shM = &shD[minus_coeff.m, 0]
                PM = &PhD[minus_coeff.m, 0]
                _qw_accumulate_dyn_one(
                    nt,
                    plus_coeff.dx, minus_coeff.dx,
                    plus_coeff.dx2half, minus_coeff.dx2half,
                    logv,
                    hP, shP, PP, hM, shM, PM, singular_dyn, has_dyn_sing,
                    Srow, outrow,
                )
                if not skip_static:
                    sta_val = _qw_block_sta_value(
                        plus_coeff.m, minus_coeff.m,
                        plus_coeff.dx, minus_coeff.dx,
                        plus_coeff.dx2half, minus_coeff.dx2half, logv,
                        hS, shS, PhS, singular_sta, has_sta_sing,
                    )
                    Sigma0[local_j] += weighted_S_beta[ik] * sta_val

    return Sigma_t_arr, Sigma0_arr


def qw_sigma_kernel_geometry_triangular(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex[::1] singular_dyn,
    double complex singular_sta,
    bint has_dyn_sing,
    bint has_sta_sing,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[:, ::1] log_interval,
    Py_ssize_t rank,
    Py_ssize_t nranks,
    Py_ssize_t tile_size,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    bint skip_static=False,
):
    """Accumulate untransformed qW Sigma over upper-triangular pair tiles.

    This kernel uses the symmetry of the radial pair block: off-diagonal pairs
    update both target rows, while diagonal pairs update only one row.  The
    returned arrays contain this MPI rank's contribution to the full k grid;
    the Python caller performs the allreduce before final transforms.
    """
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
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
    cdef double pdx
    cdef double mdx
    cdef double pdx2
    cdef double mdx2
    cdef double logv
    cdef double complex sta_val = 0.0
    cdef double complex* outj
    cdef double complex* outk
    cdef double complex* Srowj
    cdef double complex* Srowk
    cdef double complex* diagrow
    cdef double complex* hP
    cdef double complex* shP
    cdef double complex* PP
    cdef double complex* hM
    cdef double complex* shM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* pdxrow
    cdef double* mdxrow
    cdef double* pdx2row
    cdef double* mdx2row
    cdef double* logrow

    if nranks < 1:
        raise ValueError("nranks must be positive.")
    if rank < 0 or rank >= nranks:
        raise ValueError("invalid rank for triangular qW sigma kernel.")
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    _validate_qw_sigma_inputs(
        weighted_S_t, weighted_S_beta, hD, shD, PhD, hS, shS, PhS,
        singular_dyn, use_diag, dyn_avg, sta_avg,
    )
    _validate_qw_pair_geometry_shape(
        plus_m, minus_m, plus_dx, minus_dx, plus_dx2half, minus_dx2half,
        log_interval, nk, nk, "full qW pair geometry shape mismatch.",
    )

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
                    pdxrow = &plus_dx[j, 0]
                    mdxrow = &minus_dx[j, 0]
                    pdx2row = &plus_dx2half[j, 0]
                    mdx2row = &minus_dx2half[j, 0]
                    logrow = &log_interval[j, 0]
                    outj = &Sigma_t[j, 0]
                    Srowj = &weighted_S_t[j, 0]
                    k0 = kb
                    if kb == jb:
                        k0 = j
                    for k in range(k0, kend):
                        Srowk = &weighted_S_t[k, 0]
                        if use_diag and k == j:
                            diagrow = &dyn_avg[j, 0]
                            for it in range(nt):
                                outj[it] += Srowk[it] * diagrow[it]
                            if not skip_static:
                                sta_val = sta_avg[j]
                                Sigma0[j] += weighted_S_beta[k] * sta_val
                            continue

                        pm = <Py_ssize_t>pmrow[k]
                        mm = <Py_ssize_t>mmrow[k]
                        pdx = pdxrow[k]
                        mdx = mdxrow[k]
                        pdx2 = pdx2row[k]
                        mdx2 = mdx2row[k]
                        logv = logrow[k]
                        hP = &hD[pm, 0]
                        shP = &shD[pm, 0]
                        PP = &PhD[pm, 0]
                        hM = &hD[mm, 0]
                        shM = &shD[mm, 0]
                        PM = &PhD[mm, 0]
                        if not skip_static:
                            sta_val = _qw_block_sta_value(
                                pm, mm, pdx, mdx, pdx2, mdx2, logv,
                                hS, shS, PhS, singular_sta, has_sta_sing,
                            )
                            Sigma0[j] += weighted_S_beta[k] * sta_val

                        if k != j:
                            outk = &Sigma_t[k, 0]
                            _qw_accumulate_dyn_symmetric(
                                nt, pdx, mdx, pdx2, mdx2, logv,
                                hP, shP, PP, hM, shM, PM,
                                singular_dyn, has_dyn_sing,
                                Srowj, Srowk, outj, outk,
                            )
                            if not skip_static:
                                Sigma0[k] += weighted_S_beta[j] * sta_val
                        else:
                            _qw_accumulate_dyn_one(
                                nt, pdx, mdx, pdx2, mdx2, logv,
                                hP, shP, PP, hM, shM, PM,
                                singular_dyn, has_dyn_sing,
                                Srowk, outj,
                            )

                tile_id += 1
                kb += tile_size
            jb += tile_size

    return Sigma_t_arr, Sigma0_arr


def qw_sigma_kernel_geometry_triangular_omp(
    double complex[:, ::1] weighted_S_t,
    double complex[::1] weighted_S_beta,
    double complex[:, ::1] hD,
    double complex[:, ::1] shD,
    double complex[:, ::1] PhD,
    double complex[::1] hS,
    double complex[::1] shS,
    double complex[::1] PhS,
    double complex[::1] singular_dyn,
    double complex singular_sta,
    bint has_dyn_sing,
    bint has_sta_sing,
    int[:, ::1] plus_m,
    int[:, ::1] minus_m,
    double[:, ::1] plus_dx,
    double[:, ::1] minus_dx,
    double[:, ::1] plus_dx2half,
    double[:, ::1] minus_dx2half,
    double[:, ::1] log_interval,
    Py_ssize_t rank,
    Py_ssize_t nranks,
    Py_ssize_t tile_size,
    bint use_diag,
    double complex[:, ::1] dyn_avg,
    double complex[::1] sta_avg,
    bint skip_static,
    Py_ssize_t num_threads,
):
    """OpenMP triangular qW Sigma accumulation with private outputs.

    Work is split by rank-owned triangular tiles and each OpenMP thread writes
    to private full-grid ``Sigma_t``/``Sigma0`` buffers before a final reduction.
    The Python adapter chooses ``num_threads`` only after applying its private
    buffer memory guard.
    """
    cdef Py_ssize_t nk = weighted_S_t.shape[0]
    cdef Py_ssize_t nt = weighted_S_t.shape[1]
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
    cdef double pdx
    cdef double mdx
    cdef double pdx2
    cdef double mdx2
    cdef double logv
    cdef double complex sta_val = 0.0
    cdef double complex* outj
    cdef double complex* outk
    cdef double complex* Srowj
    cdef double complex* Srowk
    cdef double complex* diagrow
    cdef double complex* hP
    cdef double complex* shP
    cdef double complex* PP
    cdef double complex* hM
    cdef double complex* shM
    cdef double complex* PM
    cdef int* pmrow
    cdef int* mmrow
    cdef double* pdxrow
    cdef double* mdxrow
    cdef double* pdx2row
    cdef double* mdx2row
    cdef double* logrow

    if num_threads <= 1:
        return qw_sigma_kernel_geometry_triangular(
            weighted_S_t,
            weighted_S_beta,
            hD,
            shD,
            PhD,
            hS,
            shS,
            PhS,
            singular_dyn,
            singular_sta,
            has_dyn_sing,
            has_sta_sing,
            plus_m,
            minus_m,
            plus_dx,
            minus_dx,
            plus_dx2half,
            minus_dx2half,
            log_interval,
            rank,
            nranks,
            tile_size,
            use_diag,
            dyn_avg,
            sta_avg,
            skip_static,
        )
    if nranks < 1:
        raise ValueError("nranks must be positive.")
    if rank < 0 or rank >= nranks:
        raise ValueError("invalid rank for triangular qW sigma kernel.")
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    _validate_qw_sigma_inputs(
        weighted_S_t, weighted_S_beta, hD, shD, PhD, hS, shS, PhS,
        singular_dyn, use_diag, dyn_avg, sta_avg,
    )
    _validate_qw_pair_geometry_shape(
        plus_m, minus_m, plus_dx, minus_dx, plus_dx2half, minus_dx2half,
        log_interval, nk, nk, "full qW pair geometry shape mismatch.",
    )

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
                pdxrow = &plus_dx[j, 0]
                mdxrow = &minus_dx[j, 0]
                pdx2row = &plus_dx2half[j, 0]
                mdx2row = &minus_dx2half[j, 0]
                logrow = &log_interval[j, 0]
                outj = &Sigma_t_priv[tid, j, 0]
                Srowj = &weighted_S_t[j, 0]
                k0 = kb
                if kb == jb:
                    k0 = j
                for k in range(k0, kend):
                    Srowk = &weighted_S_t[k, 0]
                    if use_diag and k == j:
                        diagrow = &dyn_avg[j, 0]
                        for it in range(nt):
                            outj[it] += Srowk[it] * diagrow[it]
                        if not skip_static:
                            sta_val = sta_avg[j]
                            Sigma0_priv[tid, j] += weighted_S_beta[k] * sta_val
                        continue

                    pm = <Py_ssize_t>pmrow[k]
                    mm = <Py_ssize_t>mmrow[k]
                    pdx = pdxrow[k]
                    mdx = mdxrow[k]
                    pdx2 = pdx2row[k]
                    mdx2 = mdx2row[k]
                    logv = logrow[k]
                    hP = &hD[pm, 0]
                    shP = &shD[pm, 0]
                    PP = &PhD[pm, 0]
                    hM = &hD[mm, 0]
                    shM = &shD[mm, 0]
                    PM = &PhD[mm, 0]
                    if not skip_static:
                        sta_val = _qw_block_sta_value(
                            pm, mm, pdx, mdx, pdx2, mdx2, logv,
                            hS, shS, PhS, singular_sta, has_sta_sing,
                        )
                        Sigma0_priv[tid, j] += weighted_S_beta[k] * sta_val

                    if k != j:
                        outk = &Sigma_t_priv[tid, k, 0]
                        _qw_accumulate_dyn_symmetric(
                            nt, pdx, mdx, pdx2, mdx2, logv,
                            hP, shP, PP, hM, shM, PM,
                            singular_dyn, has_dyn_sing,
                            Srowj, Srowk, outj, outk,
                        )
                        if not skip_static:
                            Sigma0_priv[tid, k] += weighted_S_beta[j] * sta_val
                    else:
                        _qw_accumulate_dyn_one(
                            nt, pdx, mdx, pdx2, mdx2, logv,
                            hP, shP, PP, hM, shM, PM,
                            singular_dyn, has_dyn_sing,
                            Srowk, outj,
                        )

        for j in prange(nk, schedule="static", num_threads=num_threads):
            outj = &Sigma_t[j, 0]
            for tid in range(num_threads):
                Sigma0[j] += Sigma0_priv[tid, j]
                Srowj = &Sigma_t_priv[tid, j, 0]
                for it in range(nt):
                    outj[it] += Srowj[it]

    return Sigma_t_arr, Sigma0_arr
