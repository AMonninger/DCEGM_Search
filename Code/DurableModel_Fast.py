"""
Simplified durable–nondurable consumption-saving model.

Port of the one-period notebook suite into a self-contained class API:

    from DurableModel import DurableConsumer
    # Search + stochastic prices (Combined notebook):
    model = DurableConsumer(cycles=1, SearchCost=0.1, sigma_eps=0.0, PriceShkStd=0.1)
    # EV Type I taste only (TasteShock notebook; kappa=0 is the symmetric logit):
    model = DurableConsumer(cycles=1, SearchCost=0.0, sigma_eps=0.05, PriceShkStd=0.0)
    # Zorzi–Beraja taste: same logit with a utility cost kappa of adjusting:
    model = DurableConsumer(cycles=1, SearchCost=0.0, sigma_eps=0.05, kappa=0.1, PriceShkStd=0.0)
    model.solve()
    sol = model.solution[0]   # first period; solution[-1] is terminal
    sol.cFunc(n, m), sol.dFunc(n, m), sol.adjusting(n, m)

Channels (turn off with zeros):
  - Directed search: SearchCost (mutually exclusive with EV taste);
    search_method="closed" (default, conjugate ε*) or "grid" (101-point argmax)
  - EV Type I taste shocks: sigma_eps (closed-form logit / log-sum; no xi grid).
    kappa is an optional utility cost of adjusting; kappa=0 keeps the symmetric logit.
  - Stochastic prices: PriceShkStd

Supports the four pedagogical cases: taste only, search only, price only, search+price.
Requires LTV=0, ScaleBool=False, MaintCost=OpCost=0 (simplified).
"""
from __future__ import annotations

from copy import deepcopy
import os
import sys

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from numba import njit, prange

_cuda_lib = os.path.join(sys.prefix, "Library")
if os.path.isfile(os.path.join(_cuda_lib, "bin", "cudart64_12.dll")):
    os.environ.setdefault("CUDA_PATH", _cuda_lib)

try:
    import cupy as _cp
    _GPU_AVAILABLE = bool(_cp.cuda.is_available())
except Exception:
    _cp = None
    _GPU_AVAILABLE = False

# Numba prange FOC is faster than CuPy at Combined-test grids (~400 x 800).
# Set DCEGM_GPU_FOC=1 to force the GPU FOC path.
_USE_GPU_FOC = _GPU_AVAILABLE and os.environ.get("DCEGM_GPU_FOC", "0") == "1"

from HARK.econforgeinterp import LinearFast
from HARK.interpolation import ValueFuncCRRA, LowerEnvelope2D
from HARK.ConsumptionSaving.ConsIndShockModel import utility, utility_inv, utilityP_inv
from HARK.dcegm import calc_nondecreasing_segments, calc_cross_points
from HARK.distribution import MeanOneLogNormal
from HARK.utilities import make_grid_exp_mult
from HARK.core import MetricObject

from utility import (
    construct_grid,
    CRRAutilityP_inv_njit,
    CRRAutility_njit,
    MargValueFuncCRRA_dur,
    HANKIncShkDstn_Truncated,
    update_prob_grid_list_with_trunc,
    compute_cost_matrix,
    directed_search_function_scaled_matrix,
    directed_search_function_closed_form,
    FUES_allPol,
    jump_to_grid_dur,
    gen_tran_matrix_dur_sparse,
    compute_erg_dstn_sparse,
    J_from_F,
)


def _eval2d_batch_gpu(x_grid_g, y_grid_g, f_val_g, xi_g, yi_g):
    """Bilinear interpolation with linear extrapolation (LinearFast-equivalent)."""
    cp = _cp
    nx, ny = len(x_grid_g), len(y_grid_g)
    ix = cp.clip(cp.searchsorted(x_grid_g, xi_g) - 1, 0, nx - 2).astype(cp.int32)
    iy = cp.clip(cp.searchsorted(y_grid_g, yi_g) - 1, 0, ny - 2).astype(cp.int32)
    x0, x1 = x_grid_g[ix], x_grid_g[ix + 1]
    y0, y1 = y_grid_g[iy], y_grid_g[iy + 1]
    tx, ty = (xi_g - x0) / (x1 - x0), (yi_g - y0) / (y1 - y0)
    return (
        (1.0 - tx) * (1.0 - ty) * f_val_g[ix, iy]
        + (1.0 - tx) * ty * f_val_g[ix, iy + 1]
        + tx * (1.0 - ty) * f_val_g[ix + 1, iy]
        + tx * ty * f_val_g[ix + 1, iy + 1]
    )


def _eval_lf(lf, x, y):
    """Evaluate LinearFast on broadcastable x, y.

    Stay on CPU: at Combined-test sizes a CuPy round-trip is dominated by H2D/D2H.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shp = np.broadcast(x, y).shape
    return np.asarray(
        lf(np.broadcast_to(x, shp), np.broadcast_to(y, shp)), dtype=float
    ).reshape(shp)


def _gpu_foc_residual(d_g, a_g, dGrid_g, aGrid_g, eop1_g, eopd_g, price, alpha, d_ubar, CRRA):
    eP1 = _eval2d_batch_gpu(dGrid_g, aGrid_g, eop1_g, d_g, a_g)
    ePD = _eval2d_batch_gpu(dGrid_g, aGrid_g, eopd_g, d_g, a_g)
    C = (eP1 / (alpha * (d_g + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA)))) ** (
        1.0 / (alpha * (1.0 - CRRA) - 1.0)
    )
    Term1 = (
        (1.0 - alpha)
        * C ** (alpha * (1.0 - CRRA))
        * (d_g + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA) - 1.0)
    )
    return Term1 - price * eP1 + ePD


def _gpu_bisect_all(
    d_br, a_lo, a_hi, f_lo_sign, dGrid_g, aGrid_g, eop1_g, eopd_g,
    price, alpha, d_ubar, CRRA, n_iter=60,
):
    f_lo_s = f_lo_sign.astype(_cp.float64)
    for _ in range(n_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = _gpu_foc_residual(
            d_br, a_mid, dGrid_g, aGrid_g, eop1_g, eopd_g, price, alpha, d_ubar, CRRA
        )
        upd = _cp.sign(f_mid) == f_lo_s
        a_lo = _cp.where(upd, a_mid, a_lo)
        f_lo_s = _cp.where(upd, _cp.sign(f_mid), f_lo_s)
        a_hi = _cp.where(~upd, a_mid, a_hi)
    return 0.5 * (a_lo + a_hi)


@njit(cache=True)
def _eval2d(x_grid, y_grid, f_val, xi, yi):
    nx, ny = len(x_grid), len(y_grid)
    ix = min(max(int(np.searchsorted(x_grid, xi)) - 1, 0), nx - 2)
    iy = min(max(int(np.searchsorted(y_grid, yi)) - 1, 0), ny - 2)
    tx = (xi - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    ty = (yi - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    return (
        (1.0 - tx) * (1.0 - ty) * f_val[ix, iy]
        + tx * (1.0 - ty) * f_val[ix + 1, iy]
        + (1.0 - tx) * ty * f_val[ix, iy + 1]
        + tx * ty * f_val[ix + 1, iy + 1]
    )


@njit(cache=True)
def _foc_pt(d, a, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA):
    eP1 = _eval2d(dGrid, aGrid, eop1, d, a)
    ePD = _eval2d(dGrid, aGrid, eopd, d, a)
    C = (eP1 / (alpha * (d + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA)))) ** (
        1.0 / (alpha * (1.0 - CRRA) - 1.0)
    )
    Term1 = (
        (1.0 - alpha)
        * C ** (alpha * (1.0 - CRRA))
        * (d + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA) - 1.0)
    )
    return Term1 - price * eP1 + ePD


@njit(parallel=True, cache=True)
def _scan_foc(d_adj, a_scan, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA):
    nD, nA = len(d_adj), len(a_scan)
    f = np.empty((nD, nA))
    for i in prange(nD):
        d = d_adj[i]
        for j in range(nA):
            f[i, j] = _foc_pt(
                d, a_scan[j], dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA
            )
    return f


@njit(parallel=True, cache=True)
def _bisect_many(
    d_br, a_lo, a_hi, f_lo_sign, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA,
    n_iter=60,
):
    n = len(d_br)
    out = np.empty(n)
    for i in prange(n):
        lo, hi, sgn = a_lo[i], a_hi[i], f_lo_sign[i]
        d = d_br[i]
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            fm = _foc_pt(d, mid, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA)
            if np.sign(fm) == sgn:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


def _cpu_foc_roots(d_adj, a_scan, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA):
    d_adj = np.ascontiguousarray(d_adj, dtype=np.float64)
    a_scan = np.ascontiguousarray(a_scan, dtype=np.float64)
    dGrid = np.ascontiguousarray(dGrid, dtype=np.float64)
    aGrid = np.ascontiguousarray(aGrid, dtype=np.float64)
    eop1 = np.ascontiguousarray(eop1, dtype=np.float64)
    eopd = np.ascontiguousarray(eopd, dtype=np.float64)
    f = _scan_foc(d_adj, a_scan, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA)
    f_scan = np.where(np.isnan(f), 0.0, f)
    row, col = np.where(np.sign(f_scan)[:, :-1] != np.sign(f_scan)[:, 1:])
    empty = (np.empty(0), np.empty(0))
    if row.size == 0:
        return empty
    f_lo, f_hi = f[row, col], f[row, col + 1]
    ok = np.isfinite(f_lo) & np.isfinite(f_hi) & (f_lo * f_hi < 0)
    row, col = row[ok], col[ok]
    if row.size == 0:
        return empty
    a_rt = _bisect_many(
        d_adj[row], a_scan[col], a_scan[col + 1], np.sign(f[row, col]),
        dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA,
    )
    d_rt = d_adj[row]
    ok = a_rt >= -1e-8
    return a_rt[ok], d_rt[ok]


@njit(cache=True)
def _unique_sorted(vals):
    vals = np.sort(vals)
    n = len(vals)
    if n == 0:
        return vals
    out = np.empty(n, dtype=np.float64)
    out[0] = vals[0]
    k = 1
    for i in range(1, n):
        if vals[i] != vals[i - 1]:
            out[k] = vals[i]
            k += 1
    return out[:k]


@njit(cache=True)
def _interp_no_extrap(x, y, xq):
    """Linear interpolation; NaN outside [x[0], x[-1]]. Matches HARK envelope interp."""
    n = len(x)
    nq = len(xq)
    out = np.empty(nq, dtype=np.float64)
    for i in range(nq):
        out[i] = np.nan
    if n == 0:
        return out
    x0 = x[0]
    x1 = x[n - 1]
    for i in range(nq):
        xi = xq[i]
        if xi < x0 or xi > x1:
            continue
        j = np.searchsorted(x, xi)
        if j == 0:
            out[i] = y[0]
        elif x[j] == xi:
            out[i] = y[j]
        else:
            t = (xi - x[j - 1]) / (x[j] - x[j - 1])
            out[i] = (1.0 - t) * y[j - 1] + t * y[j]
    return out


@njit(cache=True)
def _interp_extrap(x, y, xq):
    """Linear interpolation with linear extrapolation (LinearFast 1-D)."""
    n = len(x)
    nq = len(xq)
    out = np.empty(nq, dtype=np.float64)
    if n == 0:
        for i in range(nq):
            out[i] = np.nan
        return out
    if n == 1:
        for i in range(nq):
            out[i] = y[0]
        return out
    for i in range(nq):
        j = np.searchsorted(x, xq[i])
        if j < 1:
            j = 1
        elif j >= n:
            j = n - 1
        dx = x[j] - x[j - 1]
        if dx == 0.0:
            out[i] = y[j]
        else:
            t = (xq[i] - x[j - 1]) / dx
            out[i] = (1.0 - t) * y[j - 1] + t * y[j]
    return out


@njit(cache=True)
def _insert_envelope_points(x, y, inds, xtra_x, xtra_y, xtra_ind):
    """Insert extras at np.searchsorted positions (stable, like np.insert)."""
    n = len(x)
    m = len(xtra_x)
    if m == 0:
        return x, y, inds
    slots = np.empty(m, dtype=np.int64)
    for i in range(m):
        slots[i] = np.searchsorted(x, xtra_x[i])
    for i in range(1, m):
        key_s = slots[i]
        kx, ky, ki = xtra_x[i], xtra_y[i], xtra_ind[i]
        j = i - 1
        while j >= 0 and slots[j] > key_s:
            slots[j + 1] = slots[j]
            xtra_x[j + 1] = xtra_x[j]
            xtra_y[j + 1] = xtra_y[j]
            xtra_ind[j + 1] = xtra_ind[j]
            j -= 1
        slots[j + 1] = key_s
        xtra_x[j + 1] = kx
        xtra_y[j + 1] = ky
        xtra_ind[j + 1] = ki
    out_n = n + m
    xo = np.empty(out_n, dtype=np.float64)
    yo = np.empty(out_n, dtype=np.float64)
    io = np.empty(out_n, dtype=np.int64)
    i = 0
    e = 0
    w = 0
    while w < out_n:
        if e < m and slots[e] <= i:
            xo[w] = xtra_x[e]
            yo[w] = xtra_y[e]
            io[w] = xtra_ind[e]
            e += 1
        else:
            xo[w] = x[i]
            yo[w] = y[i]
            io[w] = inds[i]
            i += 1
        w += 1
    return xo, yo, io


@njit(cache=True)
def _fast_upper_envelope(starts, ends, m_egm, v_egm):
    """
    Packed-array port of HARK ``upper_envelope`` (crossings on).

    Returns (m_upper, v_upper, inds_upper, ok). ok is False if a grid column
    is all-NaN (HARK ``nanargmax`` raises ValueError).
    """
    n_seg = len(starts)
    empty_m = np.empty(0, dtype=np.float64)
    empty_i = np.empty(0, dtype=np.int64)
    if n_seg == 0:
        return empty_m, empty_m, empty_i, False
    x = _unique_sorted(m_egm)
    nx = len(x)
    if nx == 0:
        return empty_m, empty_m, empty_i, False
    y_cond = np.empty((n_seg, nx), dtype=np.float64)
    for s in range(n_seg):
        i0 = starts[s]
        i1 = ends[s]
        slen = i1 - i0 + 1
        if slen == 1:
            for j in range(nx):
                y_cond[s, j] = np.nan
            ind = np.searchsorted(x, m_egm[i0])
            if 0 <= ind < nx:
                y_cond[s, ind] = v_egm[i0]
        else:
            row = _interp_no_extrap(m_egm[i0 : i1 + 1], v_egm[i0 : i1 + 1], x)
            for j in range(nx):
                y_cond[s, j] = row[j]
    env_inds = np.empty(nx, dtype=np.int64)
    y = np.empty(nx, dtype=np.float64)
    for j in range(nx):
        best = -np.inf
        bi = -1
        for s in range(n_seg):
            v = y_cond[s, j]
            if v == v and v > best:
                best = v
                bi = s
        if bi < 0:
            return empty_m, empty_m, empty_i, False
        env_inds[j] = bi
        y[j] = best
    xing_points, xing_lines = calc_cross_points(x, y_cond, env_inds)
    n_xp = xing_points.shape[0]
    if n_xp > 0:
        n_fin = 0
        for i in range(n_xp):
            if xing_points[i, 0] == xing_points[i, 0]:
                n_fin += 1
        if n_fin > 0:
            xtra_x = np.empty(2 * n_fin, dtype=np.float64)
            xtra_y = np.empty(2 * n_fin, dtype=np.float64)
            xtra_ind = np.empty(2 * n_fin, dtype=np.int64)
            k = 0
            for i in range(n_xp):
                xx = xing_points[i, 0]
                if xx == xx:
                    yy = xing_points[i, 1]
                    xtra_x[k] = xx
                    xtra_y[k] = yy
                    xtra_ind[k] = xing_lines[i, 0]
                    k += 1
                    xtra_x[k] = np.nextafter(xx, xx + 1.0)
                    xtra_y[k] = yy
                    xtra_ind[k] = xing_lines[i, 1]
                    k += 1
            x, y, env_inds = _insert_envelope_points(
                x, y, env_inds, xtra_x, xtra_y, xtra_ind
            )
    return x, y, env_inds, True


@njit(cache=True)
def _fast_c_on_envelope(starts, ends, m_egm, c_egm, m_upper, inds_upper):
    """Map EGM consumption onto the envelope m-grid (HARK LinearFast / quirk)."""
    n_u = len(m_upper)
    c_upper = np.empty(n_u, dtype=np.float64)
    for i in range(n_u):
        c_upper[i] = np.nan
    if len(starts) == 0:
        return c_upper
    n_egm = len(m_egm)
    slen0 = ends[0] - starts[0] + 1
    if starts[len(starts) - 1] + 1 == n_egm and slen0 >= 2:
        i0 = starts[0]
        i1 = ends[0]
        return _interp_extrap(m_egm[i0 : i1 + 1], c_egm[i0 : i1 + 1], m_upper)
    n_seg = len(starts)
    for k in range(n_seg):
        i0 = starts[k]
        i1 = ends[k]
        slen = i1 - i0 + 1
        if slen == 1:
            ck = c_egm[i0]
            for i in range(n_u):
                if inds_upper[i] == k:
                    c_upper[i] = ck
        elif slen >= 2:
            n_hit = 0
            for i in range(n_u):
                if inds_upper[i] == k:
                    n_hit += 1
            if n_hit == 0:
                continue
            mq = np.empty(n_hit, dtype=np.float64)
            loc = np.empty(n_hit, dtype=np.int64)
            t = 0
            for i in range(n_u):
                if inds_upper[i] == k:
                    mq[t] = m_upper[i]
                    loc[t] = i
                    t += 1
            cq = _interp_extrap(m_egm[i0 : i1 + 1], c_egm[i0 : i1 + 1], mq)
            for t in range(n_hit):
                c_upper[loc[t]] = cq[t]
    return c_upper


@njit(cache=True)
def _any_finite(a):
    for i in range(len(a)):
        v = a[i]
        if v == v and np.isfinite(v):
            return True
    return False


@njit(cache=True)
def _fallback_c_on_m(mNrmGrid, BoroCnstNat):
    return _interp_extrap(
        np.array([BoroCnstNat, 1.0]), np.array([0.0, 1.0]), mNrmGrid
    )


@njit(parallel=True, cache=True)
def _fast_keeper_rows(
    dNrmGrid, aNrmGrid, mNrmGrid, EndOfPrdvPOne, EndOfPrdv, alpha, d_ubar, CRRA, BoroCnstNat,
):
    """EGM + DCEGM envelope for all keeper n-rows; c on ``mNrmGrid``."""
    nN = len(dNrmGrid)
    nA = len(aNrmGrid)
    nM = len(mNrmGrid)
    c_unc = np.empty((nN, nM), dtype=np.float64)
    n_dec = np.zeros(nN, dtype=np.int64)
    n_tot = np.zeros(nN, dtype=np.int64)
    for i_n in prange(nN):
        n0 = dNrmGrid[i_n]
        n_aux = np.empty(nA, dtype=np.float64)
        for i in range(nA):
            n_aux[i] = n0
        eop1 = EndOfPrdvPOne[i_n]
        eopv = EndOfPrdv[i_n]
        c_egm = CRRAutilityP_inv_njit(eop1, n_aux, alpha, d_ubar, CRRA)
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility_njit(c_egm, n_aux, alpha, d_ubar, CRRA) + eopv
        vKeept = np.empty(nA, dtype=np.float64)
        for i in range(nA):
            vv = v_egm[i]
            if vv < -700.0:
                vv = -700.0
            elif vv > 700.0:
                vv = 700.0
            vKeept[i] = np.exp(vv)
        starts, ends = calc_nondecreasing_segments(m_egm, vKeept)
        n_nondec = len(starts)
        n_d = n_nondec - 1
        if n_d < 0:
            n_d = 0
        n_dec[i_n] = n_d
        n_tot[i_n] = n_nondec + n_d
        any_fin = False
        for j in range(n_nondec):
            for t in range(starts[j], ends[j] + 1):
                vt = vKeept[t]
                if vt == vt and np.isfinite(vt):
                    any_fin = True
                    break
            if any_fin:
                break
        if n_nondec == 0 or not any_fin:
            c_unc[i_n, :] = np.maximum(_fallback_c_on_m(mNrmGrid, BoroCnstNat), 0.0)
            continue
        m_u, _, inds, ok = _fast_upper_envelope(starts, ends, m_egm, vKeept)
        if not ok:
            c_unc[i_n, :] = np.maximum(_fallback_c_on_m(mNrmGrid, BoroCnstNat), 0.0)
            continue
        c_u = _fast_c_on_envelope(starts, ends, m_egm, c_egm, m_u, inds)
        if c_u[0] != 0.0:
            n_u = len(m_u)
            m_i = np.empty(n_u + 1, dtype=np.float64)
            c_i = np.empty(n_u + 1, dtype=np.float64)
            m_i[0] = BoroCnstNat
            c_i[0] = 0.0
            m_i[1:] = m_u
            c_i[1:] = c_u
        else:
            m_i = m_u
            c_i = c_u
        if len(m_i) < 2 or not _any_finite(c_i):
            c_unc[i_n, :] = np.maximum(_fallback_c_on_m(mNrmGrid, BoroCnstNat), 0.0)
        else:
            c_unc[i_n, :] = np.maximum(_interp_extrap(m_i, c_i, mNrmGrid), 0.0)
    return c_unc, n_dec, n_tot


def _gpu_foc_ctx(d_adj, a_scan, dGrid, aGrid, eop1, eopd):
    dAdj_g = _cp.asarray(np.ascontiguousarray(d_adj, dtype=np.float64))
    a_scan_g = _cp.asarray(np.ascontiguousarray(a_scan, dtype=np.float64))
    nD, nA = len(d_adj), len(a_scan)
    return dict(
        dAdj_g=dAdj_g,
        a_scan_g=a_scan_g,
        d_mat=dAdj_g[:, None] * _cp.ones((1, nA)),
        a_mat=_cp.ones((nD, 1)) * a_scan_g[None, :],
        dGrid_g=_cp.asarray(np.ascontiguousarray(dGrid, dtype=np.float64)),
        aGrid_g=_cp.asarray(np.ascontiguousarray(aGrid, dtype=np.float64)),
        eop1_g=_cp.asarray(np.ascontiguousarray(eop1, dtype=np.float64)),
        eopd_g=_cp.asarray(np.ascontiguousarray(eopd, dtype=np.float64)),
        nD=nD,
        nA=nA,
    )


def _gpu_foc_from_ctx(ctx, price, alpha, d_ubar, CRRA):
    f = _gpu_foc_residual(
        ctx["d_mat"].ravel(), ctx["a_mat"].ravel(),
        ctx["dGrid_g"], ctx["aGrid_g"], ctx["eop1_g"], ctx["eopd_g"],
        price, alpha, d_ubar, CRRA,
    ).reshape(ctx["nD"], ctx["nA"])
    f_scan = _cp.where(_cp.isnan(f), 0.0, f)
    row, col = _cp.where(_cp.sign(f_scan)[:, :-1] != _cp.sign(f_scan)[:, 1:])
    empty = (np.empty(0), np.empty(0))
    if len(row) == 0:
        return empty
    f_lo, f_hi = f[row, col], f[row, col + 1]
    ok = _cp.isfinite(f_lo) & _cp.isfinite(f_hi) & (f_lo * f_hi < 0)
    row, col = row[ok], col[ok]
    if len(row) == 0:
        return empty
    a_rt = _gpu_bisect_all(
        ctx["dAdj_g"][row], ctx["a_mat"][row, col], ctx["a_mat"][row, col + 1],
        _cp.sign(f[row, col]),
        ctx["dGrid_g"], ctx["aGrid_g"], ctx["eop1_g"], ctx["eopd_g"],
        price, alpha, d_ubar, CRRA,
    )
    d_rt = ctx["dAdj_g"][row]
    ok = a_rt >= -1e-8
    return _cp.asnumpy(a_rt[ok]), _cp.asnumpy(d_rt[ok])


def _gpu_foc_roots(d_adj, a_scan, dGrid, aGrid, eop1, eopd, price, alpha, d_ubar, CRRA):
    return _gpu_foc_from_ctx(
        _gpu_foc_ctx(d_adj, a_scan, dGrid, aGrid, eop1, eopd),
        price, alpha, d_ubar, CRRA,
    )


def _gpu_warmup():
    if not _GPU_AVAILABLE:
        return
    try:
        xg = _cp.array([0.0, 1.0, 2.0])
        fg = _cp.ones((3, 3))
        _eval2d_batch_gpu(xg, xg, fg, _cp.array([0.5, 1.5]), _cp.array([0.5, 1.5]))
        d = np.array([0.5, 1.0])
        a = np.array([0.0, 1.0, 2.0])
        e = np.ones((2, 3))
        _gpu_foc_roots(d, a, d, a, e, e, 1.0, 0.75, 1e-4, 2.0)
        _cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


_gpu_warmup()


def _cpu_warmup():
    d = np.array([0.5, 1.0])
    a = np.array([0.0, 1.0, 2.0])
    e = np.ones((2, 3))
    _cpu_foc_roots(d, a, d, a, e, e, 1.0, 0.75, 1e-4, 2.0)
    x = np.linspace(0.0, 1.0, 12)
    FUES_allPol(x, np.sqrt(x), 0.5 * x, x, x, b=0.0)
    a_k = np.linspace(0.0, 2.0, 8)
    d_k = np.array([0.5, 1.0, 1.5])
    m_k = np.linspace(0.0, 3.0, 6)
    e1 = np.ones((3, 8))
    ev = np.ones((3, 8))
    _fast_keeper_rows(d_k, a_k, m_k, e1, ev, 0.75, 1e-4, 2.0, 0.0)


_cpu_warmup()


def TranShkMean_Func(taxrate, labor, wage):
    return (1.0 - taxrate) * labor * wage


def calcLogSumChoiceProbs(Vals, sigma):
    """
    Inclusive value and logit probabilities for i.i.d. EV Type I taste shocks.

    Matches ``calcLogSumChoiceProbs(..., smoothtype='type1')`` in the retirement
    code and ``DurableModel_TasteShock_One_Period.ipynb``. Non-finite entries are
    treated as infeasible (sentinel before the exp, else ``(-inf)-(-inf)`` → NaN).
    """
    Vals = np.asarray(Vals, dtype=float)
    if sigma == 0.0:
        Vals_rank = np.where(np.isfinite(Vals), Vals, -np.inf)
        Pflat = np.argmax(Vals_rank, axis=0)
        V = np.full(Vals[0].shape, -np.inf)
        Probs = np.zeros(Vals.shape)
        for i in range(Vals.shape[0]):
            optimal = Pflat == i
            V[optimal] = Vals_rank[i][optimal]
            Probs[i][optimal] = 1.0
        return V, Probs

    finite = np.isfinite(Vals)
    Vals_work = np.where(finite, Vals, -1.0e300)
    maxV = np.max(Vals_work, axis=0)
    sumexp = np.sum(np.exp((Vals_work - maxV) / sigma), axis=0)
    LogSumV = maxV + sigma * np.log(sumexp)
    Probs = np.exp((Vals_work - LogSumV) / sigma)

    all_bad = ~np.any(finite, axis=0)
    LogSumV = np.where(all_bad, -np.inf, LogSumV)
    Probs = np.where(all_bad, 0.0, Probs)
    Probs = np.where(finite, Probs, 0.0)
    mass = Probs.sum(axis=0, keepdims=True)
    Probs = np.where(mass > 0.0, Probs / np.where(mass > 0.0, mass, 1.0), Probs)
    return LogSumV, Probs


def _mix_marg(P0, x0, P1, x1):
    """Probability-weighted mix; propagate +inf when a positive-prob branch is non-finite."""
    bad = ((P0 > 0) & ~np.isfinite(x0)) | ((P1 > 0) & ~np.isfinite(x1))
    x0s = np.where(np.isfinite(x0), x0, 0.0)
    x1s = np.where(np.isfinite(x1), x1, 0.0)
    return np.where(bad, np.inf, P0 * x0s + P1 * x1s)


def combine_ev_taste_price(
    price_vals, price_weights,
    cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
    cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
    sigma, CRRA_,
    kappa=0.0,
):
    """
    EV Type I (or hard-max) combine over keep vs adjust, then integrate over prices.

    For each price node p_k the adjuster level value is shifted by a utility
    cost ``kappa`` (Zorzi–Beraja location) and then
      v = sigma * log(exp(vKeep/sigma) + exp((vAdj(p_k) - kappa)/sigma)),
      P^adj = logit weight on the shifted adjuster.
    ``kappa=0`` is the symmetric log-sum. ``sigma=0`` collapses to
    max{vKeep, vAdj - kappa}. Results are then averaged with price weights.
    Adjuster arrays: (n_price, n_n, n_m). Keeper: (n_n, n_m).
    ``adjusting`` returned is E_p[P^adj].
    Marginal values use those probabilities and the unshifted branch
    derivatives: kappa does not depend on the state.
    """
    n_price = cAdj.shape[0]
    keep_shape = cKeep.shape

    vKeep_level = utility(inv_vKeep, CRRA_)
    vAdj_level = utility(inv_vAdj, CRRA_)
    if kappa != 0.0:
        vAdj_level = np.where(np.isfinite(vAdj_level), vAdj_level - kappa, vAdj_level)

    barC = np.zeros(keep_shape)
    barD = np.zeros(keep_shape)
    barA = np.zeros(keep_shape)
    barV = np.zeros(keep_shape)
    barVP = np.zeros(keep_shape)
    barVPD = np.zeros(keep_shape)
    barAdj = np.zeros(keep_shape)

    for i_p in range(n_price):
        w = price_weights[i_p]
        Vals = np.stack([vKeep_level, vAdj_level[i_p]], axis=0)
        V, Probs = calcLogSumChoiceProbs(Vals, sigma)
        Pkeep, Padj = Probs[0], Probs[1]

        barC += w * (Pkeep * cKeep + Padj * cAdj[i_p])
        barD += w * (Pkeep * dKeep + Padj * dAdj[i_p])
        barA += w * (Pkeep * aKeep + Padj * aAdj[i_p])
        barV += w * V
        barVP += w * _mix_marg(Pkeep, vPKeep, Padj, vPAdj[i_p])
        barVPD += w * _mix_marg(Pkeep, vPDKeep, Padj, vPDAdj[i_p])
        barAdj += w * Padj

    return barC, barD, barA, barV, barVP, barVPD, barAdj


def combine_search_price(
    price_vals, price_weights,
    cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
    cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
    nGrid, mGrid, SearchCost_func_, e_grid_, CRRA_,
    search_method="closed",
):
    """
    Nested combine over price nodes with directed search (Combined notebook):
      v = max_eps[(1-eps) vKeep + eps * vAdj(p) - sigma(eps)],
    then integrate with mu_p.

    ``search_method`` is ``"closed"`` (conjugate ε* = max{0, 1-exp(-Δ/S)}) or
    ``"grid"`` (argmax over ``e_grid_``, the original 101-point kernel).

    Shutoffs: PriceShkStd=0, SearchCost=0 (hard max via zero costs).
    Adjuster arrays: (n_price, n_n, n_m). Keeper: (n_n, n_m).
    """
    if search_method not in ("grid", "closed"):
        raise ValueError("search_method must be 'grid' or 'closed'.")

    n_price = cAdj.shape[0]
    keep_shape = cKeep.shape

    vKeep_level = utility(inv_vKeep, CRRA_)
    vAdj_level = utility(inv_vAdj, CRRA_)

    nm = nGrid[:, None] + mGrid[None, :]
    search_cost_matrix = np.asarray(SearchCost_func_(nm.ravel()), dtype=float).reshape(
        keep_shape
    )
    Costs_matrix = (
        compute_cost_matrix(search_cost_matrix, e_grid_)
        if search_method == "grid"
        else None
    )
    mNrmGridMin = np.zeros(1)

    barC = np.zeros(keep_shape)
    barD = np.zeros(keep_shape)
    barA = np.zeros(keep_shape)
    barV = np.zeros(keep_shape)
    barVP = np.zeros(keep_shape)
    barVPD = np.zeros(keep_shape)
    barEps = np.zeros(keep_shape)

    for i_p in range(n_price):
        w = price_weights[i_p]
        vAdj_p = vAdj_level[i_p]
        vPAdj_p = vPAdj[i_p]
        vPDAdj_p = vPDAdj[i_p]

        vAdj_aux = np.where(vAdj_p == -np.inf, -1e12, vAdj_p)
        vKeep_aux = np.where(vKeep_level == -np.inf, -1e12, vKeep_level)
        vPAdj_aux = np.where(vPAdj_p == np.inf, 1e12, vPAdj_p)
        vPDAdj_aux = np.where(vPDAdj_p == np.inf, 1e12, vPDAdj_p)
        vPKeep_aux = np.where(vPKeep == np.inf, 1e12, vPKeep)
        vPDKeep_aux = np.where(vPDKeep == np.inf, 1e12, vPDKeep)

        if search_method == "grid":
            (
                eps_ij, _cost_ij, v_ij,
                c_ij, d_ij, a_ij, vP_ij, vPD_ij,
            ) = directed_search_function_scaled_matrix(
                nGrid, mGrid, mNrmGridMin, e_grid_, Costs_matrix,
                vAdj_aux, vKeep_aux,
                cAdj[i_p], cKeep,
                dAdj[i_p], dKeep,
                aAdj[i_p], aKeep,
                vPAdj_aux, vPKeep_aux,
                vPDAdj_aux, vPDKeep_aux,
            )
        else:
            (
                eps_ij, _cost_ij, v_ij,
                c_ij, d_ij, a_ij, vP_ij, vPD_ij,
            ) = directed_search_function_closed_form(
                nGrid, mGrid, mNrmGridMin, search_cost_matrix,
                vAdj_aux, vKeep_aux,
                cAdj[i_p], cKeep,
                dAdj[i_p], dKeep,
                aAdj[i_p], aKeep,
                vPAdj_aux, vPKeep_aux,
                vPDAdj_aux, vPDKeep_aux,
            )

        v_ij = np.asarray(v_ij, dtype=float)
        vP_ij = np.asarray(vP_ij, dtype=float)
        vPD_ij = np.asarray(vPD_ij, dtype=float)

        cond_inf_v = (
            ((eps_ij == 1.0) & (vAdj_p == -np.inf))
            | ((eps_ij == 0.0) & (vKeep_level == -np.inf))
            | (((eps_ij > 0.0) & (eps_ij < 1.0))
               & ((vAdj_p == -np.inf) | (vKeep_level == -np.inf)))
        )
        v_ij[cond_inf_v] = -np.inf

        cond_inf_vP = (
            ((eps_ij == 1.0) & (vPAdj_p == np.inf))
            | ((eps_ij == 0.0) & (vPKeep == np.inf))
            | (((eps_ij > 0.0) & (eps_ij < 1.0))
               & ((vPAdj_p == np.inf) | (vPKeep == np.inf)))
        )
        vP_ij[cond_inf_vP] = np.inf

        cond_inf_vPD = (
            ((eps_ij == 1.0) & (vPDAdj_p == np.inf))
            | ((eps_ij == 0.0) & (vPDKeep == np.inf))
            | (((eps_ij > 0.0) & (eps_ij < 1.0))
               & ((vPDAdj_p == np.inf) | (vPDKeep == np.inf)))
        )
        vPD_ij[cond_inf_vPD] = np.inf

        barC += w * c_ij
        barD += w * d_ij
        barA += w * a_ij
        barV += w * v_ij
        barVP += w * vP_ij
        barVPD += w * vPD_ij
        barEps += w * eps_ij

    return barC, barD, barA, barV, barVP, barVPD, barEps


class DurableConsumerSolution(MetricObject):
    """One-period solution objects for plotting and continuation."""

    distance_criteria = ["inv_vPFunc", "inv_vPDFunc"]
    # distance_criteria = ["vPFunc", "vPDFunc"]

    def __init__(
        self,
        cFunc=None,
        dFunc=None,
        aFunc=None,
        vFunc=None,
        inv_vFunc=None,
        vPFunc=None,
        vPDFunc=None,
        inv_vPFunc=None,
        inv_vPDFunc=None,
        adjusting=None,
        cFuncKeep=None,
        dFuncKeep=None,
        cFuncAdj=None,
        dFuncAdj=None,
        cFuncAdj_1d=None,
        dFuncAdj_1d=None,
        cFuncAdj_1d_list=None,
        dFuncAdj_1d_list=None,
        inv_vFuncKeep=None,
        inv_vFuncAdj_1d_list=None,
        adjusting_arr=None,
        cFunc_arr=None,
        dFunc_arr=None,
        nNrmGrid=None,
        mNrmGrid=None,
        # Pre-/post-FUES adjuster clouds (lists over price nodes; also p≈1 aliases)
        xAdj_pre_list=None,
        cAdj_pre_list=None,
        dAdj_pre_list=None,
        aAdj_pre_list=None,
        vAdj_pre_list=None,
        xAdj_post_list=None,
        cAdj_post_list=None,
        dAdj_post_list=None,
        aAdj_post_list=None,
        vAdj_post_list=None,
        xAdj_pre=None,
        cAdj_pre=None,
        dAdj_pre=None,
        aAdj_pre=None,
        vAdj_pre=None,
        xAdj_post=None,
        cAdj_post=None,
        dAdj_post=None,
        aAdj_post=None,
        vAdj_post=None,
        # DCEGM / FUES diagnostics (None for terminal)
        keep_decreasing_segments=None,
        keep_segments_total=None,
        keep_share_decreasing=None,
        adj_kinks_removed=None,
        adj_points_total=None,
        adj_share_removed=None,
    ):
        self.cFunc = cFunc
        self.dFunc = dFunc
        self.aFunc = aFunc
        self.vFunc = vFunc
        self.inv_vFunc = inv_vFunc
        self.vPFunc = vPFunc
        self.vPDFunc = vPDFunc
        self.inv_vPFunc = inv_vPFunc
        self.inv_vPDFunc = inv_vPDFunc
        self.adjusting = adjusting
        self.cFuncKeep = cFuncKeep
        self.dFuncKeep = dFuncKeep
        self.cFuncAdj = cFuncAdj
        self.dFuncAdj = dFuncAdj
        # 1-D adjuster policies in liquidity ℓ (representative price ≈ 1)
        self.cFuncAdj_1d = cFuncAdj_1d
        self.dFuncAdj_1d = dFuncAdj_1d
        # All price-node-specific 1-D adjuster policies
        self.cFuncAdj_1d_list = cFuncAdj_1d_list
        self.dFuncAdj_1d_list = dFuncAdj_1d_list
        # Conditional values: keeper on (n,m), adjuster 1-D on liquidity x
        self.inv_vFuncKeep = inv_vFuncKeep
        self.inv_vFuncAdj_1d_list = inv_vFuncAdj_1d_list
        self.adjusting_arr = adjusting_arr
        self.cFunc_arr = cFunc_arr
        self.dFunc_arr = dFunc_arr
        self.nNrmGrid = nNrmGrid
        self.mNrmGrid = mNrmGrid
        # Pre-FUES FOC candidate clouds (one array per price node)
        self.xAdj_pre_list = xAdj_pre_list
        self.cAdj_pre_list = cAdj_pre_list
        self.dAdj_pre_list = dAdj_pre_list
        self.aAdj_pre_list = aAdj_pre_list
        self.vAdj_pre_list = vAdj_pre_list
        # Post-FUES refined clouds (one array per price node)
        self.xAdj_post_list = xAdj_post_list
        self.cAdj_post_list = cAdj_post_list
        self.dAdj_post_list = dAdj_post_list
        self.aAdj_post_list = aAdj_post_list
        self.vAdj_post_list = vAdj_post_list
        # Convenience: representative price ≈ 1
        self.xAdj_pre = xAdj_pre
        self.cAdj_pre = cAdj_pre
        self.dAdj_pre = dAdj_pre
        self.aAdj_pre = aAdj_pre
        self.vAdj_pre = vAdj_pre
        self.xAdj_post = xAdj_post
        self.cAdj_post = cAdj_post
        self.dAdj_post = dAdj_post
        self.aAdj_post = aAdj_post
        self.vAdj_post = vAdj_post
        # Keeper (DCEGM decreasing segments) and adjuster (FUES) kink counts
        self.keep_decreasing_segments = keep_decreasing_segments
        self.keep_segments_total = keep_segments_total
        self.keep_share_decreasing = keep_share_decreasing
        self.adj_kinks_removed = adj_kinks_removed
        self.adj_points_total = adj_points_total
        self.adj_share_removed = adj_share_removed


# Default parameters (Combined notebook)
durable_defaults = dict(
    CRRA=2.0,
    DiscFac=0.975,
    LivPrb=1.0,
    alpha=0.75,
    d_ubar=1e-4,
    dDepr=0.05,
    adjC=0.15,
    MaintCost=0.0,
    OpCost=0.0,
    Rfree=1.04 ** (0.25),
    RSpread=0.0,
    PermGroFac=1.0,
    LTV=0.0,
    BoroCnstArt=0.0,
    # Income
    PermShkStd=0.0,
    PermShkCount=1,
    TranShkStd=0.2,
    TranShkCount=7,
    UnempPrb=0.0,
    IncUnemp=0.0,
    IncUnemp_Std=0.1,
    truncation=2,
    taxrate=0.0,
    labor=1.0,
    wage=1.0,
    # Channels (EV taste sigma_eps vs search are mutually exclusive).
    # kappa is a utility cost of adjusting inside the logit (Zorzi–Beraja).
    # kappa=0 keeps the symmetric EV Type I log-sum.
    sigma_eps=0.0,
    kappa=0.0,
    SearchCost=0.1,
    ScaleBool=False,
    e_grid_count=101,
    search_method="closed",  # "closed" conjugate ε*; "grid" 101-point argmax
    PriceShkCount=5,
    PriceShkStd=0.1,
    # EoP method
    use_expected=True,
    use_TransitionMatrix=False,
    # Grids
    nNrmMax=5.0,
    nNrmCount=100,
    mNrmMax=8.0,
    mNrmCount=100,
    xNrmMax=13.0,
    xNrmCount=200,
    aNrmMax=8.0,
    aNrmCount=100,
    NestFac=1,
    grid_type="exp_mult",
    TranMatrix_mCount=1010,
    TranMatrix_bCount=1000,
    N_scan=800,
    # Steady-state histogram / transition-matrix grids (distribution over (n, m))
    TranMatrixmMin=1e-4,
    TranMatrixmMax=8.0,
    TranMatrixmCount=1000,
    TranMatrixnMin=0.0,
    TranMatrixnMax=5.0,
    TranMatrixnCount=100,
    nNrmInitMean=0.0,   # newborn durable stock seed (used only when LivPrb < 1)
    mNrmInitMean=1.0,   # newborn market-resource seed (used only when LivPrb < 1)
    neutral_measure=False,
    # Horizon (background convention): cycles is the MODE selector, T_cycle the
    # number of periods.
    #   cycles = 1 => finite horizon with T_cycle periods before the terminal
    #                 (solution has T_cycle + 1 entries; solution[-1] is terminal)
    #   cycles = 0 => infinite horizon (iterate the one-period operator to convergence)
    cycles=1,
    T_cycle=1,       # number of periods before terminal when cycles >= 1
    tol=1e-6,        # infinite-horizon convergence tolerance (matches background)
    min_cycles=0,    # minimum iterations before accepting convergence (cycles=0)
    max_cycles=500,  # iteration cap for cycles=0 (matches background max_cycles)
    # Aggregation used for the infinite-horizon convergence distance:
    #   "max"  -> max absolute difference (matches HARK's default behavior)
    #   "mse"  -> mean squared error
    #   "rmse" -> root mean squared error
    distance_metric="mse",
    # When True, solve steps emit diagnostic figures (e.g. the 1-D adjuster
    # consumption/durable policies plotted once per period/iteration).
    diagnostics=False,
    # When False, solve() uses a pre-set self.solution_terminal as the terminal
    # instead of computing the analytic terminal (used by the Jacobian solver).
    analytic_terminal=True,
    jac_dx=1e-4,     # perturbation size for sequence-space Jacobians
    verbose=False,
)


class DurableConsumer:
    """
    Simplified durable consumer with optional EV taste, search, and price channels.

    Discrete-choice smoothers (mutually exclusive):
      - Directed search when ``SearchCost > 0`` (Combined / Search notebooks).
        ``search_method="closed"`` uses the conjugate ε*; ``"grid"`` keeps the
        original 101-point argmax over ``e_grid``.
      - EV Type I taste shocks when ``SearchCost == 0``. ``sigma_eps`` is the
        scale (0 is a hard max). ``kappa`` is a utility cost subtracted from
        the adjuster value before the log-sum; ``kappa=0`` is the symmetric logit.
      - Hard max when ``sigma_eps``, ``kappa``, and ``SearchCost`` are zero

    Stochastic prices (``PriceShkStd``) nest with either smoother.

    Parameters
    ----------
    cycles : int
        Horizon mode selector (background convention).
        ``cycles = 1`` solves a finite horizon with ``T_cycle`` periods before
        the terminal (solution has ``T_cycle + 1`` entries; ``solution[-1]`` is
        terminal). ``cycles = 0`` solves the infinite horizon: the one-period
        operator is iterated from the terminal seed until the solution
        ``distance`` falls below ``tol`` after at least ``min_cycles``
        iterations (or ``max_cycles`` iterations are reached).
    T_cycle : int
        Number of periods before the terminal when ``cycles >= 1``
        (T_cycle=1 solves T-1 and T).
    tol : float
        Convergence tolerance for the infinite-horizon iteration (cycles=0).
    min_cycles : int
        Minimum number of iterations before convergence is accepted when
        ``cycles=0``. Use e.g. ``min_cycles=200`` to force a burn-in even if
        the distance falls below ``tol`` earlier. Must satisfy
        ``0 <= min_cycles <= max_cycles``.
    max_cycles : int
        Maximum number of iterations for the infinite-horizon iteration.
    distance_metric : str
        How to aggregate the per-point differences of the solution
        ``distance_criteria`` into the scalar convergence distance used in the
        infinite-horizon iteration (``cycles=0``). One of:
        ``"max"`` (maximum absolute difference; matches HARK's default),
        ``"mse"`` (mean squared error), or ``"rmse"`` (root mean squared error).
    diagnostics : bool
        When True, the solver emits diagnostic figures while solving. Currently
        this plots the 1-D (representative price ~= 1) adjuster consumption
        ``cFuncAdj`` and durable ``dFuncAdj`` policies once per period/iteration.
    **kwds
        Override any entry in durable_defaults.
    """

    def __init__(self, **kwds):
        if "sigma_xi" in kwds or "TasteShkCount" in kwds:
            raise ValueError(
                "Multiplicative log-normal taste shocks (sigma_xi / TasteShkCount) were "
                "removed. Use sigma_eps for EV Type I taste shocks (and SearchCost=0)."
            )
        params = dict(durable_defaults)
        params.update(kwds)
        for k, v in params.items():
            setattr(self, k, v)

        self.tau = self.adjC
        self.aXtraMax = self.aNrmMax
        if self.ScaleBool:
            raise NotImplementedError("Constant-cost search branch only (ScaleBool=False).")
        if abs(self.LTV) > 1e-14:
            raise NotImplementedError("Simplified DurableConsumer requires LTV=0.")
        if not (bool(self.use_expected) ^ bool(self.use_TransitionMatrix)):
            raise ValueError("Set exactly one of use_expected / use_TransitionMatrix.")
        if self.SearchCost > 0.0 and (self.sigma_eps > 0.0 or self.kappa != 0.0):
            raise ValueError(
                "Search and EV taste shocks (including a utility cost kappa) are "
                "alternative discrete smoothers; set SearchCost=0, or set "
                "sigma_eps=0 and kappa=0."
            )
        if self.search_method not in ("grid", "closed"):
            raise ValueError("search_method must be 'grid' or 'closed'.")

        self.solution = None
        self.solution_terminal = None
        self._updated = False

    # ------------------------------------------------------------------ utils
    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def CRRAutility(self, C, D):
        u_inner = (C ** self.alpha) * ((D + self.d_ubar) ** (1.0 - self.alpha))
        return utility(u_inner, self.CRRA)

    def CRRAutility_inv(self, C, D):
        return utility_inv(self.CRRAutility(C, D), self.CRRA)

    def CRRAutilityP(self, C, D):
        return (
            self.alpha
            * C ** (self.alpha * (1.0 - self.CRRA) - 1.0)
            * (D + self.d_ubar) ** ((1.0 - self.alpha) * (1.0 - self.CRRA))
        )

    def CRRAutilityPD(self, C, D):
        return (
            (1.0 - self.alpha)
            * (C ** (self.alpha * (1.0 - self.CRRA)))
            * (D + self.d_ubar) ** ((1.0 - self.alpha) * (1.0 - self.CRRA) - 1.0)
        )

    def CRRAutilityP_inv(self, Cmarg, D):
        power = 1.0 / (self.alpha * (1.0 - self.CRRA) - 1.0)
        return (
            Cmarg
            / (
                self.alpha
                * (D + self.d_ubar) ** ((1.0 - self.alpha) * (1.0 - self.CRRA))
            )
        ) ** power

    def CRRAutilityPD_inv(self, Dmarg, D):
        """Invert u_d(c, d) = Dmarg for c, holding d fixed."""
        expo = self.alpha * (1.0 - self.CRRA)
        denom = (1.0 - self.alpha) * (D + self.d_ubar) ** (
            (1.0 - self.alpha) * (1.0 - self.CRRA) - 1.0
        )
        return (Dmarg / denom) ** (1.0 / expo)

    def tau_func(self, x):
        return np.ones_like(np.asarray(x, dtype=float)) * self.tau

    # --------------------------------------------------------------- setup
    @staticmethod
    def _as_scalar(x):
        """First element if ``x`` is a sequence; otherwise ``x`` itself."""
        if isinstance(x, (list, tuple, np.ndarray)):
            return x[0]
        return x

    def _build_inc_shk(
        self,
        labor=None,
        taxrate=None,
        wage=None,
        UnempPrb=None,
        IncUnemp=None,
        PermShkStd=None,
        TranShkStd=None,
    ):
        """Build one ``HANKIncShkDstn_Truncated`` for the given income parameters."""
        labor = self._as_scalar(self.labor if labor is None else labor)
        taxrate = self._as_scalar(self.taxrate if taxrate is None else taxrate)
        wage = self._as_scalar(self.wage if wage is None else wage)
        UnempPrb = self._as_scalar(self.UnempPrb if UnempPrb is None else UnempPrb)
        IncUnemp = self._as_scalar(self.IncUnemp if IncUnemp is None else IncUnemp)
        PermShkStd = self._as_scalar(self.PermShkStd if PermShkStd is None else PermShkStd)
        TranShkStd = self._as_scalar(self.TranShkStd if TranShkStd is None else TranShkStd)
        return HANKIncShkDstn_Truncated(
            sigma_Perm=PermShkStd,
            sigma_Tran=TranShkStd,
            n_approx_Perm=self.PermShkCount if PermShkStd > 0.0 else 1,
            n_approx_Tran=self.TranShkCount if TranShkStd > 0.0 else 1,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            taxrate=taxrate,
            TranShkMean_Func=TranShkMean_Func,
            labor=labor,
            wage=wage,
            truncation=self.truncation,
            neutral_measure=False,
        )

    def _activate_income(self, shk, prob_grid=None):
        """
        Install ``shk`` as the active single-period income distribution used by
        ``_end_of_period`` / ``solve_one_period``. Optionally install
        ``prob_grid`` when ``use_TransitionMatrix`` is True.
        """
        if not hasattr(shk, "pmv"):
            raise TypeError(
                "_activate_income expects a single DiscreteDistribution with .pmv, "
                f"got {type(shk)!r}."
            )
        self.IncShkDstn = shk
        self.ShkPrbsNext = shk.pmv
        self.PermShkValsNext = shk.atoms[0]
        self.TranShkValsNext = shk.atoms[1]
        self.PermShkValsNext_only = np.unique(self.PermShkValsNext)
        self.ShkPrbsNext_PermShk = (
            np.ones_like(self.PermShkValsNext_only) / len(self.PermShkValsNext_only)
        )
        self.TranShkMinNext = float(np.min(self.TranShkValsNext))
        self.PermShkMinNext = float(np.min(self.PermShkValsNext))
        if self.use_TransitionMatrix and prob_grid is not None:
            self.probGrid_income = prob_grid

    def _ensure_income_tran_grids(self, shk):
        """Build ``mNrmGrid_income`` / ``bNrmGrid_income`` sized from ``shk``."""
        PermShkMin = float(np.min(shk.atoms[0]))
        TranShkMax = float(np.max(shk.atoms[1]))
        mNrmMax_income = (
            self.Rfree / (self.PermGroFac * PermShkMin) * self.aXtraMax + TranShkMax
        )
        self.mNrmGrid_income = np.asarray(
            make_grid_exp_mult(0.0001, mNrmMax_income, self.TranMatrix_mCount, timestonest=1)
        ).flatten()
        self.bNrmGrid_income = np.asarray(
            make_grid_exp_mult(0.0, mNrmMax_income - 1.0, self.TranMatrix_bCount, timestonest=1)
        ).flatten()

    def _build_prob_grid_income(
        self,
        shk,
        labor=None,
        taxrate=None,
        wage=None,
        UnempPrb=None,
        IncUnemp=None,
        TranShkStd=None,
    ):
        """
        Build one ``probGrid_income`` lottery for ``shk`` / income scalars.

        Requires ``mNrmGrid_income`` and ``bNrmGrid_income`` to already exist
        (call ``_ensure_income_tran_grids`` first). Grids are held fixed across
        a finite-horizon labor path so only the lottery changes with labor.
        """
        labor = self._as_scalar(self.labor if labor is None else labor)
        taxrate = self._as_scalar(self.taxrate if taxrate is None else taxrate)
        wage = self._as_scalar(self.wage if wage is None else wage)
        UnempPrb = self._as_scalar(self.UnempPrb if UnempPrb is None else UnempPrb)
        IncUnemp = self._as_scalar(self.IncUnemp if IncUnemp is None else IncUnemp)
        TranShkStd = self._as_scalar(self.TranShkStd if TranShkStd is None else TranShkStd)
        TranShkMin = float(np.min(shk.atoms[1]))
        mu_transformed = (
            (1.0 - UnempPrb * IncUnemp) / (1.0 - UnempPrb)
            if UnempPrb < 1.0
            else 1.0
        )
        b_mesh, m_mesh = np.meshgrid(
            self.bNrmGrid_income, self.mNrmGrid_income, indexing="ij"
        )
        diff_grid = m_mesh - b_mesh
        z = TranShkMean_Func(taxrate, labor, wage)
        return update_prob_grid_list_with_trunc(
            1,
            diff_grid,
            np.array([TranShkStd]),
            np.array([UnempPrb]),
            np.array([IncUnemp if UnempPrb > 0 else 1e-12]),
            np.array([mu_transformed]),
            np.array([TranShkMin]),
            np.array([z]),
            self.IncUnemp_Std,
        )[0]

    def update(self):
        """Build grids, shock distributions, income process, and search cost."""
        # Price shocks
        if self.PriceShkStd > 0.0:
            dstn = MeanOneLogNormal(sigma=self.PriceShkStd).discretize(self.PriceShkCount)
            self.PriceShkVals = np.asarray(dstn.atoms[0], dtype=float)
            self.PriceShkWeights = np.asarray(dstn.pmv, dtype=float)
        else:
            self.PriceShkVals = np.array([1.0])
            self.PriceShkWeights = np.array([1.0])
        self.nPrice = len(self.PriceShkVals)

        # State grids
        gtype, nest = self.grid_type, self.NestFac
        self.nNrmGrid = np.unique(
            np.append(0.0, construct_grid(0.001, self.nNrmMax, self.nNrmCount - 1, gtype, nest))
        )
        # add n = 0.5 to the grid (for figure purposes)
        self.nNrmGrid = np.sort(np.unique(np.append(self.nNrmGrid, 0.5)))

        self.mNrmGrid = np.unique(
            np.append(0.0, construct_grid(0.001, self.mNrmMax, self.mNrmCount - 1, gtype, nest))
        )
        self.xNrmGrid = np.unique(
            np.append(0.0, construct_grid(0.001, self.xNrmMax, self.xNrmCount - 1, gtype, nest))
        )
        self.aNrmGrid = np.unique(
            np.append(0.0, construct_grid(0.001, self.aNrmMax, self.aNrmCount - 1, gtype, nest))
        )
        self.dNrmGrid = self.nNrmGrid.copy()

        self.e_grid = np.linspace(0.0, 1.0, self.e_grid_count)
        self.SearchCost_func = LinearFast(
            np.ones_like(self.xNrmGrid) * self.SearchCost, [self.xNrmGrid]
        )

        # Income (single-period SS representation)
        shk = self._build_inc_shk()
        self._activate_income(shk)
        self.IncShkDstn_path = None
        self.probGrid_income_path = None

        if self.use_TransitionMatrix:
            self._ensure_income_tran_grids(shk)
            self.probGrid_income = self._build_prob_grid_income(shk)
        else:
            self.mNrmGrid_income = None
            self.bNrmGrid_income = None
            self.probGrid_income = None

        self._updated = True
        discrete = (
            "search" if self.SearchCost > 0
            else (
                "EV taste" if self.sigma_eps > 0
                else ("hard max + utility cost" if self.kappa != 0.0 else "hard max")
            )
        )
        self._log(
            f"Updated. Discrete: {discrete}"
            + (
                f" (sigma_eps={self.sigma_eps}, kappa={self.kappa})"
                if self.SearchCost == 0 and (self.sigma_eps > 0 or self.kappa != 0.0)
                else ""
            )
            + (
                f" (S={self.SearchCost}, search_method={self.search_method})"
                if self.SearchCost > 0
                else ""
            )
            + f"; price: {'ON' if self.PriceShkStd > 0 else 'OFF'} (nPrice={self.nPrice})"
        )

    def _combine(
        self,
        cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
        cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
        nGrid, mGrid,
    ):
        """
        Discrete combine + price integration.

        Uses directed search when SearchCost > 0 (Combined notebook); otherwise
        EV Type I / hard-max via sigma_eps, with optional utility cost kappa
        (TasteShock / Price / baseline notebooks). kappa=0 is the symmetric logit.
        """
        if self.SearchCost > 0.0:
            return combine_search_price(
                self.PriceShkVals, self.PriceShkWeights,
                cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
                cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
                nGrid, mGrid, self.SearchCost_func, self.e_grid, self.CRRA,
                search_method=self.search_method,
            )
        return combine_ev_taste_price(
            self.PriceShkVals, self.PriceShkWeights,
            cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
            cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
            self.sigma_eps, self.CRRA,
            kappa=self.kappa,
        )

    # ----------------------------------------------------------- terminal
    def solve_terminal(self):
        """Terminal period: keep c=m,d=n; adjust c=αℓ, d=(1-α)ℓ/p; then combine."""
        if not self._updated:
            self.update()

        alpha, CRRA = self.alpha, self.CRRA
        mNrmGrid_T = np.append(self.mNrmGrid, self.mNrmGrid[-1] + 1.0)
        nNrmGrid_T = self.nNrmGrid.copy()
        keep_shape = (len(nNrmGrid_T), len(mNrmGrid_T))

        cFuncKeep_T_arr = np.zeros(keep_shape)
        dFuncKeep_T_arr = np.zeros(keep_shape)
        aFuncKeep_T_arr = np.zeros(keep_shape)
        inv_vFuncKeep_T_arr = np.zeros(keep_shape)
        for i_n, n0 in enumerate(nNrmGrid_T):
            dFuncKeep_T_arr[i_n, :] = n0
            cFuncKeep_T_arr[i_n, :] = mNrmGrid_T
            aFuncKeep_T_arr[i_n, :] = 0.0
            inv_vFuncKeep_T_arr[i_n, :] = self.CRRAutility_inv(
                cFuncKeep_T_arr[i_n, :], dFuncKeep_T_arr[i_n, :]
            )
        vPFuncKeep_T_arr = self.CRRAutilityP(cFuncKeep_T_arr, dFuncKeep_T_arr)
        vPDFuncKeep_T_arr = self.CRRAutilityPD(cFuncKeep_T_arr, dFuncKeep_T_arr)
        cFuncKeep_T = LinearFast(cFuncKeep_T_arr, [nNrmGrid_T, mNrmGrid_T])
        dFuncKeep_T = LinearFast(dFuncKeep_T_arr, [nNrmGrid_T, mNrmGrid_T])

        nPrice = self.nPrice
        PriceShkVals = self.PriceShkVals
        shape_T = (nPrice, len(nNrmGrid_T), len(mNrmGrid_T))
        cFuncAdj_SOL_T = np.zeros(shape_T)
        dFuncAdj_SOL_T = np.zeros(shape_T)
        aFuncAdj_SOL_T = np.zeros(shape_T)
        inv_vFuncAdj_SOL_T = np.zeros(shape_T)
        vPDFuncAdj_SOL_T = np.zeros(shape_T)

        for i_p, p0 in enumerate(PriceShkVals):
            for i_n, n0 in enumerate(nNrmGrid_T):
                n_aux = np.full(len(mNrmGrid_T), n0)
                tau_nm = self.tau_func(n_aux + mNrmGrid_T)
                x_aux = np.clip(mNrmGrid_T + (1.0 - tau_nm) * n_aux, 0.0, None)
                dFuncAdj_SOL_T[i_p, i_n, :] = (1.0 - alpha) * x_aux / p0
                cFuncAdj_SOL_T[i_p, i_n, :] = alpha * x_aux
                aFuncAdj_SOL_T[i_p, i_n, :] = (
                    x_aux - cFuncAdj_SOL_T[i_p, i_n, :] - p0 * dFuncAdj_SOL_T[i_p, i_n, :]
                )
                inv_vFuncAdj_SOL_T[i_p, i_n, :] = self.CRRAutility_inv(
                    cFuncAdj_SOL_T[i_p, i_n, :], dFuncAdj_SOL_T[i_p, i_n, :]
                )
                vPDFuncAdj_SOL_T[i_p, i_n, :] = (1.0 - tau_nm) * self.CRRAutilityP(
                    cFuncAdj_SOL_T[i_p, i_n, :], dFuncAdj_SOL_T[i_p, i_n, :]
                )
        vPFuncAdj_SOL_T = self.CRRAutilityP(cFuncAdj_SOL_T, dFuncAdj_SOL_T)

        if nPrice >= 2:
            cFuncAdj_NM_T = LinearFast(cFuncAdj_SOL_T, [PriceShkVals, nNrmGrid_T, mNrmGrid_T])
            dFuncAdj_NM_T = LinearFast(dFuncAdj_SOL_T, [PriceShkVals, nNrmGrid_T, mNrmGrid_T])
        else:
            cFuncAdj_NM_T = LinearFast(cFuncAdj_SOL_T[0], [nNrmGrid_T, mNrmGrid_T])
            dFuncAdj_NM_T = LinearFast(dFuncAdj_SOL_T[0], [nNrmGrid_T, mNrmGrid_T])

        (
            cFunc_T_arr, dFunc_T_arr, aFunc_T_arr,
            barV_T_arr, vPFunc_T_arr, vPDFunc_T_arr, adjusting_T_arr,
        ) = self._combine(
            cFuncAdj_SOL_T, dFuncAdj_SOL_T, aFuncAdj_SOL_T,
            inv_vFuncAdj_SOL_T, vPFuncAdj_SOL_T, vPDFuncAdj_SOL_T,
            cFuncKeep_T_arr, dFuncKeep_T_arr, aFuncKeep_T_arr,
            inv_vFuncKeep_T_arr, vPFuncKeep_T_arr, vPDFuncKeep_T_arr,
            nNrmGrid_T, mNrmGrid_T,
        )

        inv_vFunc_T_arr = utility_inv(barV_T_arr, CRRA)
        cFunc_T = LinearFast(cFunc_T_arr, [nNrmGrid_T, mNrmGrid_T])
        dFunc_T = LinearFast(dFunc_T_arr, [nNrmGrid_T, mNrmGrid_T])
        inv_vFunc_T = LinearFast(inv_vFunc_T_arr, [nNrmGrid_T, mNrmGrid_T])
        vFunc_T = ValueFuncCRRA(inv_vFunc_T, CRRA)
        vPFunc_T = LinearFast(vPFunc_T_arr, [nNrmGrid_T, mNrmGrid_T])
        vPDFunc_T = LinearFast(vPDFunc_T_arr, [nNrmGrid_T, mNrmGrid_T])
        inv_vPFunc_T = LinearFast(utilityP_inv(vPFunc_T_arr, CRRA), [nNrmGrid_T, mNrmGrid_T])
        inv_vPDFunc_T = LinearFast(utilityP_inv(vPDFunc_T_arr, CRRA), [nNrmGrid_T, mNrmGrid_T])
        adjusting_T = LinearFast(adjusting_T_arr, [nNrmGrid_T, mNrmGrid_T])
        aFunc_T = LinearFast(aFunc_T_arr, [nNrmGrid_T, mNrmGrid_T])

        i_p0 = int(np.argmin(np.abs(PriceShkVals - 1.0)))
        # 1-D adjuster in liquidity (analytical terminal)
        c_1d = alpha * self.xNrmGrid
        d_1d = (1.0 - alpha) * self.xNrmGrid / PriceShkVals[i_p0]
        c_1d_list = [
            LinearFast(alpha * self.xNrmGrid, [self.xNrmGrid])
            for _ in PriceShkVals
        ]
        d_1d_list = [
            LinearFast((1.0 - alpha) * self.xNrmGrid / p0, [self.xNrmGrid])
            for p0 in PriceShkVals
        ]
        inv_vFuncKeep_T = LinearFast(inv_vFuncKeep_T_arr, [nNrmGrid_T, mNrmGrid_T])
        inv_vAdj_1d_list_T = [
            LinearFast(
                self.CRRAutility_inv(
                    alpha * self.xNrmGrid, (1.0 - alpha) * self.xNrmGrid / p0
                ),
                [self.xNrmGrid],
            )
            for p0 in PriceShkVals
        ]
        sol = DurableConsumerSolution(
            cFunc=cFunc_T,
            dFunc=dFunc_T,
            aFunc=aFunc_T,
            vFunc=vFunc_T,
            inv_vFunc=inv_vFunc_T,
            vPFunc=vPFunc_T,
            vPDFunc=vPDFunc_T,
            inv_vPFunc=inv_vPFunc_T,
            inv_vPDFunc=inv_vPDFunc_T,
            adjusting=adjusting_T,
            cFuncKeep=cFuncKeep_T,
            dFuncKeep=dFuncKeep_T,
            cFuncAdj=cFuncAdj_NM_T if nPrice == 1 else LinearFast(cFuncAdj_SOL_T[i_p0], [nNrmGrid_T, mNrmGrid_T]),
            dFuncAdj=dFuncAdj_NM_T if nPrice == 1 else LinearFast(dFuncAdj_SOL_T[i_p0], [nNrmGrid_T, mNrmGrid_T]),
            cFuncAdj_1d=LinearFast(c_1d, [self.xNrmGrid]),
            dFuncAdj_1d=LinearFast(d_1d, [self.xNrmGrid]),
            cFuncAdj_1d_list=c_1d_list,
            dFuncAdj_1d_list=d_1d_list,
            inv_vFuncKeep=inv_vFuncKeep_T,
            inv_vFuncAdj_1d_list=inv_vAdj_1d_list_T,
            adjusting_arr=adjusting_T_arr,
            cFunc_arr=cFunc_T_arr,
            dFunc_arr=dFunc_T_arr,
            nNrmGrid=nNrmGrid_T,
            mNrmGrid=mNrmGrid_T,
        )
        self._log("Terminal combined. Mean E[eps*]:", float(adjusting_T_arr.mean()))
        return sol

    # --------------------------------------------------------------- EoP
    def _end_of_period(self, solution_next):
        """Compute end-of-period marginal values from next-period solution."""
        CRRA = self.CRRA
        inv_vPFuncNext = solution_next.inv_vPFunc
        inv_vPDFuncNext = solution_next.inv_vPDFunc
        inv_vFunc_next = solution_next.inv_vFunc
        aNrmGrid = self.aNrmGrid
        dNrmGrid = self.dNrmGrid
        R_vec = np.full(len(aNrmGrid), self.Rfree)
        post_shape = (len(dNrmGrid), len(aNrmGrid))
        EndOfPrdvPOne = np.zeros(post_shape)
        EndOfPrdvPTwo = np.zeros(post_shape)
        EndOfPrdvPD = np.zeros(post_shape)
        EndOfPrdv = np.zeros(post_shape)
        inv_EndOfPrdv = np.zeros(post_shape)

        if self.use_expected:
            psi = np.asarray(self.IncShkDstn.atoms[0], dtype=float)
            theta = np.asarray(self.IncShkDstn.atoms[1], dtype=float)
            prb = np.asarray(self.IncShkDstn.pmv, dtype=float)
            n_q = ((1.0 - self.dDepr) * dNrmGrid)[None, :, None] / psi[:, None, None]
            m_q = (
                R_vec[None, None, :]
                / (self.PermGroFac * psi[:, None, None])
                * aNrmGrid[None, None, :]
                + theta[:, None, None]
            )
            inv_vp = np.clip(_eval_lf(inv_vPFuncNext, n_q, m_q), 0, None)
            inv_vpd = np.clip(_eval_lf(inv_vPDFuncNext, n_q, m_q), 0, None)
            vp = inv_vp ** (-CRRA)
            vpd = inv_vpd ** (-CRRA)
            v = utility_inv(_eval_lf(inv_vFunc_next, n_q, m_q), CRRA)
            w_p = (prb * psi ** (-CRRA))[:, None, None]
            w_v = (prb * psi ** (1.0 - CRRA) * self.PermGroFac ** (1.0 - CRRA))[:, None, None]
            e_vp = np.sum(w_p * vp, axis=0)
            e_vpd = np.sum(w_p * vpd, axis=0)
            e_v = np.sum(w_v * v, axis=0)
            fac = self.DiscFac * self.LivPrb * self.PermGroFac ** (-CRRA)
            EndOfPrdvPOne = fac * R_vec * e_vp
            EndOfPrdvPTwo = fac * (R_vec + self.OpCost + self.MaintCost * self.dDepr) * e_vp
            EndOfPrdvPD = fac * (1.0 - self.dDepr * (1.0 - self.MaintCost)) * e_vpd
            EndOfPrdv = self.DiscFac * self.LivPrb * e_v
            inv_EndOfPrdv = utility_inv(EndOfPrdv, CRRA)
        else:
            nNrmGridNow = dNrmGrid
            _n2d = np.repeat(nNrmGridNow[:, np.newaxis], len(self.mNrmGrid_income), axis=1)
            _m2d = np.tile(self.mNrmGrid_income[np.newaxis, :], (len(nNrmGridNow), 1))

            inv_vPFuncNext_array = np.clip(_eval_lf(inv_vPFuncNext, _n2d, _m2d), 0, None)
            inv_vPDFuncNext_array = np.clip(_eval_lf(inv_vPDFuncNext, _n2d, _m2d), 0, None)
            inv_vFuncNext_array = np.clip(_eval_lf(inv_vFunc_next, _n2d, _m2d), 0, None)

            vFuncNext_array = utility(inv_vFuncNext_array, CRRA)
            vPDFuncNext_array = inv_vPDFuncNext_array ** (-CRRA)
            vPFuncNext_array = inv_vPFuncNext_array ** (-CRRA)
            vPDFuncNext_array[np.isinf(vPDFuncNext_array)] = 1e16
            vPFuncNext_array[np.isinf(vPFuncNext_array)] = 1e16
            vFuncNext_array[np.isinf(vFuncNext_array)] = -1e16

            Interm_vP_array = np.dot(self.probGrid_income, vPFuncNext_array.T)
            Interm_vPD_array = np.dot(self.probGrid_income, vPDFuncNext_array.T)
            Interm_v_array = np.dot(self.probGrid_income, vFuncNext_array.T)

            Interm_inv_vP_array = Interm_vP_array ** (-1.0 / CRRA)
            Interm_inv_vPD_array = Interm_vPD_array ** (-1.0 / CRRA)
            Interm_inv_v_array = utility_inv(Interm_v_array, CRRA)

            Interm_inv_vP_Interp_func = LinearFast(
                Interm_inv_vP_array, [self.bNrmGrid_income, nNrmGridNow]
            )
            Interm_inv_vPD_Interp_func = LinearFast(
                Interm_inv_vPD_array, [self.bNrmGrid_income, nNrmGridNow]
            )
            Interm_inv_v_Interp_func = LinearFast(
                Interm_inv_v_array, [self.bNrmGrid_income, nNrmGridNow]
            )

            BoroCnstNat2D_b = np.array([0.0 - self.TranShkMinNext])
            bNrmGrid_incomeNew = np.unique(
                np.sort(np.append(self.bNrmGrid_income, BoroCnstNat2D_b))
            )
            Interm_inv_vP_withzero_array = np.zeros(
                (len(bNrmGrid_incomeNew), len(nNrmGridNow))
            )
            Interm_inv_vPD_withzero_array = np.zeros_like(Interm_inv_vP_withzero_array)
            Interm_inv_v_withzero_array = np.zeros_like(Interm_inv_vP_withzero_array)

            idx = int(np.where(bNrmGrid_incomeNew >= BoroCnstNat2D_b[0])[0][0])
            b_slice = bNrmGrid_incomeNew[idx + 1 :]
            if b_slice.size:
                b_2d = np.repeat(b_slice[:, None], len(nNrmGridNow), axis=1)
                n_2d = np.repeat(nNrmGridNow[None, :], len(b_slice), axis=0)
                Interm_inv_vP_withzero_array[idx + 1 :, :] = _eval_lf(
                    Interm_inv_vP_Interp_func, b_2d, n_2d
                )
                Interm_inv_vPD_withzero_array[idx + 1 :, :] = _eval_lf(
                    Interm_inv_vPD_Interp_func, b_2d, n_2d
                )
                Interm_inv_v_withzero_array[idx + 1 :, :] = _eval_lf(
                    Interm_inv_v_Interp_func, b_2d, n_2d
                )

            Interm_inv_vP_Interp_func = LinearFast(
                Interm_inv_vP_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow]
            )
            Interm_inv_vPD_Interp_func = LinearFast(
                Interm_inv_vPD_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow]
            )
            Interm_inv_v_Interp_func = LinearFast(
                Interm_inv_v_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow]
            )

            nD, nSh, nA = len(dNrmGrid), len(self.PermShkValsNext_only), len(aNrmGrid)
            _base_w = (
                self.ShkPrbsNext_PermShk
                * self.PermShkValsNext_only ** (-CRRA)
                * self.PermGroFac ** (-CRRA)
                * self.DiscFac
                * self.LivPrb
            )
            _base_wv = (
                self.ShkPrbsNext_PermShk
                * self.PermShkValsNext_only ** (1.0 - CRRA)
                * self.PermGroFac ** (1.0 - CRRA)
                * self.DiscFac
                * self.LivPrb
            )
            _d_surv = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * dNrmGrid
            _d_lam = dNrmGrid * (self.OpCost + self.dDepr * self.MaintCost)

            n_next_3d = (
                _d_surv[:, np.newaxis, np.newaxis]
                / self.PermShkValsNext_only[np.newaxis, :, np.newaxis]
            ) * np.ones((1, 1, nA))
            b_next_3d = (
                (
                    R_vec[np.newaxis, np.newaxis, :] * aNrmGrid[np.newaxis, np.newaxis, :]
                    - _d_lam[:, np.newaxis, np.newaxis]
                )
                / (self.PermGroFac * self.PermShkValsNext_only[np.newaxis, :, np.newaxis])
            )
            b_flat = b_next_3d.ravel()
            n_flat = n_next_3d.ravel()
            shp3 = (nD, nSh, nA)

            vP_3d = _eval_lf(Interm_inv_vP_Interp_func, b_flat, n_flat).reshape(shp3) ** (-CRRA)
            vPD_3d = _eval_lf(Interm_inv_vPD_Interp_func, b_flat, n_flat).reshape(shp3) ** (-CRRA)
            v_3d = utility(
                np.clip(_eval_lf(Interm_inv_v_Interp_func, b_flat, n_flat), 0.0, None), CRRA
            ).reshape(shp3)

            w = _base_w[np.newaxis, :, np.newaxis]
            wv = _base_wv[np.newaxis, :, np.newaxis]
            EndOfPrdvPOne = np.sum(w * R_vec[np.newaxis, np.newaxis, :] * vP_3d, axis=1)
            EndOfPrdvPTwo = np.sum(
                w
                * (R_vec + self.OpCost + self.MaintCost * self.dDepr)[np.newaxis, np.newaxis, :]
                * vP_3d,
                axis=1,
            )
            EndOfPrdvPD = np.sum(
                w * (1.0 - self.dDepr * (1.0 - self.MaintCost)) * vPD_3d, axis=1
            )
            EndOfPrdv = np.sum(wv * v_3d, axis=1)
            inv_EndOfPrdv = utility_inv(EndOfPrdv, CRRA)

        EndOfPrdvPOne[np.isnan(EndOfPrdvPOne)] = np.inf
        EndOfPrdvPTwo[np.isnan(EndOfPrdvPTwo)] = np.inf
        EndOfPrdvPD[np.isnan(EndOfPrdvPD)] = np.inf
        inv_EndOfPrdv = np.nan_to_num(inv_EndOfPrdv, nan=0.0, posinf=0.0, neginf=0.0)

        return dict(
            EndOfPrdvPOne=EndOfPrdvPOne,
            EndOfPrdvPTwo=EndOfPrdvPTwo,
            EndOfPrdvPD=EndOfPrdvPD,
            EndOfPrdv=EndOfPrdv,
            inv_EndOfPrdv=inv_EndOfPrdv,
            EndOfPrdvPOneFunc=LinearFast(EndOfPrdvPOne, [dNrmGrid, aNrmGrid]),
            EndOfPrdvPTwoFunc=LinearFast(EndOfPrdvPTwo, [dNrmGrid, aNrmGrid]),
            EndOfPrdvPDFunc=LinearFast(EndOfPrdvPD, [dNrmGrid, aNrmGrid]),
            inv_EndOfPrdvFunc=LinearFast(inv_EndOfPrdv, [dNrmGrid, aNrmGrid]),
            EndOfPrdvFunc=ValueFuncCRRA(
                LinearFast(inv_EndOfPrdv, [dNrmGrid, aNrmGrid]), CRRA
            ),
            R_vec=R_vec,
        )

    # ------------------------------------------------------------- keeper
    def _solve_keeper(self, eop):
        alpha, CRRA, d_ubar = self.alpha, self.CRRA, self.d_ubar
        aNrmGrid, dNrmGrid = self.aNrmGrid, self.dNrmGrid
        nNrmGrid, mNrmGrid = self.nNrmGrid, self.mNrmGrid
        EndOfPrdvPOne = eop["EndOfPrdvPOne"]
        EndOfPrdv = eop["EndOfPrdv"]
        EndOfPrdvFunc = eop["EndOfPrdvFunc"]
        EndOfPrdvPDFunc = eop["EndOfPrdvPDFunc"]
        BoroCnstNat = 0.0

        cFuncKeepUnc_arr, n_dec_arr, n_tot_arr = _fast_keeper_rows(
            np.ascontiguousarray(dNrmGrid, dtype=np.float64),
            np.ascontiguousarray(aNrmGrid, dtype=np.float64),
            np.ascontiguousarray(mNrmGrid, dtype=np.float64),
            np.ascontiguousarray(EndOfPrdvPOne, dtype=np.float64),
            np.ascontiguousarray(EndOfPrdv, dtype=np.float64),
            alpha, d_ubar, CRRA, BoroCnstNat,
        )
        decreasing_segments_counter = int(n_dec_arr.sum())
        segments_total_counter = int(n_tot_arr.sum())

        keep_shape = (len(nNrmGrid), len(mNrmGrid))
        dFuncKeepUnc_arr = np.repeat(nNrmGrid[:, None], len(mNrmGrid), axis=1)

        cFuncKeepUnc = LinearFast(cFuncKeepUnc_arr, [nNrmGrid, mNrmGrid])
        dFuncKeepUnc = LinearFast(dFuncKeepUnc_arr, [nNrmGrid, mNrmGrid])
        cFuncKeepCnst = LinearFast(
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        )
        cFuncKeep = LowerEnvelope2D(cFuncKeepUnc, cFuncKeepCnst, nan_bool=False)
        dFuncKeep = dFuncKeepUnc

        n_2d = dFuncKeepUnc_arr
        m_2d = np.tile(mNrmGrid, (len(nNrmGrid), 1))
        cKeep_arr = np.clip(cFuncKeep(n_2d, m_2d), 0.0, None)
        aFuncKeep_arr = np.maximum(m_2d - cKeep_arr, BoroCnstNat)
        inv_vFuncKeep_arr = utility_inv(
            self.CRRAutility(cKeep_arr, n_2d) + EndOfPrdvFunc(n_2d, aFuncKeep_arr), CRRA
        )
        vPDFuncKeep_arr = self.CRRAutilityPD(cKeep_arr, n_2d) + EndOfPrdvPDFunc(
            n_2d, aFuncKeep_arr
        )

        vPFuncKeep_arr = self.CRRAutilityP(cKeep_arr, dFuncKeepUnc_arr)
        inv_vFuncKeep = LinearFast(inv_vFuncKeep_arr, [nNrmGrid, mNrmGrid])
        vFuncKeep = ValueFuncCRRA(inv_vFuncKeep, CRRA)
        aFuncKeep = LinearFast(aFuncKeep_arr, [nNrmGrid, mNrmGrid])
        vPDFuncKeep = LinearFast(vPDFuncKeep_arr, [nNrmGrid, mNrmGrid])
        vPFuncKeep = MargValueFuncCRRA_dur(
            cFuncKeep, dFuncKeep, alpha, d_ubar, CRRA
        )

        keep_share_decreasing = (
            decreasing_segments_counter / segments_total_counter
            if segments_total_counter
            else float("nan")
        )
        return dict(
            cFuncKeep=cFuncKeep,
            dFuncKeep=dFuncKeep,
            aFuncKeep=aFuncKeep,
            aFuncKeep_arr=aFuncKeep_arr,
            cKeep_arr=cKeep_arr,
            dKeep_arr=dFuncKeepUnc_arr,
            inv_vFuncKeep=inv_vFuncKeep,
            inv_vFuncKeep_arr=inv_vFuncKeep_arr,
            vFuncKeep=vFuncKeep,
            vPFuncKeep=vPFuncKeep,
            vPFuncKeep_arr=vPFuncKeep_arr,
            vPDFuncKeep=vPDFuncKeep,
            vPDFuncKeep_arr=vPDFuncKeep_arr,
            keep_shape=keep_shape,
            keep_decreasing_segments=int(decreasing_segments_counter),
            keep_segments_total=int(segments_total_counter),
            keep_share_decreasing=float(keep_share_decreasing),
        )

    # ----------------------------------------------------------- adjuster
    def _solve_adjuster(self, eop):
        alpha, CRRA, d_ubar = self.alpha, self.CRRA, self.d_ubar
        aNrmGrid = self.aNrmGrid
        dNrmGrid = self.dNrmGrid
        nNrmGrid, mNrmGrid = self.nNrmGrid, self.mNrmGrid
        EndOfPrdvPOneFunc = eop["EndOfPrdvPOneFunc"]
        EndOfPrdvPDFunc = eop["EndOfPrdvPDFunc"]
        EndOfPrdvFunc = eop["EndOfPrdvFunc"]
        PriceShkVals = self.PriceShkVals
        nPrice = self.nPrice
        a_min, a_max = float(aNrmGrid[0]), float(aNrmGrid[-1])
        N_scan = self.N_scan
        dNrmGridAdj = np.asarray(
            construct_grid(
                dNrmGrid[0], dNrmGrid[-1], len(self.xNrmGrid), self.grid_type, self.NestFac
            ),
            dtype=float,
        )
        BoroCnstNat1D = 0.0
        a_scan = np.asarray(construct_grid(0.0, a_max, N_scan, "exp_mult", 1), dtype=float) + a_min
        gpu_ctx = (
            _gpu_foc_ctx(
                dNrmGridAdj, a_scan, dNrmGrid, aNrmGrid,
                eop["EndOfPrdvPOne"], eop["EndOfPrdvPD"],
            )
            if _USE_GPU_FOC else None
        )

        cFuncAdj_list, dFuncAdj_list, aFuncAdj_list = [], [], []
        inv_vFuncAdj_1d_list = []
        # Pre-/post-FUES clouds (one entry per price)
        xAdj_pre_list, cAdj_pre_list, dAdj_pre_list = [], [], []
        aAdj_pre_list, vAdj_pre_list = [], []
        xAdj_post_list, cAdj_post_list, dAdj_post_list = [], [], []
        aAdj_post_list, vAdj_post_list = [], []
        adj_removedpoints_counter = 0
        adj_totalpoints_counter = 0

        for i_p, p0 in enumerate(PriceShkVals):
            if gpu_ctx is not None:
                a_root, d_0 = _gpu_foc_from_ctx(gpu_ctx, p0, alpha, d_ubar, CRRA)
            else:
                a_root, d_0 = _cpu_foc_roots(
                    dNrmGridAdj, a_scan, dNrmGrid, aNrmGrid,
                    eop["EndOfPrdvPOne"], eop["EndOfPrdvPD"],
                    p0, alpha, d_ubar, CRRA,
                )

            if a_root.size:
                c_0 = CRRAutilityP_inv_njit(
                    EndOfPrdvPOneFunc(d_0, a_root), d_0, alpha, d_ubar, CRRA
                )
                x_0 = a_root + p0 * d_0 + c_0
                v_0 = self.CRRAutility(c_0, d_0) + EndOfPrdvFunc(d_0, a_root)
            else:
                a_root = c_0 = x_0 = d_0 = v_0 = np.array([])

            aAdj_array, cAdj_array, xAdj_array, dAdj_array, vAdj_array = (
                a_root, c_0, x_0, d_0, v_0
            )
            self._log(f"p[{i_p}]={p0:.4f}: FOC roots = {len(xAdj_array)}")

            def _static_adj_on_x(price):
                """Fallback when FOC/FUES yields no usable points."""
                x_u = np.unique(np.append(BoroCnstNat1D, self.xNrmGrid))
                c_u = alpha * x_u
                d_u = (1.0 - alpha) * x_u / price
                a_u = np.maximum(x_u - c_u - price * d_u, 0.0)
                v_u = self.CRRAutility(c_u, d_u) + EndOfPrdvFunc(d_u, a_u)
                inv_u = utility_inv(v_u, CRRA)
                inv_u = np.nan_to_num(inv_u, nan=0.0, posinf=0.0, neginf=0.0)
                return x_u, c_u, d_u, a_u, v_u, inv_u

            # Sort pre-FUES cloud by liquidity (also the stored pre arrays)
            if xAdj_array.size > 0:
                order = np.argsort(xAdj_array)
                xAdj_array = xAdj_array[order]
                vAdj_array = vAdj_array[order]
                cAdj_array = cAdj_array[order]
                dAdj_array = dAdj_array[order]
                aAdj_array = aAdj_array[order]

            xAdj_pre_list.append(np.asarray(xAdj_array, dtype=float).copy())
            cAdj_pre_list.append(np.asarray(cAdj_array, dtype=float).copy())
            dAdj_pre_list.append(np.asarray(dAdj_array, dtype=float).copy())
            aAdj_pre_list.append(np.asarray(aAdj_array, dtype=float).copy())
            vAdj_pre_list.append(np.asarray(vAdj_array, dtype=float).copy())

            use_fallback = xAdj_array.size < 2
            if not use_fallback:
                xAdj_upper, vAdj_upper, cAdj_upper, dAdj_upper, aAdj_upper, _ = FUES_allPol(
                    xAdj_array, vAdj_array, cAdj_array, dAdj_array, aAdj_array, b=BoroCnstNat1D
                )
                use_fallback = len(xAdj_upper) == 0
                if not use_fallback:
                    adj_removedpoints_counter += len(xAdj_array) - len(xAdj_upper)
                    adj_totalpoints_counter += len(xAdj_array)

            if use_fallback:
                self._log(f"  fallback static adjuster for p={p0:.4f}")
                x_upper, c_upper, d_upper, a_upper, v_upper, inv_v_upper = _static_adj_on_x(p0)
                # Post-FUES = fallback policy grid
                xAdj_post_list.append(np.asarray(x_upper, dtype=float).copy())
                cAdj_post_list.append(np.asarray(c_upper, dtype=float).copy())
                dAdj_post_list.append(np.asarray(d_upper, dtype=float).copy())
                aAdj_post_list.append(np.asarray(a_upper, dtype=float).copy())
                vAdj_post_list.append(np.asarray(v_upper, dtype=float).copy())
            else:
                # Raw FUES output (before boundary prepend / tail trim)
                xAdj_post_list.append(np.asarray(xAdj_upper, dtype=float).copy())
                cAdj_post_list.append(np.asarray(cAdj_upper, dtype=float).copy())
                dAdj_post_list.append(np.asarray(dAdj_upper, dtype=float).copy())
                aAdj_post_list.append(np.asarray(aAdj_upper, dtype=float).copy())
                vAdj_post_list.append(np.asarray(vAdj_upper, dtype=float).copy())

                inv_vAdj_upper = utility_inv(vAdj_upper, CRRA)
                xAdj_upper[0] = np.round(xAdj_upper[0], decimals=3)
                dAdj_upper[0] = np.round(dAdj_upper[0], decimals=3)
                aAdj_upper[0] = np.round(aAdj_upper[0], decimals=3)
                cAdj_upper[0] = np.round(cAdj_upper[0], decimals=3)
                inv_vAdj_upper[0] = np.round(inv_vAdj_upper[0], decimals=3)

                x_upper = np.append(BoroCnstNat1D, xAdj_upper)
                d_upper = np.append(0.0, dAdj_upper)
                c_upper = np.append(0.0, cAdj_upper)
                a_upper = np.append(BoroCnstNat1D, aAdj_upper)
                inv_v_upper = np.append(0.0, inv_vAdj_upper)

                while len(x_upper) > 2 and (
                    c_upper[-1] < c_upper[-2]
                    or d_upper[-1] < d_upper[-2]
                    or inv_v_upper[-1] < inv_v_upper[-2]
                ):
                    x_upper, c_upper, d_upper, a_upper, inv_v_upper = (
                        x_upper[:-1],
                        c_upper[:-1],
                        d_upper[:-1],
                        a_upper[:-1],
                        inv_v_upper[:-1],
                    )

            cFuncAdj_p = LinearFast(c_upper, [x_upper])
            dFuncAdj_p = LinearFast(d_upper, [x_upper])
            aFuncAdj_p = LinearFast(a_upper, [x_upper])
            cFuncAdj_list.append(cFuncAdj_p)
            dFuncAdj_list.append(dFuncAdj_p)
            aFuncAdj_list.append(aFuncAdj_p)

            # Recompute inv_v on xNrmGrid: linear interp of inv_v between 0 and
            # the first FUES node poorly approximates the nonlinear transform.
            c_aux = np.maximum(cFuncAdj_p(self.xNrmGrid), 0.0)
            d_aux = np.maximum(dFuncAdj_p(self.xNrmGrid), 0.0)
            a_aux = self.xNrmGrid - c_aux - p0 * d_aux
            vAdj_aux = self.CRRAutility(c_aux, d_aux) + EndOfPrdvFunc(d_aux, a_aux)
            inv_vAdj_aux_array = utility_inv(vAdj_aux, CRRA)
            inv_vAdj_aux_array = np.nan_to_num(
                inv_vAdj_aux_array, nan=0.0, posinf=0.0, neginf=0.0
            )
            inv_vFuncAdj_1d_list.append(LinearFast(inv_vAdj_aux_array, [self.xNrmGrid]))

        n_2d = nNrmGrid[:, None]
        m_2d = mNrmGrid[None, :]
        tau_nm = self.tau_func(n_2d + m_2d)
        x_aux = np.clip(m_2d + (1.0 - tau_nm) * n_2d, 0.0, None)
        x_flat = x_aux.ravel()
        shp_nm = x_aux.shape

        cFuncAdj_NM_arr = np.empty((nPrice,) + shp_nm)
        dFuncAdj_NM_arr = np.empty_like(cFuncAdj_NM_arr)
        aFuncAdj_NM_arr = np.empty_like(cFuncAdj_NM_arr)
        inv_vFuncAdj_NM_arr = np.empty_like(cFuncAdj_NM_arr)
        vPFuncAdj_NM_arr = np.empty_like(cFuncAdj_NM_arr)
        vPDFuncAdj_NM_arr = np.empty_like(cFuncAdj_NM_arr)

        for i_p, p0 in enumerate(PriceShkVals):
            c_i = np.maximum(cFuncAdj_list[i_p](x_flat), 0.0).reshape(shp_nm)
            d_i = np.maximum(dFuncAdj_list[i_p](x_flat), 0.0).reshape(shp_nm)
            cFuncAdj_NM_arr[i_p] = c_i
            dFuncAdj_NM_arr[i_p] = d_i
            aFuncAdj_NM_arr[i_p] = np.maximum(x_aux - c_i - p0 * d_i, 0.0)
            inv_vFuncAdj_NM_arr[i_p] = np.asarray(
                inv_vFuncAdj_1d_list[i_p](x_flat), dtype=float
            ).reshape(shp_nm)
            vPFuncAdj_NM_arr[i_p] = self.CRRAutilityP(c_i, d_i)
            vPDFuncAdj_NM_arr[i_p] = (1.0 - tau_nm) * vPFuncAdj_NM_arr[i_p]

        i_p0 = int(np.argmin(np.abs(PriceShkVals - 1.0)))
        cFuncAdj_NM = LinearFast(cFuncAdj_NM_arr[i_p0], [nNrmGrid, mNrmGrid])
        dFuncAdj_NM = LinearFast(dFuncAdj_NM_arr[i_p0], [nNrmGrid, mNrmGrid])

        adj_share_removed = (
            adj_removedpoints_counter / adj_totalpoints_counter
            if adj_totalpoints_counter
            else float("nan")
        )

        # Diagnostic plot of the 1-D adjuster policies (price ~= 1), one per period.
        if getattr(self, "diagnostics", False):
            self._plot_adjuster_1d(cFuncAdj_list, dFuncAdj_list, i_p0)
            print("length of adjuster: ", len(x_upper))
            print("min and max c: ", np.min(c_upper), np.max(c_upper))
            print("min and max d: ", np.min(d_upper), np.max(d_upper))

        return dict(
            cFuncAdj_NM_arr=cFuncAdj_NM_arr,
            dFuncAdj_NM_arr=dFuncAdj_NM_arr,
            aFuncAdj_NM_arr=aFuncAdj_NM_arr,
            inv_vFuncAdj_NM_arr=inv_vFuncAdj_NM_arr,
            vPFuncAdj_NM_arr=vPFuncAdj_NM_arr,
            vPDFuncAdj_NM_arr=vPDFuncAdj_NM_arr,
            cFuncAdj=cFuncAdj_NM,
            dFuncAdj=dFuncAdj_NM,
            cFuncAdj_list=cFuncAdj_list,
            dFuncAdj_list=dFuncAdj_list,
            inv_vFuncAdj_1d_list=inv_vFuncAdj_1d_list,
            xAdj_pre_list=xAdj_pre_list,
            cAdj_pre_list=cAdj_pre_list,
            dAdj_pre_list=dAdj_pre_list,
            aAdj_pre_list=aAdj_pre_list,
            vAdj_pre_list=vAdj_pre_list,
            xAdj_post_list=xAdj_post_list,
            cAdj_post_list=cAdj_post_list,
            dAdj_post_list=dAdj_post_list,
            aAdj_post_list=aAdj_post_list,
            vAdj_post_list=vAdj_post_list,
            i_p0=i_p0,
            adj_kinks_removed=int(adj_removedpoints_counter),
            adj_points_total=int(adj_totalpoints_counter),
            adj_share_removed=float(adj_share_removed),
        )

    # ------------------------------------------------------ diagnostics
    def _plot_adjuster_1d(self, cFuncAdj_list, dFuncAdj_list, i_p0):
        """
        Diagnostic plot of the 1-D adjuster policies at the representative price
        (``p ~= 1``, index ``i_p0``): consumption ``cFuncAdj`` and durable
        ``dFuncAdj`` as functions of adjuster liquidity ``x`` (``self.xNrmGrid``).

        Called once per period/iteration when ``self.diagnostics`` is True.
        """
        x = self.xNrmGrid
        cAdj = np.maximum(cFuncAdj_list[i_p0](x), 0.0)
        dAdj = np.maximum(dFuncAdj_list[i_p0](x), 0.0)

        self._diag_period = getattr(self, "_diag_period", 0) + 1

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(x, cAdj)
        axes[0].set_title("cFuncAdj (1D)")
        axes[0].set_xlabel("x (adjuster liquidity)")
        axes[0].set_ylabel("c")
        axes[1].plot(x, dAdj)
        axes[1].set_title("dFuncAdj (1D)")
        axes[1].set_xlabel("x (adjuster liquidity)")
        axes[1].set_ylabel("d")
        fig.suptitle(f"Adjuster 1D policies - period/iter {self._diag_period}")
        fig.tight_layout()
        plt.show()

    # -------------------------------------------------------- one period
    def solve_one_period(self, solution_next):
        """Solve one period given the next-period solution."""
        eop = self._end_of_period(solution_next)
        keep = self._solve_keeper(eop)
        adj = self._solve_adjuster(eop)

        (
            cFunc_arr, dFunc_arr, aFunc_arr,
            barV_arr, vPFunc_arr, vPDFunc_arr, adjusting_arr,
        ) = self._combine(
            adj["cFuncAdj_NM_arr"], adj["dFuncAdj_NM_arr"], adj["aFuncAdj_NM_arr"],
            adj["inv_vFuncAdj_NM_arr"], adj["vPFuncAdj_NM_arr"], adj["vPDFuncAdj_NM_arr"],
            keep["cKeep_arr"], keep["dKeep_arr"], keep["aFuncKeep_arr"],
            keep["inv_vFuncKeep_arr"], keep["vPFuncKeep_arr"], keep["vPDFuncKeep_arr"],
            self.nNrmGrid, self.mNrmGrid,
        )

        inv_vFunc_arr = utility_inv(barV_arr, self.CRRA)
        cFunc = LinearFast(cFunc_arr, [self.nNrmGrid, self.mNrmGrid])
        dFunc = LinearFast(dFunc_arr, [self.nNrmGrid, self.mNrmGrid])
        aFunc = LinearFast(aFunc_arr, [self.nNrmGrid, self.mNrmGrid])
        inv_vFunc = LinearFast(inv_vFunc_arr, [self.nNrmGrid, self.mNrmGrid])
        vFunc = ValueFuncCRRA(inv_vFunc, self.CRRA)
        vPFunc = LinearFast(vPFunc_arr, [self.nNrmGrid, self.mNrmGrid])
        vPDFunc = LinearFast(vPDFunc_arr, [self.nNrmGrid, self.mNrmGrid])
        inv_vPFunc = LinearFast(utilityP_inv(vPFunc_arr, self.CRRA), [self.nNrmGrid, self.mNrmGrid])
        inv_vPDFunc = LinearFast(
            utilityP_inv(vPDFunc_arr, self.CRRA), [self.nNrmGrid, self.mNrmGrid]
        )
        adjusting = LinearFast(adjusting_arr, [self.nNrmGrid, self.mNrmGrid])

        # Diagnostic
        if getattr(self, "diagnostics", False):
            print("min and max inv_vPFunc: ", np.min(vPFunc_arr), np.max(vPFunc_arr))
            print("min and max inv_vPDFunc: ", np.min(vPDFunc_arr), np.max(vPDFunc_arr))
            print("Average adjuster probability: ", float(adjusting_arr.mean()))
        self._log("Period combined. Mean E[eps*]:", float(adjusting_arr.mean()))
        i_p0 = int(np.argmin(np.abs(self.PriceShkVals - 1.0)))
        return DurableConsumerSolution(
            cFunc=cFunc,
            dFunc=dFunc,
            aFunc=aFunc,
            vFunc=vFunc,
            inv_vFunc=inv_vFunc,
            vPFunc=vPFunc,
            vPDFunc=vPDFunc,
            inv_vPFunc=inv_vPFunc,
            inv_vPDFunc=inv_vPDFunc,
            adjusting=adjusting,
            cFuncKeep=keep["cFuncKeep"],
            dFuncKeep=keep["dFuncKeep"],
            cFuncAdj=adj["cFuncAdj"],
            dFuncAdj=adj["dFuncAdj"],
            cFuncAdj_1d=adj["cFuncAdj_list"][i_p0],
            dFuncAdj_1d=adj["dFuncAdj_list"][i_p0],
            cFuncAdj_1d_list=adj["cFuncAdj_list"],
            dFuncAdj_1d_list=adj["dFuncAdj_list"],
            inv_vFuncKeep=keep["inv_vFuncKeep"],
            inv_vFuncAdj_1d_list=adj["inv_vFuncAdj_1d_list"],
            adjusting_arr=adjusting_arr,
            cFunc_arr=cFunc_arr,
            dFunc_arr=dFunc_arr,
            nNrmGrid=self.nNrmGrid,
            mNrmGrid=self.mNrmGrid,
            xAdj_pre_list=adj["xAdj_pre_list"],
            cAdj_pre_list=adj["cAdj_pre_list"],
            dAdj_pre_list=adj["dAdj_pre_list"],
            aAdj_pre_list=adj["aAdj_pre_list"],
            vAdj_pre_list=adj["vAdj_pre_list"],
            xAdj_post_list=adj["xAdj_post_list"],
            cAdj_post_list=adj["cAdj_post_list"],
            dAdj_post_list=adj["dAdj_post_list"],
            aAdj_post_list=adj["aAdj_post_list"],
            vAdj_post_list=adj["vAdj_post_list"],
            xAdj_pre=adj["xAdj_pre_list"][i_p0],
            cAdj_pre=adj["cAdj_pre_list"][i_p0],
            dAdj_pre=adj["dAdj_pre_list"][i_p0],
            aAdj_pre=adj["aAdj_pre_list"][i_p0],
            vAdj_pre=adj["vAdj_pre_list"][i_p0],
            xAdj_post=adj["xAdj_post_list"][i_p0],
            cAdj_post=adj["cAdj_post_list"][i_p0],
            dAdj_post=adj["dAdj_post_list"][i_p0],
            aAdj_post=adj["aAdj_post_list"][i_p0],
            vAdj_post=adj["vAdj_post_list"][i_p0],
            keep_decreasing_segments=keep["keep_decreasing_segments"],
            keep_segments_total=keep["keep_segments_total"],
            keep_share_decreasing=keep["keep_share_decreasing"],
            adj_kinks_removed=adj["adj_kinks_removed"],
            adj_points_total=adj["adj_points_total"],
            adj_share_removed=adj["adj_share_removed"],
        )

    # ----------------------------------------------------- distance metric
    def _criterion_diffs(self, sol_a, sol_b):
        """
        Return the element-wise differences of the two solutions' value grids
        for every entry in ``distance_criteria``, as a single flat array.

        Each criterion (``inv_vPFunc``, ``inv_vPDFunc``, ...) is a ``LinearFast``
        interpolant; its underlying value grid lives in ``.f_val``. Plain arrays
        are compared directly. The grids themselves are fixed across iterations
        and are intentionally not pooled into the metric.

        The criterion arrays are ``utilityP_inv(vP, CRRA) = vP ** (-1/CRRA)``,
        which is ``+inf`` wherever marginal value is exactly zero. To avoid
        ``inf - inf = nan`` (which would poison the mean/max), only points where
        both arrays are finite are kept.
        """
        diffs = []
        for name in sol_a.distance_criteria:
            obj_a = getattr(sol_a, name)
            obj_b = getattr(sol_b, name)
            arr_a = np.asarray(getattr(obj_a, "f_val", obj_a), dtype=float)
            arr_b = np.asarray(getattr(obj_b, "f_val", obj_b), dtype=float)
            if arr_a.shape != arr_b.shape:
                return None  # shapes disagree -> treat as "very far apart"
            finite = np.isfinite(arr_a) & np.isfinite(arr_b)
            diffs.append((arr_a[finite] - arr_b[finite]).ravel())
        diffs = [d for d in diffs if d.size > 0]
        if not diffs:
            return np.array([0.0])
        return np.concatenate(diffs)

    def _solution_distance(self, sol, prev):
        """
        Scalar convergence distance between consecutive solutions, aggregated
        according to ``self.distance_metric`` ("max", "mse", or "rmse").

        The "max" option reproduces HARK's default ``sol.distance(prev)``.
        """
        metric = str(getattr(self, "distance_metric", "max")).lower()
        diffs = self._criterion_diffs(sol, prev)
        if diffs is None:
            return 1000.0  # mirrors HARK's failsafe for shape mismatches
        if metric == "max":
            return float(np.max(np.abs(diffs)))
        elif metric == "mse":
            return float(np.mean(diffs ** 2))
        elif metric == "rmse":
            return float(np.sqrt(np.mean(diffs ** 2)))
        raise ValueError(
            f"Unknown distance_metric {metric!r}; choose 'max', 'mse', or 'rmse'."
        )

    # ---------------------------------------------------------------- solve
    def solve(self):
        """
        Solve the model.

        Finite horizon (``cycles >= 1``): sets ``self.solution`` to a list of
        length ``T_cycle + 1``, where ``solution[t]`` is period t (0 = first) and
        ``solution[-1]`` is terminal.

        Infinite horizon (``cycles == 0``): runs one discarded warm-up period
        (Numba / LinearFast / FUES on the real grids), then iterates the
        one-period operator from the terminal seed until the convergence
        distance falls below ``self.tol`` *and* at least ``self.min_cycles``
        iterations have run, or until ``self.max_cycles`` is reached. The distance aggregates the
        differences of the solution ``distance_criteria`` according to
        ``self.distance_metric`` ("max", "mse", or "rmse"). Sets
        ``self.solution`` to a single-element list holding the converged (limit)
        solution, and stores ``self.distance`` (final scalar distance) and
        ``self.completed_cycles`` (number of iterations performed).
        """
        self.update()
        if not self.analytic_terminal and self.solution_terminal is not None:
            terminal = self.solution_terminal
        else:
            terminal = self.solve_terminal()
            self.solution_terminal = terminal

        if int(self.cycles) == 0:
            min_cycles = int(getattr(self, "min_cycles", 0))
            max_cycles = int(self.max_cycles)
            if min_cycles < 0:
                raise ValueError(f"min_cycles must be >= 0; got {min_cycles}")
            if min_cycles > max_cycles:
                raise ValueError(
                    f"min_cycles ({min_cycles}) cannot exceed max_cycles ({max_cycles})"
                )
            prev = terminal
            # Compile/cache kernels on the actual grids. Discard the result so
            # the counted path still starts from the terminal seed.
            self._log("Infinite-horizon warm-up period (not counted).")
            self.solve_one_period(terminal)
            for it in range(max_cycles):
                sol = self.solve_one_period(prev)
                dist = self._solution_distance(sol, prev)
                ###
                if getattr(self, "diagnostics", False):
                    print(dist)
                ###
                self._log(
                    f"Infinite-horizon iter {it}: {self.distance_metric} "
                    f"distance = {dist:.3e}"
                )
                # it is 0-based; completed count after this step is it + 1
                if dist < self.tol and (it + 1) >= min_cycles:
                    self.distance = dist
                    self.completed_cycles = it + 1
                    self._log(f"Converged after {it + 1} iterations (distance {dist:.3e}).")
                    self.solution = [sol]
                    return self.solution
                prev = sol
            self.distance = dist
            self.completed_cycles = max_cycles
            self._log(
                f"Warning: reached max_cycles={max_cycles} without converging "
                f"(last distance {dist:.3e})."
            )
            self.solution = [prev]
            return self.solution

        sols = [terminal]
        for _ in range(int(self.T_cycle)):
            sols.insert(0, self.solve_one_period(sols[0]))
        self.solution = sols
        return self.solution

    def _solve_perturbed_finite(self, param_path):
        """
        Finite-horizon backward induction seeded by ``self.solution_terminal``
        (which must be pre-set to the steady-state solution), with per-period
        parameter values.

        ``param_path`` maps a parameter name to a list of length ``T_cycle``
        giving that parameter's value in each period (index 0 = earliest period,
        index T_cycle-1 = the period just before the terminal).

        Supports:
        - "direct" parameters consumed in ``_end_of_period`` (``Rfree``,
          ``RSpread``, ``DiscFac``, ``PermGroFac``) via set-and-restore;
        - income-mean parameters in ``_JAC_INCOME`` (currently ``labor``) by
          precomputing ``IncShkDstn_path[t]`` (and ``probGrid_income_path[t]``
          when ``use_TransitionMatrix``) and activating period ``t`` before
          each backward step.

        Produces ``self.solution`` of length ``T_cycle + 1`` where
        ``solution[k]`` is period k and ``solution[-1]`` is the terminal seed.
        Also stores ``self.IncShkDstn_path`` (list of length ``T_cycle``, or
        ``None`` if no income shock) for ``calc_transition_matrix_dur``.
        """
        self.update()
        if self.solution_terminal is None:
            raise ValueError(
                "_solve_perturbed_finite requires self.solution_terminal to be set."
            )
        T = int(self.T_cycle)
        saved = {k: getattr(self, k) for k in param_path}

        def _path_or_ss(name, t):
            if name in param_path:
                return self._as_scalar(param_path[name][t])
            return self._as_scalar(getattr(self, name))

        # Precompute per-period income objects when an income shock is present.
        self.IncShkDstn_path = None
        self.probGrid_income_path = None
        if any(k in self._JAC_INCOME for k in param_path):
            IncShkDstn_path = []
            probGrid_income_path = [] if self.use_TransitionMatrix else None
            for t in range(T):
                labor_t = _path_or_ss("labor", t)
                tax_t = _path_or_ss("taxrate", t)
                wage_t = _path_or_ss("wage", t)
                shk_t = self._build_inc_shk(labor=labor_t, taxrate=tax_t, wage=wage_t)
                IncShkDstn_path.append(shk_t)
                if self.use_TransitionMatrix:
                    # Keep m/b grids from update(); only the lottery tracks labor.
                    probGrid_income_path.append(
                        self._build_prob_grid_income(
                            shk_t, labor=labor_t, taxrate=tax_t, wage=wage_t
                        )
                    )
            self.IncShkDstn_path = IncShkDstn_path
            self.probGrid_income_path = probGrid_income_path

        sols = [self.solution_terminal]
        try:
            # Backward step j produces period index (T-1-j): j=0 -> period T-1.
            for j in range(T):
                period = T - 1 - j
                for k, path in param_path.items():
                    setattr(self, k, path[period])
                if self.IncShkDstn_path is not None:
                    prob_t = (
                        self.probGrid_income_path[period]
                        if self.probGrid_income_path is not None
                        else None
                    )
                    self._activate_income(self.IncShkDstn_path[period], prob_t)
                sols.insert(0, self.solve_one_period(sols[0]))
        finally:
            for k, v in saved.items():
                setattr(self, k, v)
            # Restore SS income on the agent (paths remain for transition matrices).
            shk_ss = self._build_inc_shk()
            prob_ss = (
                self._build_prob_grid_income(shk_ss)
                if self.use_TransitionMatrix
                else None
            )
            self._activate_income(shk_ss, prob_ss)
        self.solution = sols
        return self.solution

    # ------------------------------------------------- transition matrix / SS
    def define_distribution_grid_dur(self, dist_mGrid=None, dist_nGrid=None, timestonest=None):
        """
        Build the histogram grids over which the stationary distribution lives.

        Simplified vs. background: LTV=0 is enforced, so ``mNrmMin = 0`` and the
        negative-``m`` branch is dropped. Grids are the same for the infinite
        (``cycles=0``) and finite (``T_cycle``) cases.
        """
        if timestonest is None:
            timestonest = self.NestFac

        if not hasattr(dist_mGrid, "__len__"):
            self.dist_mGrid = np.asarray(
                make_grid_exp_mult(
                    ming=self.TranMatrixmMin,
                    maxg=self.TranMatrixmMax,
                    ng=int(self.TranMatrixmCount),
                    timestonest=timestonest,
                )
            ).flatten()
        else:
            self.dist_mGrid = dist_mGrid

        if not hasattr(dist_nGrid, "__len__"):
            self.dist_nGrid = construct_grid(
                self.TranMatrixnMin,
                self.TranMatrixnMax,
                int(self.TranMatrixnCount),
                self.grid_type,
                self.NestFac,
            )
        else:
            self.dist_nGrid = dist_nGrid

    def _tran_matrix_one_period(self, sol, shk):
        """Policy grids + sparse transition matrix for a single solved period."""
        dist_nGrid = self.dist_nGrid
        dist_mGrid = self.dist_mGrid
        shape = (len(dist_nGrid), len(dist_mGrid))

        cPol = np.zeros(shape)
        dPol = np.zeros(shape)
        aPol = np.zeros(shape)
        adjPol = np.zeros(shape)
        dexPol = np.zeros(shape)
        for i_n in range(len(dist_nGrid)):
            n_aux = np.ones(len(dist_mGrid)) * dist_nGrid[i_n]
            cPol[i_n] = sol.cFunc(n_aux, dist_mGrid)
            dPol[i_n] = sol.dFunc(n_aux, dist_mGrid)
            dexPol[i_n] = dPol[i_n] - n_aux
            adjPol[i_n] = np.clip(sol.adjusting(n_aux, dist_mGrid), 0.0, 1.0)
            aPol[i_n] = sol.aFunc(n_aux, dist_mGrid)
        texPol = cPol + dexPol

        if np.any(np.isnan(cPol)):
            print("cPol_Grid has NaNs")

        # Scalar Rfree/RSpread/LivPrb in the simplified model (no time variation).
        R_vec = np.where(aPol < 0, self.Rfree + self.RSpread, self.Rfree)
        bNext = (R_vec * aPol) - dPol * (self.OpCost + self.dDepr * self.MaintCost)
        nNext_unshocked = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * dPol

        shk_prbs = shk.pmv
        perm_shks = shk.atoms[0]
        tran_shks = shk.atoms[1]
        LivPrb = self.LivPrb

        NewBornDist = jump_to_grid_dur(
            np.ones_like(tran_shks) * self.nNrmInitMean,
            np.ones_like(tran_shks) * self.mNrmInitMean,
            shk_prbs,
            dist_nGrid,
            dist_mGrid,
        )

        rows, cols, data, nStates = gen_tran_matrix_dur_sparse(
            dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
            shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist,
        )
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        data = np.array(data, dtype=np.float64)
        TranMatrix = sps.csr_matrix((data, (rows, cols)), shape=(nStates, nStates))

        return dict(
            cPol=cPol, dPol=dPol, aPol=aPol, adjPol=adjPol,
            dexPol=dexPol, texPol=texPol, TranMatrix=TranMatrix,
        )

    def _resolve_period_shk(self, k, shk_dstn=None):
        """
        Income distribution for period ``k`` of a finite-horizon transition.

        Resolution order:
        1. ``shk_dstn[k]`` if ``shk_dstn`` is a sequence of distributions;
        2. ``shk_dstn`` if it is a single distribution (broadcast);
        3. ``self.IncShkDstn_path[k]`` if a path was stored by
           ``_solve_perturbed_finite``;
        4. ``self.IncShkDstn`` (steady-state single distribution).
        """
        if shk_dstn is not None:
            if hasattr(shk_dstn, "pmv"):
                return shk_dstn
            return shk_dstn[k]
        path = getattr(self, "IncShkDstn_path", None)
        if path is not None:
            return path[k]
        return self.IncShkDstn

    def calc_transition_matrix_dur(self, shk_dstn=None):
        """
        Evaluate solved policies on the histogram grid and build the transition
        matrix. For ``cycles=0`` a single matrix is stored; for the finite case a
        list over ``T_cycle`` periods is stored.

        Finite-horizon periods use the period-specific income distribution from
        ``IncShkDstn_path`` (or ``shk_dstn[k]``) when available, so a labor
        impulse at period ``k`` enters both policy and the transition lottery.

        No CarCrash channel and no Harmenberg neutral measure in the simplified
        port (default has degenerate permanent shocks).
        """
        if int(self.cycles) == 0:
            shk = shk_dstn if shk_dstn is not None else self.IncShkDstn
            if not hasattr(shk, "pmv"):
                shk = shk[0]
            out = self._tran_matrix_one_period(self.solution[0], shk)
            self.cPol_Grid = out["cPol"]
            self.dPol_Grid = out["dPol"]
            self.aPol_Grid = out["aPol"]
            self.adjPol_Grid = out["adjPol"]
            self.dexPol_Grid = out["dexPol"]
            self.texPol_Grid = out["texPol"]
            self.tran_matrix = out["TranMatrix"]
        else:
            self.cPol_Grid = []
            self.dPol_Grid = []
            self.aPol_Grid = []
            self.adjPol_Grid = []
            self.dexPol_Grid = []
            self.texPol_Grid = []
            self.tran_matrix = []
            for k in range(int(self.T_cycle)):
                shk_k = self._resolve_period_shk(k, shk_dstn)
                out = self._tran_matrix_one_period(self.solution[k], shk_k)
                self.cPol_Grid.append(out["cPol"])
                self.dPol_Grid.append(out["dPol"])
                self.aPol_Grid.append(out["aPol"])
                self.adjPol_Grid.append(out["adjPol"])
                self.dexPol_Grid.append(out["dexPol"])
                self.texPol_Grid.append(out["texPol"])
                self.tran_matrix.append(out["TranMatrix"])

    def calc_ergodic_dist(self, transition_matrix=None):
        """Stationary distribution (eigenvector for eigenvalue 1) of the matrix."""
        if transition_matrix is None:
            transition_matrix = self.tran_matrix
        if isinstance(transition_matrix, list):
            transition_matrix = transition_matrix[0]

        ergodic_distr = compute_erg_dstn_sparse(transition_matrix)
        self.vec_erg_dstn = ergodic_distr
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.dist_nGrid), len(self.dist_mGrid))
        )

    def compute_steady_state(self):
        """
        Solve the infinite-horizon problem, build the transition matrix and its
        ergodic distribution, and return steady-state aggregates.
        """
        if self.solution is None or int(self.cycles) != 0:
            self.cycles = 0
            self.update()
            self.solve()

        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur()

        self.c_ss = self.cPol_Grid
        self.d_ss = self.dPol_Grid
        self.dex_ss = self.dexPol_Grid
        self.a_ss = self.aPol_Grid
        self.adj_ss = self.adjPol_Grid

        self.calc_ergodic_dist()
        ss_dstn = self.vec_erg_dstn

        self.A_ss = self.a_ss.flatten() @ ss_dstn
        self.C_ss = self.c_ss.flatten() @ ss_dstn
        self.D_ss = self.d_ss.flatten() @ ss_dstn
        self.DEX_ss = self.dex_ss.flatten() @ ss_dstn
        self.TEX_ss = self.C_ss + self.DEX_ss
        self.ADJ_ss = self.adj_ss.flatten() @ ss_dstn
        self._ss_method = "interp"

        return self.A_ss, self.C_ss, self.D_ss, self.DEX_ss, self.TEX_ss, self.ADJ_ss

    # --------------------------------------------------- sequence-space Jacobian
    # Direct parameters: enter ``_end_of_period`` as scalars (set-and-restore).
    # Income parameters: enter via per-period ``IncShkDstn_path`` rebuild.
    _JAC_SUPPORTED = ("Rfree", "RSpread", "DiscFac", "PermGroFac", "labor")
    _JAC_INCOME = ("labor",)
    # Not implemented yet (need richer income rebuild / LivPrb newborn channel).
    _JAC_NOT_IMPLEMENTED = (
        "LivPrb", "PermShkStd", "TranShkStd", "UnempPrb", "IncUnemp",
        "taxrate", "wage",
    )

    def _params_dict(self):
        """Reconstruct a kwargs dict of current parameter values (fresh-instance)."""
        return {k: getattr(self, k) for k in durable_defaults}

    def _make_finite_agent(self, T):
        """Fresh finite-horizon agent seeded by this model's steady state."""
        params = self._params_dict()
        params["cycles"] = 1
        params["T_cycle"] = int(T)
        params["analytic_terminal"] = False  # terminal = steady state
        agent = DurableConsumer(**params)
        agent.solution_terminal = deepcopy(self.solution[0])
        return agent

    def calc_jacobian(self, shk_param, T):
        """
        Sequence-space (fake-news) Jacobians of aggregate ``C, DEX, TEX, D, A,
        ADJ`` with respect to ``shk_param`` over a horizon ``T``.

        Requires ``compute_steady_state()`` to have been run on ``self`` first
        (uses the stored SS policy grids, transition matrix, and ergodic
        distribution). Returns ``(J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ)``.

        Supported shocks (``_JAC_SUPPORTED``):
        - direct: ``Rfree``, ``RSpread``, ``DiscFac``, ``PermGroFac``
        - income: ``labor`` (rebuilds ``IncShkDstn_path[t]`` / optional
          ``probGrid_income_path[t]``; finite-horizon transitions use period-``k``
          shocks)

        Other shocks print a not-implemented message and return ``None``.
        """
        if shk_param in self._JAC_NOT_IMPLEMENTED:
            print(
                f"calc_jacobian: shock parameter '{shk_param}' is not implemented "
                "yet (needs a per-period income process rebuild / non-trivial "
                "survival distribution). Supported: "
                f"{', '.join(self._JAC_SUPPORTED)}."
            )
            return None
        if shk_param not in self._JAC_SUPPORTED:
            print(
                f"calc_jacobian: unknown shock parameter '{shk_param}'. "
                f"Supported: {', '.join(self._JAC_SUPPORTED)}."
            )
            return None
        if getattr(self, "vec_erg_dstn", None) is None:
            raise RuntimeError(
                "calc_jacobian requires a steady state: call compute_steady_state() first."
            )

        T = int(T)
        dx = self.jac_dx
        ss_val = self._as_scalar(getattr(self, shk_param))

        # Steady-state objects (single arrays / matrix for cycles=0).
        a_ss = self.aPol_Grid
        c_ss = self.cPol_Grid
        d_ss = self.dPol_Grid
        dex_ss = self.dexPol_Grid
        adj_ss = self.adjPol_Grid
        tranmat_ss = self.tran_matrix
        Dist_ss = self.vec_erg_dstn.T

        # -- Finite-horizon agent: shock at period i = T-1 (second to last) -----
        FinHorizonAgent = self._make_finite_agent(T)
        i = T - 1
        path = [ss_val] * T
        path[i] = ss_val + dx
        FinHorizonAgent._solve_perturbed_finite({shk_param: path})
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid
        # Append steady-state grids (index T) — not added by calc_transition_matrix.
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        adj_t.append(self.adj_ss)

        # STEP 1: policy deviations and their aggregates (curly Y).
        A_curl_s = np.zeros(T)
        C_curl_s = np.zeros(T)
        D_curl_s = np.zeros(T)
        DEX_curl_s = np.zeros(T)
        ADJ_curl_s = np.zeros(T)
        for k in range(T):
            A_curl_s[k] = ((a_t[T - k] - a_ss).flatten() @ Dist_ss) / dx
            C_curl_s[k] = ((c_t[T - k] - c_ss).flatten() @ Dist_ss) / dx
            D_curl_s[k] = ((d_t[T - k] - d_ss).flatten() @ Dist_ss) / dx
            DEX_curl_s[k] = ((dex_t[T - k] - dex_ss).flatten() @ Dist_ss) / dx
            ADJ_curl_s[k] = ((adj_t[T - k] - adj_ss).flatten() @ Dist_ss) / dx

        # STEP 1 (curly D): distribution deviations.
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)
        nStates = len(self.dist_nGrid) * len(self.dist_mGrid)
        Dist_curl_s = np.zeros((T, nStates))
        for k in range(T):
            dlambda = tranmat_t[T - k] - tranmat_ss
            Dist_curl_s[k] = (dlambda @ Dist_ss) / dx

        # STEP 2: expectation vectors propagated by the SS transition matrix.
        exp_vecs_a = np.zeros((T, nStates))
        exp_vecs_c = np.zeros((T, nStates))
        exp_vecs_d = np.zeros((T, nStates))
        exp_vecs_dex = np.zeros((T, nStates))
        exp_vecs_adj = np.zeros((T, nStates))
        ev_a = a_ss.flatten()
        ev_c = c_ss.flatten()
        ev_d = d_ss.flatten()
        ev_dex = dex_ss.flatten()
        ev_adj = adj_ss.flatten()
        for k in range(T):
            exp_vecs_a[k] = ev_a
            exp_vecs_c[k] = ev_c
            exp_vecs_d[k] = ev_d
            exp_vecs_dex[k] = ev_dex
            exp_vecs_adj[k] = ev_adj
            ev_a = ev_a @ tranmat_ss
            ev_c = ev_c @ tranmat_ss
            ev_d = ev_d @ tranmat_ss
            ev_dex = ev_dex @ tranmat_ss
            ev_adj = ev_adj @ tranmat_ss

        # STEP 3: fake-news matrices.
        Curl_F_A = np.zeros((T, T))
        Curl_F_C = np.zeros((T, T))
        Curl_F_D = np.zeros((T, T))
        Curl_F_DEX = np.zeros((T, T))
        Curl_F_ADJ = np.zeros((T, T))
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s
        for r in range(T - 1):
            for s in range(T):
                Curl_F_A[r + 1][s] = np.dot(exp_vecs_a[r], Dist_curl_s[s])
                Curl_F_C[r + 1][s] = np.dot(exp_vecs_c[r], Dist_curl_s[s])
                Curl_F_D[r + 1][s] = np.dot(exp_vecs_d[r], Dist_curl_s[s])
                Curl_F_DEX[r + 1][s] = np.dot(exp_vecs_dex[r], Dist_curl_s[s])
                Curl_F_ADJ[r + 1][s] = np.dot(exp_vecs_adj[r], Dist_curl_s[s])

        # STEP 4: Jacobians from fake-news matrices.
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        # -- Zeroth column via a 2-period agent shocked at period 0 -------------
        ZerothColAgent = self._make_finite_agent(2)
        path0 = [ss_val + dx, ss_val]
        ZerothColAgent._solve_perturbed_finite({shk_param: path0})
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_zeroth = ZerothColAgent.tran_matrix[0]
        dstn = self.vec_erg_dstn.T

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)
        for k in range(T):
            if k == 0:
                dstn = tranmat_zeroth @ dstn
            else:
                dstn = tranmat_ss @ dstn
            A_t_no_sim[k] = self.aPol_Grid.flatten() @ dstn
            C_t_no_sim[k] = self.cPol_Grid.flatten() @ dstn
            D_t_no_sim[k] = self.dPol_Grid.flatten() @ dstn
            DEX_t_no_sim[k] = self.dexPol_Grid.flatten() @ dstn
            ADJ_t_no_sim[k] = self.adjPol_Grid.flatten() @ dstn

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    # -------------------------------------------- cond SS / Jacobian (search)
    def _nm_dist_mesh(self):
        n = np.asarray(self.dist_nGrid, dtype=float)
        m = np.asarray(self.dist_mGrid, dtype=float)
        n_aux = np.repeat(n[:, None], len(m), axis=1)
        m_aux = np.tile(m, (len(n), 1))
        return n, m, n_aux, m_aux

    def _tran_matrix_from_pols(self, cPol, dPol, aPol, shk):
        """Sparse transition matrix from mixed policies on the histogram grid."""
        dist_nGrid, dist_mGrid = self.dist_nGrid, self.dist_mGrid
        R_vec = np.where(aPol < 0, self.Rfree + self.RSpread, self.Rfree)
        bNext = (R_vec * aPol) - dPol * (self.OpCost + self.dDepr * self.MaintCost)
        nNext_unshocked = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * dPol
        NewBornDist = jump_to_grid_dur(
            np.ones_like(shk.atoms[1]) * self.nNrmInitMean,
            np.ones_like(shk.atoms[1]) * self.mNrmInitMean,
            shk.pmv, dist_nGrid, dist_mGrid,
        )
        rows, cols, data, nStates = gen_tran_matrix_dur_sparse(
            dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
            shk.pmv, shk.atoms[0], shk.atoms[1], self.LivPrb, NewBornDist,
        )
        return sps.csr_matrix(
            (np.asarray(data, dtype=np.float64),
             (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(nStates, nStates),
        )

    def _search_on_dist_grid(self, sol, sol_next, period_params=None):
        """
        Interpolate keeper / adjuster conditionals onto the histogram grid
        and re-combine (search / EV taste / hard max via ``_combine``).
        Does not use ``sol.adjusting``.

        For EV taste / hard max (``SearchCost == 0``), mix probabilities use
        the solver's stored ``inv_vFuncKeep`` / ``inv_vFuncAdj_1d_list``.
        Rebuilding ``u(c)+EOP`` from interpolated policies mismeasures
        ``Δv`` (logit / ``σ_ε`` is very sensitive) and can flip Jacobians.
        Directed search still reconstructs values on the dist grid so ``ε*``
        can be relocated relative to the solution-grid border.
        """
        saved = {}
        saved_inc = None
        try:
            if period_params:
                period_params = dict(period_params)
                k_inc = period_params.pop("_period_k", None)
                for key, val in period_params.items():
                    saved[key] = getattr(self, key)
                    setattr(self, key, val)
                if k_inc is not None and getattr(self, "IncShkDstn_path", None) is not None:
                    saved_inc = (
                        self.IncShkDstn,
                        getattr(self, "probGrid_income", None),
                    )
                    prob_t = (
                        self.probGrid_income_path[k_inc]
                        if getattr(self, "probGrid_income_path", None) is not None
                        else None
                    )
                    self._activate_income(self.IncShkDstn_path[k_inc], prob_t)
            nGrid, mGrid, n_aux, m_aux = self._nm_dist_mesh()
            shape = n_aux.shape

            cKeep = np.clip(np.asarray(sol.cFuncKeep(n_aux, m_aux), dtype=float), 0.0, None)
            dKeep = np.asarray(sol.dFuncKeep(n_aux, m_aux), dtype=float)
            aKeep = np.maximum(m_aux - cKeep, 0.0)

            nPrice = int(self.nPrice)
            price_vals = np.asarray(self.PriceShkVals, dtype=float)
            cAdj = np.zeros((nPrice,) + shape)
            dAdj = np.zeros_like(cAdj)
            aAdj = np.zeros_like(cAdj)
            tau_nm = self.tau_func(n_aux + m_aux)
            x_aux = np.clip(m_aux + (1.0 - tau_nm) * n_aux, 0.0, None)
            x_flat = x_aux.ravel()

            c_list = sol.cFuncAdj_1d_list
            d_list = sol.dFuncAdj_1d_list
            for i_p, p0 in enumerate(price_vals):
                if c_list is not None:
                    c_i = np.maximum(
                        np.asarray(c_list[i_p](x_flat), dtype=float), 0.0
                    ).reshape(shape)
                    d_i = np.maximum(
                        np.asarray(d_list[i_p](x_flat), dtype=float), 0.0
                    ).reshape(shape)
                else:
                    c_i = np.maximum(np.asarray(sol.cFuncAdj(n_aux, m_aux), dtype=float), 0.0)
                    d_i = np.maximum(np.asarray(sol.dFuncAdj(n_aux, m_aux), dtype=float), 0.0)
                cAdj[i_p] = c_i
                dAdj[i_p] = d_i
                aAdj[i_p] = np.maximum(x_aux - c_i - p0 * d_i, 0.0)

            inv_vKeep_fn = getattr(sol, "inv_vFuncKeep", None)
            inv_vAdj_list = getattr(sol, "inv_vFuncAdj_1d_list", None)
            use_sol_values = (
                self.SearchCost <= 0.0
                and inv_vKeep_fn is not None
                and inv_vAdj_list is not None
            )

            inv_vAdj = np.zeros_like(cAdj)
            vPAdj = np.zeros_like(cAdj)
            vPDAdj = np.zeros_like(cAdj)
            vPKeep = self.CRRAutilityP(cKeep, dKeep)
            if use_sol_values:
                inv_vKeep = np.asarray(inv_vKeep_fn(n_aux, m_aux), dtype=float)
                vPDKeep = self.CRRAutilityPD(cKeep, dKeep)
                for i_p in range(nPrice):
                    inv_vAdj[i_p] = np.asarray(
                        inv_vAdj_list[i_p](x_flat), dtype=float
                    ).reshape(shape)
                    vPAdj[i_p] = self.CRRAutilityP(cAdj[i_p], dAdj[i_p])
                    vPDAdj[i_p] = (1.0 - tau_nm) * vPAdj[i_p]
            else:
                eop = self._end_of_period(sol_next)
                EndOfPrdvFunc = eop["EndOfPrdvFunc"]
                EndOfPrdvPDFunc = eop["EndOfPrdvPDFunc"]
                vKeep = self.CRRAutility(cKeep, dKeep) + EndOfPrdvFunc(dKeep, aKeep)
                inv_vKeep = utility_inv(vKeep, self.CRRA)
                vPDKeep = self.CRRAutilityPD(cKeep, dKeep) + EndOfPrdvPDFunc(
                    dKeep, aKeep
                )
                for i_p in range(nPrice):
                    v_adj = self.CRRAutility(cAdj[i_p], dAdj[i_p]) + EndOfPrdvFunc(
                        dAdj[i_p], aAdj[i_p]
                    )
                    inv_vAdj[i_p] = utility_inv(v_adj, self.CRRA)
                    vPAdj[i_p] = self.CRRAutilityP(cAdj[i_p], dAdj[i_p])
                    vPDAdj[i_p] = (1.0 - tau_nm) * vPAdj[i_p]

            c, d, a, _v, _vp, _vpd, eps = self._combine(
                cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
                cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
                nGrid, mGrid,
            )
            return dict(
                c=np.asarray(c, dtype=float),
                d=np.asarray(d, dtype=float),
                a=np.asarray(a, dtype=float),
                eps=np.clip(np.asarray(eps, dtype=float), 0.0, 1.0),
            )
        finally:
            for key, val in saved.items():
                setattr(self, key, val)
            if saved_inc is not None:
                shk_ss, prob_ss = saved_inc
                self._activate_income(shk_ss, prob_ss)

    def _cond_one_period(self, sol, sol_next, shk, period_params=None):
        """Mixed policies + transition matrix after re-search on the dist grid."""
        cond = self._search_on_dist_grid(sol, sol_next, period_params)
        n_aux = np.repeat(self.dist_nGrid[:, None], len(self.dist_mGrid), axis=1)
        dex = cond["d"] - n_aux
        return dict(
            cPol=cond["c"],
            dPol=cond["d"],
            aPol=cond["a"],
            adjPol=cond["eps"],
            dexPol=dex,
            texPol=cond["c"] + dex,
            TranMatrix=self._tran_matrix_from_pols(cond["c"], cond["d"], cond["a"], shk),
        )

    def calc_transition_matrix_dur_cond(self, shk_dstn=None, param_path=None):
        """Like ``calc_transition_matrix_dur``, but re-combines on the dist grid."""
        if int(self.cycles) == 0:
            shk = shk_dstn if shk_dstn is not None else self.IncShkDstn
            if not hasattr(shk, "pmv"):
                shk = shk[0]
            out = self._cond_one_period(self.solution[0], self.solution[0], shk)
            self.cPol_Grid = out["cPol"]
            self.dPol_Grid = out["dPol"]
            self.aPol_Grid = out["aPol"]
            self.adjPol_Grid = out["adjPol"]
            self.dexPol_Grid = out["dexPol"]
            self.texPol_Grid = out["texPol"]
            self.tran_matrix = out["TranMatrix"]
        else:
            self.cPol_Grid = []
            self.dPol_Grid = []
            self.aPol_Grid = []
            self.adjPol_Grid = []
            self.dexPol_Grid = []
            self.texPol_Grid = []
            self.tran_matrix = []
            T = int(self.T_cycle)
            for k in range(T):
                shk_k = self._resolve_period_shk(k, shk_dstn)
                period_params = None
                if param_path:
                    period_params = {name: path[k] for name, path in param_path.items()}
                    period_params["_period_k"] = k
                sol_next = self.solution[k + 1]
                out = self._cond_one_period(
                    self.solution[k], sol_next, shk_k, period_params
                )
                self.cPol_Grid.append(out["cPol"])
                self.dPol_Grid.append(out["dPol"])
                self.aPol_Grid.append(out["aPol"])
                self.adjPol_Grid.append(out["adjPol"])
                self.dexPol_Grid.append(out["dexPol"])
                self.texPol_Grid.append(out["texPol"])
                self.tran_matrix.append(out["TranMatrix"])

    def compute_steady_state_cond(self):
        """
        Infinite-horizon SS with discrete choice re-combined on the histogram
        grid (conditional keeper / adjuster policies, then ``_combine``).

        Same return and stored attributes as ``compute_steady_state``.
        """
        if self.solution is None or int(self.cycles) != 0:
            self.cycles = 0
            self.update()
            self.solve()

        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur_cond()

        self.c_ss = self.cPol_Grid
        self.d_ss = self.dPol_Grid
        self.dex_ss = self.dexPol_Grid
        self.a_ss = self.aPol_Grid
        self.adj_ss = self.adjPol_Grid

        self.calc_ergodic_dist()
        ss_dstn = self.vec_erg_dstn

        self.A_ss = self.a_ss.flatten() @ ss_dstn
        self.C_ss = self.c_ss.flatten() @ ss_dstn
        self.D_ss = self.d_ss.flatten() @ ss_dstn
        self.DEX_ss = self.dex_ss.flatten() @ ss_dstn
        self.TEX_ss = self.C_ss + self.DEX_ss
        self.ADJ_ss = self.adj_ss.flatten() @ ss_dstn
        self._ss_method = "cond"

        return self.A_ss, self.C_ss, self.D_ss, self.DEX_ss, self.TEX_ss, self.ADJ_ss

    def calc_jacobian_cond(self, shk_param, T):
        """
        Fake-news Jacobians using histogram-grid re-combined policies.

        Same signature and return as ``calc_jacobian``. Calls
        ``compute_steady_state_cond()`` if that SS has not been built yet.
        """
        if shk_param in self._JAC_NOT_IMPLEMENTED:
            print(
                f"calc_jacobian_cond: shock parameter '{shk_param}' is not implemented "
                "yet. Supported: "
                f"{', '.join(self._JAC_SUPPORTED)}."
            )
            return None
        if shk_param not in self._JAC_SUPPORTED:
            print(
                f"calc_jacobian_cond: unknown shock parameter '{shk_param}'. "
                f"Supported: {', '.join(self._JAC_SUPPORTED)}."
            )
            return None
        if getattr(self, "_ss_method", None) != "cond" or getattr(
            self, "vec_erg_dstn", None
        ) is None:
            self.compute_steady_state_cond()

        T = int(T)
        dx = self.jac_dx
        ss_val = self._as_scalar(getattr(self, shk_param))

        a_ss = self.aPol_Grid
        c_ss = self.cPol_Grid
        d_ss = self.dPol_Grid
        dex_ss = self.dexPol_Grid
        adj_ss = self.adjPol_Grid
        tranmat_ss = self.tran_matrix
        Dist_ss = self.vec_erg_dstn.T

        FinHorizonAgent = self._make_finite_agent(T)
        path = [ss_val] * T
        path[T - 1] = ss_val + dx
        FinHorizonAgent._solve_perturbed_finite({shk_param: path})
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur_cond(param_path={shk_param: path})

        a_t = list(FinHorizonAgent.aPol_Grid) + [self.a_ss]
        c_t = list(FinHorizonAgent.cPol_Grid) + [self.c_ss]
        d_t = list(FinHorizonAgent.dPol_Grid) + [self.d_ss]
        dex_t = list(FinHorizonAgent.dexPol_Grid) + [self.dex_ss]
        adj_t = list(FinHorizonAgent.adjPol_Grid) + [self.adj_ss]
        tranmat_t = list(FinHorizonAgent.tran_matrix) + [tranmat_ss]

        A_curl_s = np.zeros(T)
        C_curl_s = np.zeros(T)
        D_curl_s = np.zeros(T)
        DEX_curl_s = np.zeros(T)
        ADJ_curl_s = np.zeros(T)
        nStates = len(self.dist_nGrid) * len(self.dist_mGrid)
        Dist_curl_s = np.zeros((T, nStates))
        for k in range(T):
            A_curl_s[k] = ((a_t[T - k] - a_ss).flatten() @ Dist_ss) / dx
            C_curl_s[k] = ((c_t[T - k] - c_ss).flatten() @ Dist_ss) / dx
            D_curl_s[k] = ((d_t[T - k] - d_ss).flatten() @ Dist_ss) / dx
            DEX_curl_s[k] = ((dex_t[T - k] - dex_ss).flatten() @ Dist_ss) / dx
            ADJ_curl_s[k] = ((adj_t[T - k] - adj_ss).flatten() @ Dist_ss) / dx
            Dist_curl_s[k] = ((tranmat_t[T - k] - tranmat_ss) @ Dist_ss) / dx

        exp_vecs_a = np.zeros((T, nStates))
        exp_vecs_c = np.zeros((T, nStates))
        exp_vecs_d = np.zeros((T, nStates))
        exp_vecs_dex = np.zeros((T, nStates))
        exp_vecs_adj = np.zeros((T, nStates))
        ev_a, ev_c, ev_d = a_ss.flatten(), c_ss.flatten(), d_ss.flatten()
        ev_dex, ev_adj = dex_ss.flatten(), adj_ss.flatten()
        for k in range(T):
            exp_vecs_a[k] = ev_a
            exp_vecs_c[k] = ev_c
            exp_vecs_d[k] = ev_d
            exp_vecs_dex[k] = ev_dex
            exp_vecs_adj[k] = ev_adj
            ev_a = ev_a @ tranmat_ss
            ev_c = ev_c @ tranmat_ss
            ev_d = ev_d @ tranmat_ss
            ev_dex = ev_dex @ tranmat_ss
            ev_adj = ev_adj @ tranmat_ss

        Curl_F_A = np.zeros((T, T))
        Curl_F_C = np.zeros((T, T))
        Curl_F_D = np.zeros((T, T))
        Curl_F_DEX = np.zeros((T, T))
        Curl_F_ADJ = np.zeros((T, T))
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s
        for r in range(T - 1):
            for s in range(T):
                Curl_F_A[r + 1][s] = np.dot(exp_vecs_a[r], Dist_curl_s[s])
                Curl_F_C[r + 1][s] = np.dot(exp_vecs_c[r], Dist_curl_s[s])
                Curl_F_D[r + 1][s] = np.dot(exp_vecs_d[r], Dist_curl_s[s])
                Curl_F_DEX[r + 1][s] = np.dot(exp_vecs_dex[r], Dist_curl_s[s])
                Curl_F_ADJ[r + 1][s] = np.dot(exp_vecs_adj[r], Dist_curl_s[s])

        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        ZerothColAgent = self._make_finite_agent(2)
        path0 = [ss_val + dx, ss_val]
        ZerothColAgent._solve_perturbed_finite({shk_param: path0})
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur_cond(param_path={shk_param: path0})
        tranmat_zeroth = ZerothColAgent.tran_matrix[0]

        dstn = self.vec_erg_dstn.T
        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)
        for k in range(T):
            dstn = (tranmat_zeroth if k == 0 else tranmat_ss) @ dstn
            A_t_no_sim[k] = self.aPol_Grid.flatten() @ dstn
            C_t_no_sim[k] = self.cPol_Grid.flatten() @ dstn
            D_t_no_sim[k] = self.dPol_Grid.flatten() @ dstn
            DEX_t_no_sim[k] = self.dexPol_Grid.flatten() @ dstn
            ADJ_t_no_sim[k] = self.adjPol_Grid.flatten() @ dstn

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    # ---------------------------------------------------------- Euler errors
    @staticmethod
    def _interior_nodes(grid, densify=1, trim=0.0):
        """``densify`` interior points in each grid cell, then trim a fraction of the span."""
        grid = np.asarray(grid, dtype=float)
        densify = int(densify)
        if densify < 1 or grid.size < 2:
            return np.array([], dtype=float)
        pieces = [
            np.linspace(lo, hi, densify + 2)[1:-1]
            for lo, hi in zip(grid[:-1], grid[1:])
        ]
        nodes = np.concatenate(pieces)
        span = float(grid[-1] - grid[0])
        lo = float(grid[0]) + float(trim) * span
        hi = float(grid[-1]) - float(trim) * span
        return nodes[(nodes > lo) & (nodes < hi)]

    @staticmethod
    def _judd_ee(c, c_impl):
        """Judd (1992) consumption-unit residual: log10(|c - c_impl| / c + 1e-16)."""
        c = np.asarray(c, dtype=float)
        c_impl = np.asarray(c_impl, dtype=float)
        rel = np.abs(c - c_impl) / np.maximum(np.abs(c), 1e-16)
        return np.log10(rel + 1e-16)

    @staticmethod
    def _ee_summary(ee):
        ee = np.asarray(ee, dtype=float)
        ee = ee[np.isfinite(ee)]
        n = int(ee.size)
        if n == 0:
            return dict(
                mean=float("nan"),
                p95=float("nan"),
                n=0,
                share_gt_neg3=float("nan"),
                share_gt_neg4=float("nan"),
            )
        return dict(
            mean=float(np.mean(ee)),
            p95=float(np.percentile(ee, 95)),
            n=n,
            share_gt_neg3=float(np.mean(ee > -3.0)),
            share_gt_neg4=float(np.mean(ee > -4.0)),
        )

    @staticmethod
    def _assign_ee(sol, prefix, summary):
        setattr(sol, f"{prefix}_mean", summary["mean"])
        setattr(sol, f"{prefix}_p95", summary["p95"])
        setattr(sol, f"{prefix}_n", summary["n"])
        setattr(sol, f"{prefix}_share_gt_neg3", summary["share_gt_neg3"])
        setattr(sol, f"{prefix}_share_gt_neg4", summary["share_gt_neg4"])

    def _gothic_from_next(self, sol_next, d, a):
        """
        End-of-period marginal values at (d, a) from next period's stored
        ``inv_vPFunc`` / ``inv_vPDFunc``, using the same income quadrature as EGM.

        Returns ``(eop1, eopd, ok)`` where ``ok`` is False if any shock node
        would extrapolate the continuation interpolants.
        """
        d = np.asarray(d, dtype=float).ravel()
        a = np.asarray(a, dtype=float).ravel()
        n_pts = d.size
        eop1 = np.full(n_pts, np.nan)
        eopd = np.full(n_pts, np.nan)
        ok = np.zeros(n_pts, dtype=bool)
        if n_pts == 0:
            return eop1, eopd, ok

        psi = np.asarray(self.PermShkValsNext, dtype=float)
        theta = np.asarray(self.TranShkValsNext, dtype=float)
        omega = np.asarray(self.ShkPrbsNext, dtype=float)
        n_next = ((1.0 - self.dDepr) * d[:, None]) / psi[None, :]
        m_next = (
            self.Rfree / (self.PermGroFac * psi[None, :]) * a[:, None]
            + theta[None, :]
        )
        n_grid = np.asarray(sol_next.nNrmGrid, dtype=float)
        m_grid = np.asarray(sol_next.mNrmGrid, dtype=float)
        ok = np.all(
            (n_next >= n_grid[0])
            & (n_next <= n_grid[-1])
            & (m_next >= m_grid[0])
            & (m_next <= m_grid[-1]),
            axis=1,
        )
        if not np.any(ok):
            return eop1, eopd, ok

        n_flat = n_next[ok].ravel()
        m_flat = m_next[ok].ravel()
        crra = self.CRRA
        vp = np.clip(sol_next.inv_vPFunc(n_flat, m_flat), 0.0, None) ** (-crra)
        vpd = np.clip(sol_next.inv_vPDFunc(n_flat, m_flat), 0.0, None) ** (-crra)
        vp = np.where(np.isfinite(vp), vp, 1e16)
        vpd = np.where(np.isfinite(vpd), vpd, 1e16)
        n_ok = int(np.sum(ok))
        vp = vp.reshape(n_ok, psi.size)
        vpd = vpd.reshape(n_ok, psi.size)

        wgt = omega[None, :] * (psi[None, :] ** (-crra))
        e_vp = np.sum(wgt * vp, axis=1)
        e_vpd = np.sum(wgt * vpd, axis=1)
        fac = self.DiscFac * self.LivPrb * self.PermGroFac ** (-crra)
        eop1[ok] = fac * self.Rfree * e_vp
        eopd[ok] = fac * (1.0 - self.dDepr * (1.0 - self.MaintCost)) * e_vpd
        return eop1, eopd, ok

    def _euler_consumption(self, sol_next, c, d, a, a_guard, c_guard):
        """Branch-conditional consumption Euler: Judd residual vs u_c^{-1}(gothic_m)."""
        c = np.asarray(c, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()
        a = np.asarray(a, dtype=float).ravel()
        base_ok = (
            np.isfinite(c)
            & np.isfinite(d)
            & np.isfinite(a)
            & (a > a_guard)
            & (c > c_guard)
            & (d >= -1e-12)
        )
        eop1, eopd, ok_next = self._gothic_from_next(sol_next, d, a)
        ok = base_ok & ok_next & np.isfinite(eop1) & (eop1 > 0.0)
        c_impl = np.full_like(c, np.nan)
        if np.any(ok):
            with np.errstate(divide="ignore", invalid="ignore"):
                c_impl[ok] = self.CRRAutilityP_inv(eop1[ok], d[ok])
        ok &= np.isfinite(c_impl) & (c_impl > c_guard)
        ee = np.full_like(c, np.nan)
        ee[ok] = self._judd_ee(c[ok], c_impl[ok])
        return ee, eop1, eopd, ok

    def compute_euler_errors(
        self,
        densify=2,
        trim_m=0.05,
        trim_n=0.02,
        a_guard=1e-6,
        c_guard=None,
    ):
        """
        Off-grid, branch-conditional Euler residuals (Judd 1992 consumption units).

        Evaluates keeper ``c(n,m)`` and adjuster ``(c,d)(ℓ)`` on a staggered
        ``(n,m)`` mesh that is not a subset of the solution grid. Continuation
        comes from *next* period's stored ``inv_vPFunc`` / ``inv_vPDFunc`` with
        the same income quadrature as EGM — not from the current period's
        end-of-period interpolants, and never from the mixed policy ``c̄``.

        Constrained points (``a`` below ``a_guard``), tiny consumption, and
        states that would extrapolate the continuation interpolants are dropped.

        Stores on each non-terminal ``solution[t]``:

        * ``ee_c_keep_{mean,p95,n,share_gt_neg3,share_gt_neg4}``
        * ``ee_c_adj_{...}``
        * ``ee_d_adj_{...}``  (durable FOC converted to consumption units)

        Infinite horizon (``cycles=0``): the unique solution is its own
        continuation. Finite horizon: ``solution[t]`` uses ``solution[t+1]``;
        the terminal period is skipped.

        Returns a list of per-period summary dicts.
        """
        sols = getattr(self, "solution", None)
        if not sols:
            raise ValueError("compute_euler_errors requires a solved model (call solve first).")
        if c_guard is None:
            c_guard = max(float(self.d_ubar), 1e-8)
        a_guard = float(a_guard)

        n_test = self._interior_nodes(self.nNrmGrid, densify=densify, trim=trim_n)
        m_test = self._interior_nodes(self.mNrmGrid, densify=densify, trim=trim_m)
        if n_test.size == 0 or m_test.size == 0:
            raise ValueError("Euler test mesh is empty; relax trim_n / trim_m or densify.")
        nn, mm = np.meshgrid(n_test, m_test, indexing="ij")
        n_flat = nn.ravel()
        m_flat = mm.ravel()

        i_p0 = int(np.argmin(np.abs(self.PriceShkVals - 1.0)))
        p0 = float(self.PriceShkVals[i_p0])
        tau = float(self.tau)
        ell = m_flat + (1.0 - tau) * n_flat

        n_period = len(sols)
        # Infinite horizon: one solution, continuation is itself.
        # Finite horizon: skip the terminal (last) entry.
        if n_period == 1:
            pairs = [(0, sols[0], sols[0])]
        else:
            pairs = [(t, sols[t], sols[t + 1]) for t in range(n_period - 1)]

        empty = self._ee_summary(np.array([]))
        reports = []
        for t, sol, sol_next in pairs:
            if sol.cFuncKeep is None or sol_next.inv_vPFunc is None:
                self._assign_ee(sol, "ee_c_keep", empty)
                self._assign_ee(sol, "ee_c_adj", empty)
                self._assign_ee(sol, "ee_d_adj", empty)
                reports.append(
                    dict(period=t, c_keep=empty, c_adj=empty, d_adj=empty)
                )
                continue

            # ----- keeper: d = n, a = m - c -----
            c_keep = np.clip(np.asarray(sol.cFuncKeep(n_flat, m_flat), dtype=float).ravel(), 0.0, None)
            d_keep = n_flat
            a_keep = m_flat - c_keep
            ee_c_keep, _, _, _ = self._euler_consumption(
                sol_next, c_keep, d_keep, a_keep, a_guard, c_guard
            )
            sum_c_keep = self._ee_summary(ee_c_keep)

            # ----- adjuster at representative p ≈ 1, policies in liquidity ℓ -----
            c_adj_fn = sol.cFuncAdj_1d
            d_adj_fn = sol.dFuncAdj_1d
            if c_adj_fn is None or d_adj_fn is None:
                sum_c_adj = empty
                sum_d_adj = empty
            else:
                x_grid = getattr(c_adj_fn, "grid_list", [self.xNrmGrid])[0]
                x_lo, x_hi = float(np.asarray(x_grid)[0]), float(np.asarray(x_grid)[-1])
                in_x = (ell >= x_lo) & (ell <= x_hi)
                c_adj = np.full_like(ell, np.nan)
                d_adj = np.full_like(ell, np.nan)
                if np.any(in_x):
                    c_adj[in_x] = np.clip(np.asarray(c_adj_fn(ell[in_x]), dtype=float).ravel(), 0.0, None)
                    d_adj[in_x] = np.clip(np.asarray(d_adj_fn(ell[in_x]), dtype=float).ravel(), 0.0, None)
                a_adj = ell - c_adj - p0 * d_adj
                ee_c_adj, eop1, eopd, ok_c = self._euler_consumption(
                    sol_next, c_adj, d_adj, a_adj, a_guard, c_guard
                )
                ee_c_adj = np.where(in_x, ee_c_adj, np.nan)
                sum_c_adj = self._ee_summary(ee_c_adj)

                # Durable FOC in consumption units: hold d fixed, invert u_d.
                target = p0 * eop1 - eopd
                ok_d = (
                    ok_c
                    & in_x
                    & np.isfinite(target)
                    & (target > 0.0)
                    & (d_adj > 0.0)
                )
                c_d_impl = np.full_like(c_adj, np.nan)
                if np.any(ok_d):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        c_d_impl[ok_d] = self.CRRAutilityPD_inv(target[ok_d], d_adj[ok_d])
                ok_d &= np.isfinite(c_d_impl) & (c_d_impl > c_guard)
                ee_d_adj = np.full_like(c_adj, np.nan)
                ee_d_adj[ok_d] = self._judd_ee(c_adj[ok_d], c_d_impl[ok_d])
                sum_d_adj = self._ee_summary(ee_d_adj)

            self._assign_ee(sol, "ee_c_keep", sum_c_keep)
            self._assign_ee(sol, "ee_c_adj", sum_c_adj)
            self._assign_ee(sol, "ee_d_adj", sum_d_adj)
            reports.append(
                dict(period=t, c_keep=sum_c_keep, c_adj=sum_c_adj, d_adj=sum_d_adj)
            )
        return reports

    def print_kink_stats(self, label=None):
        """
        Print keeper (DCEGM decreasing segments) and adjuster (FUES) stats
        for each period.

        Terminal period has no EGM/FUES envelope step, so those rows are marked n/a.
        """
        if label:
            print(f"=== {label}: kink / segment stats by period ===")
        else:
            print("=== Kink / segment stats by period ===")
        sols = getattr(self, "solution", None)
        if not sols:
            print("  (no solution)")
            return
        for t, sol in enumerate(sols):
            if sol.keep_decreasing_segments is None and sol.adj_kinks_removed is None:
                print(f"Period {t} (terminal): n/a")
                continue
            print(f"Period {t}:")
            print(
                f"  Keeper — number of decreasing segments: {sol.keep_decreasing_segments}, "
                f"share of decreasing segments: {sol.keep_share_decreasing}"
            )
            print(
                f"  Adjuster — number of kinks removed: {sol.adj_kinks_removed}, "
                f"share of removed points: {sol.adj_share_removed}"
            )
            if getattr(sol, "ee_c_keep_mean", None) is not None:
                def _ee_line(name, prefix):
                    mean = getattr(sol, f"{prefix}_mean")
                    p95 = getattr(sol, f"{prefix}_p95")
                    n = getattr(sol, f"{prefix}_n")
                    share = getattr(sol, f"{prefix}_share_gt_neg3")
                    if n == 0 or (isinstance(mean, float) and np.isnan(mean)):
                        print(f"  Euler {name}: n/a")
                        return
                    print(
                        f"  Euler {name}: mean={mean:.3f}, p95={p95:.3f}, n={n}"
                        f", share>1e-3={share:.3f}"
                    )

                _ee_line("EE_c keep", "ee_c_keep")
                _ee_line("EE_c adj", "ee_c_adj")
                _ee_line("EE_d adj", "ee_d_adj")


__all__ = [
    "DurableConsumer",
    "DurableConsumerSolution",
    "durable_defaults",
    "calcLogSumChoiceProbs",
    "combine_ev_taste_price",
    "combine_search_price",
]
