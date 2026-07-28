"""
Utilities for Problems with Durable goods.

Contains only the functions required by
DurableModel_FUES_Search_Fast_Model20_Baseline.py,
sourced from utilities_Durable_20240904.py and the
previous utilities_Durable.py (now utilities_Durable_20260312.py).
"""
import numpy as np
import numba
from scipy import sparse as sp
from numba import njit, float64, int32, prange
from numba.typed import List

import scipy.optimize as sp_optimize
import scipy.sparse.linalg as spla

from HARK.utilities import make_grid_exp_mult
from HARK.core import MetricObject
from HARK.distribution import (
    combine_indep_dstns,
    DiscreteDistribution,
    MeanOneLogNormal,
    add_discrete_outcome_constant_mean,
    TimeVaryingDiscreteDistribution,
)
from copy import deepcopy


##########################################################################################################
### Grid Construction
##########################################################################################################

def construct_grid(Min, Max, Count, grid_type, NestFac):
    """
    Constructs grids in different grid-types (linear, exp_mult, nonlinear,
    exp_mult_extended).
    """
    exp_nest = NestFac
    Grid = None
    if grid_type == "linear":
        Grid = np.linspace(Min, Max, Count)
    elif grid_type == "exp_mult":
        Grid = make_grid_exp_mult(ming=Min, maxg=Max, ng=Count, timestonest=exp_nest)
    elif grid_type == "nonlinear":
        Grid = nonlinspace(Min, Max, Count, 1.1)
    elif grid_type == "exp_mult_extended":
        Count = np.int(Count / 2)
        Grid1 = make_grid_exp_mult(ming=Min, maxg=Max, ng=Count, timestonest=exp_nest)
        Grid2 = np.linspace(Min, Max, Count)
        Grid = np.unique(np.concatenate((Grid1, Grid2[:-1])))
    else:
        raise Exception(
            "grid_type not recognized in __init__."
            + "Please ensure grid_type is 'linear', 'nonlinear', or 'exp_mult'"
        )
    return Grid


##########################################################################################################
### Utility Functions
##########################################################################################################

@njit("(float64[:])(float64[:], float64)", cache=True)
def utility_njit(c, gam):
    if gam == 1:
        return np.log(c)
    else:
        return c ** (1.0 - gam) / (1.0 - gam)


def CRRAutilityP_dur(c, d, alpha, d_ubar, gam):
    """Marginal utility of consumption with durable goods (CRRA)."""
    return ((alpha * c ** (alpha * (1 - gam) - 1)) * (d + d_ubar) ** ((1 - alpha) * (1 - gam)))


def inv_CRRAutilityP_dur(c, d, alpha, d_ubar, gam):
    """Inverse marginal utility of consumption with durable goods (CRRA)."""
    vP = ((alpha * c ** (alpha * (1 - gam) - 1)) * (d + d_ubar) ** ((1 - alpha) * (1 - gam)))
    return vP ** (-1.0 / gam)


@njit("(float64[:])(float64[:], float64[:], float64, float64, float64)", cache=True)
def CRRAutilityP_inv_njit(c, d, alpha, d_ubar, CRRA):
    power = 1 / ((alpha * (1 - CRRA) - 1))
    return (c / (alpha * (d + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** power


@njit("(float64[:](float64[:], float64[:], float64, float64, float64))", cache=True)
def CRRAutility_njit(c, d, alpha, d_ubar, CRRA):
    c_prime = np.power(c, alpha)
    d_prime = np.power(d + d_ubar, 1 - alpha)
    u_inner = c_prime * d_prime
    return utility_njit(u_inner, CRRA)


class MargValueFuncCRRA_dur(MetricObject):
    """
    Marginal value function for durable goods models where
    dvdm(state) = u'(c(state)) (CRRA utility).
    """
    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, dFunc, alpha, d_ubar, CRRA):
        self.cFunc = deepcopy(cFunc)
        self.dFunc = deepcopy(dFunc)
        self.CRRA = CRRA
        self.alpha = alpha
        self.d_ubar = d_ubar
        self.grid_list = cFunc.grid_list if hasattr(cFunc, 'grid_list') else None

    def __call__(self, *cFuncArgs):
        return CRRAutilityP_dur(
            self.cFunc(*cFuncArgs), self.dFunc(*cFuncArgs),
            alpha=self.alpha, d_ubar=self.d_ubar, gam=self.CRRA
        )


class inv_MargValueFuncCRRA_dur(MetricObject):
    """
    Inverse marginal value function for durable goods models (CRRA utility).
    """
    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, dFunc, alpha, d_ubar, CRRA):
        self.cFunc = deepcopy(cFunc)
        self.dFunc = deepcopy(dFunc)
        self.CRRA = CRRA
        self.alpha = alpha
        self.d_ubar = d_ubar
        self.grid_list = cFunc.grid_list if hasattr(cFunc, 'grid_list') else None

    def __call__(self, *cFuncArgs):
        return inv_CRRAutilityP_dur(
            self.cFunc(*cFuncArgs), self.dFunc(*cFuncArgs),
            alpha=self.alpha, d_ubar=self.d_ubar, gam=self.CRRA
        )


##########################################################################################################
### Income Process / Distributions
##########################################################################################################

class LognormPermIncShk_Truncated(DiscreteDistribution):
    """
    Discretized lognormal permanent income shock with truncation support.
    """

    def __init__(self, sigma, n_approx, truncation, neutral_measure=False, seed=0):
        logn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        if neutral_measure:
            logn_approx.pmv = (logn_approx.atoms * logn_approx.pmv).flatten()
        else:
            logn_approx.atoms = np.clip(
                logn_approx.atoms, 1 - truncation * sigma, 1 + truncation * sigma
            )
        super().__init__(pmv=logn_approx.pmv, atoms=logn_approx.atoms, seed=seed)


class MixtureTranIncShk_HANK_Truncated(DiscreteDistribution):
    """
    Transitory income shock distribution: mixture of lognormal and
    unemployment shock, with truncation.
    """

    def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, wage, labor,
                 taxrate, TranShkMean_Func, truncation, seed=0):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        dstn_approx.atoms = np.clip(
            dstn_approx.atoms, 1 - truncation * sigma, 1 + truncation * sigma
        )
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )
        dstn_approx.atoms = dstn_approx.atoms * TranShkMean_Func(taxrate, labor, wage)
        TranShk_min = np.max([1 - truncation * sigma, 0.0])
        dstn_approx.pmv = np.append(0.0, dstn_approx.pmv)
        dstn_approx.atoms = np.append(TranShk_min, dstn_approx.atoms)
        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


def add_discrete_outcome_constant_mean_jacobian(distribution, p, x, p_old, x_old, sort=False):
    """
    Adds a discrete outcome (x, p) to a distribution, scaling other atoms
    using OLD (p_old, x_old) values to preserve the old mean scaling.
    """
    if type(distribution) != TimeVaryingDiscreteDistribution:
        scaling_factor = (1 - p_old * x_old) / (1 - p_old)
        atoms = np.append(x, distribution.atoms * scaling_factor)
        pmv = np.append(p, distribution.pmv * (1 - p))
        if sort:
            indices = np.argsort(atoms)
            atoms = atoms[indices]
            pmv = pmv[indices]
        return DiscreteDistribution(pmv, atoms)
    else:
        return TimeVaryingDiscreteDistribution(
            [
                add_discrete_outcome_constant_mean_jacobian(d, p, x, p_old, x_old)
                for d in distribution.distributions
            ],
            seed=distribution.seed,
        )


class MixtureTranIncShk_HANK_Truncated_Jacobian(DiscreteDistribution):
    """
    Transitory income shock distribution for Jacobian computations:
    mixture of lognormal and unemployment shock, scaled by old (p_old, x_old).
    """

    def __init__(self, sigma, UnempPrb, IncUnemp, UnempPrb_old, IncUnemp_old,
                 n_approx, wage, labor, taxrate, TranShkMean_Func, truncation, seed=0):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        dstn_approx.atoms = np.clip(
            dstn_approx.atoms, 1 - truncation * sigma, 1 + truncation * sigma
        )
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean_jacobian(
                dstn_approx, p=UnempPrb, x=IncUnemp,
                p_old=UnempPrb_old, x_old=IncUnemp_old
            )
        dstn_approx.atoms = dstn_approx.atoms * TranShkMean_Func(taxrate, labor, wage)
        TranShk_min = np.max([1 - truncation * sigma, 0.0])
        dstn_approx.pmv = np.append(0.0, dstn_approx.pmv)
        dstn_approx.atoms = np.append(TranShk_min, dstn_approx.atoms)
        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


class HANKIncShkDstn_Truncated(DiscreteDistribution):
    """
    Joint distribution of permanent and transitory income shocks (HANK),
    with truncation support.
    """

    def __init__(self, sigma_Perm, sigma_Tran, n_approx_Perm, n_approx_Tran,
                 UnempPrb, IncUnemp, taxrate, TranShkMean_Func, labor, wage,
                 truncation, neutral_measure=False, seed=0):
        perm_dstn = LognormPermIncShk_Truncated(
            sigma=sigma_Perm, n_approx=n_approx_Perm,
            neutral_measure=neutral_measure, truncation=truncation,
        )
        tran_dstn = MixtureTranIncShk_HANK_Truncated(
            sigma=sigma_Tran, UnempPrb=UnempPrb, IncUnemp=IncUnemp,
            n_approx=n_approx_Tran, wage=wage, labor=labor,
            taxrate=taxrate, TranShkMean_Func=TranShkMean_Func, truncation=truncation,
        )
        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)
        super().__init__(pmv=joint_dstn.pmv, atoms=joint_dstn.atoms, seed=seed)


class HANKIncShkDstn_Truncated_Jacobian(DiscreteDistribution):
    """
    Joint distribution of permanent and transitory income shocks (HANK)
    for Jacobian computations, with truncation support.
    """

    def __init__(self, sigma_Perm, sigma_Tran, n_approx_Perm, n_approx_Tran,
                 UnempPrb, IncUnemp, UnempPrb_old, IncUnemp_old,
                 taxrate, TranShkMean_Func, labor, wage,
                 truncation, neutral_measure=False, seed=0):
        perm_dstn = LognormPermIncShk_Truncated(
            sigma=sigma_Perm, n_approx=n_approx_Perm,
            neutral_measure=neutral_measure, truncation=truncation,
        )
        tran_dstn = MixtureTranIncShk_HANK_Truncated_Jacobian(
            sigma=sigma_Tran, UnempPrb=UnempPrb, IncUnemp=IncUnemp,
            UnempPrb_old=UnempPrb_old, IncUnemp_old=IncUnemp_old,
            n_approx=n_approx_Tran, wage=wage, labor=labor,
            taxrate=taxrate, TranShkMean_Func=TranShkMean_Func, truncation=truncation,
        )
        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)
        super().__init__(pmv=joint_dstn.pmv, atoms=joint_dstn.atoms, seed=seed)


##########################################################################################################
### Transition Matrix Simulation
##########################################################################################################

@numba.njit
def jump_to_grid_dur(n_vals, m_vals, probs, dist_nGrid, dist_mGrid):
    """
    Distributes (n_vals, m_vals) onto (dist_nGrid, dist_mGrid), maintaining means.
    Returns flattened probability grid.
    """
    probGrid = np.zeros((len(dist_nGrid), len(dist_mGrid)))

    nIndex = np.digitize(n_vals, dist_nGrid) - 1
    nIndex[n_vals <= dist_nGrid[0]] = -1
    nIndex[n_vals >= dist_nGrid[-1]] = len(dist_nGrid) - 1

    mIndex = np.digitize(m_vals, dist_mGrid) - 1
    mIndex[m_vals <= dist_mGrid[0]] = -1
    mIndex[m_vals >= dist_mGrid[-1]] = len(dist_mGrid) - 1

    for i in range(len(n_vals)):
        # --- n weights for shock i ---
        if nIndex[i] == -1:
            nlowerIndex = 0
            nupperIndex = 0
            nlowerWeight = 1.0
            nupperWeight = 0.0
        elif nIndex[i] == len(dist_nGrid) - 1:
            nlowerIndex = len(dist_nGrid) - 1
            nupperIndex = len(dist_nGrid) - 1
            nlowerWeight = 1.0
            nupperWeight = 0.0
        else:
            nlowerIndex = nIndex[i]
            nupperIndex = nIndex[i] + 1
            nlowerWeight = (dist_nGrid[nupperIndex] - n_vals[i]) / (
                dist_nGrid[nupperIndex] - dist_nGrid[nlowerIndex])
            nupperWeight = 1.0 - nlowerWeight

        # --- m weights for shock i ---
        if mIndex[i] == -1:
            mlowerIndex = 0
            mupperIndex = 0
            mlowerWeight = 1.0
            mupperWeight = 0.0
        elif mIndex[i] == len(dist_mGrid) - 1:
            mlowerIndex = len(dist_mGrid) - 1
            mupperIndex = len(dist_mGrid) - 1
            mlowerWeight = 1.0
            mupperWeight = 0.0
        else:
            mlowerIndex = mIndex[i]
            mupperIndex = mIndex[i] + 1
            mlowerWeight = (dist_mGrid[mupperIndex] - m_vals[i]) / (
                dist_mGrid[mupperIndex] - dist_mGrid[mlowerIndex])
            mupperWeight = 1.0 - mlowerWeight

        probGrid[nlowerIndex][mlowerIndex] += probs[i] * nlowerWeight * mlowerWeight
        probGrid[nlowerIndex][mupperIndex] += probs[i] * nlowerWeight * mupperWeight
        probGrid[nupperIndex][mlowerIndex] += probs[i] * nupperWeight * mlowerWeight
        probGrid[nupperIndex][mupperIndex] += probs[i] * nupperWeight * mupperWeight

    return probGrid.flatten()


def jump_to_grid_dur_shock(dbNext, dnNext_unshocked, Dist_ss_2d,
                            bNext_ss, nNext_unshocked_ss,
                            shk_prbs, perm_shks, tran_shks, LivPrb,
                            dist_nGrid, dist_mGrid):
    """
    Analytically computes the linearised distribution perturbation (curlyD)
    from next-period state deviations, using derivatives of bilinear
    interpolation weights (SSJ-style: avoids differencing full transition
    matrices).

    For each grid point and income shock the function differentiates through
    the bilinear mass-assignment step of jump_to_grid_dur:
        d(w_n_lower)/d(n_next) = -1 / cell_width_n

    Parameters
    ----------
    dbNext : (n_n x n_m) array
        Total derivative of next-period bank balance w.r.t. the shock
        (includes BOTH the direct effect from shocks that enter the
        transition equation — e.g. Rfree — AND the indirect policy
        response).  Caller must pre-compute this as
            (bNext_plus - bNext_minus) / (2*dx)
        using the actual perturbed interest rate for the shocked period.
    dnNext_unshocked : (n_n x n_m) array
        Total derivative of next-period unshocked durable stock w.r.t. the
        shock.  For shocks that do not directly enter the depreciation
        equation (Rfree, labor, UnempPrb) this equals
            (1 - dDepr*(1 - MaintCost)) * da_d.
    Dist_ss_2d : (n_n x n_m) array
        Steady-state distribution on the distribution grid.
    bNext_ss : (n_n x n_m) array
        SS bank balance next period  =  R_ss*a_ss - d_ss*(OpCost + dDepr*MaintCost).
    nNext_unshocked_ss : (n_n x n_m) array
        SS unshocked durable next period = (1 - dDepr*(1-MaintCost))*d_ss.
    shk_prbs, perm_shks, tran_shks : 1-D arrays (n_shocks,)
        Income shock distribution (Harmenberg neutral measure).
    LivPrb : float
    dist_nGrid, dist_mGrid : 1-D arrays
        Distribution grids.

    Returns
    -------
    curlyD : 1-D array of shape (n_n * n_m,)
        Linearised distribution response. Sums to zero (mass-preserving).
    """
    n_n      = len(dist_nGrid)
    n_m      = len(dist_mGrid)
    n_states = n_n * n_m

    # Caller already provides full derivatives of next-period states
    dn_next_unshocked = dnNext_unshocked

    D_flat  = Dist_ss_2d.ravel()
    curlyD  = np.zeros(n_states)

    for js in range(len(shk_prbs)):
        p    = perm_shks[js]
        tr   = tran_shks[js]
        prob = shk_prbs[js]

        # SS next-period states (all grid points at once)
        n_next = (nNext_unshocked_ss / p).ravel()
        m_next = (bNext_ss / p + tr).ravel()

        # Policy-deviation shifts at next-period states
        dn = (dn_next_unshocked / p).ravel()
        dm = (dbNext / p).ravel()

        # Lower grid-cell indices
        jn_raw = np.searchsorted(dist_nGrid, n_next) - 1
        jm_raw = np.searchsorted(dist_mGrid, m_next) - 1

        # Boundary masks (derivative is zero at boundary: mass just piles up)
        at_n_bnd = (n_next <= dist_nGrid[0]) | (n_next >= dist_nGrid[-1])
        at_m_bnd = (m_next <= dist_mGrid[0]) | (m_next >= dist_mGrid[-1])

        jn_lo = np.clip(jn_raw, 0, n_n - 2)
        jn_hi = jn_lo + 1
        jm_lo = np.clip(jm_raw, 0, n_m - 2)
        jm_hi = jm_lo + 1

        # Cell widths
        n_cw = dist_nGrid[jn_hi] - dist_nGrid[jn_lo]
        m_cw = dist_mGrid[jm_hi] - dist_mGrid[jm_lo]

        # SS bilinear weights (clamped for safety)
        w_n_lo = np.clip((dist_nGrid[jn_hi] - n_next) / n_cw, 0.0, 1.0)
        w_m_lo = np.clip((dist_mGrid[jm_hi] - m_next) / m_cw, 0.0, 1.0)
        w_n_hi = 1.0 - w_n_lo
        w_m_hi = 1.0 - w_m_lo

        # Analytical weight derivatives
        dw_n = np.where(at_n_bnd, 0.0, -dn / n_cw)
        dw_m = np.where(at_m_bnd, 0.0, -dm / m_cw)

        coef = D_flat * (prob * LivPrb)

        # Flat destination indices for the four bilinear cells
        i_ll = jn_lo * n_m + jm_lo
        i_lh = jn_lo * n_m + jm_hi
        i_hl = jn_hi * n_m + jm_lo
        i_hh = jn_hi * n_m + jm_hi

        # Signs verified: total change per source point sums to zero
        curlyD += np.bincount(i_ll, weights=coef * ( dw_n * w_m_lo + w_n_lo * dw_m), minlength=n_states)
        curlyD += np.bincount(i_lh, weights=coef * ( dw_n * w_m_hi - w_n_lo * dw_m), minlength=n_states)
        curlyD += np.bincount(i_hl, weights=coef * (-dw_n * w_m_lo + w_n_hi * dw_m), minlength=n_states)
        curlyD += np.bincount(i_hh, weights=coef * (-dw_n * w_m_hi - w_n_hi * dw_m), minlength=n_states)

    return curlyD



@numba.njit(parallel=True)
def gen_tran_matrix_dur_CarCrash(dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
                                  shk_prbs, perm_shks, tran_shks, carcrash_shks,
                                  LivPrb, NewBornDist):
    TranMatrix = np.zeros((len(dist_nGrid) * len(dist_mGrid),
                           len(dist_nGrid) * len(dist_mGrid)))
    for i in numba.prange(len(dist_nGrid)):
        for j in numba.prange(len(dist_mGrid)):
            nNext_ij = carcrash_shks * (nNext_unshocked[i, j] / perm_shks)
            mNext_ij = bNext[i, j] / perm_shks + tran_shks
            TranMatrix[:, i * len(dist_mGrid) + j] += (
                LivPrb * jump_to_grid_dur(nNext_ij, mNext_ij, shk_prbs, dist_nGrid, dist_mGrid)
                + (1.0 - LivPrb) * NewBornDist
            )
    return TranMatrix


@numba.njit
def gen_tran_matrix_dur_sparse(dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
                                shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist):
    nStates = len(dist_nGrid) * len(dist_mGrid)
    rows = List()
    cols = List()
    data = List()
    for i in numba.prange(len(dist_nGrid)):
        for j in range(len(dist_mGrid)):
            col = i * len(dist_mGrid) + j
            nNext_ij = nNext_unshocked[i, j] / perm_shks
            mNext_ij = bNext[i, j] / perm_shks + tran_shks
            probs = (
                LivPrb * jump_to_grid_dur(nNext_ij, mNext_ij, shk_prbs, dist_nGrid, dist_mGrid)
                + (1.0 - LivPrb) * NewBornDist
            )
            for r in range(nStates):
                if probs[r] != 0.0:
                    rows.append(r)
                    cols.append(col)
                    data.append(probs[r])
    return rows, cols, data, nStates




def compute_erg_dstn_sparse(transition_matrix: sp.csr_matrix):
    """
    Compute ergodic distribution from a sparse transition matrix.
    Memory-efficient alternative to compute_erg_dstn.
    """
    n = transition_matrix.shape[0]
    T = transition_matrix - sp.eye(n, format='csr')
    b = np.zeros(n, dtype=np.float64)
    b[0] = 1.0
    T.data[T.indptr[0]:T.indptr[1]] = 0.0
    T[0, :] = b
    s_hat = spla.spsolve(T, b)
    ergodic_distr = s_hat.real / np.sum(s_hat.real)
    return ergodic_distr


##########################################################################################################
### Integration / Taking Expectations
##########################################################################################################

def m_nrm_next(shocks, a_nrm, Rfree, PermGroFac):
    """Normalized market resources next period."""
    return Rfree / (PermGroFac * shocks[0]) * a_nrm + shocks[1]


def n_nrm_next(shocks, a_nrm, n_nrm, dDepr):
    """Normalized durable stock next period."""
    aux = np.ones(len(a_nrm)) * ((1 - dDepr) * n_nrm)
    return aux.reshape(len(a_nrm), 1) / (shocks[0])


def vp_next_vPFunc(shocks, vPFuncNext, aNrm, nNrmGrid, CRRA, dDepr, Rfree, PermGroFac):
    return shocks[0] ** (-CRRA) * vPFuncNext(
        n_nrm_next(shocks, aNrm, nNrmGrid, dDepr),
        m_nrm_next(shocks, aNrm, Rfree, PermGroFac)
    )


def vp_next_vPDFunc(shocks, inv_vPDFuncNext, aNrm, nNrmGrid, CRRA, dDepr, Rfree, PermGroFac):
    return shocks[0] ** (-CRRA) * (
        np.clip(
            inv_vPDFuncNext(
                n_nrm_next(shocks, aNrm, nNrmGrid, dDepr),
                m_nrm_next(shocks, aNrm, Rfree, PermGroFac)
            ), 0, None
        )
    ) ** -CRRA


def vFunc_next(shocks, utility_inv, inv_vFunc_next, aNrm, nNrmGrid, CRRA, dDepr, Rfree, PermGroFac):
    inv_vFunc_plus_array = inv_vFunc_next(
        n_nrm_next(shocks, aNrm, nNrmGrid, dDepr),
        m_nrm_next(shocks, aNrm, Rfree, PermGroFac)
    )
    vFunc_plus_array = utility_inv(inv_vFunc_plus_array, CRRA)
    return shocks[0] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA) * vFunc_plus_array


##########################################################################################################
### S,s Region Detection and Solution
##########################################################################################################

def sign_change_indices(arr):
    indices = []
    for i in range(1, arr.shape[0]):
        a = arr[i - 1]
        b = arr[i]
        sa = 1 if a > 0 else -1 if a < 0 else 0
        sb = 1 if b > 0 else -1 if b < 0 else 0
        if sa != sb and sa != 0 and sb != 0:
            indices.append(i - 1)
    return np.array(indices, dtype=np.int32)


def find_roots_in_interval(function, a_min, a_max, d, N):
    a_aux = construct_grid(0.0, a_max, N, 'exp_mult', 1)
    a_aux = a_aux + a_min
    d_aux = np.ones(N) * d
    function_array = function(a_aux, d_aux)
    function_array[np.isnan(function_array)] = 0
    sign_changes = np.where(np.diff(np.sign(function_array)))[0]
    brackets = [(a_aux[i], a_aux[i + 1]) for i in sign_changes]
    roots = []
    for bracket in brackets:
        # Re-evaluate endpoints without NaN→0 substitution to detect spurious brackets.
        # Spurious sign changes arise when the adjuster's borrowing constraint extends
        # below the natural constraint grid: EndOfPrdvP = ∞ there, so f = NaN → 0.
        # That 0→negative crossing is not a real root — skip it.
        fa = function(bracket[0], d)
        fb = function(bracket[1], d)
        if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0):
            continue
        result = sp_optimize.root_scalar(function, bracket=bracket, args=(d,))
        if result.converged:
            roots.append(result.root)
    return np.array(roots)



def calc_SS_fast_terminal(nNrmGrid, mNrmGrid, mNrmMinNow, inv_vFuncAdj_array, inv_vFuncKeep_array, tol):
    """
    Identifies S,s bounds (lower, center, upper) for terminal-period solution,
    supporting a variable borrowing constraint (mNrmMinNow per n point).
    Returns lSuS_array and lSuS_idx_array.
    """
    shape = (len(nNrmGrid), 3)
    lSuS_array = np.zeros(shape)
    lSuS_idx_array = np.zeros(shape, dtype=np.int32)
    for i_n in range(len(nNrmGrid)):
        start_idx = len(mNrmMinNow) - i_n if len(mNrmMinNow) > 1 else 0
        inv_vFunc_Diff = inv_vFuncAdj_array[i_n, start_idx:] - inv_vFuncKeep_array[i_n, start_idx:]
        sign_changes = sign_change_indices(inv_vFunc_Diff + tol)

        min_val = np.inf
        min_idx = -1
        for i in range(inv_vFunc_Diff.shape[0]):
            val = inv_vFunc_Diff[i]
            if val < min_val:
                min_val = val
                min_idx = i

        lSuS_array[i_n, 1] = mNrmGrid[min_idx + start_idx] if min_idx != -1 else mNrmMinNow[i_n]
        lSuS_idx_array[i_n, 1] = int(min_idx + start_idx if min_idx != -1 else 0)

        if len(sign_changes) <= 1:
            lSuS_array[i_n, 0] = lSuS_array[i_n, 1]
            lSuS_array[i_n, 2] = lSuS_array[i_n, 1]
            lSuS_idx_array[i_n, 0] = lSuS_idx_array[i_n, 1]
            lSuS_idx_array[i_n, 2] = lSuS_idx_array[i_n, 1]
        elif len(sign_changes) == 2:
            lS_idx = sign_changes[0]
            lSuS_array[i_n, 0] = mNrmGrid[lS_idx + start_idx] if lS_idx != 0 else mNrmMinNow[i_n]
            uS_idx = sign_changes[1]
            lSuS_array[i_n, 2] = mNrmGrid[uS_idx + start_idx] if uS_idx != 0 else mNrmMinNow[i_n]
            lSuS_idx_array[i_n, 0] = int(lS_idx + start_idx if lS_idx != 0 else 0)
            lSuS_idx_array[i_n, 2] = int(uS_idx + start_idx if uS_idx != 0 else 0)
        else:
            lS_candidates = np.where(inv_vFunc_Diff < -tol)[0]
            if len(lS_candidates) == 0:
                lS = 0
            else:
                lS_idx = np.where(lS_candidates <= min_idx)[-1]
                lS = lS_candidates[lS_idx[0]]
            uS_candidates = np.where(inv_vFunc_Diff[lS:] > -tol)[0]
            uS = len(mNrmGrid) - 1 if len(uS_candidates) == 0 else uS_candidates[0] + lS
            lSuS_array[i_n, 0] = mNrmGrid[lS + start_idx] if lS != 0 else mNrmMinNow[i_n]
            lSuS_array[i_n, 2] = mNrmGrid[uS + start_idx] if uS != 0 else mNrmMinNow[i_n]
            lSuS_idx_array[i_n, 0] = int(lS + start_idx if lS != 0 else 0)
            lSuS_idx_array[i_n, 2] = int(uS + start_idx if uS != 0 else 0)

    return lSuS_array, lSuS_idx_array



def durable_solution_function_FUES_fast_search_noSS(adjusting_array,
                                                     cFuncAdj_NM_array, dFuncAdj_NM_array, aFuncAdj_NM_array,
                                                     inv_vFuncAdj_NM_array, vPFuncAdj_NM_array, vPDFuncAdj_NM_array,
                                                     cFuncKeep_NM_array, dFuncKeep_NM_array, aFuncKeep_NM_array,
                                                     inv_vFuncKeep_NM_array, vPFuncKeep_NM_array, vPDFuncKeep_NM_array, tol):
    """
    Combines adjuster and keeper solutions using a value-comparison mask,
    without assuming a single S,s band.
    """
    mask_boolean = (inv_vFuncAdj_NM_array + tol >= inv_vFuncKeep_NM_array)
    cFunc_array = np.where(mask_boolean, cFuncAdj_NM_array, cFuncKeep_NM_array)
    dFunc_array = np.where(mask_boolean, dFuncAdj_NM_array, dFuncKeep_NM_array)
    aFunc_array = np.where(mask_boolean, aFuncAdj_NM_array, aFuncKeep_NM_array)
    inv_vFunc_array = np.where(mask_boolean, inv_vFuncAdj_NM_array, inv_vFuncKeep_NM_array)
    vPFunc_array = np.where(mask_boolean, vPFuncAdj_NM_array, vPFuncKeep_NM_array)
    vPDFunc_array = np.where(mask_boolean, vPDFuncAdj_NM_array, vPDFuncKeep_NM_array)
    adjusting_array = np.where(mask_boolean, np.ones_like(inv_vFuncAdj_NM_array),
                                np.zeros_like(inv_vFuncKeep_NM_array))
    return cFunc_array, dFunc_array, aFunc_array, inv_vFunc_array, vPFunc_array, vPDFunc_array, adjusting_array


##########################################################################################################
### Search Cost
##########################################################################################################

@njit(fastmath=True)
def fast_lognorm_pdf(x, s, mu):
    """Lognormal PDF with safe handling of x <= 0."""
    pdf = np.zeros_like(x)
    for i in range(x.size):
        if x[i] > 0:
            pdf[i] = (1.0 / (x[i] * s * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(x[i]) - mu) ** 2) / (2 * s ** 2)
            )
    return pdf


@njit(fastmath=True)
def compute_mixture_pdf(x_vals, s_emp, mu_emp, unemp_prob, s_unemp, mu_unemp, z):
    """
    Mixture PDF (employed + unemployed lognormal), applying scale factor z
    by shifting mu -> mu + log(z).
    """
    mu_emp_z = mu_emp + np.log(z)
    mu_unemp_z = mu_unemp + np.log(z)
    if unemp_prob > 0.0:
        pdf_emp = fast_lognorm_pdf(x_vals, s_emp, mu_emp_z)
        pdf_unemp = fast_lognorm_pdf(x_vals, s_unemp, mu_unemp_z)
        return (1.0 - unemp_prob) * pdf_emp + unemp_prob * pdf_unemp
    else:
        return fast_lognorm_pdf(x_vals, s_emp, mu_emp_z)


@njit(parallel=True, fastmath=True)
def compute_cost_matrix(search_cost_matrix, e_grid):
    """Builds a 3-D cost matrix (n x m x e) from a 2-D search cost matrix."""
    n_size, m_size = search_cost_matrix.shape
    e_size = len(e_grid)
    g_e = np.empty(e_size)
    for j in range(e_size):
        if e_grid[j] < 1.0:
            g_e[j] = e_grid[j] + np.log(1.0 - e_grid[j]) * (1.0 - e_grid[j])
        else:
            g_e[j] = e_grid[j]
    Costs_matrix = np.zeros((n_size, m_size, e_size))
    for i_n in prange(n_size):
        for i_m in range(m_size):
            cost_val = search_cost_matrix[i_n, i_m]
            for j in range(e_size):
                Costs_matrix[i_n, i_m, j] = cost_val * g_e[j]
            Costs_matrix[i_n, i_m, e_size - 1] = cost_val
    return Costs_matrix


@njit(parallel=True, fastmath=True)
def update_prob_grid_list_with_trunc(
    T_cycle, diff_grid, TranShkStd, UnempPrb, IncUnemp,
    mu_transformed_list, TranShk_min_list, z, IncUnemp_Std
):
    """
    Computes a list of normalized probability grids (one per period) using
    a truncated mixture lognormal PDF.
    """
    nb, nm = diff_grid.shape
    probGridList = []
    for t in range(T_cycle):
        s_emp = TranShkStd[t]
        mu_emp = np.log(mu_transformed_list[t]) - 0.5 * s_emp ** 2
        unemp_prob = UnempPrb[t]
        s_unemp = IncUnemp_Std
        mu_unemp = np.log(IncUnemp[t]) - 0.5 * s_unemp ** 2
        TranShk_min = TranShk_min_list[t]
        z_t = z[t]
        probGrid = np.empty((nb, nm))
        for i in prange(nb):
            row_vals = diff_grid[i]
            row_pdf = compute_mixture_pdf(row_vals, s_emp, mu_emp, unemp_prob, s_unemp, mu_unemp, z_t)
            for j in range(nm):
                if row_vals[j] < TranShk_min:
                    row_pdf[j] = 0.0
            row_sum = np.sum(row_pdf)
            if row_sum > 0.0:
                probGrid[i] = row_pdf / row_sum
            else:
                probGrid[i] = 0.0
        probGridList.append(probGrid)
    return probGridList


##########################################################################################################
### Sequence Space Jacobian (Fake News Algorithm)
##########################################################################################################

@numba.njit
def J_from_F(F):
    J = F.copy()
    for t in range(1, F.shape[0]):
        J[1:, t] += J[:-1, t - 1]
    return J


@numba.njit
def compute_ZerothCol(T, tranmat_t_zeroth_col_dense, tranmat_ss, dstn_t_zeroth_col,
                      a_ss_flat, c_ss_flat, d_ss_flat, dex_ss_flat, adj_ss_flat):

    A_t_no_sim = np.zeros(T)
    C_t_no_sim = np.zeros(T)
    D_t_no_sim = np.zeros(T)
    DEX_t_no_sim = np.zeros(T)
    ADJ_t_no_sim = np.zeros(T)

    for i in range(T):
        if i == 0:
            dstn_t_zeroth_col = np.dot(tranmat_t_zeroth_col_dense, dstn_t_zeroth_col)
        else:
            dstn_t_zeroth_col = np.dot(tranmat_ss, dstn_t_zeroth_col)

        A_t_no_sim[i] = np.dot(a_ss_flat, dstn_t_zeroth_col)
        C_t_no_sim[i] = np.dot(c_ss_flat, dstn_t_zeroth_col)
        D_t_no_sim[i] = np.dot(d_ss_flat, dstn_t_zeroth_col)
        DEX_t_no_sim[i] = np.dot(dex_ss_flat, dstn_t_zeroth_col)
        ADJ_t_no_sim[i] = np.dot(adj_ss_flat, dstn_t_zeroth_col)

    return A_t_no_sim, C_t_no_sim, D_t_no_sim, DEX_t_no_sim, ADJ_t_no_sim


@numba.njit
def FakeNews_njit(T, dx, a_ss, c_ss, d_ss, dex_ss, adj_ss, a_t, c_t, d_t, dex_t, adj_t,
                  tranmat_ss, tranmat_t, Dist_ss):
    a_ss_flat = a_ss.flatten()
    c_ss_flat = c_ss.flatten()
    d_ss_flat = d_ss.flatten()
    dex_ss_flat = dex_ss.flatten()
    adj_ss_flat = adj_ss.flatten()

    ########
    # STEP1 # of fake news algorithm
    ########
    Pol_Grid_deviations_shape = (T, a_ss.shape[0], a_ss.shape[1])
    da0_s = np.empty(Pol_Grid_deviations_shape)
    dc0_s = np.empty(Pol_Grid_deviations_shape)
    dd0_s = np.empty(Pol_Grid_deviations_shape)
    ddex0_s = np.empty(Pol_Grid_deviations_shape)
    dadj0_s = np.empty(Pol_Grid_deviations_shape)

    for i in range(T):
        da0_s[i] = a_t[T - i] - a_ss
        dc0_s[i] = c_t[T - i] - c_ss
        dd0_s[i] = d_t[T - i] - d_ss
        ddex0_s[i] = dex_t[T - i] - dex_ss
        dadj0_s[i] = adj_t[T - i] - adj_ss

    dA0_s = np.empty(T)
    dC0_s = np.empty(T)
    dD0_s = np.empty(T)
    dDEX0_s = np.empty(T)
    dADJ0_s = np.empty(T)

    for i in range(T):
        dA0_s[i] = np.dot(da0_s[i].flatten(), Dist_ss)
        dC0_s[i] = np.dot(dc0_s[i].flatten(), Dist_ss)
        dD0_s[i] = np.dot(dd0_s[i].flatten(), Dist_ss)
        dDEX0_s[i] = np.dot(ddex0_s[i].flatten(), Dist_ss)
        dADJ0_s[i] = np.dot(dadj0_s[i].flatten(), Dist_ss)

    A_curl_s = dA0_s / dx
    C_curl_s = dC0_s / dx
    D_curl_s = dD0_s / dx
    DEX_curl_s = dDEX0_s / dx
    ADJ_curl_s = dADJ0_s / dx

    dist_shape = (T, Dist_ss.size)
    dDist0_s = np.zeros(dist_shape)

    for i in range(T):
        dlambda0_s_i = tranmat_t[T - i] - tranmat_ss
        dDist0_s[i] = np.dot(dlambda0_s_i, Dist_ss)

    Dist_curl_s = dDist0_s / dx

    ########
    # STEP2 # of fake news algorithm
    ########
    exp_vecs_a = np.empty((T, a_ss_flat.size))
    exp_vecs_c = np.empty((T, c_ss_flat.size))
    exp_vecs_d = np.empty((T, d_ss_flat.size))
    exp_vecs_dex = np.empty((T, dex_ss_flat.size))
    exp_vecs_adj = np.empty((T, adj_ss_flat.size))

    exp_vec_a = a_ss_flat
    exp_vec_c = c_ss_flat
    exp_vec_d = d_ss_flat
    exp_vec_dex = dex_ss_flat
    exp_vec_adj = adj_ss_flat

    for i in range(T):
        exp_vecs_a[i] = exp_vec_a
        exp_vec_a = np.dot(exp_vec_a, tranmat_ss)

        exp_vecs_c[i] = exp_vec_c
        exp_vec_c = np.dot(exp_vec_c, tranmat_ss)

        exp_vecs_d[i] = exp_vec_d
        exp_vec_d = np.dot(exp_vec_d, tranmat_ss)

        exp_vecs_dex[i] = exp_vec_dex
        exp_vec_dex = np.dot(exp_vec_dex, tranmat_ss)

        exp_vecs_adj[i] = exp_vec_adj
        exp_vec_adj = np.dot(exp_vec_adj, tranmat_ss)

    #########
    # STEP3 # of the algorithm (equation 26 of the published paper)
    #########
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

    for i in range(T - 1):
        for j in range(T):
            Curl_F_A[i + 1, j] = np.dot(exp_vecs_a[i], Dist_curl_s[j])
            Curl_F_C[i + 1, j] = np.dot(exp_vecs_c[i], Dist_curl_s[j])
            Curl_F_D[i + 1, j] = np.dot(exp_vecs_d[i], Dist_curl_s[j])
            Curl_F_DEX[i + 1, j] = np.dot(exp_vecs_dex[i], Dist_curl_s[j])
            Curl_F_ADJ[i + 1, j] = np.dot(exp_vecs_adj[i], Dist_curl_s[j])

    ########
    # STEP4 # of the algorithm
    ########
    J_A = J_from_F(Curl_F_A)
    J_C = J_from_F(Curl_F_C)
    J_D = J_from_F(Curl_F_D)
    J_DEX = J_from_F(Curl_F_DEX)
    J_TEX = J_C + J_DEX
    J_ADJ = J_from_F(Curl_F_ADJ)

    return J_A, J_C, J_D, J_DEX, J_TEX, J_ADJ


##########################################################################################################
### Directed Search
##########################################################################################################


@njit(parallel=True, fastmath=True)
def directed_search_function_scaled_matrix(nNrmGridNow, mNrmGridTotal, mNrmGridMin, e_grid,
                                            Costs_matrix, vFuncAdj_SOL_array, vFuncKeep_SOL_array,
                                            cFuncAdj_SOL_array, cFuncKeep_SOL_array,
                                            dFuncAdj_SOL_array, dFuncKeep_SOL_array,
                                            aFuncAdj_SOL_array, aFuncKeep_SOL_array,
                                            vPFuncAdj_SOL_array, vPFuncKeep_SOL_array,
                                            vPDFuncAdj_SOL_array, vPDFuncKeep_SOL_array):
    n_size = len(nNrmGridNow)
    m_size = len(mNrmGridTotal)

    e_matrix = np.zeros((n_size, m_size))
    vFunc_array = np.zeros((n_size, m_size))
    DirectedSearchCost_array = np.zeros((n_size, m_size))

    vFunc_min = vFuncAdj_SOL_array[0, 0]

    for i_n in prange(n_size):
        start_idx = len(mNrmGridMin) - i_n if len(mNrmGridMin) > 0 else 0
        for i_m in prange(m_size):
            if i_m < start_idx:
                e_matrix[i_n, i_m] = 1
                vFunc_array[i_n, i_m] = -np.inf
                continue

            vFuncAdj_aux = vFuncAdj_SOL_array[i_n, i_m]
            vFuncKeep_aux = vFuncKeep_SOL_array[i_n, i_m]

            if nNrmGridNow[i_n] == 0:
                e_matrix[i_n, i_m] = 1
                vFunc_array[i_n, i_m] = vFuncAdj_aux - Costs_matrix[i_n, i_m][-1]
                DirectedSearchCost_array[i_n, i_m] = Costs_matrix[i_n, i_m][-1]
            elif vFuncKeep_aux > vFuncAdj_aux:
                e_matrix[i_n, i_m] = 0
                vFunc_array[i_n, i_m] = vFuncKeep_aux
                DirectedSearchCost_array[i_n, i_m] = 0
            elif vFuncKeep_aux == vFunc_min and vFuncAdj_aux == vFunc_min:
                e_matrix[i_n, i_m] = 1
                vFunc_array[i_n, i_m] = vFuncAdj_aux - Costs_matrix[i_n, i_m][-1]
                DirectedSearchCost_array[i_n, i_m] = Costs_matrix[i_n, i_m][-1]
            else:
                values = e_grid * vFuncAdj_aux + (1 - e_grid) * vFuncKeep_aux - Costs_matrix[i_n, i_m]
                values[np.isnan(values)] = -np.inf
                max_position = np.argmax(values)
                e_matrix[i_n, i_m] = e_grid[max_position]
                vFunc_array[i_n, i_m] = values[max_position]
                DirectedSearchCost_array[i_n, i_m] = Costs_matrix[i_n, i_m, max_position]

    cFunc_array = e_matrix * cFuncAdj_SOL_array + (1 - e_matrix) * cFuncKeep_SOL_array
    dFunc_array = e_matrix * dFuncAdj_SOL_array + (1 - e_matrix) * dFuncKeep_SOL_array
    aFunc_array = e_matrix * aFuncAdj_SOL_array + (1 - e_matrix) * aFuncKeep_SOL_array
    vPFunc_array = e_matrix * vPFuncAdj_SOL_array + (1 - e_matrix) * vPFuncKeep_SOL_array
    vPDFunc_array = e_matrix * vPDFuncAdj_SOL_array + (1 - e_matrix) * vPDFuncKeep_SOL_array

    return e_matrix, DirectedSearchCost_array, vFunc_array, cFunc_array, dFunc_array, aFunc_array, vPFunc_array, vPDFunc_array


@njit(parallel=True, fastmath=True)
def directed_search_function_scaled_matrix_limit(nNrmGridNow, mNrmGridTotal, mNrmGridMin,
                                                  idx_nlimit, idx_mlimit, e_grid,
                                                  Costs_matrix, vFuncAdj_SOL_array, vFuncKeep_SOL_array,
                                                  cFuncAdj_SOL_array, cFuncKeep_SOL_array,
                                                  dFuncAdj_SOL_array, dFuncKeep_SOL_array,
                                                  aFuncAdj_SOL_array, aFuncKeep_SOL_array,
                                                  vPFuncAdj_SOL_array, vPFuncKeep_SOL_array,
                                                  vPDFuncAdj_SOL_array, vPDFuncKeep_SOL_array):
    n_size = len(nNrmGridNow)
    m_size = len(mNrmGridTotal)

    e_matrix = np.zeros((n_size, m_size))
    vFunc_array = np.zeros((n_size, m_size))
    DirectedSearchCost_array = np.zeros((n_size, m_size))

    vFunc_min = vFuncAdj_SOL_array[0, 0]

    for i_n in prange(n_size):
        start_idx = len(mNrmGridMin) - i_n if len(mNrmGridMin) > 1 else 0

        if idx_nlimit >= 0 and i_n > idx_nlimit:
            e_matrix[i_n] = 1
            vFunc_array[i_n, start_idx:] = vFuncAdj_SOL_array[i_n, start_idx:] - Costs_matrix[i_n, start_idx:, -1]
            DirectedSearchCost_array[i_n, start_idx:] = Costs_matrix[i_n, start_idx:, -1]
            continue

        for i_m in range(m_size):
            if i_m < start_idx:
                e_matrix[i_n, i_m] = 1
                vFunc_array[i_n, i_m] = -np.inf
                continue

            if idx_mlimit >= 0 and i_m > idx_mlimit:
                e_matrix[i_n, i_m:] = 1
                vFunc_array[i_n, i_m:] = vFuncAdj_SOL_array[i_n, i_m:] - Costs_matrix[i_n, i_m:, -1]
                DirectedSearchCost_array[i_n, i_m:] = Costs_matrix[i_n, i_m:, -1]
                break

            vFuncAdj_aux = vFuncAdj_SOL_array[i_n, i_m]
            vFuncKeep_aux = vFuncKeep_SOL_array[i_n, i_m]

            if nNrmGridNow[i_n] == 0:
                e_matrix[i_n, i_m] = 1
                vFunc_array[i_n, i_m] = vFuncAdj_aux - Costs_matrix[i_n, i_m, -1]
                DirectedSearchCost_array[i_n, i_m] = Costs_matrix[i_n, i_m, -1]
            elif vFuncKeep_aux > vFuncAdj_aux:
                e_matrix[i_n, i_m] = 0
                vFunc_array[i_n, i_m] = vFuncKeep_aux
                DirectedSearchCost_array[i_n, i_m] = 0
            elif vFuncKeep_aux == vFunc_min and vFuncAdj_aux == vFunc_min:
                e_matrix[i_n, i_m] = 1
                vFunc_array[i_n, i_m] = vFuncAdj_aux - Costs_matrix[i_n, i_m, -1]
                DirectedSearchCost_array[i_n, i_m] = Costs_matrix[i_n, i_m, -1]
            else:
                values = e_grid * vFuncAdj_aux + (1 - e_grid) * vFuncKeep_aux - Costs_matrix[i_n, i_m]
                values[np.isnan(values)] = -np.inf
                max_position = np.argmax(values)
                e_matrix[i_n, i_m] = e_grid[max_position]
                vFunc_array[i_n, i_m] = values[max_position]
                DirectedSearchCost_array[i_n, i_m] = Costs_matrix[i_n, i_m, max_position]

    cFunc_array = e_matrix * cFuncAdj_SOL_array + (1 - e_matrix) * cFuncKeep_SOL_array
    dFunc_array = e_matrix * dFuncAdj_SOL_array + (1 - e_matrix) * dFuncKeep_SOL_array
    aFunc_array = e_matrix * aFuncAdj_SOL_array + (1 - e_matrix) * aFuncKeep_SOL_array
    vPFunc_array = e_matrix * vPFuncAdj_SOL_array + (1 - e_matrix) * vPFuncKeep_SOL_array
    vPDFunc_array = e_matrix * vPDFuncAdj_SOL_array + (1 - e_matrix) * vPDFuncKeep_SOL_array

    return e_matrix, DirectedSearchCost_array, vFunc_array, cFunc_array, dFunc_array, aFunc_array, vPFunc_array, vPDFunc_array
