"""
Simplified durable–nondurable consumption-saving model.

Port of the one-period notebook suite into a self-contained class API:

    from DurableModel import DurableConsumer
    # Search + stochastic prices (Combined notebook):
    model = DurableConsumer(cycles=1, SearchCost=0.1, sigma_eps=0.0, PriceShkStd=0.1)
    # EV Type I taste only (TasteShock notebook):
    model = DurableConsumer(cycles=1, SearchCost=0.0, sigma_eps=0.05, PriceShkStd=0.0)
    model.solve()
    sol = model.solution[0]   # first period; solution[-1] is terminal
    sol.cFunc(n, m), sol.dFunc(n, m), sol.adjusting(n, m)

Channels (turn off with zeros):
  - Directed search: SearchCost (mutually exclusive with EV taste)
  - EV Type I taste shocks: sigma_eps (closed-form logit / log-sum; no xi grid)
  - Stochastic prices: PriceShkStd

Supports the four pedagogical cases: taste only, search only, price only, search+price.
Requires LTV=0, ScaleBool=False, MaintCost=OpCost=0 (simplified).
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.sparse as sps

from HARK.econforgeinterp import LinearFast
from HARK.interpolation import ValueFuncCRRA, LowerEnvelope2D, MargValueFuncCRRA
from HARK.ConsumptionSaving.ConsIndShockModel import utility, utility_inv, utilityP_inv
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope
from HARK.distribution import expected, MeanOneLogNormal
from HARK.utilities import make_grid_exp_mult
from HARK.core import MetricObject

from utility import (
    construct_grid,
    CRRAutilityP_inv_njit,
    CRRAutility_njit,
    MargValueFuncCRRA_dur,
    find_roots_in_interval,
    HANKIncShkDstn_Truncated,
    vp_next_vPDFunc,
    vFunc_next,
    update_prob_grid_list_with_trunc,
    compute_cost_matrix,
    directed_search_function_scaled_matrix,
    FUES_allPol,
    jump_to_grid_dur,
    gen_tran_matrix_dur_sparse,
    compute_erg_dstn_sparse,
    J_from_F,
)


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
):
    """
    EV Type I (or hard-max) combine over keep vs adjust, then integrate over prices.

    For each price node p_k:
      v = sigma * log(exp(vKeep/sigma) + exp(vAdj(p_k)/sigma)),
      P^adj = logit weight on adjust,
    then average with price weights mu_p. ``sigma=0`` collapses to hard max.
    Adjuster arrays: (n_price, n_n, n_m). Keeper: (n_n, n_m).
    ``adjusting`` returned is E_p[P^adj].
    """
    n_price = cAdj.shape[0]
    keep_shape = cKeep.shape

    vKeep_level = utility(inv_vKeep, CRRA_)
    vAdj_level = utility(inv_vAdj, CRRA_)

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
):
    """
    Nested combine over price nodes with directed search (Combined notebook):
      v = max_eps[(1-eps) vKeep + eps * vAdj(p) - sigma(eps)],
    then integrate with mu_p.

    Shutoffs: PriceShkStd=0, SearchCost=0 (hard max via zero costs).
    Adjuster arrays: (n_price, n_n, n_m). Keeper: (n_n, n_m).
    """
    n_price = cAdj.shape[0]
    keep_shape = cKeep.shape

    vKeep_level = utility(inv_vKeep, CRRA_)
    vAdj_level = utility(inv_vAdj, CRRA_)

    search_cost_matrix = np.zeros(keep_shape)
    for i_n, n0 in enumerate(nGrid):
        n_aux = np.full(len(mGrid), n0)
        search_cost_matrix[i_n, :] = SearchCost_func_(n_aux + mGrid)
    Costs_matrix = compute_cost_matrix(search_cost_matrix, e_grid_)
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

        v_ij = np.array(v_ij, copy=True)
        vP_ij = np.array(vP_ij, copy=True)
        vPD_ij = np.array(vPD_ij, copy=True)

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
    adjC=0.3,
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
    # Channels (EV taste sigma_eps vs search are mutually exclusive)
    sigma_eps=0.0,
    SearchCost=0.1,
    ScaleBool=False,
    e_grid_count=101,
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
    max_cycles=500,  # iteration cap for cycles=0 (matches background max_cycles)
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
      - Directed search when ``SearchCost > 0`` (Combined / Search notebooks)
      - EV Type I taste shocks when ``sigma_eps > 0`` and ``SearchCost == 0``
      - Hard max when both are zero

    Stochastic prices (``PriceShkStd``) nest with either smoother.

    Parameters
    ----------
    cycles : int
        Horizon mode selector (background convention).
        ``cycles = 1`` solves a finite horizon with ``T_cycle`` periods before
        the terminal (solution has ``T_cycle + 1`` entries; ``solution[-1]`` is
        terminal). ``cycles = 0`` solves the infinite horizon: the one-period
        operator is iterated from the terminal seed until the solution
        ``distance`` falls below ``tol`` (or ``max_cycles`` iterations are
        reached).
    T_cycle : int
        Number of periods before the terminal when ``cycles >= 1``
        (T_cycle=1 solves T-1 and T).
    tol : float
        Convergence tolerance for the infinite-horizon iteration (cycles=0).
    max_cycles : int
        Maximum number of iterations for the infinite-horizon iteration.
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
        if self.SearchCost > 0.0 and self.sigma_eps > 0.0:
            raise ValueError(
                "Search and EV Type I taste shocks are alternative discrete smoothers; "
                "set either SearchCost=0 or sigma_eps=0."
            )

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

    def tau_func(self, x):
        return np.ones_like(np.asarray(x, dtype=float)) * self.tau

    # --------------------------------------------------------------- setup
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

        # Income
        self.IncShkDstn = HANKIncShkDstn_Truncated(
            sigma_Perm=self.PermShkStd,
            sigma_Tran=self.TranShkStd,
            n_approx_Perm=self.PermShkCount if self.PermShkStd > 0.0 else 1,
            n_approx_Tran=self.TranShkCount if self.TranShkStd > 0.0 else 1,
            UnempPrb=self.UnempPrb,
            IncUnemp=self.IncUnemp,
            taxrate=self.taxrate,
            TranShkMean_Func=TranShkMean_Func,
            labor=self.labor,
            wage=self.wage,
            truncation=self.truncation,
            neutral_measure=False,
        )
        self.ShkPrbsNext = self.IncShkDstn.pmv
        self.PermShkValsNext = self.IncShkDstn.atoms[0]
        self.TranShkValsNext = self.IncShkDstn.atoms[1]
        self.PermShkValsNext_only = np.unique(self.PermShkValsNext)
        self.ShkPrbsNext_PermShk = (
            np.ones_like(self.PermShkValsNext_only) / len(self.PermShkValsNext_only)
        )
        self.TranShkMinNext = float(np.min(self.TranShkValsNext))
        self.PermShkMinNext = float(np.min(self.PermShkValsNext))

        if self.use_TransitionMatrix:
            mNrmMax_income = (
                self.Rfree
                / (self.PermGroFac * self.PermShkMinNext)
                * self.aXtraMax
                + float(np.max(self.TranShkValsNext))
            )
            self.mNrmGrid_income = np.asarray(
                make_grid_exp_mult(0.0001, mNrmMax_income, self.TranMatrix_mCount, timestonest=1)
            ).flatten()
            self.bNrmGrid_income = np.asarray(
                make_grid_exp_mult(0.0, mNrmMax_income - 1.0, self.TranMatrix_bCount, timestonest=1)
            ).flatten()
            mu_transformed = (
                (1.0 - self.UnempPrb * self.IncUnemp) / (1.0 - self.UnempPrb)
                if self.UnempPrb < 1.0
                else 1.0
            )
            b_mesh, m_mesh = np.meshgrid(
                self.bNrmGrid_income, self.mNrmGrid_income, indexing="ij"
            )
            diff_grid = m_mesh - b_mesh
            z = TranShkMean_Func(self.taxrate, self.labor, self.wage)
            self.probGrid_income = update_prob_grid_list_with_trunc(
                1,
                diff_grid,
                np.array([self.TranShkStd]),
                np.array([self.UnempPrb]),
                np.array([self.IncUnemp if self.UnempPrb > 0 else 1e-12]),
                np.array([mu_transformed]),
                np.array([self.TranShkMinNext]),
                np.array([z]),
                self.IncUnemp_Std,
            )[0]
        else:
            self.mNrmGrid_income = None
            self.bNrmGrid_income = None
            self.probGrid_income = None

        self._updated = True
        discrete = (
            "search" if self.SearchCost > 0
            else ("EV taste" if self.sigma_eps > 0 else "hard max")
        )
        self._log(
            f"Updated. Discrete: {discrete}"
            + (f" (sigma_eps={self.sigma_eps})" if self.sigma_eps > 0 and self.SearchCost == 0 else "")
            + (f" (S={self.SearchCost})" if self.SearchCost > 0 else "")
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
        EV Type I / hard-max via sigma_eps (TasteShock / Price / baseline notebooks).
        """
        if self.SearchCost > 0.0:
            return combine_search_price(
                self.PriceShkVals, self.PriceShkWeights,
                cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
                cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
                nGrid, mGrid, self.SearchCost_func, self.e_grid, self.CRRA,
            )
        return combine_ev_taste_price(
            self.PriceShkVals, self.PriceShkWeights,
            cAdj, dAdj, aAdj, inv_vAdj, vPAdj, vPDAdj,
            cKeep, dKeep, aKeep, inv_vKeep, vPKeep, vPDKeep,
            self.sigma_eps, self.CRRA,
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
            for i_d in range(len(dNrmGrid)):
                EndOfPrdvPOne[i_d] = (
                    self.DiscFac * self.LivPrb * self.PermGroFac ** (-CRRA) * R_vec
                    * expected(
                        vp_next_vPDFunc,
                        self.IncShkDstn,
                        args=(
                            inv_vPFuncNext, aNrmGrid, dNrmGrid[i_d], CRRA,
                            self.dDepr, R_vec, self.PermGroFac,
                        ),
                    )
                )
                EndOfPrdvPTwo[i_d] = (
                    self.DiscFac * self.LivPrb * self.PermGroFac ** (-CRRA)
                    * (R_vec + self.OpCost + self.MaintCost * self.dDepr)
                    * expected(
                        vp_next_vPDFunc,
                        self.IncShkDstn,
                        args=(
                            inv_vPFuncNext, aNrmGrid, dNrmGrid[i_d], CRRA,
                            self.dDepr, R_vec, self.PermGroFac,
                        ),
                    )
                )
                EndOfPrdvPD[i_d] = (
                    self.DiscFac * self.LivPrb * self.PermGroFac ** (-CRRA)
                    * (1.0 - self.dDepr * (1.0 - self.MaintCost))
                    * expected(
                        vp_next_vPDFunc,
                        self.IncShkDstn,
                        args=(
                            inv_vPDFuncNext, aNrmGrid, dNrmGrid[i_d], CRRA,
                            self.dDepr, R_vec, self.PermGroFac,
                        ),
                    )
                )
                EndOfPrdv[i_d] = (
                    self.DiscFac * self.LivPrb
                    * expected(
                        vFunc_next,
                        self.IncShkDstn,
                        args=(
                            utility_inv, inv_vFunc_next, aNrmGrid, dNrmGrid[i_d],
                            CRRA, self.dDepr, R_vec, self.PermGroFac,
                        ),
                    )
                )
                # inv_EndOfPrdv[i_d, 1:] = utility_inv(EndOfPrdv[i_d, 1:], CRRA)
                inv_EndOfPrdv[i_d] = utility_inv(EndOfPrdv[i_d], CRRA)
        else:
            nNrmGridNow = dNrmGrid
            _n2d = np.repeat(nNrmGridNow[:, np.newaxis], len(self.mNrmGrid_income), axis=1)
            _m2d = np.tile(self.mNrmGrid_income[np.newaxis, :], (len(nNrmGridNow), 1))

            inv_vPFuncNext_array = np.clip(inv_vPFuncNext(_n2d, _m2d), 0, None)
            inv_vPDFuncNext_array = np.clip(inv_vPDFuncNext(_n2d, _m2d), 0, None)
            inv_vFuncNext_array = np.clip(inv_vFunc_next(_n2d, _m2d), 0, None)

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

            for i_n in range(len(nNrmGridNow)):
                n_aux_grid = np.ones(len(bNrmGrid_incomeNew)) * nNrmGridNow[i_n]
                idx = int(np.where(bNrmGrid_incomeNew >= BoroCnstNat2D_b[0])[0][0])
                b_slice = bNrmGrid_incomeNew[idx + 1 :]
                n_slice = n_aux_grid[idx + 1 :]
                Interm_inv_vP_withzero_array[idx + 1 :, i_n] = Interm_inv_vP_Interp_func(
                    b_slice, n_slice
                )
                Interm_inv_vPD_withzero_array[idx + 1 :, i_n] = Interm_inv_vPD_Interp_func(
                    b_slice, n_slice
                )
                Interm_inv_v_withzero_array[idx + 1 :, i_n] = Interm_inv_v_Interp_func(
                    b_slice, n_slice
                )

            Interm_inv_vP_Interp_func = LinearFast(
                Interm_inv_vP_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow]
            )
            Interm_vPfunc_Integration = MargValueFuncCRRA(Interm_inv_vP_Interp_func, CRRA)
            Interm_inv_vPD_Interp_func = LinearFast(
                Interm_inv_vPD_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow]
            )
            Interm_vPDfunc_Integration = MargValueFuncCRRA(Interm_inv_vPD_Interp_func, CRRA)
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

            vP_3d = Interm_vPfunc_Integration(b_flat, n_flat).reshape(nD, nSh, nA)
            vPD_3d = Interm_vPDfunc_Integration(b_flat, n_flat).reshape(nD, nSh, nA)
            vinv_flat = np.clip(Interm_inv_v_Interp_func(b_flat, n_flat), 0.0, None)
            v_3d = utility(vinv_flat, CRRA).reshape(nD, nSh, nA)

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
        vTransf = lambda x: np.exp(np.clip(x, -700, 700))

        cFuncKeepList = []
        decreasing_segments_counter = 0
        segments_total_counter = 0
        for i_n, n0 in enumerate(dNrmGrid):
            n_aux = np.full(len(aNrmGrid), n0)
            EndOfPrdvPNow = EndOfPrdvPOne[i_n]
            EndOfPrdvNow = EndOfPrdv[i_n]
            cKeep_egm = CRRAutilityP_inv_njit(EndOfPrdvPNow, n_aux, alpha, d_ubar, CRRA)
            mKeep_egm = aNrmGrid + cKeep_egm
            vKeep_egm = CRRAutility_njit(cKeep_egm, n_aux, alpha, d_ubar, CRRA) + EndOfPrdvNow

            vKeept = vTransf(vKeep_egm)
            start, end = calc_nondecreasing_segments(mKeep_egm, vKeept)

            # Decreasing segments = breaks between nondecreasing pieces (= kinks)
            n_nondec = len(start)
            n_dec = max(n_nondec - 1, 0)
            decreasing_segments_counter += n_dec
            segments_total_counter += n_nondec + n_dec

            segments, m_segments, c_segments = [], [], []
            for j in range(len(start)):
                idx = range(start[j], end[j] + 1)
                segments.append([mKeep_egm[idx], vKeept[idx]])
                m_segments.append(mKeep_egm[idx])
                c_segments.append(cKeep_egm[idx])

            if len(segments) == 0 or all(
                np.all(~np.isfinite(seg[1])) for seg in segments
            ):
                # Degenerate EGM: fall back to consuming all resources
                c_interp = np.array([0.0, 1.0])
                m_interp = np.array([BoroCnstNat, 1.0])
                cFuncKeepList.append(LinearFast(c_interp, [m_interp]))
                continue

            try:
                mKeep_upper, _, inds_upper = upper_envelope(segments)
            except ValueError:
                # upper_envelope can fail with All-NaN slices on degenerate segments
                c_interp = np.array([0.0, 1.0])
                m_interp = np.array([BoroCnstNat, 1.0])
                cFuncKeepList.append(LinearFast(c_interp, [m_interp]))
                continue

            cKeep_upper = np.full_like(mKeep_upper, np.nan)
            for k, c_segm in enumerate(c_segments):
                if start[-1] + 1 == len(mKeep_egm):
                    cKeep_upper = LinearFast(c_segments[0], [m_segments[0]])(mKeep_upper)
                    break
                elif len(m_segments[k]) == 1:
                    cKeep_upper[inds_upper == k] = c_segm
                else:
                    cKeep_upper[inds_upper == k] = LinearFast(c_segm, [m_segments[k]])(
                        mKeep_upper[inds_upper == k]
                    )

            if cKeep_upper[0] != 0.0:
                c_interp = np.insert(cKeep_upper, 0, 0.0)
                m_interp = np.insert(mKeep_upper, 0, BoroCnstNat)
            else:
                c_interp, m_interp = cKeep_upper, mKeep_upper
            cFuncKeepList.append(LinearFast(c_interp, [m_interp]))

        keep_shape = (len(nNrmGrid), len(mNrmGrid))
        cFuncKeepUnc_arr = np.zeros(keep_shape)
        dFuncKeepUnc_arr = np.zeros(keep_shape)
        for i_n, n0 in enumerate(nNrmGrid):
            dFuncKeepUnc_arr[i_n, :] = n0
            cFuncKeepUnc_arr[i_n, :] = np.maximum(cFuncKeepList[i_n](mNrmGrid), 0.0)

        cFuncKeepUnc = LinearFast(cFuncKeepUnc_arr, [nNrmGrid, mNrmGrid])
        dFuncKeepUnc = LinearFast(dFuncKeepUnc_arr, [nNrmGrid, mNrmGrid])
        cFuncKeepCnst = LinearFast(
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        )
        cFuncKeep = LowerEnvelope2D(cFuncKeepUnc, cFuncKeepCnst, nan_bool=False)
        dFuncKeep = dFuncKeepUnc

        inv_vFuncKeep_arr = np.zeros(keep_shape)
        aFuncKeep_arr = np.zeros(keep_shape)
        vPDFuncKeep_arr = np.zeros(keep_shape)
        cKeep_arr = np.zeros(keep_shape)
        for i_n, n0 in enumerate(nNrmGrid):
            n_aux = np.full(len(mNrmGrid), n0)
            cNrm = np.clip(cFuncKeep(n_aux, mNrmGrid), 0.0, None)
            aNrm = np.maximum(mNrmGrid - cNrm, BoroCnstNat)
            cKeep_arr[i_n, :] = cNrm
            aFuncKeep_arr[i_n, :] = aNrm
            vNrm = self.CRRAutility(cNrm, n_aux) + EndOfPrdvFunc(n_aux, aNrm)
            inv_vFuncKeep_arr[i_n, :] = utility_inv(vNrm, CRRA)
            vPDFuncKeep_arr[i_n, :] = self.CRRAutilityPD(cNrm, n_aux) + EndOfPrdvPDFunc(
                n_aux, aNrm
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
        dNrmGridAdj = construct_grid(
            dNrmGrid[0], dNrmGrid[-1], len(self.xNrmGrid), self.grid_type, self.NestFac
        )
        BoroCnstNat1D = 0.0

        def rootfinding_function(a, d, price):
            a = np.atleast_1d(np.asarray(a, dtype=float))
            d = np.atleast_1d(np.asarray(d, dtype=float))
            if d.size == 1:
                d = np.full_like(a, float(np.asarray(d).ravel()[0]))
            eop1 = EndOfPrdvPOneFunc(d, a)
            eopd = EndOfPrdvPDFunc(d, a)
            c = CRRAutilityP_inv_njit(eop1, d, alpha, d_ubar, CRRA)
            return self.CRRAutilityPD(c, d) - price * eop1 + eopd

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
            aAdj_list, cAdj_list, xAdj_list, dAdj_list, vAdj_list = [], [], [], [], []
            for d_aux in dNrmGridAdj:
                a_root = find_roots_in_interval(
                    lambda a, d, _p=p0: rootfinding_function(a, d, _p),
                    a_min, a_max, d_aux, N_scan,
                )
                if a_root.size == 0:
                    continue
                a_root = a_root[a_root >= -1e-8]
                if a_root.size == 0:
                    continue
                d_0 = np.full_like(a_root, d_aux)
                c_0 = CRRAutilityP_inv_njit(
                    EndOfPrdvPOneFunc(d_0, a_root), d_0, alpha, d_ubar, CRRA
                )
                x_0 = a_root + p0 * d_0 + c_0
                v_0 = self.CRRAutility(c_0, d_0) + EndOfPrdvFunc(d_0, a_root)
                aAdj_list.append(a_root)
                cAdj_list.append(c_0)
                xAdj_list.append(x_0)
                dAdj_list.append(d_0)
                vAdj_list.append(v_0)

            aAdj_array = np.concatenate(aAdj_list) if aAdj_list else np.array([])
            cAdj_array = np.concatenate(cAdj_list) if cAdj_list else np.array([])
            xAdj_array = np.concatenate(xAdj_list) if xAdj_list else np.array([])
            dAdj_array = np.concatenate(dAdj_list) if dAdj_list else np.array([])
            vAdj_array = np.concatenate(vAdj_list) if vAdj_list else np.array([])
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

        shape_nm = (nPrice, len(nNrmGrid), len(mNrmGrid))
        cFuncAdj_NM_arr = np.zeros(shape_nm)
        dFuncAdj_NM_arr = np.zeros(shape_nm)
        aFuncAdj_NM_arr = np.zeros(shape_nm)
        inv_vFuncAdj_NM_arr = np.zeros(shape_nm)
        vPFuncAdj_NM_arr = np.zeros(shape_nm)
        vPDFuncAdj_NM_arr = np.zeros(shape_nm)

        for i_p, p0 in enumerate(PriceShkVals):
            for i_n, n0 in enumerate(nNrmGrid):
                n_aux = np.full(len(mNrmGrid), n0)
                tau_nm = self.tau_func(n_aux + mNrmGrid)
                x_aux = np.clip(mNrmGrid + (1.0 - tau_nm) * n_aux, 0.0, None)
                c_i = np.maximum(cFuncAdj_list[i_p](x_aux), 0.0)
                d_i = np.maximum(dFuncAdj_list[i_p](x_aux), 0.0)
                cFuncAdj_NM_arr[i_p, i_n, :] = c_i
                dFuncAdj_NM_arr[i_p, i_n, :] = d_i
                aFuncAdj_NM_arr[i_p, i_n, :] = np.maximum(x_aux - c_i - p0 * d_i, 0.0)
                inv_vFuncAdj_NM_arr[i_p, i_n, :] = inv_vFuncAdj_1d_list[i_p](x_aux)
                vPFuncAdj_NM_arr[i_p, i_n, :] = self.CRRAutilityP(c_i, d_i)
                vPDFuncAdj_NM_arr[i_p, i_n, :] = (1.0 - tau_nm) * vPFuncAdj_NM_arr[i_p, i_n, :]

        i_p0 = int(np.argmin(np.abs(PriceShkVals - 1.0)))
        cFuncAdj_NM = LinearFast(cFuncAdj_NM_arr[i_p0], [nNrmGrid, mNrmGrid])
        dFuncAdj_NM = LinearFast(dFuncAdj_NM_arr[i_p0], [nNrmGrid, mNrmGrid])

        adj_share_removed = (
            adj_removedpoints_counter / adj_totalpoints_counter
            if adj_totalpoints_counter
            else float("nan")
        )
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

    # ---------------------------------------------------------------- solve
    def solve(self):
        """
        Solve the model.

        Finite horizon (``cycles >= 1``): sets ``self.solution`` to a list of
        length ``T_cycle + 1``, where ``solution[t]`` is period t (0 = first) and
        ``solution[-1]`` is terminal.

        Infinite horizon (``cycles == 0``): iterates the one-period operator
        from the terminal seed until ``sol.distance(prev) < self.tol`` (using the
        solution ``distance_criteria``) or ``self.max_cycles`` is reached. Sets
        ``self.solution`` to a single-element list holding the converged (limit)
        solution.
        """
        self.update()
        if not self.analytic_terminal and self.solution_terminal is not None:
            terminal = self.solution_terminal
        else:
            terminal = self.solve_terminal()
            self.solution_terminal = terminal

        if int(self.cycles) == 0:
            prev = terminal
            for it in range(int(self.max_cycles)):
                sol = self.solve_one_period(prev)
                dist = sol.distance(prev)
                self._log(f"Infinite-horizon iter {it}: distance = {dist:.3e}")
                if dist < self.tol:
                    self._log(f"Converged after {it + 1} iterations (distance {dist:.3e}).")
                    self.solution = [sol]
                    return self.solution
                prev = sol
            self._log(
                f"Warning: reached max_cycles={self.max_cycles} without converging "
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
        index T_cycle-1 = the period just before the terminal). Only the
        "direct" parameters consumed in ``_end_of_period`` are supported.

        Produces ``self.solution`` of length ``T_cycle + 1`` where
        ``solution[k]`` is period k and ``solution[-1]`` is the terminal seed.
        """
        self.update()
        if self.solution_terminal is None:
            raise ValueError(
                "_solve_perturbed_finite requires self.solution_terminal to be set."
            )
        T = int(self.T_cycle)
        saved = {k: getattr(self, k) for k in param_path}
        sols = [self.solution_terminal]
        try:
            # Backward step j produces period index (T-1-j): j=0 -> period T-1.
            for j in range(T):
                period = T - 1 - j
                for k, path in param_path.items():
                    setattr(self, k, path[period])
                sols.insert(0, self.solve_one_period(sols[0]))
        finally:
            for k, v in saved.items():
                setattr(self, k, v)
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

    def calc_transition_matrix_dur(self, shk_dstn=None):
        """
        Evaluate solved policies on the histogram grid and build the transition
        matrix. For ``cycles=0`` a single matrix is stored; for the finite case a
        list over ``T_cycle`` periods is stored.

        No CarCrash channel and no Harmenberg neutral measure in the simplified
        port (default has degenerate permanent shocks).
        """
        if shk_dstn is None:
            shk_dstn = self.IncShkDstn

        if int(self.cycles) == 0:
            out = self._tran_matrix_one_period(self.solution[0], shk_dstn)
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
                out = self._tran_matrix_one_period(self.solution[k], shk_dstn)
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

        return self.A_ss, self.C_ss, self.D_ss, self.DEX_ss, self.TEX_ss, self.ADJ_ss

    # --------------------------------------------------- sequence-space Jacobian
    # Parameters whose Jacobian is currently supported (consumed directly in
    # _end_of_period, so time variation is handled by set-and-restore).
    _JAC_SUPPORTED = ("Rfree", "RSpread", "DiscFac", "PermGroFac")
    # Not implemented yet (need a per-period income process rebuild, or the
    # newborn/survival distribution which is trivial at LivPrb=1).
    _JAC_NOT_IMPLEMENTED = (
        "LivPrb", "PermShkStd", "TranShkStd", "UnempPrb", "IncUnemp",
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

        Only the "direct" parameters in ``_JAC_SUPPORTED`` are implemented; other
        shocks print a not-implemented message and return ``None``.
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
        ss_val = getattr(self, shk_param)

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


__all__ = [
    "DurableConsumer",
    "DurableConsumerSolution",
    "durable_defaults",
    "calcLogSumChoiceProbs",
    "combine_ev_taste_price",
    "combine_search_price",
]
