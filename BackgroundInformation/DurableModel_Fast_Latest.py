"""
This file contains classes and functions for representing, solving, and simulating
a consumer type with idiosyncratic shocks to permanent and transitory income,
who can consume durable and non-durable goods. There is an adjustment cost for
adjusting the durable stock as well as a depreciation rate.

Author:
Adrian Monninger

Using Code from Tianyang He: https://github.com/econ-ark/HARK/pull/825/files
and Jeppe Druedahl: https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks

Structure is based on ConsRiskyContribModel from Mateo

########################################################################################################################
Improvements:
1. Adjuster problem in last period: analytical solution instead of optimisation problem
2. Adjustment is evaluated at array instead of function
3. Inverse Value Function is the actual inverse and not -1/vFunc
4. mNrmGrid is endogeneously created
5. vFunc is similar to IndShockConsumer if the latter is produced with a spline interpolation instead of a cubic one
6. Consumption Functions uses LowerEnvelope with a Constrained and Unconstrained function.
    - Constrained Part is similar to Terminal Solution
7. Natural Borrowing Constrained is established and the aXtraGrid is similar to IndShockConsumer
8. Renamed uPFunc to vPFunc
9. Fixed vPFunc: DID I?
10. Postdecision Functions work with expected
11. Solve the Keeper with Lower-Envelope. Solve the Adjuster over inv_vFunc over Keeper Solution function (Not only unconstraint part)
12. Adjust aNrmGrid and dNrmGrid such that m_next and n_next are smaller than state-space of solution
13. Use Rootfinding and FUES instead of Nested EGM
14. Solve Keeper and Adjuster in one loop (if possible)
15. Truncated Income Distribution
16. Add directed search
17. Add RSpread which is higher than RFree + Maintenance and Operating costs
 
Clean Version of DurableModel_FUES_Search_Fast_20241008


"""
import time
from matplotlib.colors import ListedColormap
import scipy.optimize as sp_optimize

import numpy as np
from joblib import Parallel, delayed
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, init_idiosyncratic_shocks
from HARK.econforgeinterp import LinearFast, DecayInterp
from HARK.interpolation import LowerEnvelope, LowerEnvelope2D, ValueFuncCRRA, MargValueFuncCRRA
import matplotlib.pyplot as plt
from HARK.distribution import (
    IndexDistribution
)
# Additional:
from HARK.core import MetricObject, NullFunc # Basic HARK features
from copy import deepcopy
from HARK.ConsumptionSaving.ConsIndShockModel import utility, utility_inv, utilityP_inv
from HARK.utilities import make_grid_exp_mult, construct_assets_grid
from HARK.distribution import expected, Lognormal
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope

import scipy as sp

from utilities_Durable import (
    # Grid construction
    construct_grid,
    # Utility functions
    CRRAutilityP_inv_njit, CRRAutility_njit,
    MargValueFuncCRRA_dur, inv_MargValueFuncCRRA_dur,
    # Transition matrix simulation
    jump_to_grid_dur, jump_to_grid_dur_shock, gen_tran_matrix_dur_CarCrash,
    gen_tran_matrix_dur_sparse, compute_erg_dstn_sparse,
    # Integration / expectations
    vp_next_vPFunc, vp_next_vPDFunc, vFunc_next,
    # S,s region detection and solution
    find_roots_in_interval, calc_SS_fast_terminal, 
    durable_solution_function_FUES_fast_search_noSS,
    # Income process / distributions
    HANKIncShkDstn_Truncated, HANKIncShkDstn_Truncated_Jacobian,
    LognormPermIncShk_Truncated,
    MixtureTranIncShk_HANK_Truncated, MixtureTranIncShk_HANK_Truncated_Jacobian,
    # Search cost
    compute_cost_matrix, update_prob_grid_list_with_trunc,
    # Sequence space Jacobian (fake news algorithm)
    FakeNews_njit, J_from_F, compute_ZerothCol,
    # Directed search
    directed_search_function_scaled_matrix,
    directed_search_function_scaled_matrix_limit,
)
# from fast_upper_envelope import fast_upper_envelope
from fast_upper_envelope_copy import fast_upper_envelope
from FUES import FUES, FUES_allPol

from numba import njit, prange

# ---------------------------------------------------------------------------
# GPU (CuPy) acceleration — optional, transparent fall-back to CPU
# ---------------------------------------------------------------------------
try:
    import cupy as _cp
    _GPU_AVAILABLE = _cp.cuda.is_available()
except (ImportError, Exception):
    _cp = None
    _GPU_AVAILABLE = False

# Set to True from a notebook to print a per-section timing table each period.
# Example: import DurableModel_Fast_Latest as _dfm; _dfm._PROFILE_DURABLE = True
_PROFILE_DURABLE = False

# ---------------------------------------------------------------------------
# GPU FP32 cache for probGrid_income.
# probGrid_income is a (nB × nM) dense matrix — every (b,m) cell gets a
# nonzero probability from the continuous log-normal income density f(m-b).
# Dense np.dot on CPU takes ~3 s/period (BLAS FP64 at ~50 GFLOPS effective).
# GPU FP32 cuBLAS at 15 TFLOPS is ~100× faster.  We cache the FP32 GPU array
# to avoid repeated H2D transfers (paid once for time_inv cases).
# Cache keyed on id(arr): safe for time_inv (same object) and time_vary
# (new array object each period → fresh upload).
# ---------------------------------------------------------------------------
_PROB_GPU_CACHE: dict = {}   # { id(arr): cp.ndarray float32 on GPU }

# ---------------------------------------------------------------------------
# GPU kernel warmup — called once at import time to pre-compile CuPy JIT
# kernels (searchsorted, bilinear arithmetic, power, clip).  Without this,
# the first period pays ~0.5 s of CuPy JIT compilation latency.
# ---------------------------------------------------------------------------
def _gpu_warmup():
    if not _GPU_AVAILABLE:
        return
    try:
        _xg = _cp.array([0.0, 1.0, 2.0], dtype=_cp.float64)
        _yg = _cp.array([0.0, 1.0, 2.0], dtype=_cp.float64)
        _fg = _cp.ones((3, 3), dtype=_cp.float64)
        _xi = _cp.array([0.5, 1.5], dtype=_cp.float64)
        _yi = _cp.array([0.5, 1.5], dtype=_cp.float64)
        _r  = _eval2d_batch_gpu(_xg, _yg, _fg, _xi, _yi)
        # warm up power, clip, sum kernels used in eop_batch
        _r2 = _cp.clip(_r, 0.0, None) ** (-2.0)
        _r3 = (_r ** (1.0 - 2.0)) / (1.0 - 2.0)
        _cp.sum(_r2)
        # warm up FP32 matmul used in dot3
        _A = _cp.ones((4, 4), dtype=_cp.float32)
        _B = _cp.ones((4, 4), dtype=_cp.float32)
        _cp.dot(_A, _B)
        _cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass

_gpu_warmup()

# ---------------------------------------------------------------------------
# Fast numba kernels for the adjuster root-finding loop
# ---------------------------------------------------------------------------
# Numba audit (GPU status):
#   _eval2d            : @njit — single-point scalar kernel; called inside other @njit
#                        functions. GPU not applicable (scalar dispatch).
#   _scan_rootfinding_2d : @njit(parallel=True) — parallel CPU scan over (nD,nA).
#                        Superseded by _gpu_foc_residual + _eval2d_batch_gpu when GPU
#                        is available. Retained as CPU fallback (not currently invoked
#                        because per-d a_min offsets require per-row scan grids).
#   _compute_lambda_arrays : @njit(parallel=True) — parallel CPU lambda assembly.
#                        Could be ported to GPU; currently fast enough on CPU.
#   _bisect_root       : @njit — single-bracket serial bisection. Superseded by
#                        _gpu_bisect_all for GPU path. Retained as CPU reference.
# ---------------------------------------------------------------------------

@njit
def _eval2d(x_grid, y_grid, f_val, xi, yi):
    """
    2D bilinear interpolation on a rectangular grid with linear extrapolation.
    Equivalent to LinearFast(f_val, [x_grid, y_grid])(xi, yi) but callable
    inside @njit(parallel=True) without the extrap_options restriction.
    """
    nx = len(x_grid)
    ny = len(y_grid)
    # Clamped index (extrapolates linearly when tx/ty falls outside [0,1])
    ix = min(max(int(np.searchsorted(x_grid, xi)) - 1, 0), nx - 2)
    iy = min(max(int(np.searchsorted(y_grid, yi)) - 1, 0), ny - 2)
    tx = (xi - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    ty = (yi - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    return ((1.0 - tx) * (1.0 - ty) * f_val[ix,     iy    ]
            +       tx  * (1.0 - ty) * f_val[ix + 1, iy    ]
            + (1.0 - tx) *       ty  * f_val[ix,     iy + 1]
            +       tx  *       ty   * f_val[ix + 1, iy + 1])


def _eval2d_batch_gpu(x_grid_g, y_grid_g, f_val_g, xi_g, yi_g):
    """
    GPU-vectorised 2-D bilinear interpolation with linear extrapolation.
    Numerically equivalent to LinearFast(f_val, [x_grid, y_grid])(xi, yi)
    but runs entirely on the GPU via CuPy.

    Parameters
    ----------
    x_grid_g : cp.ndarray (nx,)   first-axis grid (e.g. b-grid or n-grid)
    y_grid_g : cp.ndarray (ny,)   second-axis grid
    f_val_g  : cp.ndarray (nx, ny) function values
    xi_g     : cp.ndarray (N,)    query points on x-axis
    yi_g     : cp.ndarray (N,)    query points on y-axis

    Returns
    -------
    cp.ndarray (N,)
    """
    cp = _cp
    nx = len(x_grid_g)
    ny = len(y_grid_g)
    ix = cp.clip(cp.searchsorted(x_grid_g, xi_g) - 1, 0, nx - 2).astype(cp.int32)
    iy = cp.clip(cp.searchsorted(y_grid_g, yi_g) - 1, 0, ny - 2).astype(cp.int32)
    x0 = x_grid_g[ix];  x1 = x_grid_g[ix + 1]
    y0 = y_grid_g[iy];  y1 = y_grid_g[iy + 1]
    tx = (xi_g - x0) / (x1 - x0)
    ty = (yi_g - y0) / (y1 - y0)
    return ((1.0 - tx) * (1.0 - ty) * f_val_g[ix,     iy    ]
            + (1.0 - tx) * ty        * f_val_g[ix,     iy + 1]
            + tx * (1.0 - ty)        * f_val_g[ix + 1, iy    ]
            + tx * ty                * f_val_g[ix + 1, iy + 1])


def _gpu_foc_residual(d_g, a_g, dGrid_g, aGrid_g,
                       eop1_g, eop2_g, eopd_g,
                       alpha, d_ubar, CRRA):
    """
    GPU-vectorised FOC residual for the adjuster root-finding scan.

    residual = Term1(C, D) - EndOfPrdvPTwo(D, a) + EndOfPrdvPD(D, a)

    where C solves the Euler equation:
        C = (EndOfPrdvPOne(D,a) / (alpha*(D+d_ubar)^((1-alpha)*(1-CRRA))))
              ^ (1 / (alpha*(1-CRRA) - 1))

    Parameters
    ----------
    d_g, a_g   : cp.ndarray (N,)   query (d, a) pairs
    dGrid_g, aGrid_g : cp.ndarray  EoP grid axes
    eop1_g, eop2_g, eopd_g : cp.ndarray (nD, nA)  raw EoP arrays on GPU
    alpha, d_ubar, CRRA : floats

    Returns cp.ndarray (N,) — residual values (NaN where interpolation
    extrapolates below the natural borrowing constraint).
    """
    eP1 = _eval2d_batch_gpu(dGrid_g, aGrid_g, eop1_g, d_g, a_g)
    eP2 = _eval2d_batch_gpu(dGrid_g, aGrid_g, eop2_g, d_g, a_g)
    ePD = _eval2d_batch_gpu(dGrid_g, aGrid_g, eopd_g, d_g, a_g)
    denom  = alpha * (d_g + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA))
    C_exp  = 1.0 / (alpha * (1.0 - CRRA) - 1.0)
    C      = (eP1 / denom) ** C_exp
    exp_C  = alpha * (1.0 - CRRA)
    exp_PD = (1.0 - alpha) * (1.0 - CRRA) - 1.0
    Term1  = (1.0 - alpha) * C ** exp_C * (d_g + d_ubar) ** exp_PD
    return Term1 - eP2 + ePD


def _gpu_bisect_all(d_brack_g, a_lo_g, a_hi_g, f_lo_sign_g,
                     dGrid_g, aGrid_g, eop1_g, eop2_g, eopd_g,
                     alpha, d_ubar, CRRA, n_iter=60):
    """
    Run ``n_iter`` bisection iterations over ALL brackets simultaneously on the
    GPU.  Returns a_roots (midpoint of final sub-intervals).

    After 60 iterations the interval width is reduced by 2^-60 ≈ 1e-18,
    which is well below machine epsilon.

    Parameters
    ----------
    d_brack_g   : cp.ndarray (B,)  d-value for each bracket
    a_lo_g, a_hi_g : cp.ndarray (B,)  initial bracket bounds
    f_lo_sign_g : cp.ndarray (B,)  sign of f at a_lo (±1 or 0)
    """
    a_lo   = a_lo_g.copy()
    a_hi   = a_hi_g.copy()
    f_lo_s = f_lo_sign_g.astype(_cp.float64)
    for _ in range(n_iter):
        a_mid  = 0.5 * (a_lo + a_hi)
        f_mid  = _gpu_foc_residual(d_brack_g, a_mid, dGrid_g, aGrid_g,
                                    eop1_g, eop2_g, eopd_g,
                                    alpha, d_ubar, CRRA)
        update_lo = (_cp.sign(f_mid) == f_lo_s)
        a_lo   = _cp.where(update_lo, a_mid, a_lo)
        f_lo_s = _cp.where(update_lo, _cp.sign(f_mid), f_lo_s)
        a_hi   = _cp.where(~update_lo, a_mid, a_hi)
    return 0.5 * (a_lo + a_hi)


@njit(parallel=True)
def _scan_rootfinding_2d(d_adj_grid, a_scan_grid,
                          d_eop_grid, a_eop_grid,
                          eop_P1, eop_P2, eop_PD,
                          alpha, d_ubar, CRRA):
    """
    Evaluate the adjuster FOC residual f(d, a) over an entire 2-D grid of
    (d_adj, a_scan) points in parallel.  Replaces the serial scan inside
    find_roots_in_interval for the main hot loop.

    Returns f_vals[nD, nA]:  positive → f > 0, negative → f < 0.
    Sign changes along axis-1 indicate root brackets.
    """
    nD = len(d_adj_grid)
    nA = len(a_scan_grid)
    f_vals = np.empty((nD, nA))
    for i_d in prange(nD):
        d = d_adj_grid[i_d]
        denom_base = alpha * (d + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA))
        exp_PD = (1.0 - alpha) * (1.0 - CRRA) - 1.0  # exponent for (d+d_ubar) in Term1
        exp_C  = alpha * (1.0 - CRRA)                 # exponent for C in Term1
        C_exp  = 1.0 / (alpha * (1.0 - CRRA) - 1.0)  # exponent in inverse Euler
        for i_a in range(nA):
            a = a_scan_grid[i_a]
            eP1  = _eval2d(d_eop_grid, a_eop_grid, eop_P1, d, a)
            # Inverse Euler equation for C
            C    = (eP1 / denom_base) ** C_exp
            Term1 = (1.0 - alpha) * C ** exp_C * (d + d_ubar) ** exp_PD
            Term2 = _eval2d(d_eop_grid, a_eop_grid, eop_P2, d, a)
            Term3 = _eval2d(d_eop_grid, a_eop_grid, eop_PD, d, a)
            f_vals[i_d, i_a] = Term1 - Term2 + Term3
    return f_vals


@njit(parallel=True)
def _compute_lambda_arrays(nNrmGridTotal, mNrmGridTotal, cFuncKeep_SOL_array,
                            mNrmMinNow_size,
                            d_grid, a_grid,
                            inv_eop_PD_vals, inv_lambdaM_vals,
                            CRRA):
    """
    Compute LambdaN_array and LambdaM_array for all (i_n, i_m) in parallel.

    LambdaN[i_n, i_m] = inv_EndOfPrdvPD(n, a_keep) ** (-CRRA)
    LambdaM[i_n, i_m] = inv_LambdaM(n, a_keep)    ** (-CRRA)

    where a_keep = mNrmGridTotal[i_m] - cFuncKeep_SOL_array[i_n, i_m].

    Replaces the serial Python loop in the SOL-array assembly block
    (TransitionMatrix branch) that calls 2-D LinearFast objects row by row.
    """
    nN = len(nNrmGridTotal)
    nM = len(mNrmGridTotal)
    LambdaN = np.zeros((nN, nM))
    LambdaM = np.zeros((nN, nM))
    for i_n in prange(nN):
        n = nNrmGridTotal[i_n]
        idx_start = (mNrmMinNow_size - i_n) if mNrmMinNow_size > 1 else 0
        for i_m in range(idx_start, nM):
            a_keep = mNrmGridTotal[i_m] - cFuncKeep_SOL_array[i_n, i_m]
            inv_l = _eval2d(d_grid, a_grid, inv_eop_PD_vals, n, a_keep)
            inv_m = _eval2d(d_grid, a_grid, inv_lambdaM_vals, n, a_keep)
            if inv_l > 0.0:
                LambdaN[i_n, i_m] = inv_l ** (-CRRA)
            if inv_m > 0.0:
                LambdaM[i_n, i_m] = inv_m ** (-CRRA)
    return LambdaN, LambdaM


@njit
def _bisect_root(d, a_lo, a_hi, f_lo,
                 d_eop_grid, a_eop_grid,
                 eop_P1, eop_P2, eop_PD,
                 alpha, d_ubar, CRRA, tol=1e-10, max_iter=60):
    """Bisect within [a_lo, a_hi] for the FOC residual."""
    denom_base = alpha * (d + d_ubar) ** ((1.0 - alpha) * (1.0 - CRRA))
    C_exp  = 1.0 / (alpha * (1.0 - CRRA) - 1.0)
    exp_PD = (1.0 - alpha) * (1.0 - CRRA) - 1.0
    exp_C  = alpha * (1.0 - CRRA)
    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        eP1   = _eval2d(d_eop_grid, a_eop_grid, eop_P1, d, a_mid)
        C     = (eP1 / denom_base) ** C_exp
        f_mid = ((1.0 - alpha) * C ** exp_C * (d + d_ubar) ** exp_PD
                 - _eval2d(d_eop_grid, a_eop_grid, eop_P2, d, a_mid)
                 + _eval2d(d_eop_grid, a_eop_grid, eop_PD,  d, a_mid))
        if abs(a_hi - a_lo) < tol:
            break
        if f_lo * f_mid <= 0.0:
            a_hi = a_mid
        else:
            a_lo = a_mid
            f_lo = f_mid
    return 0.5 * (a_lo + a_hi)


def TranShkMean_Func(taxrate, labor, wage):
    z = (1 - taxrate) * labor * wage
    return z

########################################################################################################################
# Make a dictionary to specify an idiosyncratic income shocks consumer
init_durable = dict(
    init_idiosyncratic_shocks,
    **{
        "alpha": 0.9, # Cobb-Douglas parameter for non-durable good consumption in utility function
        "dDepr": 0.1, # Depreciation Rate of Durable Stock
        "adjC": 0.15, # Adjustment costs
        "d_ubar": 1e-4, # Minimum durable stock for utility function
        # For Grids
        "nNrmMin": 0.0,
        "nNrmMax": 10,
        "nNrmCount": 100,
        # "mNrmMin": 0.0,
        # "mNrmMax": 10,
        "mNrmCount": 100,
        "xNrmCount": 200,
        "BoroCnstdNrm": 0,  # Borrowing Constraint of durable goods.
        "tol": 1e-6, # Tolerance for optimizer/ Acceptable difference before switching from adjuster to keeper
        "dNrmInitMean": 0, # Initial mean of end-of-period durable stock.
        "dNrmInitStd": 0, # Initial standard deviation of end-of-period durable stock.
        "nNrmInitMean": 1, # Initial mean of beginning-of-period durable stock.
        "mNrmInitMean": 1, # Initial mean of market resources.
        "NestFac": 3, # To construct grids differently
        "grid_type": 'exp_mult',
        "solve_terminal": True, # Boolean for solving terminal solution
        "terminal_method": 'terminal', # Either 'terminal_PF' for Perfect Foresight or 'terminal_Keep' for only Keep option
        "expected_method": 'expected', #Method to calculate next period values. Either "expected", "TransitionMatrix", or by Hand.
        "UpperEnvelope": 'DCEGM', # Which UpperEnvelope function do we use? DCEGM, FUES, or FUESBonn
        "BoroCnstArt":[None],
        "TranMatrixnCount": 100,
        "TranMatrixmCount": 200,
        "TranMatrixnMax": 1000,
        "TranMatrixmMax": 1000,
        "TranMatrixnMin": 0,
        "TranMatrixmMin": 1e-6,
        "Limit": False, # Boolean if Limit should be calculated or not
        "extrap_method": "decay_prop",
        "taxrate": [0.0],
        "labor": [1.0],
        "wage": [1.0],
        "TranShkMean_Func": [TranShkMean_Func],
        "truncation": 2,
        "SearchCost": 0.1,
        "ScaleBool": False, ### Sacling Adjustment Costs and Search Cost
        "ScaleFac": 1.0, ### Scaling Factor for Adjustment Costs and Search Cost
        "ScaleInt": [400.0, 500.0], ### Scale interval for Adjustment Costs
        "CarCrashBool": False, ### Boolean if Car Crash should be included
        "CarCrashPrb": 0.0, ### Probability of Car Crash
        "CarCrashMin": 0.0, ### Minimum durable stock after Car Crash
        "LTV": [1.0], ### Loan to Value Ratio: Percentage of durable stock that can be borrowed against
        "max_cycles": 500, # Maximum number of cycles to run
        "RSpread": [0.05], # Interest Rate for Borrowing
        ### maintenance parameter and operational costs
        "MaintCost": 0.8, # maintenance costs
        "OpCost": 0.048, # operational costs        
    }
)
# ### Adding taxrate, wage, labor
# init_durable['taxrate'] = [0.0] * init_durable['T_cycle']
# init_durable['labor'] = [1.0] * init_durable['T_cycle']
# init_durable['wage'] = [1.0] * init_durable['T_cycle']

class UnempRisk_Durable_Latest(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "alpha",
        "dDepr",
        "adjC",
        "d_ubar",
        "nNrmMin",
        "nNrmMax",
        "nNrmCount",
        "mNrmCount",
        "xNrmCount",
        "BoroCnstdNrm",
        "tol",
        "dNrmInitMean",
        "dNrmInitStd",
        "nNrmInitMean",
        "mNrmInitMean",
        "NestFac",
        "grid_type",
        "solve_terminal",
        "terminal_method",
        "expected_method",
        "UpperEnvelope",
        "Limit",
        "extrap_method",
        "truncation",
        "SearchCost",
        "ScaleBool",
        "ScaleFac",
        "ScaleInt",
        "CarCrashBool",
        "CarCrashPrb",
        "CarCrashMin",
        "max_cycles",
        "MaintCost",
        "OpCost"

    ]
    # Allow Rfree, Discount Rate, and Artificial Borrowing Constraint to vary over time.
    time_inv_.remove("DiscFac")
    time_inv_.remove("BoroCnstArt")
    time_vary_ = IndShockConsumerType.time_vary_ + ["Rfree", "RSpread", "DiscFac", "PermShkStd", "TranShkStd", "UnempPrb", "IncUnemp", "UnempPrbRet", "IncUnempRet", "BoroCnstArt", "taxrate", "wage", "labor", "LTV"]
    '''
    Adding the new state variable:
    nNrm: Beginning of period durable stock normalized by permanent income
    dNrm: End of period durable stock normalized by permanent income
    '''

    state_vars = IndShockConsumerType.state_vars + [
    "nNrm",
    "dNrm",
    ]

    def __init__(self, **kwds): # verbose=1, quiet=False,
        params = init_durable.copy()
        params.update(kwds)
        # Initialize a basic consumer type
        IndShockConsumerType.__init__(self, **params) # verbose=verbose, quiet=quiet,
        self.time_inv = deepcopy(self.time_inv_)

        self.def_utility_funcs()
        # Set the solver for the durable model, and update various constructed attributes
        self.solve_one_period = solve_DurableConsumer
        self.update()

    def def_utility_funcs(self):
        # i. U(C,D)
        u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
        self.CRRAutility = lambda C, D: utility(u_inner(C, D, self.d_ubar, self.alpha), self.CRRA)

        # ii. Inverse U(C,D)
        self.CRRAutility_inv = lambda C, D: utility_inv(self.CRRAutility(C, D), self.CRRA)

        # iii. uPC U(C,D) wrt C
        self.CRRAutilityP = lambda C, D: (
                                             ((self.alpha * C ** (self.alpha * (1 - self.CRRA) - 1)) * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA))))
        # iv. Inverse uPC U(C,D) wrt C
        self.CRRAutilityP_inv = lambda C, D: (
                (C / (self.alpha * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA)))) ** (1 / (self.alpha * (1 - self.CRRA) - 1)))

        # v. uPD U(C,D) wrt D
        self.CRRAutilityPD = lambda C, D: (
                    (1 - self.alpha) * (C ** (self.alpha * (1 - self.CRRA))) * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA) - 1))

    def pre_solve(self):
        # self.solve_fricitonless() # Solve Frictionless Solution
        # self.solve_frictionless_1D() # Solve Frictionless Solution
        if self.solve_terminal == True:
            if self.terminal_method == 'terminal_PF':
                self.solve_terminal_solution_PF()
                self.update_solution_terminal_PF()
            elif self.terminal_method == 'terminal_Adj':
                self.solve_terminal_solution_Adj()
                self.update_solution_terminal_Adj()
            elif self.terminal_method == 'terminal_Keep':
                self.solve_terminal_solution_Keep()
                self.update_solution_solution_Keep()
            elif self.terminal_method == 'terminal_Keep_PF':
                self.solve_terminal_solution_Keep_PF()
                self.update_solution_solution_Keep_PF()
            elif self.terminal_method == 'terminal_Frictionless':
                self.solve_terminal_solution_Frictionless()
            else:
                self.solve_terminal_solution()
                self.update_solution_terminal()

    def update_params(self):
        ### Check parameters
        # 1) If infinite horizon, set T_cycle to 1.
        if self.cycles == 0:
            self.T_cycle = 1
        
        # 2) Make sure that time varying parameters are of the right length. If not, assume that they are constant.
        for var_name in self.time_vary_:
            var_value = getattr(self, var_name)
            if isinstance(var_value, float):
                setattr(self, var_name, [var_value] * self.T_cycle)
            elif len(var_value) < self.T_cycle:
                setattr(self, var_name, [var_value[0]] * self.T_cycle)
            elif len(var_value) > self.T_cycle:
                setattr(self, var_name, var_value[:self.T_cycle])

    def update(self):
        '''
        We need to initialize multiple grids:
        1. Normalized durable stock grid: nNrmGrid
        2. Normalized market resource grid: mNrmGrid
        3. Normalized market resources + durable stock grid including adjustment costs: xNrmGrid
        4. Normalized asset grid: aNrmGrid
        '''
        timeBOOL = False
        if timeBOOL:
            tic = time.time()


        self.update_params()
        if timeBOOL:
            toc = time.time()
            print("update_params: {:.2f} seconds".format(toc - tic))

        if timeBOOL:
            tic = time.time()
        self.update_income_process_truncated()
        if timeBOOL:
            toc = time.time()
            print("update_income_process_truncated: {:.2f} seconds".format(toc - tic))
        # self.update_income_process()

        ### Beginning of period grids
        if timeBOOL:
            tic = time.time()
        self.updatenNrmGrid()
        self.updatemNrmGrid()
        self.updatexNrmGrid()
        if timeBOOL:
            toc = time.time()
            print("updatenNrmGrid, updatemNrmGrid, updatexNrmGrid: {:.2f} seconds".format(toc - tic))
        
        if timeBOOL:
            tic = time.time()
        self.scale_adjustment_costs() # Scale Adjustment Costs
        self.scale_search_costs() # Scale Search Costs
        if timeBOOL:
            toc = time.time()
            print("scale_search_costs: {:.2f} seconds".format(toc - tic))

        ### End of period grids
        self.update_assets_grid() # aXtraGrid
        self.updatedNrmGrid() # dNrmGrid

        if timeBOOL:
            tic = time.time()
        if self.expected_method == 'TransitionMatrix':
            self.updatemNrmGrid_income()
            self.updatebNrmGrid_income()
            # self.updateBM_TranMatrix()
            self.updateBM_TranMatrix_numba()
        else:
            self.mNrmGrid_income = np.array([0.0])
            self.bNrmGrid_income = np.array([0.0])
            nNrmGridNow = self.dNrmGrid * (1 - self.dDepr)
            self.zero_grid = -nNrmGridNow * (1 - self.tau_func(nNrmGridNow))# * self.LTV[0]
            self.probGrid_income = np.array([0.0])
            self.add_to_time_inv('mNrmGrid_income') 
            self.add_to_time_inv('bNrmGrid_income')
            self.add_to_time_inv('zero_grid')
            self.add_to_time_inv('probGrid_income')
        if timeBOOL:
            toc = time.time()
            print("probGrid_income: {:.2f} seconds".format(toc - tic))

    def update_income_process_truncated(self):
        """
        Updates this agent's income process based on his own attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        """
        # Make sure they have the same length
        if len(self.taxrate) < self.T_cycle:
            self.taxrate = [self.taxrate[0]] * self.T_cycle

        if len(self.TranShkMean_Func) < self.T_cycle:
            self.TranShkMean_Func = [self.TranShkMean_Func[0]] * self.T_cycle

        if len(self.labor) < self.T_cycle:
            self.labor = [self.labor[0]] * self.T_cycle

        if len(self.wage) < self.T_cycle:
            self.wage = [self.wage[0]] * self.T_cycle

        (
            IncShkDstn,
            PermShkDstn,
            TranShkDstn,
        ) = self.construct_lognormal_income_process_unemployment_truncated()

        self.IncShkDstn = IncShkDstn
        self.PermShkDstn = PermShkDstn
        self.TranShkDstn = TranShkDstn
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")

    def update_income_process_truncated_jacobian(self, UnempPrb_old, IncUnemp_old):
        """
        Updates this agent's income process based on his own attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        """
        # Make sure they have the same length
        if len(self.taxrate) < self.T_cycle:
            self.taxrate = [self.taxrate[0]] * self.T_cycle

        if len(self.TranShkMean_Func) < self.T_cycle:
            self.TranShkMean_Func = [self.TranShkMean_Func[0]] * self.T_cycle

        if len(self.labor) < self.T_cycle:
            self.labor = [self.labor[0]] * self.T_cycle

        if len(self.wage) < self.T_cycle:
            self.wage = [self.wage[0]] * self.T_cycle

        (
            IncShkDstn,
            PermShkDstn,
            TranShkDstn,
        ) = self.construct_lognormal_income_process_unemployment_truncated_jacobian(UnempPrb_old, IncUnemp_old)

        self.IncShkDstn = IncShkDstn
        self.PermShkDstn = PermShkDstn
        self.TranShkDstn = TranShkDstn
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")

    def updatenNrmGrid(self): #Grid of Normalized Durable Stock
        # Start with 0 and 0.001
        self.nNrmGrid = construct_grid(0.001, self.nNrmMax, self.nNrmCount - 1, self.grid_type, self.NestFac)
        self.nNrmGrid = np.unique(np.append(0.0, self.nNrmGrid))
        self.add_to_time_inv('nNrmGrid')      

    def updatemNrmGrid(self): # Grid of Normalized Market resouces if d\neq n
        """
        Assures that the solution grid for beginning of period market resources is larger than the grid for end of period assets.
        """
        #### For t_cycle == 1: self.Rfree is float and not an array
        if self.T_cycle == 1:
            mNrmMax = self.Rfree / (self.PermGroFac[0] * self.IncShkDstn[0].atoms[0][0]) *  self.aXtraMax + self.IncShkDstn[0].atoms[1][-1]
        else:
            mNrmMax = self.Rfree[0] / (self.PermGroFac[0] * self.IncShkDstn[0].atoms[0][0]) *  self.aXtraMax + self.IncShkDstn[0].atoms[1][-1]
        
        mNrmMin = self.aXtraMin*0.01
        self.mNrmGrid = construct_grid(mNrmMin, mNrmMax, self.aXtraCount - 1, self.grid_type, self.NestFac)
        self.mNrmGrid = np.sort(np.unique(np.append(0.0, self.mNrmGrid)))

        # ### Add borrowing constraint
        # self.mNrmGrid = np.unique(np.sort(np.append(self.mNrmGrid, self.zero_grid)))
        self.add_to_time_inv('mNrmGrid')

    def updatexNrmGrid(self): # x = m + (1 - Adjc) d
        # self.xNrmGrid = construct_grid(self.mNrmMin + (1 - self.adjC) * self.nNrmMin,self.mNrmMax + (1 - self.adjC) * self.nNrmMax ,self.xNrmCount, self.grid_type, self.NestFac)
        xNrmMin = self.mNrmGrid[1] # Start from zero
        xNrmMax = self.mNrmGrid[-1] + self.nNrmGrid[-1]
        self.xNrmGrid = construct_grid(xNrmMin, xNrmMax ,self.xNrmCount, self.grid_type, self.NestFac)
        self.xNrmGrid = np.sort(np.unique(np.append(0.0, self.xNrmGrid)))
        self.add_to_time_inv('xNrmGrid')

    def update_assets_grid(self):
        """
        Updates this agent's end-of-period assets grid by constructing a multi-
        exponentially spaced grid of aXtra values.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.aXtraGrid = construct_assets_grid(self)
        self.add_to_time_inv("aXtraGrid")

    def updatedNrmGrid(self): # Grid of End of Period Durable Stock
        """
        Updates this agent's end-of-period durable stock grid by constructing a multi-exponentially spaced grid.
        """
        self.dNrmGrid = construct_grid(self.nNrmGrid[0], self.nNrmMax, self.nNrmCount, self.grid_type, self.NestFac)
        self.dNrmGrid = np.sort(np.unique(np.append(0.0, self.dNrmGrid)))
        self.add_to_time_inv('dNrmGrid')

        # ### Create a zero grid for durable stock
        # self.zero_grid =  -self.dNrmGrid * (1 - self.dDepr) * (1 - self.tau_func(self.dNrmGrid)) * self.LTV[0]
        # self.add_to_time_inv('zero_grid')
        # print("Use LTV or 1 as agent could resell in next period?") --? Natural Borrowing Constraint is 1. Next period you can sell it no matter what.

    def updatemNrmGrid_income(self):
        """
        Updates this agent's dense beginning-of-period market resources grid by constructing a dense(!) multi-exponentially spaced grid.

        1. Calculate max based on a Grid and shocks
        2. Calculate Borrowing Constraint based on durable stock and human wealth.
        3. Create a grid from 0 to maximum
        4. Add an extra layer of grids for negative numbers.         
        """
        if self.T_cycle == 1:
            mNrmMax = self.Rfree / (self.PermGroFac[0] * self.IncShkDstn[0].atoms[0][0]) *  self.aXtraMax + self.IncShkDstn[0].atoms[1][-1]
        else:
            mNrmMax = self.Rfree[0] / (self.PermGroFac[0] * self.IncShkDstn[0].atoms[0][0]) *  self.aXtraMax + self.IncShkDstn[0].atoms[1][-1]

        # BoroCnstNatDur = -self.dNrmGrid * (1 - self.dDepr) * (1 - self.tau_func_1D(self.dNrmGrid)) * self.LTV[0] * self.Rfree[0] + 0.0001
        if self.LTV[0] == 0:
            BoroCnstNatDur = 0.0
        else:
            BoroCnstNatDur = -self.dNrmGrid * (1 - self.dDepr) * (1 - self.tau_func(self.dNrmGrid)) #self.zero_grid --> Natural Borrowing Constraint has always LTV of 1

        # Borrowing Constraint of assets is human wealth. At the limit, we can borrow against the limit of human wealth.
        # hNrm_Limit = 1/(1 - (self.PermGroFac[0]/self.Rfree[0]))
        hNrm_Limit = 0.0 #-0.5968473526227219
        BoroCnstNatAss = - hNrm_Limit

        BorCnstNatTotal = BoroCnstNatDur + BoroCnstNatAss  
        self.mNrmGrid_income = make_grid_exp_mult(0.0001, mNrmMax, 1010, timestonest=1)
        
        if self.LTV[0] != 0.0:
            ### Add the extra grid for each item in the BoroCnstNatDur grid
            # extra_grid = construct_grid(0.001, 5, 20, self.grid_type, self.NestFac)
            extra_grid = construct_grid(0.001, 5, 10, self.grid_type, self.NestFac)
            combined_grid_flat = np.sort((BorCnstNatTotal[:, np.newaxis] + extra_grid).flatten())

            self.mNrmGrid_income = np.unique(np.sort(np.append(self.mNrmGrid_income, combined_grid_flat)))
        self.mNrmGrid_income = self.mNrmGrid_income.flatten()

        self.add_to_time_inv('mNrmGrid_income')        

    def updatebNrmGrid_income(self):
        """
        Updates this agent's dense beginning-of-period market resources grid by constructing a dense(!) multi-exponentially spaced grid.

        1. Take m-Grid
        2. Shift zero-Grid by minimum of transitory shock
        """
       
        if self.T_cycle == 1:
            mNrmMax = self.Rfree / (self.PermGroFac[0] * self.IncShkDstn[0].atoms[0][0]) *  self.aXtraMax + self.IncShkDstn[0].atoms[1][-1]
        else:
            mNrmMax = self.Rfree[0] / (self.PermGroFac[0] * self.IncShkDstn[0].atoms[0][0]) *  self.aXtraMax + self.IncShkDstn[0].atoms[1][-1]

        TranShkValsNext = self.IncShkDstn[0].atoms[1]
        TranShk_min = np.min(TranShkValsNext) 
        TranShk_max = np.max(TranShkValsNext)

        # self.bNrmGrid_income = make_grid_exp_mult(0.0, mNrmMax - shock_min, 1000, timestonest=1)
        self.bNrmGrid_income = make_grid_exp_mult(0.0, mNrmMax - 1, 1000, timestonest=1)
        self.bNrmGrid_income = self.bNrmGrid_income

        if self.LTV[0] == 0.0:
            BoroCnstNatDur = 0.0
        else:
            BoroCnstNatDur = -self.dNrmGrid * (1 - self.dDepr) * (1 - self.tau_func(self.dNrmGrid)) #self.zero_grid --> Natural Borrowing Constraint has always LTV of 1

        # Borrowing Constraint of assets is human wealth. At the limit, we can borrow against the limit of human wealth.
        # hNrm_Limit = 1/(1 - (self.PermGroFac[0]/self.Rfree[0]))
        hNrm_Limit = 0.0 #-0.5968473526227219
        BoroCnstNatAss = - hNrm_Limit
        BorCnstNatTotal = BoroCnstNatDur + BoroCnstNatAss

        # BoroCnstNatDur = self.zero_grid
        # extra_grid = construct_grid(0.001, 5, 20, self.grid_type, self.NestFac)
        extra_grid = construct_grid(0.001, 5, 10, self.grid_type, self.NestFac)

        extra_grid = extra_grid - TranShk_min

        if self.LTV[0] != 0.0:
            ### Add the extra grid for each item in the BoroCnstNatDur grid
            combined_grid_flat = np.sort((BorCnstNatTotal[:, np.newaxis] + extra_grid).flatten())

            self.bNrmGrid_income = np.unique(np.sort(np.append(self.bNrmGrid_income, combined_grid_flat)))
        self.bNrmGrid_income = self.bNrmGrid_income.flatten()

        self.add_to_time_inv('bNrmGrid_income')    


    def updateBM_TranMatrix_numba(self):
        """
        Optimized version with Numba, includes truncation at TranShk_min.
        Avoids recomputing identical probGrid entries for repeated parameters.
        """
        b_grid = self.bNrmGrid_income
        m_grid = self.mNrmGrid_income
        IncUnemp_Std = 0.1

        lists = [self.TranShkStd, self.UnempPrb, self.IncUnemp, self.UnempPrbRet, self.IncUnempRet, 
                 self.TranShkMean_Func, self.taxrate, self.labor, self.wage]
        TimeVaryingInc = any(len(set(lst)) > 1 for lst in lists)

        mu_transformed_list = [
            (1.0 - self.UnempPrb[t] * self.IncUnemp[t]) / (1.0 - self.UnempPrb[t]) for t in range(self.T_cycle)
        ]
        TranShk_min_list = [np.min(self.IncShkDstn[t].atoms[1]) for t in range(self.T_cycle)]

        # Mesh only once
        b_mesh, m_mesh = np.meshgrid(self.bNrmGrid_income, self.mNrmGrid_income, indexing='ij')
        diff_grid = m_mesh - b_mesh  # shape (nb, nm)

        if TimeVaryingInc:
            probGridList = []
            cache = {}

            for t in range(self.T_cycle):
                # Create a hashable key for the parameter set
                key = (
                    self.TranShkStd[t],
                    self.UnempPrb[t],
                    self.IncUnemp[t],
                    mu_transformed_list[t],
                    TranShk_min_list[t],
                    self.TranShkMean_Func[t],
                    self.taxrate[t],
                    self.labor[t],
                    self.wage[t]
                )

                if key not in cache:
                    z = self.TranShkMean_Func[t](self.taxrate[t], self.labor[t], self.wage[t])
                    cache[key] = update_prob_grid_list_with_trunc(
                        1, diff_grid,
                        np.array([self.TranShkStd[t]]),
                        np.array([self.UnempPrb[t]]),
                        np.array([self.IncUnemp[t]]),
                        np.array([mu_transformed_list[t]]),
                        np.array([TranShk_min_list[t]]),
                        np.array([z]),
                        IncUnemp_Std
                    )[0]  # extract the single matrix

                probGridList.append(cache[key])

            self.probGrid_income = probGridList
            self.add_to_time_vary('probGrid_income')

        else:
            z = self.TranShkMean_Func[0](self.taxrate[0], self.labor[0], self.wage[0])
            probGridList = update_prob_grid_list_with_trunc(
                1, diff_grid,
                np.array([self.TranShkStd[0]]),
                np.array([self.UnempPrb[0]]),
                np.array([self.IncUnemp[0]]),
                np.array([mu_transformed_list[0]]),
                np.array([TranShk_min_list[0]]),
                np.array([z]),
                IncUnemp_Std
            )
            self.probGrid_income = probGridList[0]
            self.add_to_time_inv('probGrid_income')

    def updateBM_TranMatrix_numba_jacobian(self, UnempPrb_ss, IncUnemp_ss):
        """
        Optimized version with Numba, includes truncation at TranShk_min.
        Avoids recomputing identical probGrid entries for repeated parameters.
        """
        b_grid = self.bNrmGrid_income
        m_grid = self.mNrmGrid_income
        IncUnemp_Std = 0.1

        lists = [self.TranShkStd, self.UnempPrb, self.IncUnemp, self.UnempPrbRet, self.IncUnempRet, 
                 self.TranShkMean_Func, self.taxrate, self.labor, self.wage]
        TimeVaryingInc = any(len(set(lst)) > 1 for lst in lists)

        mu_transformed_list = [
            (1.0 - UnempPrb_ss * IncUnemp_ss) / (1.0 - UnempPrb_ss) for t in range(self.T_cycle)
        ]
        TranShk_min_list = [np.min(self.IncShkDstn[t].atoms[1]) for t in range(self.T_cycle)]

        # Mesh only once
        b_mesh, m_mesh = np.meshgrid(self.bNrmGrid_income, self.mNrmGrid_income, indexing='ij')
        diff_grid = m_mesh - b_mesh  # shape (nb, nm)

        if TimeVaryingInc:
            probGridList = []
            cache = {}

            for t in range(self.T_cycle):
                # Create a hashable key for the parameter set
                key = (
                    self.TranShkStd[t],
                    self.UnempPrb[t],
                    self.IncUnemp[t],
                    mu_transformed_list[t],
                    TranShk_min_list[t],
                    self.TranShkMean_Func[t],
                    self.taxrate[t],
                    self.labor[t],
                    self.wage[t]
                )

                if key not in cache:
                    z = self.TranShkMean_Func[t](self.taxrate[t], self.labor[t], self.wage[t])
                    cache[key] = update_prob_grid_list_with_trunc(
                        1, diff_grid,
                        np.array([self.TranShkStd[t]]),
                        np.array([self.UnempPrb[t]]),
                        np.array([self.IncUnemp[t]]),
                        np.array([mu_transformed_list[t]]),
                        np.array([TranShk_min_list[t]]),
                        np.array([z]),
                        IncUnemp_Std
                    )[0]  # extract the single matrix

                probGridList.append(cache[key])

            self.probGrid_income = probGridList
            self.add_to_time_vary('probGrid_income')

        else:
            z = self.TranShkMean_Func[0](self.taxrate[0], self.labor[0], self.wage[0])
            probGridList = update_prob_grid_list_with_trunc(
                1, diff_grid,
                np.array([self.TranShkStd[0]]),
                np.array([self.UnempPrb[0]]),
                np.array([self.IncUnemp[0]]),
                np.array([mu_transformed_list[0]]),
                np.array([TranShk_min_list[0]]),
                np.array([z]),
                IncUnemp_Std
            )
            self.probGrid_income = probGridList[0]
            self.add_to_time_inv('probGrid_income')


    # Scale adjustment costs
    def scale_adjustment_costs(self):

        """
        Compute a 1D scaled search cost function over x ∈ (-∞, x_max] such that they converge to 0.
        The function is:
            - tau_s for x <= p1 (including negative x)
            - quadratic decline between p1 and p2
            - zero after p2
        """
        if self.ScaleBool == False:
            # Constant function (always SearchCost)
            x_grid = self.xNrmGrid
            tau_x = np.ones_like(x_grid) * self.adjC
            self.tau_func = LinearFast(tau_x, [x_grid])

        else:
            p1, p2 = self.ScaleInt
            tau_s = self.adjC
            x_grid = self.xNrmGrid

            tau_x = np.zeros_like(x_grid)

            # Region 1: constant tau_s for x <= p1 (including negatives)
            mask1 = x_grid <= p1
            tau_x[mask1] = tau_s

            # Region 2: quadratic decline for p1 < x <= (p1+p2)/2
            mid = (p1 + p2) / 2
            mask2 = (x_grid > p1) & (x_grid <= mid)
            tau_x[mask2] = tau_s - (2 * tau_s / ((p2 - p1) ** 2)) * (x_grid[mask2] - p1) ** 2

            # Region 3: quadratic decline for mid < x <= p2
            mask3 = (x_grid > mid) & (x_grid <= p2)
            tau_x[mask3] = (2 * tau_s / ((p2 - p1) ** 2)) * (x_grid[mask3] - p2) ** 2

            # Region 4: x > p2 remains 0
            tau_x = np.clip(tau_x, 0, tau_s)

            self.tau_func = LinearFast(tau_x, [x_grid])

        self.add_to_time_inv('tau_func') 
  
    # Scale adjustment costs
    def scale_search_costs(self):

        """
        Compute a 1D scaled search cost function over x ∈ (-∞, x_max] such that they converge to 0.
        The function is:
            - tau_s for x <= p1 (including negative x)
            - quadratic decline between p1 and p2
            - zero after p2
        """
        if self.ScaleBool == False:
            # Constant function (always SearchCost)
            x_grid = self.xNrmGrid
            tau_x = np.ones_like(x_grid) * self.SearchCost
            self.SearchCost_func = LinearFast(tau_x, [x_grid])
            # return

        else:
            p1, p2 = self.ScaleInt
            tau_s = self.SearchCost
            x_grid = self.xNrmGrid

            tau_x = np.zeros_like(x_grid)

            # Region 1: constant tau_s for x <= p1 (including negatives)
            mask1 = x_grid <= p1
            tau_x[mask1] = tau_s

            # Region 2: quadratic decline for p1 < x <= (p1+p2)/2
            mid = (p1 + p2) / 2
            mask2 = (x_grid > p1) & (x_grid <= mid)
            tau_x[mask2] = tau_s - (2 * tau_s / ((p2 - p1) ** 2)) * (x_grid[mask2] - p1) ** 2

            # Region 3: quadratic decline for mid < x <= p2
            mask3 = (x_grid > mid) & (x_grid <= p2)
            tau_x[mask3] = (2 * tau_s / ((p2 - p1) ** 2)) * (x_grid[mask3] - p2) ** 2

            # Region 4: x > p2 remains 0
            tau_x = np.clip(tau_x, 0, tau_s)

            self.SearchCost_func = LinearFast(tau_x, [x_grid])

        self.add_to_time_inv('SearchCost_func') 

    # Solve last period
    def solve_terminal_solution(self):
        '''
        Solves the terminal period. As there is no bequest, the agent consumes all income.
        Market resources and durable stock can be split into durable and non-durable consumption.

        Keepers problem: consume every market resource (m) given durable stock (d).

        Adjuster problem: Calculate optimal share of durables which is c = alpha*xNrmGrid; d = (1-alpha)*xNrmGrid

        Optimal Solution: Use either keeper or adjuster solution depending on the higher utility.

        :return:
        cFunc : function
            The consumption function for this period, defined over durable stock and market
            resources: c = cFunc(n,m).
        dFunc: function
            The end of period durable stock function for this period, defined over beginning of period durable stock and
            market resources: d = dFunc(n,m)
        vFunc : function
            The beginning-of-period value function for this period, defined over durable stock
            and market resources: v = vFunc(n,m).
        cFuncKeep : function
            The consumption function for this periods' keeper problem, defined over durable stock and market
            resources: c = cFunc(n,m).
        dFuncKeep: function
            The durable consumption function for this periods' keeper problem, defined over durable stock and
            market resources: d = dFunc(n,m).
        vFuncKeep : function
            The beginning-of-period value function for this periods' keeper problem, defined over durable stock
            and market resources: v = vFunc(n,m).
        cFuncAdj : function
            The consumption function for this periods' adjuster problem, defined over sum of durable stock and market
            resources: c = cFunc(x).
        dFuncAdj: function
            The durable consumption function for this periods' adjuster problem, defined over sum of durable stock and
            market resources: d = dFunc(x)
        vFuncAdj : function
            The beginning-of-period value function for this periods' adjuster problem, defined over sum of durable stock
            and market resources: v = vFunc(x).
        adjusting : function
            The adjusting function for this period indicates if for a given durable stock and market resources, the agent
            adjusts the durable stock or not: adjusting = adjusting(n,m).
        '''
        timeBOOL = False
        ### i) Create Grids
        # Durable stock grid:
        nNrmGridNow = self.nNrmGrid
        # Market resources grid: add minimum grid point (max borrowing against durable stock).
        if self.LTV[0] == 0.0:
            mNrmMinNow = np.zeros(1)
        else:
            # Collateral includes OpCost: in the terminal period there is no future income or
            # liquid savings, so operating costs on the surviving durable stock must also be
            # serviceable from the collateral. The effective survival value net of OpCost is
            # (1 - dDepr*(1-MaintCost) - OpCost), which replaces the standard (1 - dDepr*(1-MaintCost)).
            mNrmMinNow = - nNrmGridNow * (1 - self.adjC) * (1 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * self.LTV[0]
            if not np.any(mNrmMinNow < 0):   # e.g. dDepr=1 → collateral_term=0
                mNrmMinNow = np.zeros(1)

        mNrmGridNow = np.unique(np.sort(np.append(self.mNrmGrid, mNrmMinNow)))
        ### add max + 1
        mNrmGridNow = np.append(mNrmGridNow, mNrmGridNow[-1] + 1)
        if timeBOOL:
            tic = time.time()

        # a) keeper problem: keep durable stock and consume everything else
        keep_shape = (len(nNrmGridNow),len(mNrmGridNow))
        cFuncKeep_terminal_array = np.zeros(keep_shape)
        dFuncKeep_terminal_array = np.zeros(keep_shape)
        inv_vFuncKeep_terminal_array = np.zeros(keep_shape)
        
        ### Consume everything and borrow against durable stock: Budget constraint = m + (1 - dDepr)* n = c
        ### Solution is 0 for points below mNrmGridNow
        for i_n in range(len(nNrmGridNow)):
            n_aux = np.ones(len(mNrmGridNow)) * nNrmGridNow[i_n]
            # Non-durable consumption
            if len(mNrmMinNow) > 1:
                idx_start = len(mNrmMinNow) - i_n
            else:
                idx_start = 0
            dFuncKeep_terminal_array[i_n, idx_start:] = n_aux[idx_start:]
            adjC_scaled = self.tau_func(n_aux[idx_start:] + mNrmGridNow[idx_start:])
            # OpCost included: see mNrmMinNow note above.
            cFuncKeep_terminal_array[i_n, idx_start:] = mNrmGridNow[idx_start:] + (1 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * (1 - adjC_scaled) * self.LTV[0] * n_aux[idx_start:]
            inv_vFuncKeep_terminal_array[i_n, idx_start:] = self.CRRAutility_inv(cFuncKeep_terminal_array[i_n, idx_start:], dFuncKeep_terminal_array[i_n, idx_start:])
            
            
        vPDFuncKeep_terminal_array = self.CRRAutilityPD(cFuncKeep_terminal_array, dFuncKeep_terminal_array)
        vPFuncKeep_terminal_array = self.CRRAutilityP(cFuncKeep_terminal_array, dFuncKeep_terminal_array)

        aFuncKeep_terminal_array = mNrmGridNow - cFuncKeep_terminal_array
        
        # iii) Make Functions
        # a) Policy Functions
        self.cFuncKeep_terminal = LinearFast(cFuncKeep_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.dFuncKeep_terminal = LinearFast(dFuncKeep_terminal_array, [nNrmGridNow, mNrmGridNow])
        
        # b) Value Functions
        self.inv_vFuncKeep_terminal = LinearFast(inv_vFuncKeep_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.vFuncKeep_terminal = ValueFuncCRRA(self.inv_vFuncKeep_terminal, self.CRRA)
        
        # c) Marginal Value Functions
        self.vPFuncKeep_terminal = MargValueFuncCRRA_dur(self.cFuncKeep_terminal, self.dFuncKeep_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.inv_vPFuncKeep_terminal = inv_MargValueFuncCRRA_dur(self.cFuncKeep_terminal, self.dFuncKeep_terminal, self.alpha, self.d_ubar, self.CRRA)

        if timeBOOL:
            toc = time.time()
            print("Keeper Problem solved in ", toc - tic, " seconds.")

        if timeBOOL:
            tic = time.time()
        # b) adjuster problem: (Note this is an analytical solution)
        if self.alpha == 1.0:
            ### Non-Durable case
            cFuncAdj_terminal_array = self.xNrmGrid
            dFuncAdj_terminal_array = np.zeros_like(self.xNrmGrid)
            inv_vFuncAdj_terminal_array = self.CRRAutility_inv(cFuncAdj_terminal_array, dFuncAdj_terminal_array)
        else:
            # Calculate optimal ratio of c/d.
            gam_terminal = (self.alpha)/(1 - self.alpha) * (1 - (1 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * (1 - self.tau_func(self.xNrmGrid)) * self.LTV[0])
            term_d_terminal = 1/(1 - self.alpha) * (1 - (1 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * (1 - self.tau_func(self.xNrmGrid)) * self.LTV[0])

            dFuncAdj_terminal_array = self.xNrmGrid/term_d_terminal
            cFuncAdj_terminal_array = dFuncAdj_terminal_array * gam_terminal
            inv_vFuncAdj_terminal_array = self.CRRAutility_inv(cFuncAdj_terminal_array, dFuncAdj_terminal_array)


        ###
        # Create Functions
        self.cFuncAdj_terminal = LinearFast(cFuncAdj_terminal_array, [self.xNrmGrid])
        self.dFuncAdj_terminal = LinearFast(dFuncAdj_terminal_array, [self.xNrmGrid])
        self.inv_vFuncAdj_terminal = LinearFast(inv_vFuncAdj_terminal_array, [self.xNrmGrid])
        self.vFuncAdj_terminal = ValueFuncCRRA(self.inv_vFuncAdj_terminal, self.CRRA)
        self.vPFuncAdj_terminal = MargValueFuncCRRA_dur(self.cFuncAdj_terminal, self.dFuncAdj_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.inv_vPFuncAdj_terminal = inv_MargValueFuncCRRA_dur(self.cFuncAdj_terminal, self.dFuncAdj_terminal, self.alpha, self.d_ubar, self.CRRA)
        # vPDFuncAdj = (1-tau) * u_c(c*, d*): the (1-tau) factor arises because the adjuster's
        # value increases by (1-tau) for each extra unit of n (it can be liquidated net of cost).
        tau_1D = self.tau_func(self.xNrmGrid)
        vPDFuncAdj_terminal_array = (1 - tau_1D) * self.CRRAutilityP(cFuncAdj_terminal_array, dFuncAdj_terminal_array)
        inv_vPDFuncAdj_terminal_array = utilityP_inv(vPDFuncAdj_terminal_array, self.CRRA)
        self.vPDFuncAdj_terminal = LinearFast(vPDFuncAdj_terminal_array, [self.xNrmGrid])
        self.inv_vPDFuncAdj_terminal = LinearFast(inv_vPDFuncAdj_terminal_array, [self.xNrmGrid])

        if timeBOOL:
            toc = time.time()
            print("Adjuster Problem solved in ", toc - tic, " seconds.")

        if timeBOOL:
            tic = time.time()

        ### Create Frictionless Functions (tau = 0, actual LTV)
        # Terminal-period frictionless problem: this is the tau->0 limit of the actual
        # terminal adjuster solution.  The adjuster uses:
        #   gamma(x) = alpha/(1-alpha) * [1 - (1 - delta*(1-chi) - OpCost)*(1-tau(x))*LTV]
        # As x -> inf, tau(x) -> 0, so the limit is:
        #   p_d = 1 - (1 - delta*(1-chi) - OpCost) * LTV
        # With LTV=1 this simplifies to delta*(1-chi)+OpCost, but for general LTV the
        # actual LTV must be used so that the frictionless function is the correct limit.
        # Note: gamma here does NOT involve R because there is no next period.
        # The non-terminal limit function (used as an extrapolation device) has
        #   gamma = alpha/(1-alpha) * (1 + (delta+OpCost-1)/R), which differs.
        if self.alpha == 1:
            cFuncAdj_Frictionless_terminal_array = self.xNrmGrid
            dFuncAdj_Frictionless_terminal_array = np.zeros_like(self.xNrmGrid)
        else:
            # effective user cost = 1 - residual_value * LTV  (tau = 0 limit)
            p_d_Frictionless = 1.0 - (1.0 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * self.LTV[0]
            gam_terminal_Frictionless    = self.alpha / (1 - self.alpha) * p_d_Frictionless
            term_d_terminal_Frictionless = p_d_Frictionless / (1 - self.alpha)   # = p_d/(1-alpha)

            dFuncAdj_Frictionless_terminal_array = self.xNrmGrid / term_d_terminal_Frictionless
            cFuncAdj_Frictionless_terminal_array = dFuncAdj_Frictionless_terminal_array * gam_terminal_Frictionless

        qFuncAdj_Frictionless_terminal_array   = cFuncAdj_Frictionless_terminal_array + dFuncAdj_Frictionless_terminal_array
        inv_vFuncAdj_Frictionless_array = self.CRRAutility_inv(cFuncAdj_Frictionless_terminal_array, dFuncAdj_Frictionless_terminal_array)

        ### Create Functions
        self.qFuncAdj_Frictionless_terminal   = LinearFast(qFuncAdj_Frictionless_terminal_array, [self.xNrmGrid])
        self.cFuncAdj_Frictionless_terminal   = LinearFast(cFuncAdj_Frictionless_terminal_array, [self.xNrmGrid])
        self.dFuncAdj_Frictionless_terminal   = LinearFast(dFuncAdj_Frictionless_terminal_array, [self.xNrmGrid])
        self.inv_vFuncAdj_Frictionless_terminal = LinearFast(inv_vFuncAdj_Frictionless_array, [self.xNrmGrid])
        # MargValueFuncCRRA expects the Cobb-Douglas composite c^alpha*d^(1-alpha)
        # (same input as ValueFuncCRRA) so that value and marginal-value are consistent.
        self.vPFuncAdj_Frictionless_terminal  = MargValueFuncCRRA(self.inv_vFuncAdj_Frictionless_terminal, self.CRRA)
        self.vFuncAdj_Frictionless_terminal   = ValueFuncCRRA(self.inv_vFuncAdj_Frictionless_terminal, self.CRRA)

        if timeBOOL:
            toc = time.time()
            print("Frictionless Problem solved in ", toc - tic, " seconds.")

        if timeBOOL:
            tic = time.time()
        # c) Create Consumption Function: Using analytical solution
        ### Create adjuster function in nxm
        # Create empty container
        shape = (len(nNrmGridNow), len(mNrmGridNow))
        cFuncAdj_SOL_array = np.zeros(shape)
        dFuncAdj_SOL_array = np.zeros(shape)
        aFuncAdj_SOL_array = np.zeros(shape)
        inv_vFuncAdj_SOL_array = np.zeros(shape)
        vPDFuncAdj_SOL_array = np.zeros(shape)
        adjusting_array = np.ones(shape)
        
        for i_n in range(len(nNrmGridNow)):
            ### Start_id
            if len(mNrmMinNow) > 1:
                idx_start = len(mNrmMinNow) - i_n 
            else:
                idx_start = 0
            n_aux = np.ones(len(mNrmGridNow)) * nNrmGridNow[i_n]
            
            # tau is determined by beginning-of-period state (n, m), before the adjustment decision.
            adjC_scaled = self.tau_func(n_aux + mNrmGridNow) # adjC_scaled = self.tau_func((1 - self.dDepr) * n_aux + mNrmGridNow)
            x_aux = np.clip(mNrmGridNow + (1 - adjC_scaled) * n_aux, 0, None)

            if self.alpha == 1.0:
                ### Non-Durable case
                dFuncAdj_SOL_array[i_n, idx_start:] = np.zeros_like(x_aux[idx_start:])
                cFuncAdj_SOL_array[i_n, idx_start:] = x_aux[idx_start:]
            else:
                gam_terminal = (self.alpha)/(1 - self.alpha) * (1 - (1 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * (1 - adjC_scaled) * self.LTV[0])
                term_d_terminal = 1/(1 - self.alpha) * (1 - (1 - self.dDepr * (1 - self.MaintCost) - self.OpCost) * (1 - adjC_scaled) * self.LTV[0])
                
                dFuncAdj_SOL_array[i_n, idx_start:] =  x_aux[idx_start:] / term_d_terminal[idx_start:]
                cFuncAdj_SOL_array[i_n, idx_start:] =  dFuncAdj_SOL_array[i_n, idx_start:] * gam_terminal[idx_start:]
            


            aFuncAdj_SOL_array[i_n] = x_aux - cFuncAdj_SOL_array[i_n] - dFuncAdj_SOL_array[i_n]
            inv_vFuncAdj_SOL_array[i_n, idx_start:] = self.CRRAutility_inv(cFuncAdj_SOL_array[i_n, idx_start:], dFuncAdj_SOL_array[i_n, idx_start:])  # Adjust
            vPDFuncAdj_SOL_array[i_n] = (1 - adjC_scaled) * self.CRRAutilityP(cFuncAdj_SOL_array[i_n], dFuncAdj_SOL_array[i_n])  # Adjust


        vPFuncAdj_SOL_array = self.CRRAutilityP(cFuncAdj_SOL_array, dFuncAdj_SOL_array)
        inv_vPFuncAdj_SOL_array = utilityP_inv(vPFuncAdj_SOL_array, self.CRRA)
        inv_vPDFuncAdj_SOL_array = utilityP_inv(vPDFuncAdj_SOL_array, self.CRRA)

        ### Is this a violation of the budget constraint?
        # timeBOOL if budget constraint is violated
        if timeBOOL:
            toc = time.time()
            print("Solutions in array ", toc - tic, " seconds.")

        if timeBOOL:
            tic = time.time()
        ### When adjC and SearchCost are zero: Frictionless and take adjuster solution
        if self.adjC == 0 and self.SearchCost == 0:
            ### Frictionless Case: Use adjuster solution
            cFunc_terminal_array = cFuncAdj_SOL_array
            dFunc_terminal_array = dFuncAdj_SOL_array
            aFunc_terminal_array = aFuncAdj_SOL_array
            inv_vFunc_terminal_array = inv_vFuncAdj_SOL_array
            vPFunc_terminal_array = vPFuncAdj_SOL_array
            vPDFunc_terminal_array = vPDFuncAdj_SOL_array
            adjusting_terminal_array = np.ones(shape)
            lSuS_array = np.zeros((len(nNrmGridNow),2))

        else:
            if self.SearchCost == 0:
                print("No Search Cost")
                #### No SS-Structure (allow for multiple thresholds)
                cFunc_terminal_array, dFunc_terminal_array, aFunc_terminal_array, inv_vFunc_terminal_array, vPFunc_terminal_array, vPDFunc_terminal_array, adjusting_terminal_array = durable_solution_function_FUES_fast_search_noSS(
                    adjusting_array,
                    cFuncAdj_SOL_array, dFuncAdj_SOL_array, aFuncAdj_SOL_array,
                    inv_vFuncAdj_SOL_array, vPFuncAdj_SOL_array, vPDFuncAdj_SOL_array,
                    cFuncKeep_terminal_array, dFuncKeep_terminal_array, aFuncKeep_terminal_array,
                    inv_vFuncKeep_terminal_array, vPFuncKeep_terminal_array, vPDFuncKeep_terminal_array, 1e-4) #tol)
                        
                vFunc_terminal_array = utility_inv(inv_vFunc_terminal_array, self.CRRA) # Not necessary, but for consistency.
            
            else: # Directed Search
                print("Search Cost")
                # inv_v stores u_inner = CRRAutility_inv(c,d); recover true value via utility(u_inner).
                # utility_inv(u_inner) would only equal v for CRRA = 2.
                vFuncAdj_SOL_array = utility(inv_vFuncAdj_SOL_array, self.CRRA)
                vFuncKeep_SOL_array = utility(inv_vFuncKeep_terminal_array, self.CRRA)
                
                ### Replace -np.inf with -1e12
                vFuncAdj_SOL_aux_array = np.where(vFuncAdj_SOL_array == -np.inf, -1e12, vFuncAdj_SOL_array)
                vFuncKeep_SOL_aux_array = np.where(vFuncKeep_SOL_array == -np.inf, -1e12, vFuncKeep_SOL_array)

                vPFuncAdj_SOL_aux_array = np.where(vPFuncAdj_SOL_array == np.inf, 1e12, vPFuncAdj_SOL_array)
                vPDFuncAdj_SOL_aux_array = np.where(vPDFuncAdj_SOL_array == np.inf, 1e12, vPDFuncAdj_SOL_array)
                vPFuncKeep_terminal_aux_array = np.where(vPFuncKeep_terminal_array == np.inf, 1e12, vPFuncKeep_terminal_array)
                vPDFuncKeep_terminal_aux_array = np.where(vPDFuncKeep_terminal_array == np.inf, 1e12, vPDFuncKeep_terminal_array)

                ### Optimal effort for Directed Search
                e_grid = np.linspace(0,1,101)

                # Step 1: Precompute search cost matrix
                search_cost_matrix = np.zeros(shape)

                for i_n in range(len(nNrmGridNow)):
                    n_aux = np.ones(len(mNrmGridNow)) * nNrmGridNow[i_n]
                    search_cost_matrix[i_n, :] = self.SearchCost_func(n_aux + mNrmGridNow)

                # Step 2: Compute costs matrix with Numba
                Costs_matrix = compute_cost_matrix(search_cost_matrix, e_grid)

                adjusting_terminal_array, DirectedSearchCost_terminal_array, vFunc_terminal_array, cFunc_terminal_array, dFunc_terminal_array, aFunc_terminal_array, vPFunc_terminal_array, vPDFunc_terminal_array = directed_search_function_scaled_matrix(nNrmGridNow, mNrmGridNow, mNrmMinNow, e_grid, Costs_matrix, vFuncAdj_SOL_aux_array, vFuncKeep_SOL_aux_array,
                cFuncAdj_SOL_array, cFuncKeep_terminal_array, dFuncAdj_SOL_array, dFuncKeep_terminal_array, aFuncAdj_SOL_array, aFuncKeep_terminal_array, vPFuncAdj_SOL_aux_array, vPFuncKeep_terminal_aux_array, vPDFuncAdj_SOL_aux_array, vPDFuncKeep_terminal_aux_array)

                ### replace -np.inf if adjusting_terminal_array == 1 and vPFunc_terminal_array == -np.inf or adjusting_terminal_array == 0 and vPFunc_keep_array == -np.inf or adjusting_terminal_array between 0 and 1 and one is -np.inf


                ### Replace -1e12 with -np.inf
                cond_inf = (
                    ((adjusting_terminal_array == 1.0) & (vFuncAdj_SOL_array == -np.inf)) |
                    ((adjusting_terminal_array == 0.0) & (vFuncKeep_SOL_array == -np.inf)) |
                    (((adjusting_terminal_array > 0.0) & (adjusting_terminal_array < 1.0)) &
                    ((vFuncAdj_SOL_array == -np.inf) | (vFuncKeep_SOL_array == -np.inf)))
                )

                vFunc_terminal_array[cond_inf] = -np.inf
                # vFunc_terminal_array = np.where(vFunc_terminal_array == -1e12, -np.inf, vFunc_terminal_array)
                inv_vFunc_terminal_array = utility_inv(vFunc_terminal_array, self.CRRA)

                # Conditions for restoring np.inf
                cond_inf = (
                    ((adjusting_terminal_array == 1.0) & (vPFuncAdj_SOL_array == np.inf)) |
                    ((adjusting_terminal_array == 0.0) & (vPFuncKeep_terminal_array == np.inf)) |
                    (((adjusting_terminal_array > 0.0) & (adjusting_terminal_array < 1.0)) &
                    ((vPFuncAdj_SOL_array == np.inf) | (vPFuncKeep_terminal_array == np.inf)))
                )

                # Apply condition
                vPFunc_terminal_array[cond_inf] = np.inf

                # Conditions for restoring np.inf
                cond_inf = (
                    ((adjusting_terminal_array == 1.0) & (vPDFuncAdj_SOL_array == np.inf)) |
                    ((adjusting_terminal_array == 0.0) & (vPDFuncKeep_terminal_array == np.inf)) |
                    (((adjusting_terminal_array > 0.0) & (adjusting_terminal_array < 1.0)) &
                    ((vPDFuncAdj_SOL_array == np.inf) | (vPDFuncKeep_terminal_array == np.inf)))
                )

                # Apply condition
                vPDFunc_terminal_array[cond_inf] = np.inf
                
                # Directly take from original arrays when adjusting=1 or adjusting=0
                mask_adj1 = (adjusting_terminal_array == 1.0)
                mask_adj0 = (adjusting_terminal_array == 0.0)
                vPFunc_terminal_array[mask_adj1] = vPFuncAdj_SOL_array[mask_adj1]
                vPFunc_terminal_array[mask_adj0] = vPFuncKeep_terminal_array[mask_adj0]

                vPDFunc_terminal_array[mask_adj1] = vPDFuncAdj_SOL_array[mask_adj1]
                vPDFunc_terminal_array[mask_adj0] = vPDFuncKeep_terminal_array[mask_adj0]

        if timeBOOL:
            toc = time.time()
            print("Combining solutions in ", toc - tic, " seconds.")
        ### Create Marginal Value Arrays from inverses
        inv_vPFunc_terminal_array = utilityP_inv(vPFunc_terminal_array, self.CRRA)
        inv_vPDFunc_terminal_array = utilityP_inv(vPDFunc_terminal_array, self.CRRA)
       
        # Interpolation
        # a) Policy Functions
        self.cFunc_terminal = LinearFast(cFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.dFunc_terminal = LinearFast(dFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.aFunc_terminal = LinearFast(aFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.adjusting_terminal = LinearFast(adjusting_terminal_array, [nNrmGridNow, mNrmGridNow])

        # b) Value Functions
        self.inv_vFunc_terminal = LinearFast(inv_vFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.vFunc_terminal = ValueFuncCRRA(self.inv_vFunc_terminal, self.CRRA)

        # c) Marginal Value Functions
        self.vPDFunc_terminal = LinearFast(vPDFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.inv_vPDFunc_terminal = LinearFast(inv_vPDFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.vPFunc_terminal = LinearFast(vPFunc_terminal_array, [nNrmGridNow, mNrmGridNow])
        self.inv_vPFunc_terminal = LinearFast(inv_vPFunc_terminal_array, [nNrmGridNow, mNrmGridNow])

        self.xNrmMin = np.min(self.xNrmGrid)
        self.mNrmMin = mNrmMinNow
        self.hNrm = 0.0
        self.MPCmin = self.alpha#1.0
        self.MPCmax = self.alpha#1.0


        lSuS_array, lSuS_idx_array = calc_SS_fast_terminal(nNrmGridNow, mNrmGridNow, mNrmMinNow, inv_vFuncAdj_SOL_array, inv_vFuncKeep_terminal_array, 1e-14)
        self.lSuS_array = lSuS_array
        self.adjusting_array = adjusting_terminal_array

        p1 = self.ScaleInt[0] 
        p2 = self.ScaleInt[1] 
        nNrmGridTotal_eval = nNrmGridNow[nNrmGridNow <= p2 * 2]
        mNrmGridTotal_eval = mNrmGridNow[mNrmGridNow <= p2 * 2]

        vPFunc_terminal_array = vPFunc_terminal_array[:len(nNrmGridTotal_eval), :len(mNrmGridTotal_eval)]
        vPDFunc_eval_array = vPDFunc_terminal_array[:len(nNrmGridTotal_eval), :len(mNrmGridTotal_eval)]
        self.vPFunc_eval = LinearFast(vPFunc_terminal_array, [nNrmGridTotal_eval, mNrmGridTotal_eval])
        self.vPDFunc_eval =  LinearFast(vPDFunc_eval_array, [nNrmGridTotal_eval, mNrmGridTotal_eval])


    def update_solution_terminal(self):
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep = self.cFuncKeep_terminal,
            cFuncAdj = self.cFuncAdj_terminal,
            dFuncKeep = self.dFuncKeep_terminal,
            dFuncAdj = self.dFuncAdj_terminal,
            inv_vFuncKeep = self.inv_vFuncKeep_terminal,
            inv_vFuncAdj = self.inv_vFuncAdj_terminal,
            cFunc = self.cFunc_terminal,
            dFunc = self.dFunc_terminal,
            aFunc = self.aFunc_terminal,
            inv_vPFunc = self.inv_vPFunc_terminal,
            inv_vPDFunc = self.inv_vPDFunc_terminal,
            inv_vFunc = self.inv_vFunc_terminal,
            adjusting = self.adjusting_terminal,
            xNrmMin = self.xNrmMin,
            mNrmMin = self.mNrmMin,
            hNrm = self.hNrm,
            MPCmin = self.MPCmin,
            MPCmax = self.MPCmax,
            cFuncCnst = self.cFunc_terminal,
            dFuncCnst = self.dFunc_terminal,
            ### For evaluation
            vPFunc_eval = self.vPFunc_eval,
            vPDFunc_eval = self.vPDFunc_eval,
            ### Frictionless Solution
            qFunc_Frictionless = self.qFuncAdj_Frictionless_terminal,
            vFunc_Frictionless = self.vFuncAdj_Frictionless_terminal,
            vPFunc_Frictionless = self.vPFuncAdj_Frictionless_terminal,
    )
        
    def calc_limit_inf(self):
        '''
        Calculates the limit policy functions for the infinite horizon case.
        This corresponds to the perfect foresight (no uncertainty) and no frictions (adjustment cost is 0).       
        '''
        alpha = self.alpha
        dDepr = self.dDepr
        Rfree = self.Rfree[0]

        Thorn_R = ((self.Rfree[0] * self.DiscFac[0])**(1/self.CRRA))/(self.Rfree[0])
        # Thorn_R = ((self.Rfree[0] * self.DiscFac[0] * self.LivPrb[0])**(1/self.CRRA))/(self.Rfree[0])
        self.MPCminNow = 1 - Thorn_R
        self.hNrmNow = 1/(1 - (self.PermGroFac[0]/self.Rfree[0]))

        ### Get gamma and Lambda_inverse
        if alpha <1:
            self.gam = (alpha/(1 - alpha)) * (Rfree - 1 + dDepr)/Rfree ## Includes Durables
            Lambda_inv = (1 + self.gam) / (1 + self.gam*(Rfree - 1 + dDepr)/Rfree)
        else:
            self.gam = 1
            Lambda_inv = 1

        MPXminNow = np.min([self.MPCminNow * Lambda_inv, 1.0])

        ### Limit Function:
        LimitexFunc1D = lambda x: MPXminNow * (self.hNrmNow + x)
        LimitcFunc1D = lambda x: MPXminNow * (self.hNrmNow + x) * (self.gam/(1 + self.gam))
        LimitdFunc1D = lambda x: MPXminNow * (self.hNrmNow + x) * (1/(1 + self.gam))

        LimitexFunc2D = lambda n, m: MPXminNow * (self.hNrmNow + n + m)
        LimitcFunc2D = lambda n, m: MPXminNow * (self.hNrmNow + n + m) * (self.gam/(1 + self.gam))
        LimitdFunc2D = lambda n, m: MPXminNow * (self.hNrmNow + n + m) * (1/(1 + self.gam))

        # Limit Gradient (Frictionless case)
        LimitexGrad1D = lambda x: [MPXminNow * np.ones_like(x)]
        LimitcGrad1D = lambda x: [MPXminNow * np.ones_like(x) * (self.gam/(1 + self.gam))]
        LimitdGrad1D = lambda x: [MPXminNow * np.ones_like(x) * (1/(1 + self.gam))]

        LimitexGrad2D = lambda n, m: LimitexGrad1D(n + m)
        LimitcGrad2D = lambda n, m: LimitcGrad1D(n + m)
        LimitdGrad2D = lambda n, m: LimitdGrad1D(n + m)

        ### Limit Value Function:
        Term = 1/((1 - self.CRRA)*(1 - Thorn_R))
        # LimitvFunc1D = lambda x: Term * LimitexFunc1D(x)**(1 - self.CRRA)
        # LimitvFunc2D = lambda n, m: Term * LimitexFunc2D(n, m)**(1 - self.CRRA)

        # Limitinv_vFunc1D = lambda x: utility_inv(LimitvFunc1D(x), self.CRRA)
        # Limitinv_vFunc2D = lambda n, m: utility_inv(LimitvFunc2D(n, m), self.CRRA)


        gam_inv = self.gam**(-1)
        term_inv = ((gam_inv ** self.alpha) / (1 + gam_inv))**(1 - self.CRRA)
        LimitvFunc1D = lambda x: Term * term_inv * LimitexFunc1D(x)**(1 - self.CRRA)
        LimitvFunc2D = lambda n, m: Term * term_inv * LimitexFunc2D(n, m)**(1 - self.CRRA)

        Limitinv_vFunc1D = lambda x: utility_inv(LimitvFunc1D(x), self.CRRA)
        Limitinv_vFunc2D = lambda n, m: utility_inv(LimitvFunc2D(n, m), self.CRRA)

        return {
            "LimitexFunc1D": LimitexFunc1D,
            "LimitcFunc1D": LimitcFunc1D,
            "LimitdFunc1D": LimitdFunc1D,
            "LimitvFunc1D": LimitvFunc1D,
            "Limitinv_vFunc1D": Limitinv_vFunc1D,
            "LimitexGrad1D": LimitexGrad1D,
            "LimitcGrad1D": LimitcGrad1D,
            "LimitdGrad1D": LimitdGrad1D,
            "LimitexFunc2D": LimitexFunc2D,
            "LimitcFunc2D": LimitcFunc2D,
            "LimitdFunc2D": LimitdFunc2D,
            "LimitvFunc2D": LimitvFunc2D,
            "Limitinv_vFunc2D": Limitinv_vFunc2D,
            "LimitexGrad2D": LimitexGrad2D,
            "LimitdGrad2D": LimitdGrad2D,
            "LimitcGrad2D": LimitcGrad2D,
        }

        # Solve last period
    def solve_terminal_solution_PF(self):
     
        '''     
        Solves the terminal period assuming frictionless (adjustment cost is 0) perfect foresight (no uncertainty) solution.
        
        Market resources and durable stock can be split into durable and non-durable consumption.

        Keepers problem: consume every market resource (m) given durable stock (d).

        Adjuster problem: Calculate optimal share of durables which is c = alpha*xNrmGrid; d = (1-alpha)*xNrmGrid

        Optimal Solution: Use either keeper or adjuster solution depending on the higher utility.

        :return:
        cFunc : function
            The consumption function for this period, defined over durable stock and market
            resources: c = cFunc(n,m).
        dFunc: function
            The end of period durable stock function for this period, defined over beginning of period durable stock and
            market resources: d = dFunc(n,m)
        vFunc : function
            The beginning-of-period value function for this period, defined over durable stock
            and market resources: v = vFunc(n,m).
        cFuncKeep : function
            The consumption function for this periods' keeper problem, defined over durable stock and market
            resources: c = cFunc(n,m).
        dFuncKeep: function
            The durable consumption function for this periods' keeper problem, defined over durable stock and
            market resources: d = dFunc(n,m).
        vFuncKeep : function
            The beginning-of-period value function for this periods' keeper problem, defined over durable stock
            and market resources: v = vFunc(n,m).
        cFuncAdj : function
            The consumption function for this periods' adjuster problem, defined over sum of durable stock and market
            resources: c = cFunc(x).
        dFuncAdj: function
            The durable consumption function for this periods' adjuster problem, defined over sum of durable stock and
            market resources: d = dFunc(x)
        vFuncAdj : function
            The beginning-of-period value function for this periods' adjuster problem, defined over sum of durable stock
            and market resources: v = vFunc(x).
        adjusting : function
            The adjusting function for this period indicates if for a given durable stock and market resources, the agent
            adjusts the durable stock or not: adjusting = adjusting(n,m).
        '''
        LimitNow = self.calc_limit_inf()

        """
        Without Lower envelope.
        """
        # Total Solution Functions are 2D
        self.cFunc_terminal = LimitNow["LimitcFunc2D"]
        self.dFunc_terminal = LimitNow["LimitdFunc2D"]
        self.vFunc_terminal = LimitNow["LimitvFunc2D"]
        self.inv_vFunc_terminal = LimitNow["Limitinv_vFunc2D"]
        self.vPFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.vPDFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.inv_vPDFunc_terminal = inv_MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.inv_vPFunc_terminal = inv_MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)

        ### Add zero-grid
        self.nNrmGridNow = self.nNrmGrid
        zero_grid_m = -(self.nNrmGrid + self.hNrmNow)
        self.mNrmGridNow = np.unique(np.sort(np.append(self.mNrmGrid, zero_grid_m)))

        solution_shape = (len(self.nNrmGridNow), len(self.mNrmGridNow))
        vPDFunc_terminal_array = np.zeros(solution_shape)
        vPFunc_terminal_array = np.zeros(solution_shape)

        for i_n in range(len(self.nNrmGridNow)):
            n_aux = np.ones(len(self.mNrmGridNow)) * self.nNrmGridNow[i_n]
            c_aux = self.cFunc_terminal(n_aux, self.mNrmGridNow)
            d_aux = self.dFunc_terminal(n_aux, self.mNrmGridNow)
            vPDFunc_terminal_array[i_n] = self.CRRAutilityP(c_aux, d_aux)
            vPFunc_terminal_array[i_n] = self.CRRAutilityP(c_aux, d_aux)
        
        # Interpolation
        ### Replace all nan with inf
        vPDFunc_terminal_array[np.isnan(vPDFunc_terminal_array)] = np.inf
        inv_vPDFunc_terminal_array = utilityP_inv(vPDFunc_terminal_array, self.CRRA)
        vPFunc_terminal_array[np.isnan(vPFunc_terminal_array)] = np.inf
        inv_vPFunc_terminal_array = utilityP_inv(vPFunc_terminal_array, self.CRRA)

        self.inv_vPDFunc_terminal = LinearFast(inv_vPDFunc_terminal_array, [self.nNrmGridNow, self.mNrmGridNow])
        self.inv_vPFunc_terminal = LinearFast(inv_vPFunc_terminal_array, [self.nNrmGridNow, self.mNrmGridNow])


        ### Adjusting Solutuion Functions are 1D
        self.qFuncAdj_terminal = LimitNow["LimitexFunc1D"]
        self.cFuncAdj_terminal = LimitNow["LimitcFunc1D"]
        self.dFuncAdj_terminal = LimitNow["LimitdFunc1D"]
        self.vFuncAdj_terminal = LimitNow["LimitvFunc1D"]
        self.inv_vFuncAdj_terminal = LimitNow["Limitinv_vFunc1D"]
        self.vPFuncAdj_terminal = MargValueFuncCRRA(self.qFuncAdj_terminal, self.CRRA)
        self.vPDFuncAdj_terminal = MargValueFuncCRRA(self.qFuncAdj_terminal, self.CRRA)
        
        shape = (len(self.nNrmGridNow), len(self.mNrmGrid))
        self.adjusting_terminal = LinearFast(np.ones(shape), [self.nNrmGridNow, self.mNrmGrid])

        ### Other functions I need
        # a) keeper problem: keep durable stock and consume everything else: For constraint part
        self.cFuncKeep_terminal = LinearFast(np.array([[0.0,1.0],[0.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])
        self.dFuncKeep_terminal = LinearFast(np.array([[0.0,0.0],[1.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])

        self.lSuS_array = np.zeros((len(self.nNrmGrid), 2))
        self.adjusting_array = np.ones(shape)
        self.xNrmMin = self.xNrmGrid[0]

        ### Frictionless Solution with Constraint + Unconstraint part
        self.cFunc_Frictionless_terminal = LimitNow["LimitcFunc1D"]
        self.dFunc_Frictionless_terminal = LimitNow["LimitdFunc1D"] 
        self.qFunc_Frictionless_terminal = LimitNow["LimitexFunc1D"]


        # cFunc_Frictionless_terminal_array = (1/ (1 + self.gam)) * (self.xNrmGrid) 
        # dFunc_FrictionlessCnst_terminal_array = (self.gam/(1 + self.gam)) *  (self.xNrmGrid)
        # cFunc_FrictionlessCnst_terminal = LinearFast(cFunc_FrictionlessCnst_terminal_array, [self.xNrmGrid])
        # dFunc_FrictionlessCnst_terminal = LinearFast(dFunc_FrictionlessCnst_terminal_array, [self.xNrmGrid])
        # qFunc_FrictionlessCnst_terminal = LinearFast(self.xNrmGrid, [self.xNrmGrid])

        # # Take Lower Envelope
        # self.cFunc_Frictionless = LowerEnvelope(cFunc_FrictionlessUnc_terminal, cFunc_FrictionlessCnst_terminal)
        # self.dFunc_Frictionless = LowerEnvelope(dFunc_FrictionlessUnc_terminal, dFunc_FrictionlessCnst_terminal)
        # self.qFunc_Frictionless = LowerEnvelope(qFunc_FrictionlessUnc_terminal, qFunc_FrictionlessCnst_terminal)

        # Value Functions
        self.inv_vFunc_Frictionless_terminal_array = utility_inv(self.CRRAutility(self.cFunc_Frictionless_terminal(self.xNrmGrid), self.dFunc_Frictionless_terminal(self.xNrmGrid)) + self.DiscFac[0] * self.LivPrb[0] * LimitNow["LimitvFunc1D"](self.dFunc_Frictionless_terminal(self.xNrmGrid) * (1 - self.dDepr) + self.Rfree[0] * (self.xNrmGrid - self.dFunc_Frictionless_terminal(self.xNrmGrid) - self.cFunc_Frictionless_terminal(self.xNrmGrid))+1), self.CRRA)
        self.inv_vFunc_Frictionless_terminal = LinearFast(self.inv_vFunc_Frictionless_terminal_array, [self.xNrmGrid])
        self.vFunc_Frictionless_terminal = ValueFuncCRRA(self.inv_vFunc_Frictionless_terminal, self.CRRA)
        self.vPFunc_Frictionless_terminal = MargValueFuncCRRA(self.qFunc_Frictionless_terminal, self.CRRA)
        # self.vPDFunc_Frictionless = MargValueFuncCRRA_dur(self.cFunc_Frictionless, self.dFunc_Frictionless, self.alpha, self.d_ubar, self.CRRA)

        self.xNrmMin = -self.hNrmNow #This is not the mimimum of the grid, but the minimum you can borrow your assets without the durable stock.
        self.mNrmMin = zero_grid_m
        self.hNrm = self.hNrmNow
        self.MPCmin = LimitNow["LimitcGrad1D"](1.0)
        self.MPCmax = LimitNow["LimitcGrad1D"](1.0)
        ###
        lSuS_array = None
        self.lSuS_array = lSuS_array
        self.adjusting_array = np.ones(solution_shape)

    def update_solution_terminal_PF(self):
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep = self.cFunc_terminal,
            cFuncAdj = self.cFuncAdj_terminal,
            dFuncKeep = self.dFunc_terminal,
            dFuncAdj = self.dFuncAdj_terminal,
            inv_vFuncKeep = self.inv_vFunc_terminal,
            inv_vFuncAdj = self.inv_vFuncAdj_terminal,
            cFunc = self.cFunc_terminal,
            dFunc = self.dFunc_terminal,
            aFunc = NullFunc,
            inv_vPFunc = self.inv_vPFunc_terminal,
            inv_vPDFunc = self.inv_vPDFunc_terminal,
            inv_vFunc = self.inv_vFunc_terminal,
            adjusting = self.adjusting_terminal,
            xNrmMin = self.xNrmMin,
            mNrmMin = self.mNrmMin,
            hNrm = self.hNrmNow,
            MPCmin = self.MPCminNow,
            MPCmax = self.MPCminNow, #self.MPCmax,
            # # Constraint functions
            cFuncCnst = self.cFuncKeep_terminal,
            dFuncCnst = self.dFuncKeep_terminal,
            vPFunc_eval = self.vPFunc_terminal,
            vPDFunc_eval = self.vPDFunc_terminal,
            ### Frictionless Solution
            vFunc_Frictionless = self.vFunc_Frictionless_terminal,
            vPFunc_Frictionless = self.vPFunc_Frictionless_terminal,
    )

    def solve_terminal_solution_Keep(self):
        """
        Agent cannot adjust, but receives the present discounted value of durable stock as a loan.
        Therefore, non-durable consumption is beginning-of-period market resources (m) plus the 
        PDV of durables ((1 - delta) * d/R) and durable consumption is the beginning of period
        durable stock (n).
        """
        solution_shape = (len(self.nNrmGrid), len(self.mNrmGrid))
        cFunc_array = np.zeros(solution_shape)
        dFunc_array = np.zeros(solution_shape)
        inv_vFunc_array = np.zeros(solution_shape)
        vPDFunc_array = np.zeros(solution_shape)


        #PDV of Beginning-of-Period Durable Stock
        PDVnNrmGrid = (1 - self.dDepr) * self.nNrmGrid/(self.Rfree[0])

        for i_n in range(len(self.nNrmGrid)):
            cFunc_array[i_n] = self.mNrmGrid #+ PDVnNrmGrid[i_n]
            dFunc_array[i_n] = np.ones_like(self.mNrmGrid) * self.nNrmGrid[i_n]
            inv_vFunc_array[i_n] = self.CRRAutility_inv(cFunc_array[i_n], dFunc_array[i_n])
            vPDFunc_array[i_n] = self.CRRAutilityPD(cFunc_array[i_n], dFunc_array[i_n])

        self.cFunc_terminal = LinearFast(cFunc_array, [self.nNrmGrid, self.mNrmGrid])
        self.dFunc_terminal = LinearFast(dFunc_array, [self.nNrmGrid, self.mNrmGrid])
        self.inv_vFunc_terminal = LinearFast(inv_vFunc_array, [self.nNrmGrid, self.mNrmGrid])
        self.vFunc_terminal =  ValueFuncCRRA(self.inv_vFunc_terminal, self.CRRA)
        self.vPDFunc_terminal = LinearFast(vPDFunc_array, [self.nNrmGrid, self.mNrmGrid])
        self.vPFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.adjusting_terminal = LinearFast(np.zeros(solution_shape), [self.nNrmGrid, self.mNrmGrid])

        ### Infinite Horizon solution
        self.xNrmMin = 0.0
        self.hNrmNow = 0.0
        self.MPCminNow = 1.0 #+ (1 - self.dDepr)/(self.Rfree[0]) #(1. - Thorn_R)
        self.MPCmaxNow = 1.0 #+ (1 - self.dDepr)/(self.Rfree[0])
        self.lSuS_array = np.zeros((len(self.nNrmGrid), 2))
        self.adjusting_terminal_array = np.ones(solution_shape)


    def update_solution_solution_Keep(self):
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep = self.cFunc_terminal,
            cFuncAdj = NullFunc,
            dFuncKeep = self.dFunc_terminal,
            dFuncAdj = NullFunc,
            vFuncKeep = self.vFunc_terminal,
            vFuncAdj = NullFunc,
            vPFuncKeep = self.vPDFunc_terminal,
            vPFuncAdj = NullFunc,
            inv_vFuncKeep = self.inv_vFunc_terminal,
            inv_vFuncAdj = NullFunc,
            cFunc = self.cFunc_terminal,
            dFunc = self.dFunc_terminal,
            aFunc = NullFunc,
            vFunc = self.vFunc_terminal,
            vPFunc = self.vPFunc_terminal,
            vPDFunc = self.vPDFunc_terminal,
            inv_vFunc = self.inv_vFunc_terminal,
            adjusting = self.adjusting_terminal,
            mNrmGrid=self.mNrmGrid,
            nNrmGrid=self.nNrmGrid,
            dNrmGrid = self.dNrmGrid,
            xNrmMin = self.xNrmMin,
            hNrm = self.hNrmNow,
            MPCmin = self.MPCminNow,
            MPCmax = self.MPCmaxNow,
            aXtraGrid=self.aXtraGrid,
            # # Constraint functions
            cFuncCnst = self.cFunc_terminal,
            dFuncCnst = self.dFunc_terminal,
            # # Comparing Upper Envelope
            EndOfPrdvP = None,
            EndOfPrdvPD = None,
            EndOfPrdv = None,
            aNrmGridNow = None,
            BoroCnstNat = None,
            xAdj_array = self.xNrmGrid,
            dAdj_array = None, 
            cAdj_array = None, 
            vPFunc_eval = self.vPFunc_terminal,
            vPDFunc_eval = self.vPDFunc_terminal,
            lSuS_array = self.lSuS_array,
            adjusting_array = self.adjusting_terminal_array,
        )



    def solve_terminal_solution_Keep_PF(self):
        """
        Agent cannot adjust, but receives the present discounted value of durable stock as a loan.
        Therefore, non-durable consumption is beginning-of-period market resources (m) plus the 
        PDV of durables ((1 - delta) * d/R) and durable consumption is the beginning of period
        durable stock (n).
        
        In Period T + 1: Perfect Foresight.
        For Frictionless case: Use Perfect Foresight solution.
        """

        ### 1) Get PF Solution (For Periods afterwards)
        LimitNow = self.calc_limit_inf()
        cFuncUnc_Frictionless = LimitNow["LimitcFunc1D"]
        dFuncUnc_Frictionless = LimitNow["LimitdFunc1D"]
        vFuncUnc_Frictionless = LimitNow["LimitvFunc1D"]
        inv_vFuncUnc_Frictionless = LimitNow["Limitinv_vFunc1D"]
        vPFuncUnc_Frictionless = MargValueFuncCRRA_dur(cFuncUnc_Frictionless, dFuncUnc_Frictionless, self.alpha, self.d_ubar, self.CRRA)
        vPDFuncUnc_Frictionless = MargValueFuncCRRA_dur(cFuncUnc_Frictionless, dFuncUnc_Frictionless, self.alpha, self.d_ubar, self.CRRA)

        ### 2) Frictionless for period T (With constraint + PF next period)
        ### Frictionless constraint: 
        """
        x = c + d. 
        With c = gam * d --> x = (1 + gam) * d. --> d = x/(1 + gam)

        v = U(c,d) + beta * v(x_t+1); where x_{t+1} = 1 + (1 - delta) * d 
        """
        dFuncCnst_Frictionless_array = self.xNrmGrid/(1 + self.gam) # Not correct given next period
        cFuncCnst_Frictionless_array = self.gam * dFuncCnst_Frictionless_array # Not correct given next period
        cFuncCnst_Frictionless = LinearFast(cFuncCnst_Frictionless_array, [self.xNrmGrid])
        dFuncCnst_Frictionless = LinearFast(dFuncCnst_Frictionless_array, [self.xNrmGrid])

        ### Make Lower Envelope
        cFunc_Frictionless = LowerEnvelope(cFuncUnc_Frictionless, cFuncCnst_Frictionless)
        dFunc_Frictionless = LowerEnvelope(dFuncUnc_Frictionless, dFuncCnst_Frictionless)
        vPFunc_Frictionless = MargValueFuncCRRA_dur(cFunc_Frictionless, dFunc_Frictionless, self.alpha, self.d_ubar, self.CRRA)
        vPDFunc_Frictionless = MargValueFuncCRRA_dur(cFunc_Frictionless, dFunc_Frictionless, self.alpha, self.d_ubar, self.CRRA)

        ### Make inv_vFunc taking constraint and unconstraint part into account and assume PF in next period.
        x_nextCnst = (1 - self.dDepr) * dFuncCnst_Frictionless(self.xNrmGrid) + 1 #Constraint means a = 0 --> m = 1
        vFuncCnst_Frictionless_array = self.CRRAutility(cFuncCnst_Frictionless(self.xNrmGrid), dFuncCnst_Frictionless(self.xNrmGrid)) + self.DiscFac[0] * self.LivPrb[0] * vFuncUnc_Frictionless(x_nextCnst)
        inv_vFuncCnst_Frictionless_array = utility_inv(vFuncCnst_Frictionless_array, self.CRRA)
        inv_vFuncCnst_Frictionless = LinearFast(inv_vFuncCnst_Frictionless_array, [self.xNrmGrid])

        ### Stack them together over xNrmGrid
        aFunc_Frictionless_array = self.xNrmGrid - cFunc_Frictionless(self.xNrmGrid) - dFunc_Frictionless(self.xNrmGrid)
        # Find idx where a > 1e-4
        idx = np.where(aFunc_Frictionless_array > 1e-4)[0][0]
        inv_vFunc_Frictionless_array = np.zeros_like(self.xNrmGrid)
        inv_vFunc_Frictionless_array[:idx] = inv_vFuncCnst_Frictionless(self.xNrmGrid[:idx])
        inv_vFunc_Frictionless_array[idx:] = inv_vFuncUnc_Frictionless(self.xNrmGrid[idx:])

        # Make Functions
        inv_vFunc_Frictionless = LinearFast(inv_vFunc_Frictionless_array, [self.xNrmGrid])
        vFunc_Frictionless = ValueFuncCRRA(inv_vFunc_Frictionless, self.CRRA)

        self.cFunc_Frictionless = cFunc_Frictionless
        self.dFunc_Frictionless = dFunc_Frictionless
        self.inv_vFunc_Frictionless = inv_vFunc_Frictionless
        self.vFunc_Frictionless = vFunc_Frictionless
        self.vPDFunc_Frictionless = vPDFunc_Frictionless
        self.vPFunc_Frictionless = vPFunc_Frictionless

        ### 3) Make Terminal Solution (Keep + PF next period)
        ### Create Solution Functions
        solution_shape = (len(self.nNrmGrid), len(self.mNrmGrid))
        cFunc_array = np.zeros(solution_shape)
        dFunc_array = np.zeros(solution_shape)
        aFunc_array = np.zeros(solution_shape)
        inv_vFunc_array = np.zeros(solution_shape)
        vPDFunc_array = np.zeros(solution_shape)
        mNrmGridNow = np.zeros(solution_shape)

        cFuncKeepList = []
        mNrmGrid_endo_array = []
        
        # Natural Borrowing Constraint (With TranShk and PermShk == 1)
        BoroCnstNat = (
                (-1)
                * (self.PermGroFac[0])
                / self.Rfree[0]
        )

        aNrmGridNow = self.aXtraGrid + BoroCnstNat #np.append(0.0, self.aXtraGrid)

        for i_n in range(len(self.nNrmGrid)):
            ### Calculate End Of Period Marginal Value Function using aNrmGrid and assuming PF
            ### Get next period's m and n (assume Perfect Foresight)
            m_next = self.Rfree[0] * aNrmGridNow + 1
            n_next = dFunc_array[i_n] * (1 - self.dDepr)
            x_next = m_next + n_next ### NO ADJUSTMENT COSTS IN T+1

            # EndOfPrdvP = self.DiscFac[0] * self.LivPrb[0] * self.Rfree[0] * self.PermGroFac[0] ** (-self.CRRA) * vPFuncFrictionless(n_next, m_next)
            EndOfPrdvP = self.DiscFac[0] * self.LivPrb[0] * self.Rfree[0] * self.PermGroFac[0] ** (-self.CRRA) * vPFuncUnc_Frictionless(x_next)

            cFunc_array = utilityP_inv(EndOfPrdvP, self.CRRA)
            dFunc_array = np.ones_like(aNrmGridNow) * self.nNrmGrid[i_n]
            mNrmGridNow = aNrmGridNow + cFunc_array

            cKeep_for_interpolation = np.insert(cFunc_array, 0, 0.0, axis=-1)
            mKeep_for_interpolation = np.insert(mNrmGridNow, 0, 0.0, axis=-1)
            
            # Create List of cFuncs and array of all endogenous mNrmGrid points
            cFuncKeepList.append(LinearFast(cKeep_for_interpolation, [mKeep_for_interpolation]))
            mNrmGrid_endo_array = np.append(mNrmGrid_endo_array, mKeep_for_interpolation)

        ### Post
        ### Keeper:
        # Overall mNrmGrid
        mNrmGridKeeper = np.unique(mNrmGrid_endo_array)
        mNrmGridKeeper = np.unique(mNrmGridKeeper)
        mNrmGridKeeper.sort()

        # Create cFuncKeep and dFuncKeep
        # Create empty container
        keep_shape = (len(self.nNrmGrid), len(mNrmGridKeeper))
        cFuncKeepUnc_array = np.zeros(keep_shape)
        dFuncKeepUnc_array = np.zeros(keep_shape)

        for i_n in range(len(self.nNrmGrid)):
            dFuncKeepUnc_array[i_n] = np.ones(len(mNrmGridKeeper)) * self.nNrmGrid[i_n]
            cFuncKeepUnc_array[i_n] = cFuncKeepList[i_n](mNrmGridKeeper)

        ### Make Functions
        cFuncKeepUnc = LinearFast(cFuncKeepUnc_array, [self.nNrmGrid, mNrmGridKeeper])
        dFuncKeepUnc = LinearFast(dFuncKeepUnc_array, [self.nNrmGrid, mNrmGridKeeper])

        # Make Consumption function similar to use_points_for_interpolation
        # Constraint part: 
        cFuncKeepCnst = LinearFast(np.array([[0.0,1.0],[0.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])
        # dFuncKeepCnst = LinearFast(np.array([[0.0,0.0],[1.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])

        self.cFunc_terminal = LowerEnvelope2D(cFuncKeepUnc, cFuncKeepCnst, nan_bool=False)
        self.dFunc_terminal = dFuncKeepUnc
        # self.dFunc_terminal = LowerEnvelope2D(dFuncKeepUnc, dFuncKeepCnst, nan_bool=False)

        ### Calculate Value Function and Marginal Value Function wrt Durable Stock
        for i_n in range(len(self.nNrmGrid)):
            n_aux = np.ones(len(self.mNrmGrid)) * self.nNrmGrid[i_n]
            cFunc_array = self.cFunc_terminal(n_aux, self.mNrmGrid)
            dFunc_array = n_aux #self.dFunc_terminal(n_aux, self.mNrmGrid)
            aFunc_array = self.mNrmGrid - cFunc_array

            # Next Period Values
            m_next = self.Rfree[0] * aFunc_array + 1
            n_next = dFunc_array * (1 - self.dDepr)
            x_next = m_next + n_next ### NO ADJUSTMENT COSTS IN T+1

            ### vFunc is U(C,D) + beta * LivPrb * E[vFunc(n',m')]
            # vFuncNow = self.CRRAutility(cFunc_array, dFunc_array) + self.DiscFac[0] * self.LivPrb[0] * vFuncUnc2D_Frictionless(n_next, m_next)
            vFuncNow = self.CRRAutility(cFunc_array, dFunc_array) + self.DiscFac[0] * self.LivPrb[0] * vFuncUnc_Frictionless(x_next)
            # vFuncNow = 1/(1 - (self.DiscFac[0] * self.LivPrb[0])) * self.CRRAutility(cFunc_array, dFunc_array) # Will always consume c, d (geometric series)
            # Geometric series including depreciation
            # vFuncNow = 1/(1 - (self.DiscFac[0] * self.LivPrb[0])) * self.CRRAutility(cFunc_array, dFunc_array/self.dDepr)
            inv_vFunc_array[i_n,1:] = utility_inv(vFuncNow[1:], self.CRRA)

            ### vPFunc is U'(C,D) + beta * LivPrb * (1 - dDepr) * E[vPDFunc(n',m')] Which is the Keepers solution
            # Lambda = (1 - self.dDepr) * self.DiscFac[0] * self.LivPrb[0] * vPDFuncFrictionless(n_next, m_next)
            Lambda = (1 - self.dDepr) * self.DiscFac[0] * self.LivPrb[0] * vPDFuncUnc_Frictionless(x_next)
            vPDFunc_array[i_n] = self.CRRAutilityPD(cFunc_array, dFunc_array) + Lambda

        inv_vFunc_array[0,0] = 0.0
        self.inv_vFunc_terminal = LinearFast(inv_vFunc_array, [self.nNrmGrid, self.mNrmGrid])
        self.vFunc_terminal =  ValueFuncCRRA(self.inv_vFunc_terminal, self.CRRA)
        # self.vPDFunc_terminal = LinearFast(vPDFunc_array, [self.nNrmGrid, self.mNrmGrid])
        # self.vPDFunc_terminal = BilinearInterp(vPDFunc_array, self.nNrmGrid, self.mNrmGrid)
        self.vPFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.vPDFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA) ### THIS IS THE ADJUSTERs VPD NOT OF THE KEEPER> TEST
        self.adjusting_terminal = LinearFast(np.zeros(solution_shape), [self.nNrmGrid, self.mNrmGrid])

        ### Infinite Horizon solution
        self.xNrmMin = 0.0
        self.hNrmNow = 1/(1 - (self.PermGroFac[0]/self.Rfree[0])) ### PF
        # self.hNrmNow = 0.0
        # Thorn_R = ((self.Rfree[0] * self.DiscFac[0])**(1/self.CRRA))/(self.Rfree[0])
        Thorn_R = ((self.Rfree[0] * self.DiscFac[0] * self.LivPrb[0])**(1/self.CRRA))/(self.Rfree[0])
        self.MPCminNow = 1 - Thorn_R
        # self.MPCminNow = 0.0 #+ (1 - self.dDepr)/(self.Rfree[0]) #(1. - Thorn_R)
        self.MPCmaxNow = 1.0 ### Constraint part #+ (1 - self.dDepr)/(self.Rfree[0])
        self.lSuS_array = np.zeros((len(self.nNrmGrid), 2))
        self.adjusting_terminal_array = np.zeros(solution_shape)

    def update_solution_solution_Keep_PF(self):
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep = self.cFunc_terminal,
            cFuncAdj = NullFunc,
            dFuncKeep = self.dFunc_terminal,
            dFuncAdj = NullFunc,
            vFuncKeep = self.vFunc_terminal,
            vFuncAdj = NullFunc,
            vPFuncKeep = self.vPDFunc_terminal,
            vPFuncAdj = NullFunc,
            inv_vFuncKeep = self.inv_vFunc_terminal,
            inv_vFuncAdj = NullFunc,
            cFunc = self.cFunc_terminal,
            dFunc = self.dFunc_terminal,
            aFunc = NullFunc,
            vFunc = self.vFunc_terminal,
            vPFunc = self.vPFunc_terminal,
            vPDFunc = self.vPDFunc_terminal,
            inv_vFunc = self.inv_vFunc_terminal,
            adjusting = self.adjusting_terminal,
            mNrmGrid=self.mNrmGrid,
            nNrmGrid=self.nNrmGrid,
            dNrmGrid = self.dNrmGrid,
            xNrmMin = self.xNrmMin,
            hNrm = self.hNrmNow,
            MPCmin = self.MPCminNow,
            MPCmax = self.MPCmaxNow,
            aXtraGrid=self.aXtraGrid,
            # # Constraint functions
            cFuncCnst = self.cFunc_terminal,
            dFuncCnst = self.dFunc_terminal,
            # # Comparing Upper Envelope
            EndOfPrdvP = None,
            EndOfPrdvPD = None,
            EndOfPrdv = None,
            aNrmGridNow = None,
            BoroCnstNat = None,
            xAdj_array = self.xNrmGrid,
            dAdj_array = None, 
            cAdj_array = None, 
            vPFunc_eval = self.vPFunc_terminal,
            vPDFunc_eval = self.vPDFunc_terminal,
            lSuS_array = self.lSuS_array,
            adjusting_array = self.adjusting_terminal_array,
            # d_kink = 0.0,
        )

    def solve_terminal_solution_Frictionless(self):
        """
        Solve the Frictionless infinite horizon version of the Problem
        """
        InfHorizon = deepcopy(self)
        InfHorizon.cycles = 0
        InfHorizon.SearchCost = 0
        InfHorizon.adjC = 0
        InfHorizon.terminal_method = "terminal_PF"
        InfHorizon.update()
        InfHorizon.solve()

        self.solution_terminal = deepcopy(InfHorizon.solution[0]) 

    def solve_terminal_solution_Adj(self):
        """
        Solves terminal Solution for the Adjuster Case. In this case, the agent always adjusts the durable stock such that c/d is the right ratio.
        """
        # a) keeper problem: keep durable stock and consume everything else
        self.cFuncKeep_terminal = LinearFast(np.array([[0.0,1.0],[0.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])
        self.dFuncKeep_terminal = LinearFast(np.array([[0.0,0.0],[1.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])

        keep_shape = (len(self.nNrmGridNow),len(self.mNrmGrid))

        # # Value and marginal value Functions
        # # i) empty container
        inv_vFuncKeep_terminal_array = np.zeros(keep_shape)
        # # ii) fill arrays
        for i_m in range(len(self.mNrmGrid)):
            if self.mNrmGrid[i_m] == 0:  # forced c = 0
                inv_vFuncKeep_terminal_array[:, i_m] = 0
                continue
            m_aux = np.ones(len(self.nNrmGridNow)) * self.mNrmGrid[i_m]
            inv_vFuncKeep_terminal_array[:,i_m] = self.CRRAutility_inv(m_aux, self.nNrmGridNow)

        # iii) Make Functions
        self.inv_vFuncKeep_terminal = LinearFast(inv_vFuncKeep_terminal_array, [self.nNrmGridNow, self.mNrmGrid])
        self.vFuncKeep_terminal = ValueFuncCRRA( self.inv_vFuncKeep_terminal, self.CRRA)
        self.vPFuncKeep_terminal = MargValueFuncCRRA_dur(self.cFuncKeep_terminal, self.dFuncKeep_terminal, self.CRRA,
                                                        self.alpha, self.d_ubar)
        # b) adjuster problem: (Note this is an analytical solution)
        cFuncAdj_terminal_array = self.alpha * self.xNrmGrid
        dFuncAdj_terminal_array = (1 - self.alpha) * self.xNrmGrid

        inv_vFuncAdj_terminal_array = self.CRRAutility_inv(self.xNrmGrid - dFuncAdj_terminal_array, dFuncAdj_terminal_array)

        # Create Functions in 1D for adjuster
        self.cFuncAdj_terminal = LinearFast(cFuncAdj_terminal_array, [self.xNrmGrid])
        self.dFuncAdj_terminal = LinearFast(dFuncAdj_terminal_array, [self.xNrmGrid])
        self.inv_vFuncAdj_terminal = LinearFast(inv_vFuncAdj_terminal_array, [self.xNrmGrid])
        self.vFuncAdj_terminal = ValueFuncCRRA(self.inv_vFuncAdj_terminal, self.CRRA)
        self.vPFuncAdj_terminal = MargValueFuncCRRA_dur(self.cFuncAdj_terminal, self.dFuncAdj_terminal, self.CRRA,
                                                          self.alpha, self.d_ubar)
        
        # Adjuster in nxm
        ### Prepare matrices
        solution_shape = (len(self.nNrmGridNow), len(self.mNrmGrid))
        vFuncAdj_SOL_array = np.zeros(solution_shape)
        inv_vFuncAdj_SOL_array = np.zeros(solution_shape)
        cFuncAdj_SOL_array = np.zeros(solution_shape)
        dFuncAdj_SOL_array = np.zeros(solution_shape)
        vPDFuncAdj_SOL_array = np.zeros(solution_shape)
        adjusting_array = np.ones(solution_shape)
        aFuncAdj_Sol_array = np.zeros(solution_shape)
        if self.ScaleBool == True:
            adjC_scaled = self.adjC #adjC_scaled_func(self.adjC, self.mNrmGrid, self.Rfree[0], self.ScaleFac)
            # adjC_scaled = self.adjC/(self.mNrmGrid * (self.Rfree[0] - 1 /self.Rfree[0]))
            # adjC_scaled = np.clip(adjC_scaled, 0, self.adjC)
            # adjC_scaled[self.mNrmGrid == 0] = self.adjC
        else:
            adjC_scaled = self.adjC

        for i_n in range(len(self.nNrmGrid)):
            n_aux = np.ones_like(self.mNrmGrid) * self.nNrmGrid[i_n]
            x_aux = (1 - adjC_scaled) * n_aux + self.mNrmGrid

            vFuncAdj_SOL_array[i_n] = self.vFuncAdj_terminal(x_aux)
            inv_vFuncAdj_SOL_array[i_n] =  self.inv_vFuncAdj_terminal(x_aux)
            cFuncAdj_SOL_array[i_n] = self.cFuncAdj_terminal(x_aux)
            dFuncAdj_SOL_array[i_n] = self.dFuncAdj_terminal(x_aux)   
            vPDFuncAdj_SOL_array[i_n] = (1 - adjC_scaled) * self.CRRAutilityP(cFuncAdj_SOL_array[i_n], dFuncAdj_SOL_array[i_n])  # Adjust

        # Create functions in 2D
        self.adjusting_terminal = LinearFast(adjusting_array, [self.nNrmGrid, self.mNrmGrid])
        self.cFunc_terminal = LinearFast(cFuncAdj_SOL_array, [self.nNrmGrid, self.mNrmGrid])
        self.dFunc_terminal = LinearFast(dFuncAdj_SOL_array, [self.nNrmGrid, self.mNrmGrid])
        self.inv_vFunc_terminal = LinearFast(inv_vFuncAdj_SOL_array, [self.nNrmGrid, self.mNrmGrid])
        self.vPDFunc_terminal = LinearFast(vPDFuncAdj_SOL_array, [self.nNrmGrid, self.mNrmGrid])
        self.vFunc_terminal = ValueFuncCRRA(self.inv_vFunc_terminal, self.CRRA)
        self.vPFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.aFunc_terminal = LinearFast(aFuncAdj_Sol_array, [self.nNrmGrid, self.mNrmGrid])

        self.hNrmNow = 0.0
        self.MPCminNow = self.alpha
        self.MPCmaxNow = 1.0
        self.lSuS_array = np.zeros((len(self.nNrmGrid), 2))
        self.adjusting_terminal_array = adjusting_array
        

        # ### Evaluation function 
        # nNrmGrid_eval = self.nNrmGrid[self.nNrmGrid >= ]
    def update_solution_terminal_Adj(self):
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep = self.cFuncKeep_terminal,
            cFuncAdj = self.cFuncAdj_terminal,
            dFuncKeep = self.dFuncKeep_terminal,
            dFuncAdj =self.dFuncAdj_terminal,
            vFuncKeep = self.vFuncKeep_terminal,
            vFuncAdj = self.vFuncAdj_terminal,
            vPFuncKeep = self.vPDFunc_terminal,
            vPFuncAdj = self.vPFuncAdj_terminal,
            inv_vFuncKeep = self.inv_vFuncKeep_terminal,
            inv_vFuncAdj = self.inv_vFuncAdj_terminal,

            cFunc = self.cFunc_terminal,
            dFunc = self.dFunc_terminal,
            aFunc = self.aFunc_terminal,
            vFunc = self.vFunc_terminal,
            vPFunc = self.vPFunc_terminal,
            vPDFunc = self.vPDFunc_terminal,
            inv_vFunc = self.inv_vFunc_terminal,
            adjusting = self.adjusting_terminal,
            mNrmGrid=self.mNrmGrid,
            nNrmGrid=self.nNrmGrid,
            dNrmGrid = self.dNrmGrid,
            xNrmMin = self.xNrmMin,
            hNrm = self.hNrmNow,
            MPCmin = self.MPCminNow,
            MPCmax = self.MPCmaxNow,
            aXtraGrid=self.aXtraGrid,
            # # Constraint functions
            cFuncCnst = self.cFuncKeep_terminal,
            dFuncCnst = self.dFuncKeep_terminal,
            # # Comparing Upper Envelope
            EndOfPrdvP = None,
            EndOfPrdvPD = None,
            EndOfPrdv = None,
            aNrmGridNow = None,
            BoroCnstNat = None,
            xAdj_array = self.xNrmGrid,
            dAdj_array = None, 
            cAdj_array = None, 
            vPFunc_eval = self.vPFunc_terminal,
            vPDFunc_eval = self.vPDFunc_terminal,
            lSuS_array = self.lSuS_array,
            adjusting_array = self.adjusting_terminal_array
        )        
    
    ################################################################################################################
    def construct_lognormal_income_process_unemployment_truncated(self):
        """
        Generates a list of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation PermShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  Transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        PermShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        PermShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        TranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        TranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float or [float]
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float or None
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float or [float]
            Transitory income received when unemployed.
        IncUnempRet : float or None
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.
        truncation : int
            Number of SD of Shocks until truncation

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distributiony]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.
        """
        # Unpack the parameters from the input
        T_cycle = self.T_cycle
        PermShkStd = self.PermShkStd
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd
        TranShkCount = self.TranShkCount
        T_retire = self.T_retire
        UnempPrb = self.UnempPrb
        IncUnemp = self.IncUnemp
        UnempPrbRet = self.UnempPrbRet
        IncUnempRet = self.IncUnempRet
        truncation = self.truncation

        # Make sure they have the same length
        if len(self.taxrate) < self.T_cycle:
            self.taxrate = [self.taxrate[0]] * self.T_cycle

        if len(self.TranShkMean_Func) < self.T_cycle:
            self.TranShkMean_Func = [self.TranShkMean_Func[0]] * self.T_cycle

        if len(self.labor) < self.T_cycle:
            self.labor = [self.labor[0]] * self.T_cycle

        if len(self.wage) < self.T_cycle:
            self.wage = [self.wage[0]] * self.T_cycle

        truncation_list = [self.truncation]  * self.T_cycle
        taxrate = self.taxrate
        TranShkMean_Func = self.TranShkMean_Func
        labor = self.labor
        wage = self.wage

        if T_retire > 0:
            normal_length = T_retire
            retire_length = T_cycle - T_retire
        else:
            normal_length = T_cycle
            retire_length = 0

        if all(
                [
                    isinstance(x, (float, int)) or (x is None)
                    for x in [UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet]
                ]
        ):

            UnempPrb_list = [UnempPrb] * normal_length + [UnempPrbRet] * retire_length
            IncUnemp_list = [IncUnemp] * normal_length + [IncUnempRet] * retire_length

        elif all([isinstance(x, list) for x in [UnempPrb, IncUnemp]]):

            UnempPrb_list = UnempPrb
            IncUnemp_list = IncUnemp

        else:

            raise Exception(
                "Unemployment must be specified either using floats for UnempPrb,"
                + "IncUnemp, UnempPrbRet, and IncUnempRet, in which case the "
                + "unemployment probability and income change only with retirement, or "
                + "using lists of length T_cycle for UnempPrb and IncUnemp, specifying "
                + "each feature at every age."
            )

        PermShkCount_list = [PermShkCount] * normal_length + [1] * retire_length
        TranShkCount_list = [TranShkCount] * normal_length + [1] * retire_length

        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        neutral_measure_list = [self.neutral_measure] * len(PermShkCount_list)

        IncShkDstn = IndexDistribution(
            engine=HANKIncShkDstn_Truncated,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
                "truncation": truncation_list,
            },
            RNG=self.RNG,
        )

        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk_Truncated,
            conditional={
                "sigma": PermShkStd,
                "n_approx": PermShkCount_list,
                "neutral_measure": neutral_measure_list,
                "truncation": truncation_list,
            },
        )

        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk_HANK_Truncated,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "n_approx": TranShkCount_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
                "truncation": truncation_list,

            },
        )

        return IncShkDstn, PermShkDstn, TranShkDstn

    def construct_lognormal_income_process_unemployment_truncated_jacobian(self, UnempPrb_old, IncUnemp_old):
        """
        Generates a list of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation PermShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  Transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        PermShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        PermShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        TranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        TranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float or [float]
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float or None
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float or [float]
            Transitory income received when unemployed.
        IncUnempRet : float or None
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.
        truncation : int
            Number of SD of Shocks until truncation

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distributiony]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.
        """
        # Unpack the parameters from the input
        T_cycle = self.T_cycle
        PermShkStd = self.PermShkStd
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd
        TranShkCount = self.TranShkCount
        T_retire = self.T_retire
        UnempPrb = self.UnempPrb
        IncUnemp = self.IncUnemp
        UnempPrbRet = self.UnempPrbRet
        IncUnempRet = self.IncUnempRet
        truncation = self.truncation

        # Make sure they have the same length
        if len(self.taxrate) < self.T_cycle:
            self.taxrate = [self.taxrate[0]] * self.T_cycle

        if len(self.TranShkMean_Func) < self.T_cycle:
            self.TranShkMean_Func = [self.TranShkMean_Func[0]] * self.T_cycle

        if len(self.labor) < self.T_cycle:
            self.labor = [self.labor[0]] * self.T_cycle

        if len(self.wage) < self.T_cycle:
            self.wage = [self.wage[0]] * self.T_cycle

        truncation_list = [self.truncation]  * self.T_cycle
        taxrate = self.taxrate
        TranShkMean_Func = self.TranShkMean_Func
        labor = self.labor
        wage = self.wage

        if T_retire > 0:
            normal_length = T_retire
            retire_length = T_cycle - T_retire
        else:
            normal_length = T_cycle
            retire_length = 0

        if all(
                [
                    isinstance(x, (float, int)) or (x is None)
                    for x in [UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet]
                ]
        ):

            UnempPrb_list = [UnempPrb] * normal_length + [UnempPrbRet] * retire_length
            IncUnemp_list = [IncUnemp] * normal_length + [IncUnempRet] * retire_length

        elif all([isinstance(x, list) for x in [UnempPrb, IncUnemp]]):

            UnempPrb_list = UnempPrb
            IncUnemp_list = IncUnemp

        else:

            raise Exception(
                "Unemployment must be specified either using floats for UnempPrb,"
                + "IncUnemp, UnempPrbRet, and IncUnempRet, in which case the "
                + "unemployment probability and income change only with retirement, or "
                + "using lists of length T_cycle for UnempPrb and IncUnemp, specifying "
                + "each feature at every age."
            )

        PermShkCount_list = [PermShkCount] * normal_length + [1] * retire_length
        TranShkCount_list = [TranShkCount] * normal_length + [1] * retire_length

        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        neutral_measure_list = [self.neutral_measure] * len(PermShkCount_list)
        UnempPrb_old_List = [UnempPrb_old] * len(PermShkCount_list)
        IncUnemp_old_List = [IncUnemp_old] * len(PermShkCount_list)
        IncShkDstn = IndexDistribution(
            engine=HANKIncShkDstn_Truncated_Jacobian,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "UnempPrb_old": UnempPrb_old_List,
                "IncUnemp_old": IncUnemp_old_List,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
                "truncation": truncation_list,
            },
            RNG=self.RNG,
        )

        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk_Truncated,
            conditional={
                "sigma": PermShkStd,
                "n_approx": PermShkCount_list,
                "neutral_measure": neutral_measure_list,
                "truncation": truncation_list,
            },
        )

        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk_HANK_Truncated_Jacobian,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "UnempPrb_old": UnempPrb_old_List,
                "IncUnemp_old": IncUnemp_old_List,
                "n_approx": TranShkCount_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
                "truncation": truncation_list,

            },
        )

        return IncShkDstn, PermShkDstn, TranShkDstn
    
    ####################################################################################################################
    def calc_limit(self, t):
        '''
        Calculates limit function for each policy function. Use it after the model is solved.
        '''

        alpha = self.alpha
        dDepr = self.dDepr
        # adjC = self.adjC
        # Rfree = self.Rfree[min(t, len(self.Rfree)-1)]
        Rfree = self.Rfree[t]

        self.MPCminNow = self.solution[t].MPCmin
        self.hNrmNow = self.solution[t].hNrm
        ### Get gamma and Lambda_inverse
        if alpha <1:
            gam = (alpha/(1 - alpha)) * (Rfree - 1 + dDepr)/Rfree ## Includes Durables
            Lambda_inv = (1 + gam) / (1 + gam*(Rfree - 1 + dDepr)/Rfree)
        else:
            gam = 1
            Lambda_inv = 1

        MPXminNow = np.min([self.MPCminNow * Lambda_inv, 1.0])

        ### Limit Function:
        LimitexFunc1D = lambda x: MPXminNow * (self.hNrmNow + x)
        LimitcFunc1D = lambda x: MPXminNow * (self.hNrmNow + x) * (gam/(1 + gam))
        LimitdFunc1D = lambda x: MPXminNow * (self.hNrmNow + x) * (1/(1 + gam))
        LimitaFunc1D = lambda x: x - (MPXminNow * (self.hNrmNow + x))

        LimitexFunc2D = lambda n, m: MPXminNow * (self.hNrmNow + n + m)
        LimitcFunc2D = lambda n, m: MPXminNow * (self.hNrmNow + n + m) * (gam/(1 + gam))
        LimitdFunc2D = lambda n, m: MPXminNow * (self.hNrmNow + n + m) * (1/(1 + gam))
        LimitaFunc2D = lambda n, m: (n + m) - (MPXminNow * (self.hNrmNow + n + m))
        LimitadjFunc2D = lambda n, m: 1

        # Limit Gradient (Frictionless case)
        LimitexGrad1D = lambda x: [MPXminNow * np.ones_like(x)]
        LimitcGrad1D = lambda x: [MPXminNow * np.ones_like(x) * (gam/(1 + gam))]
        LimitdGrad1D = lambda x: [MPXminNow * np.ones_like(x) * (1/(1 + gam))]
        LimitaGrad1D = lambda x: [np.ones_like(x) - MPXminNow * np.ones_like(x)]

        LimitexGrad2D = lambda n, m: LimitexGrad1D(n + m)
        LimitcGrad2D = lambda n, m: LimitcGrad1D(n + m)
        LimitdGrad2D = lambda n, m: LimitdGrad1D(n + m)
        LimitaGrad2D = lambda n, m: LimitaGrad1D(n + m)
        LimitadjGrad2D = lambda n, m: [0]

        return {
            "LimitexFunc1D": LimitexFunc1D,
            "LimitcFunc1D": LimitcFunc1D,
            "LimitdFunc1D": LimitdFunc1D,
            "LimitaFunc1D": LimitaFunc1D,
            "LimitexFunc2D": LimitexFunc2D,
            "LimitcFunc2D": LimitcFunc2D,
            "LimitdFunc2D": LimitdFunc2D,
            "LimitaFunc2D": LimitaFunc2D,
            "LimitadjFunc2D": LimitadjFunc2D,
            "LimitexGrad1D": LimitexGrad1D,
            "LimitdGrad1D": LimitdGrad1D,
            "LimitcGrad1D": LimitcGrad1D,
            "LimitaGrad1D": LimitaGrad1D,
            "LimitexGrad2D": LimitexGrad2D,
            "LimitdGrad2D": LimitdGrad2D,
            "LimitcGrad2D": LimitcGrad2D,
            "LimitaGrad2D": LimitaGrad2D,
            "LimitadjGrad2D": LimitadjGrad2D,
                
        }
    def add_limit(self):
        '''
        This function adds a limit function to each policy function. Use it after the model is solved.
        '''
        # print("here")
        for t in range(self.T_cycle):
            LimitNow = self.calc_limit(t)

            self.solution[t].cFunc =  DecayInterp(
                self.solution[0].cFunc,
                limit_fun= LimitNow["LimitcFunc2D"],
                limit_grad=LimitNow["LimitcGrad2D"],
                extrap_method=self.extrap_method
                )
            
            self.solution[t].dFunc =  DecayInterp(
                self.solution[0].dFunc,
                limit_fun= LimitNow["LimitdFunc2D"],
                limit_grad=LimitNow["LimitdGrad2D"],
                extrap_method=self.extrap_method
                )

            self.solution[t].aFunc =  DecayInterp(
                self.solution[0].aFunc,
                limit_fun= LimitNow["LimitaFunc2D"],
                limit_grad=LimitNow["LimitaGrad2D"],
                extrap_method=self.extrap_method
                )

            self.solution[t].adjusting =  DecayInterp(
                self.solution[0].adjusting,
                limit_fun= LimitNow["LimitadjFunc2D"],
                limit_grad=LimitNow["LimitadjGrad2D"],
                extrap_method=self.extrap_method
                )
                        
            self.solution[t].cFuncAdj =  DecayInterp(
                self.solution[0].cFuncAdj,
                limit_fun= LimitNow["LimitcFunc1D"],
                limit_grad=LimitNow["LimitcGrad1D"],
                extrap_method=self.extrap_method
                )
            
            self.solution[t].dFuncAdj =  DecayInterp(
                self.solution[0].dFuncAdj,
                limit_fun= LimitNow["LimitdFunc1D"],
                limit_grad=LimitNow["LimitdGrad1D"],
                extrap_method=self.extrap_method
                )
       
    ####################################################################################################################
    ### SIMULATION PART STARTS HERE: Only functions are shown which are changed
    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Additionally, creates nNrm (normalized durable stock) drawn from lognormal distribution.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)
        # Adding nNrm for birth
        N = np.sum(which_agents)  # Number of new consumers to make
        # self.state_now["nNrm"][which_agents] = Lognormal(
        self.state_now["dNrm"][which_agents] = (Lognormal(
            mu=self.dNrmInitMean,
            sigma=self.dNrmInitStd,
            seed=self.RNG.integers(0, 2**31 - 1),
        ).draw(N))/(1 - self.dDepr) #* self.shocks['PermShk']
        if self.dDepr < 1:
            self.state_now["dNrm"][which_agents] = self.state_now["dNrm"][which_agents]/(1 - self.dDepr)
        else:
            self.state_now["dNrm"][which_agents] = 0
        
        # As this is End-of-Period durable stock, divide by depreciation and shocks such that beginning of period durable
        # stock is equal to draw.
        return None
    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = np.ones(self.AgentCount)
        if "Rfree" in self.time_inv:
            RfreeNow = RfreeNow * self.Rfree
        elif "Rfree" in self.time_vary:
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                RfreeNow[these] = self.Rfree[t]
        
        RSpreadNow = np.zeros(self.AgentCount) 
        if "RSpreadNow" in self.time_inv:
            RSpreadNow = RSpreadNow * self.RSpread
        elif "RSpreadNow" in self.time_vary:
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                RSpreadNow[these] = self.RSpread[t]

        return RfreeNow, RSpreadNow
    
    def transition(self): # Adding nNrm # With Maintenance, Operation Costs, and RSpread
        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]
        dNrmPrev = self.state_prev["dNrm"]
        RfreeNow, RSpreadNow = self.get_Rfree()
        
        RSpreadeffNow = np.zeros(self.AgentCount) 
        
        # RSpreadNow is self.RSpread[t] if self.state_prev["aNrm"] < 0 and 0 otherwise
        RSpreadeffNow[self.state_prev["aNrm"] < 0] = RSpreadNow[self.state_prev["aNrm"] < 0]

        # Calculate new states: normalized market resources and permanent income level
        # Updated permanent income level
        pLvlNow = pLvlPrev * self.shocks["PermShk"]
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev["PlvlAgg"] * self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = (RfreeNow + RSpreadeffNow) / self.shocks["PermShk"]

        bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income

        # Car payments
        carPayNow = dNrmPrev * (self.OpCost + self.dDepr * self.MaintCost) / self.shocks["PermShk"]
        
        # Market resources after income
        mNrmNow = bNrmNow - carPayNow + self.shocks["TranShk"]

        nNrmNow = dNrmPrev * (1 - self.dDepr * (1 - self.MaintCost))/self.shocks['PermShk'] # Normalized Durable Stock
        
        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None, None, nNrmNow, None
        # Nones are aNrm, aLvl, dNrm as those are defined by the choices made in the period.

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        # Added
        dNrmNow = np.zeros(self.AgentCount) + np.nan
        # exNrmNow = np.zeros(self.AgentCount) + np.nan
        # adjusting = np.zeros(self.AgentCount) + np.nan

        MPCnow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            # CHANGE: cFunc has dimensions (nNrm, mNrm) Optimal c given the durable stock nNrm and market resources
            cNrmNow[these] = self.solution[t].cFunc(self.state_now['nNrm'][these], self.state_now['mNrm'][these])
            dNrmNow[these] = self.solution[t].dFunc(self.state_now['nNrm'][these], self.state_now['mNrm'][these])
            # exNrmNow[these] = self.solution[t].exFunc(self.state_now['nNrm'][these], self.state_now['mNrm'][these])
            # adjusting[these] = self.solution[t].adjusting(self.state_now['nNrm'][these], self.state_now['mNrm'][these])

            # cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
            #     self.state_now["mNrm"][these]
            # )
        self.controls['cNrm'] = cNrmNow
        self.controls['dNrm'] = dNrmNow
        # self.controls['exNrm'] = exNrmNow
        # self.controls['adjusting'] = adjusting
        return None

        # # MPCnow is not really a control
        # self.MPCnow = MPCnow
        # return None

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sol = self.solution[self.t_cycle[0]]
        n = self.state_now['nNrm']
        m = self.state_now['mNrm']

        self.state_now['aNrm'] = np.zeros(self.AgentCount)
        self.state_now['searcheffort'] = np.zeros(self.AgentCount)
        self.state_now['adjusting'] = np.zeros(self.AgentCount)

        adjC_scaled = self.tau_func(m + n)
        xNrm = m + (1 - adjC_scaled) * n

        self.state_now['searcheffort'] = np.clip(sol.adjusting(n, m), 0, 1)
        random_draws = np.random.rand(self.AgentCount)
        self.state_now['adjusting'] = (
            random_draws < self.state_now['searcheffort']
        ).astype(int)

        # Adjuster assets: 2D (n, m) policy (same grid as transition matrix)
        if not isinstance(sol.aFuncAdj, NullFunc):
            aNrmAdj = sol.aFuncAdj(n, m)
        elif self.alpha >= 1:
            aNrmAdj = (
                xNrm
                - sol.cFuncAdj(np.zeros_like(n), xNrm)
                - sol.dFuncAdj(np.zeros_like(n), xNrm)
            )
        else:
            aNrmAdj = xNrm - sol.cFuncAdj(xNrm) - sol.dFuncAdj(xNrm)

        # Keeper assets: 2D (n, m) policy
        if not isinstance(sol.aFuncKeep, NullFunc):
            aNrmKeep = sol.aFuncKeep(n, m)
        else:
            aNrmKeep = m - sol.cFuncKeep(n, m)

        adj = self.state_now['adjusting']
        self.state_now['aNrm'] = np.where(adj == 1, aNrmAdj, aNrmKeep)

        # Useful in some cases to precalculate asset level
        self.state_now['aLvl'] = self.state_now['aNrm'] * self.state_now['pLvl']

        # Add durable stocks normalized and in levels
        self.state_now['dNrm'] = self.controls['dNrm'] #Durable consumption this period is equal to the stock next period
        self.state_now['dLvl'] = self.state_now['dNrm'] * self.state_now['pLvl']
        self.state_now['nLvl'] = self.state_now['nNrm'] * self.state_now['pLvl']
        self.state_now['mLvl'] = self.state_now['mNrm'] * self.state_now['pLvl']
        # moves now to prev
        # super().get_poststates() # If this is true, then calculates poststates based on ConsIndShockModel

        return None

    ### TRANSITION MATRIX:
    def define_distribution_grid_dur(self, dist_mGrid=None, dist_nGrid=None, m_density=0, timestonest = None):

        """
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and durable stock may be prespecified
        as dist_mGrid and dist_nGrid, respectively. If not then default grid is computed based off given parameters.

        Added: durable grid

        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources

        dist_nGrid : np.array
                 Prespecified grid for distribution over normalized durable stock.

        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.

        TranMatrixmCount: float
                Number of gridpoints for market resources grid.

        TranMatrixnCount: float
                Number of gridpoints for durable stock grid.

        Returns
        -------
        None
        """

        if not hasattr(self,
                       "neutral_measure"):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
            self.neutral_measure = False

        # if TranMatrixmCount == None:
        #     TranMatrixmCount = self.aXtraCount
        # else:
        #     m_points = num_pointsM
        #
        # if num_pointsN == None:
        #     n_points = self.nNrmCount
        # else:
        #     n_points = num_pointsN

        if not isinstance(timestonest, int):
            timestonest = self.aXtraNestFac
        else:
            timestonest = timestonest

        if self.cycles == 0:
            ### Baseline
            if not hasattr(dist_mGrid, '__len__'):
                aMin = np.min(self.solution[0].mNrmMin)
                if aMin < 0:
                    # Make sure it is an integer
                    TranMatrixmCountNow = int(self.TranMatrixmCount/2)
                else:
                    TranMatrixmCountNow = int(self.TranMatrixmCount)

                aXtra_Grid = make_grid_exp_mult(
                    ming=self.TranMatrixmMin, maxg=self.TranMatrixmMax, ng=TranMatrixmCountNow,
                    timestonest=timestonest)  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid, -1)
                    axtra_shifted = np.insert(axtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid, new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)

                # self.dist_mGrid = aXtra_Grid
                ### Add negative values to the grid
                
                if aMin < 0:
                    aXtra_Neg = make_grid_exp_mult(0.0, -aMin, ng=TranMatrixmCountNow, timestonest=timestonest)
                    aXtra_Neg = aXtra_Neg * (-1)  # Make negative values
                    aXtra_Grid = np.concatenate((aXtra_Neg, aXtra_Grid))  # Add negative values to the grid
                    aXtra_Grid = np.unique(aXtra_Grid)  # Remove duplicates

                # ### Add negative values to the grid
                # aXtra_Grid = np.append(aXtra_Grid, self.solution[0].mNrmMin)
                # aXtra_Grid = np.unique(aXtra_Grid)

                self.dist_mGrid = aXtra_Grid
            else:
                self.dist_mGrid = dist_mGrid  # If grid of market resources prespecified then use as mgrid

            if not hasattr(dist_nGrid, '__len__'):

                nXtra_Grid = construct_grid(self.TranMatrixnMin, self.TranMatrixnMax, self.TranMatrixnCount, self.grid_type, self.NestFac)
                for i in range(m_density):
                    nxtra_shifted = np.delete(nXtra_Grid, -1)
                    nxtra_shifted = np.insert(nxtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = nXtra_Grid - nxtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_N_grid = nxtra_shifted + dist_betw_pts_half
                    nXtra_Grid = np.concatenate((nXtra_Grid, new_N_grid))
                    nXtra_Grid = np.sort(nXtra_Grid)
                self.dist_nGrid = nXtra_Grid
            else:
                self.dist_nGrid = dist_nGrid  # If grid of durable stock prespecified then use it as ngrid

        elif self.cycles > 1:
            raise Exception('define_distribution_grid requires cycles = 0 or cycles = 1')

        elif self.T_cycle != 0:
            ### Baseline
            if not hasattr(dist_mGrid, '__len__'):
                aMin = np.min(self.solution[0].mNrmMin)
                if aMin < 0:
                    TranMatrixmCountNow = int(self.TranMatrixmCount/2)
                else:
                    TranMatrixmCountNow = int(self.TranMatrixmCount)

                aXtra_Grid = make_grid_exp_mult(
                    ming=self.TranMatrixmMin, maxg=self.TranMatrixmMax, ng=TranMatrixmCountNow,
                    timestonest=timestonest)  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid, -1)
                    axtra_shifted = np.insert(axtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid, new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)

                # self.dist_mGrid = aXtra_Grid
                ### Add negative values to the grid
                aMin = np.min(self.solution[0].mNrmMin)
                if aMin < 0:
                    aXtra_Neg = make_grid_exp_mult(0.0, -aMin, ng=TranMatrixmCountNow, timestonest=timestonest)
                    aXtra_Neg = aXtra_Neg * (-1)  # Make negative values
                    aXtra_Grid = np.concatenate((aXtra_Neg, aXtra_Grid))  # Add negative values to the grid
                    aXtra_Grid = np.unique(aXtra_Grid)  # Remove duplicates
                self.dist_mGrid = aXtra_Grid
            else:
                self.dist_mGrid = dist_mGrid  # If grid of market resources prespecified then use as mgrid

            if not hasattr(dist_nGrid, '__len__'):

                nXtra_Grid = construct_grid(self.TranMatrixnMin, self.TranMatrixnMax, self.TranMatrixnCount, self.grid_type, self.NestFac)
                for i in range(m_density):
                    nxtra_shifted = np.delete(nXtra_Grid, -1)
                    nxtra_shifted = np.insert(nxtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = nXtra_Grid - nxtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_N_grid = nxtra_shifted + dist_betw_pts_half
                    nXtra_Grid = np.concatenate((nXtra_Grid, new_N_grid))
                    nXtra_Grid = np.sort(nXtra_Grid)
                self.dist_nGrid = nXtra_Grid
            else:
                self.dist_nGrid = dist_nGrid  # If grid of durable stock prespecified then use it as ngrid

    # 3. Calculate transition matrix
    def calc_transition_matrix_dur(self, shk_dstn=None):
        '''
        Calculates how the distribution of agents across market resources
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem.
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.


        Parameters
        ----------
            shk_dstn: list
                list of income shock distributions
        Returns
        -------
        None

        '''
        if self.cycles == 0:
            if not hasattr(shk_dstn, 'pmv'):
                shk_dstn = self.IncShkDstn

            dist_nGrid = self.dist_nGrid  # Grid of durable stock
            dist_mGrid = self.dist_mGrid  # Grid of market resources

            ### Baseline
            # Get Steady State Policy Grids. They depend on the discrete choice 
            SS_Pol_shape = (len(dist_nGrid), len(dist_mGrid))
            self.aPol_Grid = np.zeros(SS_Pol_shape)
            self.cPol_Grid = np.zeros(SS_Pol_shape)
            self.dPol_Grid = np.zeros(SS_Pol_shape)
            self.adjPol_Grid = np.zeros(SS_Pol_shape)
            self.dexPol_Grid = np.zeros(SS_Pol_shape)
            
            for i_n in range(len(dist_nGrid)):
                n_aux = np.ones(len(dist_mGrid)) * dist_nGrid[i_n]
                self.cPol_Grid[i_n] = self.solution[0].cFunc(n_aux, dist_mGrid)  # Steady State Consumption Policy Grid
                self.dPol_Grid[i_n] = self.solution[0].dFunc(n_aux, dist_mGrid)
                self.dexPol_Grid[i_n] = self.solution[0].dFunc(n_aux, dist_mGrid) - n_aux
                # Depending on adjusting or not, we have a different budget constraint
                self.adjPol_Grid[i_n] = self.solution[0].adjusting(n_aux, dist_mGrid)
                # ### bound by 0 and 1
                self.adjPol_Grid[i_n] = np.clip(self.adjPol_Grid[i_n], 0, 1)
                self.aPol_Grid[i_n] = self.solution[0].aFunc(n_aux, dist_mGrid)
            
            # Total Expenditure Policy Grid
            self.texPol_Grid = self.cPol_Grid + self.dexPol_Grid

            ### Check for NaNs
            if np.any(np.isnan(self.cPol_Grid)):
                print('cPol_Grid has NaNs')
            
            # Obtain shock values and shock probabilities from income distribution
            R_vec = np.where(self.aPol_Grid < 0, self.Rfree[0] + self.RSpread[0], self.Rfree[0])
            bNext = (R_vec * self.aPol_Grid) - self.dPol_Grid * (self.OpCost + self.dDepr * self.MaintCost)  # Bank Balances next period (Interest rate * assets)
            nNext_unshocked = (1 - self.dDepr * (1 - self.MaintCost)) * self.dPol_Grid  # Durable Stock next period after depreciation
            
            shk_prbs = shk_dstn[0].pmv #shk_dstn[0].pmf  # Probability of shocks
            tran_shks = shk_dstn[0].atoms[1] #shk_dstn[0].X[1]  # Transitory shocks
            perm_shks = shk_dstn[0].atoms[0] #shk_dstn[0].X[0]  # Permanent shocks
            LivPrb = self.LivPrb[0]  # Update probability of staying alive

            if self.CarCrashBool == True:
                ### Adjust shock probability to account for CarCrash Shocks
                ### perm shock x tran shock x car crash shock
                tran_shks_prob = np.zeros(self.TranShkCount + 1)
                tran_shks_prob[0] = self.UnempPrb[0]
                tran_shks_prob[1:] = (1 - self.UnempPrb[0])/self.TranShkCount
                perm_shks_prob = np.ones(self.PermShkCount)/self.PermShkCount
                carcrash_shks_prob = np.array([self.CarCrashPrb, 1 - self.CarCrashPrb])

                carcrash_shks_raw = np.array([self.CarCrashMin, 1])
                tran_shks_raw = np.unique(tran_shks)
                perm_shks_raw = np.unique(perm_shks)

                ### Make new shock probabilities
                shk_prbs_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))
                for i in range(len(perm_shks_prob)):
                    for j in range(len(tran_shks_prob)):
                        for k in range(len(carcrash_shks_prob)):
                            shk_prbs_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = perm_shks_prob[i] * tran_shks_prob[j] * carcrash_shks_prob[k]

                ### Make new shock matrix
                tran_shks_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))
                perm_shks_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))
                carcrash_shks_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))

                for i in range(len(perm_shks_prob)):
                    for j in range(len(tran_shks_prob)):
                        for k in range(len(carcrash_shks_prob)):
                            tran_shks_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = tran_shks_raw[j]
                            perm_shks_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = perm_shks_raw[i]
                            carcrash_shks_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = carcrash_shks_raw[k]

                NewBornDist = jump_to_grid_dur(np.ones_like(tran_shks_new) * self.nNrmInitMean, np.ones_like(tran_shks_new) * self.mNrmInitMean, shk_prbs_new, dist_nGrid, dist_mGrid)
                # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks_new), np.ones_like(tran_shks_new), shk_prbs_new, dist_nGrid, dist_mGrid) # no initial durable stock

                ### Generate transition matrix (dense) then store as sparse.
                # Use the CarCrash generator with the expanded (perm x tran x
                # carcrash) shock grid so the steady-state transition is
                # consistent with the finite-horizon (Jacobian) transition.
                TranMatrix = gen_tran_matrix_dur_CarCrash(
                    dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
                    shk_prbs_new, perm_shks_new, tran_shks_new, carcrash_shks_new,
                    LivPrb, NewBornDist,
                )
                self.tran_matrix = sp.sparse.csr_matrix(TranMatrix.astype(np.float64))

            else:
                # New borns have this distribution (assumes start with no assets, no durable stock and permanent income=1)
                # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), np.zeros_like(tran_shks), shk_prbs, dist_nGrid, dist_mGrid)
                NewBornDist = jump_to_grid_dur(np.ones_like(tran_shks) * self.nNrmInitMean, np.ones_like(tran_shks) * self.mNrmInitMean, shk_prbs, dist_nGrid, dist_mGrid)
                # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), np.ones_like(tran_shks), shk_prbs, dist_nGrid, dist_mGrid) # no initial durable stock

                # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), np.zeros_like(tran_shks), shk_prbs, dist_nGrid, dist_mGrid)

                # # Generate Transition Matrix
                # tran_matrix = gen_tran_matrix_dur(dist_nGrid, dist_mGrid, bNext, nNext_unshocked, shk_prbs, perm_shks, tran_shks, LivPrb,
                #                     NewBornDist)

                ### Generate Sparse matrix
                rows, cols, data, nStates = gen_tran_matrix_dur_sparse(
                    dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
                    shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist
                )

                # Convert numba lists -> numpy arrays
                rows = np.array(rows, dtype=np.int32)
                cols = np.array(cols, dtype=np.int32)
                data = np.array(data, dtype=np.float64)  # float64: differencing in Jacobians is noise-sensitive

                self.tran_matrix = sp.sparse.csr_matrix((data, (rows, cols)), shape=(nStates, nStates))

        elif self.cycles > 1:
            print('calc_transition_matrix requires cycles = 0 or cycles = 1')

        elif self.T_cycle != 0: # finite horizon problem

            if not hasattr(shk_dstn, 'pmv'):
                shk_dstn = self.IncShkDstn

            # Get Lists of Policy Grids. They depend on the discrete choice 
            self.cPol_Grid = []  # List of non-durable consumption policy grids for each period in T_cycle
            self.dPol_Grid = [] # List of durable consumption policy grids for each period in T_cycle
            self.aPol_Grid = []  # List of asset policy grids for each period in T_cycle
            self.adjPol_Grid = [] # List of adjusting policy grids for each period in T_cycle
            self.dexPol_Grid = [] # List of durable expenditure policy grid for each period in T_cycle
            self.texPol_Grid = [] # List of total expenditure policy grid for each period in T_cycle
            self.tran_matrix = []  # List of transition matrices

            dist_mGrid = self.dist_mGrid # Grid of market resources

            for k in range(self.T_cycle):

                # If grid is time varying
                if type(self.dist_nGrid) == list:
                    dist_nGrid = self.dist_nGrid[k]  # Durable stock grid this period
                else:
                    dist_nGrid = self.dist_nGrid  # If here then use prespecified durable stock grid

                ### Baseline
                Pol_shape = (len(dist_nGrid), len(dist_mGrid))
                CNow = np.zeros(Pol_shape)
                DNow = np.zeros(Pol_shape)
                AdjNow = np.zeros(Pol_shape)
                DexNow = np.zeros(Pol_shape)
                aNext = np.zeros(Pol_shape)

                for i_n in range(len(dist_nGrid)):
                    n_aux = np.ones(len(dist_mGrid)) * dist_nGrid[i_n]

                    CNow[i_n] = self.solution[k].cFunc(n_aux, dist_mGrid)  # Steady State Consumption Policy Grid
                    DNow[i_n] = self.solution[k].dFunc(n_aux, dist_mGrid)
                    DexNow[i_n] = self.solution[k].dFunc(n_aux, dist_mGrid) - n_aux

                    AdjNow[i_n] = self.solution[k].adjusting(n_aux, dist_mGrid)
                    ### bound by 0 and 1
                    AdjNow[i_n] = np.clip(AdjNow[i_n], 0, 1)

                    aNext[i_n] = self.solution[k].aFunc(n_aux, dist_mGrid)


                TexNow = CNow + DexNow

                ### Add policies to lists
                self.cPol_Grid.append(CNow) 
                self.dPol_Grid.append(DNow)
                self.adjPol_Grid.append(AdjNow) 
                self.aPol_Grid.append(aNext) 
                self.dexPol_Grid.append(DexNow) 
                self.texPol_Grid.append(TexNow) 
                
                if type(self.Rfree) == list:
                    R_vec = np.where(aNext < 0, self.Rfree[k] + self.RSpread[k], self.Rfree[k])
                    bNext = (R_vec * aNext) - DNow * (self.OpCost + self.dDepr * self.MaintCost)
                else:
                    R_vec = np.where(aNext < 0, self.Rfree + self.RSpread, self.Rfree)
                    bNext = (R_vec * aNext) - DNow * (self.OpCost + self.dDepr * self.MaintCost)

                nNext_unshocked = (1 - self.dDepr * (1 - self.MaintCost)) * DNow
                
                # Obtain shocks and shock probabilities from income distribution this period
                shk_prbs = shk_dstn[k].pmv #shk_dstn[k].pmf  # Probability of shocks this period
                tran_shks = shk_dstn[k].atoms[1] #shk_dstn[k].X[1]  # Transitory shocks this period
                perm_shks = shk_dstn[k].atoms[0] #shk_dstn[k].X[0]  # Permanent shocks this period
                LivPrb = self.LivPrb[k]  # Update probability of staying alive this period

                if self.CarCrashBool == True:
                    ### Adjust shock probability to account for CarCrash Shocks
                    ### perm shock x tran shock x car crash shock
                    tran_shks_prob = np.zeros(self.TranShkCount + 1)
                    tran_shks_prob[0] = self.UnempPrb[k]
                    tran_shks_prob[1:] = (1 - self.UnempPrb[k])/self.TranShkCount
                    perm_shks_prob = np.ones(self.PermShkCount)/self.PermShkCount
                    carcrash_shks_prob = np.array([self.CarCrashPrb, 1 - self.CarCrashPrb])

                    carcrash_shks_raw = np.array([self.CarCrashMin, 1])
                    tran_shks_raw = np.unique(tran_shks)
                    perm_shks_raw = np.unique(perm_shks)

                    ### Make new shock probabilities
                    shk_prbs_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))
                    for i in range(len(perm_shks_prob)):
                        for j in range(len(tran_shks_prob)):
                            for k in range(len(carcrash_shks_prob)):
                                shk_prbs_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = perm_shks_prob[i] * tran_shks_prob[j] * carcrash_shks_prob[k]

                    ### Make new shock matrix
                    tran_shks_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))
                    perm_shks_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))
                    carcrash_shks_new = np.zeros((len(perm_shks_prob) * len(tran_shks_prob)* len(carcrash_shks_prob)))

                    for i in range(len(perm_shks_prob)):
                        for j in range(len(tran_shks_prob)):
                            for k in range(len(carcrash_shks_prob)):
                                tran_shks_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = tran_shks_raw[j]
                                perm_shks_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = perm_shks_raw[i]
                                carcrash_shks_new[i * len(tran_shks_prob) * len(carcrash_shks_prob) + j * len(carcrash_shks_prob) + k] = carcrash_shks_raw[k]

                    NewBornDist = jump_to_grid_dur(np.ones_like(tran_shks_new) * self.nNrmInitMean, np.ones_like(tran_shks_new) * self.mNrmInitMean, shk_prbs_new, dist_nGrid, dist_mGrid)
                    # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks_new), np.ones_like(tran_shks_new), shk_prbs_new, dist_nGrid, dist_mGrid) # no initial durable stock
                    # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks_new), np.zeros_like(tran_shks_new), shk_prbs_new, dist_nGrid, dist_mGrid)

                    # # Generate Transition Matrix
                    # TranMatrix = gen_tran_matrix_dur_CarCrash(dist_nGrid, dist_mGrid, bNext, nNext_unshocked, shk_prbs_new, perm_shks_new, tran_shks_new, carcrash_shks_new, LivPrb,
                    #                     NewBornDist)

                    ### Generate transition matrix (dense) then store as sparse.
                    # gen_tran_matrix_dur_CarCrash returns a DENSE (nStates x nStates)
                    # array, so we wrap it directly rather than re-assembling from
                    # (rows, cols, data) — those are not produced on this branch.
                    TranMatrix = gen_tran_matrix_dur_CarCrash(
                        dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
                        shk_prbs_new, perm_shks_new, tran_shks_new, carcrash_shks_new,
                        LivPrb, NewBornDist,
                    )
                    TranMatrix = sp.sparse.csr_matrix(TranMatrix.astype(np.float64))

                else:
                    # New borns have this distribution (assumes start with no assets, no durable stock and permanent income=1)
                    # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), np.zeros_like(tran_shks), shk_prbs, dist_nGrid, dist_mGrid)
                    NewBornDist = jump_to_grid_dur(np.ones_like(tran_shks) * self.nNrmInitMean, np.ones_like(tran_shks) * self.mNrmInitMean, shk_prbs, dist_nGrid, dist_mGrid)
                    # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), np.ones_like(tran_shks), shk_prbs, dist_nGrid, dist_mGrid) # no initial durable stock
                    # NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), np.zeros_like(tran_shks), shk_prbs, dist_nGrid, dist_mGrid)

                    # # Generate Transition Matrix
                    # TranMatrix = gen_tran_matrix_dur(dist_nGrid, dist_mGrid, bNext, nNext_unshocked, shk_prbs,
                    #                                     perm_shks, tran_shks, LivPrb,
                    #                                     NewBornDist)

                    ### Generate sparse transition matrix:
                    rows, cols, data, nStates = gen_tran_matrix_dur_sparse(
                        dist_nGrid, dist_mGrid, bNext, nNext_unshocked,
                        shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist
                    )

                    # Convert numba lists -> numpy arrays
                    rows = np.array(rows, dtype=np.int32)
                    cols = np.array(cols, dtype=np.int32)
                    data = np.array(data, dtype=np.float64)  # float64: differencing in Jacobians is noise-sensitive

                    TranMatrix = sp.sparse.csr_matrix((data, (rows, cols)), shape=(nStates, nStates))
                
                
                self.tran_matrix.append(TranMatrix)
                # TranMatrix_sparse = sp.sparse.csr_matrix(TranMatrix)
                # self.tran_matrix.append(TranMatrix_sparse)


    ### Old Version of transition_matrix: END
    # 4. Calculate ergodic distribution: calc_ergodic_dist
    def calc_ergodic_dist(self, transition_matrix=None, method=None):

        '''
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.

        Parameters
        ----------
        transition_matrix: List
                    transition matrix whose ergordic distribution is to be solved
        Returns
        -------
        None
        '''

        if not isinstance(transition_matrix, list):
            transition_matrix = [self.tran_matrix]

        # ergodic_distr = compute_erg_dstn(transition_matrix[0]) # This Version is much faster

        ### Sparse Version
        ergodic_distr = compute_erg_dstn_sparse(transition_matrix[0])

        # # ergodic_distr = compute_erg_dstn(transition_matrix[0], method)
        # if method == 'Pontus':
        #     ergodic_distr = compute_erg_dstn(transition_matrix[0])
        # else:
        #     eigen, ergodic_distr = sp.sparse.linalg.eigs(transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1,
        #                                       which='LM')  # Solve for ergodic distribution
        #     ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)



        self.vec_erg_dstn = ergodic_distr  # distribution as a vector
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.dist_nGrid), len(self.dist_mGrid)))  # distribution reshaped into len(ngrid) by len(mgrid) array

    def compute_steady_state(self):
        # Compute steady state to perturb around
        # Only need to solve if we have not already solved it or if the cycles are greater than 0
        if not hasattr(self, 'solution') or self.cycles>0:
            self.cycles = 0
            self.update()
            self.solve()
        # Add Limit
        # if self.dDepr <1:
        #     self.add_limit()

        # else:
        #     solution_shape = (len(self.nNrmGrid) , len(self.mNrmGrid))
        #     cFunc_array = np.zeros(solution_shape)
        #     dFunc_array = np.zeros(solution_shape)
        #     exFunc_array = np.zeros(solution_shape)
        #     for i_n in range(len(self.nNrmGrid)):
        #         n_aux = np.ones_like(self.mNrmGrid) * self.nNrmGrid[i_n]
        #         cFunc_array[i_n] = self.solution[0].cFunc(n_aux, self.mNrmGrid)
        #         dFunc_array[i_n] = self.solution[0].dFunc(n_aux, self.mNrmGrid)
        #         exFunc_array[i_n] = self.solution[0].exFunc(n_aux, self.mNrmGrid)
        #     self.solution[0].cFunc = LinearFast(cFunc_array, [self.nNrmGrid, self.mNrmGrid])
        #     self.solution[0].dFunc = LinearFast(dFunc_array, [self.nNrmGrid, self.mNrmGrid])
        #     self.solution[0].exFunc = LinearFast(exFunc_array, [self.nNrmGrid, self.mNrmGrid])
        #     self.add_limit()

        # Use Harmenberg Measure
        self.neutral_measure = True
        self.update_income_process_truncated()

        # Non stochastic simuation
        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur()

        self.c_ss = self.cPol_Grid  # Normalized Consumption Policy grid
        self.d_ss = self.dPol_Grid  # Normalized Durable Policy grid
        self.dex_ss = self.dexPol_Grid # Normalized Durable Expenditure Policy grid
        self.a_ss = self.aPol_Grid  # Normalized Asset Policy grid
        self.adj_ss = self.adjPol_Grid

        self.calc_ergodic_dist()  # Calculate ergodic distribution
        # Steady State Distribution as a vector (m*p x 1) where m is the number of gridpoints on the market resources grid
        ss_dstn = self.vec_erg_dstn

        # Need to flatten policy grids. Sparse Version
        self.A_ss = self.a_ss.flatten() @ ss_dstn
        self.C_ss = self.c_ss.flatten() @ ss_dstn
        self.D_ss = self.d_ss.flatten() @ ss_dstn
        self.DEX_ss = self.dex_ss.flatten() @ ss_dstn
        self.TEX_ss = self.C_ss + self.DEX_ss
        self.ADJ_ss = self.adj_ss.flatten() @ ss_dstn

        return self.A_ss, self.C_ss, self.D_ss, self.DEX_ss, self.TEX_ss, self.ADJ_ss

    def compute_steady_state_level(self, N_pGrid=50):
        """
        Compute steady-state cross-sectional distributions of LEVEL variables
        (physical units, not normalized by permanent income P).

        Uses the *independence approximation*: the cross-sectional distribution
        of normalized states (nNrm, mNrm) and permanent income P are treated as
        independent.  The approximation error is small when expected lifetime is
        long (high LivPrb) because the contemporaneous ψ shock is only one of
        ~1/(1−LivPrb) accumulated shocks in log(P), keeping the cross-sectional
        correlation between P and normalized states around −σ_ψ / σ_{log P} ≈ −0.1.

        Level value = normalized_value × P  for monetary variables.
        ADJ is scale-invariant (it is a 0/1 indicator), so its percentile array
        is identical to the normalized case.

        Calls compute_steady_state() automatically if it has not yet been run.
        The first call computes one extra transition matrix (neutral_measure=False)
        to obtain the unweighted ergodic distribution; subsequent calls reuse the
        cached result stored in self.vec_erg_dstn_unweighted.

        Parameters
        ----------
        N_pGrid : int
            Number of quadrature points for the P distribution (default 50).

        Returns
        -------
        dict
            Keys: 'A', 'C', 'D', 'DEX', 'TEX', 'ADJ', 'N', 'M'
            Values: ndarray of shape (101,) — percentiles p0 through p100 in
            LEVEL (un-normalized) units.

        Also stores each array as self.{key}_ss_pctile (e.g. self.A_ss_pctile).
        """
        if not hasattr(self, 'vec_erg_dstn'):
            self.compute_steady_state()

        # ------------------------------------------------------------------
        # Step 1: unweighted normalized ergodic distribution
        #
        # compute_steady_state() uses neutral_measure=True (Harmenberg P-weighted
        # distribution), which is correct for means but not for the independence
        # approximation.  We need the UNWEIGHTED cross-sectional distribution.
        # We recompute the transition matrix once with neutral_measure=False,
        # cache the result, then restore the neutral-measure state.
        # ------------------------------------------------------------------
        if not hasattr(self, 'vec_erg_dstn_unweighted'):
            saved_tran_matrix = self.tran_matrix   # sparse CSR (cycles=0 case)

            self.neutral_measure = False
            self.update_income_process_truncated()
            self.calc_transition_matrix_dur()
            self.calc_ergodic_dist()
            self.vec_erg_dstn_unweighted = self.vec_erg_dstn.copy()

            # Restore: put back the neutral-measure TM (already computed),
            # then re-run the cheap eigenvector step to restore vec_erg_dstn.
            self.neutral_measure = True
            self.update_income_process_truncated()
            self.tran_matrix = saved_tran_matrix
            self.calc_ergodic_dist()

        ss_norm = self.vec_erg_dstn_unweighted   # (N_n * N_m,)

        # ------------------------------------------------------------------
        # Step 2: ergodic distribution of permanent income P
        #
        # An agent of age a has log(P_a) = Σ_{s=1}^{a} log(ψ_s).
        # Age ~ Geometric(1 − LivPrb).  By Wald's identities:
        #   E[log P]   = E[log ψ] / (1 − LivPrb)
        #   Var[log P] = Var[log ψ] / (1 − LivPrb)
        #              + LivPrb * (E[log ψ])^2 / (1 − LivPrb)^2
        # For a mean-1 lognormal shock: E[log ψ] = −σ_ψ²/2, Var[log ψ] = σ_ψ².
        # ------------------------------------------------------------------
        sigma_psi  = float(self.PermShkStd[0])
        death_rate = 1.0 - float(self.LivPrb[0])

        if sigma_psi < 1e-10 or death_rate < 1e-10:
            p_nodes   = np.array([1.0])
            p_weights = np.array([1.0])
        else:
            mu_log_psi  = -sigma_psi**2 / 2.0
            E_log_P     =  mu_log_psi / death_rate
            Var_log_P   = (sigma_psi**2 / death_rate
                           + float(self.LivPrb[0]) * mu_log_psi**2 / death_rate**2)
            sigma_log_P = np.sqrt(Var_log_P)

            log_p_nodes = np.linspace(E_log_P - 4.0 * sigma_log_P,
                                      E_log_P + 4.0 * sigma_log_P, N_pGrid)
            p_nodes = np.exp(log_p_nodes)

            # Gaussian PDF in log-space; normalize to sum-to-1 weights
            log_p_pdf = np.exp(-0.5 * ((log_p_nodes - E_log_P) / sigma_log_P)**2)
            p_weights = log_p_pdf / log_p_pdf.sum()

            # Rescale nodes so that E[P] = Σ w_l * P_l = 1 exactly
            # (corrects minor discretization / CLT approximation error)
            p_nodes = p_nodes / (p_weights @ p_nodes)

        # ------------------------------------------------------------------
        # Step 3: level percentiles via outer product (independence approx.)
        #
        # Joint mass at (normalized state k, P grid point l):
        #   w_{k,l} = ss_norm[k] * p_weights[l]
        # Level value: x_level = normalized_val[k] * p_nodes[l]
        # ------------------------------------------------------------------
        q_levels = np.arange(0, 101, dtype=float)

        def _level_wquantile(policy_2d):
            """Percentiles of level = policy_2d × P under independence."""
            norm_vals  = policy_2d.flatten()
            level_vals = np.outer(norm_vals, p_nodes).ravel()
            joint_mass = np.outer(ss_norm, p_weights).ravel()
            idx        = np.argsort(level_vals)
            cdf        = np.clip(np.cumsum(joint_mass[idx]), 0.0, 1.0)
            return np.interp(q_levels / 100.0, cdf, level_vals[idx])

        def _wquantile(policy_2d):
            """Percentiles of a scale-invariant variable (no P scaling)."""
            vals = policy_2d.flatten()
            idx  = np.argsort(vals)
            cdf  = np.clip(np.cumsum(ss_norm[idx]), 0.0, 1.0)
            return np.interp(q_levels / 100.0, cdf, vals[idx])

        n_grid_2d, m_grid_2d = np.meshgrid(
            self.dist_nGrid, self.dist_mGrid, indexing='ij'
        )

        pctile = {
            "A":   _level_wquantile(self.a_ss),
            "C":   _level_wquantile(self.c_ss),
            "D":   _level_wquantile(self.d_ss),
            "DEX": _level_wquantile(self.dex_ss),
            "TEX": _level_wquantile(self.c_ss + self.dex_ss),
            "ADJ": _wquantile(self.adj_ss),      # scale-invariant: 0/1 indicator
            "N":   _level_wquantile(n_grid_2d),
            "M":   _level_wquantile(m_grid_2d),
        }

        for key, arr in pctile.items():
            setattr(self, f"{key}_ss_pctile", arr)

        return pctile

    def calc_jacobian(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Non-Durable Consumption with respect to shk_param

        DEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param       

        TEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Stock with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        ADJJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Adjusting Probability with respect to shk_param
        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon
        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = UnempRisk_Durable_Latest(**params)
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state

        # # delete Rfree from time invariant list since it varies overtime
        # FinHorizonAgent.del_from_time_inv("Rfree")
        # # Add Rfree to time varying list to be able to introduce time varying interest rates
        # FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 1e-4 #0.001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        FinHorizonAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        FinHorizonAgent.solve()

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        # FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        a_t = FinHorizonAgent.aPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # NOTE: the steady-state policy grids are appended to the finite-horizon
        # policy lists once, in the block below.  Do NOT append here as well:
        # c_t/a_t/... are references to the agent's own lists, so appending twice
        # would place the SS grid at both index T and T+1.

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Stock Policy
        dex_ss = self.dexPol_Grid  # steady state Durable Expenditure Policy
        adj_ss = self.adjPol_Grid  # steady state Adjusting Policy

        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        adj_t.append(self.adj_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        dd0_s = []  # Deviation of Durable Stock policy from steady state policy
        ddex0_s = []  # Deviation of Durable Expenditure policy from steady state policy
        dadj0_s = []  # Deviation of Adjusting policy from steady state policy

        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)
            dd0_s.append(d_t[T - i] - d_ss)
            ddex0_s.append(dex_t[T - i] - dex_ss)
            dadj0_s.append(adj_t[T - i] - adj_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)
        dd0_s = np.array(dd0_s)
        ddex0_s = np.array(ddex0_s)
        dadj0_s = np.array(dadj0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        Dist_ss = self.vec_erg_dstn.T#[0] # Shape: ((nxm) x 1)
        dA0_s = []
        dC0_s = []
        dD0_s = []
        dDEX0_s = []
        dADJ0_s = []

        ### Sparse Version
        for i in range(T):
            dA0_s.append(da0_s[i].flatten() @ Dist_ss)
            dC0_s.append(dc0_s[i].flatten() @ Dist_ss)
            dD0_s.append(dd0_s[i].flatten() @ Dist_ss)
            dDEX0_s.append(ddex0_s[i].flatten() @ Dist_ss)
            dADJ0_s.append(dadj0_s[i].flatten() @ Dist_ss)

        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        dA0_s = np.array(dA0_s)
        dC0_s = np.array(dC0_s)
        dD0_s = np.array(dD0_s)
        dDEX0_s = np.array(dDEX0_s)
        dADJ0_s = np.array(dADJ0_s)

        A_curl_s = dA0_s / dx # Shape (T,1)
        C_curl_s = dC0_s / dx
        D_curl_s = dD0_s / dx
        DEX_curl_s = dDEX0_s / dx
        ADJ_curl_s = dADJ0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        ### Does not make a huge array, but keeps it for each T
        dDist0_s_shape = (T, len(self.dist_nGrid) * len(self.dist_mGrid)) # Shape: T x (nxm)
        dDist0_s = np.zeros(dDist0_s_shape)

        ### Sparse Version
        for i in range(T):
            dlambda0_s_i = tranmat_t[T - i] - tranmat_ss   # sparse - sparse = sparse
            dDist0_s[i] = dlambda0_s_i @ Dist_ss           # sparse × dense = dense

        Dist_curl_s = dDist0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors: Shape T x n x m
        exp_vecs_a = []
        exp_vecs_c = []
        exp_vecs_d = []
        exp_vecs_dex = []
        exp_vecs_adj = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss.flatten()
        exp_vec_c = c_ss.flatten()
        exp_vec_d = d_ss.flatten()
        exp_vec_dex = dex_ss.flatten()
        exp_vec_adj = adj_ss.flatten()

        ### Sparse Version
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = exp_vec_a.flatten() @ tranmat_ss    # sparse matmul

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = exp_vec_c.flatten() @ tranmat_ss

            exp_vecs_d.append(exp_vec_d)
            exp_vec_d = exp_vec_d.flatten() @ tranmat_ss

            exp_vecs_dex.append(exp_vec_dex)
            exp_vec_dex = exp_vec_dex.flatten() @ tranmat_ss

            exp_vecs_adj.append(exp_vec_adj)
            exp_vec_adj = exp_vec_adj.flatten() @ tranmat_ss

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)
        exp_vecs_d = np.array(exp_vecs_d)
        exp_vecs_dex = np.array(exp_vecs_dex)
        exp_vecs_adj = np.array(exp_vecs_adj)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption
        Curl_F_D = np.zeros((T, T))  # Fake news matrix for durable stock
        Curl_F_DEX = np.zeros((T, T))  # Fake news matrix for durable expenditure
        Curl_F_ADJ = np.zeros((T, T))  # Fake news matrix for durable expenditure

        # First row of Fake News Matrix: Shaper ((n x m) x 1)
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i].flatten(), Dist_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i].flatten(), Dist_curl_s[j])
                Curl_F_D[i + 1][j] = np.dot(exp_vecs_d[i].flatten(), Dist_curl_s[j])
                Curl_F_DEX[i + 1][j] = np.dot(exp_vecs_dex[i].flatten(), Dist_curl_s[j])
                Curl_F_ADJ[i + 1][j] = np.dot(exp_vecs_adj[i].flatten(), Dist_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########

        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]


        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update()
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        ZerothColAgent.solve()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix

        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T#[0]

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        ### Sparse Version
        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    def calc_jacobian_Baseline(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Non-Durable Consumption with respect to shk_param

        DEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param       

        TEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Stock with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        ADJJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Adjusting Probability with respect to shk_param
        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon
        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = UnempRisk_Durable_Latest(**params)
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state

        # # delete Rfree from time invariant list since it varies overtime
        # FinHorizonAgent.del_from_time_inv("Rfree")
        # # Add Rfree to time varying list to be able to introduce time varying interest rates
        # FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 1e-4 #0.001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        FinHorizonAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        FinHorizonAgent.solve()

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        # FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        a_t = FinHorizonAgent.aPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # Append steady state policy grid into list of policy grids as HARK does not provide the initial policy
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        a_t.append(self.a_ss)
        adj_t.append(self.adj_ss)

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Stock Policy
        dex_ss = self.dexPol_Grid  # steady state Durable Expenditure Policy
        adj_ss = self.adjPol_Grid  # steady state Adjusting Policy

        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        adj_t.append(self.adj_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        dd0_s = []  # Deviation of Durable Stock policy from steady state policy
        ddex0_s = []  # Deviation of Durable Expenditure policy from steady state policy
        dadj0_s = []  # Deviation of Adjusting policy from steady state policy

        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)
            dd0_s.append(d_t[T - i] - d_ss)
            ddex0_s.append(dex_t[T - i] - dex_ss)
            dadj0_s.append(adj_t[T - i] - adj_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)
        dd0_s = np.array(dd0_s)
        ddex0_s = np.array(ddex0_s)
        dadj0_s = np.array(dadj0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        Dist_ss = self.vec_erg_dstn.T#[0] # Shape: ((nxm) x 1)
        dA0_s = []
        dC0_s = []
        dD0_s = []
        dDEX0_s = []
        dADJ0_s = []

        ### Sparse Version
        for i in range(T):
            dA0_s.append(da0_s[i].flatten() @ Dist_ss)
            dC0_s.append(dc0_s[i].flatten() @ Dist_ss)
            dD0_s.append(dd0_s[i].flatten() @ Dist_ss)
            dDEX0_s.append(ddex0_s[i].flatten() @ Dist_ss)
            dADJ0_s.append(dadj0_s[i].flatten() @ Dist_ss)

        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        dA0_s = np.array(dA0_s)
        dC0_s = np.array(dC0_s)
        dD0_s = np.array(dD0_s)
        dDEX0_s = np.array(dDEX0_s)
        dADJ0_s = np.array(dADJ0_s)

        A_curl_s = dA0_s / dx # Shape (T,1)
        C_curl_s = dC0_s / dx
        D_curl_s = dD0_s / dx
        DEX_curl_s = dDEX0_s / dx
        ADJ_curl_s = dADJ0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        ### Does not make a huge array, but keeps it for each T
        dDist0_s_shape = (T, len(self.dist_nGrid) * len(self.dist_mGrid)) # Shape: T x (nxm)
        dDist0_s = np.zeros(dDist0_s_shape)

        ### Sparse Version
        for i in range(T):
            dlambda0_s_i = tranmat_t[T - i] - tranmat_ss   # sparse - sparse = sparse
            dDist0_s[i] = dlambda0_s_i @ Dist_ss           # sparse × dense = dense

        Dist_curl_s = dDist0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors: Shape T x n x m
        exp_vecs_a = []
        exp_vecs_c = []
        exp_vecs_d = []
        exp_vecs_dex = []
        exp_vecs_adj = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss.flatten()
        exp_vec_c = c_ss.flatten()
        exp_vec_d = d_ss.flatten()
        exp_vec_dex = dex_ss.flatten()
        exp_vec_adj = adj_ss.flatten()

        ### Sparse Version
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = exp_vec_a.flatten() @ tranmat_ss    # sparse matmul

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = exp_vec_c.flatten() @ tranmat_ss

            exp_vecs_d.append(exp_vec_d)
            exp_vec_d = exp_vec_d.flatten() @ tranmat_ss

            exp_vecs_dex.append(exp_vec_dex)
            exp_vec_dex = exp_vec_dex.flatten() @ tranmat_ss

            exp_vecs_adj.append(exp_vec_adj)
            exp_vec_adj = exp_vec_adj.flatten() @ tranmat_ss

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)
        exp_vecs_d = np.array(exp_vecs_d)
        exp_vecs_dex = np.array(exp_vecs_dex)
        exp_vecs_adj = np.array(exp_vecs_adj)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption
        Curl_F_D = np.zeros((T, T))  # Fake news matrix for durable stock
        Curl_F_DEX = np.zeros((T, T))  # Fake news matrix for durable expenditure
        Curl_F_ADJ = np.zeros((T, T))  # Fake news matrix for durable expenditure

        # First row of Fake News Matrix: Shaper ((n x m) x 1)
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i].flatten(), Dist_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i].flatten(), Dist_curl_s[j])
                Curl_F_D[i + 1][j] = np.dot(exp_vecs_d[i].flatten(), Dist_curl_s[j])
                Curl_F_DEX[i + 1][j] = np.dot(exp_vecs_dex[i].flatten(), Dist_curl_s[j])
                Curl_F_ADJ[i + 1][j] = np.dot(exp_vecs_adj[i].flatten(), Dist_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########
        
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]
        
        
        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update()
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        
        # Solve
        ZerothColAgent.solve()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix

        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T#[0]

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        ### Sparse Version
        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ



    def calc_jacobian_zeroth_column(self, shk_param, T):
        """
        Same as ``calc_jacobian``, but with a coherent zeroth column:
        period 0 aggregates shocked ZerothColAgent policies on Dist_ss;
        t>=1 evolves the distribution then aggregates with SS policies.

        Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Non-Durable Consumption with respect to shk_param

        DEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param       

        TEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Stock with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        ADJJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Adjusting Probability with respect to shk_param
        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon
        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = UnempRisk_Durable_Latest(**params)
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state

        # # delete Rfree from time invariant list since it varies overtime
        # FinHorizonAgent.del_from_time_inv("Rfree")
        # # Add Rfree to time varying list to be able to introduce time varying interest rates
        # FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 1e-4 #0.001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        FinHorizonAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        FinHorizonAgent.solve()

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        # FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        a_t = FinHorizonAgent.aPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # NOTE: the steady-state policy grids are appended to the finite-horizon
        # policy lists once, in the block below.  Do NOT append here as well:
        # c_t/a_t/... are references to the agent's own lists, so appending twice
        # would place the SS grid at both index T and T+1.

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Stock Policy
        dex_ss = self.dexPol_Grid  # steady state Durable Expenditure Policy
        adj_ss = self.adjPol_Grid  # steady state Adjusting Policy

        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        adj_t.append(self.adj_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        dd0_s = []  # Deviation of Durable Stock policy from steady state policy
        ddex0_s = []  # Deviation of Durable Expenditure policy from steady state policy
        dadj0_s = []  # Deviation of Adjusting policy from steady state policy

        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)
            dd0_s.append(d_t[T - i] - d_ss)
            ddex0_s.append(dex_t[T - i] - dex_ss)
            dadj0_s.append(adj_t[T - i] - adj_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)
        dd0_s = np.array(dd0_s)
        ddex0_s = np.array(ddex0_s)
        dadj0_s = np.array(dadj0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        Dist_ss = self.vec_erg_dstn.T#[0] # Shape: ((nxm) x 1)
        dA0_s = []
        dC0_s = []
        dD0_s = []
        dDEX0_s = []
        dADJ0_s = []

        ### Sparse Version
        for i in range(T):
            dA0_s.append(da0_s[i].flatten() @ Dist_ss)
            dC0_s.append(dc0_s[i].flatten() @ Dist_ss)
            dD0_s.append(dd0_s[i].flatten() @ Dist_ss)
            dDEX0_s.append(ddex0_s[i].flatten() @ Dist_ss)
            dADJ0_s.append(dadj0_s[i].flatten() @ Dist_ss)

        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        dA0_s = np.array(dA0_s)
        dC0_s = np.array(dC0_s)
        dD0_s = np.array(dD0_s)
        dDEX0_s = np.array(dDEX0_s)
        dADJ0_s = np.array(dADJ0_s)

        A_curl_s = dA0_s / dx # Shape (T,1)
        C_curl_s = dC0_s / dx
        D_curl_s = dD0_s / dx
        DEX_curl_s = dDEX0_s / dx
        ADJ_curl_s = dADJ0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        ### Does not make a huge array, but keeps it for each T
        dDist0_s_shape = (T, len(self.dist_nGrid) * len(self.dist_mGrid)) # Shape: T x (nxm)
        dDist0_s = np.zeros(dDist0_s_shape)

        ### Sparse Version
        for i in range(T):
            dlambda0_s_i = tranmat_t[T - i] - tranmat_ss   # sparse - sparse = sparse
            dDist0_s[i] = dlambda0_s_i @ Dist_ss           # sparse × dense = dense

        Dist_curl_s = dDist0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors: Shape T x n x m
        exp_vecs_a = []
        exp_vecs_c = []
        exp_vecs_d = []
        exp_vecs_dex = []
        exp_vecs_adj = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss.flatten()
        exp_vec_c = c_ss.flatten()
        exp_vec_d = d_ss.flatten()
        exp_vec_dex = dex_ss.flatten()
        exp_vec_adj = adj_ss.flatten()

        ### Sparse Version
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = exp_vec_a.flatten() @ tranmat_ss    # sparse matmul

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = exp_vec_c.flatten() @ tranmat_ss

            exp_vecs_d.append(exp_vec_d)
            exp_vec_d = exp_vec_d.flatten() @ tranmat_ss

            exp_vecs_dex.append(exp_vec_dex)
            exp_vec_dex = exp_vec_dex.flatten() @ tranmat_ss

            exp_vecs_adj.append(exp_vec_adj)
            exp_vec_adj = exp_vec_adj.flatten() @ tranmat_ss

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)
        exp_vecs_d = np.array(exp_vecs_d)
        exp_vecs_dex = np.array(exp_vecs_dex)
        exp_vecs_adj = np.array(exp_vecs_adj)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption
        Curl_F_D = np.zeros((T, T))  # Fake news matrix for durable stock
        Curl_F_DEX = np.zeros((T, T))  # Fake news matrix for durable expenditure
        Curl_F_ADJ = np.zeros((T, T))  # Fake news matrix for durable expenditure

        # First row of Fake News Matrix: Shaper ((n x m) x 1)
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i].flatten(), Dist_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i].flatten(), Dist_curl_s[j])
                Curl_F_D[i + 1][j] = np.dot(exp_vecs_d[i].flatten(), Dist_curl_s[j])
                Curl_F_DEX[i + 1][j] = np.dot(exp_vecs_dex[i].flatten(), Dist_curl_s[j])
                Curl_F_ADJ[i + 1][j] = np.dot(exp_vecs_adj[i].flatten(), Dist_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########

        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]


        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update()
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        ZerothColAgent.solve()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix

        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        Dist_ss_zc = self.vec_erg_dstn.T  # SS distribution

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        # Period 0: shocked policies @ Dist_ss (coherent stock-flow accounting).
        # t>=1: shocked then SS transitions; aggregate with SS policies.
        A_t_no_sim[0] = ZerothColAgent.aPol_Grid[0].flatten() @ Dist_ss_zc
        C_t_no_sim[0] = ZerothColAgent.cPol_Grid[0].flatten() @ Dist_ss_zc
        D_t_no_sim[0] = ZerothColAgent.dPol_Grid[0].flatten() @ Dist_ss_zc
        DEX_t_no_sim[0] = ZerothColAgent.dexPol_Grid[0].flatten() @ Dist_ss_zc
        ADJ_t_no_sim[0] = ZerothColAgent.adjPol_Grid[0].flatten() @ Dist_ss_zc

        dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ Dist_ss_zc
        for i in range(1, T):
            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col
            dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    def calc_jacobian_real(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Only uses ex-post realizations of the shocks without any anticipation.Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Non-Durable Consumption with respect to shk_param

        DEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param       

        TEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Stock with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        ADJJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Adjusting Probability with respect to shk_param
        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon
        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = UnempRisk_Durable_Latest(**params)
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state

        # # delete Rfree from time invariant list since it varies overtime
        # FinHorizonAgent.del_from_time_inv("Rfree")
        # # Add Rfree to time varying list to be able to introduce time varying interest rates
        # FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 1e-4 #0.001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        FinHorizonAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        FinHorizonAgent.solve()

        # Replace solution with SS.
        for t in range(T):
            FinHorizonAgent.solution[t] = deepcopy(self.solution[0])

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        # FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        a_t = FinHorizonAgent.aPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # NOTE: the steady-state policy grids are appended to the finite-horizon
        # policy lists once, in the block below.  Do NOT append here as well:
        # c_t/a_t/... are references to the agent's own lists, so appending twice
        # would place the SS grid at both index T and T+1.

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Stock Policy
        dex_ss = self.dexPol_Grid  # steady state Durable Expenditure Policy
        adj_ss = self.adjPol_Grid  # steady state Adjusting Policy

        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        adj_t.append(self.adj_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        dd0_s = []  # Deviation of Durable Stock policy from steady state policy
        ddex0_s = []  # Deviation of Durable Expenditure policy from steady state policy
        dadj0_s = []  # Deviation of Adjusting policy from steady state policy

        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)
            dd0_s.append(d_t[T - i] - d_ss)
            ddex0_s.append(dex_t[T - i] - dex_ss)
            dadj0_s.append(adj_t[T - i] - adj_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)
        dd0_s = np.array(dd0_s)
        ddex0_s = np.array(ddex0_s)
        dadj0_s = np.array(dadj0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        Dist_ss = self.vec_erg_dstn.T#[0] # Shape: ((nxm) x 1)
        dA0_s = []
        dC0_s = []
        dD0_s = []
        dDEX0_s = []
        dADJ0_s = []

        ### Sparse Version
        for i in range(T):
            dA0_s.append(da0_s[i].flatten() @ Dist_ss)
            dC0_s.append(dc0_s[i].flatten() @ Dist_ss)
            dD0_s.append(dd0_s[i].flatten() @ Dist_ss)
            dDEX0_s.append(ddex0_s[i].flatten() @ Dist_ss)
            dADJ0_s.append(dadj0_s[i].flatten() @ Dist_ss)

        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        dA0_s = np.array(dA0_s)
        dC0_s = np.array(dC0_s)
        dD0_s = np.array(dD0_s)
        dDEX0_s = np.array(dDEX0_s)
        dADJ0_s = np.array(dADJ0_s)

        A_curl_s = dA0_s / dx # Shape (T,1)
        C_curl_s = dC0_s / dx
        D_curl_s = dD0_s / dx
        DEX_curl_s = dDEX0_s / dx
        ADJ_curl_s = dADJ0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        ### Does not make a huge array, but keeps it for each T
        dDist0_s_shape = (T, len(self.dist_nGrid) * len(self.dist_mGrid)) # Shape: T x (nxm)
        dDist0_s = np.zeros(dDist0_s_shape)

        ### Sparse Version
        for i in range(T):
            dlambda0_s_i = tranmat_t[T - i] - tranmat_ss   # sparse - sparse = sparse
            dDist0_s[i] = dlambda0_s_i @ Dist_ss           # sparse × dense = dense

        Dist_curl_s = dDist0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors: Shape T x n x m
        exp_vecs_a = []
        exp_vecs_c = []
        exp_vecs_d = []
        exp_vecs_dex = []
        exp_vecs_adj = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss.flatten()
        exp_vec_c = c_ss.flatten()
        exp_vec_d = d_ss.flatten()
        exp_vec_dex = dex_ss.flatten()
        exp_vec_adj = adj_ss.flatten()

        ### Sparse Version
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = exp_vec_a.flatten() @ tranmat_ss    # sparse matmul

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = exp_vec_c.flatten() @ tranmat_ss

            exp_vecs_d.append(exp_vec_d)
            exp_vec_d = exp_vec_d.flatten() @ tranmat_ss

            exp_vecs_dex.append(exp_vec_dex)
            exp_vec_dex = exp_vec_dex.flatten() @ tranmat_ss

            exp_vecs_adj.append(exp_vec_adj)
            exp_vec_adj = exp_vec_adj.flatten() @ tranmat_ss

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)
        exp_vecs_d = np.array(exp_vecs_d)
        exp_vecs_dex = np.array(exp_vecs_dex)
        exp_vecs_adj = np.array(exp_vecs_adj)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption
        Curl_F_D = np.zeros((T, T))  # Fake news matrix for durable stock
        Curl_F_DEX = np.zeros((T, T))  # Fake news matrix for durable expenditure
        Curl_F_ADJ = np.zeros((T, T))  # Fake news matrix for durable expenditure

        # First row of Fake News Matrix: Shaper ((n x m) x 1)
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i].flatten(), Dist_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i].flatten(), Dist_curl_s[j])
                Curl_F_D[i + 1][j] = np.dot(exp_vecs_d[i].flatten(), Dist_curl_s[j])
                Curl_F_DEX[i + 1][j] = np.dot(exp_vecs_dex[i].flatten(), Dist_curl_s[j])
                Curl_F_ADJ[i + 1][j] = np.dot(exp_vecs_adj[i].flatten(), Dist_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########

        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]


        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update()
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Solve
        ZerothColAgent.solve()

        # Replace solution with SS.
        for t in range(len(ZerothColAgent.solution)):
            ZerothColAgent.solution[t] = deepcopy(self.solution[0])

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix

        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T#[0]

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        ### Sparse Version
        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    def calc_jacobian_smooth(self, shk_param, T, use_gpu=False):
        """
        Smooth Jacobian with three improvements over calc_jacobian:

        (1) Centered two-sided finite differences with dx=1e-4.
        (2) Fresh SS distribution grid using the model's configured
            TranMatrixmCount / TranMatrixnCount (set these before calling for
            a finer grid).  Policy evaluation skips sparse-matrix construction
            for finite-horizon agents, so scaling the grid is much cheaper.
        (3) Analytical (SSJ-style) curlyD via jump_to_grid_dur_shock instead of
            differencing full transition matrices — the main source of kinks.

        Speed notes
        -----------
        * Two backward solves are run (+dx, -dx) instead of one; with the
          transition-matrix step skipped for finite-horizon agents the total
          cost is roughly 2-3× the original calc_jacobian.
        * For finer grids set ss.TranMatrixmCount / ss.TranMatrixnCount before
          calling.  The method rebuilds the SS distribution on whatever counts
          are currently set on the model.
        * use_gpu=True offloads the expectation-vector loop and dense Curl_F
          matrix fill to an Nvidia GPU via CuPy.  Falls back to CPU silently
          when CuPy is not available.

        Parameters
        ----------
        shk_param : str
        T         : int  Jacobian horizon
        use_gpu   : bool (default False)

        Returns
        -------
        J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ : (T x T) numpy arrays
        """
        # ─── GPU setup (optional) ────────────────────────────────────────────
        _gpu_ok = False
        if use_gpu:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpsp
                _gpu_ok = True
            except ImportError:
                pass  # fall back to CPU silently

        # ─── 0. Re-build SS transition matrix / ergodic dist on fresh grid ───
        # Save only the attributes that will be overwritten so we can restore them.
        _save_attrs = [
            'dist_mGrid', 'dist_nGrid', 'tran_matrix', 'vec_erg_dstn', 'erg_dstn',
            'aPol_Grid', 'cPol_Grid', 'dPol_Grid', 'adjPol_Grid', 'dexPol_Grid', 'texPol_Grid',
            'a_ss', 'c_ss', 'd_ss', 'dex_ss', 'adj_ss',
            'A_ss', 'C_ss', 'D_ss', 'DEX_ss', 'TEX_ss', 'ADJ_ss',
        ]
        saved = {a: getattr(self, a) for a in _save_attrs if hasattr(self, a)}

        # Rebuild distribution grid using whatever TranMatrixmCount/nCount are set.
        # No dist_mGrid / dist_nGrid override — keeps the state space manageable.
        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur()
        self.calc_ergodic_dist()

        self.c_ss   = self.cPol_Grid
        self.d_ss   = self.dPol_Grid
        self.dex_ss = self.dexPol_Grid
        self.a_ss   = self.aPol_Grid
        self.adj_ss = self.adjPol_Grid
        ss_dstn = self.vec_erg_dstn
        self.A_ss   = self.a_ss.flatten()   @ ss_dstn
        self.C_ss   = self.c_ss.flatten()   @ ss_dstn
        self.D_ss   = self.d_ss.flatten()   @ ss_dstn
        self.DEX_ss = self.dex_ss.flatten() @ ss_dstn
        self.TEX_ss = self.C_ss + self.DEX_ss
        self.ADJ_ss = self.adj_ss.flatten() @ ss_dstn

        dist_mGrid = self.dist_mGrid
        dist_nGrid = self.dist_nGrid
        tranmat_ss = self.tran_matrix          # scipy sparse CSR
        Dist_ss    = self.vec_erg_dstn         # 1-D (n_n * n_m,)
        Dist_ss_2d = Dist_ss.reshape(len(dist_nGrid), len(dist_mGrid))
        a_ss   = self.a_ss;  c_ss  = self.c_ss;  d_ss   = self.d_ss
        dex_ss = self.dex_ss; adj_ss = self.adj_ss
        n_n      = len(dist_nGrid)
        n_m      = len(dist_mGrid)
        n_states = n_n * n_m

        # Precompute SS next-period state quantities for analytical curlyD
        R_vec_ss           = np.where(a_ss < 0, self.Rfree[0] + self.RSpread[0], self.Rfree[0])
        bNext_ss           = R_vec_ss * a_ss - d_ss * (self.OpCost + self.dDepr * self.MaintCost)
        nNext_unshocked_ss = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * d_ss
        shk_prbs  = self.IncShkDstn[0].pmv
        tran_shks = self.IncShkDstn[0].atoms[1]
        perm_shks = self.IncShkDstn[0].atoms[0]

        # ─── 1. Solve +dx and -dx finite-horizon agents ───────────────────────
        dx      = 1e-4
        twodx   = 2.0 * dx
        i_shock = T - 1

        params_base = deepcopy(self.__dict__["parameters"])
        params_base["T_cycle"] = T
        params_base["cycles"]  = 1
        for _p in ["Rfree", "RSpread", "DiscFac", "LivPrb", "PermGroFac",
                   "PermShkStd", "TranShkStd", "UnempPrb", "UnempPrbRet",
                   "IncUnemp", "IncUnempRet", "BoroCnstArt"]:
            params_base[_p] = T * [getattr(self, _p)[0]]

        base_val = (getattr(self, shk_param)[0]
                    if isinstance(getattr(self, shk_param), list)
                    else getattr(self, shk_param))

        def _eval_pols_on_grid(agent):
            """Evaluate agent's policy functions on (dist_nGrid x dist_mGrid).
            Skips sparse-matrix construction — only bilinear lookups needed."""
            aPol, cPol, dPol, dexPol, adjPol = [], [], [], [], []
            for k in range(T):
                aNext  = np.zeros((n_n, n_m))
                CNow   = np.zeros((n_n, n_m))
                DNow   = np.zeros((n_n, n_m))
                DexNow = np.zeros((n_n, n_m))
                AdjNow = np.zeros((n_n, n_m))
                for i_n in range(n_n):
                    n_aux      = np.full(n_m, dist_nGrid[i_n])
                    CNow[i_n]   = agent.solution[k].cFunc(n_aux, dist_mGrid)
                    DNow[i_n]   = agent.solution[k].dFunc(n_aux, dist_mGrid)
                    DexNow[i_n] = DNow[i_n] - n_aux
                    AdjNow[i_n] = np.clip(agent.solution[k].adjusting(n_aux, dist_mGrid), 0.0, 1.0)
                    aNext[i_n]  = agent.solution[k].aFunc(n_aux, dist_mGrid)
                aPol.append(aNext);  cPol.append(CNow)
                dPol.append(DNow);   dexPol.append(DexNow);  adjPol.append(AdjNow)
            return aPol, cPol, dPol, dexPol, adjPol

        def _solve_fin_agent(sign):
            params = deepcopy(params_base)
            agent = UnempRisk_Durable_Latest(**params)
            agent.solve_terminal = False
            agent.solution_terminal = deepcopy(self.solution[0])
            agent.IncShkDstn = T * [self.IncShkDstn[0]]
            agent.del_from_time_inv(shk_param)
            agent.add_to_time_vary(shk_param)
            perturbed = (i_shock * [base_val]
                         + [base_val + sign * dx]
                         + (T - i_shock - 1) * [base_val])
            setattr(agent, shk_param, perturbed)
            agent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            agent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            agent.solve()
            agent.neutral_measure = True
            agent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            pols = _eval_pols_on_grid(agent)
            # Also return the perturbed income distribution at i_shock so that
            # the directly-shocked period's curlyD can be computed numerically,
            # capturing direct income-distribution effects (labor, UnempPrb) and
            # direct interest-rate effects (Rfree, RSpread) in one shot.
            shk_dstn_at_shock = agent.IncShkDstn[i_shock]
            return pols, shk_dstn_at_shock

        (a_p_list, c_p_list, d_p_list, dex_p_list, adj_p_list), shk_dstn_p = _solve_fin_agent(+1)
        (a_m_list, c_m_list, d_m_list, dex_m_list, adj_m_list), shk_dstn_m = _solve_fin_agent(-1)

        # ─── 2. Centered policy deviations (Step 1 of fake-news) ─────────────
        # Append SS as the t=T entry (same indexing convention as calc_jacobian)
        a_p_list.append(a_ss);    a_m_list.append(a_ss)
        c_p_list.append(c_ss);    c_m_list.append(c_ss)
        d_p_list.append(d_ss);    d_m_list.append(d_ss)
        dex_p_list.append(dex_ss); dex_m_list.append(dex_ss)
        adj_p_list.append(adj_ss); adj_m_list.append(adj_ss)

        da0_s   = np.array([(a_p_list[T-i]   - a_m_list[T-i])   / twodx for i in range(T)])
        dc0_s   = np.array([(c_p_list[T-i]   - c_m_list[T-i])   / twodx for i in range(T)])
        dd0_s   = np.array([(d_p_list[T-i]   - d_m_list[T-i])   / twodx for i in range(T)])
        ddex0_s = np.array([(dex_p_list[T-i] - dex_m_list[T-i]) / twodx for i in range(T)])
        dadj0_s = np.array([(adj_p_list[T-i] - adj_m_list[T-i]) / twodx for i in range(T)])

        A_curl_s   = da0_s.reshape(T, -1)   @ Dist_ss
        C_curl_s   = dc0_s.reshape(T, -1)   @ Dist_ss
        D_curl_s   = dd0_s.reshape(T, -1)   @ Dist_ss
        DEX_curl_s = ddex0_s.reshape(T, -1) @ Dist_ss
        ADJ_curl_s = dadj0_s.reshape(T, -1) @ Dist_ss

        # ─── 3. Distribution perturbation (curlyD) ───────────────────────────
        # Strategy:
        #   i = 1  (k = i_shock): NUMERICAL — build two sparse transition
        #          matrices using the agents' *actual* perturbed IncShkDstn.
        #          This captures every direct-shock channel in one shot:
        #            • Rfree/RSpread: R enters bNext directly
        #            • labor/UnempPrb: income shock values / probs change
        #   i ≥ 2  (k < i_shock, anticipation periods): ANALYTICAL via
        #          jump_to_grid_dur_shock.  At these periods both agents use
        #          the SS income distribution; only policies differ (through
        #          anticipation of the future shock), so the analytical
        #          linearisation is exact to O(dx²).

        Dist_curl_s = np.zeros((T, n_states))

        # ── i = 1: numerical curlyD at the directly-shocked period ──────────
        # Build bNext / nNext for each agent using the correct R at i_shock
        rfree_p_ = (base_val + dx) if shk_param == "Rfree"   else self.Rfree[0]
        rfree_m_ = (base_val - dx) if shk_param == "Rfree"   else self.Rfree[0]
        rsprd_p_ = (base_val + dx) if shk_param == "RSpread" else self.RSpread[0]
        rsprd_m_ = (base_val - dx) if shk_param == "RSpread" else self.RSpread[0]
        _R_p_shk = np.where(a_p_list[i_shock] < 0, rfree_p_ + rsprd_p_, rfree_p_)
        _R_m_shk = np.where(a_m_list[i_shock] < 0, rfree_m_ + rsprd_m_, rfree_m_)
        _bNext_p = _R_p_shk * a_p_list[i_shock] - d_p_list[i_shock] * (self.OpCost + self.dDepr * self.MaintCost)
        _bNext_m = _R_m_shk * a_m_list[i_shock] - d_m_list[i_shock] * (self.OpCost + self.dDepr * self.MaintCost)
        _nNext_p = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * d_p_list[i_shock]
        _nNext_m = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * d_m_list[i_shock]

        for _sign, _bN, _nN, _shk_dstn in [(+1, _bNext_p, _nNext_p, shk_dstn_p),
                                             (-1, _bNext_m, _nNext_m, shk_dstn_m)]:
            _prbs = _shk_dstn.pmv
            _tr   = _shk_dstn.atoms[1]
            _pm   = _shk_dstn.atoms[0]
            _nbd  = jump_to_grid_dur(np.ones_like(_tr), np.ones_like(_tr),
                                     _prbs, dist_nGrid, dist_mGrid)
            _rows, _cols, _data, _ = gen_tran_matrix_dur_sparse(
                dist_nGrid, dist_mGrid, _bN, _nN,
                _prbs, _pm, _tr, self.LivPrb[0], _nbd,
            )
            _tm = sp.sparse.csr_matrix(
                (np.array(_data, dtype=np.float64),
                 (np.array(_rows, dtype=np.int32), np.array(_cols, dtype=np.int32))),
                shape=(n_states, n_states),
            )
            if _sign == +1:
                _T_plus = _tm
            else:
                _T_minus = _tm

        Dist_curl_s[1] = (_T_plus - _T_minus) @ Dist_ss / twodx

        # ── i ≥ 2: analytical curlyD for anticipation periods ───────────────
        for i in range(2, T):
            k = T - i   # period index (< i_shock at all anticipation periods)

            da_a_i = (a_p_list[k] - a_m_list[k]) / twodx
            da_d_i = (d_p_list[k] - d_m_list[k]) / twodx

            # Both agents use SS income distribution here; only policies differ.
            # R is also SS (shock hasn't hit yet at period k < i_shock).
            bNext_p = R_vec_ss * a_p_list[k] - d_p_list[k] * (self.OpCost + self.dDepr * self.MaintCost)
            bNext_m = R_vec_ss * a_m_list[k] - d_m_list[k] * (self.OpCost + self.dDepr * self.MaintCost)
            dbNext_i = (bNext_p - bNext_m) / twodx
            dnNext_i = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * da_d_i

            Dist_curl_s[i] = jump_to_grid_dur_shock(
                dbNext_i, dnNext_i, Dist_ss_2d,
                bNext_ss, nNext_unshocked_ss,
                shk_prbs, perm_shks, tran_shks, self.LivPrb[0],
                dist_nGrid, dist_mGrid,
            )

        # ─── 4. Steps 2-3: expectation vectors → fake-news matrices ──────────
        Curl_F_A   = np.zeros((T, T))
        Curl_F_C   = np.zeros((T, T))
        Curl_F_D   = np.zeros((T, T))
        Curl_F_DEX = np.zeros((T, T))
        Curl_F_ADJ = np.zeros((T, T))

        Curl_F_A[0]   = A_curl_s
        Curl_F_C[0]   = C_curl_s
        Curl_F_D[0]   = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        if _gpu_ok:
            # GPU path: transfer ev vectors and Dist_curl_s to device
            Dc_T_gpu = cp.asarray(Dist_curl_s.T)   # (n_states, T) on GPU
            tm_gpu   = cpsp.csr_matrix(tranmat_ss)  # sparse on GPU
            ev_a   = cp.asarray(a_ss.flatten())
            ev_c   = cp.asarray(c_ss.flatten())
            ev_d   = cp.asarray(d_ss.flatten())
            ev_dex = cp.asarray(dex_ss.flatten())
            ev_adj = cp.asarray(adj_ss.flatten())
            for i in range(T - 1):
                Curl_F_A[i+1]   = cp.asnumpy(ev_a   @ Dc_T_gpu)
                Curl_F_C[i+1]   = cp.asnumpy(ev_c   @ Dc_T_gpu)
                Curl_F_D[i+1]   = cp.asnumpy(ev_d   @ Dc_T_gpu)
                Curl_F_DEX[i+1] = cp.asnumpy(ev_dex @ Dc_T_gpu)
                Curl_F_ADJ[i+1] = cp.asnumpy(ev_adj @ Dc_T_gpu)
                ev_a   = ev_a   @ tm_gpu
                ev_c   = ev_c   @ tm_gpu
                ev_d   = ev_d   @ tm_gpu
                ev_dex = ev_dex @ tm_gpu
                ev_adj = ev_adj @ tm_gpu
        else:
            # CPU path: memory-efficient, one vector per variable
            Dc_T   = Dist_curl_s.T   # (n_states, T)
            ev_a   = a_ss.flatten()
            ev_c   = c_ss.flatten()
            ev_d   = d_ss.flatten()
            ev_dex = dex_ss.flatten()
            ev_adj = adj_ss.flatten()
            for i in range(T - 1):
                Curl_F_A[i+1]   = ev_a   @ Dc_T
                Curl_F_C[i+1]   = ev_c   @ Dc_T
                Curl_F_D[i+1]   = ev_d   @ Dc_T
                Curl_F_DEX[i+1] = ev_dex @ Dc_T
                Curl_F_ADJ[i+1] = ev_adj @ Dc_T
                ev_a   = ev_a   @ tranmat_ss
                ev_c   = ev_c   @ tranmat_ss
                ev_d   = ev_d   @ tranmat_ss
                ev_dex = ev_dex @ tranmat_ss
                ev_adj = ev_adj @ tranmat_ss

        # ─── 5. Jacobian from fake-news matrix ───────────────────────────────
        J_A   = J_from_F(Curl_F_A)
        J_C   = J_from_F(Curl_F_C)
        J_D   = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        # ─── 6. Zeroth column (finite-horizon agent, as in calc_jacobian) ─────
        params_zc = deepcopy(self.__dict__["parameters"])
        params_zc["T_cycle"] = 2
        params_zc["cycles"] = 1
        for _p in ["Rfree", "RSpread", "DiscFac", "LivPrb", "PermGroFac",
                   "PermShkStd", "TranShkStd", "UnempPrb", "UnempPrbRet",
                   "IncUnemp", "IncUnempRet", "BoroCnstArt"]:
            params_zc[_p] = 2 * [getattr(self, _p)[0]]

        ZerothColAgent = UnempRisk_Durable_Latest(**params_zc)
        ZerothColAgent.solve_terminal = False
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                [getattr(self, shk_param)[0] + dx]
                + (params_zc["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )
        else:
            peturbed_list = (
                [getattr(self, shk_param) + dx]
                + (params_zc["T_cycle"] - 1) * [getattr(self, shk_param)]
            )
        setattr(ZerothColAgent, shk_param, peturbed_list)

        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.solve()
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col_dense = ZerothColAgent.tran_matrix[0]
        dstn_t_zeroth_col = self.vec_erg_dstn.T

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        # ─── 7. Restore every attribute overwritten in step 0 ────────────────
        for attr, val in saved.items():
            setattr(self, attr, val)

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    # ──────────────────────────────────────────────────────────────────────────
    def calc_jacobian_jax(self, shk_param, T):
        """
        Jacobian via JAX forward-mode autodiff (JVP) for the curlyD step.

        Architecture
        ─────────────
        • curlyY  (policy responses weighted by Dist_ss)
              Centered finite differences — identical to calc_jacobian_smooth.
              The HARK solver is not JAX-differentiable; migrating it to JAX
              would extend autodiff coverage to curlyY as well.

        • curlyD  (distribution perturbation)
              ALL periods use ``jax.jvp`` through a JAX-native re-implementation
              of the bilinear transition matrix (``_T_dot_Dist``).  This replaces:
                – the numerical FD of full sparse matrices for the directly-shocked
                  period (i = 1), and
                – the hand-derived analytical formula (jump_to_grid_dur_shock) for
                  anticipation periods (i ≥ 2).
              Benefits vs finite-difference / analytical curlyD:
                1. All shock types handled uniformly — no per-shock special cases.
                2. Kinks in the policy function at discrete-choice thresholds are
                   differentiated through correctly: JAX gives the local subgradient
                   at the operating point rather than averaging over a kink region.
                3. Adding new shock types requires only specifying their tangent
                   vectors; no analytical derivation of dT/d_shock is needed.

        • Fake-news assembly and zeroth column
              Identical to calc_jacobian_smooth.

        Parameters
        ----------
        shk_param : str    Name of the shock parameter (e.g. "Rfree", "labor").
        T         : int    Jacobian horizon (number of periods).

        Returns
        -------
        J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ : (T × T) numpy arrays
        """
        # ── JAX setup ─────────────────────────────────────────────────────────
        try:
            import jax
            import jax.numpy as jnp
            # x64 must be enabled before any JAX computation in this process.
            # Calling update() here is safe when JAX has not yet traced anything
            # for the first time; if JAX was already imported with float32 the
            # user should add  jax.config.update("jax_enable_x64", True)  near
            # the top of their script before the first import of this module.
            jax.config.update("jax_enable_x64", True)
        except ImportError as _jax_err:
            raise ImportError(
                f"JAX failed to import: {_jax_err}\n\n"
                "Install JAX (CPU):  pip install jax\n"
                "Install JAX (GPU):  pip install jax[cuda12]\n"
                "If numpy DLL errors appear, reinstall numpy via conda:\n"
                "  conda install numpy numpy-base --force-reinstall"
            ) from _jax_err

        # ── 0. Rebuild SS distribution on a fresh grid ────────────────────────
        _save_attrs = [
            'dist_mGrid', 'dist_nGrid', 'tran_matrix', 'vec_erg_dstn', 'erg_dstn',
            'aPol_Grid', 'cPol_Grid', 'dPol_Grid', 'adjPol_Grid', 'dexPol_Grid',
            'texPol_Grid',
            'a_ss', 'c_ss', 'd_ss', 'dex_ss', 'adj_ss',
            'A_ss', 'C_ss', 'D_ss', 'DEX_ss', 'TEX_ss', 'ADJ_ss',
        ]
        saved = {a: getattr(self, a) for a in _save_attrs if hasattr(self, a)}

        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur()
        self.calc_ergodic_dist()

        self.c_ss   = self.cPol_Grid
        self.d_ss   = self.dPol_Grid
        self.dex_ss = self.dexPol_Grid
        self.a_ss   = self.aPol_Grid
        self.adj_ss = self.adjPol_Grid
        ss_dstn = self.vec_erg_dstn
        self.A_ss   = self.a_ss.flatten()   @ ss_dstn
        self.C_ss   = self.c_ss.flatten()   @ ss_dstn
        self.D_ss   = self.d_ss.flatten()   @ ss_dstn
        self.DEX_ss = self.dex_ss.flatten() @ ss_dstn
        self.TEX_ss = self.C_ss + self.DEX_ss
        self.ADJ_ss = self.adj_ss.flatten() @ ss_dstn

        dist_mGrid = self.dist_mGrid
        dist_nGrid = self.dist_nGrid
        tranmat_ss = self.tran_matrix           # scipy sparse CSR
        Dist_ss    = self.vec_erg_dstn          # 1-D (n_n * n_m,)
        a_ss   = self.a_ss;  c_ss  = self.c_ss;  d_ss   = self.d_ss
        dex_ss = self.dex_ss; adj_ss = self.adj_ss
        n_n      = len(dist_nGrid)
        n_m      = len(dist_mGrid)
        n_states = n_n * n_m

        # ── Convert static model quantities to JAX arrays (float64) ──────────
        dng_j = jnp.array(dist_nGrid, dtype=jnp.float64)
        dmg_j = jnp.array(dist_mGrid, dtype=jnp.float64)
        D_j   = jnp.array(Dist_ss,    dtype=jnp.float64)

        ss_prbs_j = jnp.array(self.IncShkDstn[0].pmv,      dtype=jnp.float64)
        ss_tran_j = jnp.array(self.IncShkDstn[0].atoms[1], dtype=jnp.float64)
        ss_perm_j = jnp.array(self.IncShkDstn[0].atoms[0], dtype=jnp.float64)

        LivPrb_f = float(self.LivPrb[0])
        OpC_f    = float(self.OpCost)
        dDp_f    = float(self.dDepr)
        MnC_f    = float(self.MaintCost)
        rfree_f  = float(self.Rfree[0])
        rsprd_f  = float(self.RSpread[0])

        a_ss_j = jnp.array(a_ss.ravel(), dtype=jnp.float64)
        d_ss_j = jnp.array(d_ss.ravel(), dtype=jnp.float64)

        # Newborn distribution: bilinear mass of the single point (n=1, m=1).
        # Independent of shock parameters (the original gen_tran_matrix_dur_sparse
        # places newborns at ones_like(tran_shks) regardless of income values).
        _jnl = int(np.clip(np.searchsorted(dist_nGrid, 1.0) - 1, 0, n_n - 2))
        _jml = int(np.clip(np.searchsorted(dist_mGrid, 1.0) - 1, 0, n_m - 2))
        _jnh, _jmh = _jnl + 1, _jml + 1
        _ncw = dist_nGrid[_jnh] - dist_nGrid[_jnl]
        _mcw = dist_mGrid[_jmh] - dist_mGrid[_jml]
        _wnl = float(np.clip((dist_nGrid[_jnh] - 1.0) / _ncw, 0.0, 1.0))
        _wml = float(np.clip((dist_mGrid[_jmh] - 1.0) / _mcw, 0.0, 1.0))
        _nbd = np.zeros(n_states)
        _nbd[_jnl * n_m + _jml] = _wnl * _wml
        _nbd[_jnl * n_m + _jmh] = _wnl * (1.0 - _wml)
        _nbd[_jnh * n_m + _jml] = (1.0 - _wnl) * _wml
        _nbd[_jnh * n_m + _jmh] = (1.0 - _wnl) * (1.0 - _wml)
        NBD_j = jnp.array(_nbd, dtype=jnp.float64)

        # ── Core JAX function: T(policies, shock_params) @ Dist_ss ───────────
        # Fully differentiable w.r.t. all floating-point arguments via jax.jvp.
        #
        # Implementation: for each income shock (tran_s, perm_s, prob_s) we
        # compute, for every source state, where bNext and nNext land on the
        # distribution grid, then scatter-add the bilinear weights × Dist_ss
        # mass to the output vector.  jax.lax.scan loops over shocks so the
        # whole routine is JAX-traceable.
        #
        # Gradient flow through the bilinear weights:
        #   wmlo = clip((dmg_j[jmhi] - mN_s) / mcw, 0, 1)
        # gives  d(wmlo)/d(mN_s) = -1/mcw  (exactly the analytical formula
        # in jump_to_grid_dur_shock), so the JVP reproduces the analytical
        # result for smooth policies and generalises it to kinks.
        @jax.jit
        def _T_dot_Dist(a_f, d_f, rfree, rsprd, tran_j, perm_j, probs_j):
            """Compute T @ Dist_ss in JAX, differentiable w.r.t. all floats."""
            R_vec = jnp.where(a_f < 0.0, rfree + rsprd, rfree)
            bN    = R_vec * a_f - d_f * (OpC_f + dDp_f * MnC_f)   # (n_states,)
            nN    = (1.0 - dDp_f * (1.0 - MnC_f)) * d_f            # (n_states,)

            def _shock_step(carry, xs):
                tran_s, perm_s, prob_s = xs
                nN_s = nN / perm_s
                mN_s = bN / perm_s + tran_s

                jnlo = jnp.clip(
                    jnp.searchsorted(dng_j, nN_s, side='right') - 1, 0, n_n - 2)
                jmlo = jnp.clip(
                    jnp.searchsorted(dmg_j, mN_s, side='right') - 1, 0, n_m - 2)
                jnhi = jnlo + 1
                jmhi = jmlo + 1

                ncw = dng_j[jnhi] - dng_j[jnlo]
                mcw = dmg_j[jmhi] - dmg_j[jmlo]

                # Bilinear weights (gradient flows through these w.r.t. a_f, d_f,
                # rfree, tran_s — NOT through the integer indices jnlo/jmlo).
                wnlo = jnp.clip((dng_j[jnhi] - nN_s) / ncw, 0.0, 1.0)
                wmlo = jnp.clip((dmg_j[jmhi] - mN_s) / mcw, 0.0, 1.0)
                wnhi = 1.0 - wnlo
                wmhi = 1.0 - wmlo

                # Source mass: Dist_ss weighted by survival probability × shock prob
                sw = D_j * (LivPrb_f * prob_s)

                # Scatter to the four destination cells for every source state.
                # JAX's at[idx].add handles duplicate destination indices correctly
                # (reduce-scatter), making this equivalent to a sparse matrix row sum.
                out = carry
                out = out.at[jnlo * n_m + jmlo].add(sw * wnlo * wmlo)
                out = out.at[jnlo * n_m + jmhi].add(sw * wnlo * wmhi)
                out = out.at[jnhi * n_m + jmlo].add(sw * wnhi * wmlo)
                out = out.at[jnhi * n_m + jmhi].add(sw * wnhi * wmhi)
                return out, None

            alive, _ = jax.lax.scan(
                _shock_step, jnp.zeros(n_states), (tran_j, perm_j, probs_j))
            # Add newborn mass (independent of policies and shock params)
            return alive + (1.0 - LivPrb_f) * jnp.sum(D_j) * NBD_j

        # ── 1. Solve +dx and -dx finite-horizon agents ────────────────────────
        dx      = 1e-4
        twodx   = 2.0 * dx
        i_shock = T - 1

        params_base = deepcopy(self.__dict__["parameters"])
        params_base["T_cycle"] = T
        params_base["cycles"]  = 1
        for _p in ["Rfree", "RSpread", "DiscFac", "LivPrb", "PermGroFac",
                   "PermShkStd", "TranShkStd", "UnempPrb", "UnempPrbRet",
                   "IncUnemp", "IncUnempRet", "BoroCnstArt"]:
            params_base[_p] = T * [getattr(self, _p)[0]]

        base_val = (getattr(self, shk_param)[0]
                    if isinstance(getattr(self, shk_param), list)
                    else getattr(self, shk_param))

        def _eval_pols_on_grid(agent):
            aPol, cPol, dPol, dexPol, adjPol = [], [], [], [], []
            for k in range(T):
                aNext  = np.zeros((n_n, n_m))
                CNow   = np.zeros((n_n, n_m))
                DNow   = np.zeros((n_n, n_m))
                DexNow = np.zeros((n_n, n_m))
                AdjNow = np.zeros((n_n, n_m))
                for i_n in range(n_n):
                    n_aux      = np.full(n_m, dist_nGrid[i_n])
                    CNow[i_n]   = agent.solution[k].cFunc(n_aux, dist_mGrid)
                    DNow[i_n]   = agent.solution[k].dFunc(n_aux, dist_mGrid)
                    DexNow[i_n] = DNow[i_n] - n_aux
                    AdjNow[i_n] = np.clip(agent.solution[k].adjusting(n_aux, dist_mGrid), 0.0, 1.0)
                    aNext[i_n]  = agent.solution[k].aFunc(n_aux, dist_mGrid)
                aPol.append(aNext);  cPol.append(CNow)
                dPol.append(DNow);   dexPol.append(DexNow);  adjPol.append(AdjNow)
            return aPol, cPol, dPol, dexPol, adjPol

        def _solve_fin_agent(sign):
            params = deepcopy(params_base)
            agent  = UnempRisk_Durable_Latest(**params)
            agent.solve_terminal = False
            agent.solution_terminal = deepcopy(self.solution[0])
            agent.IncShkDstn = T * [self.IncShkDstn[0]]
            agent.del_from_time_inv(shk_param)
            agent.add_to_time_vary(shk_param)
            perturbed = (i_shock * [base_val]
                         + [base_val + sign * dx]
                         + (T - i_shock - 1) * [base_val])
            setattr(agent, shk_param, perturbed)
            agent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            agent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            agent.solve()
            agent.neutral_measure = True
            agent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            pols = _eval_pols_on_grid(agent)
            shk_dstn_at_shock = agent.IncShkDstn[i_shock]
            return pols, shk_dstn_at_shock

        (a_p_list, c_p_list, d_p_list, dex_p_list, adj_p_list), shk_dstn_p = _solve_fin_agent(+1)
        (a_m_list, c_m_list, d_m_list, dex_m_list, adj_m_list), shk_dstn_m = _solve_fin_agent(-1)

        # ── 2. Centered policy deviations (Step 1 of fake-news) ───────────────
        a_p_list.append(a_ss);    a_m_list.append(a_ss)
        c_p_list.append(c_ss);    c_m_list.append(c_ss)
        d_p_list.append(d_ss);    d_m_list.append(d_ss)
        dex_p_list.append(dex_ss); dex_m_list.append(dex_ss)
        adj_p_list.append(adj_ss); adj_m_list.append(adj_ss)

        da0_s   = np.array([(a_p_list[T-i]   - a_m_list[T-i])   / twodx for i in range(T)])
        dc0_s   = np.array([(c_p_list[T-i]   - c_m_list[T-i])   / twodx for i in range(T)])
        dd0_s   = np.array([(d_p_list[T-i]   - d_m_list[T-i])   / twodx for i in range(T)])
        ddex0_s = np.array([(dex_p_list[T-i] - dex_m_list[T-i]) / twodx for i in range(T)])
        dadj0_s = np.array([(adj_p_list[T-i] - adj_m_list[T-i]) / twodx for i in range(T)])

        A_curl_s   = da0_s.reshape(T, -1)   @ Dist_ss
        C_curl_s   = dc0_s.reshape(T, -1)   @ Dist_ss
        D_curl_s   = dd0_s.reshape(T, -1)   @ Dist_ss
        DEX_curl_s = ddex0_s.reshape(T, -1) @ Dist_ss
        ADJ_curl_s = dadj0_s.reshape(T, -1) @ Dist_ss

        # ── 3. curlyD via JAX JVP ──────────────────────────────────────────────
        # For every period we apply jax.jvp to _T_dot_Dist with the appropriate
        # tangent vectors.  This is equivalent to, but more general than, the
        # hybrid (numerical for i=1, jump_to_grid_dur_shock for i≥2) approach in
        # calc_jacobian_smooth:
        #
        #   i = 1  (k = i_shock): tangents cover policies AND all shock channels
        #          (Rfree via rfree/rsprd, labor/UnempPrb via tran_j/probs_j).
        #          No special-casing per shock type is required.
        #
        #   i ≥ 2  (anticipation periods): only policy tangents are non-zero
        #          (shock hasn't arrived yet).  The JVP through the bilinear
        #          weights reproduces jump_to_grid_dur_shock analytically.
        #
        # For non-convex models the benefit is most pronounced at kink periods:
        # finite differences average over kinks whereas JVP gives the exact
        # local derivative of _T_dot_Dist at the SS operating point, regardless
        # of policy non-linearities.
        Dist_curl_s = np.zeros((T, n_states))

        # Tangents for income-distribution inputs at i_shock (centered FD on
        # the perturbed IncShkDstn returned by each ±dx agent).  We include the
        # permanent-shock atoms as well (atoms[0]); these move for shocks such as
        # PermShkStd, and omitting them would bias the directly-shocked curlyD.
        d_tran_j = (jnp.array(shk_dstn_p.atoms[1], dtype=jnp.float64)
                    - jnp.array(shk_dstn_m.atoms[1], dtype=jnp.float64)) / twodx
        d_perm_j = (jnp.array(shk_dstn_p.atoms[0], dtype=jnp.float64)
                    - jnp.array(shk_dstn_m.atoms[0], dtype=jnp.float64)) / twodx
        d_prbs_j = (jnp.array(shk_dstn_p.pmv, dtype=jnp.float64)
                    - jnp.array(shk_dstn_m.pmv, dtype=jnp.float64)) / twodx

        # Tangents for Rfree / RSpread (non-zero only for the matching shock)
        d_rfree_f = 1.0 if shk_param == "Rfree"   else 0.0
        d_rsprd_f = 1.0 if shk_param == "RSpread" else 0.0

        # Policy tangents at the directly-shocked period
        da_ish_j = jnp.array(
            (a_p_list[i_shock] - a_m_list[i_shock]).ravel() / twodx,
            dtype=jnp.float64)
        dd_ish_j = jnp.array(
            (d_p_list[i_shock] - d_m_list[i_shock]).ravel() / twodx,
            dtype=jnp.float64)

        # ── i = 1: JVP for the directly-shocked period ────────────────────────
        # All shock channels are captured in one shot via their tangent vectors
        # (policies, Rfree/RSpread, and the transitory/permanent income atoms
        #  and probabilities).
        def _T_ishock(a_f, d_f, rfree, rsprd, tran_j, perm_j, prbs_j):
            return _T_dot_Dist(a_f, d_f, rfree, rsprd, tran_j, perm_j, prbs_j)

        primals_ish  = (a_ss_j, d_ss_j, rfree_f, rsprd_f, ss_tran_j, ss_perm_j, ss_prbs_j)
        tangents_ish = (da_ish_j, dd_ish_j, d_rfree_f, d_rsprd_f, d_tran_j, d_perm_j, d_prbs_j)
        _, _curlyD_1 = jax.jvp(_T_ishock, primals_ish, tangents_ish)
        Dist_curl_s[1] = np.array(_curlyD_1)

        # ── i ≥ 2: JVP for anticipation periods ───────────────────────────────
        # At k < i_shock the income distribution and R are both at SS;
        # only the policy grid values change (anticipation of the future shock).
        def _T_antip(a_f, d_f):
            return _T_dot_Dist(a_f, d_f, rfree_f, rsprd_f,
                               ss_tran_j, ss_perm_j, ss_prbs_j)

        for i in range(2, T):
            k      = T - i
            da_k_j = jnp.array(
                (a_p_list[k] - a_m_list[k]).ravel() / twodx, dtype=jnp.float64)
            dd_k_j = jnp.array(
                (d_p_list[k] - d_m_list[k]).ravel() / twodx, dtype=jnp.float64)
            _, _curlyD_i = jax.jvp(_T_antip, (a_ss_j, d_ss_j), (da_k_j, dd_k_j))
            Dist_curl_s[i] = np.array(_curlyD_i)

        # ── 4. Steps 2-3: expectation vectors → fake-news matrices ────────────
        Curl_F_A   = np.zeros((T, T))
        Curl_F_C   = np.zeros((T, T))
        Curl_F_D   = np.zeros((T, T))
        Curl_F_DEX = np.zeros((T, T))
        Curl_F_ADJ = np.zeros((T, T))

        Curl_F_A[0]   = A_curl_s
        Curl_F_C[0]   = C_curl_s
        Curl_F_D[0]   = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        Dc_T   = Dist_curl_s.T
        ev_a   = a_ss.flatten()
        ev_c   = c_ss.flatten()
        ev_d   = d_ss.flatten()
        ev_dex = dex_ss.flatten()
        ev_adj = adj_ss.flatten()
        for i in range(T - 1):
            Curl_F_A[i+1]   = ev_a   @ Dc_T
            Curl_F_C[i+1]   = ev_c   @ Dc_T
            Curl_F_D[i+1]   = ev_d   @ Dc_T
            Curl_F_DEX[i+1] = ev_dex @ Dc_T
            Curl_F_ADJ[i+1] = ev_adj @ Dc_T
            ev_a   = ev_a   @ tranmat_ss
            ev_c   = ev_c   @ tranmat_ss
            ev_d   = ev_d   @ tranmat_ss
            ev_dex = ev_dex @ tranmat_ss
            ev_adj = ev_adj @ tranmat_ss

        # ── 5. Jacobian from fake-news matrix ─────────────────────────────────
        J_A   = J_from_F(Curl_F_A)
        J_C   = J_from_F(Curl_F_C)
        J_D   = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        # ── 6. Zeroth column (finite-horizon agent, as in calc_jacobian) ──────
        params_zc = deepcopy(self.__dict__["parameters"])
        params_zc["T_cycle"] = 2
        params_zc["cycles"] = 1
        for _p in ["Rfree", "RSpread", "DiscFac", "LivPrb", "PermGroFac",
                   "PermShkStd", "TranShkStd", "UnempPrb", "UnempPrbRet",
                   "IncUnemp", "IncUnempRet", "BoroCnstArt"]:
            params_zc[_p] = 2 * [getattr(self, _p)[0]]

        ZerothColAgent = UnempRisk_Durable_Latest(**params_zc)
        ZerothColAgent.solve_terminal = False
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                [getattr(self, shk_param)[0] + dx]
                + (params_zc["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )
        else:
            peturbed_list = (
                [getattr(self, shk_param) + dx]
                + (params_zc["T_cycle"] - 1) * [getattr(self, shk_param)]
            )
        setattr(ZerothColAgent, shk_param, peturbed_list)

        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.solve()
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col_dense = ZerothColAgent.tran_matrix[0]
        dstn_t_zeroth_col = self.vec_erg_dstn.T

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        # ── 7. Restore overwritten attributes ─────────────────────────────────
        for attr, val in saved.items():
            setattr(self, attr, val)

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ

    def check_curlyD_analytical_vs_numerical(self, shk_param, T, periods=None,
                                             dx=1e-4, verbose=True):
        """
        Sanity check for the linearised distribution response (curlyD) used by
        ``calc_jacobian_smooth`` (for anticipation periods i >= 2) and, through
        the same math, ``calc_jacobian_jax``.

        For each *anticipation* period k = T - i (with k < i_shock, where the
        income distribution AND the interest rate are still at steady state and
        only the policy functions differ) it compares two estimates of curlyD:

          * ANALYTICAL  -- ``jump_to_grid_dur_shock`` (closed-form derivative of
            the bilinear transition), and
          * NUMERICAL   -- centered finite difference of two *float64* sparse
            transition matrices built from the same +-dx policies:
                (T_plus - T_minus) @ Dist_ss / (2 dx).

        In smooth regions the two agree to O(dx^2); they differ only near policy
        kinks.  A large, systematic discrepancy (especially a sign flip) points
        to an error in ``jump_to_grid_dur_shock``.

        Parameters
        ----------
        shk_param : str        Shock parameter name (e.g. "Rfree", "labor").
        T         : int        Jacobian horizon.
        periods   : list[int] or None
            Anticipation indices i (each with 2 <= i <= T-1) to check.  These
            match the row index of ``Dist_curl_s`` in the smooth/jax methods.
            Defaults to a spread of up to five valid indices.
        dx        : float      Perturbation size (default 1e-4).
        verbose   : bool       Print a summary table.

        Returns
        -------
        results : dict
            {i: {"period_k", "max_abs_diff", "max_rel_diff", "l2_rel_diff",
                 "mass_analytical", "mass_numerical",
                 "curlyD_analytical", "curlyD_numerical"}}
        """
        # ── 0. Rebuild SS distribution on a fresh grid (as in the smooth method) ──
        _save_attrs = [
            'dist_mGrid', 'dist_nGrid', 'tran_matrix', 'vec_erg_dstn', 'erg_dstn',
            'aPol_Grid', 'cPol_Grid', 'dPol_Grid', 'adjPol_Grid', 'dexPol_Grid',
            'texPol_Grid',
            'a_ss', 'c_ss', 'd_ss', 'dex_ss', 'adj_ss',
            'A_ss', 'C_ss', 'D_ss', 'DEX_ss', 'TEX_ss', 'ADJ_ss',
        ]
        saved = {a: getattr(self, a) for a in _save_attrs if hasattr(self, a)}

        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur()
        self.calc_ergodic_dist()

        a_ss = self.aPol_Grid
        d_ss = self.dPol_Grid

        dist_mGrid = self.dist_mGrid
        dist_nGrid = self.dist_nGrid
        Dist_ss    = self.vec_erg_dstn
        n_n        = len(dist_nGrid)
        n_m        = len(dist_mGrid)
        n_states   = n_n * n_m
        Dist_ss_2d = Dist_ss.reshape(n_n, n_m)

        # SS income shock arrays (assumed neutral measure, as at steady state)
        shk_prbs  = self.IncShkDstn[0].pmv
        tran_shks = self.IncShkDstn[0].atoms[1]
        perm_shks = self.IncShkDstn[0].atoms[0]
        LivPrb    = self.LivPrb[0]

        # SS next-period state quantities for the analytical curlyD
        R_vec_ss           = np.where(a_ss < 0, self.Rfree[0] + self.RSpread[0], self.Rfree[0])
        bNext_ss           = R_vec_ss * a_ss - d_ss * (self.OpCost + self.dDepr * self.MaintCost)
        nNext_unshocked_ss = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * d_ss

        # Newborn distribution (invariant to the shock; needed by the sparse gen)
        _nbd = jump_to_grid_dur(np.ones_like(tran_shks), np.ones_like(tran_shks),
                                shk_prbs, dist_nGrid, dist_mGrid)

        # ── 1. Solve +dx and -dx finite-horizon agents ────────────────────────
        twodx   = 2.0 * dx
        i_shock = T - 1

        params_base = deepcopy(self.__dict__["parameters"])
        params_base["T_cycle"] = T
        params_base["cycles"]  = 1
        for _p in ["Rfree", "RSpread", "DiscFac", "LivPrb", "PermGroFac",
                   "PermShkStd", "TranShkStd", "UnempPrb", "UnempPrbRet",
                   "IncUnemp", "IncUnempRet", "BoroCnstArt"]:
            params_base[_p] = T * [getattr(self, _p)[0]]

        base_val = (getattr(self, shk_param)[0]
                    if isinstance(getattr(self, shk_param), list)
                    else getattr(self, shk_param))

        def _eval_pols_on_grid(agent):
            aPol, dPol = [], []
            for k in range(T):
                aNext = np.zeros((n_n, n_m))
                DNow  = np.zeros((n_n, n_m))
                for i_n in range(n_n):
                    n_aux      = np.full(n_m, dist_nGrid[i_n])
                    DNow[i_n]  = agent.solution[k].dFunc(n_aux, dist_mGrid)
                    aNext[i_n] = agent.solution[k].aFunc(n_aux, dist_mGrid)
                aPol.append(aNext);  dPol.append(DNow)
            return aPol, dPol

        def _solve_fin_agent(sign):
            params = deepcopy(params_base)
            agent  = UnempRisk_Durable_Latest(**params)
            agent.solve_terminal = False
            agent.solution_terminal = deepcopy(self.solution[0])
            agent.IncShkDstn = T * [self.IncShkDstn[0]]
            agent.del_from_time_inv(shk_param)
            agent.add_to_time_vary(shk_param)
            perturbed = (i_shock * [base_val]
                         + [base_val + sign * dx]
                         + (T - i_shock - 1) * [base_val])
            setattr(agent, shk_param, perturbed)
            agent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            agent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            agent.solve()
            agent.neutral_measure = True
            agent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
            return _eval_pols_on_grid(agent)

        a_p_list, d_p_list = _solve_fin_agent(+1)
        a_m_list, d_m_list = _solve_fin_agent(-1)

        # ── 2. Numerical curlyD helper for an anticipation period ─────────────
        # k < i_shock: income distribution and R are at SS, only policies differ.
        def _build_tm(a_pol, d_pol):
            R_vec = np.where(a_pol < 0, self.Rfree[0] + self.RSpread[0], self.Rfree[0])
            bN = R_vec * a_pol - d_pol * (self.OpCost + self.dDepr * self.MaintCost)
            nN = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * d_pol
            rows, cols, data, _ = gen_tran_matrix_dur_sparse(
                dist_nGrid, dist_mGrid, bN, nN,
                shk_prbs, perm_shks, tran_shks, LivPrb, _nbd,
            )
            return sp.sparse.csr_matrix(
                (np.array(data, dtype=np.float64),
                 (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                shape=(n_states, n_states),
            )

        # ── 3. Pick which anticipation indices to check ───────────────────────
        valid = list(range(2, T))  # i such that k = T - i is a strict anticipation period
        if periods is None:
            if len(valid) <= 5:
                periods = valid
            else:
                periods = sorted(set(np.linspace(valid[0], valid[-1], 5).astype(int)))
        else:
            periods = [i for i in periods if i in valid]

        results = {}
        for i in periods:
            k = T - i

            da_d_i  = (d_p_list[k] - d_m_list[k]) / twodx
            bNext_p = R_vec_ss * a_p_list[k] - d_p_list[k] * (self.OpCost + self.dDepr * self.MaintCost)
            bNext_m = R_vec_ss * a_m_list[k] - d_m_list[k] * (self.OpCost + self.dDepr * self.MaintCost)
            dbNext_i = (bNext_p - bNext_m) / twodx
            dnNext_i = (1.0 - self.dDepr * (1.0 - self.MaintCost)) * da_d_i

            curlyD_ana = jump_to_grid_dur_shock(
                dbNext_i, dnNext_i, Dist_ss_2d,
                bNext_ss, nNext_unshocked_ss,
                shk_prbs, perm_shks, tran_shks, LivPrb,
                dist_nGrid, dist_mGrid,
            )

            Tp = _build_tm(a_p_list[k], d_p_list[k])
            Tm = _build_tm(a_m_list[k], d_m_list[k])
            curlyD_num = (Tp - Tm) @ Dist_ss / twodx

            diff      = curlyD_ana - curlyD_num
            denom     = np.maximum(np.abs(curlyD_num).max(), 1e-30)
            max_abs   = np.abs(diff).max()
            max_rel   = max_abs / denom
            l2_rel    = np.linalg.norm(diff) / max(np.linalg.norm(curlyD_num), 1e-30)

            results[i] = {
                "period_k": k,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "l2_rel_diff":  l2_rel,
                "mass_analytical": curlyD_ana.sum(),
                "mass_numerical":  curlyD_num.sum(),
                "curlyD_analytical": curlyD_ana,
                "curlyD_numerical":  curlyD_num,
            }

        # ── 4. Restore overwritten attributes ─────────────────────────────────
        for attr, val in saved.items():
            setattr(self, attr, val)

        if verbose:
            print(f"curlyD analytical-vs-numerical check | shk_param={shk_param}, "
                  f"T={T}, dx={dx:g}")
            print(f"{'i':>4} {'k':>4} {'max_abs':>12} {'max_rel':>12} "
                  f"{'l2_rel':>12} {'mass_ana':>12} {'mass_num':>12}")
            for i in periods:
                r = results[i]
                print(f"{i:>4} {r['period_k']:>4} {r['max_abs_diff']:>12.3e} "
                      f"{r['max_rel_diff']:>12.3e} {r['l2_rel_diff']:>12.3e} "
                      f"{r['mass_analytical']:>12.3e} {r['mass_numerical']:>12.3e}")
            print("(curlyD is mass-preserving, so mass_* should be ~0; "
                  "l2_rel should be small away from kinks.)")

        return results

    def calc_MPC(self):
        """
        Calculates the MPC, eg the Jacobian of aggregate consumption of a shock of labor

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        MPC: numpy.array
            Marginal Propensity to Consume Non-Durable Goods with respect to labor shock

        MPC_DEX: numpy.array
            Marginal Propensity to Consume Durable Goods with respect to labor shock

        MPC_TEX: numpy.array
            Marginal Propensity to Consume Non-Durable and Durable Goods with respect to labor shock
        """

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########
        shk_param = "labor"
        dx = 1e-4 #0.001  # Size of perturbation
        T = 2  # Dimension of Jacobian Matrix
        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix
        
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]
        
        
        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update()
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        ZerothColAgent.updateBM_TranMatrix_numba_jacobian(self.UnempPrb[0], self.IncUnemp[0])
        
        # Solve
        ZerothColAgent.solve()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        # ZerothColAgent.update_income_process_truncated()
        ZerothColAgent.update_income_process_truncated_jacobian(self.UnempPrb[0], self.IncUnemp[0])

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix

        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T#[0]

        C_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)

        ### Sparse Version
        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col

        J_C = (C_t_no_sim - self.C_ss) / dx
        J_DEX = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX = J_C + J_DEX

        MPC = J_C.T[0]
        MPC_DEX = J_DEX.T[0]
        MPC_TEX = J_TEX.T[0]
        return MPC, MPC_DEX, MPC_TEX


    def calc_jacobian_Unemp(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Non-Durable Consumption with respect to shk_param

        DEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param       

        TEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Stock with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        ADJJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Adjusting Probability with respect to shk_param
        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon
        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = UnempRisk_Durable_Latest(**params)
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 1e-4 #0.001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.updateBM_TranMatrix_numba()

        # Solve
        FinHorizonAgent.solve()

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        # FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.update_income_process_truncated()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        a_t = FinHorizonAgent.aPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # NOTE: the steady-state policy grids are appended to the finite-horizon
        # policy lists once, in the block below.  Do NOT append here as well:
        # c_t/a_t/... are references to the agent's own lists, so appending twice
        # would place the SS grid at both index T and T+1.

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Stock Policy
        dex_ss = self.dexPol_Grid  # steady state Durable Expenditure Policy
        adj_ss = self.adjPol_Grid  # steady state Adjusting Policy

        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        dex_t.append(self.dex_ss)
        adj_t.append(self.adj_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        dd0_s = []  # Deviation of Durable Stock policy from steady state policy
        ddex0_s = []  # Deviation of Durable Expenditure policy from steady state policy
        dadj0_s = []  # Deviation of Adjusting policy from steady state policy

        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)
            dd0_s.append(d_t[T - i] - d_ss)
            ddex0_s.append(dex_t[T - i] - dex_ss)
            dadj0_s.append(adj_t[T - i] - adj_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)
        dd0_s = np.array(dd0_s)
        ddex0_s = np.array(ddex0_s)
        dadj0_s = np.array(dadj0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        Dist_ss = self.vec_erg_dstn.T#[0] # Shape: ((nxm) x 1)
        dA0_s = []
        dC0_s = []
        dD0_s = []
        dDEX0_s = []
        dADJ0_s = []

        ### Sparse Version
        for i in range(T):
            dA0_s.append(da0_s[i].flatten() @ Dist_ss)
            dC0_s.append(dc0_s[i].flatten() @ Dist_ss)
            dD0_s.append(dd0_s[i].flatten() @ Dist_ss)
            dDEX0_s.append(ddex0_s[i].flatten() @ Dist_ss)
            dADJ0_s.append(dadj0_s[i].flatten() @ Dist_ss)

        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        dA0_s = np.array(dA0_s)
        dC0_s = np.array(dC0_s)
        dD0_s = np.array(dD0_s)
        dDEX0_s = np.array(dDEX0_s)
        dADJ0_s = np.array(dADJ0_s)

        A_curl_s = dA0_s / dx # Shape (T,1)
        C_curl_s = dC0_s / dx
        D_curl_s = dD0_s / dx
        DEX_curl_s = dDEX0_s / dx
        ADJ_curl_s = dADJ0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)


        ### Does not make a huge array, but keeps it for each T
        dDist0_s_shape = (T, len(self.dist_nGrid) * len(self.dist_mGrid)) # Shape: T x (nxm)
        dDist0_s = np.zeros(dDist0_s_shape)

        ### Sparse Version
        for i in range(T):
            dlambda0_s_i = tranmat_t[T - i] - tranmat_ss   # sparse - sparse = sparse
            dDist0_s[i] = dlambda0_s_i @ Dist_ss           # sparse × dense = dense

        Dist_curl_s = dDist0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors: Shape T x n x m
        exp_vecs_a = []
        exp_vecs_c = []
        exp_vecs_d = []
        exp_vecs_dex = []
        exp_vecs_adj = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss.flatten()
        exp_vec_c = c_ss.flatten()
        exp_vec_d = d_ss.flatten()
        exp_vec_dex = dex_ss.flatten()
        exp_vec_adj = adj_ss.flatten()

        ### Sparse Version
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = exp_vec_a.flatten() @ tranmat_ss    # sparse matmul

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = exp_vec_c.flatten() @ tranmat_ss

            exp_vecs_d.append(exp_vec_d)
            exp_vec_d = exp_vec_d.flatten() @ tranmat_ss

            exp_vecs_dex.append(exp_vec_dex)
            exp_vec_dex = exp_vec_dex.flatten() @ tranmat_ss

            exp_vecs_adj.append(exp_vec_adj)
            exp_vec_adj = exp_vec_adj.flatten() @ tranmat_ss

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)
        exp_vecs_d = np.array(exp_vecs_d)
        exp_vecs_dex = np.array(exp_vecs_dex)
        exp_vecs_adj = np.array(exp_vecs_adj)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption
        Curl_F_D = np.zeros((T, T))  # Fake news matrix for durable stock
        Curl_F_DEX = np.zeros((T, T))  # Fake news matrix for durable expenditure
        Curl_F_ADJ = np.zeros((T, T))  # Fake news matrix for durable expenditure

        # First row of Fake News Matrix: Shaper ((n x m) x 1)
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_DEX[0] = DEX_curl_s
        Curl_F_ADJ[0] = ADJ_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i].flatten(), Dist_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i].flatten(), Dist_curl_s[j])
                Curl_F_D[i + 1][j] = np.dot(exp_vecs_d[i].flatten(), Dist_curl_s[j])
                Curl_F_DEX[i + 1][j] = np.dot(exp_vecs_dex[i].flatten(), Dist_curl_s[j])
                Curl_F_ADJ[i + 1][j] = np.dot(exp_vecs_adj[i].flatten(), Dist_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_DEX = J_from_F(Curl_F_DEX)
        J_TEX = J_C + J_DEX
        J_ADJ = J_from_F(Curl_F_ADJ)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########
        
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]
        
        
        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update()
        ZerothColAgent.update_income_process_truncated()
        ZerothColAgent.updateBM_TranMatrix_numba()
        
        # Solve
        ZerothColAgent.solve()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated()

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix

        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T#[0]

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        DEX_t_no_sim = np.zeros(T)
        ADJ_t_no_sim = np.zeros(T)

        ### Sparse Version
        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = tranmat_t_zeroth_col_dense @ dstn_t_zeroth_col
            else:
                dstn_t_zeroth_col = tranmat_ss @ dstn_t_zeroth_col

            A_t_no_sim[i] = self.aPol_Grid.flatten() @ dstn_t_zeroth_col
            C_t_no_sim[i] = self.cPol_Grid.flatten() @ dstn_t_zeroth_col
            D_t_no_sim[i] = self.dPol_Grid.flatten() @ dstn_t_zeroth_col
            DEX_t_no_sim[i] = self.dexPol_Grid.flatten() @ dstn_t_zeroth_col
            ADJ_t_no_sim[i] = self.adjPol_Grid.flatten() @ dstn_t_zeroth_col

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ




    ### njit version of calc_jacobian
    def calc_jacobian_njit(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Non-Durable Consumption with respect to shk_param

        DEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param       

        TEXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Stock with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        ADJJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Adjusting Probability with respect to shk_param
        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = UnempRisk_Durable_Latest(**params)
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 1e-4 #0.001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        # FinHorizonAgent.update_income_process_truncated()
        FinHorizonAgent.update()

        # Solve
        FinHorizonAgent.solve()

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.update_income_process_truncated()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        a_t = FinHorizonAgent.aPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # NOTE: the steady-state policy grids are appended to the finite-horizon
        # policy lists once, in the block below.  Do NOT append here as well:
        # c_t/a_t/... are references to the agent's own lists, so appending twice
        # would place the SS grid at both index T and T+1.

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Stock Policy
        dex_ss = self.dexPol_Grid  # steady state Durable Expenditure Policy
        adj_ss = self.adjPol_Grid  # steady state Adjusting Policy
        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # Steady state distribution of market resources
        Dist_ss = self.vec_erg_dstn.T# [0]
        
        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        dex_t = FinHorizonAgent.dexPol_Grid
        adj_t = FinHorizonAgent.adjPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t = FinHorizonAgent.aPol_Grid + [a_ss]
        c_t = FinHorizonAgent.cPol_Grid + [c_ss]
        d_t = FinHorizonAgent.dPol_Grid + [d_ss]
        dex_t = FinHorizonAgent.dexPol_Grid + [dex_ss]
        adj_t = FinHorizonAgent.adjPol_Grid + [adj_ss]
        tranmat_t = FinHorizonAgent.tran_matrix + [tranmat_ss]

        ### All in One
        J_A, J_C, J_D, J_DEX, J_TEX, J_ADJ = FakeNews_njit(T, dx, a_ss, c_ss, d_ss, dex_ss, adj_ss, a_t, c_t, d_t, dex_t, adj_t, tranmat_ss, tranmat_t, Dist_ss)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########
        
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # Required for finite horizon

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["RSpread"] = params["T_cycle"] * [self.RSpread[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]
        
        
        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = UnempRisk_Durable_Latest(**params)
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        # ZerothColAgent.update_income_process_truncated()
        ZerothColAgent.update()

        # Solve
        ZerothColAgent.solve()

        # Add Limit
        # if self.dDepr <1:
        #     ZerothColAgent.add_limit()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_truncated()

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix
        tranmat_t_zeroth_col_dense = tranmat_t_zeroth_col[0] #sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T#[0]

        # Call the optimized compute function
        A_t_no_sim, C_t_no_sim, D_t_no_sim, DEX_t_no_sim, ADJ_t_no_sim = compute_ZerothCol(
            T,
            tranmat_t_zeroth_col_dense,
            tranmat_ss,
            dstn_t_zeroth_col,
            a_ss.flatten(), c_ss.flatten(), d_ss.flatten(), dex_ss.flatten(), adj_ss.flatten()
        )
        
        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_DEX.T[0] = (DEX_t_no_sim - self.DEX_ss) / dx
        J_TEX.T[0] = J_C.T[0] + J_DEX.T[0]
        J_ADJ.T[0] = (ADJ_t_no_sim - self.ADJ_ss) / dx

        return J_C, J_DEX, J_TEX, J_D, J_A, J_ADJ
    
    
class DurableConsumerSolution(MetricObject):
    """
    SAME AS ConsumerSolution BUT WITH DURABLE CONSUMPTION FUNCTION
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    The solution has 3 steps:
    Step 1: Compute the post-decision functions EndOfPrdv and EndOfPrdvP on a grid over the post-decision states n_t and a_t
    Step 2: Solve the keeper problem on a grid over the pre-decision states n_t, m_t, where the combined EGM and upper
            envelope Algorithm is applied for each n_t.
    Step 3: Solve the adjuster problem using interpolation of the keeper value function found in step 2.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over durable stock and market
        resources: c = cFunc(n,m).
    dFunc: function
        The durable consumption function for this period, defined over durable stock and
        market resources: d = dFunc(n,m)
    vFunc : function
        The beginning-of-period value function for this period, defined over durable stock
        and market resources: v = vFunc(n,m).
    vPFunc: function
        The beginning-of-period marginal utility function for this period, defined over durable stock
        and market resources: uP = vPFunc(n,m).
    cFuncKeep : function
        The consumption function for this periods' keeper problem, defined over durable stock and market
        resources: c = cFunc(n,m).
    dFuncKeep: function
        The durable consumption function for this periods' keeper problem, defined over durable stock and
        market resources: d = dFunc(n,m).
    vFuncKeep : function
        The beginning-of-period value function for this periods' keeper problem, defined over durable stock
        and market resources: v = vFunc(n,m).
    vPFuncKeep: function
        The beginning-of-period marginal utility function for this periods' keeper problem, defined over durable stock
        and market resources: uP = vPFunc(n,m).
    cFuncAdj : function
        The consumption function for this periods' adjuster problem, defined over sum of durable stock and market
        resources: c = cFunc(x).
    dFuncAdj: function
        The durable consumption function for this periods' adjuster problem, defined over sum of durable stock and
        market resources: d = dFunc(x)
    vFuncAdj : function
        The beginning-of-period value function for this periods' adjuster problem, defined over sum of durable stock
        and market resources: v = vFunc(x).
    vPFuncAdj: function
        The beginning-of-period marginal utility function for this periods' adjuster problem, defined over durable stock
        and market resources: uP = vPFunc(n,m).
    adjusting : function
        The adjusting function for this period indicates if for a given durable stock and market resources, the agent
        adjusts the durable stock or not: adjusting = adjusting(n,m).

    """
    distance_criteria = ["inv_vPFunc", "inv_vPDFunc"] # Default (Other possibilities could be the vPFunc and vPDFunc as well as the Keeper and Adjuster Specific ones.)

    def __init__(
        self,
        # Policy Functions
        cFunc = None,
        cFuncAdj = None,
        cFuncKeep = None,
        dFunc = None, 
        dFuncAdj = None,
        dFuncKeep = None,
        aFunc = None,
        aFuncAdj = None,
        aFuncKeep = None,
        adjusting = None,

        # (Marginal) Value Functions as inverse
        inv_vPFunc = None,
        inv_vPDFunc = None,
        inv_vFunc = None,
        inv_vFuncAdj = None,
        inv_vFuncKeep = None,
        # Adjuster
        # mNrmGrid = None,
        # nNrmGrid = None,
        # dNrmGrid = None,
        xNrmMin = None,
        mNrmMin = None,
        MPCmin = None,
        hNrm = None,
        MPCmax = None,
        # aXtraGrid=None,
        # inv_vFuncKeepUnc = None,
        # cFuncKeepUnc = None,
        # exFuncKeepUnc= None,
        # # Constraint functions
        cFuncCnst = None,
        dFuncCnst = None,
        # # Comparing upper envelope
        # EndOfPrdvPOne=None,
        # EndOfPrdvPTwo=None,
        # EndOfPrdvPD = None,
        # EndOfPrdv=None,
        # aNrmGridNow=None,
        # c_egm_array=None,
        # m_egm_array=None,
        # v_egm_array=None,
        # BoroCnstNat = None,
        # m_thresh_max = None,
        # d_upper_max = None,
        # d_kink = None,
        # kink = None,
        # xAdj_array = None,
        # dAdj_array = None,
        # cAdj_array = None,

        vPFunc_eval = None,
        vPDFunc_eval = None,
        # vPDFuncKeep = None, 
        # vPDFuncAdj = None,

        # lSuS_array = None,
        # adjusting_array = None,
        # d_kink = None,

        # tau_func = None,
        # SearchCost_func = None,

        ### Frictonless Solution
        qFunc_Frictionless = None,
        # cFunc_Frictionless = None,
        # dFunc_Frictionless = None,
        vFunc_Frictionless = None,
        vPFunc_Frictionless = None,
        # inv_vFunc_Frictionless = None,
    ):

        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.cFuncAdj = cFuncAdj if cFuncAdj is not None else NullFunc()
        self.cFuncKeep = cFuncKeep if cFuncKeep is not None else NullFunc()
        self.dFunc = dFunc if dFunc is not None else NullFunc() # NEW
        self.dFuncAdj = dFuncAdj if dFuncAdj is not None else NullFunc()
        self.dFuncKeep = dFuncKeep if dFuncKeep is not None else NullFunc()
        self.aFunc = aFunc if aFunc is not None else NullFunc() # NEW
        self.aFuncAdj = aFuncAdj if aFuncAdj is not None else NullFunc()
        self.aFuncKeep = aFuncKeep if aFuncKeep is not None else NullFunc()
        # self.vFunc = vFunc if vFunc is not None else NullFunc()
        # self.vFuncAdj = vFuncAdj if vFuncAdj is not None else NullFunc()
        # self.vFuncKeep = vFuncKeep if vFuncKeep is not None else NullFunc()
        # self.vPFunc = vPFunc if vPFunc is not None else NullFunc()
        # self.vPFuncKeep = vPFuncKeep if vPFuncKeep is not None else NullFunc()
        # self.vPFuncAdj = vPFuncAdj if vPFuncAdj is not None else NullFunc()
        # self.vPDFunc = vPDFunc if vPDFunc is not None else NullFunc()
        self.inv_vPFunc = inv_vPFunc if inv_vPFunc is not None else NullFunc()
        self.inv_vPDFunc = inv_vPDFunc if inv_vPDFunc is not None else NullFunc()
        self.inv_vFunc = inv_vFunc if inv_vFunc is not None else NullFunc()
        self.inv_vFuncAdj = inv_vFuncAdj if inv_vFuncAdj is not None else NullFunc()
        self.inv_vFuncKeep = inv_vFuncKeep if inv_vFuncKeep is not None else NullFunc()
        self.adjusting = adjusting if adjusting is not None else NullFunc()
        # self.mNrmGrid = mNrmGrid if mNrmGrid is not None else NullFunc()
        # self.nNrmGrid = nNrmGrid if nNrmGrid is not None else NullFunc()
        # self.dNrmGrid = dNrmGrid if dNrmGrid is not None else NullFunc()
        self.xNrmMin = xNrmMin
        self.mNrmMin = mNrmMin if mNrmMin is not None else NullFunc()
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        # self.aXtraGrid = aXtraGrid
        # # Constraint functions
        self.cFuncCnst = cFuncCnst if cFuncCnst is not None else NullFunc()
        self.dFuncCnst = dFuncCnst if dFuncCnst is not None else NullFunc()
        # self.EndOfPrdvPOne = EndOfPrdvPOne if EndOfPrdvPOne is not None else NullFunc()
        # self.EndOfPrdvPTwo = EndOfPrdvPTwo if EndOfPrdvPTwo is not None else NullFunc()
        # self.EndOfPrdvPD = EndOfPrdvPD if EndOfPrdvPD is not None else NullFunc()
        # self.EndOfPrdv = EndOfPrdv if EndOfPrdv is not None else NullFunc()
        # self.aNrmGridNow = aNrmGridNow if aNrmGridNow is not None else NullFunc()
        # self.BoroCnstNat = BoroCnstNat if BoroCnstNat is not None else NullFunc()
        # self.xAdj_array = xAdj_array if xAdj_array is not None else NullFunc()
        # self.dAdj_array = dAdj_array if dAdj_array is not None else NullFunc()
        # self.cAdj_array = cAdj_array if cAdj_array is not None else NullFunc()

        ### Evaluation Functions
        self.vPFunc_eval = vPFunc_eval if vPFunc_eval is not None else NullFunc()
        self.vPDFunc_eval = vPDFunc_eval if vPDFunc_eval is not None else NullFunc()

        ### SS-structure
        # self.lSuS_array = lSuS_array if lSuS_array is not None else NullFunc()
        # self.adjusting_array = adjusting_array if adjusting_array is not None else NullFunc()
        # self.d_kink = d_kink if d_kink is not None else NullFunc()

        # self.tau_func = tau_func if tau_func is not None else NullFunc()
        # self.SearchCost_func = SearchCost_func if SearchCost_func is not None else NullFunc()

        ### Frictionless Solution
        self.qFunc_Frictionless = qFunc_Frictionless if qFunc_Frictionless is not None else NullFunc()
        # self.cFunc_Frictionless = cFunc_Frictionless if cFunc_Frictionless is not None else NullFunc()
        # self.dFunc_Frictionless = dFunc_Frictionless if dFunc_Frictionless is not None else NullFunc()
        self.vFunc_Frictionless = vFunc_Frictionless if vFunc_Frictionless is not None else NullFunc()
        self.vPFunc_Frictionless = vPFunc_Frictionless if vPFunc_Frictionless is not None else NullFunc()
        # self.inv_vFunc_Frictionless = inv_vFunc_Frictionless if inv_vFunc_Frictionless is not None else NullFunc()

def solve_DurableConsumer(
        solution_next,
        IncShkDstn,
        # All sorts of parameters
        alpha,  # Cobb-Douglas parameter for non-durable good consumption in utility function
        dDepr,  # Depreciation Rate of Durable Stock
        adjC,  # Adjustment costs
        d_ubar,  # Minimum durable stock for utility function
        CRRA,
        DiscFac,
        LivPrb,
        Rfree,
        PermGroFac,
        # Grids:
        mNrmGrid, #mNrmCount would be enough as well
        nNrmGrid,
        xNrmGrid, #xNrmCount would be enough as well
        grid_type,
        NestFac,
        # Borrowing Constraint:
        BoroCnstdNrm, # Cannot have negative durable Stock
        tol, # tolerance for optimization function and when to adjust vs keep
        BoroCnstArt,
        expected_method,
        aXtraGrid,
        UpperEnvelope,
        Limit,
        extrap_method,
        SearchCost,
        ScaleBool,
        ScaleFac,
        ScaleInt,
        ### Integration
        bNrmGrid_income, mNrmGrid_income, probGrid_income,
        ### Car Crash Shock
        CarCrashBool,
        CarCrashPrb,
        CarCrashMin,
        # tau_func,
        LTV,
        tau_func,
        SearchCost_func,
        #
        MaintCost,
        OpCost,
        RSpread, 
):
    ####################################################################################################################
    # 1. Update utility functions:
    # i. U(C,D)
    # global cFuncAdjUnc
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # v. uPD U(C,D) wrt D
    CRRAutilityPD = lambda C, D: (
                (1 - alpha) * (C ** (alpha * (1 - CRRA))) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA) - 1))


    ### Create Functions for root-finding
    def rootfinding_function(a_root, D):
        C = CRRAutilityP_inv(EndOfPrdvPOne(D, a_root), D)
        Term1 = CRRAutilityPD(C, D)
        Term2 = EndOfPrdvPTwo(D, a_root) 
        Term3 = EndOfPrdvPD(D, a_root) 
        return Term1 - Term2 + Term3

    def max_constraint_withBorrowingConstraint(X, d):

        a = -d * (1 - dDepr * (1 - MaintCost) - OpCost) * (1 - adjC) + xNrmMinNow  # Adjusted bounds
        c = X - a - d  # Non-durable consumption
        EndOfPrdv_aux = EndOfPrdv(d, a)
        CRRA_utility_aux = CRRAutility(c, d)
        utility = CRRA_utility_aux + EndOfPrdv_aux
        utility_inv_val = utility_inv(utility, CRRA)
        return utility_inv_val, a, c

    ### Find the kink point at which a = - BoroCnstArt1D - d * (1 - dDepr) * (1 - adjC)
    def find_xkink(x0, dFuncAdjUnc, aFuncAdjUnc, BoroCnstArt, LTV, dDepr, adjC, MaintCost, OpCost):
        """
        Finds the kink point x0 such that a0 = -BoroCnstArt - d0 * (1 - dDepr) * (1 - adjC),
        assuming linear interpolation between dFuncAdjUnc and aFuncAdjUnc.
        """
        d0 = dFuncAdjUnc(x0)
        a0 = aFuncAdjUnc(x0)
        agoal = -BoroCnstArt - d0 * (1 - dDepr * (1 - MaintCost) - OpCost) * (1 - adjC) * LTV
        # Calculate and return the difference:
        diff = a0 - agoal
        return diff
    
    ####################################################################################################################
    Errors = 0
    Figure = False
    # Scale = False
    timeBOOL = _PROFILE_DURABLE
    _t = {}                                   # section timings (only populated when timeBOOL)
    _t0_total = time.perf_counter()
    ####################################################################################################################
    # print('new period')
    # 1) Shock values:
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    iShock = len(PermShkValsNext)

    ### Only PermShock (For 2nd step of Integratio method)
    PermShkValsNext_only = np.unique(PermShkValsNext)
    ShkPrbsNext_PermShk = np.ones_like(PermShkValsNext_only)/len(PermShkValsNext_only)
    
    # Car Crash
    CarCrashVal = np.array([CarCrashMin, 1.0])
    CarCrashPrb_array = np.array([CarCrashPrb, 1 - CarCrashPrb])

    # 2. Unpack next period's solution
    inv_vFunc_next = solution_next.inv_vFunc
    inv_vPFuncNext = solution_next.inv_vPFunc
    inv_vPDFuncNext = solution_next.inv_vPDFunc
    qFunc_Frictionless_Next  = solution_next.qFunc_Frictionless
    vPFunc_Frictionless_Next = solution_next.vPFunc_Frictionless
    vFunc_Frictionless_Next  = solution_next.vFunc_Frictionless

    # Constraint function for depreciation = 1 and Keeper
    cFuncCnst = solution_next.cFuncCnst
    dFuncCnst = solution_next.dFuncCnst

    p1 = ScaleInt[0]
    p2 = ScaleInt[1]

    ### durable grids
    nNrmGridNow = nNrmGrid
    dNrmGridNow = nNrmGrid

    ### NATURAL BORROWING CONSTRAINT START
    # NATURAL BORROWING CONSTRAINT. Similar to def def_BoroCnst(self, BoroCnstArt):
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    TranShk_min = np.min(TranShkValsNext) 
    DiscFacEff = DiscFac * LivPrb
    WorstIncPrb = np.sum(
        ShkPrbsNext[
            (PermShkValsNext * TranShkValsNext)
            == (PermShkMinNext * TranShkMinNext)
            ]
    )

    # Update the bounding MPCs and PDV of human wealth
    if alpha <1:
        gam = (alpha/(1 - alpha)) * (Rfree + dDepr + OpCost - 1)/Rfree ## Includes Durables
        Lambda_inv = (1 + gam) / (1 + gam + (Rfree  + dDepr + OpCost - 1)/Rfree)
    else:
        gam = 1
        Lambda_inv = 1
        
    PatFac = (((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree) #* ((1 + gam)/(1 + gam*(Rfree - 1 + dDepr)/Rfree)) ## Includes Durables


    Thorn_R = ((Rfree * DiscFac)**(1/CRRA))/(Rfree)


    MPCminNow = (1.0 / (1.0 + PatFac / solution_next.MPCmin))
    MPXminNow = np.min([MPCminNow * Lambda_inv, 1.0])

    Ex_IncNext = np.dot(
        ShkPrbsNext, TranShkValsNext * PermShkValsNext
    )
    hNrmNow = (
            PermGroFac / Rfree * (Ex_IncNext + solution_next.hNrm)
    )
    MPCmaxNow = 1.0 / (
            1.0
            + (WorstIncPrb ** (1.0 / CRRA))
            * PatFac
            / solution_next.MPCmax
    )

    # Calculate the minimum allowable value of money resources in this period
    ### For xNrmGrid (given d is 0)
    BoroCnstNat1D_NoR = (
            (solution_next.xNrmMin - TranShkMinNext)
            * (PermGroFac * PermShkMinNext)
    )
    if BoroCnstNat1D_NoR < 0:
        BoroCnstNat1D =  BoroCnstNat1D_NoR/ (Rfree + RSpread)
    else:
        BoroCnstNat1D =  BoroCnstNat1D_NoR/ Rfree

    ### Given d is > 0. Can borrow against durable stock. BUT: account for interest rates as well.
    # Use solution_next.mNrmMin (array indexed by nNrmGrid) evaluated at the worst-case
    # next-period durable stock. The worst case occurs at ψ_min, because a lower permanent
    # shock inflates n' = (1-δ)*d/ψ, pushing next-period mNrmMin further negative while
    # simultaneously making m' harder to achieve (R*a smaller in normalised terms).
    nNext_worst = np.clip(
        (1 - dDepr) * dNrmGridNow / PermShkMinNext,
        nNrmGridNow[0], nNrmGridNow[-1]
    )
    if isinstance(solution_next.mNrmMin, np.ndarray):
        if len(solution_next.mNrmMin) == len(nNrmGridNow):
            # Normal case: full 2-D constraint array, one entry per n-grid point.
            mNrmMinNext_worst = np.interp(nNext_worst, nNrmGridNow, solution_next.mNrmMin)
        elif len(solution_next.mNrmMin) == 1:
            # Collapsed constant (e.g. dDepr=1 → zero collateral term): broadcast.
            mNrmMinNext_worst = np.full(len(dNrmGridNow), solution_next.mNrmMin[0])
        else:
            # Unexpected length mismatch — fall back to old scalar approximation.
            mNrmMinNext_worst = (
                solution_next.xNrmMin
                - dNrmGridNow * (1 - dDepr) * (1 - tau_func(0.0)) * LTV
            )
    else:
        # Fall back to the old scalar approximation if mNrmMin is not yet an array
        mNrmMinNext_worst = (
            solution_next.xNrmMin
            - dNrmGridNow * (1 - dDepr) * (1 - tau_func(0.0)) * LTV
        )
    BoroCnstNat2D_NoR = (mNrmMinNext_worst - TranShkMinNext) * (PermGroFac * PermShkMinNext)

    ### BoroCnstNat2D_NoR = (
            # (solution_next.xNrmMin - TranShkMinNext - dNrmGridNow * (1 - dDepr) * (1 - tau_func
            # (0.0)) * LTV) * (PermGroFac * PermShkMinNext)
    BoroCnstNat2D_NoR[BoroCnstNat2D_NoR < 0] /= (Rfree + RSpread)
    BoroCnstNat2D_NoR[BoroCnstNat2D_NoR > 0] /= Rfree

    BoroCnstNat2D = BoroCnstNat2D_NoR

   
    ### For bank balances (before transitory shocks)
    BoroCnstNat2D_b = solution_next.mNrmMin - TranShkMinNext #TranShkMinNext #BoroCnstNat2D - ( TranShkMinNext * (PermGroFac * PermShkMinNext)/ Rfree)

    # BoroCnstNat = nNrmGrid + BoroCnstNat
    # Note: need to be sure to handle BoroCnstArt==None appropriately.
    # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
    # However in Py3, this raises a TypeError. Thus here we need to directly
    # address the situation in which BoroCnstArt == None:
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat2D
        xNrmMinNow = BoroCnstNat1D # Natural borrowing constraint only depends on assets, as here c == d == 0
    else:
        BoroCnstArt2D = BoroCnstArt - dNrmGridNow * (1 - tau_func(np.zeros_like(dNrmGridNow))) * (1 - dDepr * (1 - MaintCost) - OpCost) * LTV * (PermGroFac * PermShkMinNext) / Rfree  #
        if LTV == 0:
            mNrmMinNow = np.array([BoroCnstArt])
        else:
            mNrmMinNow = BoroCnstArt2D
        xNrmMinNow = BoroCnstArt #np.clip(BoroCnstTotalNat, 0.0, BoroCnstArt) # Natural borrowing constraint only depends on assets, as here c == d == 0
    
    if BoroCnstNat1D < np.min([xNrmMinNow]):
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow

 
    ### Add grid of all Natural Borrowing Constraints to aXtraGrid
    aNrmGridNow = np.asarray(aXtraGrid)
    aNrmGridNow = np.append(aNrmGridNow, aXtraGrid[0] + BoroCnstNat2D) 
    aNrmGridNow = np.append(aNrmGridNow, BoroCnstNat2D) 
    aNrmGridNow = np.unique(np.sort(aNrmGridNow))   

    ### Add another layer of grid between aXtraGrid[0] and BoroCnstTotalNat[0]
    aNrmGrid_add = np.linspace(aXtraGrid[0], BoroCnstNat2D[0], 100)
    aNrmGridNow = np.append(aNrmGridNow, aNrmGrid_add)
    aNrmGridNow = np.unique(np.sort(aNrmGridNow))

    ### Add another layer
    aNrmGridNow = np.append(aNrmGridNow, aXtraGrid[0] + BoroCnstNat2D + 0.1)
    aNrmGridNow = np.append(aNrmGridNow, aXtraGrid[0] + BoroCnstNat2D + 1.0) 
    aNrmGridNow = np.unique(np.sort(aNrmGridNow))

    ### Create R_vec
    R_vec = np.where(aNrmGridNow < 0, Rfree + RSpread, Rfree)


    mNrmGridNow = np.append(mNrmGrid, mNrmMinNow) #BoroCnstNat
    mNrmGridNow = np.unique(np.append(mNrmGridNow, np.max(mNrmGrid)))

    # zero_grid_m = -nNrmGridNow * (1 - tau_func(nNrmGridNow, np.zeros_like(nNrmGridNow)))
    ### Add to mNrmGridNow
    mNrmGridNow = np.append(mNrmGridNow, mNrmMinNow)
    mNrmGridNow = np.unique(np.sort(mNrmGridNow))
    
    ### One Grid for Unconstrained (minimum is BoroCnstNat) and one for Total Solution (minimum is BoroCnstArt)
    xNrmGridNow = xNrmGrid + BoroCnstNat1D
    if BoroCnstArt is None:
        xNrmGridTotal = np.unique(xNrmGridNow)
    else:
        xNrmGridTotal = np.unique(xNrmGrid + BoroCnstArt)

    #########################################################################################################################################################################
    # 3. Post decision function:
    '''
    Compute the post-decision functions $w_t$ and $q_t$ on a grid over the post-decision states $d_t, a_t$.
    End of period value function (EndOfPrdvFunc in IndShockConsumer): EndOfPrdv
    End-of-period marginal values (EndOfPrdvP  in IndShockConsumer): EndOfPrdvP
    
    w_t(d_t, a_t) = \beta E[v_{t+1} (n_{t+1}, m_{t+1})].
    u_c(c_t,n_t) &= \alpha c_t^{\alpha(1 - \rho) - 1} n_t^{(1 - \alpha)(1 - \rho)} = q_t
    '''
    if timeBOOL:
        tic = time.perf_counter()
    ### i. Initialize End of Period Functions
    post_shape = (len(dNrmGridNow), len(aNrmGridNow))
    EndOfPrdvPOne_array = np.zeros(post_shape)
    EndOfPrdvPTwo_array = np.zeros(post_shape)
    EndOfPrdvPD_array = np.zeros(post_shape)
    EndOfPrdv_array = np.zeros(post_shape)
    LambdaM_array = np.zeros(post_shape)
    # inv_EndOfPrdv_array_shape = (len(dNrmGridNow), len(aNrmGridNow)+1)
    inv_EndOfPrdv_array = np.zeros(post_shape)
    if expected_method == "expected":
        # print("Not done yet")
        for i_d in range(len(dNrmGridNow)):
            EndOfPrdvPOne_array[i_d] = (
                    DiscFac * LivPrb * PermGroFac ** (-CRRA) * R_vec * 
                    expected(vp_next_vPDFunc, IncShkDstn, args=(
                        inv_vPFuncNext, aNrmGridNow, dNrmGridNow[i_d],
                        CRRA, dDepr, R_vec, PermGroFac)))
            
            EndOfPrdvPTwo_array[i_d] = (DiscFac * LivPrb * PermGroFac ** (-CRRA) * (R_vec + OpCost + MaintCost * dDepr) * 
                                        expected(vp_next_vPDFunc, IncShkDstn, args=(
                        inv_vPFuncNext, aNrmGridNow, dNrmGridNow[i_d],
                        CRRA, dDepr, R_vec, PermGroFac)))

            EndOfPrdvPD_array[i_d] = (
                    DiscFac * LivPrb * PermGroFac ** (-CRRA) * (1 - dDepr * (1 - MaintCost)) *
                    expected(vp_next_vPDFunc, IncShkDstn, args=(
                        inv_vPDFuncNext, aNrmGridNow, dNrmGridNow[i_d],
                        CRRA, dDepr, R_vec, PermGroFac)))
            
            EndOfPrdv_array[i_d] = (
                    DiscFac * LivPrb * expected(vFunc_next, IncShkDstn, args=(
                utility_inv, inv_vFunc_next, aNrmGridNow, dNrmGridNow[i_d], CRRA, dDepr,
                R_vec, PermGroFac)))
            # inv_EndOfPrdv_array[i_d,1:] = utility_inv(EndOfPrdv_array[i_d], CRRA)
            inv_EndOfPrdv_array[i_d,1:] = utility_inv(EndOfPrdv_array[i_d,1:], CRRA)

        # Replace EndOfPrdvPOne_array nan with np.inf
        EndOfPrdvPOne_array[np.isnan(EndOfPrdvPOne_array)] = np.inf
        # Replace EndOfPrdvPTwo_array nan with np.inf
        EndOfPrdvPTwo_array[np.isnan(EndOfPrdvPTwo_array)] = np.inf
        # Replace EndOfPrdvPD_array nan with np.inf
        EndOfPrdvPD_array[np.isnan(EndOfPrdvPD_array)] = np.inf

    elif expected_method == "TransitionMatrix":
        ### Interim Step: From b to m
        # Empty Container
        vPnext_array_shape = (len(nNrmGridNow), len(mNrmGrid_income))
        inv_vPDFuncNext_array = np.zeros(vPnext_array_shape)
        inv_vFuncNext_array = np.zeros(vPnext_array_shape)
        inv_vPFuncNext_array = np.zeros(vPnext_array_shape)

        # Vectorised: build full 2-D (nN × nM) grids and evaluate all rows at once.
        # LinearFast.eval_linear is numba-compiled → single batch call is much
        # faster than a Python for-loop over nNrmGridNow.
        if timeBOOL:
            _tic_init = time.perf_counter()
        _n2d = np.repeat(nNrmGridNow[:, np.newaxis], len(mNrmGrid_income), axis=1)   # (nN, nM)
        _m2d = np.tile(mNrmGrid_income[np.newaxis, :], (len(nNrmGridNow), 1))         # (nN, nM)
        if _GPU_AVAILABLE:
            # GPU path: interpolate AND apply inverse transforms entirely on GPU.
            # Keeping arrays on GPU eliminates:
            #   • D2H transfer of 3 × (nN, nM) FP64 arrays (~86 MB)
            #   • CPU power ops x**(-CRRA) on 3 × 3.6 M elements (~150 ms)
            #   • H2D re-upload for dot3 (~86 MB → FP32 cast on GPU instead)
            _n_flat_g = _cp.asarray(np.ascontiguousarray(_n2d.ravel()))
            _m_flat_g = _cp.asarray(np.ascontiguousarray(_m2d.ravel()))
            _shape2d  = _n2d.shape
            _xg0 = _cp.asarray(np.ascontiguousarray(inv_vPFuncNext.grid_list[0]))
            _yg0 = _cp.asarray(np.ascontiguousarray(inv_vPFuncNext.grid_list[1]))
            _inv_vP_g  = _cp.clip(
                _eval2d_batch_gpu(_xg0, _yg0,
                                  _cp.asarray(np.ascontiguousarray(inv_vPFuncNext.f_val)),
                                  _n_flat_g, _m_flat_g), 0.0, None).reshape(_shape2d)
            _xg1 = _cp.asarray(np.ascontiguousarray(inv_vPDFuncNext.grid_list[0]))
            _yg1 = _cp.asarray(np.ascontiguousarray(inv_vPDFuncNext.grid_list[1]))
            _inv_vPD_g = _cp.clip(
                _eval2d_batch_gpu(_xg1, _yg1,
                                  _cp.asarray(np.ascontiguousarray(inv_vPDFuncNext.f_val)),
                                  _n_flat_g, _m_flat_g), 0.0, None).reshape(_shape2d)
            _xg2 = _cp.asarray(np.ascontiguousarray(inv_vFunc_next.grid_list[0]))
            _yg2 = _cp.asarray(np.ascontiguousarray(inv_vFunc_next.grid_list[1]))
            _inv_vF_g  = _cp.clip(
                _eval2d_batch_gpu(_xg2, _yg2,
                                  _cp.asarray(np.ascontiguousarray(inv_vFunc_next.f_val)),
                                  _n_flat_g, _m_flat_g), 0.0, None).reshape(_shape2d)
            # Apply inverse transforms on GPU (saves ~150 ms CPU x**(-CRRA) ops)
            _vP_g  = _inv_vP_g  ** (-CRRA)
            _vPD_g = _inv_vPD_g ** (-CRRA)
            _vF_g  = (_inv_vF_g ** (1.0 - CRRA)) / (1.0 - CRRA)
            # Replace inf with sentinel values on GPU
            _vP_g  = _cp.where(_cp.isinf(_vP_g),  _cp.float64(1e16),  _vP_g)
            _vPD_g = _cp.where(_cp.isinf(_vPD_g), _cp.float64(1e16),  _vPD_g)
            _vF_g  = _cp.where(_cp.isinf(_vF_g),  _cp.float64(-1e16), _vF_g)
            _gpu_vstack_ready = True
        else:
            inv_vPFuncNext_array  = np.clip(inv_vPFuncNext(_n2d, _m2d),  0, None)
            inv_vPDFuncNext_array = np.clip(inv_vPDFuncNext(_n2d, _m2d), 0, None)
            inv_vFuncNext_array   = np.clip(inv_vFunc_next(_n2d, _m2d),  0, None)
            _gpu_vstack_ready = False
        if timeBOOL:
            _t['init_batch'] = time.perf_counter() - _tic_init

        ### Take inverse (CPU path only; GPU path computed transforms above)
        # inv_vFunc stores u_inner = utility_inv(v, CRRA); recover true value via utility(u_inner).
        if not _gpu_vstack_ready:
            vFuncNext_array = utility(inv_vFuncNext_array, CRRA)
            vPDFuncNext_array = inv_vPDFuncNext_array ** (-CRRA)
            vPFuncNext_array  = inv_vPFuncNext_array  ** (-CRRA)
            vPDFuncNext_array[np.isinf(vPDFuncNext_array)] = 1e16
            vPFuncNext_array[np.isinf(vPFuncNext_array)]   = 1e16
            vFuncNext_array[np.isinf(vFuncNext_array)]     = -1e16

        ### Step 4: Pre-multiply the Markov matrix by the matrix you just constructed. The resulting
        ###         matrix of intermediate expected values will have b row-wise.
        if timeBOOL:
            _tic_dot = time.perf_counter()
            if 'probGrid_shape' not in _t:
                _t['probGrid_shape'] = probGrid_income.shape
                _nN_for_log = _vP_g.shape[0] if _gpu_vstack_ready else vPFuncNext_array.shape[0]
                _t['vPFuncNext_shape'] = (_nN_for_log, probGrid_income.shape[1])

        # probGrid_income is a fully dense (nB × nM) matrix built from the
        # continuous log-normal income density — every cell is nonzero.
        # CPU BLAS FP64 takes ~3 s. GPU FP32 cuBLAS at 15 TFLOPS takes ~15 ms.
        #
        # All 3 matmuls are batched into ONE DGEMM:
        #   probGrid (nB, nM) @ stack([vP, vPD, v]).T (nM, 3·nN) → (nB, 3·nN)
        # then sliced into three (nB, nN) arrays.
        #
        # probGrid is cached on GPU (FP32) to avoid repeated H2D transfers.
        if _GPU_AVAILABLE:
            _pid = id(probGrid_income)
            if _pid not in _PROB_GPU_CACHE:
                _PROB_GPU_CACHE.clear()     # evict stale entries (time_vary)
                _PROB_GPU_CACHE[_pid] = _cp.asarray(
                    np.ascontiguousarray(probGrid_income), dtype=_cp.float32)
                if timeBOOL:
                    _t['probGrid_upload'] = time.perf_counter() - _tic_dot
            _prob_g32 = _PROB_GPU_CACHE[_pid]

            if _gpu_vstack_ready:
                # Arrays already on GPU — no H2D needed; cast FP32 directly on GPU
                _nN_dot = _vP_g.shape[0]
                _vstack_g32 = _cp.hstack([_vP_g.T, _vPD_g.T, _vF_g.T]).astype(_cp.float32)
            else:
                # Fallback: upload from CPU (only when GPU init_batch unavailable)
                _nN_dot = vPFuncNext_array.shape[0]
                _vstack = np.ascontiguousarray(
                    np.hstack([vPFuncNext_array.T,
                               vPDFuncNext_array.T,
                               vFuncNext_array.T]))
                _vstack_g32 = _cp.asarray(_vstack, dtype=_cp.float32)

            # Single batched FP32 matmul on GPU
            _res_g32 = _prob_g32 @ _vstack_g32         # (nB, 3·nN) FP32
            _res = _cp.asnumpy(_res_g32).astype(np.float64)

            Interm_vP_array  = _res[:,          :_nN_dot]
            Interm_vPD_array = _res[:,   _nN_dot:2*_nN_dot]
            Interm_v_array   = _res[:, 2*_nN_dot:]
        else:
            # CPU path: dense BLAS FP64
            Interm_vP_array  = np.dot(probGrid_income, vPFuncNext_array.T)
            Interm_vPD_array = np.dot(probGrid_income, vPDFuncNext_array.T)
            Interm_v_array   = np.dot(probGrid_income, vFuncNext_array.T)

        Interm_inv_vP_array  = Interm_vP_array  ** (-1 / CRRA)
        Interm_inv_vPD_array = Interm_vPD_array ** (-1 / CRRA)
        Interm_inv_v_array   = utility_inv(Interm_v_array, CRRA)
        if timeBOOL:
            _t['dot3'] = time.perf_counter() - _tic_dot
       
        # 2) Add a zero point
        ### a) Interpolate functions
        if timeBOOL:
            _tic_lf1 = time.perf_counter()
        Interm_inv_vP_Interp_func = LinearFast(Interm_inv_vP_array, [bNrmGrid_income, nNrmGridNow])
        Interm_vPfunc_Integration = MargValueFuncCRRA(Interm_inv_vP_Interp_func, CRRA)

        Interm_inv_vPD_Interp_func = LinearFast(Interm_inv_vPD_array, [bNrmGrid_income, nNrmGridNow])
        Interm_vPDfunc_Integration = MargValueFuncCRRA(Interm_inv_vPD_Interp_func, CRRA)

        Interm_inv_v_Interp_func = LinearFast(Interm_inv_v_array, [bNrmGrid_income, nNrmGridNow])
        Interm_vFunc_Integration = ValueFuncCRRA(Interm_inv_v_Interp_func, CRRA)
        if timeBOOL:
            _t['lf1'] = time.perf_counter() - _tic_lf1

        ### b) Add a zero-grid to bNrmGrid_income
        """
        If mNextMin starts at 0. bNextMin should be 0-TransShkMinNext.
        """
        bNrmGrid_incomeNew = np.unique(np.sort(np.append(bNrmGrid_income, BoroCnstNat2D_b)))

        ### c) For each nNrmGridNow, add a zero point to the Interm_inv_vP_array, Interm_inv_vPD_array, Interm_inv_v_array and interpolate for the other points.
        Interm_inv_vP_withzero_array = np.zeros((len(bNrmGrid_incomeNew), len(nNrmGridNow)))
        Interm_inv_vPD_withzero_array = np.zeros((len(bNrmGrid_incomeNew), len(nNrmGridNow)))
        Interm_inv_v_withzero_array = np.zeros((len(bNrmGrid_incomeNew), len(nNrmGridNow)))
        if timeBOOL:
            _tic_wz = time.perf_counter()
        if _GPU_AVAILABLE:
            # GPU path: evaluate all (nBnew × nN) query points in one batch,
            # then zero out the constrained region per column.
            _nBnew = len(bNrmGrid_incomeNew)
            _nN_wz = len(nNrmGridNow)
            # Compute per-column start index on CPU (cheap)
            _idx_arr_wz = np.empty(_nN_wz, dtype=np.int64)
            for _i in range(_nN_wz):
                if dDepr == 1.0:
                    _idx_arr_wz[_i] = 0
                elif len(BoroCnstNat2D_b) == 1:
                    _idx_arr_wz[_i] = int(np.where(bNrmGrid_incomeNew >= BoroCnstNat2D_b)[0][0])
                else:
                    _idx_arr_wz[_i] = int(np.where(bNrmGrid_incomeNew >= BoroCnstNat2D_b[_i])[0][0])
            # Build (nBnew, nN) query grids
            _bq_g = (_cp.asarray(bNrmGrid_incomeNew)[:, _cp.newaxis]
                     * _cp.ones((1, _nN_wz), dtype=_cp.float64))           # (nBnew, nN)
            _nq_g = (_cp.ones((_nBnew, 1), dtype=_cp.float64)
                     * _cp.asarray(nNrmGridNow)[_cp.newaxis, :])            # (nBnew, nN)
            _bq_flat = _bq_g.ravel()
            _nq_flat = _nq_g.ravel()
            # GPU interpolation grids (from the pre-computed LinearFast objects)
            _bGrid_wz = _cp.asarray(np.ascontiguousarray(Interm_inv_vP_Interp_func.grid_list[0]))
            _nGrid_wz = _cp.asarray(np.ascontiguousarray(Interm_inv_vP_Interp_func.grid_list[1]))
            _fvP_wz   = _cp.asarray(np.ascontiguousarray(Interm_inv_vP_Interp_func.f_val))
            _fvPD_wz  = _cp.asarray(np.ascontiguousarray(Interm_inv_vPD_Interp_func.f_val))
            _fv_wz    = _cp.asarray(np.ascontiguousarray(Interm_inv_v_Interp_func.f_val))
            # Batch GPU interpolation
            _rP_wz  = _eval2d_batch_gpu(_bGrid_wz, _nGrid_wz, _fvP_wz,  _bq_flat, _nq_flat).reshape(_nBnew, _nN_wz)
            _rPD_wz = _eval2d_batch_gpu(_bGrid_wz, _nGrid_wz, _fvPD_wz, _bq_flat, _nq_flat).reshape(_nBnew, _nN_wz)
            _rv_wz  = _eval2d_batch_gpu(_bGrid_wz, _nGrid_wz, _fv_wz,   _bq_flat, _nq_flat).reshape(_nBnew, _nN_wz)
            # Mask: row j is valid for column i if j > idx_arr_wz[i]
            _row_idx_wz = _cp.arange(_nBnew)[:, _cp.newaxis]              # (nBnew, 1)
            _col_idx_wz = _cp.asarray(_idx_arr_wz)[_cp.newaxis, :]        # (1, nN)
            _mask_wz = _row_idx_wz > _col_idx_wz                          # (nBnew, nN)
            Interm_inv_vP_withzero_array  = _cp.asnumpy(_cp.where(_mask_wz, _rP_wz,  0.0))
            Interm_inv_vPD_withzero_array = _cp.asnumpy(_cp.where(_mask_wz, _rPD_wz, 0.0))
            Interm_inv_v_withzero_array   = _cp.asnumpy(_cp.where(_mask_wz, _rv_wz,  0.0))
        else:
            # CPU path: parallelised over n-rows (joblib threads).
            def _one_interm_row(i_n):
                n_aux_grid = np.ones(len(bNrmGrid_incomeNew)) * nNrmGridNow[i_n]
                if dDepr == 1.0:
                    idx = 0
                elif len(BoroCnstNat2D_b) == 1:
                    idx = np.where(bNrmGrid_incomeNew >= BoroCnstNat2D_b)[0][0]
                else:
                    idx = np.where(bNrmGrid_incomeNew >= BoroCnstNat2D_b[i_n])[0][0]
                b_slice = bNrmGrid_incomeNew[idx + 1:]
                n_slice = n_aux_grid[idx + 1:]
                vP_row  = Interm_inv_vP_Interp_func(b_slice, n_slice)
                vPD_row = Interm_inv_vPD_Interp_func(b_slice, n_slice)
                v_row   = Interm_inv_v_Interp_func(b_slice, n_slice)
                return idx, vP_row, vPD_row, v_row

            _interm_rows = Parallel(n_jobs=-1, backend='threading')(
                delayed(_one_interm_row)(i_n) for i_n in range(len(nNrmGridNow))
            )
            for i_n, (idx, vP_row, vPD_row, v_row) in enumerate(_interm_rows):
                Interm_inv_vP_withzero_array[idx + 1:, i_n]  = vP_row
                Interm_inv_vPD_withzero_array[idx + 1:, i_n] = vPD_row
                Interm_inv_v_withzero_array[idx + 1:, i_n]   = v_row
        if timeBOOL:
            _t['withzero'] = time.perf_counter() - _tic_wz
            

        # Interpolate
        if timeBOOL:
            _tic_lf2 = time.perf_counter()
        Interm_inv_vP_Interp_func = LinearFast(Interm_inv_vP_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow])
        Interm_vPfunc_Integration = MargValueFuncCRRA(Interm_inv_vP_Interp_func, CRRA)

        Interm_inv_vPD_Interp_func = LinearFast(Interm_inv_vPD_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow])
        Interm_vPDfunc_Integration = MargValueFuncCRRA(Interm_inv_vPD_Interp_func, CRRA)

        Interm_inv_v_Interp_func = LinearFast(Interm_inv_v_withzero_array, [bNrmGrid_incomeNew, nNrmGridNow])
        Interm_vFunc_Integration = ValueFuncCRRA(Interm_inv_v_Interp_func, CRRA)
        if timeBOOL:
            _t['lf2'] = time.perf_counter() - _tic_lf2

        ### Step 5: Using the pre-specified exogenous grid of a, compute expected end-of-period
        ###         (marginal) value using the typical discretized approximation to the permanent shock
        ###         distribution. For any (a) end-of-period state, there is a finite set of future (b)
        ###         points that can be reached under the discretization.

        ### End-of-period Marginal Value Function (Step 2)
        ### With Car Crash Shock
        if CarCrashBool == True:
            for i_d in range(len(dNrmGridNow)):
                for ishock in range(len(PermShkValsNext_only)):
                    for iCarCrash in range(len(CarCrashPrb_array)):
                        n_next = CarCrashVal[iCarCrash] * np.ones(len(aNrmGridNow)) * ((1 - dDepr * (1 - MaintCost)) * dNrmGridNow[i_d]) / (PermShkValsNext_only[ishock])
                        b_next = (R_vec * aNrmGridNow -  dNrmGridNow[i_d] * (OpCost + dDepr * MaintCost)) / (PermGroFac * PermShkValsNext_only[ishock]) 

                        # Use total solution. This already incorporates the optimal adjuster/keeper region
                        Interm_vP_InterpNext_array = Interm_vPfunc_Integration(b_next, n_next)
                        Interm_vPD_InterpNext_array = Interm_vPDfunc_Integration(b_next, n_next)
                        Interm_inv_v_InterpNext_array = np.clip(Interm_inv_v_Interp_func(b_next, n_next), 0, None)
                        Interm_v_InterpNext_array = utility(Interm_inv_v_InterpNext_array, CRRA)

                        EndOfPrdvPOne_array[i_d] += CarCrashPrb_array[iCarCrash] * ShkPrbsNext_PermShk[ishock] * PermShkValsNext_only[ishock] ** (-CRRA) * \
                            DiscFac * LivPrb * (PermGroFac ** (-CRRA)) * R_vec * Interm_vP_InterpNext_array
                        EndOfPrdvPTwo_array[i_d] += CarCrashPrb_array[iCarCrash] * ShkPrbsNext_PermShk[ishock] * PermShkValsNext_only[ishock] ** (-CRRA) * \
                            DiscFac * LivPrb * (PermGroFac ** (-CRRA)) * (R_vec + OpCost + MaintCost * dDepr) * Interm_vP_InterpNext_array                        
                        EndOfPrdvPD_array[i_d] += CarCrashPrb_array[iCarCrash] * ShkPrbsNext_PermShk[ishock] * PermShkValsNext_only[ishock] ** (-CRRA) * \
                            DiscFac * LivPrb * (PermGroFac ** (-CRRA)) * (1 - dDepr* (1 - MaintCost)) * Interm_vPD_InterpNext_array
                        EndOfPrdv_array[i_d] += CarCrashPrb_array[iCarCrash] * ShkPrbsNext_PermShk[ishock] * PermShkValsNext_only[ishock] ** (1.0 - CRRA) * PermGroFac ** (
                                1.0 - CRRA) * DiscFac * LivPrb * Interm_v_InterpNext_array
                        LambdaM_array[i_d] += CarCrashPrb_array[iCarCrash] * ShkPrbsNext_PermShk[ishock] * PermShkValsNext_only[ishock] ** (-CRRA) * \
                            DiscFac * LivPrb * (PermGroFac ** (-CRRA)) * (OpCost + MaintCost * dDepr) * Interm_vP_InterpNext_array
                inv_EndOfPrdv_array[i_d] = utility_inv(EndOfPrdv_array[i_d], CRRA) 
        ### Without Car Crash Shock — fully vectorised over (d, shock, a)
        else:
            if timeBOOL:
                _tic_eop = time.perf_counter()
            nD  = len(dNrmGridNow)
            nSh = len(PermShkValsNext_only)
            nA  = len(aNrmGridNow)

            # Per-shock scalar weights (nSh,)
            _base_w  = (ShkPrbsNext_PermShk
                        * PermShkValsNext_only ** (-CRRA)
                        * PermGroFac ** (-CRRA)
                        * DiscFac * LivPrb)          # used for vP, vPD, LambdaM
            _base_wv = (ShkPrbsNext_PermShk
                        * PermShkValsNext_only ** (1.0 - CRRA)
                        * PermGroFac ** (1.0 - CRRA)
                        * DiscFac * LivPrb)           # used for v

            # Durable-survival coefficient per d-point (nD,)
            _d_surv  = (1 - dDepr * (1 - MaintCost)) * dNrmGridNow   # (nD,)
            _d_lam   = dNrmGridNow * (OpCost + dDepr * MaintCost)     # (nD,)

            if _GPU_AVAILABLE:
                # Build (nD, nSh, nA) query grids directly on GPU.
                # Saves ~85 ms CPU array construction + 44 MB H2D per period.
                # H2D cost for inputs: (nD + nD + nSh + nA + nA) × 8B ≈ 22 KB (negligible).
                _d_surv_eop_g = _cp.asarray(_d_surv)
                _d_lam_eop_g  = _cp.asarray(_d_lam)
                _shk_eop_g    = _cp.asarray(PermShkValsNext_only)
                _a_eop_g      = _cp.asarray(aNrmGridNow)
                _R_eop_g      = _cp.asarray(R_vec)
                n_flat_g = (_d_surv_eop_g[:, _cp.newaxis, _cp.newaxis]
                            / _shk_eop_g[_cp.newaxis, :, _cp.newaxis]
                            * _cp.ones((1, 1, nA), dtype=_cp.float64)).ravel()
                b_flat_g = ((_R_eop_g[_cp.newaxis, _cp.newaxis, :]
                             * _a_eop_g[_cp.newaxis, _cp.newaxis, :]
                             - _d_lam_eop_g[:, _cp.newaxis, _cp.newaxis])
                            / (PermGroFac * _shk_eop_g[_cp.newaxis, :, _cp.newaxis])).ravel()
            else:
                # CPU fallback: build 3-D arrays then flatten
                n_next_3d = (
                    _d_surv[:, np.newaxis, np.newaxis]
                    / PermShkValsNext_only[np.newaxis, :, np.newaxis]
                ) * np.ones((1, 1, nA))
                b_next_3d = (
                    (R_vec[np.newaxis, np.newaxis, :] * aNrmGridNow[np.newaxis, np.newaxis, :]
                     - _d_lam[:, np.newaxis, np.newaxis])
                    / (PermGroFac * PermShkValsNext_only[np.newaxis, :, np.newaxis])
                )
                b_flat_g = b_next_3d.ravel()   # named _g for unified reference below (CPU ndarray)
                n_flat_g = n_next_3d.ravel()

            if timeBOOL:
                _t['eop_grid'] = time.perf_counter() - _tic_eop
                _tic_eop_gpu = time.perf_counter()

            if _GPU_AVAILABLE:
                # ---- GPU path (~15x faster than CPU LinearFast) ----
                # Extract raw arrays from the LinearFast / MargValueFuncCRRA wrappers.
                # MargValueFuncCRRA (HARK.interpolation) stores its inv-interp
                # func as .cFunc; LinearFast stores grids as .grid_list, values
                # as .f_val.
                _lf_vP   = Interm_vPfunc_Integration.cFunc   # LinearFast (inv_vP)
                _lf_vPD  = Interm_vPDfunc_Integration.cFunc  # LinearFast (inv_vPD)
                _lf_vinv = Interm_inv_v_Interp_func           # LinearFast (inv_v)
                _bGrid_g = _cp.asarray(np.ascontiguousarray(_lf_vP.grid_list[0]))
                _nGrid_g = _cp.asarray(np.ascontiguousarray(_lf_vP.grid_list[1]))
                _fvP_g   = _cp.asarray(np.ascontiguousarray(_lf_vP.f_val))
                _fvPD_g  = _cp.asarray(np.ascontiguousarray(_lf_vPD.f_val))
                _fvinv_g = _cp.asarray(np.ascontiguousarray(_lf_vinv.f_val))
                # n_flat_g / b_flat_g already on GPU (built above)
                _b_g = b_flat_g
                _n_g = n_flat_g

                # Bilinear interpolation on GPU
                inv_vP_g   = _eval2d_batch_gpu(_bGrid_g, _nGrid_g, _fvP_g,   _b_g, _n_g)
                inv_vPD_g  = _eval2d_batch_gpu(_bGrid_g, _nGrid_g, _fvPD_g,  _b_g, _n_g)
                inv_vinv_g = _eval2d_batch_gpu(_bGrid_g, _nGrid_g, _fvinv_g, _b_g, _n_g)

                # Apply MargValueFuncCRRA (**-CRRA) and utility transforms on GPU
                vP_3d_g  = inv_vP_g.reshape(nD, nSh, nA) ** (-CRRA)
                vPD_3d_g = inv_vPD_g.reshape(nD, nSh, nA) ** (-CRRA)
                vinv_g   = _cp.clip(inv_vinv_g, 0.0, None)
                v_3d_g   = (vinv_g.reshape(nD, nSh, nA) ** (1.0 - CRRA)) / (1.0 - CRRA)

                # Weighted sums entirely on GPU
                w_g  = _cp.asarray(_base_w)[_cp.newaxis, :, _cp.newaxis]
                wv_g = _cp.asarray(_base_wv)[_cp.newaxis, :, _cp.newaxis]
                R_g  = _cp.asarray(R_vec)[_cp.newaxis, _cp.newaxis, :]
                if timeBOOL:
                    _cp.cuda.runtime.deviceSynchronize()
                    _t['eop_gpu_compute'] = time.perf_counter() - _tic_eop_gpu
                    _tic_eop_d2h = time.perf_counter()

                EndOfPrdvPOne_array = _cp.asnumpy(_cp.sum(w_g * R_g * vP_3d_g, axis=1))
                EndOfPrdvPTwo_array = _cp.asnumpy(_cp.sum(
                    w_g * (R_g + OpCost + MaintCost * dDepr) * vP_3d_g, axis=1))
                EndOfPrdvPD_array   = _cp.asnumpy(_cp.sum(
                    w_g * float(1 - dDepr * (1 - MaintCost)) * vPD_3d_g, axis=1))
                EndOfPrdv_array     = _cp.asnumpy(_cp.sum(wv_g * v_3d_g, axis=1))
                LambdaM_array       = _cp.asnumpy(_cp.sum(
                    w_g * float(OpCost + MaintCost * dDepr) * vP_3d_g, axis=1))
                inv_EndOfPrdv_array = utility_inv(EndOfPrdv_array, CRRA)
                if timeBOOL:
                    _t['eop_d2h'] = time.perf_counter() - _tic_eop_d2h
            else:
                # ---- CPU path ----
                vP_flat   = Interm_vPfunc_Integration(b_flat_g, n_flat_g)
                vPD_flat  = Interm_vPDfunc_Integration(b_flat_g, n_flat_g)
                vinv_flat = np.clip(Interm_inv_v_Interp_func(b_flat_g, n_flat_g), 0.0, None)
                v_flat    = utility(vinv_flat, CRRA)

                vP_3d  = vP_flat.reshape(nD, nSh, nA)
                vPD_3d = vPD_flat.reshape(nD, nSh, nA)
                v_3d   = v_flat.reshape(nD, nSh, nA)

                w  = _base_w[np.newaxis, :, np.newaxis]
                wv = _base_wv[np.newaxis, :, np.newaxis]

                EndOfPrdvPOne_array = np.sum(
                    w * R_vec[np.newaxis, np.newaxis, :] * vP_3d, axis=1)
                EndOfPrdvPTwo_array = np.sum(
                    w * (R_vec + OpCost + MaintCost * dDepr)[np.newaxis, np.newaxis, :] * vP_3d, axis=1)
                EndOfPrdvPD_array = np.sum(
                    w * (1 - dDepr * (1 - MaintCost)) * vPD_3d, axis=1)
                EndOfPrdv_array = np.sum(wv * v_3d, axis=1)
                LambdaM_array = np.sum(
                    w * (OpCost + MaintCost * dDepr) * vP_3d, axis=1)
                inv_EndOfPrdv_array = utility_inv(EndOfPrdv_array, CRRA)
            if timeBOOL:
                _t['eop_batch'] = time.perf_counter() - _tic_eop

    else:
        ### ii. Loop states
        # for i_d in range(len(nNrmGridNow)):
        for i_d in range(len(dNrmGridNow)):
            for ishock in range(iShock):
                n_plus = np.ones(len(aNrmGridNow)) * ((1 - dDepr * (1 - MaintCost)) * dNrmGridNow[i_d]) / (PermShkValsNext_only[ishock])
                m_plus = (R_vec * aNrmGridNow -  dNrmGridNow[i_d] * (OpCost + dDepr * MaintCost)) / (PermGroFac * PermShkValsNext_only[ishock]) + TranShkValsNext[ishock]

                # Use total solution. This already incorporates the optimal adjuster/keeper region
                vPFunc_plus_array = CRRAutilityP(cFuncNext(n_plus, m_plus), dFuncNext(n_plus, m_plus))
                inv_vPDFunc_plus_array = inv_vPDFuncNext(n_plus, m_plus)
                vPDFunc_plus_array = inv_vPDFunc_plus_array**-CRRA
                inv_vFunc_plus_array = inv_vFunc_next(n_plus, m_plus)
                vFunc_plus_array = utility(inv_vFunc_plus_array, CRRA)

                EndOfPrdvPOne_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (-CRRA) * \
                                    DiscFac * LivPrb * R_vec * PermGroFac ** (-CRRA) * vPFunc_plus_array # weighted post decision function
                EndOfPrdvPTwo_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (-CRRA) * \
                                    DiscFac * LivPrb * (R_vec + OpCost + dDepr * MaintCost) * PermGroFac ** (-CRRA) * vPFunc_plus_array # weighted post decision function
                EndOfPrdvPD_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (-CRRA) * \
                                     LivPrb * PermGroFac ** (-CRRA) * DiscFac * (1 - dDepr * (1 - MaintCost)) * vPDFunc_plus_array
                EndOfPrdv_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (1.0 - CRRA) * PermGroFac ** (
                            1.0 - CRRA) * DiscFac * LivPrb * vFunc_plus_array  # weighted value function
                LambdaM_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (-CRRA) * \
                                    DiscFac * LivPrb * (OpCost + dDepr * MaintCost) * PermGroFac ** (-CRRA) * vPFunc_plus_array # weighted post decision function
            inv_EndOfPrdv_array[i_d] = utility_inv(EndOfPrdv_array[i_d], CRRA)

    ### Inverses
    inv_EndOfPrdvPD_array = utilityP_inv(EndOfPrdvPD_array, CRRA)
    inv_LambdaM_array = utilityP_inv(LambdaM_array, CRRA)

    # vi. transform post decision value function
    # Make sure inv_EndOfPrdv_array is 0 when Natural Borrowing Constraint is binding.
    # Vectorised: compute per-row cutoff indices via searchsorted, then use a
    # boolean mask to set all constrained cells in one numpy pass.
    _idx_zero_vec = np.searchsorted(aNrmGridNow, BoroCnstNat2D, side='left')  # (nD,)
    _col_idx = np.arange(len(aNrmGridNow))                                      # (nA,)
    _mask_cnst = _col_idx[np.newaxis, :] <= _idx_zero_vec[:, np.newaxis]        # (nD, nA)
    inv_EndOfPrdv_array[_mask_cnst]  = 0.0
    EndOfPrdvPOne_array[_mask_cnst]  = np.inf
    EndOfPrdvPTwo_array[_mask_cnst]  = np.inf
    EndOfPrdvPD_array[_mask_cnst]    = np.inf
    inv_EndOfPrdvPD_array[_mask_cnst] = 0.0
    if (OpCost + dDepr * MaintCost) > 0.0:
        LambdaM_array[_mask_cnst]    = np.inf
        inv_LambdaM_array[_mask_cnst] = 0.0
    else:
        LambdaM_array[:, :]    = 0.0       # infinity × 0 → nan; use 0 directly
        inv_LambdaM_array[:, :] = np.inf


    inv_EndOfPrdv = LinearFast(inv_EndOfPrdv_array, [dNrmGridNow, aNrmGridNow])
    EndOfPrdv = ValueFuncCRRA(inv_EndOfPrdv, CRRA)
    if alpha == 1 or dDepr == 1:
        EndOfPrdvPD_array = np.zeros(post_shape)

    # ix. Interpolate and make functions
    EndOfPrdvPOne = LinearFast(EndOfPrdvPOne_array, [dNrmGridNow, aNrmGridNow])
    EndOfPrdvPTwo = LinearFast(EndOfPrdvPTwo_array, [dNrmGridNow, aNrmGridNow])
    EndOfPrdvPD = LinearFast(EndOfPrdvPD_array, [dNrmGridNow, aNrmGridNow])
    LambdaM = LinearFast(LambdaM_array, [dNrmGridNow, aNrmGridNow])
    inv_EndOfPrdvPD = LinearFast(inv_EndOfPrdvPD_array, [dNrmGridNow, aNrmGridNow])
    inv_LambdaM = LinearFast(inv_LambdaM_array, [dNrmGridNow, aNrmGridNow])

    # Check for Nan's
    if np.isnan(EndOfPrdvPOne_array).any():
        print("Nan in EndOfPrdvPOne")
        Errors = 1
    if np.isnan(EndOfPrdvPTwo_array).any():
        print("Nan in EndOfPrdvPTwo")
        Errors = 1
    if np.isnan(inv_EndOfPrdv_array).any():
        print("NaN in EndOfPrdv")
        Errors = 1
    if np.isnan(EndOfPrdvPD_array).any():
        print("Nan in EndOfPrdvPD")
        Errors = 1
    if np.isnan(LambdaM_array).any():
        print("Nan in LambdaM")
        Errors = 1
    # # ### if non nan maximum is larger than zero
    # # if np.nanmax(LambdaM_array) > 0:
    # #     print("Warning: LambdaM_array has non-nan maximum greater than zero")
    # #     Errors = 1
    # # Check for kink at the last point
    # if (EndOfPrdvPOne_array[:,-1] > EndOfPrdvPOne_array[:,-2]).any():
    #     print("EndOfPrdvPOne is increasing at last point: increase aXtraGrid")
    #     Errors = 1
    # # Check for kink at the last point
    # if (EndOfPrdvPOne_array[-1,:] > EndOfPrdvPOne_array[-2,:]).any():
    #     print("EndOfPrdvPOne is increasing at last point: d direction")
    #     Errors = 1

    #     ### print i_n for which this is the case
    #     for i_m in range(len(EndOfPrdvPOne_array)):
    #         if (EndOfPrdvPOne_array[-1, i_m] > EndOfPrdvPOne_array[-2, i_m]):
    #             print(f"EndOfPrdvPOne is increasing at last point: increase aXtraGrid for i_m={i_m}")

    if timeBOOL:
        _t['eop_total'] = time.perf_counter() - tic
        print("End of Period took: {:.3f} s".format(_t['eop_total']))

    if timeBOOL:
        tic = time.perf_counter()


    # # ########################################################################################################
    # if alpha < 1:
    #     ### Solve Frictionless (perfect-foresight limit) model for non-terminal period
    #     # Optimal c/d ratio with no frictions (gam defined at line ~5008 above):
    #     #   gam = alpha/(1-alpha) * (1 + (dDepr + OpCost - 1)/Rfree)
    #     # The chi (MaintCost) terms cancel in the ell_{t+1} transition, so the
    #     # effective durable cost factor and Lambda use OpCost, NOT MaintCost.
    #     #
    #     # EGM via marginal-value Euler equation (see EGM step below).
    #     if timeBOOL:
    #         tic = time.time()

    #     ### End-of-period net-wealth grid
    #     # w_t = a_t + d_t*(1 - dDepr - OpCost)/Rfree  (chi terms cancel)
    #     wXtraGrid = construct_grid(aXtraGrid[0],
    #                                aXtraGrid[-1] + dNrmGridNow[-1] * (1 - dDepr - OpCost) / Rfree,
    #                                100, grid_type, NestFac)
    #     wNrmGridNow = wXtraGrid + BoroCnstNat1D

    #     ### Beginning-of-next-period resources: ell_{t+1} = R*w_t + 1
    #     xNrmGridNext_Frictionless = wNrmGridNow * Rfree + 1

    #     ### EGM step via marginal-value Euler equation
    #     # The Euler equation for total spending q = c+d holds without correction only
    #     # when gam is the same across periods. At the terminal boundary gam changes
    #     # (gam_T uses the LTV-based p_d; gam_{non-T} uses the interior FOC), so the
    #     # q-based recursion carries a large error.  Instead, use the marginal-value
    #     # representation: u_c = beta*R * v'_{t+1}(ell_{t+1}) and invert via
    #     #   u_c = K_lambda * x^{-rho},  K_lambda = alpha * gam^{alpha-1}
    #     # to recover the Cobb-Douglas composite x = c^alpha * d^{1-alpha}.
    #     # This automatically accounts for any gam discontinuity at the boundary.
    #     K_lambda = alpha * gam ** (alpha - 1)
    #     vP_next_array = DiscFac * LivPrb * Rfree * vPFunc_Frictionless_Next(xNrmGridNext_Frictionless)
    #     x_array = (K_lambda / vP_next_array) ** (1.0 / CRRA)

    #     ### From composite x = gam^alpha * d, recover c and d using gam_t (non-terminal)
    #     dFunc_Frictionless_array = x_array / gam ** alpha
    #     cFunc_Frictionless_array = gam * dFunc_Frictionless_array
    #     qFunc_Frictionless_array = cFunc_Frictionless_array + dFunc_Frictionless_array

    #     ### Endogenous grid: ell_t = q_t * Lambda + w_t
    #     # Lambda = (gam + 1 + (dDepr + OpCost - 1)/Rfree) / (1 + gam)
    #     Lambda_Frictionless = (gam + 1 + (dDepr + OpCost - 1) / Rfree) / (1 + gam)
    #     xNrmGrid_Frictionless = qFunc_Frictionless_array * Lambda_Frictionless + wNrmGridNow

    #     ### Add a zero point at the natural borrowing constraint
    #     qFunc_Frictionless_array = np.append(0.0, qFunc_Frictionless_array)
    #     dFunc_Frictionless_array = np.append(0.0, dFunc_Frictionless_array)
    #     cFunc_Frictionless_array = np.append(0.0, cFunc_Frictionless_array)
    #     xNrmGrid_Frictionless    = np.append(BoroCnstNat1D, xNrmGrid_Frictionless)

    #     ### Interpolate policy functions
    #     qFunc_Frictionless = LinearFast(qFunc_Frictionless_array, [xNrmGrid_Frictionless])
    #     cFunc_Frictionless = LinearFast(cFunc_Frictionless_array, [xNrmGrid_Frictionless])
    #     dFunc_Frictionless = LinearFast(dFunc_Frictionless_array, [xNrmGrid_Frictionless])

    #     ### Value function: v = u(c,d) + beta*zeta*v_{t+1}(ell_{t+1})
    #     # Evaluate on the exogenous xNrmGridNow (skip the zero point, prepend 0)
    #     c_aux     = cFunc_Frictionless(xNrmGridNow[1:])
    #     d_aux     = dFunc_Frictionless(xNrmGridNow[1:])
    #     a_aux     = xNrmGridNow[1:] - c_aux - d_aux
    #     nNext_aux = d_aux * (1 - dDepr * (1 - MaintCost))
    #     mNext_aux = a_aux * Rfree - d_aux * (OpCost + dDepr * MaintCost) + 1
    #     EndOfPrdv_Frictionless_aux = DiscFac * LivPrb * vFunc_Frictionless_Next(nNext_aux + mNext_aux)

    #     vFunc_Frictionless_array     = CRRAutility(c_aux, d_aux) + EndOfPrdv_Frictionless_aux
    #     inv_vFunc_Frictionless_array = utility_inv(vFunc_Frictionless_array, CRRA)
    #     inv_vFunc_Frictionless_array = np.append(0.0, inv_vFunc_Frictionless_array)

    #     inv_vFunc_Frictionless = LinearFast(inv_vFunc_Frictionless_array, [xNrmGridNow])
    #     vFunc_Frictionless     = ValueFuncCRRA(inv_vFunc_Frictionless, CRRA)
    #     # MargValueFuncCRRA uses the same inv_v representation as ValueFuncCRRA
    #     vPFunc_Frictionless    = MargValueFuncCRRA(inv_vFunc_Frictionless, CRRA)

    #     if timeBOOL:
    #         toc = time.time()
    #         print("Frictionless End of Period took: {:.2f} seconds".format(toc - tic))

    # else:
    #     qFunc_Frictionless  = qFunc_Frictionless_Next
    #     vFunc_Frictionless  = vFunc_Frictionless_Next
    #     vPFunc_Frictionless = vPFunc_Frictionless_Next

    ######################################################################################################################
    ### Keeper and Adjuster Separated
    if timeBOOL:
        tic = time.perf_counter()
    if (adjC != 0 or SearchCost != 0) or alpha == 1:
        ### Preparations
        # Keeper
        cFuncKeepList = []

        # Note: dNrmGridNow and nNrmGridNow are the same object; the keeper retains the
        # current durable stock, so d = n.  The loop below therefore iterates over
        # initial durable holdings even though the grid is labelled "d".
        # Parallelised over the n-grid using threads (numpy/numba release the GIL).
        vTransf = lambda x: np.exp(x)   # value transformation used by DCEGM

        def _one_keeper_row(i_n, n_aux):
            n_aux_array = np.ones(len(aNrmGridNow), dtype=np.float64) * n_aux
            EndOfPrdvPNow = EndOfPrdvPOne_array[i_n]
            EndOfPrdvNow  = EndOfPrdv_array[i_n]

            cKeep_egm = CRRAutilityP_inv_njit(EndOfPrdvPNow, n_aux_array, alpha, d_ubar, CRRA)
            mKeep_egm = aNrmGridNow + cKeep_egm
            vKeep_egm = CRRAutility_njit(cKeep_egm, n_aux_array, alpha, d_ubar, CRRA) + EndOfPrdvNow

            if UpperEnvelope == "DCEGM":
                vKeept_egm = vTransf(vKeep_egm)
                start, end = calc_nondecreasing_segments(mKeep_egm, vKeept_egm)
                segments, m_segments, c_segments = [], [], []
                for j in range(len(start)):
                    idx = range(start[j], end[j] + 1)
                    segments.append([mKeep_egm[idx], vKeept_egm[idx]])
                    m_segments.append(mKeep_egm[idx])
                    c_segments.append(cKeep_egm[idx])
                mKeep_upper, _, inds_upper = upper_envelope(segments)
                cKeep_upper = np.zeros_like(mKeep_upper) + np.nan
                for k, c_segm in enumerate(c_segments):
                    if start[-1] + 1 == len(mKeep_egm):
                        c_Interp = LinearFast(c_segments[0], [m_segments[0]])
                        cKeep_upper = c_Interp(mKeep_upper)
                    elif len(m_segments[k]) == 1:
                        cKeep_upper[inds_upper == k] = c_segm
                    else:
                        cKeep_upper[inds_upper == k] = LinearFast(c_segm, [m_segments[k]])(mKeep_upper[inds_upper == k])
            elif UpperEnvelope == "FUES":
                mKeep_upper, vKeep_upper, cKeep_upper, aKeep_upper, _ \
                    = FUES(mKeep_egm, vKeep_egm, cKeep_egm, aNrmGridNow)
            elif UpperEnvelope == "FUESBonn":
                mKeep_upper, vKeep_upper, cKeep_upper \
                    = fast_upper_envelope(endog_grid=mKeep_egm, value=vKeep_egm,
                                          policy=cKeep_egm, exog_grid=aNrmGridNow,
                                          jump_thresh=1, lower_bound_wealth=1e-10)

            if cKeep_upper[0] != 0.0:
                c_interp = np.insert(cKeep_upper, 0, 0.0, axis=-1)
                m_interp = np.insert(mKeep_upper, 0, BoroCnstNat2D[i_n], axis=-1)
            else:
                c_interp = cKeep_upper
                m_interp = mKeep_upper
            return c_interp, m_interp

        _keeper_rows = Parallel(n_jobs=-1, backend='threading')(
            delayed(_one_keeper_row)(i_n, n_aux)
            for i_n, n_aux in enumerate(dNrmGridNow)
        )
        cFuncKeepList = [LinearFast(c, [m]) for c, m in _keeper_rows]

        ### Post
        ### Keeper:
        mNrmGridKeeper = mNrmGridNow
        # Create cFuncKeep and dFuncKeep
        # Create empty container
        keep_shape = (len(nNrmGridNow), len(mNrmGridKeeper))
        cFuncKeepUnc_array = np.zeros(keep_shape)
        dFuncKeepUnc_array = np.zeros(keep_shape)

        for i_n in range(len(nNrmGridNow)):
            dFuncKeepUnc_array[i_n] = np.ones(len(mNrmGridKeeper)) * nNrmGridNow[i_n]
            cFuncKeepUnc_array[i_n] = cFuncKeepList[i_n](mNrmGridKeeper)
            ### Make sure cFunc is not negative
            cFuncKeepUnc_array[i_n] = np.maximum(cFuncKeepUnc_array[i_n], 0.0)
        
        ### Make Functions
        cFuncKeepUnc = LinearFast(cFuncKeepUnc_array, [nNrmGridNow, mNrmGridKeeper])
        dFuncKeepUnc = LinearFast(dFuncKeepUnc_array, [nNrmGridNow, mNrmGridKeeper])

        # Make Consumption function similar to use_points_for_interpolation
        # Constraint part: From minimum m onwards, consume everything.
        if np.array(mNrmMinNow).size == 1:
            # LTV == 0: constraint is c = m (consume everything at the floor).
            # The fixed [0,1] grid is safe because LinearFast extrapolates linearly,
            # giving c(n, m) = m for all (n, m) outside [0,1].
            cFuncKeepCnst = LinearFast(np.array([[0.0,1.0],[0.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])

        else:
            mNrmMinNow_aux = np.append(np.sort(mNrmMinNow), np.max([mNrmMinNow]) + 1)
            nN = len(nNrmGridNow)
            nM = len(mNrmMinNow_aux)
            cFuncKeepCnst_array = np.zeros((nN, nM))
            if LTV == 0:
                # Every row is identical: cumsum of abs-diff from index 0
                _cCnst = np.cumsum(np.abs(np.diff(mNrmMinNow_aux)))
                cFuncKeepCnst_array[:, 1:] = _cCnst[np.newaxis, :]
            else:
                # Row i_n starts at start_idx = nN - i_n (triangular fill).
                # Original: cCnst[i_n, start_idx+j] = cumsum(absdiff(slice[start_idx-1:]))[j]
                # = sum_{k=start_idx-1}^{start_idx-1+j} |diff_k|
                # Use a zero-padded cumsum so each row's offset is subtracted correctly.
                _absdiff = np.abs(np.diff(mNrmMinNow_aux))
                _cumsum_padded = np.concatenate([[0.0], np.cumsum(_absdiff)])
                for i_n in range(nN):
                    start_idx = nN - i_n
                    cFuncKeepCnst_array[i_n, start_idx:] = (
                        _cumsum_padded[start_idx:] - _cumsum_padded[start_idx - 1])

            cFuncKeepCnst = LinearFast(cFuncKeepCnst_array, [nNrmGridNow, mNrmMinNow_aux])

        cFuncKeep = LowerEnvelope2D(cFuncKeepUnc, cFuncKeepCnst, nan_bool=False)
        dFuncKeep = dFuncKeepUnc
        vPFuncKeep = MargValueFuncCRRA_dur(cFuncKeep, dFuncKeep, alpha, d_ubar, CRRA)

        ### VALUE FUNCTIONS
        ### Keeper — parallelised over n-grid rows (threads release GIL for numpy)
        vFunc_shape = (len(nNrmGridNow), len(mNrmGridNow))
        inv_vFuncKeep_array = np.zeros(vFunc_shape)
        _mNrmMinNow_size = np.array(mNrmMinNow).size

        def _one_vfunc_row(i_n, n_aux):
            start_idx = (len(mNrmMinNow) - i_n) if _mNrmMinNow_size > 1 else 0
            n_aux_arr = np.ones(len(mNrmGridNow[start_idx:])) * n_aux
            m_aux_arr = mNrmGridNow[start_idx:]
            cNrm = np.clip(cFuncKeep(n_aux_arr, m_aux_arr), 0.0, None)
            aNrm = np.maximum(m_aux_arr - cNrm, BoroCnstNat2D[i_n])
            vNrm = CRRAutility(cNrm, n_aux_arr) + EndOfPrdv(n_aux_arr, aNrm)
            return start_idx, utility_inv(vNrm, CRRA)

        _vfunc_rows = Parallel(n_jobs=-1, backend='threading')(
            delayed(_one_vfunc_row)(i_n, n_aux)
            for i_n, n_aux in enumerate(nNrmGridNow)
        )
        for i_n, (start_idx, inv_v_row) in enumerate(_vfunc_rows):
            inv_vFuncKeep_array[i_n, start_idx:] = inv_v_row

        inv_vFuncKeep = LinearFast(inv_vFuncKeep_array, [nNrmGridNow, mNrmGridNow])
        vFuncKeep = ValueFuncCRRA(inv_vFuncKeep, CRRA)
        
        ### check if below zero
        if np.any(inv_vFuncKeep_array < 0):
            print("inv_vFuncKeep_array is below zero")
            Errors = 1
    if timeBOOL:
        _t['keeper'] = time.perf_counter() - tic
        print("Keeper took: {:.3f} s".format(_t['keeper']))
    ##########################################################################################################################
    ##########################################################################################################################
    ### Solve Adjuster Problem
    """
    First: Solve Unconstraint part of adjuster problem. This step includes FUES as FOC are neccessary, but not sufficient.
    Second: Solve Constraint part of adjuster problem.
    Third: Compare the two solutions and use whichever has the highest value.
    Fourth: Decay Interpolation for frictionless PF solution.
    """
    if timeBOOL:
        tic = time.perf_counter()
    if alpha < 1:
        # adjuster part
        ### Preparations
        a_min = np.min(aNrmGridNow) #np.min(aXtraGrid) 
        a_max = np.max(aNrmGridNow) #np.max(aXtraGrid)
        N = 1000 #Numbers of Roots
        aAdj_array = []
        xAdj_array = []
        cAdj_array = []
        dAdj_array = []
        vAdj_array = []

        ### Solve adjuster for a grid which is shorter than the grid we use to calculate end of Period functions.
        ### This is necessary because the adjuster problem is not well defined for points at the end of the grid.
        # dNrmAdjMax = nNrmGridNow[-1] - 1 #2 #nNrmGridNow[-1] / ((1 - dDepr)/np.min(PermShkValsNext)+0.01)
        ### Take maximum of PF solution as maximum point
        dNrmAdjMax = nNrmGridNow[-1] #d_PF_max #20 #p2
        dNrmAdjMin = nNrmGridNow[0]
        dNrmGridAdj = construct_grid(dNrmAdjMin, dNrmAdjMax,  len(xNrmGrid), grid_type, NestFac)

        ### Unconstraint part — GPU scan + GPU bisection when available,
        ### otherwise joblib threads with scipy root_scalar (baseline-identical).
        if _GPU_AVAILABLE:
            # ---- GPU scan + GPU bisection ----
            # 1) Build per-d scan grids on GPU (each row offset by a_min_d[i]).
            _base_scan = np.asarray(construct_grid(0.0, a_max, N, 'exp_mult', 1))  # (N,)
            _a_min_pd  = np.interp(dNrmGridAdj, dNrmGridNow, BoroCnstNat2D)        # (nD,)
            _dAdj_g    = _cp.asarray(np.ascontiguousarray(dNrmGridAdj))             # (nD,)
            _base_g    = _cp.asarray(np.ascontiguousarray(_base_scan))              # (N,)
            _amin_g    = _cp.asarray(np.ascontiguousarray(_a_min_pd))               # (nD,)
            _a_mat_g   = _amin_g[:, None] + _base_g[None, :]                       # (nD, N)
            _d_mat_g   = _dAdj_g[:, None] * _cp.ones((1, N), dtype=_cp.float64)    # (nD, N)

            # Upload EoP arrays (already numpy by this point)
            _dGrid_adj_g = _cp.asarray(np.ascontiguousarray(dNrmGridNow))
            _aGrid_adj_g = _cp.asarray(np.ascontiguousarray(aNrmGridNow))
            _eop1_g_adj  = _cp.asarray(np.ascontiguousarray(EndOfPrdvPOne_array))
            _eop2_g_adj  = _cp.asarray(np.ascontiguousarray(EndOfPrdvPTwo_array))
            _eopd_g_adj  = _cp.asarray(np.ascontiguousarray(EndOfPrdvPD_array))
            _eopv_g_adj  = _cp.asarray(np.ascontiguousarray(inv_EndOfPrdv_array))

            # 2) Evaluate FOC residual on GPU for all (nD × N) points.
            _f_g = _gpu_foc_residual(
                _d_mat_g.ravel(), _a_mat_g.ravel(),
                _dGrid_adj_g, _aGrid_adj_g,
                _eop1_g_adj, _eop2_g_adj, _eopd_g_adj,
                alpha, d_ubar, CRRA
            ).reshape(len(dNrmGridAdj), N)                              # (nD, N)

            # 3) Sign-change detection (NaN → 0 as in find_roots_in_interval).
            _f_scan_g = _cp.where(_cp.isnan(_f_g), 0.0, _f_g)
            _signs_g  = _cp.sign(_f_scan_g)
            _brack_mask_g = _signs_g[:, :-1] != _signs_g[:, 1:]        # (nD, N-1)
            _row_idx_g, _col_idx_g = _cp.where(_brack_mask_g)          # flat bracket indices

            if len(_row_idx_g) > 0:
                # 4) Spurious-bracket filter: both endpoints must be finite and
                #    sign(f_lo)*sign(f_hi) < 0 (same logic as find_roots_in_interval).
                _f_lo_g  = _f_g[_row_idx_g, _col_idx_g]
                _f_hi_g  = _f_g[_row_idx_g, _col_idx_g + 1]
                _valid_b = (_cp.isfinite(_f_lo_g) & _cp.isfinite(_f_hi_g)
                            & (_f_lo_g * _f_hi_g < 0))
                _row_v   = _row_idx_g[_valid_b]
                _col_v   = _col_idx_g[_valid_b]

                if len(_row_v) > 0:
                    # 5) GPU bisection over all valid brackets simultaneously.
                    _d_br_g  = _dAdj_g[_row_v]                         # (B,)
                    _alo_g   = _a_mat_g[_row_v, _col_v]                 # (B,)
                    _ahi_g   = _a_mat_g[_row_v, _col_v + 1]             # (B,)
                    _flos_g  = _cp.sign(_f_g[_row_v, _col_v])           # (B,)
                    _a_rt_g  = _gpu_bisect_all(
                        _d_br_g, _alo_g, _ahi_g, _flos_g,
                        _dGrid_adj_g, _aGrid_adj_g,
                        _eop1_g_adj, _eop2_g_adj, _eopd_g_adj,
                        alpha, d_ubar, CRRA, n_iter=60
                    )                                                   # (B,)

                    # 6) Validity check: a_root >= agoal_root - 1e-8.
                    _agoal_g = (-BoroCnstArt
                                - _d_br_g * (1 - dDepr*(1-MaintCost) - OpCost)*(1-adjC)*LTV)
                    _valid_r = _a_rt_g >= _agoal_g - 1e-8
                    _a_rt_g  = _a_rt_g[_valid_r]
                    _d_r_g   = _d_br_g[_valid_r]

                    if len(_a_rt_g) > 0:
                        # 7) Compute c, x, v on GPU.
                        _eop1_rt = _eval2d_batch_gpu(
                            _dGrid_adj_g, _aGrid_adj_g, _eop1_g_adj, _d_r_g, _a_rt_g)
                        _pow_c   = 1.0 / (alpha * (1.0 - CRRA) - 1.0)
                        _c_r_g   = (_eop1_rt
                                    / (alpha * (_d_r_g + d_ubar)**((1-alpha)*(1-CRRA)))
                                    ) ** _pow_c
                        _x_r_g   = _a_rt_g + _d_r_g + _c_r_g
                        _u_in_g  = _c_r_g**alpha * (_d_r_g + d_ubar)**(1.0-alpha)
                        _cu_g    = _u_in_g**(1.0-CRRA) / (1.0-CRRA)
                        _inv_ev  = _eval2d_batch_gpu(
                            _dGrid_adj_g, _aGrid_adj_g, _eopv_g_adj, _d_r_g, _a_rt_g)
                        _ev_g    = _inv_ev**(1.0-CRRA) / (1.0-CRRA)
                        _v_r_g   = _cu_g + _ev_g

                        # 8) Transfer to CPU and accumulate.
                        aAdj_array = np.append(aAdj_array, _cp.asnumpy(_a_rt_g))
                        cAdj_array = np.append(cAdj_array, _cp.asnumpy(_c_r_g))
                        xAdj_array = np.append(xAdj_array, _cp.asnumpy(_x_r_g))
                        dAdj_array = np.append(dAdj_array, _cp.asnumpy(_d_r_g))
                        vAdj_array = np.append(vAdj_array, _cp.asnumpy(_v_r_g))
        else:
            # ---- CPU path: joblib threads + scipy root_scalar ----
            def _one_adj_row(d_aux):
                a_min_d = np.interp(d_aux, dNrmGridNow, BoroCnstNat2D)
                a_root = find_roots_in_interval(rootfinding_function, a_min_d, a_max, d_aux, N)
                if a_root.size == 0:
                    return None
                d_0 = np.ones_like(a_root) * d_aux
                agoal_root = -BoroCnstArt - d_0 * (1 - dDepr*(1-MaintCost) - OpCost)*(1-adjC)*LTV
                valid_mask = a_root >= agoal_root - 1e-8
                if not np.any(valid_mask):
                    return None
                a_root = a_root[valid_mask]
                d_0    = d_0[valid_mask]
                c_0    = CRRAutilityP_inv_njit(EndOfPrdvPOne(d_0, a_root), d_0, alpha, d_ubar, CRRA)
                x_0    = a_root + d_0 + c_0
                v_0    = CRRAutility(c_0, d_0) + EndOfPrdv(d_0, a_root)
                return a_root, c_0, x_0, d_0, v_0

            _adj_results = Parallel(n_jobs=-1, backend='threading')(
                delayed(_one_adj_row)(d_aux) for d_aux in dNrmGridAdj
            )
            for res in _adj_results:
                if res is None:
                    continue
                a_root, c_0, x_0, d_0, v_0 = res
                aAdj_array = np.append(aAdj_array, a_root)
                cAdj_array = np.append(cAdj_array, c_0)
                xAdj_array = np.append(xAdj_array, x_0)
                dAdj_array = np.append(dAdj_array, d_0)
                vAdj_array = np.append(vAdj_array, v_0)

        xAdj_upper, vAdj_upper, cAdj_upper, dAdj_upper, aAdj_upper, _ \
            = FUES_allPol(xAdj_array, vAdj_array, cAdj_array, dAdj_array, aAdj_array, b=BoroCnstNat1D) #, m_bar=1.5)
        inv_vAdj_upper = utility_inv(vAdj_upper, CRRA)

        print("Number of Roots:", len(xAdj_array))
        print("Number of Kinks:", len(xAdj_array) - len(xAdj_upper))

        ### Round to 1e-3 to make it vary less
        xAdj_upper[0] = np.round(xAdj_upper[0], decimals=3)
        dAdj_upper[0] = np.round(dAdj_upper[0], decimals=3)
        aAdj_upper[0] = np.round(aAdj_upper[0], decimals=3)
        cAdj_upper[0] = np.round(cAdj_upper[0], decimals=3)
        inv_vAdj_upper[0] = np.round(inv_vAdj_upper[0], decimals=3)

        ### Add zero point: Natural Borrowing Constraint without durables (no non-durable nor durable consumption)
        xAdj_upper = np.append(BoroCnstNat1D, xAdj_upper)
        dAdj_upper = np.append(0, dAdj_upper)
        cAdj_upper = np.append(0, cAdj_upper)
        aAdj_upper = np.append(BoroCnstNat1D, aAdj_upper)
        inv_vAdj_upper = np.append(0, inv_vAdj_upper)

        ### Make Function
        cFuncAdjUnc = LinearFast(cAdj_upper, [xAdj_upper])
        dFuncAdjUnc = LinearFast(dAdj_upper, [xAdj_upper])
        inv_vFuncAdjUnc = LinearFast(inv_vAdj_upper, [xAdj_upper])
        aFuncAdjUnc = LinearFast(aAdj_upper, [xAdj_upper])
   
        # Find the kink point where the collateral constraint just binds.
        # Widen the bracket by +10 until a sign change is found or we have
        # exhausted the entire range of valid roots.  If no sign change exists
        # (constraint never or always binding), fall through to the fallback.
        _xkink_args = (dFuncAdjUnc, aFuncAdjUnc, BoroCnstArt, LTV, dDepr, adjC, MaintCost, OpCost)
        _bracket_upper = 10.0
        _bracket_max = float(xAdj_upper[-1]) + 10.0
        solution = None
        while _bracket_upper <= _bracket_max:
            try:
                _sol = sp_optimize.root_scalar(
                    find_xkink,
                    args=_xkink_args,
                    bracket=[BoroCnstNat1D, _bracket_upper],
                    method='bisect'
                )
                solution = _sol
                break
            except ValueError:
                _bracket_upper += 10.0
        if solution is None:
            class _FakeSol:
                converged = False
            solution = _FakeSol()
        if not solution.converged:
            print("Root finding did not converge. Check the function and the parameters.")

            xAdj_comb_array = np.insert(xAdj_upper, 0, xNrmMinNow)
            dFuncAdj_comb_array = np.insert(dAdj_upper, 0, 0)
            cFuncAdj_comb_array = np.insert(cAdj_upper, 0, 0)
            aFuncAdj_comb_array = np.insert(aAdj_upper, 0, BoroCnstNat1D)
            inv_vAdj_comb_array = np.insert(inv_vAdj_upper, 0, 0)

            x0 = xAdj_upper[0]
            d0 = dAdj_upper[0]
            a0 = aAdj_upper[0]
            c0 = cAdj_upper[0]
            inv_v0 = inv_vAdj_upper[0]

        else:
            x0 = solution.root
            d0 = dFuncAdjUnc(x0)
            a0 = aFuncAdjUnc(x0)
            c0 = cFuncAdjUnc(x0)
    
            ### Calculate inv_v0
            inv_v0 = utility_inv(CRRAutility(c0, d0) + EndOfPrdv(d0, a0), CRRA)

            # ### Round to 1e-3
            # x0 = np.round(x0, decimals=3)
            # d0 = np.round(d0, decimals=3)
            # a0 = np.round(a0, decimals=3)
            # c0 = np.round(c0, decimals=3)
            # inv_v0 = np.round(inv_v0, decimals=3)


            ### If x0 is smaller than xAdj_upper[0], add kink point and zero point
            if x0 < xAdj_upper[0]:
                ### Add kink point and 0 point
                xAdj_comb_array = np.insert(xAdj_upper, 0, [xNrmMinNow, x0])
                dFuncAdj_comb_array = np.insert(dAdj_upper, 0, [0, d0])
                cFuncAdj_comb_array = np.insert(cAdj_upper, 0, [0, c0])
                aFuncAdj_comb_array = np.insert(aAdj_upper, 0, [BoroCnstNat1D, a0])
                inv_vAdj_comb_array = np.insert(inv_vAdj_upper, 0, [0, inv_v0])

            else:
                ### If x0 is larger than xAdj_upper[0], add only zero point. 
                ### Find the index at which x0 is smaller and xAdj_upper and use only these points.
                idx = np.where(xAdj_upper >= x0)[0][0]
                xAdj_comb_array = np.insert(xAdj_upper[idx:], 0, [xNrmMinNow, x0])
                dFuncAdj_comb_array = np.insert(dAdj_upper[idx:], 0, [0, d0])
                cFuncAdj_comb_array = np.insert(cAdj_upper[idx:], 0, [0, c0])
                aFuncAdj_comb_array = np.insert(aAdj_upper[idx:], 0, [BoroCnstNat1D, a0])
                inv_vAdj_comb_array = np.insert(inv_vAdj_upper[idx:], 0, [0, inv_v0])

        
        ### Check if c, d is decreasing at last grid-point
        while (cFuncAdj_comb_array[-1] < cFuncAdj_comb_array[-2]) or (dFuncAdj_comb_array[-1] < dFuncAdj_comb_array[-2]) or (inv_vAdj_comb_array[-1] < inv_vAdj_comb_array[-2]):
            cFuncAdj_comb_array = cFuncAdj_comb_array[:-1]
            xAdj_comb_array = xAdj_comb_array[:-1]
            dFuncAdj_comb_array = dFuncAdj_comb_array[:-1]
            aFuncAdj_comb_array = aFuncAdj_comb_array[:-1]
            inv_vAdj_comb_array = inv_vAdj_comb_array[:-1]

        ### Make Functions
        cFuncAdj = LinearFast(cFuncAdj_comb_array, [xAdj_comb_array])
        dFuncAdj = LinearFast(dFuncAdj_comb_array, [xAdj_comb_array])
        aFuncAdj = LinearFast(aFuncAdj_comb_array, [xAdj_comb_array])

        ########################################################
        #### Make Value Function on exogenous grid
        ###########
        ### Value Function: evaluate policies on a uniform exogenous grid to obtain
        ### a well-spaced interpolant (the endogenous xAdj_comb_array can be irregular).
        xNrmGridvFunc = construct_grid(xNrmMinNow, np.max(xAdj_comb_array), len(xNrmGridNow), grid_type, NestFac)
        c_aux = cFuncAdj(xNrmGridvFunc)
        d_aux = dFuncAdj(xNrmGridvFunc)
        a_aux = xNrmGridvFunc - d_aux - c_aux
        vAdj_aux = CRRAutility(c_aux, d_aux) + EndOfPrdv(d_aux, a_aux)
        inv_vAdj_aux_array = utility_inv(vAdj_aux, CRRA)

        ### Check if c, d is decreasing at last grid-point
        while (inv_vAdj_aux_array[-1] < inv_vAdj_aux_array[-2]):
            inv_vAdj_aux_array = inv_vAdj_aux_array[:-1]
            xNrmGridvFunc = xNrmGridvFunc[:-1]
            print("Delte last grid point of inv_vAdj_aux_array")
        inv_vFuncAdj = LinearFast(inv_vAdj_aux_array, [xNrmGridvFunc])

        if timeBOOL:
            _t['adjuster'] = time.perf_counter() - tic
            print("Adjuster took: {:.3f} s".format(_t['adjuster']))
        if timeBOOL:
            tic = time.perf_counter()

        vPFuncAdj = MargValueFuncCRRA_dur(cFuncAdj, dFuncAdj, alpha, d_ubar, CRRA)
        vFuncAdj = ValueFuncCRRA(inv_vFuncAdj, CRRA)
        inv_vFuncAdj_SOL = inv_vFuncAdj

        if timeBOOL:
            _t['adjuster_limit'] = time.perf_counter() - tic
            print("Adding Adjuster Limit took: {:.3f} s".format(_t['adjuster_limit']))
    ############################################################################################
    ############################################################################################
    if alpha < 1 and Limit==True:
        ### Solve Frictionless (perfect-foresight limit) model for non-terminal period
        # Optimal c/d ratio with no frictions (gam defined at line ~5008 above):
        #   gam = alpha/(1-alpha) * (1 + (dDepr + OpCost - 1)/Rfree)
        # The chi (MaintCost) terms cancel in the ell_{t+1} transition, so the
        # effective durable cost factor and Lambda use OpCost, NOT MaintCost.
        #
        # EGM via marginal-value Euler equation (see EGM step below).
        ### End-of-period net-wealth grid
        # w_t = a_t + d_t*(1 - dDepr - OpCost)/Rfree  (chi terms cancel)
        wXtraGrid = construct_grid(aXtraGrid[0],
                                    aXtraGrid[-1] + dNrmGridNow[-1] * (1 - dDepr - OpCost) / Rfree,
                                    100, grid_type, NestFac)
        wNrmGridNow = wXtraGrid + BoroCnstNat1D

        ### Beginning-of-next-period resources: ell_{t+1} = R*w_t + 1
        xNrmGridNext_Frictionless = wNrmGridNow * Rfree + 1

        ### EGM step via marginal-value Euler equation
        # The Euler equation for total spending q = c+d holds without correction only
        # when gam is the same across periods. At the terminal boundary gam changes
        # (gam_T uses the LTV-based p_d; gam_{non-T} uses the interior FOC), so the
        # q-based recursion carries a large error.  Instead, use the marginal-value
        # representation: u_c = beta*R * v'_{t+1}(ell_{t+1}) and invert via
        #   u_c = K_lambda * x^{-rho},  K_lambda = alpha * gam^{alpha-1}
        # to recover the Cobb-Douglas composite x = c^alpha * d^{1-alpha}.
        # This automatically accounts for any gam discontinuity at the boundary.
        K_lambda = alpha * gam ** (alpha - 1)
        vP_next_array = DiscFac * LivPrb * Rfree * vPFunc_Frictionless_Next(xNrmGridNext_Frictionless)
        x_array = (K_lambda / vP_next_array) ** (1.0 / CRRA)

        ### From composite x = gam^alpha * d, recover c and d using gam_t (non-terminal)
        dFunc_Frictionless_array = x_array / gam ** alpha
        cFunc_Frictionless_array = gam * dFunc_Frictionless_array
        qFunc_Frictionless_array = cFunc_Frictionless_array + dFunc_Frictionless_array

        ### Endogenous grid: ell_t = q_t * Lambda + w_t
        # Lambda = (gam + 1 + (dDepr + OpCost - 1)/Rfree) / (1 + gam)
        Lambda_Frictionless = (gam + 1 + (dDepr + OpCost - 1) / Rfree) / (1 + gam)
        xNrmGrid_Frictionless_array = qFunc_Frictionless_array * Lambda_Frictionless + wNrmGridNow

        ### Add a zero point at the natural borrowing constraint
        qFunc_Frictionless_array = np.append(0.0, qFunc_Frictionless_array)
        dFunc_Frictionless_array = np.append(0.0, dFunc_Frictionless_array)
        cFunc_Frictionless_array = np.append(0.0, cFunc_Frictionless_array)
        xNrmGrid_Frictionless_array    = np.append(BoroCnstNat1D, xNrmGrid_Frictionless_array)

        ### Interpolate policy functions
        qFunc_Frictionless = LinearFast(qFunc_Frictionless_array, [xNrmGrid_Frictionless_array])
        cFunc_Frictionless = LinearFast(cFunc_Frictionless_array, [xNrmGrid_Frictionless_array])
        dFunc_Frictionless = LinearFast(dFunc_Frictionless_array, [xNrmGrid_Frictionless_array])

        ### Value function: v = u(c,d) + beta*zeta*v_{t+1}(ell_{t+1})
        # Evaluate on the exogenous xNrmGridNow (skip the zero point, prepend 0)
        c_aux     = cFunc_Frictionless(xNrmGridNow[1:])
        d_aux     = dFunc_Frictionless(xNrmGridNow[1:])
        a_aux     = xNrmGridNow[1:] - c_aux - d_aux
        nNext_aux = d_aux * (1 - dDepr * (1 - MaintCost))
        mNext_aux = a_aux * Rfree - d_aux * (OpCost + dDepr * MaintCost) + 1
        EndOfPrdv_Frictionless_aux = DiscFac * LivPrb * vFunc_Frictionless_Next(nNext_aux + mNext_aux)

        vFunc_Frictionless_array     = CRRAutility(c_aux, d_aux) + EndOfPrdv_Frictionless_aux
        inv_vFunc_Frictionless_array = utility_inv(vFunc_Frictionless_array, CRRA)
        inv_vFunc_Frictionless_array = np.append(0.0, inv_vFunc_Frictionless_array)

        inv_vFunc_Frictionless = LinearFast(inv_vFunc_Frictionless_array, [xNrmGridNow])
        vFunc_Frictionless     = ValueFuncCRRA(inv_vFunc_Frictionless, CRRA)
        # MargValueFuncCRRA uses the same inv_v representation as ValueFuncCRRA
        vPFunc_Frictionless    = MargValueFuncCRRA(inv_vFunc_Frictionless, CRRA)


        ### Add the Limit Function to Adjuster solution
        ### Paste Frictionless Solution to adjuster
        # Policy functions (c, d) live on xNrmGrid_Frictionless_array (endogenous EGM grid).
        # Value function (inv_v) lives on xNrmGridNow (exogenous grid). Each needs its own
        # splice index because the two x-grids are independent.

        # ── Policy functions (c, d) ───────────────────────────────────────────────────
        # a) First frictionless EGM point beyond the adjuster grid
        idx_c = np.where(xNrmGrid_Frictionless_array > xAdj_comb_array[-1])[0][0]

        # b) Extended x-grid for policy functions (EGM grid)
        xAdj_comb_Limit_array = np.concatenate((xAdj_comb_array, xNrmGrid_Frictionless_array[idx_c:]))

        # c) Level-match at the splice point (ensure continuity)
        c_diff = cFunc_Frictionless_array[idx_c] - cFuncAdj(xNrmGrid_Frictionless_array[idx_c])
        d_diff = dFunc_Frictionless_array[idx_c] - dFuncAdj(xNrmGrid_Frictionless_array[idx_c])

        cFunc_Frictionless_adjusted_array = cFunc_Frictionless_array - c_diff
        dFunc_Frictionless_adjusted_array = dFunc_Frictionless_array - d_diff

        # d) Combine arrays
        cFuncAdj_comb_Limit_array = np.concatenate((cFuncAdj_comb_array, cFunc_Frictionless_adjusted_array[idx_c:]))
        dFuncAdj_comb_Limit_array = np.concatenate((dFuncAdj_comb_array, dFunc_Frictionless_adjusted_array[idx_c:]))

        # e) Create policy interpolants
        cFuncAdj_comb_Limit  = LinearFast(cFuncAdj_comb_Limit_array, [xAdj_comb_Limit_array])
        dFuncAdj_comb_Limit  = LinearFast(dFuncAdj_comb_Limit_array, [xAdj_comb_Limit_array])
        vPFuncAdj_comb_Limit = MargValueFuncCRRA_dur(cFuncAdj_comb_Limit, dFuncAdj_comb_Limit, alpha, d_ubar, CRRA)

        # ── Value function ────────────────────────────────────────────────────────────
        # Value function: Only create for new part
        ### Value function: v = u(c,d) + beta*zeta*v_{t+1}(ell_{t+1})
        # Evaluate on the exogenous xNrmGridNow (skip the zero point, prepend 0)
        idx_v = np.where(xNrmGrid_Frictionless_array > xNrmGridvFunc[-1])[0][0]
        c_aux     = cFunc_Frictionless_adjusted_array[idx_v:]
        d_aux     = dFunc_Frictionless_adjusted_array[idx_v:]
        a_aux     = xNrmGrid_Frictionless_array[idx_v:] - c_aux - d_aux
        nNext_aux = d_aux * (1 - dDepr * (1 - MaintCost))
        mNext_aux = a_aux * Rfree - d_aux * (OpCost + dDepr * MaintCost) + 1
        EndOfPrdv_Frictionless_aux = DiscFac * LivPrb * vFunc_Frictionless_Next(nNext_aux + mNext_aux)

        vFunc_Frictionless_Limit_array     = CRRAutility(c_aux, d_aux) + EndOfPrdv_Frictionless_aux
        inv_vFunc_Frictionless_Limit_array = utility_inv(vFunc_Frictionless_Limit_array, CRRA)

        ### Add solution
        xNrmGridvFunc_comb_Limit_array = np.concatenate((xNrmGridvFunc, xNrmGrid_Frictionless_array[idx_v:]))
        if inv_vAdj_aux_array[-1] > inv_vFunc_Frictionless_Limit_array[0]:
            print(" Frictionless is lower than adjuster at the last grid point")
            inv_v_diff = inv_vFunc_Frictionless_Limit_array[0] - inv_vFuncAdj(xNrmGrid_Frictionless_array[idx_v])
            inv_vFunc_Frictionless_Limit_array = inv_vFunc_Frictionless_Limit_array - inv_v_diff
        inv_vFunc_comb_Limit_array = np.concatenate((inv_vAdj_aux_array, inv_vFunc_Frictionless_Limit_array))
        inv_vFuncAdj_comb_Limit        = LinearFast(inv_vFunc_comb_Limit_array, [xNrmGridvFunc_comb_Limit_array])
        vFuncAdj            = ValueFuncCRRA(inv_vFuncAdj, CRRA)
        inv_vFuncAdj_SOL    = inv_vFuncAdj

        if np.min(inv_vFunc_Frictionless_Limit_array) < 0.0:
            print("error: inv_vFunc_Frictionless_Limit_array is negative")
        if np.min(inv_vFunc_comb_Limit_array) < 0.0:
            print("error: inv_vFunc_comb_Limit_array is negative")
        ### Re-define adjuster to keep the one with limit
        cFuncAdj = cFuncAdj_comb_Limit
        dFuncAdj = dFuncAdj_comb_Limit
        inv_vFuncAdj = inv_vFuncAdj_comb_Limit
        inv_vFuncAdj_SOL = inv_vFuncAdj_comb_Limit 
        vPFuncAdj = vPFuncAdj_comb_Limit

        if Figure:
            qFuncAdj = LinearFast(cFuncAdj_comb_array + dFuncAdj_comb_array, [xAdj_comb_array])
            qFuncAdjUnc = LinearFast(cAdj_upper + dAdj_upper,     [xAdj_upper])
            qFuncAdj_comb_Limit = LinearFast(cFuncAdj_comb_Limit_array + dFuncAdj_comb_Limit_array, [xAdj_comb_Limit_array])

            # ── x-grid ranges ────────────────────────────────────────────────────────
            x_ranges = [
                (float(np.atleast_1d(xNrmMinNow)[0]), 10.0,                  r'$x \in [x^{\min}, 10]$'),
                (float(np.atleast_1d(xNrmMinNow)[0]), 100.0,                 r'$x \in [x^{\min}, 100]$'),
                (float(np.atleast_1d(xNrmMinNow)[0]), 200.0,                 r'$x \in [x^{\min}, 200]$'),
                (float(np.atleast_1d(xNrmMinNow)[0]), float(xNrmGridNow[-1]), r'$x \in [x^{\min}, x^{\max}]$'),
            ]

            # Column labels and the three functions to evaluate
            col_info = [
                ('$c^{adj}(x)$',       cFuncAdj,     cFuncAdjUnc, cFunc_Frictionless, cFuncAdj_comb_Limit),
                ('$d^{adj}(x)$',       dFuncAdj,     dFuncAdjUnc, dFunc_Frictionless, dFuncAdj_comb_Limit),
                ('$q^{adj}(x)$',       qFuncAdj,     qFuncAdjUnc, qFunc_Frictionless, qFuncAdj_comb_Limit),
                ('$\\widetilde{v}^{adj}(x)$', inv_vFuncAdj, inv_vFuncAdjUnc, inv_vFunc_Frictionless, inv_vFuncAdj_comb_Limit),
            ]

            fig, axes = plt.subplots(4, 4, figsize=(18, 15))

            for row, (x_lo, x_hi, row_title) in enumerate(x_ranges):
                x_plot = np.linspace(x_lo + 1e-6, x_hi, 500)

                for col, (col_title, func_full, func_unc, func_lim, func_with_lim) in enumerate(col_info):
                    ax = axes[row, col]

                    # Full adjuster function (constrained + unconstrained)
                    ax.plot(x_plot, func_full(x_plot), lw=2, color='steelblue',  label='Full')

                    # Unconstrained part (no kink correction)
                    ax.plot(x_plot, func_unc(x_plot),  lw=1.5, color='darkorange',
                            ls='--', label='Unconstrained')

                    # Limit via EGM
                    ax.plot(x_plot, func_lim(x_plot),  lw=0.5, color='red',
                            ls=':', label='Limit')  

                    # Function with Limit
                    ax.plot(x_plot, func_with_lim(x_plot),  lw=0.5, color='black',
                            ls=':', label='Full + Limit')                                
                    # Constrained/combined array scatter (raw grid points)
                    if col == 0:
                        ax.scatter(xAdj_comb_array,
                                cFuncAdj_comb_array, s=12, color='crimson', zorder=5,
                                label='Grid pts')
                    elif col == 1:
                        ax.scatter(xAdj_comb_array,
                                dFuncAdj_comb_array, s=12, color='crimson', zorder=5,
                                label='Grid pts')
                    elif col == 2:
                        ax.scatter(xAdj_comb_array,
                                cFuncAdj_comb_array + dFuncAdj_comb_array, s=12, color='crimson', zorder=5,
                                label='Grid pts')
                    else:  # col == 3: inv_v
                        ax.scatter(xAdj_comb_array,
                                inv_vAdj_comb_array, s=12, color='crimson', zorder=5,
                                label='Grid pts')

                    # Mark the kink point (where collateral constraint starts to bind)
                    ax.axvline(x=float(x0), color='grey', ls=':', lw=1.2, label=f'Kink $x_0={x0:.2f}$')

                    ax.set_xlim(x_lo, x_hi)
                    ax.set_xlabel('$x = m + (1-\\tau) n$', fontsize=9)
                    if col == 0:
                        ax.set_ylabel(row_title, fontsize=10)
                    if row == 0:
                        ax.set_title(col_title, fontsize=11)
                    if row == 0 and col == 2:
                        ax.legend(fontsize=8, loc='upper left')

            fig.suptitle('Adjuster solution: Full vs. Unconstrained vs. Grid points',
                        fontsize=13, y=1.01)
            plt.tight_layout()
            plt.show()
            print("Figure done")
    else:
        qFunc_Frictionless=qFunc_Frictionless_Next
        vFunc_Frictionless=vFunc_Frictionless_Next
        vPFunc_Frictionless=vPFunc_Frictionless_Next        
    ############################################################################################
    # 6. Create Consumption Function:
    if timeBOOL:
        tic = time.time()
    '''
    Compares the value function for each combination of durable stock and market resources. Note that xNrmGrid is
    defined as (1-adjC)*nNrmGrid + mNrmGrid
    '''
    if alpha == 1: # Use keeper solution
        # Keeper Value Function and asset function
        vFuncmNrmGridNow = mNrmGridNow
        vFunc_shape = (len(nNrmGridNow), len(vFuncmNrmGridNow))
        inv_vFuncKeep_array = np.zeros(vFunc_shape)
        aFunc_array = np.zeros(vFunc_shape)
        vPFuncKeep_array = np.zeros(vFunc_shape)
        for i_n, n_aux in enumerate(nNrmGridNow):
            n_aux_array = np.ones(len(vFuncmNrmGridNow)-1) * n_aux
            
            ### Keeper
            # Compute expected value and marginal value on a grid of market resources
            cNrmKeepNow = cFuncKeep(n_aux_array, vFuncmNrmGridNow[1:])
            aNrmKeepNow = vFuncmNrmGridNow[1:] - cNrmKeepNow
            vNrmKeepNow = CRRAutility(cNrmKeepNow, n_aux_array) + EndOfPrdv(n_aux_array, aNrmKeepNow)

            # Construct the beginning-of-period value function
            inv_vFuncKeep_array[i_n,1:] = utility_inv(vNrmKeepNow, CRRA)  # value transformed through inverse utility
            inv_vFuncKeep_array[i_n, 0] = 0

            # aFunc
            aFunc_array[i_n, 1:] = vFuncmNrmGridNow[1:] - cFuncKeep(n_aux_array, vFuncmNrmGridNow[1:])

            n_aux_array = np.ones(len(vFuncmNrmGridNow)) * n_aux
            ### add a zero point to cNrmKeepNow
            cNrmKeepNow = np.insert(cNrmKeepNow, 0, 0)
            vPFuncKeep_array[i_n] = CRRAutilityP(cNrmKeepNow, n_aux_array)
        
        inv_vFuncKeep = LinearFast(inv_vFuncKeep_array, [nNrmGridNow, vFuncmNrmGridNow])
        vFuncKeep = ValueFuncCRRA(inv_vFuncKeep, CRRA)

          # Marginal value function
        inv_vPFuncKeep_array = utilityP_inv(vPFuncKeep_array, CRRA)

        cFunc = cFuncKeep
        dFunc = dFuncKeep
        aFunc = LinearFast(aFunc_array, [nNrmGridNow, vFuncmNrmGridNow])
        inv_vFunc = inv_vFuncKeep
        vFunc = vFuncKeep
        vPFunc = vPFuncKeep
        vPDFunc = vPFuncKeep # For alpha == 1 it's the same as vPFuncKeep
        inv_vPFunc = LinearFast(inv_vPFuncKeep_array, [nNrmGridNow, vFuncmNrmGridNow])
        inv_vPDFunc = LinearFast(inv_vPFuncKeep_array, [nNrmGridNow, vFuncmNrmGridNow]) # For alpha == 1 it's the same as inv_vPFunc
    
        solution_shape = (len(nNrmGridNow), len(mNrmGridNow))
        adjusting_array = np.ones(solution_shape)
        adjusting = LinearFast(adjusting_array, [nNrmGridNow, mNrmGridNow])

        ### For adjuster: empty container
        cFuncAdj = cFuncKeep
        dFuncAdj = dFuncKeep
        inv_vFuncAdj = inv_vFuncKeep
        vFuncAdj = vFuncKeep
        vPFuncAdj = vPFuncKeep

        ### Things we keep track off
        xAdj_upper = np.zeros_like(mNrmGridNow)
        dAdj_upper = np.zeros_like(mNrmGridNow)
        cAdj_upper = np.zeros_like(mNrmGridNow)
        vPFunc_eval = vPFunc
        vPDFunc_eval = vPDFunc
        lSuS_array = np.zeros((len(nNrmGridNow),2))
        d_kink = 0

        ### Frictionless
        cFunc_Frictionless=cFuncAdj
        dFunc_Frictionless=dFuncAdj
        qFunc_Frictionless=cFuncAdj
        vFunc_Frictionless=vFuncAdj
        inv_vFunc_Frictionless=inv_vFuncAdj
        vPFunc_Frictionless=vPFuncAdj

    elif adjC == 0 and SearchCost == 0: # Frictionless case: Use adjuster solution
        mNrmGridTotal = mNrmGridNow
        nNrmGridTotal = nNrmGridNow
        vFuncmNrmGridNow = mNrmGridNow
        #### FAST PART STARTS HERE
        # Create empty container
        shape = (len(nNrmGridTotal), len(mNrmGridTotal))
        cFuncAdj_SOL_array = np.zeros(shape)
        dFuncAdj_SOL_array = np.zeros(shape)
        aFuncAdj_SOL_array = np.zeros(shape)
        inv_vFuncAdj_SOL_array = np.zeros(shape)
        vPDFuncAdj_SOL_array = np.zeros(shape)
        adjusting_array = np.ones(shape)

        # Vectorised: build the full x_aux matrix in one numpy broadcast,
        # then call all 1-D LinearFast functions on the flattened array.
        _x_aux_2d = np.clip(
            mNrmGridTotal[np.newaxis, :] + nNrmGridTotal[:, np.newaxis], BoroCnstArt, None
        )  # shape (nNrmGridTotal, mNrmGridTotal)
        _x_flat = _x_aux_2d.ravel()
        cFuncAdj_SOL_array   = np.clip(cFuncAdj(_x_flat),      0, None).reshape(shape)
        dFuncAdj_SOL_array   = np.clip(dFuncAdj(_x_flat),      0, None).reshape(shape)
        aFuncAdj_SOL_array   = _x_aux_2d - cFuncAdj_SOL_array - dFuncAdj_SOL_array
        inv_vFuncAdj_SOL_array = np.clip(inv_vFuncAdj_SOL(_x_flat), 0, None).reshape(shape)

        vPFuncAdj_SOL_array = CRRAutilityP(cFuncAdj_SOL_array, dFuncAdj_SOL_array)
        # For adjC=0 the adjuster's marginal value of initial durable stock n is
        # (1-adjC)*u_c = u_c, so vPDFuncAdj = vPFuncAdj.
        vPDFuncAdj_SOL_array = CRRAutilityP(cFuncAdj_SOL_array, dFuncAdj_SOL_array)

        inv_vPFuncAdj_SOL_array = utilityP_inv(vPFuncAdj_SOL_array, CRRA)
        inv_vPDFuncAdj_SOL_array = utilityP_inv(vPDFuncAdj_SOL_array, CRRA)
        
        ### Frictionless Case: Take the adjuster solution
        cFunc_array = cFuncAdj_SOL_array
        dFunc_array = dFuncAdj_SOL_array
        aFunc_array = aFuncAdj_SOL_array
        inv_vFunc_array = inv_vFuncAdj_SOL_array
        vPDFunc_array = vPDFuncAdj_SOL_array
        adjusting_array = np.ones_like(cFunc_array)

        inv_vPFunc_array = inv_vPFuncAdj_SOL_array
        inv_vPDFunc_array = inv_vPDFuncAdj_SOL_array

        ### Frictionless: all consumers adjust, so keeper aliases point to adjuster solution
        cFuncKeep = LinearFast(cFunc_array, [nNrmGridTotal, mNrmGridTotal])
        dFuncKeep = LinearFast(dFunc_array, [nNrmGridTotal, mNrmGridTotal])
        inv_vFuncKeep = LinearFast(inv_vFunc_array, [nNrmGridTotal, mNrmGridTotal])
        vFuncKeep = ValueFuncCRRA(inv_vFuncKeep, CRRA)
        vPFuncKeep = MargValueFuncCRRA_dur(cFuncKeep, dFuncKeep, alpha, d_ubar, CRRA)
        lSuS_array = np.zeros((len(nNrmGridTotal),2))

    else:
        mNrmGridTotal = mNrmGridNow
        nNrmGridTotal = nNrmGridNow
        vFuncmNrmGridNow = mNrmGridNow  # required by the dDepr==1 block below

        #### FAST PART STARTS HERE
        # Create empty container
        shape = (len(nNrmGridTotal), len(mNrmGridTotal))
        cFuncAdj_SOL_array = np.zeros(shape)
        dFuncAdj_SOL_array = np.zeros(shape)
        aFuncAdj_SOL_array = np.zeros(shape)
        inv_vFuncAdj_SOL_array = np.zeros(shape)
        vPDFuncAdj_SOL_array = np.zeros(shape)

        cFuncKeep_SOL_array = np.zeros(shape)
        dFuncKeep_SOL_array = np.zeros(shape)
        aFuncKeep_SOL_array = np.zeros(shape)
        inv_vFuncKeep_SOL_array = np.zeros(shape)
        vPDFuncKeep_SOL_array = np.zeros(shape)

        adjusting_array = np.ones(shape)
        LambdaN_array = np.zeros(shape)
        LambdaM_array = np.zeros(shape)

        # ── Phase 1: Adjuster arrays ─────────────────────────────────────────
        # Build the scaled adjustment cost and x_aux for every (i_n, i_m) at once.
        _n2d_sol = nNrmGridTotal[:, np.newaxis] * np.ones_like(mNrmGridTotal)[np.newaxis, :]
        _m2d_sol = mNrmGridTotal[np.newaxis, :] * np.ones_like(nNrmGridTotal)[:, np.newaxis]
        _adjC_scaled_2d = tau_func((_n2d_sol + _m2d_sol).ravel()).reshape(shape)
        _x_aux_2d = np.clip(_m2d_sol + (1 - _adjC_scaled_2d) * _n2d_sol, BoroCnstArt, None)

        # Per-row start-index (triangular constraint boundary)
        _mNrmMinNow_sz = np.array(mNrmMinNow).size
        _idx_starts = np.array([
            (len(mNrmMinNow) - i_n) if _mNrmMinNow_sz > 1 else 0
            for i_n in range(len(nNrmGridTotal))
        ], dtype=np.intp)

        # Build column mask for valid cells: col >= idx_start[row]
        _col_idx_sol = np.arange(len(mNrmGridTotal))
        _valid_2d = _col_idx_sol[np.newaxis, :] >= _idx_starts[:, np.newaxis]   # (nN, nM)
        _invalid_2d = ~_valid_2d

        # Evaluate 1-D adjuster functions on the full flattened x_aux
        _c_adj_flat = np.zeros(shape[0] * shape[1])
        _d_adj_flat = np.zeros_like(_c_adj_flat)
        _iv_adj_flat = np.zeros_like(_c_adj_flat)
        _valid_flat = _valid_2d.ravel()
        _c_adj_flat[_valid_flat]  = cFuncAdj(_x_aux_2d.ravel()[_valid_flat])
        _d_adj_flat[_valid_flat]  = dFuncAdj(_x_aux_2d.ravel()[_valid_flat])
        _iv_adj_flat[_valid_flat] = inv_vFuncAdj_SOL(_x_aux_2d.ravel()[_valid_flat])
        cFuncAdj_SOL_array   = _c_adj_flat.reshape(shape)
        dFuncAdj_SOL_array   = _d_adj_flat.reshape(shape)
        inv_vFuncAdj_SOL_array = _iv_adj_flat.reshape(shape)
        aFuncAdj_SOL_array   = np.where(_valid_2d,
                                         _x_aux_2d - cFuncAdj_SOL_array - dFuncAdj_SOL_array,
                                         BoroCnstArt2D[:, np.newaxis])
        # vPD for adjuster: (1 - adjC_scaled) * u_c
        vPDFuncAdj_SOL_array = (1 - _adjC_scaled_2d) * CRRAutilityP(cFuncAdj_SOL_array, dFuncAdj_SOL_array)

        # ── Phase 2: Keeper arrays ───────────────────────────────────────────
        # Evaluate 2-D keeper functions on the full grid in one batch call.
        _c_keep_flat = np.zeros(shape[0] * shape[1])
        _iv_keep_flat = np.zeros_like(_c_keep_flat)
        _c_keep_flat[_valid_flat]  = np.clip(
            cFuncKeep(_n2d_sol.ravel()[_valid_flat], _m2d_sol.ravel()[_valid_flat]), 0, None)
        _iv_keep_flat[_valid_flat] = inv_vFuncKeep(
            _n2d_sol.ravel()[_valid_flat], _m2d_sol.ravel()[_valid_flat])
        cFuncKeep_SOL_array   = _c_keep_flat.reshape(shape)
        dFuncKeep_SOL_array   = _n2d_sol.copy()           # keeper retains n
        aFuncKeep_SOL_array   = np.where(_valid_2d,
                                          _m2d_sol - cFuncKeep_SOL_array,
                                          BoroCnstArt2D[:, np.newaxis])
        inv_vFuncKeep_SOL_array = _iv_keep_flat.reshape(shape)

        # ── Phase 3: Lambda arrays (TransitionMatrix branch) ─────────────────
        if expected_method == "TransitionMatrix":
            # Batch call: pass all valid (n, a_keep) points to the 2-D LinearFast
            # interpolants in one shot.  LinearFast internally calls eval_linear
            # (a numba CPUDispatcher) on the entire array → no Python per-cell loop.
            a_keep_all = _m2d_sol - cFuncKeep_SOL_array    # (nN, nM)
            _n_flat_v  = _n2d_sol.ravel()[_valid_flat]      # valid cells only
            _a_flat_v  = a_keep_all.ravel()[_valid_flat]

            _inv_l_flat = inv_EndOfPrdvPD(_n_flat_v, _a_flat_v)
            LambdaN_array = np.zeros(shape)
            LambdaN_array.ravel()[_valid_flat] = np.where(
                _inv_l_flat > 0, _inv_l_flat ** (-CRRA), 0.0)

            if (OpCost + dDepr * MaintCost) > 0:
                _inv_m_flat = inv_LambdaM(_n_flat_v, _a_flat_v)
                LambdaM_array = np.zeros(shape)
                LambdaM_array.ravel()[_valid_flat] = np.where(
                    _inv_m_flat > 0, _inv_m_flat ** (-CRRA), 0.0)
            else:
                LambdaM_array = np.zeros(shape)
        else:
            for i_n in range(len(nNrmGridTotal)):
                idx_start = int(_idx_starts[i_n])
                aNrmKeep_aux = mNrmGridTotal - cFuncKeep_SOL_array[i_n]
                n_keep_scalar = nNrmGridTotal[i_n] * np.ones(len(aNrmKeep_aux[idx_start:]))
                LambdaN_array[i_n, idx_start:] = (
                    LivPrb * PermGroFac**(-CRRA) * (1 - dDepr*(1-MaintCost)) * DiscFac *
                    expected(vp_next_vPFunc, IncShkDstn, args=(
                        vPDFuncNext, aNrmKeep_aux[idx_start:], n_keep_scalar,
                        CRRA, dDepr, Rfree, PermGroFac)))
                LambdaM_array[i_n, idx_start:] = (
                    LivPrb * PermGroFac**(-CRRA) * (OpCost + dDepr*MaintCost) * DiscFac *
                    expected(vp_next_vPFunc, IncShkDstn, args=(
                        vPFuncNext, aNrmKeep_aux[idx_start:], n_keep_scalar,
                        CRRA, dDepr, Rfree, PermGroFac)))

        # Clip inf/nan to a large finite number
        large_number = 1e12
        LambdaN_array = np.where(np.isfinite(LambdaN_array), LambdaN_array, large_number)
        LambdaM_array = np.where(np.isfinite(LambdaM_array), LambdaM_array, large_number)

        # vPD keeper = u_d(c,d) + Lambda
        Lambda = np.clip(LambdaN_array - LambdaM_array, 0.0, None)
        vPDFuncKeep_SOL_array = CRRAutilityPD(cFuncKeep_SOL_array, dFuncKeep_SOL_array) + Lambda

        if timeBOOL:
            for i_n in range(len(vPDFuncKeep_SOL_array)):
                if np.min(vPDFuncKeep_SOL_array[i_n]) <= 0.0:
                    print(f"Warning: vPDFuncKeep_SOL_array contains zero at i_n = {i_n}")

        ### Recover value function from its inverse: v = utility(inv_v, CRRA)
        vFuncAdj_SOL_array = utility(inv_vFuncAdj_SOL_array, CRRA)
        vFuncKeep_SOL_array = utility(inv_vFuncKeep_SOL_array, CRRA)

        ### Use inverse marginal value functions
        vPFuncAdj_SOL_array = CRRAutilityP(cFuncAdj_SOL_array, dFuncAdj_SOL_array)
        vPFuncKeep_SOL_array = CRRAutilityP(cFuncKeep_SOL_array, dFuncKeep_SOL_array)
        inv_vPFuncAdj_SOL_array = utilityP_inv(vPFuncAdj_SOL_array, CRRA)
        inv_vPFuncKeep_SOL_array = utilityP_inv(vPFuncKeep_SOL_array, CRRA)
        inv_vPDFuncAdj_SOL_array = utilityP_inv(vPDFuncAdj_SOL_array, CRRA)
        inv_vPDFuncKeep_SOL_array = utilityP_inv(vPDFuncKeep_SOL_array, CRRA)
        # Replace Nan's with zero
        inv_vPDFuncKeep_SOL_array = np.nan_to_num(inv_vPDFuncKeep_SOL_array)

        if SearchCost == 0:
            #### No SS-Structure (allow for multiple thresholds)
            cFunc_array, dFunc_array, aFunc_array, inv_vFunc_array, vPFunc_array, vPDFunc_array, adjusting_array = durable_solution_function_FUES_fast_search_noSS(
                adjusting_array,
                cFuncAdj_SOL_array, dFuncAdj_SOL_array, aFuncAdj_SOL_array,
                inv_vFuncAdj_SOL_array, vPFuncAdj_SOL_array, vPDFuncAdj_SOL_array,
                cFuncKeep_SOL_array, dFuncKeep_SOL_array, aFuncKeep_SOL_array,
                inv_vFuncKeep_SOL_array, vPFuncKeep_SOL_array, vPDFuncKeep_SOL_array, 1e-4) #tol)

            ### Set maximum n and m until which we need to search:
            mask_boolean = (inv_vFuncAdj_SOL_array + tol>= inv_vFuncKeep_SOL_array)
            adjusting_array = np.where(mask_boolean, np.ones_like(inv_vFuncAdj_SOL_array), np.zeros_like(inv_vFuncKeep_SOL_array))
        
            ### Calculate S,s Structure
            lSuS_array, lSuS_idx_array = calc_SS_fast_terminal(nNrmGridTotal, mNrmGridTotal, mNrmMinNow, inv_vFuncAdj_SOL_array, inv_vFuncKeep_SOL_array, tol=1e-4)

            ### a) Find i_n idx where lS = center = uS
            lS = lSuS_array[:, 0]
            center = lSuS_array[:, 1]
            uS = lSuS_array[:, 2]

            lS_idx = lSuS_idx_array[:, 0]
            uS_idx = lSuS_idx_array[:, 2]
            center_idx = lSuS_idx_array[:, 1]
            
            arr_start_idx = np.where(nNrmGridTotal > p1)[0][0]
            arr = lSuS_idx_array[arr_start_idx:]  # Slice the array from that index onwards
            # Compute range (max - min) for each row
            row_ranges = arr.max(axis=1) - arr.min(axis=1)

            # Find the index of the row with the smallest range
            best_index = np.argmin(row_ranges) + arr_start_idx  # Adjust index to account for slicing
            idxn4_p2 = np.where(nNrmGridTotal > p2 * 4)[0][0]  # Find the first index where nNrmGridTotal is greater than p2
            best_index = np.min([best_index, idxn4_p2])  # Ensure best_index is not beyond p2
            
            nlimit = nNrmGridTotal[best_index]  # Use the second element of the best row as the nlimit
            idx_nlimit = best_index
            
            ### for m-direction: Look at wholes in S,s structure. For i_m: calculate maximum adjusting_array. If it is 1, there is a whole. Look at the smallest idx_m where the whole exist and it is above m>0
            # idx_m_zero = np.where(mNrmGridTotal > 0.0)[0][0]
            idx_m = np.where(mNrmGridTotal > p2)[0][0]
            idxm4_p2 = np.where(mNrmGridTotal > p2 * 4)[0][0]
            m_limit = np.zeros_like(mNrmGridTotal)
            for i_m in range(len(mNrmGridTotal)):
                if i_m < idx_m:
                    continue
                m_limit[i_m] = 1 if np.min(adjusting_array[:idx_nlimit - 1, i_m]) == 1 else 0
            
            idx_mlimit =  np.min([np.max([np.where(m_limit == 1)[0][0], idx_m]), idxm4_p2])  # If no idx_mlimit is found, use idx_m

            for i_n in range(len(nNrmGridTotal)):
                for i_m in range(len(mNrmGridTotal)):
                    if i_m < idx_mlimit and i_n < idx_nlimit:
                        continue
                    cFunc_array[i_n, i_m] = cFuncAdj_SOL_array[i_n, i_m]
                    dFunc_array[i_n, i_m] = dFuncAdj_SOL_array[i_n, i_m]
                    aFunc_array[i_n, i_m] = aFuncAdj_SOL_array[i_n, i_m]
                    inv_vFunc_array[i_n, i_m] = inv_vFuncAdj_SOL_array[i_n, i_m]
                    vPFunc_array[i_n, i_m] = vPFuncAdj_SOL_array[i_n, i_m]
                    vPDFunc_array[i_n, i_m] = vPDFuncAdj_SOL_array[i_n, i_m]
                        
            vFunc_array = utility(inv_vFunc_array, CRRA)  # recover v from inv_v so line below is consistent

        else: # Directed Search
            ### Set maximum n and m until which we need to search:
            mask_boolean = (inv_vFuncAdj_SOL_array + tol>= inv_vFuncKeep_SOL_array)
            adjusting_array = np.where(mask_boolean, np.ones_like(inv_vFuncAdj_SOL_array), np.zeros_like(inv_vFuncKeep_SOL_array))
            ### Calculate S,s Structure
            lSuS_array, lSuS_idx_array = calc_SS_fast_terminal(nNrmGridTotal, mNrmGridTotal, mNrmMinNow, inv_vFuncAdj_SOL_array, inv_vFuncKeep_SOL_array, tol=1e-4)

            ### a) Find i_n idx where lS = center = uS
            lS = lSuS_array[:, 0]
            center = lSuS_array[:, 1]
            uS = lSuS_array[:, 2]

            lS_idx = lSuS_idx_array[:, 0]
            uS_idx = lSuS_idx_array[:, 2]
            center_idx = lSuS_idx_array[:, 1]
            
            arr_start_idx = np.where(nNrmGridTotal > p1)[0][0]
            arr = lSuS_idx_array[arr_start_idx:]  # Slice the array from that index onwards
            # Compute range (max - min) for each row
            row_ranges = arr.max(axis=1) - arr.min(axis=1)

            # Find the index of the row with the smallest range
            best_index = np.argmin(row_ranges) + arr_start_idx  # Adjust index to account for slicing
            idxn4_p2 = np.where(nNrmGridTotal > p2 * 4)[0][0]  # Find the first index where nNrmGridTotal is greater than p2
            best_index = np.min([best_index, idxn4_p2])  # Ensure best_index is not beyond p2
            
            nlimit = nNrmGridTotal[best_index]  # Use the second element of the best row as the nlimit
            idx_nlimit = best_index
            
            ### for m-direction: Look at wholes in S,s structure. For i_m: calculate maximum adjusting_array. If it is 1, there is a whole. Look at the smallest idx_m where the whole exist and it is above m>0
            # idx_m_zero = np.where(mNrmGridTotal > 0.0)[0][0]
            idx_m = np.where(mNrmGridTotal > p2)[0][0]
            idxm4_p2 = np.where(mNrmGridTotal > p2 * 4)[0][0]
            m_limit = np.zeros_like(mNrmGridTotal)
            for i_m in range(len(mNrmGridTotal)):
                if i_m < idx_m:
                    continue
                m_limit[i_m] = 1 if np.min(adjusting_array[:idx_nlimit - 1, i_m]) == 1 else 0
            
            idx_mlimit =  np.min([np.max([np.where(m_limit == 1)[0][0], idx_m]), idxm4_p2])  # If no idx_mlimit is found, use idx_m

            ### Optimal Search
            ### Optimal effort for Directed Search
            # e_grid = np.linspace(0,1,100000)
            e_grid = np.linspace(0,1,101)
            search_cost_matrix = np.zeros(shape)

            # Vectorised: evaluate the 1-D LinearFast SearchCost_func on the
            # full (nN × nM) x-grid in a single batch call.
            _x_sc = (nNrmGridTotal[:, np.newaxis] + mNrmGridTotal[np.newaxis, :]).ravel()
            search_cost_matrix = SearchCost_func(_x_sc).reshape(shape)

            # Step 2: Compute costs matrix with Numba
            Costs_matrix = compute_cost_matrix(search_cost_matrix, e_grid)


            ### Replace -np.inf with -1e12
            vFuncAdj_SOL_aux_array = np.where(vFuncAdj_SOL_array == -np.inf, -1e12, vFuncAdj_SOL_array)
            vFuncKeep_SOL_aux_array = np.where(vFuncKeep_SOL_array == -np.inf, -1e12, vFuncKeep_SOL_array)
            vPFuncAdj_SOL_aux_array = np.where(vPFuncAdj_SOL_array == np.inf, 1e12, vPFuncAdj_SOL_array)
            vPFuncKeep_SOL_aux_array = np.where(vPFuncKeep_SOL_array == np.inf, 1e12, vPFuncKeep_SOL_array)
            vPDFuncAdj_SOL_aux_array = np.where(vPDFuncAdj_SOL_array == np.inf, 1e12, vPDFuncAdj_SOL_array)
            vPDFuncKeep_SOL_aux_array = np.where(vPDFuncKeep_SOL_array == np.inf, 1e12, vPDFuncKeep_SOL_array)
            
            adjusting_array, DirectedSearchCost_array, vFunc_array, cFunc_array, dFunc_array, aFunc_array, vPFunc_array, vPDFunc_array = directed_search_function_scaled_matrix_limit(nNrmGridTotal, mNrmGridTotal, mNrmMinNow, idx_nlimit, idx_mlimit, e_grid, Costs_matrix, vFuncAdj_SOL_aux_array, vFuncKeep_SOL_aux_array,
                            cFuncAdj_SOL_array, cFuncKeep_SOL_array, 
                            dFuncAdj_SOL_array, dFuncKeep_SOL_array, 
                            aFuncAdj_SOL_array, aFuncKeep_SOL_array,
                            vPFuncAdj_SOL_aux_array, vPFuncKeep_SOL_aux_array,
                            vPDFuncAdj_SOL_aux_array, vPDFuncKeep_SOL_aux_array)
            
            ### Replace -1e12 with -np.inf
            cond_inf = (
                ((adjusting_array == 1.0) & (vFuncAdj_SOL_array == -np.inf)) |
                ((adjusting_array == 0.0) & (vFuncKeep_SOL_array == -np.inf)) |
                (((adjusting_array > 0.0) & (adjusting_array < 1.0)) &
                ((vFuncAdj_SOL_array == -np.inf) | (vFuncKeep_SOL_array == -np.inf)))
            )

            vFunc_array[cond_inf] = -np.inf

            # Conditions for restoring np.inf
            cond_inf = (
                ((adjusting_array == 1.0) & (vPFuncAdj_SOL_array == np.inf)) |
                ((adjusting_array == 0.0) & (vPFuncKeep_SOL_array == np.inf)) |
                (((adjusting_array > 0.0) & (adjusting_array < 1.0)) &
                ((vPFuncAdj_SOL_array == np.inf) | (vPFuncKeep_SOL_array == np.inf)))
            )

            # Apply condition
            vPFunc_array[cond_inf] = np.inf

            # Conditions for restoring np.inf
            cond_inf = (
                ((adjusting_array == 1.0) & (vPDFuncAdj_SOL_array == np.inf)) |
                ((adjusting_array == 0.0) & (vPDFuncKeep_SOL_array == np.inf)) |
                (((adjusting_array > 0.0) & (adjusting_array < 1.0)) &
                ((vPDFuncAdj_SOL_array == np.inf) | (vPDFuncKeep_SOL_array == np.inf)))
            )

            # Apply condition
            vPDFunc_array[cond_inf] = np.inf
            
            # Directly take from original arrays when adjusting=1 or adjusting=0
            mask_adj1 = (adjusting_array == 1.0)
            mask_adj0 = (adjusting_array == 0.0)
            vPFunc_array[mask_adj1] = vPFuncAdj_SOL_array[mask_adj1]
            vPFunc_array[mask_adj0] = vPFuncKeep_SOL_array[mask_adj0]

            vPDFunc_array[mask_adj1] = vPDFuncAdj_SOL_array[mask_adj1]
            vPDFunc_array[mask_adj0] = vPDFuncKeep_SOL_array[mask_adj0]
            
        if timeBOOL and np.min(vPDFunc_array) == 0.0:
            print("Warning: vPDFunc_array contains zero values.")

        inv_vFunc_array = utility_inv(vFunc_array, CRRA)
        inv_vPFunc_array = utilityP_inv(vPFunc_array, CRRA)
        inv_vPDFunc_array = utilityP_inv(vPDFunc_array, CRRA)

        if timeBOOL:
            toc = time.time()
            print("Assembling Solution: {:.2f} seconds".format(toc - tic))
        ########################################################################################################################################################
         
        if Figure:
            adjusting_redefined_array = adjusting_array.copy()
            ### Plot adjusting_terminal_array
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            axes = axes.flatten() 
            mNrmGridTotal = mNrmGridTotal.flatten()
            nNrmGridNow = nNrmGridNow.flatten()

            # First subplot
            pcm1 = axes[0].pcolormesh(mNrmGridTotal, nNrmGridNow, adjusting_array, cmap='viridis', shading='auto')
            fig.colorbar(pcm1, ax=axes[0], label='Adjusting Array Value')
            axes[0].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[0].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[0].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            # Add a horizontal and vertical line at the limits
            axes[0].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[0].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            # axes[0].scatter(0.0, nNrmGridNow[idx_nlimit], color='red', label='nNrmGridMax')
            # axes[0].scatter(mNrmGridTotal[idx_mlimit], 0.0, color='blue', label='mNrmGridMax')
            axes[0].set_xlabel('mNrmGridTotal')
            axes[0].set_ylabel('nNrmGridTotal')
            axes[0].set_title('2D Plot of Adjusting Array (re-fined)')
            axes[0].set_xlim(mNrmGridTotal[0], mNrmGridTotal[-1])
            axes[0].set_ylim(nNrmGridNow[0], nNrmGridNow[-1])

            # Second subplot (identical, but you can change as needed)
            pcm2 = axes[1].pcolormesh(mNrmGridTotal, nNrmGridNow, adjusting_array, cmap='viridis', shading='auto')
            fig.colorbar(pcm2, ax=axes[1], label='Adjusting Array Value')
            axes[1].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[1].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[1].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[1].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[1].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            axes[1].set_xlabel('mNrmGridTotal')
            axes[1].set_ylabel('nNrmGridTotal')
            axes[1].set_title('2D Plot of Adjusting Array (re-fined)')
            axes[1].set_xlim(np.max([-(p2*2 + 10), mNrmGridTotal[0]]), mNrmGridTotal[idx_mlimit] + 50)
            axes[1].set_ylim(nNrmGridTotal[0], nNrmGridTotal[idx_nlimit] + 50)

            pcm3 = axes[2].pcolormesh(mNrmGridTotal, nNrmGridNow, adjusting_array, cmap='viridis', shading='auto')
            fig.colorbar(pcm2, ax=axes[1], label='Adjusting Array Value')
            axes[2].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[2].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[2].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[2].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[2].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            axes[2].set_xlabel('mNrmGridTotal')
            axes[2].set_ylabel('nNrmGridTotal')
            axes[2].set_title('2D Plot of Adjusting Array (re-fined)')
            axes[2].set_xlim(np.max([-3.0, mNrmGridTotal[0]]), 3.0)
            axes[2].set_ylim(0.0, 3.0)

            # Third subplot: Difference plot
            diff_trim = inv_vFuncAdj_SOL_array.T - inv_vFuncKeep_SOL_array.T
            diff_trim = np.clip(diff_trim, -1.0, 1.0)
            diff_trim = np.nan_to_num(diff_trim)

            # Custom colormap for negative values: white to red
            neg_colors = np.linspace(0.3, 1, 256)
            neg_colormap = ListedColormap([(1, r, r) for r in neg_colors])  # Gradient from strong red to white

            # Meshgrid for plotting
            d, a = np.meshgrid(mNrmGridTotal, nNrmGridTotal, indexing='ij')

            # Contour for positive values
            pos_levels = np.linspace(0, np.max(diff_trim), 50)
            pos_contour = axes[3].contourf(d, a, np.where(diff_trim > 0, diff_trim, np.nan), 
                                        levels=pos_levels, cmap='Blues', extend='max')

            # Contour for negative values
            # neg_levels = np.linspace(np.min(diff_trim), 0, 50)
            neg_levels = np.linspace(-1.0, 0, 50)
            neg_contour = axes[3].contourf(d, a, np.where(diff_trim < 0, diff_trim, np.nan), 
                                        levels=neg_levels, cmap=neg_colormap, extend='min')

            # Add colorbars for positive and negative values
            fig.colorbar(pos_contour, ax=axes[3], label='Positive Values', orientation='vertical', shrink=0.8)
            fig.colorbar(neg_contour, ax=axes[3], label='Negative Values', orientation='vertical', shrink=0.8)

            # Customization for third subplot
            axes[3].set_title('Inverse vFunc Difference (White to Strong Red for Negatives)')
            axes[3].set_ylabel('nNrmGridTotal')
            axes[3].set_xlabel('mNrmGridTotal')
            axes[3].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[3].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[3].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[3].plot(BoroCnstArt2D, nNrmGridNow, 'k--', label='Zero Grid mNrm')
            axes[3].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[3].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            # axes[2].plot(nNrmGridTotal, Lower_Bound_array_raw, 'b--', label='Lower Bound Raw')
            # axes[2].plot(nNrmGridTotal, Upper_Bound_array_raw, 'r--', label='Upper Bound Raw')
            # axes[2].plot(nNrmGridTotal, Bounds_array[:,0], label='Lower Bound')
            # axes[2].plot(nNrmGridTotal, Bounds_array[:,1], label='Upper Bound')
            axes[3].set_xlim(mNrmGridTotal[0], mNrmGridTotal[-1])
            axes[3].set_ylim(nNrmGridTotal[0], nNrmGridTotal[-1])
            axes[3].legend()

            # Third subplot: Difference plot
            pos_contour = axes[4].contourf(d, a, np.where(diff_trim > 0, diff_trim, np.nan), 
                                        levels=pos_levels, cmap='Blues', extend='max')

            # Contour for negative values
            neg_contour = axes[4].contourf(d, a, np.where(diff_trim < 0, diff_trim, np.nan), 
                                        levels=neg_levels, cmap=neg_colormap, extend='min')

            # Add colorbars for positive and negative values
            fig.colorbar(pos_contour, ax=axes[4], label='Positive Values', orientation='vertical', shrink=0.8)
            fig.colorbar(neg_contour, ax=axes[4], label='Negative Values', orientation='vertical', shrink=0.8)

            # Customization for third subplot
            axes[4].set_title('Inverse vFunc Difference (White to Strong Red for Negatives)')
            axes[4].set_ylabel('nNrmGridTotal')
            axes[4].set_xlabel('mNrmGridTotal')
            axes[4].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[4].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[4].plot(BoroCnstArt2D, nNrmGridNow, 'k--', label='Zero Grid mNrm')
            axes[4].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[4].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[4].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            # axes[2].plot(nNrmGridTotal, Lower_Bound_array_raw, 'b--', label='Lower Bound Raw')
            # axes[2].plot(nNrmGridTotal, Upper_Bound_array_raw, 'r--', label='Upper Bound Raw')
            # axes[2].plot(nNrmGridTotal, Bounds_array[:,0], label='Lower Bound')
            # axes[2].plot(nNrmGridTotal, Bounds_array[:,1], label='Upper Bound')
            axes[4].set_xlim(np.max([-(p2*2 + 10), mNrmGridTotal[0]]), mNrmGridTotal[idx_mlimit] + 50)
            axes[4].set_ylim(nNrmGridTotal[0], nNrmGridTotal[idx_nlimit] + 50)
            axes[4].legend()

            # Third subplot: Difference plot
            pos_contour = axes[5].contourf(d, a, np.where(diff_trim > 0, diff_trim, np.nan), 
                                        levels=pos_levels, cmap='Blues', extend='max')

            # Contour for negative values
            neg_contour = axes[5].contourf(d, a, np.where(diff_trim < 0, diff_trim, np.nan), 
                                        levels=neg_levels, cmap=neg_colormap, extend='min')

            # Add colorbars for positive and negative values
            fig.colorbar(pos_contour, ax=axes[5], label='Positive Values', orientation='vertical', shrink=0.8)
            fig.colorbar(neg_contour, ax=axes[5], label='Negative Values', orientation='vertical', shrink=0.8)

            # Customization for third subplot
            axes[5].set_title('Inverse vFunc Difference (White to Strong Red for Negatives)')
            axes[5].set_ylabel('nNrmGridTotal')
            axes[5].set_xlabel('mNrmGridTotal')
            axes[5].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[5].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[5].plot(BoroCnstArt2D, nNrmGridNow, 'k--', label='Zero Grid mNrm')
            axes[5].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[5].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[5].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            # axes[2].plot(nNrmGridTotal, Lower_Bound_array_raw, 'b--', label='Lower Bound Raw')
            # axes[2].plot(nNrmGridTotal, Upper_Bound_array_raw, 'r--', label='Upper Bound Raw')
            # axes[2].plot(nNrmGridTotal, Bounds_array[:,0], label='Lower Bound')
            # axes[2].plot(nNrmGridTotal, Bounds_array[:,1], label='Upper Bound')
            axes[5].set_xlim(np.max([-3.0, mNrmGridTotal[0]]), 3.0)
            axes[5].set_ylim(0.0, 3.0)
            axes[5].legend()


            ### Plotting SS-Structure-refined
            pcm1 = axes[6].pcolormesh(mNrmGridTotal, nNrmGridNow, adjusting_redefined_array, cmap='viridis', shading='auto')
            fig.colorbar(pcm1, ax=axes[6], label='Adjusting Array Value')
            axes[6].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[6].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[6].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[6].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[6].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            axes[6].set_xlabel('mNrmGridTotal')
            axes[6].set_ylabel('nNrmGridTotal')
            axes[6].set_title('2D Plot of Adjusting Array (re-fined)')
            axes[6].set_xlim(mNrmGridTotal[0], mNrmGridTotal[-1])
            axes[6].set_ylim(nNrmGridNow[0], nNrmGridNow[-1])

            # Second subplot (identical, but you can change as needed)
            pcm2 = axes[7].pcolormesh(mNrmGridTotal, nNrmGridNow, adjusting_redefined_array, cmap='viridis', shading='auto')
            fig.colorbar(pcm2, ax=axes[7], label='Adjusting Array Value')
            axes[7].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[7].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[7].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[7].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[7].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            axes[7].set_xlabel('mNrmGridTotal')
            axes[7].set_ylabel('nNrmGridTotal')
            axes[7].set_title('2D Plot of Adjusting Array (re-fined)')
            axes[7].set_xlim(np.max([-(p2*2 + 10), mNrmGridTotal[0]]), mNrmGridTotal[idx_mlimit] + 50)
            axes[7].set_ylim(nNrmGridTotal[0], nNrmGridTotal[idx_nlimit] + 50)

            pcm3 = axes[8].pcolormesh(mNrmGridTotal, nNrmGridNow, adjusting_redefined_array, cmap='viridis', shading='auto')
            fig.colorbar(pcm3, ax=axes[8], label='Adjusting Array Value')
            axes[8].plot(lS, nNrmGridNow, label='Lower Bound')
            axes[8].plot(uS, nNrmGridNow, label='Upper Bound')
            axes[8].plot(center, nNrmGridNow, 'k--', label='Center of S,S Structure')
            axes[8].axhline(y=nNrmGridNow[idx_nlimit], color='red', linestyle='--', label='nNrmGridMax')
            axes[8].axvline(x=mNrmGridTotal[idx_mlimit], color='blue', linestyle='--', label='mNrmGridMax')
            axes[8].set_xlabel('mNrmGridTotal')
            axes[8].set_ylabel('nNrmGridTotal')
            axes[8].set_title('2D Plot of Adjusting Array (re-fined)')
            axes[8].set_xlim(np.max([-3.0, mNrmGridTotal[0]]), 3.0)
            axes[8].set_ylim(0.0, 3.0)

            plt.tight_layout()
            plt.show()
            print('SS-Structure')

        # ######################################################
        # if kink at last point (primary or secondary)
        if (cFunc_array[:, -1] < cFunc_array[:, -2]).any():
            print('cFunc becomes a decreasing function')
            Errors = 1
        if (dFunc_array[:, -1] < dFunc_array[:, -2]).any():
            print('dFunc becomes a decreasing function')    
            Errors = 1                  
        
    if alpha < 1:
        ### Functions without Limit point (for evaluation):
        ### Evaluation function only until p1. Find nNrmGridTotal idx and mNrmGridTotal idx at which they are p1
        nNrmGridTotal_eval = nNrmGridTotal[nNrmGridTotal <= p2 * 2]
        mNrmGridTotal_eval = mNrmGridTotal[mNrmGridTotal <= p2 * 2] #p2 * 2]

        vPFunc_eval_array = inv_vPFunc_array[:len(nNrmGridTotal_eval), :len(mNrmGridTotal_eval)]
        vPDFunc_eval_array = inv_vPDFunc_array[:len(nNrmGridTotal_eval), :len(mNrmGridTotal_eval)]
        vPFunc_eval = vPFunc_eval_array #LinearFast(vPFunc_eval_array, [nNrmGridTotal_eval, mNrmGridTotal_eval])
        vPDFunc_eval = vPDFunc_eval_array #LinearFast(vPDFunc_eval_array, [nNrmGridTotal_eval, mNrmGridTotal_eval])

        ### Make Functions
        cFunc = LinearFast(cFunc_array, [nNrmGridTotal, mNrmGridTotal])
        dFunc = LinearFast(dFunc_array, [nNrmGridTotal, mNrmGridTotal])
        aFunc = LinearFast(aFunc_array, [nNrmGridTotal, mNrmGridTotal])
        adjusting = LinearFast(adjusting_array, [nNrmGridTotal, mNrmGridTotal])
        inv_vPFunc = LinearFast(inv_vPFunc_array, [nNrmGridTotal, mNrmGridTotal])
        inv_vPDFunc = LinearFast(inv_vPDFunc_array, [nNrmGridTotal, mNrmGridTotal])

        ### Functions without Limit point:
        inv_vFunc = LinearFast(inv_vFunc_array, [nNrmGridTotal, mNrmGridTotal])
        if SearchCost == 0:
            vFunc = ValueFuncCRRA(inv_vFunc, CRRA)
        else:
            vFunc = LinearFast(vFunc_array, [nNrmGridTotal, mNrmGridTotal])


    ### If Depreciation Rate is one, we have an analytical solution for constraint part. We can construct the function
    ### Using the Lower Envelope over the constraint and unconstraint part
    if dDepr == 1 and alpha < 1:
        cFuncUnc = cFunc
        dFuncUnc = dFunc

        cFunc = LowerEnvelope2D(cFuncUnc, cFuncCnst, nan_bool=False)
        dFunc = dFuncUnc

        shape = (len(nNrmGridTotal), len(vFuncmNrmGridNow))
        inv_vFunc_array = np.zeros(shape)
        for i_n in range(len(nNrmGridTotal)):
            n_keep_aux = np.ones(len(vFuncmNrmGridNow) - 1) * nNrmGridTotal[i_n]

            # Compute expected value and marginal value on a grid of market resources
            cNrmNow = cFunc(n_keep_aux, vFuncmNrmGridNow[1:])
            dNrmNow = dFunc(n_keep_aux, vFuncmNrmGridNow[1:])

            # assets depend on adjustment vs no-adjustment
            aNrmNow = np.zeros(len(cNrmNow))
            x_aux = vFuncmNrmGridNow[1:] + (1 - adjC) * n_keep_aux
            inv_vFuncKeep_aux = inv_vFuncKeep(n_keep_aux, vFuncmNrmGridNow[1:])
            inv_vFuncAdj_aux = inv_vFuncAdj(x_aux)

            for c in range(len(cNrmNow)):
                if inv_vFuncKeep_aux[c] <= inv_vFuncAdj_aux[c] + tol:
                    aNrmNow[c] = x_aux[c] - cNrmNow[c] - dNrmNow[c]
                else:
                    aNrmNow[c] = vFuncmNrmGridNow[c+1] - cNrmNow[c]

            vNrmNow = CRRAutility(cNrmNow, dNrmNow) + utility(inv_EndOfPrdv(dNrmNow, aNrmNow), CRRA)
            # Construct the beginning-of-period value function
            inv_vFunc_array[i_n, 1:] = utility_inv(vNrmNow,
                                                       CRRA)  # value transformed through inverse utility
            inv_vFunc_array[i_n, 0] = 0

        inv_vFunc = LinearFast(inv_vFunc_array, [nNrmGridTotal, vFuncmNrmGridNow])
        vFunc = ValueFuncCRRA(inv_vFunc, CRRA)
        vPFunc = MargValueFuncCRRA_dur(cFunc, dFunc, alpha, d_ubar, CRRA)

    ####################################################################################################################
    # 2D adjuster/keeper asset policies for Monte Carlo poststates
    aFuncAdj_sim = None
    aFuncKeep_sim = None
    if alpha < 1:
        aFuncAdj_sim = LinearFast(
            aFuncAdj_SOL_array, [nNrmGridTotal, mNrmGridTotal]
        )
        if "aFuncKeep_SOL_array" in locals():
            aFuncKeep_sim = LinearFast(
                aFuncKeep_SOL_array, [nNrmGridTotal, mNrmGridTotal]
            )
    else:
        aFuncKeep_sim = aFunc

    # Assemble solution
    solution = DurableConsumerSolution(
        cFunc=cFunc,
        cFuncAdj=cFuncAdj,
        cFuncKeep=cFuncKeep,
        dFunc=dFunc,
        dFuncAdj=dFuncAdj,
        dFuncKeep=dFuncKeep,
        aFunc = aFunc,
        aFuncAdj = aFuncAdj_sim,
        aFuncKeep = aFuncKeep_sim,
        adjusting = adjusting,

        inv_vPDFunc = inv_vPDFunc,
        inv_vPFunc = inv_vPFunc,
        inv_vFunc=inv_vFunc,
        inv_vFuncAdj=inv_vFuncAdj,
        inv_vFuncKeep=inv_vFuncKeep,

        xNrmMin = xNrmMinNow,
        mNrmMin = mNrmMinNow,
        MPCmin = MPCminNow,
        hNrm = hNrmNow,
        MPCmax = MPCmaxEff,
        cFuncCnst = cFuncCnst,
        dFuncCnst = dFuncCnst,

        ### Evaluation
        vPFunc_eval = vPFunc_eval,
        vPDFunc_eval = vPDFunc_eval,

        ### Frictionless
        qFunc_Frictionless=qFunc_Frictionless,
        vFunc_Frictionless=vFunc_Frictionless,
        vPFunc_Frictionless=vPFunc_Frictionless,
    )
    if timeBOOL:
        _t['total'] = time.perf_counter() - _t0_total
        _sections = [
            ('init_batch',        'init: GPU next-period batch'),
            ('dot3',              'dot3: GPU FP32 probGrid @ vFunc'),
            ('probGrid_upload',   '  └ incl. H2D upload (1st call)'),
            ('lf1',               'lf1:  3x LinearFast (pre-wz)'),
            ('withzero',          'withzero: GPU batch fill'),
            ('lf2',               'lf2:  3x LinearFast (post-wz)'),
            ('eop_batch',         'eop_batch: GPU EoP sums (total)'),
            ('eop_grid',          '  └ grid build (CPU)'),
            ('eop_gpu_compute',   '  └ GPU H2D+interp+transforms'),
            ('eop_d2h',           '  └ GPU D2H download'),
            ('eop_total',         'eop_total (whole section)'),
            ('keeper',            'keeper'),
            ('adjuster',          'adjuster'),
            ('adjuster_limit',    'adjuster_limit'),
            ('total',             'TOTAL'),
        ]
        print("\n--- solve_DurableConsumer timing breakdown ---")
        print(f"  {'Section':<34} {'Time (s)':>8}")
        print(f"  {'-'*43}")
        for _key, _label in _sections:
            if _key in _t:
                print(f"  {_label:<34} {_t[_key]:>8.3f}")
        if 'probGrid_shape' in _t:
            print(f"  probGrid shape: {_t['probGrid_shape']}  "
                  f"vPFunc shape: {_t['vPFuncNext_shape']}")
        print(f"  GPU: {'ON' if _GPU_AVAILABLE else 'OFF'}")
        print("----------------------------------------------")
    return solution