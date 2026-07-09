# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed
#     formats: ipynb,py:light
#     rst2md: false
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import datetime

# print "*************************"
# print "Start time"
# print datetime.datetime.now().time()
# print "*************************"    
start_time = time.time()

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

# +
# here for now, should be
# from HARK import discontools or whatever name is chosen
from HARK.interpolation import LinearInterp

import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
# from HARK import Solution, AgentType
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.interpolation import LinearInterp
from  HARK.rewards import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv
from HARK.distribution import MeanOneLogNormal
from HARK.utilities import construct_assets_grid, make_grid_exp_mult
# from HARK.simulation import drawMeanOneLognormal
from math import sqrt
import scipy.stats as sp_stats


# might as well alias them utility, as we need CRRA
# should use these instead of hardcoded log (CRRA=1)
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityP_inv  = CRRAutilityP_inv
utility_inv   = CRRAutility_inv

Solution = ConsumerSolution
# +
# stuff that should be in HARK proper

def dcegmSegments(x, v):
    """
    Find index vectors `rise` and `fall` such that `rise` holds the indeces `i`
    such that x[i+1]>x[i] and `fall` holds indeces `j` such that either
    - x[j+1] < x[j] or,
    - x[j]>x[j-1] and v[j]<v[j-1].

    The vectors are essential to the DCEGM algorithm, as they definite the
    relevant intervals to be used to construct the upper envelope of potential
    solutions to the (necessary) first order conditions.

    Parameters
    ----------
    x : np.ndarray
        array of points where `v` is evaluated
    v : np.ndarray
        array of values of some function of `x`

    Returns
    -------
    rise : np.ndarray
        see description above
    fall : np.ndarray
        see description above
    """
    # NOTE: assumes that the first segment is in fact increasing (forced in EGM
    # by augmentation with the constrained segment).
    # elements in common grid g

    # Identify index intervals of falling and rising regions
    # We need these to construct the upper envelope because we need to discard
    # solutions from the inverted Euler equations that do not represent optimal
    # choices (the FOCs are only necessary in these models).
    #
    # `fall` is a vector of indeces that represent the first elements in all
    # of the falling segments (the curve can potentially fold several times)
    fall = np.empty(0, dtype=int) # initialize with empty and then add the last point below while-loop

    rise = np.array([0]) # Initialize such thatthe lowest point is the first grid point
    i = 1 # Initialize
    while i <= len(x) - 2:
        # Check if the next (`ip1` stands for i plus 1) grid point is below the
        # current one, such that the line is folding back.
        ip1_falls = x[i+1] < x[i] # true if grid decreases on index increment
        i_rose = x[i] > x[i-1] # true if grid decreases on index decrement
        val_fell = v[i] < v[i-1] # true if value rises on index decrement

        if (ip1_falls and i_rose) or (val_fell and i_rose):

            # we are in a region where the endogenous grid is decreasing or
            # the value function rises by stepping back in the grid.
            fall = np.append(fall, i) # add the index to the vector

            # We now iterate from the current index onwards until we find point
            # where resources rises again. Unfortunately, we need to check
            # each points, as there can be multiple spells of falling endogenous
            # grids, so we cannot use bisection or some other fast algorithm.
            k = i
            while x[k+1] < x[k]:
                k = k + 1
            # k now holds either the next index the starts a new rising
            # region, or it holds the length of M, `m_len`.

            rise = np.append(rise, k)

            # Set the index to the point where resources again is rising
            i = k

        i = i + 1

    fall = np.append(fall, len(v)-1)

    return rise, fall
# think! nanargmax makes everythign super ugly because numpy changed the wraning
# in all nan slices to a valueerror...it's nans, aaarghgghg
def calcMultilineEnvelope(M, C, V_T, commonM):
    """
    Do the envelope step of the DCEGM algorithm. Takes in market ressources,
    consumption levels, and inverse values from the EGM step. These represent
    (m, c) pairs that solve the necessary first order conditions. This function
    calculates the optimal (m, c, v_t) pairs on the commonM grid.

    Parameters
    ----------
    M : np.array
        market ressources from EGM step
    C : np.array
        consumption from EGM step
    V_T : np.array
        transformed values at the EGM grid
    commonM : np.array
        common grid to do upper envelope calculations on

    Returns
    -------


    """
    m_len = len(commonM)
    rise, fall = dcegmSegments(M, V_T)

    # Add the last point to the vector for convenience below
    num_kinks = len(fall) # number of kinks / falling EGM grids

    # Use these segments to sequentially find upper envelopes. commonVARNAME
    # means the VARNAME evaluated on the common grid with a cloumn for each kink
    # discovered in dcegmSegments. This means that commonVARNAME is a matrix
    # common grid length-by-number of segments to consider. In the end, we'll
    # use nanargmax over the columns to pick out the best (transformed) values.
    # This is why we fill the arrays with np.nan's.
    commonV_T = np.empty((m_len, num_kinks))
    commonV_T[:] = np.nan
    commonC = np.empty((m_len, num_kinks))
    commonC[:] = np.nan

    # Now, loop over all segments as defined by the "kinks" or the combination
    # of "rise" and "fall" indeces. These (rise[j], fall[j]) pairs define regions
    for j in range(num_kinks):
        # Find points in the common grid that are in the range of the points in
        # the interval defined by (rise[j], fall[j]).
        below = M[rise[j]] >= commonM # boolean array of bad indeces below
        above = M[fall[j]] <= commonM # boolen array of bad indeces above
        in_range = below + above == 0 # pick out elements that are neither

        # create range of indeces in the input arrays
        idxs = range(rise[j], fall[j]+1)
        # grab ressource values at the relevant indeces
        m_idx_j = M[idxs]

        # based in in_range, find the relevant ressource values to interpolate
        m_eval = commonM[in_range]

        # re-interpolate to common grid
        commonV_T[in_range,j] = LinearInterp(m_idx_j, V_T[idxs], lower_extrap=True)(m_eval)
        commonC[in_range,j]  = LinearInterp(m_idx_j, C[idxs], lower_extrap=True)(m_eval) # Interpolat econsumption also. May not be nesserary
    # for each row in the commonV_T matrix, see if all entries are np.nan. This
    # would mean that we have no valid value here, so we want to use this boolean
    # vector to filter out irrelevant entries of commonV_T.
    row_all_nan = np.array([np.all(np.isnan(row)) for row in commonV_T])
    # Now take the max of all these line segments.
    idx_max = np.zeros(commonM.size, dtype = int)
    idx_max[row_all_nan == False] = np.nanargmax(commonV_T[row_all_nan == False], axis=1)

    # prefix with upper for variable that are "upper enveloped"
    upperV_T = np.zeros(m_len)

    # Set the non-nan rows to the maximum over columns
    upperV_T[row_all_nan == False] = np.nanmax(commonV_T[row_all_nan == False, :], axis=1)
    # Set the rest to nan
    upperV_T[row_all_nan] = np.nan

    # Add the zero point in the bottom
    if np.isnan(upperV_T[0]):
        # in transformed space space, utility of zero-consumption (-inf) is 0.0
        upperV_T[0] = 0.0
        # commonM[0] is typically 0, so this is safe, but maybe it should be 0.0
        commonC[0]  = commonM[0]

    # Extrapolate if NaNs are introduced due to the common grid
    # going outside all the sub-line segments
    IsNaN = np.isnan(upperV_T)
    upperV_T[IsNaN] = LinearInterp(commonM[IsNaN == False], upperV_T[IsNaN == False])(commonM[IsNaN])


    LastBeforeNaN = np.append(np.diff(IsNaN)>0, 0)
    LastId = LastBeforeNaN*idx_max # Find last id-number
    idx_max[IsNaN] = LastId[IsNaN]
    # Linear index used to get optimal consumption based on "id"  from max
    ncols = commonC.shape[1]
    rowidx = np.cumsum(ncols*np.ones(len(commonM), dtype=int))-ncols
    idx_linear = np.unravel_index(rowidx+idx_max, commonC.shape)
    upperC = commonC[idx_linear]
    upperC[IsNaN] = LinearInterp(commonM[IsNaN==0], upperC[IsNaN==0])(commonM[IsNaN])

    # TODO calculate cross points of line segments to get the true vertical drops

    upperM = commonM.copy() # anticipatde this TODO

    return upperM, upperC, upperV_T

def OutcomeProbsCost_Search(Vals, sigma, beta, prob):
    '''
    Returns the optimal effort and cost function given the end-of-period 
    option-specific value functions `Vals`.
    Parameters
    ----------
    Vals : [numpy.array]
        A numpy.array that holds option-specific gothic-values at common 
        end-of-period grid points
    sigma : float
        A number that controls the size of the search frictions
    beta : float
        Discount factor
    prob : float
        Exogenous probability of receiving the option to switch
    Returns
    -------
    Probs : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    Costs : [numpy.array]
        A numpy.array that holds the cost associated with the optimal effort
    '''
    #====================
    # Assuming only two options
    #====================
    Vals[Vals==-np.inf] = -9999999
    Vals_diff = Vals[0]-Vals[1]
    Vals_diff[np.isnan(Vals_diff)] = 0
    
    if prob==1:
        effort = np.zeros(Vals_diff.shape)
    else:
        if sigma==0:
            effort = np.ones(Vals_diff.shape)
        else:
            effort = 1-1/np.exp(beta*(1-prob)*Vals_diff/sigma)
    
    effort[effort<0] = 0
    
    Probs = np.zeros(Vals.shape)
    Probs[0] = (1-effort)*prob+effort
    Probs[1] = (1-effort)*(1-prob)

    cost_effort = sigma*(effort - (1-effort)*np.log(1/(1-effort)))
    cost_effort[effort==1] = sigma*1.0

    return Probs, cost_effort


def calcLogSumChoiceProbs(Vals, sigma, smoothtype='type1'):
    '''
    Returns the final optimal value and choice probabilities given the choice
    specific value functions `Vals`. Probabilities are degenerate if sigma == 0.0.
    Parameters
    ----------
    Vals : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.
    sigma : float
        A number that controls the variance of the EV type1 taste shocks
    Returns
    -------
    V : [numpy.array]
        A numpy.array that holds the integrated value function.
    P : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    '''
    if (smoothtype!='type1' and smoothtype!='search'):
        raise ValueError("Argument smoothtype has to be either type1 or search")
        
    # Assumes that NaNs have been replaced by -numpy.inf or similar
    if sigma == 0.0:
        # We could construct a linear index here and use unravel_index.
        Pflat = np.argmax(Vals, axis=0)
        
        V = np.zeros(Vals[0].shape)
        Probs = np.zeros(Vals.shape)
        for i in range(Vals.shape[0]):
            optimalIndices = Pflat == i
            V[optimalIndices] = Vals[i][optimalIndices]
            Probs[i][optimalIndices] = 1
        return V, Probs
    
    if smoothtype=='search':
        #====================
        # Assuming only two choices
        #====================
        Vals[Vals==-np.inf] = -9999999
        Vals_diff = Vals[0]-Vals[1]
        Vals_diff[np.isnan(Vals_diff)] = 0
        effort    = 1-1/(np.exp(Vals_diff/sigma))
        effort[effort<0] = 0

        Probs = np.zeros(Vals.shape)
        Probs[0] = effort
        Probs[1] = (1-effort)

        cost_effort = sigma*(effort - (1-effort)*np.log(1/(1-effort)))
        cost_effort[effort==1] = sigma*1.0

        sumexp = effort*Vals[0] + (1-effort)*Vals[1]
        V      = -cost_effort + sumexp
    elif smoothtype=='type1':
        maxV = np.max(Vals, axis=0)

        # calculate maxV+sigma*log(sum_i=1^J exp((V[i]-maxV))/sigma)
        sumexp = np.sum(np.exp((Vals-maxV)/sigma), axis=0)
        LogSumV = np.log(sumexp)
        LogSumV = maxV + sigma*LogSumV
        V = LogSumV
        
        Probs = np.exp((Vals-LogSumV)/sigma)
    return V, Probs



# NOTE!!! calcExtraSaves needs to be updated to v3. It is not needed to produce the
# choice-specific functions
# Keep saveCommen==Fale until this is updated.
def calcExtraSaves(saveCommon, rs, ws, par, mGrid, tasteGrid, tasteWeights):
    if saveCommon:
        # To save the pre-discrete choice expected consumption and value function,
        # we need to interpolate onto the same grid between the two. We know that
        # ws.C and ws.V_T are on the ws.m grid, so we use that to interpolate.
        if par.TasteShkCount==1:
            Crs = LinearInterp(rs.m, rs.C)(mGrid)
            V_rs = numpy.divide(-1, LinearInterp(rs.m, rs.V_T)(mGrid))
            Cws = ws.C
            V_ws = numpy.divide(-1, ws.V_T)
            # Solution baseline
            V_baseline, P_baseline = calcLogSumChoiceProbs(numpy.stack((V_rs, V_ws)), 0.0, 'type1')
            C_baseline = (P_baseline*numpy.stack((Crs, Cws))).sum(axis=0)
            
            # Solution with exogenous probability for choice
            V_exo = (1-par.prob_opt)*V_ws + par.prob_opt*V_baseline
            C_exo = (1-par.prob_opt)*Cws + par.prob_opt*C_baseline
            
            # Solution after search
            V, P = calcLogSumChoiceProbs(numpy.stack((V_baseline, V_exo)), par.sigma_s, 'search')            
            effort_baseline = P[0, :]            
            C = effort_baseline*C_baseline + (1-effort_baseline)*C_exo
            V_T = numpy.divide(-1.0, V)
        else:
            mGrid_exp,tasteGrid_exp = np.meshgrid(mGrid,tasteGrid,indexing='ij')
            
            # Retired
            Crs = LinearInterp(rs.m, rs.C)(mGrid)
            Crs = numpy.matlib.repmat(numpy.vstack(Crs),1,par.TasteShkCount)
            
            V_rs = numpy.divide(-1, LinearInterp(rs.m, rs.V_T)(mGrid))
            V_rs = numpy.matlib.repmat(numpy.vstack(V_rs),1,par.TasteShkCount)

            # Worker
            Cws = ws.C
            Cws = numpy.matlib.repmat(np.vstack(Cws),1,par.TasteShkCount)
            
            V_ws_T = numpy.matlib.repmat(np.vstack(ws.V_T),1,par.TasteShkCount)
            V_ws_T_taste = V_ws_T*tasteGrid_exp
            V_ws = numpy.divide(-1, V_ws_T_taste)
            
            # Solution baseline
            V_baseline, P_baseline = calcLogSumChoiceProbs(numpy.stack((V_rs, V_ws)), 0.0, 'type1')
            C_baseline = (P_baseline*numpy.stack((Crs, Cws))).sum(axis=0)
            
            # Solution with exogenous probability for choice
            V_exo = (1-par.prob_opt)*V_ws + par.prob_opt*V_baseline
            C_exo = (1-par.prob_opt)*Cws + par.prob_opt*C_baseline
            
            # Expectation over taste shocks
            EV_baseline = numpy.dot(V_baseline,tasteWeights.T)
            EC_baseline = numpy.dot(C_baseline,tasteWeights.T)
            EV_exo = numpy.dot(V_exo,tasteWeights.T)
            EC_exo = numpy.dot(C_exo,tasteWeights.T)
            
            # Solution after search
            V, P = calcLogSumChoiceProbs(numpy.stack((EV_baseline, EV_exo)), par.sigma_s, 'search')            
            effort_baseline = P[0, :]            
            C = effort_baseline*EC_baseline + (1-effort_baseline)*EC_exo            
            V_T = numpy.divide(-1.0, V)
    else:
        C, V_T, P = None, None, None
    return C, V_T, P


RetiringDeatonParameters = namedtuple('RetiringDeatonParameters',
                                      'DiscFac CRRA DisUtil Rfree YRet YWork sigma_s sigma_t sigma_xi prob_opt TasteShkCount smoothtype')

class RetiringDeatonSolution(Solution):
    def __init__(self, ChoiceSols, M, C, V_T, P):
        self.m = M
        self.C = C
        self.V_T = V_T
        self.P = P
        self.ChoiceSols = ChoiceSols

class ChoiceSpecificSolution(Solution):
    def __init__(self, m, C, CFunc, V_T, V_TFunc):
        self.m = m
        self.C = C
        self.CFunc = CFunc
        self.V_T = V_T
        self.V_TFunc = V_TFunc

class RetiringDeaton(IndShockConsumerType):
    # Income integration method: "Discretize" (default) or "TransitionMethod"
    IncomeDiscMethod = "Discretize"
    bGridMin = 0.0
    bGridMax = 48.0
    bGridCount = 300
    mGridMin = 1e-4
    mGridMax = 52.0
    mGridCount = 310
    IncUnemp_Std = 0.1

    def __init__(self, **kwds):

        IndShockConsumerType.__init__(self, **kwds)

        # Ensure income-integration attributes exist on the instance for HARK's solver
        for attr, default in (
            ('IncomeDiscMethod', RetiringDeaton.IncomeDiscMethod),
            ('bGridMin', RetiringDeaton.bGridMin),
            ('bGridMax', RetiringDeaton.bGridMax),
            ('bGridCount', RetiringDeaton.bGridCount),
            ('mGridMin', RetiringDeaton.mGridMin),
            ('mGridMax', RetiringDeaton.mGridMax),
            ('mGridCount', RetiringDeaton.mGridCount),
            ('IncUnemp_Std', RetiringDeaton.IncUnemp_Std),
        ):
            if attr not in self.__dict__:
                setattr(self, attr, default)
        if 'income_bGrid' not in self.__dict__:
            self.income_bGrid = None
        if 'income_mGrid' not in self.__dict__:
            self.income_mGrid = None
        if 'incomeTranMatrix' not in self.__dict__:
            self.incomeTranMatrix = [None] * self.T_cycle

        self.time_inv = ['aXtraGrid', 'mGrid', 'EGMVector', 'par', 'Util', 'UtilP',
                         'UtilP_inv', 'saveCommon', 'IncomeDiscMethod', 'income_bGrid',
                         'income_mGrid']

        self.time_vary = self.time_vary + ['incomeTranMatrix']

        self.par = RetiringDeatonParameters(self.DiscFac, self.CRRA, self.DisUtil, self.Rfree, self.YRet, self.YWork, self.sigma_s, self.sigma_t, self.sigma_xi, self.prob_opt, self.TasteShkCount, self.smoothtype)
        # d == 2 is working
        # - 10.0 moves curve down to improve linear interpolation
        self.Util = lambda c, d: utility(c, self.CRRA) - self.par.DisUtil*(d-1) - 10.0
        self.UtilP = lambda c, d: utilityP(c, self.CRRA) # we require CRRA 1.0 for now...
        self.UtilP_inv = lambda u, d: utilityP_inv(u, self.CRRA) # ... so ...

        self.solve_one_period = solveRetiringDeaton

    def pre_solve(self):
        super().pre_solve()
        self._setup_income_integration()
        self.updateLast()

    def _setup_income_integration(self):
        """Build dense b/m grids and transition matrices when using TransitionMethod."""
        if self.IncomeDiscMethod != "TransitionMethod":
            self.income_bGrid = None
            self.income_mGrid = None
            self.incomeTranMatrix = [None] * self.T_cycle
            return

        bGrid = make_grid_exp_mult(self.bGridMin, self.bGridMax, self.bGridCount, timestonest=1)
        mGrid = make_grid_exp_mult(self.mGridMin, self.mGridMax, self.mGridCount, timestonest=1)
        b_offset = self.Rfree * self.aXtraGrid[0]
        m_offset = self.YWork + b_offset
        self.income_bGrid = bGrid + b_offset
        self.income_mGrid = mGrid + m_offset

        self.incomeTranMatrix = []
        for t in range(self.T_cycle):
            tran_std = self.TranShkStd[t]
            unemp_prb = self.UnempPrb[t] if hasattr(self.UnempPrb, '__getitem__') else self.UnempPrb
            inc_unemp = self.IncUnemp[t] if hasattr(self.IncUnemp, '__getitem__') else self.IncUnemp
            if unemp_prb > 0.0:
                probGrid = updateBM_TranMatrix_withUnemp(
                    tran_std, unemp_prb, inc_unemp, self.IncUnemp_Std,
                    self.income_bGrid, self.income_mGrid, YWork=self.YWork,
                )
            else:
                probGrid = updateBM_TranMatrix(
                    tran_std, self.income_bGrid, self.income_mGrid, YWork=self.YWork,
                )
            self.incomeTranMatrix.append(probGrid)

    def updateLast(self):
        """
        Updates grids and functions according to the given model parameters, and
        solves the last period.

        Parameters
        ---------
        None

        Returns
        -------
        None
        """

        self.mGrid = (self.aXtraGrid-self.aXtraGrid[0])*1.5
        self.EGMVector = numpy.zeros(self.EGMCount)
        self.tasteGrid_last = MeanOneLogNormal(sigma=self.sigma_xi).discretize(self.TasteShkCount)
        
        ChoiceSols = tuple(self.solveLastChoiceSpecific(choice) for choice in (1, 2))

        C, V_T, P = calcExtraSaves(self.saveCommon, ChoiceSols[0], ChoiceSols[1], self.par, self.mGrid, self.tasteGrid_last.atoms[0], self.tasteGrid_last.pmv)

        self.solution_terminal = RetiringDeatonSolution(ChoiceSols, self.mGrid.copy(), C, V_T, P)

    def solveLastChoiceSpecific(self, curChoice):
        """
        Solves the last period of a working agent.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        
        m, C = self.mGrid.copy(), self.mGrid.copy() # consume everything

        V_T = numpy.divide(-1.0, self.Util(self.mGrid, curChoice))

        # Interpolants
        CFunc = lambda coh: LinearInterp(m, C)(coh)
        V_TFunc = lambda coh: LinearInterp(m, V_T)(coh)
            
        return ChoiceSpecificSolution(m, C, CFunc, V_T, V_TFunc)

# +
# Functions used to solve each period
def solveRetiringDeaton(solution_next, LivPrb, PermGroFac, IncShkDstn, PermShkDstn, TranShkDstn, aXtraGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv, saveCommon, IncomeDiscMethod, income_bGrid, income_mGrid, incomeTranMatrix):
    """
    Solves a period of problem defined by the RetiringDeaton AgentType. It uses
    DCEGM to solve the mixed choice problem.

    Parameters
    ----------

    Returns
    -------

    """
    TasteShkDstn = MeanOneLogNormal(sigma=par.sigma_xi).discretize(par.TasteShkCount)
    rs = solveRetiredDeaton(solution_next, aXtraGrid, EGMVector, par, Util, UtilP, UtilP_inv)
    ws = solveWorkingDeaton(
        solution_next, aXtraGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv,
        TranShkDstn.atoms[0], TranShkDstn.pmv, TasteShkDstn.atoms[0], TasteShkDstn.pmv,
        IncomeDiscMethod=IncomeDiscMethod, income_bGrid=income_bGrid, income_mGrid=income_mGrid,
        incomeTranMatrix=incomeTranMatrix,
    )

    C, V_T, P = calcExtraSaves(saveCommon, rs, ws, par, mGrid, TasteShkDstn.atoms[0], TasteShkDstn.pmv)

    return RetiringDeatonSolution((rs, ws), mGrid, C, V_T, P)




def calcEGMStep(EGMVector, aXtraGrid, EV_tp1, cost_t, EUtilP_tp1, par, Util, UtilP, UtilP_inv, choice):

    # Allocate arrays
    m_t = numpy.copy(EGMVector)
    C_t = numpy.copy(EGMVector)
    Ev_t = numpy.copy(EGMVector)
    effort_cost_t = numpy.copy(EGMVector)

    # Calculate length of constrained region
    conLen = len(m_t)-len(aXtraGrid)

    # Calculate the expected marginal utility and expected value function
    Ev_t[conLen:] = EV_tp1
    
    # and the cost function
    effort_cost_t[conLen:] = cost_t

    # EGM step
    C_t[conLen:] = UtilP_inv(par.DiscFac*EUtilP_tp1, choice)
    m_t[conLen:] = aXtraGrid + C_t[conLen:]

    # Add points to M (and C) to solve between 0 and the first point EGM finds
    # (that is add the solution in the segment where the agent is constrained)
    m_t[0:conLen] = numpy.linspace(0, m_t[conLen]*0.99, conLen)
    C_t[0:conLen] = m_t[0:conLen]

    # Since a is the save everywhere (=0) the expected continuation value
    # is the same for all indeces 0:conLen
    Ev_t[0:conLen] = Ev_t[conLen+1]
    # Same holds for the cost function
    effort_cost_t[0:conLen] = effort_cost_t[conLen+1]

    return m_t, C_t, Ev_t, effort_cost_t

def solveRetiredDeaton(solution_next, aXtraGrid, EGMVector, par, Util, UtilP, UtilP_inv):
    choice = 1
    rs_tp1 = solution_next.ChoiceSols[0]

    # Next-period initial wealth given exogenous aXtraGrid
    m_tp1 = par.Rfree*aXtraGrid + par.YRet

    # Prepare variables for EGM step
    EC_tp1 = rs_tp1.CFunc(m_tp1)
    EV_T_tp1 = rs_tp1.V_TFunc(m_tp1)
    EV_tp1 = numpy.divide(-1.0, EV_T_tp1)

    EUtilP_tp1 = par.Rfree*UtilP(EC_tp1, choice)
    cost_t = numpy.zeros(EUtilP_tp1.shape)

    m_t, C_t, Ev, cost_t_cons = calcEGMStep(EGMVector, aXtraGrid, EV_tp1, cost_t, EUtilP_tp1, par, Util, UtilP, UtilP_inv, choice)

    V_T = numpy.divide(-1.0, Util(C_t, choice) - cost_t_cons + par.DiscFac*Ev)

    CFunc = LinearInterp(m_t, C_t)
    V_TFunc = LinearInterp(m_t, V_T)

    return ChoiceSpecificSolution(m_t, C_t, CFunc, V_T, V_TFunc)



def updateBM_TranMatrix(TranShkStd, bGrid, mGrid, YWork=1.0):
    """
    Transition matrix from b-grid to m-grid for mean-one lognormal transitory shocks.
    Rows index b, columns index m; each row sums to one.
    """
    s = TranShkStd
    mu = -0.5 * s**2
    lognorm_dist = sp_stats.lognorm(s, scale=numpy.exp(mu))
    b, m = numpy.meshgrid(bGrid, mGrid, indexing='ij')
    tran_shock = (m - b) / YWork
    probGrid = lognorm_dist.pdf(tran_shock) / YWork
    for i_b in range(len(bGrid)):
        row_sum = numpy.sum(probGrid[i_b])
        probGrid[i_b] = probGrid[i_b] / numpy.max([row_sum, 1e-6])
    return probGrid


def updateBM_TranMatrix_withUnemp(TranShkStd, UnempPrb, IncUnemp, IncUnemp_Std, bGrid, mGrid, YWork=1.0):
    """
    Transition matrix allowing unemployment via a continuous mixture PDF (no point mass).
    """
    mu_transformed = (1.0 - UnempPrb * IncUnemp) / (1.0 - UnempPrb)
    s_emp = TranShkStd
    mu_emp = numpy.log(mu_transformed) - 0.5 * s_emp**2
    s_unemp = IncUnemp_Std
    mu_unemp = numpy.log(IncUnemp) - 0.5 * s_unemp**2
    lognorm_dist_emp = sp_stats.lognorm(s=s_emp, scale=numpy.exp(mu_emp))
    lognorm_dist_unemp = sp_stats.lognorm(s=s_unemp, scale=numpy.exp(mu_unemp))

    def mixture_pdf(tran_shock):
        return ((1.0 - UnempPrb) * lognorm_dist_emp.pdf(tran_shock)
                + UnempPrb * lognorm_dist_unemp.pdf(tran_shock))

    b, m = numpy.meshgrid(bGrid, mGrid, indexing='ij')
    tran_shock = (m - b) / YWork
    probGrid = mixture_pdf(tran_shock) / YWork
    for i_b in range(len(bGrid)):
        row_sum = numpy.sum(probGrid[i_b])
        probGrid[i_b] = probGrid[i_b] / numpy.max([row_sum, 1e-6])
    return probGrid


def _integrate_via_transition(values_on_m, probGrid, income_bGrid, aXtraGrid, Rfree):
    """Apply b->m transition matrix, then interpolate onto b=Rfree*a."""
    values_on_b = probGrid @ values_on_m.reshape(-1, 1)
    values_on_b = values_on_b.flatten()
    b_at_a = Rfree * aXtraGrid
    return numpy.interp(b_at_a, income_bGrid, values_on_b)


def _finish_solveWorkingDeaton(EGMVector, aXtraGrid, mGrid, par, Util, UtilP, UtilP_inv,
                               EV_tp1, cost_t, EUtilP_tp1):
    """Common EGM and envelope step for solveWorkingDeaton."""
    choice = 2
    m_t, C_t, Ev, cost_t_cons = calcEGMStep(
        EGMVector, aXtraGrid, EV_tp1, cost_t, EUtilP_tp1, par, Util, UtilP, UtilP_inv, choice,
    )
    V_T = numpy.divide(-1.0, Util(C_t, choice) - cost_t_cons + par.DiscFac * Ev)
    m_t, C_t, V_T = calcMultilineEnvelope(m_t, C_t, V_T, mGrid)
    CFunc = LinearInterp(m_t, C_t, lower_extrap=True)
    V_TFunc = LinearInterp(m_t, V_T, lower_extrap=True)
    return ChoiceSpecificSolution(m_t, C_t, CFunc, V_T, V_TFunc)


def solveWorkingDeaton(solution_next, aXtraGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv,
                       TranInc, TranIncWeights, TasteShk, TasteShkWeights,
                       IncomeDiscMethod="Discretize", income_bGrid=None, income_mGrid=None,
                       incomeTranMatrix=None):
    if IncomeDiscMethod == "TransitionMethod":
        return _solveWorkingDeaton_transition(
            solution_next, aXtraGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv,
            TasteShk, TasteShkWeights, income_bGrid, income_mGrid, incomeTranMatrix,
        )

    choice = 2

    choiceCount = len(solution_next.ChoiceSols)

    # Next-period initial wealth given exogenous aXtraGrid
    # This needs to be made more general like the rest of the code
    if par.TasteShkCount==1:
        a_mat,tran_mat = numpy.meshgrid(aXtraGrid,TranInc,indexing='ij')
        mrs_tp1 = par.Rfree*a_mat + par.YWork*tran_mat
        mws_tp1 = par.Rfree*a_mat + par.YWork*tran_mat
    else:
        a_mat,tran_mat,taste_mat = numpy.meshgrid(aXtraGrid,TranInc,TasteShk,indexing='ij')
        mrs_tp1 = par.Rfree*a_mat + par.YWork*tran_mat
        mws_tp1 = par.Rfree*a_mat + par.YWork*tran_mat
        
    m_tp1s = (mrs_tp1, mws_tp1)

    # Prepare variables for EGM step
    # 1st entry is for worker retiring in the next period
    # 2nd entry is for worker continuing to work in the next period, i.e. D_t = {W}
    if par.TasteShkCount==1:
        C_tp1s = tuple(solution_next.ChoiceSols[d].CFunc(m_tp1s[d]) for d in range(choiceCount))
        Vs = tuple(numpy.divide(-1.0, solution_next.ChoiceSols[d].V_TFunc(m_tp1s[d])) for d in range(choiceCount))
    else:
        # Since interpolated values of the outcome-specific functions along the
        # taste dimension are all the same, we just do an interpolation along 
        # the (A_t,TransShk_t+1) dimensions and repeat the values along the 
        # (TasteShk_t) dimension.
        m_tp1s_R = m_tp1s[0][:,:,0]
        m_tp1s_W = m_tp1s[1][:,:,0]
        C_tp1_R_comp = solution_next.ChoiceSols[0].CFunc(m_tp1s_R)
        C_tp1_W_comp = solution_next.ChoiceSols[1].CFunc(m_tp1s_W)
        V_T_tp1_R_comp = solution_next.ChoiceSols[0].V_TFunc(m_tp1s_R)
        V_T_tp1_W_comp = solution_next.ChoiceSols[1].V_TFunc(m_tp1s_W)
        
        C_tp1_R = numpy.repeat(C_tp1_R_comp[:, :, numpy.newaxis], par.TasteShkCount, axis=2)
        C_tp1_W = numpy.repeat(C_tp1_W_comp[:, :, numpy.newaxis], par.TasteShkCount, axis=2)
        
        V_T_tp1_R = numpy.repeat(V_T_tp1_R_comp[:, :, numpy.newaxis], par.TasteShkCount, axis=2)
        V_T_tp1_W = numpy.repeat(V_T_tp1_W_comp[:, :, numpy.newaxis], par.TasteShkCount, axis=2)
                
        C_tp1s = (C_tp1_R,C_tp1_W)
        
        V_tp1_R = numpy.divide(-1.0, V_T_tp1_R)
        V_T_tp1_W_taste = taste_mat*V_T_tp1_W
        V_tp1_W = numpy.divide(-1.0, V_T_tp1_W_taste)
        Vs = (V_tp1_R,V_tp1_W)
    
    V_tp1_R = Vs[0]
    V_tp1_W = Vs[1]
    V_tp1_RW,inds_tp1_RW = calcLogSumChoiceProbs(numpy.stack((V_tp1_R,V_tp1_W)), 0.0, smoothtype='type1')
    ind0_tp1 = inds_tp1_RW[0,:]
    ind1_tp1 = inds_tp1_RW[1,:]    
    
    if par.TasteShkCount==1:
        gothV1 = numpy.dot(V_tp1_RW,TranIncWeights.T)
        gothV2 = numpy.dot(V_tp1_W,TranIncWeights.T)
    else:
        gothV1_pre = numpy.dot(V_tp1_RW,TasteShkWeights.T)
        gothV1 = numpy.dot(gothV1_pre,TranIncWeights.T)
        gothV2_pre = numpy.dot(V_tp1_W,TasteShkWeights.T)
        gothV2 = numpy.dot(gothV2_pre,TranIncWeights.T)
    
    # Effort this period that affects the likelihood of receiving the option
    # next period
    effort_all, cost_t = OutcomeProbsCost_Search(numpy.stack((gothV1,gothV2)), par.sigma_s, par.DiscFac, par.prob_opt)
    effort_t = effort_all[0,:]
    P1 = (1-effort_t)*par.prob_opt + effort_t
    P2 = (1-effort_t)*(1-par.prob_opt)
    
    C_tp1_R = C_tp1s[0]
    C_tp1_W = C_tp1s[1]
    
    if par.TasteShkCount==1:
        RHS1_pre = ind0_tp1*UtilP(C_tp1_R, 1) + ind1_tp1*UtilP(C_tp1_W, 2)
        RHS1 = P1*numpy.dot(RHS1_pre, TranIncWeights.T)
        RHS2_pre = UtilP(C_tp1_W, 2)
        RHS2 = P2*numpy.dot(RHS2_pre, TranIncWeights.T)
        RHS = par.Rfree*(RHS1+RHS2)        
    else:
        RHS1_pre1 = ind0_tp1*UtilP(C_tp1_R, 1) + ind1_tp1*taste_mat*UtilP(C_tp1_W, 2)
        RHS1_pre2 = numpy.dot(RHS1_pre1, TasteShkWeights.T)
        RHS1 = P1*numpy.dot(RHS1_pre2, TranIncWeights.T)        
        RHS2_pre1 = taste_mat*UtilP(C_tp1_W, 2)
        RHS2_pre2 = numpy.dot(RHS2_pre1, TasteShkWeights.T)
        RHS2 = P2*numpy.dot(RHS2_pre2, TranIncWeights.T)
        RHS = par.Rfree*(RHS1+RHS2)
        
                
    EUtilP_tp1 = RHS
    EV_tp1 = P1*gothV1 + P2*gothV2

    return _finish_solveWorkingDeaton(
        EGMVector, aXtraGrid, mGrid, par, Util, UtilP, UtilP_inv, EV_tp1, cost_t, EUtilP_tp1,
    )


def _solveWorkingDeaton_transition(solution_next, aXtraGrid, mGrid, EGMVector, par, Util, UtilP,
                                   UtilP_inv, TasteShk, TasteShkWeights, income_bGrid,
                                   income_mGrid, probGrid):
    """Working-period solver using Matt White's transition-matrix integration."""
    choiceCount = len(solution_next.ChoiceSols)
    income_mGrid = numpy.asarray(income_mGrid)
    income_bGrid = numpy.asarray(income_bGrid)

    CFunc_R = solution_next.ChoiceSols[0].CFunc
    CFunc_W = solution_next.ChoiceSols[1].CFunc
    V_TFunc_R = solution_next.ChoiceSols[0].V_TFunc
    V_TFunc_W = solution_next.ChoiceSols[1].V_TFunc

    if par.TasteShkCount == 1:
        C_R_m = CFunc_R(income_mGrid)
        C_W_m = CFunc_W(income_mGrid)
        V_R_m = numpy.divide(-1.0, V_TFunc_R(income_mGrid))
        V_W_m = numpy.divide(-1.0, V_TFunc_W(income_mGrid))
        V_RW_m, inds_m = calcLogSumChoiceProbs(
            numpy.stack((V_R_m, V_W_m)), 0.0, smoothtype='type1',
        )
        ind0_m = inds_m[0]
        ind1_m = inds_m[1]
        UtilP_mixed_m = ind0_m * UtilP(C_R_m, 1) + ind1_m * UtilP(C_W_m, 2)

        gothV1 = _integrate_via_transition(V_RW_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)
        gothV2 = _integrate_via_transition(V_W_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)
        RHS1 = _integrate_via_transition(UtilP_mixed_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)
        RHS2 = _integrate_via_transition(UtilP(C_W_m, 2), probGrid, income_bGrid, aXtraGrid, par.Rfree)
    else:
        TasteShk = numpy.asarray(TasteShk)
        TasteShkWeights = numpy.asarray(TasteShkWeights)
        C_R_m = CFunc_R(income_mGrid)
        C_W_m = CFunc_W(income_mGrid)
        V_R_m = numpy.divide(-1.0, V_TFunc_R(income_mGrid))
        V_T_W_m = V_TFunc_W(income_mGrid)
        V_W_mt = numpy.divide(-1.0, TasteShk[numpy.newaxis, :] * V_T_W_m[:, numpy.newaxis])
        V_R_mt = numpy.broadcast_to(V_R_m[:, numpy.newaxis], V_W_mt.shape)
        V_RW_mt, inds_mt = calcLogSumChoiceProbs(
            numpy.stack((V_R_mt, V_W_mt)), 0.0, smoothtype='type1',
        )
        ind0_mt = inds_mt[0]
        ind1_mt = inds_mt[1]
        UtilP_mixed_mt = (
            ind0_mt * UtilP(C_R_m[:, numpy.newaxis], 1)
            + ind1_mt * TasteShk[numpy.newaxis, :] * UtilP(C_W_m[:, numpy.newaxis], 2)
        )
        UtilP_mixed_m = numpy.dot(UtilP_mixed_mt, TasteShkWeights)
        V_RW_m = numpy.dot(V_RW_mt, TasteShkWeights)
        V_W_m = numpy.dot(V_W_mt, TasteShkWeights)
        RHS2_m = TasteShk[numpy.newaxis, :] * UtilP(C_W_m[:, numpy.newaxis], 2)
        RHS2_m = numpy.dot(RHS2_m, TasteShkWeights)

        gothV1 = _integrate_via_transition(V_RW_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)
        gothV2 = _integrate_via_transition(V_W_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)
        RHS1 = _integrate_via_transition(UtilP_mixed_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)
        RHS2 = _integrate_via_transition(RHS2_m, probGrid, income_bGrid, aXtraGrid, par.Rfree)

    effort_all, cost_t = OutcomeProbsCost_Search(
        numpy.stack((gothV1, gothV2)), par.sigma_s, par.DiscFac, par.prob_opt,
    )
    effort_t = effort_all[0, :]
    P1 = (1.0 - effort_t) * par.prob_opt + effort_t
    P2 = (1.0 - effort_t) * (1.0 - par.prob_opt)
    EUtilP_tp1 = par.Rfree * (P1 * RHS1 + P2 * RHS2)
    EV_tp1 = P1 * gothV1 + P2 * gothV2

    return _finish_solveWorkingDeaton(
        EGMVector, aXtraGrid, mGrid, par, Util, UtilP, UtilP_inv, EV_tp1, cost_t, EUtilP_tp1,
    )


# -
################################################################################
################################################################################
# # +
# CRRA = 1.0
# DiscFac = 0.98
# Rfree = 1.0
# DisUtil = 1.0
# T = 20
# sigma_s = 0.0
# sigma_t = 0.0
# sigma_xi = 0.0
# prob_opt = 1.0
# smoothtype = 'type1'

# aXtraMin = 1e-6
# aXtraMax = 400.0
# aXtraCount = 6000
# aXtraExtra = () # this is additional points to add (for precision around know problematic areas)
# aXtraNestFac = 1 # this is the times to nest

# YWork = 20.0
# YRet = 0.0 # normalized relative to work
# TranShkStd = [0.000]*T
# TranShkCount = 1
# PermShkStd = [0.0]*T
# PermShkCount = 1
# TasteShkCount = 1
# EGMCount = 7000
# saveCommon = False #True #False
# T_retire = 0 # not applicable, it's endogenous
# T_cycle = T
# LivPrb = [1.0]*T
# PermGroFac = [1.0]*T
# UnempPrb = 0.0
# UnempPrbRet = 0.0
# IncUnemp = 0.0
# IncUnempRet = 0.0


# retiring_params = {'CRRA' : CRRA,
#                    'Rfree' : Rfree,
#                    'DiscFac' : DiscFac,
#                    'DisUtil' : DisUtil,
#                    'T' : T,
#                    'sigma_s' : sigma_s,
#                    'sigma_t' : sigma_t,
#                    'sigma_xi' : sigma_xi,
#                    'prob_opt' : prob_opt,
#                    'smoothtype' : smoothtype,
#                    'aXtraMin' : aXtraMin,
#                    'aXtraMax' : aXtraMax,
#                    'aXtraCount' : aXtraCount,
#                    'aXtraExtra' : aXtraExtra,
#                    'aXtraNestFac' : aXtraNestFac,
#                    'UnempPrb' : UnempPrb,
#                    'UnempPrbRet' : UnempPrbRet,
#                    'IncUnemp' : IncUnemp,
#                    'IncUnempRet' : IncUnempRet,
#                    'YWork' : YWork,
#                    'YRet' : YRet,
#                    'TranShkStd' : TranShkStd,
#                    'TranShkCount' : TranShkCount,
#                    'PermShkStd' : PermShkStd,
#                    'PermShkCount' : PermShkCount,
#                    'TasteShkCount' : TasteShkCount,
#                    'EGMCount' : EGMCount,
#                    'T_retire' : T_retire,
#                    'T_cycle' : T_cycle,
#                    'LivPrb' : LivPrb,
#                    'PermGroFac' : PermGroFac,
#                    'saveCommon' : saveCommon}
# # -

# #model = dcegm.RetiringDeaton(saveCommon = True)
# model = RetiringDeaton(**retiring_params)


# model.solve()

# ## MAIN FIGURE
# ### +
# #
# ###================
# ##
# #fig1_1_params = copy.deepcopy(retiring_params)
# #fig1_1_params['Rfree'] = 1.0
# #fig1_1_params['DiscFac'] = 0.98
# #fig1_1_params['prob_opt'] = 1.0
# #fig1_1_params['sigma_xi'] = 0.000
# #fig1_1_params['TasteShkCount'] = 1
# #fig1_1_params['TranShkStd'] = [sqrt(0.000)]*fig1_1_params['T']
# #fig1_1_params['TranShkCount'] = 1
# #modelfig1_1 = RetiringDeaton(**fig1_1_params)
# #modelfig1_1.solve()
# #
# #fig1_2_params = copy.deepcopy(fig1_1_params)
# #fig1_2_params['sigma_s'] = 0.01
# #modelfig1_2 = RetiringDeaton(**fig1_2_params)
# #modelfig1_2.solve()
# #
# #fig1_3_params = copy.deepcopy(fig1_1_params)
# #fig1_3_params['sigma_s'] = 0.15
# #modelfig1_3 = RetiringDeaton(**fig1_3_params)
# #modelfig1_3.solve()
# #
# #fig1_4_params = copy.deepcopy(fig1_1_params)
# #fig1_4_params['sigma_s'] = 0.5
# #modelfig1_4 = RetiringDeaton(**fig1_4_params)
# #modelfig1_4.solve()
# #
# #fig1_5_params = copy.deepcopy(fig1_1_params)
# #fig1_5_params['sigma_s'] = 1.0
# #modelfig1_5 = RetiringDeaton(**fig1_5_params)
# #modelfig1_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig1_1.mGrid, modelfig1_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig1_2.mGrid, modelfig1_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig1_3.mGrid, modelfig1_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig1_4.mGrid, modelfig1_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig1_5.mGrid, modelfig1_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 1.0$, $\sigma_{\\xi} = 0.0$, $\sigma_{\eta} = 0.0$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p1_xi0_eta0.pdf", dpi=150)
# #plt.show()
# #
# ##================
# #
# ##
# #fig1a_1_params = copy.deepcopy(retiring_params)
# #fig1a_1_params['Rfree'] = 1.0
# #fig1a_1_params['DiscFac'] = 0.98
# #fig1a_1_params['prob_opt'] = 0.5
# #fig1a_1_params['sigma_xi'] = 0.000
# #fig1a_1_params['TasteShkCount'] = 1
# #fig1a_1_params['TranShkStd'] = [sqrt(0.000)]*fig1a_1_params['T']
# #fig1a_1_params['TranShkCount'] = 1
# #modelfig1a_1 = RetiringDeaton(**fig1a_1_params)
# #modelfig1a_1.solve()
# #
# #fig1a_2_params = copy.deepcopy(fig1a_1_params)
# #fig1a_2_params['sigma_s'] = 0.01
# #modelfig1a_2 = RetiringDeaton(**fig1a_2_params)
# #modelfig1a_2.solve()
# #
# #fig1a_3_params = copy.deepcopy(fig1a_1_params)
# #fig1a_3_params['sigma_s'] = 0.15
# #modelfig1a_3 = RetiringDeaton(**fig1a_3_params)
# #modelfig1a_3.solve()
# #
# #fig1a_4_params = copy.deepcopy(fig1a_1_params)
# #fig1a_4_params['sigma_s'] = 0.5
# #modelfig1a_4 = RetiringDeaton(**fig1a_4_params)
# #modelfig1a_4.solve()
# #
# #fig1a_5_params = copy.deepcopy(fig1a_1_params)
# #fig1a_5_params['sigma_s'] = 1.0
# #modelfig1a_5 = RetiringDeaton(**fig1a_5_params)
# #modelfig1a_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig1a_1.mGrid, modelfig1a_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig1a_2.mGrid, modelfig1a_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig1a_3.mGrid, modelfig1a_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig1a_4.mGrid, modelfig1a_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig1a_5.mGrid, modelfig1a_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.5$, $\sigma_{\\xi} = 0.0$, $\sigma_{\eta} = 0.0$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p0p5_xi0_eta0.pdf", dpi=150)
# #plt.show()
# #
# ##================
# #
# ##
# ###================
# ##
# #fig2_1_params = copy.deepcopy(retiring_params)
# #fig2_1_params['Rfree'] = 1.0
# #fig2_1_params['DiscFac'] = 0.98
# #fig2_1_params['prob_opt'] = 0.0
# #fig2_1_params['sigma_xi'] = 0.000
# #fig2_1_params['TasteShkCount'] = 1
# #fig2_1_params['TranShkStd'] = [sqrt(0.000)]*fig2_1_params['T']
# #fig2_1_params['TranShkCount'] = 1
# #modelfig2_1 = RetiringDeaton(**fig2_1_params)
# #modelfig2_1.solve()
# #
# #fig2_2_params = copy.deepcopy(fig2_1_params)
# #fig2_2_params['sigma_s'] = 0.01
# #modelfig2_2 = RetiringDeaton(**fig2_2_params)
# #modelfig2_2.solve()
# #
# #fig2_3_params = copy.deepcopy(fig2_1_params)
# #fig2_3_params['sigma_s'] = 0.15
# #modelfig2_3 = RetiringDeaton(**fig2_3_params)
# #modelfig2_3.solve()
# #
# #fig2_4_params = copy.deepcopy(fig2_1_params)
# #fig2_4_params['sigma_s'] = 0.5
# #modelfig2_4 = RetiringDeaton(**fig2_4_params)
# #modelfig2_4.solve()
# #
# #fig2_5_params = copy.deepcopy(fig2_1_params)
# #fig2_5_params['sigma_s'] = 1.0
# #modelfig2_5 = RetiringDeaton(**fig2_5_params)
# #modelfig2_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig2_1.mGrid, modelfig2_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig2_2.mGrid, modelfig2_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig2_3.mGrid, modelfig2_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig2_4.mGrid, modelfig2_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig2_5.mGrid, modelfig2_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.0$, $\sigma_{\eta} = 0.0$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p0_xi0_eta0.pdf", dpi=150)
# #plt.show()
# #
# ##
# ###================
# ##
# #fig3_1_params = copy.deepcopy(retiring_params)
# #fig3_1_params['Rfree'] = 1.0
# #fig3_1_params['DiscFac'] = 0.98
# #fig3_1_params['prob_opt'] = 0.0
# #fig3_1_params['sigma_xi'] = 0.01
# #fig3_1_params['TasteShkCount'] = 211
# #fig3_1_params['TranShkStd'] = [sqrt(0.000)]*fig3_1_params['T']
# #fig3_1_params['TranShkCount'] = 1
# #modelfig3_1 = RetiringDeaton(**fig3_1_params)
# #modelfig3_1.solve()
# #
# #fig3_2_params = copy.deepcopy(fig3_1_params)
# #fig3_2_params['sigma_s'] = 0.01
# #modelfig3_2 = RetiringDeaton(**fig3_2_params)
# #modelfig3_2.solve()
# #
# #fig3_3_params = copy.deepcopy(fig3_1_params)
# #fig3_3_params['sigma_s'] = 0.15
# #modelfig3_3 = RetiringDeaton(**fig3_3_params)
# #modelfig3_3.solve()
# #
# #fig3_4_params = copy.deepcopy(fig3_1_params)
# #fig3_4_params['sigma_s'] = 0.5
# #modelfig3_4 = RetiringDeaton(**fig3_4_params)
# #modelfig3_4.solve()
# #
# #fig3_5_params = copy.deepcopy(fig3_1_params)
# #fig3_5_params['sigma_s'] = 1.0
# #modelfig3_5 = RetiringDeaton(**fig3_5_params)
# #modelfig3_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig3_1.mGrid, modelfig3_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig3_2.mGrid, modelfig3_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig3_3.mGrid, modelfig3_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig3_4.mGrid, modelfig3_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig3_5.mGrid, modelfig3_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.01$, $\sigma_{\eta} = 0.0$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p0_xi0p01_eta0.pdf", dpi=150)
# #plt.show()
# #
# #
# ##================
# #
# #fig3a_1_params = copy.deepcopy(retiring_params)
# #fig3a_1_params['Rfree'] = 1.0
# #fig3a_1_params['DiscFac'] = 0.98
# #fig3a_1_params['prob_opt'] = 0.0
# #fig3a_1_params['sigma_xi'] = 0.005
# #fig3a_1_params['TasteShkCount'] = 211
# #fig3a_1_params['TranShkStd'] = [sqrt(0.000)]*fig3a_1_params['T']
# #fig3a_1_params['TranShkCount'] = 1
# #modelfig3a_1 = RetiringDeaton(**fig3a_1_params)
# #modelfig3a_1.solve()
# #
# #fig3a_2_params = copy.deepcopy(fig3a_1_params)
# #fig3a_2_params['sigma_s'] = 0.01
# #modelfig3a_2 = RetiringDeaton(**fig3a_2_params)
# #modelfig3a_2.solve()
# #
# #fig3a_3_params = copy.deepcopy(fig3a_1_params)
# #fig3a_3_params['sigma_s'] = 0.15
# #modelfig3a_3 = RetiringDeaton(**fig3a_3_params)
# #modelfig3a_3.solve()
# #
# #fig3a_4_params = copy.deepcopy(fig3a_1_params)
# #fig3a_4_params['sigma_s'] = 0.5
# #modelfig3a_4 = RetiringDeaton(**fig3a_4_params)
# #modelfig3a_4.solve()
# #
# #fig3a_5_params = copy.deepcopy(fig3a_1_params)
# #fig3a_5_params['sigma_s'] = 1.0
# #modelfig3a_5 = RetiringDeaton(**fig3a_5_params)
# #modelfig3a_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig3a_1.mGrid, modelfig3a_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig3a_2.mGrid, modelfig3a_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig3a_3.mGrid, modelfig3a_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig3a_4.mGrid, modelfig3a_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig3a_5.mGrid, modelfig3a_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.005$, $\sigma_{\eta} = 0.0$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p0_xi0p005_eta0.pdf", dpi=150)
# #plt.show()
# #
# ##================
# #
# #fig3b_1_params = copy.deepcopy(retiring_params)
# #fig3b_1_params['Rfree'] = 1.0
# #fig3b_1_params['DiscFac'] = 0.98
# #fig3b_1_params['prob_opt'] = 0.0
# #fig3b_1_params['sigma_xi'] = 0.005
# #fig3b_1_params['TasteShkCount'] = 211
# #fig3b_1_params['TranShkStd'] = [sqrt(0.005)]*fig3b_1_params['T']
# #fig3b_1_params['TranShkCount'] = 31
# #modelfig3b_1 = RetiringDeaton(**fig3b_1_params)
# #modelfig3b_1.solve()
# #
# #fig3b_2_params = copy.deepcopy(fig3b_1_params)
# #fig3b_2_params['sigma_s'] = 0.01
# #modelfig3b_2 = RetiringDeaton(**fig3b_2_params)
# #modelfig3b_2.solve()
# #
# #fig3b_3_params = copy.deepcopy(fig3b_1_params)
# #fig3b_3_params['sigma_s'] = 0.15
# #modelfig3b_3 = RetiringDeaton(**fig3b_3_params)
# #modelfig3b_3.solve()
# #
# #fig3b_4_params = copy.deepcopy(fig3b_1_params)
# #fig3b_4_params['sigma_s'] = 0.5
# #modelfig3b_4 = RetiringDeaton(**fig3b_4_params)
# #modelfig3b_4.solve()
# #
# #fig3b_5_params = copy.deepcopy(fig3b_1_params)
# #fig3b_5_params['sigma_s'] = 1.0
# #modelfig3b_5 = RetiringDeaton(**fig3b_5_params)
# #modelfig3b_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig3b_1.mGrid, modelfig3b_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig3b_2.mGrid, modelfig3b_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig3b_3.mGrid, modelfig3b_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig3b_4.mGrid, modelfig3b_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig3b_5.mGrid, modelfig3b_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.005$, $\sigma_{\eta} = \sqrt{0.005}$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p0_xi0p005_etaSqrt0p005.pdf", dpi=150)
# #plt.show()
# #
# ##================
# #
# #fig4_1_params = copy.deepcopy(retiring_params)
# #fig4_1_params['Rfree'] = 1.0
# #fig4_1_params['DiscFac'] = 0.98
# #fig4_1_params['prob_opt'] = 0.0
# #fig4_1_params['sigma_xi'] = 0.01
# #fig4_1_params['TasteShkCount'] = 211
# #fig4_1_params['TranShkStd'] = [sqrt(0.005)]*fig4_1_params['T']
# #fig4_1_params['TranShkCount'] = 31
# #modelfig4_1 = RetiringDeaton(**fig4_1_params)
# #modelfig4_1.solve()
# #
# #fig4_2_params = copy.deepcopy(fig4_1_params)
# #fig4_2_params['sigma_s'] = 0.01
# #modelfig4_2 = RetiringDeaton(**fig4_2_params)
# #modelfig4_2.solve()
# #
# #fig4_3_params = copy.deepcopy(fig4_1_params)
# #fig4_3_params['sigma_s'] = 0.15
# #modelfig4_3 = RetiringDeaton(**fig4_3_params)
# #modelfig4_3.solve()
# #
# #fig4_4_params = copy.deepcopy(fig4_1_params)
# #fig4_4_params['sigma_s'] = 0.5
# #modelfig4_4 = RetiringDeaton(**fig4_4_params)
# #modelfig4_4.solve()
# #
# #fig4_5_params = copy.deepcopy(fig4_1_params)
# #fig4_5_params['sigma_s'] = 1.0
# #modelfig4_5 = RetiringDeaton(**fig4_5_params)
# #modelfig4_5.solve()
# #
# #t=15
# #
# #plt.plot(modelfig4_1.mGrid, modelfig4_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig4_2.mGrid, modelfig4_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig4_3.mGrid, modelfig4_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.plot(modelfig4_4.mGrid, modelfig4_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# #plt.plot(modelfig4_5.mGrid, modelfig4_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.01$, $\sigma_{\eta} = \sqrt{0.005}$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/p0_xi0p01_etaSqrt0p005.pdf", dpi=150)
# #plt.show()
# #
# ##================

# fig4a_1_params = copy.deepcopy(retiring_params)
# fig4a_1_params['Rfree'] = 1.0
# fig4a_1_params['DiscFac'] = 0.98
# fig4a_1_params['prob_opt'] = 0.0
# fig4a_1_params['sigma_xi'] = 0.02
# fig4a_1_params['TasteShkCount'] = 511
# fig4a_1_params['TranShkStd'] = [sqrt(0.005)]*fig4a_1_params['T']
# fig4a_1_params['TranShkCount'] = 31
# modelfig4a_1 = RetiringDeaton(**fig4a_1_params)
# modelfig4a_1.solve()

# fig4a_2_params = copy.deepcopy(fig4a_1_params)
# fig4a_2_params['sigma_s'] = 0.01
# modelfig4a_2 = RetiringDeaton(**fig4a_2_params)
# modelfig4a_2.solve()

# fig4a_3_params = copy.deepcopy(fig4a_1_params)
# fig4a_3_params['sigma_s'] = 0.15
# modelfig4a_3 = RetiringDeaton(**fig4a_3_params)
# modelfig4a_3.solve()

# fig4a_4_params = copy.deepcopy(fig4a_1_params)
# fig4a_4_params['sigma_s'] = 0.5
# modelfig4a_4 = RetiringDeaton(**fig4a_4_params)
# modelfig4a_4.solve()

# fig4a_5_params = copy.deepcopy(fig4a_1_params)
# fig4a_5_params['sigma_s'] = 1.0
# modelfig4a_5 = RetiringDeaton(**fig4a_5_params)
# modelfig4a_5.solve()

# t=15

# plt.plot(modelfig4a_1.mGrid, modelfig4a_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# plt.plot(modelfig4a_2.mGrid, modelfig4a_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# plt.plot(modelfig4a_3.mGrid, modelfig4a_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# plt.plot(modelfig4a_4.mGrid, modelfig4a_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# plt.plot(modelfig4a_5.mGrid, modelfig4a_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# plt.xlim([15,120]);plt.ylim([15,25]);
# plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.02$, $\sigma_{\eta} = \sqrt{0.005}$")
# plt.grid(color='k', linestyle=':', linewidth=0.5);
# plt.yticks(np.arange(15, 26, step=1.0))
# plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# plt.savefig("../Figures/Figures_v3/p0_xi0p02_etaSqrt0p005.pdf", dpi=150)
# plt.show()

# ##================

# fig4b_1_params = copy.deepcopy(retiring_params)
# fig4b_1_params['Rfree'] = 1.0
# fig4b_1_params['DiscFac'] = 0.98
# fig4b_1_params['prob_opt'] = 0.0
# fig4b_1_params['sigma_xi'] = 0.02
# fig4b_1_params['TasteShkCount'] = 511
# fig4b_1_params['TranShkStd'] = [sqrt(0.000)]*fig4b_1_params['T']
# fig4b_1_params['TranShkCount'] = 1
# modelfig4b_1 = RetiringDeaton(**fig4b_1_params)
# modelfig4b_1.solve()

# fig4b_2_params = copy.deepcopy(fig4b_1_params)
# fig4b_2_params['sigma_s'] = 0.01
# modelfig4b_2 = RetiringDeaton(**fig4b_2_params)
# modelfig4b_2.solve()

# fig4b_3_params = copy.deepcopy(fig4b_1_params)
# fig4b_3_params['sigma_s'] = 0.15
# modelfig4b_3 = RetiringDeaton(**fig4b_3_params)
# modelfig4b_3.solve()

# fig4b_4_params = copy.deepcopy(fig4b_1_params)
# fig4b_4_params['sigma_s'] = 0.5
# modelfig4b_4 = RetiringDeaton(**fig4b_4_params)
# modelfig4b_4.solve()

# fig4b_5_params = copy.deepcopy(fig4b_1_params)
# fig4b_5_params['sigma_s'] = 1.0
# modelfig4b_5 = RetiringDeaton(**fig4b_5_params)
# modelfig4b_5.solve()

# t=15

# plt.plot(modelfig4b_1.mGrid, modelfig4b_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# plt.plot(modelfig4b_2.mGrid, modelfig4b_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# plt.plot(modelfig4b_3.mGrid, modelfig4b_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# plt.plot(modelfig4b_4.mGrid, modelfig4b_4.solution[t].ChoiceSols[1].C,color='k',linestyle=':')
# plt.plot(modelfig4b_5.mGrid, modelfig4b_5.solution[t].ChoiceSols[1].C,color='r',linestyle=':')
# plt.xlim([15,120]);plt.ylim([15,25]);
# plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# plt.title("$p = 0.0$, $\sigma_{\\xi} = 0.02$, $\sigma_{\eta} = 0.0$")
# plt.grid(color='k', linestyle=':', linewidth=0.5);
# plt.yticks(np.arange(15, 26, step=1.0))
# plt.legend(['$\sigma_s = 0$','$\sigma_s = 0.01$','$\sigma_s = 0.15$','$\sigma_s = 0.50$','$\sigma_s = 1.00$'],loc=4)
# plt.savefig("../Figures/Figures_v3/p0_xi0p02_eta0.pdf", dpi=150)
# plt.show()

# ##================
# #
# #fig5_1_params = copy.deepcopy(retiring_params)
# #fig5_1_params['Rfree'] = 1.0
# #fig5_1_params['DiscFac'] = 0.98
# #fig5_1_params['prob_opt'] = 0.0
# #fig5_1_params['sigma_xi'] = 0.02
# #fig5_1_params['sigma_s'] = 0.5
# #fig5_1_params['TasteShkCount'] = 211
# #fig5_1_params['TranShkStd'] = [sqrt(0.005)]*fig5_1_params['T']
# #fig5_1_params['TranShkCount'] = 31
# #modelfig5_1 = RetiringDeaton(**fig5_1_params)
# #modelfig5_1.solve()
# #
# #fig5_2_params = copy.deepcopy(fig5_1_params)
# #fig5_2_params['prob_opt'] = 0.1
# #modelfig5_2 = RetiringDeaton(**fig5_2_params)
# #modelfig5_2.solve()
# #
# #fig5_3_params = copy.deepcopy(fig5_1_params)
# #fig5_3_params['prob_opt'] = 0.5
# #modelfig5_3 = RetiringDeaton(**fig5_3_params)
# #modelfig5_3.solve()
# #
# #t=15
# #
# #plt.plot(modelfig5_1.mGrid, modelfig5_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig5_2.mGrid, modelfig5_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig5_3.mGrid, modelfig5_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p =$ Varying, $\sigma_s = 0.5$, $\sigma_{\\xi} = 0.02$, $\sigma_{\eta} = \sqrt{0.005}$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$p = 0.0$','$p = 0.1$','$p = 0.5$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/VaryP.pdf", dpi=150)
# #plt.show()
# #
# ##================
# #
# #fig6_1_params = copy.deepcopy(retiring_params)
# #fig6_1_params['Rfree'] = 1.0
# #fig6_1_params['DiscFac'] = 0.98
# #fig6_1_params['prob_opt'] = 0.0
# #fig6_1_params['sigma_xi'] = 0.02
# #fig6_1_params['sigma_s'] = 0.5
# #fig6_1_params['TasteShkCount'] = 211
# #fig6_1_params['TranShkStd'] = [sqrt(0.005)]*fig6_1_params['T']
# #fig6_1_params['TranShkCount'] = 31
# #modelfig6_1 = RetiringDeaton(**fig6_1_params)
# #modelfig6_1.solve()
# #
# #fig6_2_params = copy.deepcopy(fig6_1_params)
# #fig6_2_params['sigma_s'] = 1.0
# #modelfig6_2 = RetiringDeaton(**fig6_2_params)
# #modelfig6_2.solve()
# #
# #fig6_3_params = copy.deepcopy(fig6_1_params)
# #fig6_3_params['sigma_s'] = 2.0
# #modelfig6_3 = RetiringDeaton(**fig6_3_params)
# #modelfig6_3.solve()
# #
# #t=15
# #
# #plt.plot(modelfig6_1.mGrid, modelfig6_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig6_2.mGrid, modelfig6_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig6_3.mGrid, modelfig6_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_s =$ Varying, $\sigma_{\\xi} = 0.02$, $\sigma_{\eta} = \sqrt{0.005}$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_s = 0.5$','$\sigma_s = 1.0$','$\sigma_s = 2.0$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/VaryS.pdf", dpi=150)
# #plt.show()
# #
# ###================
# #
# #fig7_1_params = copy.deepcopy(retiring_params)
# #fig7_1_params['Rfree'] = 1.0
# #fig7_1_params['DiscFac'] = 0.98
# #fig7_1_params['prob_opt'] = 0.0
# #fig7_1_params['sigma_xi'] = 0.02
# #fig7_1_params['sigma_s'] = 0.5
# #fig7_1_params['TasteShkCount'] = 211
# #fig7_1_params['TranShkStd'] = [sqrt(0.005)]*fig7_1_params['T']
# #fig7_1_params['TranShkCount'] = 31
# #modelfig7_1 = RetiringDeaton(**fig7_1_params)
# #modelfig7_1.solve()
# #
# #fig7_2_params = copy.deepcopy(fig7_1_params)
# #fig7_2_params['sigma_xi'] = 0.025
# #modelfig7_2 = RetiringDeaton(**fig7_2_params)
# #modelfig7_2.solve()
# #
# #fig7_3_params = copy.deepcopy(fig7_1_params)
# #fig7_3_params['sigma_xi'] = 0.03
# #modelfig7_3 = RetiringDeaton(**fig7_3_params)
# #modelfig7_3.solve()
# #
# #t=15
# #
# #plt.plot(modelfig7_1.mGrid, modelfig7_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig7_2.mGrid, modelfig7_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig7_3.mGrid, modelfig7_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_s = 0.5$, $\sigma_{\\xi} =$ Varying, $\sigma_{\eta} = \sqrt{0.005}$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_{\\xi} = 0.02$','$\sigma_{\\xi} = 0.025$','$\sigma_{\\xi} = 0.03$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/VaryXi.pdf", dpi=150)
# #plt.show()
# #
# ##================
# #
# #fig8_1_params = copy.deepcopy(retiring_params)
# #fig8_1_params['Rfree'] = 1.0
# #fig8_1_params['DiscFac'] = 0.98
# #fig8_1_params['prob_opt'] = 0.0
# #fig8_1_params['sigma_xi'] = 0.02
# #fig8_1_params['sigma_s'] = 0.5
# #fig8_1_params['TasteShkCount'] = 211
# #fig8_1_params['TranShkStd'] = [sqrt(0.005)]*fig8_1_params['T']
# #fig8_1_params['TranShkCount'] = 31
# #modelfig8_1 = RetiringDeaton(**fig8_1_params)
# #modelfig8_1.solve()
# #
# #fig8_2_params = copy.deepcopy(fig8_1_params)
# #fig8_2_params['TranShkStd'] = [5.0*sqrt(0.005)]*fig8_1_params['T']
# #modelfig8_2 = RetiringDeaton(**fig8_2_params)
# #modelfig8_2.solve()
# #
# #fig8_3_params = copy.deepcopy(fig8_1_params)
# #fig8_3_params['TranShkStd'] = [7.5*sqrt(0.005)]*fig8_1_params['T']
# #modelfig8_3 = RetiringDeaton(**fig8_3_params)
# #modelfig8_3.solve()
# #
# #t=15
# #
# #plt.plot(modelfig8_1.mGrid, modelfig8_1.solution[t].ChoiceSols[1].C,color='k',linestyle='-')
# #plt.plot(modelfig8_2.mGrid, modelfig8_2.solution[t].ChoiceSols[1].C,color='r',linestyle='-')
# #plt.plot(modelfig8_3.mGrid, modelfig8_3.solution[t].ChoiceSols[1].C,color='k',linestyle='--')
# #plt.xlim([15,120]);plt.ylim([15,25]);
# #plt.xlabel("Resources, $m_t$"); plt.ylabel("Consumption, $C(m_t)$")
# #plt.title("$p = 0.0$, $\sigma_s = 0.5$, $\sigma_{\\xi} =$ Varying, $\sigma_{\eta} = \sqrt{0.005}$")
# #plt.grid(color='k', linestyle=':', linewidth=0.5);
# #plt.yticks(np.arange(15, 26, step=1.0))
# #plt.legend(['$\sigma_{\eta} = \sqrt{0.005}$','$\sigma_{\eta} = 5\sqrt{0.005}$','$\sigma_{\eta} = 7.5\sqrt{0.005}$'],loc=4)
# #plt.savefig("../Figures/Figures_v3/VaryEta.pdf", dpi=150)
# #plt.show()

# #================


# end_time = time.time()
# elapsed_time = end_time-start_time
# # print "*************************"
# # print "Time elapsed (seconds):"
# # print round(elapsed_time, 3)
# # print "Time elapsed (minutes):"
# # print round(elapsed_time/60.0, 3)
# # print "*************************"



# # -

# # # References
# # <div class="cite2c-biblio"></div>


