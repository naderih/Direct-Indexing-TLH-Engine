"""
Module: c1_heuristic_optimizer.py
V1: The "Naive" Risk-Minimizing Optimizer

OBJECTIVE:
    Minimize Tracking Error Variance (TEV).
    Harvesting is treated as a hard binary constraint (Force weight to 0).
"""
import numpy as np
import pandas as pd
import cvxpy as cp
import warnings

class HeuristicOptimizer:
    def __init__(self, 
                 harvest_threshold: float = -0.05,
                 turnover_penalty: float = 0.001):
        self.harvest_threshold = harvest_threshold
        self.turnover_penalty = turnover_penalty 

    def get_harvest_candidates(self, 
                               permnos: np.ndarray, 
                               current_prices: dict, 
                               positions: dict) -> list:
        """
        Identifies stocks that have breached the loss threshold.
        returns the harvest candidates list  
        Args:
            permnos (np.ndarray): Array of asset identifiers perfectly aligned with V_matrix.
            current_prices (dict): Mapping of {permno: today_spot_price_cad}.
            positions (dict): The Tax Ledger state: {permno: {'shares': float, 'acb_per_share': float}}.
            
        Returns:
            list: the list of harvesting candidates.
        """
        candidates = []
        for permno in permnos:
            if permno in positions and positions[permno]['shares'] > 0:
                acb = positions[permno]['acb_per_share']

                # Failsafe: if permno doesnt exist in the current prices, but exist in positions
                # assign a return of zero, basically assuming the price has not moved from acb 
                price = current_prices.get(permno, acb)

                if (price - acb) / acb <= self.harvest_threshold:
                    candidates.append(permno)
        return candidates

    def optimize(self, 
                 current_weights: pd.Series, 
                 bench_weights: pd.Series, 
                 V_matrix: pd.DataFrame, 
                 do_not_buy: list, 
                 do_not_harvest: list, 
                 current_prices: dict, 
                 positions: dict) -> pd.Series:
        """
        The core CVXPY execution engine that resolves the optimal target portfolio.
        
        It balances the dual objectives of maximizing Tax Alpha and minimizing 
        Transaction Costs, strictly bounded by Tracking Error Variance (TEV) 
        and CRA regulatory lockouts.
        
        Args:
            current_weights (pd.Series): The portfolio's current weights (Index: permno).
            bench_weights (pd.Series): The S&P 500 target weights for today (Index: permno).
            V_matrix (pd.DataFrame): The N x N factor covariance risk matrix for today.
            do_not_buy (list): Permnos restricted by the CRA 30-day forward superficial loss rule.
            do_not_harvest (list): Permnos restricted by the CRA 30-day backward rule.
            current_prices (dict): Mapping of {permno: today_spot_price_cad}.
            positions (dict): The Tax Ledger state: {permno: {'shares': float, 'acb_per_share': float}}.
            
        Returns:
            pd.Series: The optimal target weights (h*), indexed by permno, cleanly summing to 1.0.
        """
        permnos = V_matrix.index.values
        N = len(permnos)
        h_current = current_weights.reindex(permnos).fillna(0.0).values
        h_b = bench_weights.reindex(permnos).fillna(0.0).values
        V = V_matrix.values
        
        harvest_candidates = self.get_harvest_candidates(permnos, current_prices, positions)
        # initiating the optimization variable (portfolio weights)
        h = cp.Variable(N)
        
        # V1 OBJECTIVE: Strictly minimize Tracking Error plus the penalty for transaction costs
        tracking_error = cp.quad_form(h - h_b , cp.psd_wrap(V))
        turnover= cp.norm1(h - h_current)
        objective = cp.Minimize(tracking_error + (self.turnover_penalty * turnover))

        
        constraints =[
            cp.sum(h) == 1.0, # remian fully invested
            h >= 0.0 # long only 
        ]
        
        for i, permno in enumerate(permnos):
            # THE V1 FLAW: Hard liquidation constraint
            if permno in harvest_candidates and permno not in do_not_harvest:
                constraints.append(h[i] == 0.0)  # Immeditely force 100% sale for every stock which dropped more than 5%!
                
            # CRA Rules
            if permno in do_not_buy:
                constraints.append(h[i] <= h_current[i])
            if permno in do_not_harvest:
                constraints.append(h[i] >= h_current[i]) # superficial loss lockouts

        # instantiate the CVXPY optimization problem        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP)
            if h.value is None:
                raise ValueError("Solver failed.")
            
            # The math already guarantees h <= 1.0, 
            # but computationally, convex solvers likr OSQP iterate 
            # until they hit a numerical tolerance, 
            # meaning 'zero' often comes out as a microscopic negative float like -1e-12
            # I intentionally use NumPy to clip the noise and re-normalize the array sum to exactly 1.0.
            opt_w = np.clip(h.value, 0.0, 1.0) 
            return pd.Series(opt_w / np.sum(opt_w), index=permnos)
        except:
            return pd.Series(h_b, index=permnos)
        
    def weights_to_shares(self, 
                          optimal_weights: pd.Series, 
                          total_aum: float, 
                          current_prices: dict) -> dict:
        """
        The Translation Layer: Converts continuous Weights (h) to  Shares (n).
        Assuming Wealthsimple's fractional share trading (truncated to 4 decimal places).
        
        Args: 
        optimal_weights (pd.Series): The optimal weights from optimization (Index: permno).
        total_aum (float): The total CAD market value of the portfolio + cash.
        current_prices (dict): Mapping of {permno: today_spot_price_cad}.

        """
        target_shares = {}
        for permno, weight in optimal_weights.items():
            if weight > 1e-6 and permno in current_prices:
                target_dollars = weight * total_aum
                raw_shares = target_dollars / current_prices[permno]
                
                # Wealthsimple supports fractional shares.
                # I truncate at 4 decimal places (floor math) to prevent microscopic 
                # rounding-up from accidentally triggering a negative cash margin balance.
                target_shares[permno] = np.floor(raw_shares * 10000.0) / 10000.0
                
        return target_shares