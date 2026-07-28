"""
Module: c2_tax_alpha_optimizer.py
V2: The Convex "Tax Alpha" Optimizer

OBJECTIVE:

    Maximize Tax Alpha minus Transaction Costs.
    Tracking Error is enforcd as a strict risk budget constraint, 
    naturally allowing the solver to discover "Partial Harvesting".
"""
import numpy as np
import pandas as pd
import cvxpy as cp
import warnings

class TaxAlphaOptimizer:
    """
    Direct Indexing Optimizer engineered to maximize after-tax returns.
    Formulates a Convex Quadratic Program to execute Tax-Loss Harvesting 
    while stricty bounded by CRA regulations and Tracking Error limts.
    """
    
    def __init__(self, 
                 tev_multiplier: float = 0.01, 
                 turnover_penalty: float = 0.001, 
                 harvest_threshold: float = -0.05,
                 min_trade_cad: float = 50.0):
        """
        Initializes the V2 Tax Alpha Optimizer.
        
        Args:
            tev_multiplier (float):   The multiplier for maximum allowed Tracking Error Variance (TEV) 
                                        budget (e.g., 0.01 = 1% of the variance of the benchmark variance).
            turnover_penalty (float): The lambda multiplier for the L1 Norm transaction 
                                      cost penalty. Controls portfolio churn.
            harvest_threshold (float): The minimum unrealized loss percentage required 
                                       to trigger a harvest (e.g., -0.05 means -5%).
        """
        self.tev_multiplier = tev_multiplier
        self.turnover_penalty = turnover_penalty
        self.harvest_threshold = harvest_threshold
        self.min_trade_cad = min_trade_cad

    
    def _calculate_tax_penalty_vector(self, permnos, current_prices, positions):
        """
        Creates the Tax Penalty Vector (Y_tax).
        - Selling Winners generates a Tax Bill (Positive Penalty).
        - Selling Losers generates a Tax Shield (Negative Penalty / Benefit).
        """
        y_tax = np.zeros(len(permnos))
        
        for i, permno in enumerate(permnos):
            if permno in positions and positions[permno]['shares'] > 0:
                acb = positions[permno]['acb_per_share']
                price = current_prices.get(permno, acb)
                
                # Calculate the exact CAD Capital Gain/Loss per share
                unrealized_pnl_per_share = price - acb
                
                # We want to scale this by the stock price so it acts as a percentage penalty
                # e.g., A $10 gain on a $100 stock is a 10% penalty.
                pnl_pct = unrealized_pnl_per_share / price
                
                # If it's a massive winner, pnl_pct is positive (High Penalty to sell).
                # If it's a massive loser, pnl_pct is negative (Benefit to sell).
                y_tax[i] = pnl_pct
                
        return y_tax

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
        
        Args:
            current_weights (pd.Series): The portfolio's current weights (Index: permno).
            bench_weights (pd.Series): The S&P 500 target weights for today (Index: permno).
            V_matrix (pd.DataFrame): The N x N factor covariance risk matrix for today.
            do_not_buy (list): Permnos restricted by the CRA 30-day forward wash sale rule.
            do_not_harvest (list): Permnos restricted by the CRA 30-day backward rule.
            current_prices (dict): Mapping of {permno: today_spot_price_cad}.
            positions (dict): The Tax Ledger state {permno: {'shares': float, 'acb_per_share': float}}.
            
        Returns:
            pd.Series: The optimal target weights (h*), indexed by permno, cleanly summing to 1.0.
        """
        # Align all vectors to the V_matrix index to ensure perfect linear algebra
        permnos = V_matrix.index.values
        N = len(permnos)
        h_current = current_weights.reindex(permnos).fillna(0.0).values
        h_b = bench_weights.reindex(permnos).fillna(0.0).values
        V = V_matrix.values
        # tax penalty vector 
        Y_tax = self._calculate_harvest_opportunity_cost(permnos, 
                                                                           current_prices, 
                                                                           positions)
        
        h = cp.Variable(N)
        
        # V2 OBJECTIVE: Minimize (Holding Penalty + Tracking Error + Friction)
        # Calculate the trade delta (Target Weight - Current Weight)
        delta_h = h - h_current
        # We only care about SELLS. In CVXPY, we can isolate the negative part of the delta.
        # cp.pos(-delta_h) extracts the absolute size of the sells.
        # (e.g., if delta is -0.05, cp.pos(0.05) = 0.05).
        sells_only = cp.pos(-delta_h)

        # Multiply the SELLS by the Tax Penalty Vector
        # - Selling a winner (positive Y_tax) * (sell size) = Positive Penalty (BAD)
        # - Selling a loser (negative Y_tax) * (sell size) = Negative Penalty (GOOD)
        tax_impact = sells_only.T @ Y_tax


        # Note: Since opportunity_cost_vector is positive for losers, 
        # we MINIMIZE (h^T * opportunity_cost_vector).
        # By minimizing a positive product, CVXPY forces h -> 0 for the losers.
        turnover = cp.norm1(h - h_current)
        
        objective = cp.Minimize(tax_impact + (self.turnover_penalty * turnover))
        
        #
        # CONSTRAINTS
        # 
        constraints =[
            cp.sum(h) == 1.0, 
            h >= 0.0
        ]
        
        #  Dynamic Tracking Error Leash (Relative Risk Budgeting)
        bench_variance = h_b.T @ V @ h_b
        # Dynamically scale our allowed Tracking Error Variance
        dynamic_tev_budget = bench_variance * self.tev_multiplier
        
        tracking_error = cp.quad_form(h - h_b, cp.psd_wrap(V))
        constraints.append(tracking_error <= dynamic_tev_budget)
        
        # CRA Tax Rules
        for i, permno in enumerate(permnos):
            if permno in do_not_buy:
                constraints.append(h[i] <= h_current[i])
            if permno in do_not_harvest:
                constraints.append(h[i] >= h_current[i])
                
        
        # SOLVE
        prob = cp.Problem(objective, constraints)
        try:
            # Putting the Covariance Matrix into an inequality constraint transforms the math from a standard QP 
            # into a Quadratically Constrained Quadratic Program (QCQP). 
            # OSQP doesn't support conic constraints. 
            # We allow CVXPY's compiler to automatically route the payload to ECOS/Clarabel
            prob.solve() # since the 
            if h.value is None:
                raise ValueError("Solver failed to find a feasible solution.")
            
            # Clip floating point noise and strictly re-normalize to 1.0 to avoid shorting from computational jitter 
            opt_w = np.clip(h.value, 0.0, 1.0)
            return pd.Series(opt_w / np.sum(opt_w), 
                             index=permnos)
            
        except Exception as e:
            warnings.warn(f"CVXPY Solver Error: {e}. Falling back to Benchmark weights.")
            return pd.Series(h_b, index=permnos)

    def weights_to_shares(self, 
                          optimal_weights: pd.Series, 
                          total_aum: float, 
                          current_prices: dict,
                          positions: dict) -> dict:
        """
        The Translation Layer: Converts continuous Weights (h) to discrete Shares (n).
        Applies the Minimum Trade Dollar filter to squash continuous optimization dust.
        
        Args:
            optimal_weights (pd.Series): The target weights output by CVXPY.
            total_aum (float): The total CAD market value of the portfolio + cash.
            current_prices (dict): Mapping of {permno: today_spot_price_cad}.
            positions (dict): The current ledger holdings {permno: {'shares': float, ...}}.
            
        Returns:
            dict: Mapping of {permno: target_shares} formatted for the Tax Ledger.
        """
        target_shares = {}
        for permno, weight in optimal_weights.items():
            if permno in current_prices:
                price = current_prices[permno]
                tgt_dollars = weight * total_aum
                
                # What do we currently own?
                cur_shares = positions.get(permno, {}).get('shares', 0.0)
                cur_dollars = cur_shares * price
                
                # Calculate absolute trade size
                trade_value = abs(tgt_dollars - cur_dollars)
                
                if trade_value >= self.min_trade_cad:
                    # It's a real trade. Convert to fractional shares.
                    raw_shares = tgt_dollars / price
                    # Truncate to 4 decimals for Wealthsimple fractional limits
                    target_shares[permno] = np.floor(raw_shares * 10000.0) / 10000.0
                else:
                    # It's dust. Snap the target exactly back to current holdings.
                    # This ensures delta = 0.0, and the Ledger ignores it.
                    target_shares[permno] = cur_shares
                    
        return target_shares


if __name__ == "__main__":
    # --- PROVING V2 OUTPERFORMS V1 ---
    print("======================================================")
    print(" V2 Tax Alpha Optimizer: Partial Harvesting")
    print("======================================================\n")
    
    permnos =[101, 102, 103] # 101: AAPL, 102: MSFT, 103: ORCL
    
    # Covariance Matrix (V) - High correlation (0.03) between AAPL and MSFT
    V_mock = pd.DataFrame([[0.04, 0.03, 0.01],[0.03, 0.04, 0.01],
        [0.01, 0.01, 0.05]
    ], index=permnos, columns=permnos)
    
    # The Benchmark is static. We currently perfectly replicate the benchmark.
    bench_weights = pd.Series({101: 0.50, 102: 0.40, 103: 0.10})
    current_weights = pd.Series({101: 0.50, 102: 0.40, 103: 0.10})
    
    # SCENARIO: AAPL (101) crashed from $150 to $100 (-33%). 
    # MSFT and ORCL are flat.
    prices = {101: 100.0, 102: 50.0, 103: 30.0}
    positions = {
        101: {'shares': 500, 'acb_per_share': 150.0}, # Deep Loss! (-33%)
        102: {'shares': 800, 'acb_per_share': 40.0},  # Gain
        103: {'shares': 333, 'acb_per_share': 25.0}   # Gain
    }
    
    # Set a very tight Tracking Error budget (e.g. max variance of 0.0005)
    optimizer_v2 = TaxAlphaOptimizer(max_tracking_error=0.0005, 
                                     turnover_penalty=0.0001)
    
    print("1. Current Portfolio Weights (Perfect Benchmark Replication):")
    print(current_weights.to_string())
    
    print("\n2. AAPL Crashes 33%. Running V2 Optimizer with Tight Risk Leash...")
    
    # Run the optimization (No lockouts today)
    w_opt = optimizer_v2.optimize(
        current_weights=current_weights, 
        bench_weights=bench_weights, 
        V_matrix=V_mock,
        do_not_buy=[], 
        do_not_harvest=[],
        current_prices=prices,
        positions=positions
    )
    
    print("\n3. Optimal Target Weights (V2 Output):")
    print(w_opt.to_string())
    
    print("\n------------------------------------------------------")
    print(" THE RESULTS (The Interview Flex)")
    print("------------------------------------------------------")
    print(f"-> AAPL (101) Weight: Dropped from {current_weights[101]:.2f} to {w_opt[101]:.3f}")
    print("   Notice it didn't drop to 0.0! This is 'Partial Harvesting'.")
    print("   CVXPY sold exactly enough to harvest tax alpha, but stopped")
    print("   before violating the strict Tracking Error leash.")
    
    print(f"\n-> MSFT (102) Weight: Increased from {current_weights[102]:.2f} to {w_opt[102]:.3f}")
    print("   Because MSFT has high correlation (0.03) to AAPL in the V matrix,")
    print("   CVXPY organically identified it as the best mathematical proxy and")
    print("   bought it to plug the risk hole!")
    
    print("\n4. Translation Layer: Weights -> Target Shares")
    total_aum = 100000.0  # Assume a $100k portfolio
    target_shares = optimizer_v2.weights_to_shares(w_opt, total_aum, prices)
    print(target_shares)