"""
Module: c3_backtest_runner.py
Phase C: The Orchestrator - Historical Simulation Engine

OBJECTIVE:
    Runs the daily historical simulation, stepping through time to execute 
    the Direct Indexing & Tax-Loss Harvesting strategy.
    
    The flow:
    1. Update Market Prices (S_t+1)
    2. Check Harvest Scanner
    3. (If Triggered) Run CVXPY Optimizer
    4. Execute Trades in Tax Ledger
    5. Log End-of-Day State
"""

import pandas as pd
import numpy as np
from tqdm import tqdm 

# Import modules
from b1_risk_model import FactorRiskModel
from b2_tax_ledger import CRATaxLedger
from c2_tax_alpha_optimizer import TaxAlphaOptimizer
from config import DATA_CLEAN_DIR as DATA_DIR

class TLHBacktester:
    def __init__(self, 
                 start_date='2015-01-01', 
                 end_date='2023-12-31', 
                 initial_capital=1_000_000.0):
        
        print("Initializing Backtest Engine...")
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        
        # 1. Load the Point-in-Time Universe
        self.df_univ = pd.read_parquet(DATA_DIR / 'tlh_universe.parquet')
        
        self.df_univ = self.df_univ[(self.df_univ['date'] >= self.start_date) & 
                                    (self.df_univ['date'] <= self.end_date)]
        
        # Ensure fast daily querying
        self.df_univ.set_index('date', inplace=True)
        self.trading_days = self.df_univ.index.unique().sort_values()
        
        # 2. Instantiate the Core Engines
        self.risk_model = FactorRiskModel()
        
        # ledger is initiated with empty current_shares, 0.0 total_equity_cad, and no positions 
        self.ledger = CRATaxLedger(initial_capital_cad = self.initial_capital)

        self.optimizer = TaxAlphaOptimizer(tev_multiplier = 0.01,  # Allow TEV up to 1% of the market's natural variance
                                           turnover_penalty = 0.10, # Strong enough to force sparse proxy buys 
                                           harvest_threshold = -0.05)
        
        # 3. Tracking Metrics
        self.daily_history =[]

    def get_benchmark_weights(self, 
                              daily_data: pd.DataFrame) -> pd.Series:
        """
        Calculates exact S&P 500 cap-weights for the day.
        We will use this function inside our simulation to extract the 
        benchmark weights on a specific date
        """
        # Using USD Market Cap to calculate benchmark weights 
        # to avoids FX math distortion in calculations downstream
        total_mkt_cap = daily_data['mkt_cap_usd'].sum()
        weights = daily_data['mkt_cap_usd'] / total_mkt_cap
        return pd.Series(weights.values, index = daily_data['permno'])

    def run_simulation(self):
        print(f"Starting Simulation: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total Trading Days: {len(self.trading_days)}")
        
        for today in tqdm(self.trading_days, desc="Simulating Daily Trading"):
            # 1. MARKET UPDATE (S_t+1)
            daily_data = self.df_univ.loc[today]
            # if the daily date contains only 1 stock, 
            # we make sure we convert it to dataframe in the correct format
            if isinstance(daily_data, pd.Series): 
                daily_data = daily_data.to_frame().T # Edge case: only 1 stock alive
                
            benchamrk_permnos = daily_data['permno'].values
            
            # Dictionaries are much faster than series or data frames indexing 
            # Fast Dictionaries for the Ledger
            prices_cad = dict(zip(daily_data['permno'], daily_data['prc_adj_cad']))
            divs_cad = dict(zip(daily_data['permno'], daily_data['divamt_net_cad']))
            bench_w = self.get_benchmark_weights(daily_data)
            

            # 2. PASSIVE UPDATES
            current_shares, total_aum = self.ledger.get_current_holdings(prices_cad)
            self.ledger.process_dividends(current_shares, divs_cad)
            
            # 3. HARVEST SCANNER (The Trigger)
            # Check if any held stock breaches the -5% threshold
            trigger_optimization = False
            for permno, shares in current_shares.items():
                if shares > 0 and permno in self.ledger.positions:
                    acb = self.ledger.positions[permno]['acb_per_share']
                    
                    if acb <= 0: 
                        continue
                    price = prices_cad.get(permno, acb)
                    if (price - acb) / acb <= self.optimizer.harvest_threshold:
                        trigger_optimization = True
                        break
                        
            # Force an optimization on the very first day to invest the cash
            if today == self.trading_days[0]:
                trigger_optimization = True

            # 4. ACTIVE OPTIMIZATION & EXECUTION
            if trigger_optimization:
                # Calculate current weights based on today's prices
                #current_w = pd.Series({p: (s * prices_cad.get(p, 0)) / total_aum for p, s in current_shares.items()})
                current_w_dict = {}
                
                for p, s in current_shares.items():
                    acb = self.ledger.positions[p]['acb_per_share']
                    price = prices_cad.get(p, acb)
                    if pd.isna(price) or price <= 0:
                        price = acb
                    current_w_dict[p] = (s * price) / total_aum
                
                current_w = pd.Series(current_w_dict)

                # Get CRA Lockouts
                do_not_buy, do_not_harvest = self.ledger.get_optimizer_constraints(today)
                
                # Get V Matrix
                V_mat = self.risk_model.build_factor_covariance(today, benchamrk_permnos)
                
                # Run CVXPY
                opt_w = self.optimizer.optimize(
                    current_weights=current_w,
                    bench_weights=bench_w,
                    V_matrix=V_mat,
                    do_not_buy=do_not_buy,
                    do_not_harvest=do_not_harvest,
                    current_prices=prices_cad,
                    positions=self.ledger.positions
                )
                
                # Translate & Execute
                target_shares = self.optimizer.weights_to_shares(opt_w, 
                                                                 total_aum, 
                                                                 prices_cad,
                                                                 self.ledger.positions
                                                                 )
                self.ledger.execute_trades(target_shares, prices_cad, today)
                
                # Recalculate AUM after trading (accounts for transaction cash flow)
                _, total_aum = self.ledger.get_current_holdings(prices_cad)
            
            else:
                # If no trigger, just let the lockouts age by 1 day
                self.ledger.update_lockouts(today)

            # 5. LOG END OF DAY STATE (t+1)
            self.daily_history.append({
                'date': today,
                'portfolio_aum_cad': total_aum,
                'cash_cad': self.ledger.cash_cad,
                'realized_losses_cad': self.ledger.realized_losses_cad,
                'realized_gains_cad': self.ledger.realized_gains_cad,
                'trade_executed': trigger_optimization
            })

        print("\nSimulation Complete!")
        
    def generate_tearsheet(self):
        """Calculates final Tax Alpha metrics."""
        results = pd.DataFrame(self.daily_history).set_index('date')
        
        start_aum = self.initial_capital
        final_aum = results['portfolio_aum_cad'].iloc[-1]
        total_return = (final_aum - start_aum) / start_aum
        
        total_losses_harvested = results['realized_losses_cad'].iloc[-1]
        
        # Assuming a rough 50% inclusion rate and highest marginal tax bracket (e.g., 53% in Ontario)
        # Tax Shield = Harvested Losses * 50% * 53% = Roughly 26.5% cash value
        estimated_tax_shield = total_losses_harvested * 0.265
        tax_alpha_bps = (estimated_tax_shield / start_aum) * 10000
        
        print("\n=================================================")
        print(" INSTITUTIONAL TEARSHEET (Wealthsimple Direct Indexing)")
        print("=================================================")
        print(f"Initial Capital (CAD):     ${start_aum:,.2f}")
        print(f"Final AUM (CAD):           ${final_aum:,.2f}")
        print(f"Pre-Tax Total Return:      {total_return * 100:.2f}%")
        print("-------------------------------------------------")
        print(f"Total Losses Harvested:    ${total_losses_harvested:,.2f}")
        print(f"Estimated Tax Savings:     ${estimated_tax_shield:,.2f} CAD")
        print(f"Estimated Tax Alpha:       {tax_alpha_bps:.1f} bps")
        print("=================================================\n")
        
        return results

if __name__ == "__main__":
    backtester = TLHBacktester(start_date='2015-01-01', end_date='2015-12-31')
    backtester.run_simulation()
    results_df = backtester.generate_tearsheet()