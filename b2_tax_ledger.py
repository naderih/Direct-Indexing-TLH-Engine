"""
Module: b2_tax_ledger.py
Phase B: The Wealthsimple Core - Tax & Lot Accounting Engine

OBJECTIVE:
    Maintains dual-ledger portfolio accounting (USD/CAD) simulating the exact 
    frictions of a Canadian taxable account (Non-Registered).

KEY CRA MECHANICS:
    1. Average Cost Base (ACB): Pools the cost of identical properties. 
    2. Dual-Ledger FX: ACB and capital gains are calculated using the CAD spot rate on the trade date.
    3. Superficial Loss Rule (30-Day Forward & Backward): 
       - Blocks buying a stock 30 days after harvesting a loss.
       - Blocks harvesting a loss if the stock was purchased in the last 30 days.
    4. Dividend Accumulation: Collects net-of-withholding-tax dividends into the cash balance.
"""

import pandas as pd
import numpy as np

class CRATaxLedger:
    def __init__(self, initial_capital_cad=1_000_000.0):
        
        self.cash_cad = initial_capital_cad
        
        # Portfolio state: {permno: {'shares': float, 'acb_per_share': float}}
        self.positions = {}
        
        # Tax tracking
        self.realized_gains_cad = 0.0
        self.realized_losses_cad = 0.0
        
        # CRA 30-Day Superficial Loss Trackers
        self.superficial_loss_lockouts = {}  # Forward rule: {permno: expiration_date}
        self.last_buy_dates = {}      # Backward rule: {permno: last_purchase_date}
        
        # Audit Trail
        self.trade_history =[]

    def get_current_holdings(self, current_prices_cad):
        """Calculates the current CAD value of the portfolio.
        and a doctionary of permnos and shares in the portfolio  
        
        current_prices_cad is a dictionary giving us the current state of the market. 
        Keys are premnos, Values are prices
        example at a specific date: 
        current_prices_cad = {
            14593: 138.50,  # AAPL's CAD price today
            10104: 45.20,   # ORCL's CAD price today
            59408: 210.15,  # HD's CAD price today
            # ... and so on for all 500 stocks in the S&P 500 today
            }
        """
        current_shares = {}
        total_equity_cad = 0.0
        
        for permno, pos in self.positions.items():
            shares = pos['shares']
            if shares > 1e-6:
                current_shares[permno] = shares

                 # 1. Get the price. If the key is missing entirely, default to ACB.
                price = current_prices_cad.get(permno, pos['acb_per_share'])

                # 2. If the key was present but the value was NaN or 0, STILL default to ACB.
                if pd.isna(price) or price <= 0:
                    price = pos['acb_per_share']

                total_equity_cad += shares * price
     
        total_aum = total_equity_cad + self.cash_cad
        return current_shares, total_aum

    def update_lockouts(self, current_date):
        """Removes expired 30-day superficial loss lockouts."""
        expired =[p for p, exp_date in self.superficial_loss_lockouts.items() if current_date > exp_date]
        for p in expired:
            del self.superficial_loss_lockouts[p]

    def get_optimizer_constraints(self, current_date):
        """
        Generates the restricted lists for the CVXPY optimizer.
        Forces the math to find proxy stocks instead of attempting illegal CRA trades.
        """
        self.update_lockouts(current_date)
        
        # 1. Forward Rule: Cannot BUY if we harvested a loss in the last 30 days
        do_not_buy = list(self.superficial_loss_lockouts.keys())
        
        # 2. Backward Rule: Cannot HARVEST if we bought in the last 30 days
        do_not_harvest =[]
        for permno, last_buy_date in self.last_buy_dates.items():
            if (current_date - last_buy_date).days <= 30:
                do_not_harvest.append(permno)
                
        return do_not_buy, do_not_harvest

    def process_dividends(self, current_shares, div_data_cad):
        """Adds CAD-adjusted dividends to the cash balance."""
        for permno, dps in div_data_cad.items():
            if permno in current_shares and not pd.isna(dps) and dps > 0:
                payout = current_shares[permno] * dps
                self.cash_cad += payout

    def _buy(self, permno, shares, price_cad, current_date):
        """Executes a buy order, enforcing CRA ACB pooling."""
        cost = shares * price_cad
        self.cash_cad -= cost
        
        # Log for backward 30-day rule
        self.last_buy_dates[permno] = current_date  
        
        # ACB Pooling Math
        if permno not in self.positions or self.positions[permno]['shares'] < 1e-6:
            self.positions[permno] = {'shares': shares, 'acb_per_share': price_cad}
        else:
            old_shares = self.positions[permno]['shares']
            old_acb = self.positions[permno]['acb_per_share']
            new_shares = old_shares + shares
            # (Old Total Cost + New Total Cost) / New Total Shares
            new_acb = ((old_shares * old_acb) + cost) / new_shares
            self.positions[permno] = {'shares': new_shares, 'acb_per_share': new_acb}
            
        self.trade_history.append({
            'date': current_date, 'permno': permno, 'action': 'BUY', 
            'shares': shares, 'price_cad': price_cad, 'realized_pnl': 0.0
        })

    def _sell(self, permno, shares_to_sell, price_cad, current_date):
        """Executes a sell order and calculates ACB-based capital gains/losses."""
        revenue = shares_to_sell * price_cad
        self.cash_cad += revenue
        
        pos = self.positions[permno]
        acb = pos['acb_per_share']
        
        # Calculate realized PnL against the pooled Average Cost Base
        realized_pnl = (price_cad - acb) * shares_to_sell
        loss_harvested = False
        
        if realized_pnl < -1e-4:  # Allowing slight floating point tolerance
            self.realized_losses_cad += abs(realized_pnl)
            loss_harvested = True
        else:
            self.realized_gains_cad += realized_pnl
            
        # Deduct shares (ACB per share remains identical after a sell)
        pos['shares'] -= shares_to_sell
        if pos['shares'] < 1e-6:
            pos['shares'] = 0.0

        self.trade_history.append({
            'date': current_date, 'permno': permno, 'action': 'SELL', 
            'shares': shares_to_sell, 'price_cad': price_cad, 'realized_pnl': realized_pnl
        })

        return loss_harvested

    def execute_trades(self, target_shares, current_prices_cad, current_date):
        """Compares target_shares to current_shares and executes the deltas.
        
        """
        self.update_lockouts(current_date)
        current_shares, _ = self.get_current_holdings(current_prices_cad)
        
        all_permnos = set(target_shares.keys()).union(set(current_shares.keys()))
        
        # STEP 1: SELLS (Free up cash. Sells fund the Buys. )
        for permno in all_permnos:
            tgt = target_shares.get(permno, 0.0)
            cur = current_shares.get(permno, 0.0)
            delta = tgt - cur
            
            if delta < -1e-6: 
                if permno not in self.positions:
                    continue
                acb = self.positions[permno]['acb_per_share']

                safe_price = current_prices_cad.get(permno, acb)
                if pd.isna(safe_price) or safe_price <= 0:
                    safe_price = acb
                
                #failsafe to make sure the sale is not blocked 
                if permno in self.last_buy_dates:
                    if (current_date - self.last_buy_dates[permno]).days <= 30:
                        # Check if selling it would realize a loss
                        if safe_price < acb:
                            continue # Failsafe: Block illegal harvest
                
                shares_to_sell = min(abs(delta), cur) # the max we sell is the the whole position. We don't short.
                is_loss = self._sell(permno, shares_to_sell, safe_price, current_date)
                if is_loss:
                    self.superficial_loss_lockouts[permno] = current_date + pd.Timedelta(days=30)
                        
        # STEP 2: BUYS
        for permno in all_permnos:
            tgt = target_shares.get(permno, 0.0)
            cur = current_shares.get(permno, 0.0)
            if pd.isna(tgt) or pd.isna(cur):
                continue

            delta = tgt - cur
            
            if delta > 1e-6: 
                # Failsafe 1: Block illegal buys (Superficial Sale)
                if permno in self.superficial_loss_lockouts:
                    continue

                price_cad = current_prices_cad.get(permno)
                if pd.isna(price_cad) or price_cad <= 0:
                    if permno in self.positions:
                        price_cad = self.positions[permno]['acb_per_share']
                    else:
                        continue # Cannot buy a brand new stock with no valid price!

                cost = delta * price_cad
                    
                    # --- CASH PROTECTION FAILSAFE ---
                if self.cash_cad >= cost:
                    self._buy(permno, delta, price_cad, current_date)
                else:
                    # We are short on cash (caused by Dust Filter or rounding)!
                    # Buy exactly what we can afford with the remaining cash.
                    affordable_shares = np.floor((self.cash_cad / price_cad) * 10000.0) / 10000.0
                    if affordable_shares > 1e-6:
                        self._buy(permno, affordable_shares, price_cad, current_date)

if __name__ == "__main__":
    print("--- Testing CRA TaxLedger ---")
    ledger = CRATaxLedger()
    d1, d2, d3 = pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-15'), pd.to_datetime('2023-02-10')
    
    # 1. Buy at $10, Buy at $20 -> ACB becomes $15
    ledger._buy(12345, 100, 10.0, d1)
    ledger._buy(12345, 100, 20.0, d2)
    print(f"Post-Buys Position: {ledger.positions[12345]}")
    
    # 2. Sell at $12 -> Should harvest a $300 loss ($12 - $15 ACB = -$3 * 100 shares)
    is_loss = ledger._sell(12345, 100, 12.0, d3)
    print(f"Triggered Wash Sale Lockout? {is_loss}")
    print(f"Realized Losses: ${ledger.realized_losses_cad}")
    
    # 3. Check Constraints
    no_buy, no_harvest = ledger.get_optimizer_constraints(d3)
    print(f"Do Not Buy List (Forward Rule): {no_buy}")
    print(f"Do Not Harvest List (Backward Rule): {no_harvest}")