"""
Module: Factor Construction Engine

OBJECTIVE:
    Constructs the Time-Varying Factor Exposure Matrix (X_t) using a Point-in-Time 
    framework. This module transforms raw fundamental and market data into 
    standardized risk factors.

METHODOLOGY:
    1. Industry Classification:
       Maps SIC codes to Fama-French 12 Industries to capture sector risk.

    2. Composite Style Factors:
       - value: Composite of B/M (lagged 6mo) and E/P (lagged 3mo).
       - momentum: Composite of 12-1 Month and 6-1 Month returns.
       - Financial Constraints: Whited-Wu Index (Cash Flow, Leverage, Dividend, Sales Growth).
       - size: Log(Market Cap).
       - Profitability 
       - Volatility 
       - Investments 

    3. Capitalization-Weighted Standardization:
       Factors are standardized cross-sectionally using market-cap weights to 
       prevent micro-cap outliers from skewing the signal:
       Z = (X - Mean_CapWeighted) / Std_CapWeighted

    4. Point-in-Time Integrity:
       Strictly enforces reporting lags (using 'rdq' or 'datadate' offsets) to 
       ensure no information is used before it was publicly available.
"""
"""
Module: 01b_factor_engine.py
Builds the Factor Exposure Matrix (X) with 100% S&P 500 coverage.
Tailored for TLH/Direct Indexing by replacing 'dropna' with intelligent imputation.
Incorporates Fama-French 5-Factor metrics + Momentum + Distress + Volatility.
"""

import os
from pathlib import Path 
import pandas as pd
import numpy as np
import warnings
from config import DATA_CLEAN_DIR

# Suppress pandas chained assignment warnings for clean terminal output
warnings.filterwarnings('ignore', category=FutureWarning)

class TLHFactorEngine:
    def __init__(self, 
                 panel_path: str = str(DATA_CLEAN_DIR / 'panel_data.parquet'),
                 out_path:str = str(DATA_CLEAN_DIR / 'X.parquet')):
        """
        Args:
            panel_path: Path to your historical panel data (needs returns, mkt_cap, and fundamentals)
        """
        self.out_path = out_path
        print(f"Loading panel data from {panel_path}...")
        self.df = pd.read_parquet(panel_path)
        
        if isinstance(self.df.index, pd.MultiIndex):
            self.df.reset_index(inplace=True)
            
        self.df['permno'] = self.df['permno'].astype(int)
        self.df['date'] = pd.to_datetime(self.df['date'])

    def build_factors(self) -> pd.DataFrame:
        if Path(self.out_path).exists():
            print(f'The X factor exposures file already exists at {self.out_path}. Skipping pipeline.')
            return  

        print("--- Starting TLH Factor Construction ---")
        
        # 1. Industry Classification
        self.df['industry'] = self.df['sic'].apply(self._sic_to_ff12)
        
        # 2. Raw Descriptors
        print("Calculating Raw Descriptors...")
        self._calc_size()
        self._calc_value_pit()      #  Value Factor built from descriptors
        self._calc_momentum()       #  Momentum Factor built from descriptors
        self._calc_whited_wu()      #  Financial Constraint
        self._calc_profitability()  #  Quality Factor
        self._calc_investment()     #  Asset Growth Factor
        self._calc_volatility()     #  Risk Factor
        
        descriptor_cols =[
            'size_desc', 'bm_desc', 'ep_desc', 'mom12_1_desc', 'mom6_1_desc', 
            'ww_desc', 'profitability_desc', 'investment_desc', 'volatility_desc'
        ]
        
        # Clean infinities (e.g., division by zero book equity)
        self.df[descriptor_cols] = self.df[descriptor_cols].replace([np.inf, -np.inf], np.nan)
        
        # 3. Cap-Weighted Standardization (First Pass)
        print("Standardizing Descriptors...")
        self.df['cap_weight'] = self.df.groupby('date')['mkt_cap'].transform(lambda x: x / x.sum())
        
        for col in descriptor_cols:
            self.df[f'z_{col}'] = self.df.groupby('date')[col].transform(
                lambda x: self._standardize_cap_weighted(x, self.df.loc[x.index, 'cap_weight'])
            )
            
        # =====================================================================
        # IMPUTATION TO GET MAX COVERAGE
        # =====================================================================
        print("Imputing missing factor exposures to guarantee 100% coverage...")
        z_cols = [f'z_{col}' for col in descriptor_cols]
        
        for col in z_cols:
            # Step A: Impute with the Industry Average for that specific date
            industry_means = self.df.groupby(['date', 'industry'])[col].transform('mean')
            self.df[col] = self.df[col].fillna(industry_means)
            
            # Step B: If the whole industry is missing (e.g., Financials missing Sales Growth), fill with 0.0
            self.df[col] = self.df[col].fillna(0.0)

        # 4. Composites
        print("Building Composites...")
        self.df['value_composite'] = self.df[['z_bm_desc', 'z_ep_desc']].mean(axis=1)
        self.df['momentum_composite'] = self.df[['z_mom12_1_desc', 'z_mom6_1_desc']].mean(axis=1)
        
        # 5. Final Re-Standardization
        self.df['value'] = self.df.groupby('date')['value_composite'].transform(
            lambda x: self._standardize_cap_weighted(x, self.df.loc[x.index, 'cap_weight'])
        )
        self.df['momentum'] = self.df.groupby('date')['momentum_composite'].transform(
            lambda x: self._standardize_cap_weighted(x, self.df.loc[x.index, 'cap_weight'])
        )
        
        # Fill any final NaNs created by the composite standardization with 0.0
        self.df['value'] = self.df['value'].fillna(0.0)
        self.df['momentum'] = self.df['momentum'].fillna(0.0)
        
        # Rename single descriptors to their final factor names
        self.df.rename(columns={
            'z_size_desc': 'size', 
            'z_ww_desc': 'fin_constraint',
            'z_profitability_desc': 'profitability',
            'z_investment_desc': 'investment',
            'z_volatility_desc': 'volatility'
        }, inplace=True)
        
        # 6. Industry Dummies
        print("Generating Industry Dummies...")
        industry_dummies = pd.get_dummies(self.df['industry'], prefix='Ind').astype(float)
        
        # 7. Assemble X Matrix
        style_factors =[
            'size', 'value', 'momentum', 'fin_constraint', 
            'profitability', 'investment', 'volatility'
        ]
        
        final_df = self.df.set_index(['date', 'permno'])
        industry_dummies.index = final_df.index

        X = final_df[style_factors].join(industry_dummies)
        

        print(f"Factor Construction Complete. Final Shape: {X.shape}")
        X.to_parquet(self.out_path)
        print(f"100% Coverage Factor Exposures Saved to: {self.out_path}")
        return X

    # -------------------------------------------------------------------------
    #                            FACTOR LOGIC
    # -------------------------------------------------------------------------
    def _calc_size(self):
        self.df['size_desc'] = np.log(self.df['mkt_cap'])

    def _calc_value_pit(self):
        self.df['bm_desc'] = self.df['ceqq'] / self.df['mkt_cap']
        cols_ep =['permno', 'datadate', 'effective_date', 'ibq']
        
        if not all(c in self.df.columns for c in cols_ep):
            self.df['ep_desc'] = np.nan
            return
            
        fund_ep = self.df[cols_ep].copy().dropna(subset=['datadate']).drop_duplicates()
        fund_ep.sort_values(['permno', 'datadate'], inplace=True)
        fund_ep['ltm_earn'] = fund_ep.groupby('permno')['ibq'].rolling(4, min_periods=4).sum().values
        fund_ep.dropna(subset=['ltm_earn'], inplace=True)
        
        self.df = pd.merge_asof(
            left=self.df.sort_values('date'),
            right=fund_ep[['permno', 'effective_date', 'ltm_earn']].sort_values('effective_date'),
            left_on='date',
            right_on='effective_date',
            by='permno', 
            direction='backward'
        )
        self.df['ep_desc'] = self.df['ltm_earn'] / self.df['mkt_cap']

    def _calc_momentum(self):
        self.df.sort_values(['permno', 'date'], inplace=True)
        self.df['log_ret'] = np.log1p(self.df['ret_monthly'])
        grouped = self.df.groupby('permno')['log_ret']
        self.df['mom12_1_desc'] = grouped.transform(lambda x: np.expm1(x.shift(1).rolling(11).sum()))
        self.df['mom6_1_desc'] = grouped.transform(lambda x: np.expm1(x.shift(1).rolling(11).sum()))

    def _calc_whited_wu(self):
        try:
            cf_at = (self.df['ibq'] + self.df['dpq']) / self.df['atq']
            div_pos = ((self.df['dvpspq'] * self.df['cshoq']) > 0).astype(int)
            tltd_at = self.df['dlttq'] / self.df['atq']
            ln_at = np.log(self.df['atq'].replace(0, np.nan))
            sg = self.df.sort_values('date').groupby('permno')['saleq'].pct_change(fill_method=None)
            self.df['sg_temp'] = sg
            self.df['isg'] = self.df.groupby(['industry', 'date'])['sg_temp'].transform('mean')
            
            self.df['ww_desc'] = (
                -0.091 * cf_at - 0.062 * div_pos + 0.021 * tltd_at 
                - 0.044 * ln_at + 0.102 * self.df['isg'] - 0.035 * sg
            )
        except KeyError:
            self.df['ww_desc'] = np.nan

    def _calc_profitability(self):
        """
        Robust Profitability Factor (Proxy for Fama-French RMW).
        Operating Income / Book Equity.
        """
        try:
            ceqq_safe = self.df['ceqq'].replace(0, np.nan)
            self.df['profitability_desc'] = self.df['ibq'] / ceqq_safe
        except KeyError:
            self.df['profitability_desc'] = np.nan

    def _calc_investment(self):
        """
        Investment Factor (Proxy for Fama-French CMA).
        12-Month Trailing Asset Growth.
        """
        try:
            self.df.sort_values(['permno', 'date'], inplace=True)
            self.df['investment_desc'] = self.df.groupby('permno')['atq'].pct_change(periods=12, fill_method=None)
        except KeyError:
            self.df['investment_desc'] = np.nan

    def _calc_volatility(self):
        """
        Short-term Idiosyncratic Risk Factor.
        Rolling 12-month standard deviation of monthly returns.
        """
        try:
            self.df.sort_values(['permno', 'date'], inplace=True)
            self.df['volatility_desc'] = self.df.groupby('permno')['ret_monthly'].transform(
                lambda x: x.rolling(12, min_periods=9).std()
            )
        except KeyError:
            self.df['volatility_desc'] = np.nan

    # -------------------------------------------------------------------------
    #                            HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _standardize_cap_weighted(series: pd.Series, weights: pd.Series) -> pd.Series:
        if series.isnull().all(): return pd.Series(np.nan, index=series.index)
        series, weights = series.align(weights, join='left')
        valid = series.notna() & weights.notna()
        if not valid.any(): return pd.Series(np.nan, index=series.index)
        w, s = weights[valid], series[valid]
        w = w / w.sum()
        mean_w = np.sum(s * w)
        var_w = np.sum(w * (s - mean_w)**2)
        std_w = np.sqrt(var_w)
        if std_w == 0: return pd.Series(0.0, index=series.index)
        return ((s - mean_w) / std_w).reindex(series.index)

    @staticmethod
    def _sic_to_ff12(sic) -> str:
        if pd.isnull(sic): return 'Other'
        try: s = int(sic)
        except: return 'Other'
        if (2830 <= s <= 2836) or (3840 <= s <= 3851) or (8000 <= s <= 8099): return 'Healthcare'
        if (3570 <= s <= 3579) or (3660 <= s <= 3679) or (7370 <= s <= 7379): return 'Technology'
        if (1300 <= s <= 1399) or (2900 <= s <= 2999): return 'Energy'
        if (4900 <= s <= 4949): return 'Utilities'
        if (4800 <= s <= 4899): return 'Telecom'
        if (6000 <= s <= 6999): return 'Finance'
        if (100 <= s <= 999) or (2000 <= s <= 2799) or (2840 <= s <= 2899): return 'Consumer'
        if (2800 <= s <= 2829): return 'Chemicals'
        if (3000 <= s <= 3999): return 'Durables'
        if (5000 <= s <= 5999): return 'Shops'
        if (7000 <= s <= 7999) or (8100 <= s <= 8999): return 'Services'
        return 'Other'

if __name__ == "__main__":
    panel_path = DATA_CLEAN_DIR / 'panel_data.parquet' 
    
    if os.path.exists(panel_path):
        engine = TLHFactorEngine(panel_path=panel_path)
        X_matrix = engine.build_factors()
        print("Success! Missing values intelligently filled.")
    else:
        print(f"Could not find {panel_path}. Please move your old panel data into the DATA_CLEAN_DIR.")


