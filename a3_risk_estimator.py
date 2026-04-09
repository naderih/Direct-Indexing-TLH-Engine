"""
Module: 01c_risk_estimator.py
Dynamic Risk Model Estimator (Fama-MacBeth & EWMA)

OBJECTIVE:
    Constructs the F and Delta matrices for the Fundamental Factor Risk Model.
    
    1. Cross-Sectional Regression (Fama-MacBeth):
       R_{i,t} = X_{i,t} * F_t + u_{i,t}
       Extracts pure factor returns (F_t) and specific stock residuals (u_t).

    2. Risk Forecasting (EWMA):
       Forecasts the future covariance matrix (F) and idiosyncratic volatility (Delta) 
       using a 36-month Exponentially Weighted Moving Average to capture volatility clustering.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import DATA_CLEAN_DIR

class FactorRiskEstimator:
    def __init__(self, 
                 panel_path=DATA_CLEAN_DIR / 'panel_data.parquet', 
                 factors_path=DATA_CLEAN_DIR / 'X.parquet',
                 cov_path=DATA_CLEAN_DIR / 'factor_cov_matrices.parquet', 
                 vol_path= DATA_CLEAN_DIR /'idio_vol.parquet',
                 half_life=36, 
                 regression_weighting="WLS"):
        
        self.panel_path = panel_path
        self.factors_path = factors_path
        self.cov_path = cov_path
        self.vol_path = vol_path
        self.half_life = half_life
        
        # WLS (Cap-Weighted) is standard for institutional risk models so micro-caps 
        # don't dominate the factor return estimation.
        if regression_weighting not in ['OLS', 'WLS']:
            raise ValueError("regression_weighting must be 'OLS' or 'WLS'")
        self.regression_weighting = regression_weighting

    def load_data(self):
        print("Loading Panel Data and Factor Exposures (X)...")
        if not os.path.exists(self.panel_path) or not os.path.exists(self.factors_path):
            raise FileNotFoundError(f"Missing required data in {DATA_CLEAN_DIR}")
            
        self.panel_data = pd.read_parquet(self.panel_path)
        self.exposures = pd.read_parquet(self.factors_path)
        
        self._ensure_multiindex(self.panel_data)
        self._ensure_multiindex(self.exposures)

    def _ensure_multiindex(self, df):
        """Standardizes index structure to (permno, date)."""
        if df.index.names != ['permno', 'date']:
            df.reset_index(inplace=True)
            df.set_index(['permno', 'date'], inplace=True)
        df.sort_index(inplace=True)

    def run_fama_macbeth(self):
        print(f"Running Fama-MacBeth Regressions ({self.regression_weighting})...")
        
        # 1. Align Data (Inner Join ensures we only regress where we have both Returns and Exposures)
        y = self.panel_data[['ret_monthly', 'mkt_cap']]
        X = self.exposures
        
        aligned = y.join(X, how='inner')
        if aligned.empty:
            raise ValueError("Data Alignment Failed: No common (permno, date) rows found.")
            
        dates = aligned.index.get_level_values('date').unique().sort_values()
        
        factor_ret_list = []
        resid_list =[]
        factor_cols = [c for c in aligned.columns if c not in['ret_monthly', 'mkt_cap']]
        
        for date in dates:
            try:
                monthly_slice = aligned.xs(date, level='date').copy()
            except KeyError:
                continue
            
            # Drop any rows with missing returns to ensure exact alignment with weights
            monthly_slice.dropna(subset=['ret_monthly'] + factor_cols, inplace=True)
            
            y_t = monthly_slice['ret_monthly'].astype(float)
            X_t = monthly_slice[factor_cols].astype(float)

            # Constraint: Need more observations than factors to solve
            if len(y_t) < len(factor_cols) + 2:
                continue

            try:
                if self.regression_weighting == "OLS":
                    model = sm.OLS(y_t, X_t)
                    results = model.fit()
                elif self.regression_weighting == "WLS":
                    # Square root of market cap is the industry standard for WLS factor weighting
                    weights = np.sqrt(monthly_slice['mkt_cap'].astype(float))
                    results = sm.WLS(y_t, X_t, weights=weights).fit()

                # A. Store Factor Returns (Coefficients)
                f_t = results.params
                f_t.name = date
                factor_ret_list.append(f_t)
                
                # B. Store Residuals (Specific Returns)
                u_t = results.resid.to_frame('specific_ret')
                u_t['date'] = date 
                u_t.set_index('date', append=True, inplace=True) 
                resid_list.append(u_t)
                
            except Exception as e:
                # Catch singular matrix errors (e.g., dummy variable trap if data is perfectly collinear)
                continue

        if not factor_ret_list:
             raise ValueError("Regressions failed for ALL dates. Check input data quality.")

        self.factor_returns = pd.DataFrame(factor_ret_list)
        self.factor_returns.index.name = 'date'
        
        self.specific_returns = pd.concat(resid_list)
        if self.specific_returns.index.names == ['permno', 'date']:
             self.specific_returns = self.specific_returns.swaplevel('permno', 'date')
        self.specific_returns.sort_index(inplace=True)
        
        print(f"Fama-MacBeth Complete. Extracted pure factor returns for {len(self.factor_returns)} months.")

    def predict_risk_matrices(self):
        print("Forecasting Forward-Looking Risk Matrices (EWMA, Half-Life = 36 months)...")
        
        # 1. Forecast Factor Covariance Matrix (F)
        self.factor_cov_matrices = self.factor_returns.ewm(
            halflife=self.half_life, min_periods=self.half_life).cov()
            
        # Explicitly name the levels so `02_risk_model.py` can parse it flawlessly
        self.factor_cov_matrices.index.names = ['date', 'factor']

        # 2. Forecast Idiosyncratic Variance (Delta)
        u_wide = self.specific_returns['specific_ret'].unstack(level='permno')
        u_sq = u_wide ** 2
        
        self.idio_var = u_sq.ewm(halflife=self.half_life, min_periods=self.half_life).mean()
        self.idio_vol = np.sqrt(self.idio_var)
        
        # Handle pandas 'future_stack' warning gracefully depending on version
        try:
            self.idio_vol = self.idio_vol.stack(future_stack=True).to_frame('resid_vol').dropna()
        except TypeError:
            self.idio_vol = self.idio_vol.stack().to_frame('resid_vol').dropna()
            
        self.idio_vol.index.names = ['date', 'permno']
        self.idio_vol.sort_index(inplace=True)

    def save_outputs(self):
                
        self.factor_cov_matrices.to_parquet(self.cov_path)
        self.idio_vol.to_parquet(self.vol_path)
        
        print(f"\nSUCCESS: Risk forecasts saved.")
        print(f"-> F Matrix: {self.cov_path}")
        print(f"-> Delta: {self.vol_path}")

    def run_pipeline(self):
        if (self.cov_path.exists()) and (self.vol_path.exists()):
            print(f'The cov matrix and idio files already exists at {self.out_path}. Skipping pipeline.')
            return  
        
        self.load_data()
        self.run_fama_macbeth()
        self.predict_risk_matrices()
        self.save_outputs()

if __name__ == "__main__":
    estimator = FactorRiskEstimator()
    estimator.run_pipeline()