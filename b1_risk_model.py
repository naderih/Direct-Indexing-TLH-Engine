"""
Module: b1_risk_model.py
Phase B: Risk Matrix Generation

OBJECTIVE:
    Dynamically generates point-in-time covariance matrices using the Fundamental Factor Model.
    Reconstructs V = X * F * X^T + Delta for any given date and S&P 500 constituent list.
    
    Because TLH runs daily but Factor Models update monthly, this engine intelligently 
    searches backward for the most recent valid factor model and aligns it to the 
    exact daily constituents.
"""

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import warnings
from config import DATA_CLEAN_DIR

class FactorRiskModel:
    def __init__(self, 
                 x_path = DATA_CLEAN_DIR / 'X.parquet', 
                 f_path = DATA_CLEAN_DIR / 'factor_cov_matrices.parquet', 
                 idio_path = DATA_CLEAN_DIR / 'idio_vol.parquet', 
                 univ_path = DATA_CLEAN_DIR / 'tlh_universe.parquet'):
        
        print("Loading Risk Model inputs from Phase A (Quant Infrastructure)...")
        
        # 1. TLH Universe (for fallback returns and daily permno queries)
        self.df_univ = pd.read_parquet(univ_path)
        self.df_univ['date'] = pd.to_datetime(self.df_univ['date'])
        
        print("Building daily return matrix for Ledoit-Wolf failsafe...")
        self.ret_matrix = self.df_univ.pivot_table(index='date', columns='permno', values='dlyret').fillna(0.0)

        # 2. Factor Exposures (X)
        print("Loading Factor Exposures (X)...")
        self.X_data = pd.read_parquet(x_path)
        if self.X_data.index.names != ['permno', 'date']:
            self.X_data = self.X_data.reset_index().set_index(['permno', 'date'])

        # 3. Factor Covariances (F)
        print("Loading Factor Covariance Matrices (F)...")
        self.F_data = pd.read_parquet(f_path)
        
        # Ensure F_data is correctly structured as a MultiIndex: ['date', 'factor']
        if isinstance(self.F_data.index, pd.MultiIndex):
            self.F_data = self.F_data.reset_index()
            
        self.F_data['date'] = pd.to_datetime(self.F_data['date'])
        
        # The second column holds the row-wise factor names
        factor_col_name = self.F_data.columns[1]
        self.F_data.rename(columns={factor_col_name: 'factor'}, inplace=True)
        self.F_data.set_index(['date', 'factor'], inplace=True)

        # 4. Idiosyncratic Risk (Delta / resid_vol)
        print("Loading Idiosyncratic Risk (Delta)...")
        self.idio_data = pd.read_parquet(idio_path)
        if self.idio_data.index.names !=['permno', 'date']:
            self.idio_data = self.idio_data.reset_index().set_index(['permno', 'date'])

    def build_factor_covariance(self, target_date, current_permnos):
        """
        Builds the N x N covariance matrix V for the given target_date.
        """
        target_date = pd.to_datetime(target_date)
        
        # 1. Temporal AsOf Search: Find closest prior factor model date
        available_dates = self.F_data.index.get_level_values('date').unique()
        valid_dates = available_dates[available_dates <= target_date]
        
        if len(valid_dates) == 0:
            warnings.warn(f"[{target_date.date()}] No prior factor data. Using Ledoit-Wolf Fallback.")
            return self._fallback_ledoit_wolf(target_date, current_permnos)
        
        model_date = valid_dates.max()
        
        try:
            # 2. Extract X, F, and Delta components
            X_t = self.X_data.xs(model_date, level='date')
            F_t = self.F_data.xs(model_date, level='date')
            idio_t = self.idio_data.xs(model_date, level='date')

            # 3. Align Dimensions (Intersection of shared factors)
            common_factors = X_t.columns.intersection(F_t.index)
            X_t = X_t[common_factors]
            F_t = F_t.loc[common_factors, common_factors]

            # 4. Reindex to the exact S&P 500 roster trading on the target_date
            X_aligned = X_t.reindex(current_permnos).fillna(0.0)
            
            # Missing stocks get the median idiosyncratic volatility of the universe
            median_resid_vol = idio_t['resid_vol'].median() if not idio_t.empty else 0.05
            idio_aligned = idio_t.reindex(current_permnos).copy()
            idio_aligned['resid_vol'] = idio_aligned['resid_vol'].fillna(median_resid_vol)
            
            # D_sq (Variance) creation
            D_sq = idio_aligned['resid_vol'] ** 2
            spec_cov = np.diag(D_sq.values.flatten().astype(float))

            # 5. Matrix Algebra: V = X * F * X^T + Delta
            X_mat = X_aligned.values.astype(float)
            F_mat = F_t.values.astype(float)
            
            sys_cov = X_mat @ F_mat @ X_mat.T
            V_raw = sys_cov + spec_cov
            
            # Enforce exact symmetry for CVXPY PSD (Positive-Semi-Definite) safety
            V_symmetric = (V_raw + V_raw.T) / 2.0
            
            return pd.DataFrame(V_symmetric, index=current_permnos, columns=current_permnos)

        except Exception as e:
            warnings.warn(f"[{target_date.date()}] Structural build failed ({e}). Using Ledoit-Wolf.")
            return self._fallback_ledoit_wolf(target_date, current_permnos)

    def _fallback_ledoit_wolf(self, target_date, current_permnos, lookback_days=252):
        """
        Failsafe mechanism using empirical shrinkage.
        """
        historical_returns = self.ret_matrix.loc[:target_date].iloc[-lookback_days:]
        active_returns = historical_returns.reindex(columns=current_permnos).fillna(0.0)
        
        if len(active_returns) < 10:
            return pd.DataFrame(np.eye(len(current_permnos)) * 0.005, index=current_permnos, columns=current_permnos)

        lw = LedoitWolf()
        # Scale * 21 to convert daily covariance to the structural model's native monthly scale
        cov_matrix = lw.fit(active_returns).covariance_ * 21.0
        
        V_symmetric = (cov_matrix + cov_matrix.T) / 2.0
        return pd.DataFrame(V_symmetric, index=current_permnos, columns=current_permnos)

if __name__ == "__main__":
    print("--- Testing Phase B Risk Model ---")
    
    # 1. Initialize the model
    risk_model = FactorRiskModel()
    
    # 2. Pick a test date (e.g., end of 2015)
    test_date = pd.to_datetime('2015-12-31')
    print(f"\nExtracting active S&P 500 universe for {test_date.date()}...")
    
    # 3. Get the actual constituents alive on that day
    daily_univ = risk_model.df_univ[risk_model.df_univ['date'] == test_date]
    test_permnos = daily_univ['permno'].unique()
    
    if len(test_permnos) == 0:
        print(f"No active stocks found for {test_date.date()}. Check your data coverage.")
    else:
        print(f"Found {len(test_permnos)} active S&P 500 constituents. Building V Matrix...")
        
        # 4. Generate the Covariance Matrix
        V = risk_model.build_factor_covariance(test_date, test_permnos)
        
        # 5. Output and Sanity Checks
        print(f"\nSUCCESS! Generated V Matrix shape: {V.shape}")
        
        print("\n--- Sanity Checks (Crucial for CVXPY Optimizer) ---")
        nan_count = V.isna().sum().sum()
        print(f"1. Total NaNs in V matrix: {nan_count} (Should be 0)")
        
        is_symmetric = np.allclose(V, V.T, atol=1e-8)
        print(f"2. Is V strictly symmetric? {is_symmetric} (Must be True)")
        
        min_variance = np.diag(V).min()
        print(f"3. Minimum asset variance: {min_variance:.6f} (Must be > 0)")
        
        print("\nSample Output (Top 5x5):")
        print(V.iloc[:5, :5])