import pandas as pd 
import numpy as np 
import pandas_datareader.data as web
from config import (
    init_directories,
    DAILY_PRICES_PATH,
    SP500_INDEX_PATH,
    DELISTING_PATH,
    DATA_CLEAN_DIR,
    US_WITHHOLDING_TAX_RATE
)


class DataPipeline:
    """
    Ingests raw CRSP datasets, patches terminal delisting returns, applies split 
    adjustments, integrates Canadian FX rates, and outputs a unified point-in-time 
    universe tailored for a Canadian Tax-Loss Harvesting engine.
    """

    def __init__(self, 
                 out_path = DATA_CLEAN_DIR / 'tlh_universe.parquet'):
        self.df = None 
        self.out_path = out_path
        
    def load_and_merge_daily_market_data(self):
        "Loads daily prices, filters for the 2000+ era, and patches terminal returns."""
        print("Loading daily data...")
        daily_cols = [
            'PERMNO', 'DlyCalDt', 'DlyPrc', 'ShrOut', 'DlyRet', 'DlyRetx', 
            'DisFacPr', 'DisFacShr', 'SICCD', 'sprtrn', 'DisDivAmt'
        ]
        
        # Load the daily file
        self.df = pd.read_csv(
            str(DAILY_PRICES_PATH), 
            usecols=daily_cols,
            dtype={'PERMNO': 'int32', 'SICCD': 'float32', 'ShrOut': 'float32'}
        )
        self.df.columns = self.df.columns.str.lower()
        self.df.rename(columns={'dlycaldt': 'date'}, inplace=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['currency'] = 'USD'
        # 1. Temporal Filter: Restrict to Backtest Window (2000-2023)
        print("Filtering data for the 2000-2023 backtest window...")
        mask = (self.df['date'] >= '2000-01-01') & (self.df['date'] <= '2023-12-31')
        self.df = self.df[mask].copy()
        
        # 2. Load Delisting Data to patch terminal returns (Bankruptcies/M&A)
        print("Loading delisting data to capture -1.0 bankruptcy returns...")
        df_delist = pd.read_csv(
            str(DELISTING_PATH), 
            usecols=['PERMNO', 'DelistingDt', 'DelRet'],
            dtype={'permno': 'int32'}
        )
        df_delist.columns = df_delist.columns.str.lower()
        df_delist['delistingdt'] = pd.to_datetime(df_delist['delistingdt'])
        df_delist = df_delist[df_delist['delistingdt'] >= '2000-01-01']
        df_delist = df_delist[df_delist['delistingdt'] <= '2023-12-31']

        # 3. Merge and Patch
        print("Patching delisting returns into the daily price series...")
        self.df = pd.merge(
            self.df, 
            df_delist, 
            how='left', 
            left_on=['permno', 'date'], 
            right_on=['permno', 'delistingdt']
        )
        
        # Where a true delisting event occurred, overwrite the daily exchange return
        mask = self.df['delret'].notna()
        self.df.loc[mask, 'dlyret'] = self.df.loc[mask, 'delret']
        
        # Drop the redundant merge columns
        self.df.drop(columns=['delistingdt', 'delret'], inplace=True)


    def process_prices_and_dividends(self):
        
        """Applies strictly aligned cumulative split factors and applies tax drag."""
        
        print("Calculating Cumulative Split Adjustments and Withholding Taxes...")
        
        self.df['dlyprc'] = self.df['dlyprc'].abs()
        
        # ---------------------------------------------------------
        # THE CRSP 1 + FACPR FIX
        # CRSP logs the *additional* shares (e.g., 4-for-1 split = 3.0).
        # We must fill NaNs with 0.0 (no split), then ADD 1.0 to get the true divisor!
        # ---------------------------------------------------------
        self.df['disfacpr'] = self.df['disfacpr'].fillna(0.0)
        self.df['disfacshr'] = self.df['disfacshr'].fillna(0.0)
        
        self.df['true_divisor_prc'] = 1.0 + self.df['disfacpr']
        self.df['true_divisor_shr'] = 1.0 + self.df['disfacshr']
        
        # Ensure chronological order
        self.df.sort_values(['permno', 'date'], inplace=True)
        
        # Shift the divisor backward 1 day so it applies to the historical pre-split prices
        self.df['split_event_prc'] = self.df.groupby('permno')['true_divisor_prc'].shift(-1).fillna(1.0)
        self.df['split_event_shr'] = self.df.groupby('permno')['true_divisor_shr'].shift(-1).fillna(1.0)
        
        # Reverse cumprod to smoothly propagate the split backwards forever
        self.df['cfacpr'] = self.df.iloc[::-1].groupby('permno')['split_event_prc'].cumprod().iloc[::-1]
        self.df['cfacshr'] = self.df.iloc[::-1].groupby('permno')['split_event_shr'].cumprod().iloc[::-1]

        # 2. Calculate the true split-adjusted prices and shares
        self.df['prc_adj_usd'] = self.df['dlyprc'] / self.df['cfacpr']
        self.df['shrout_adj'] = self.df['shrout'] * self.df['cfacshr']
        
        # 3. Forward-Fill Missing Prices (Protects against trading halts)
        self.df['prc_adj_usd'] = self.df.groupby('permno')['prc_adj_usd'].ffill()
        
        self.df['mkt_cap_usd'] = self.df['prc_adj_usd'] * self.df['shrout_adj']
        
        # 4. Dividends must ALSO be divided by the cumulative factor to reflect historical basis!
        self.df['disdivamt'] = self.df['disdivamt'].fillna(0.0)
        self.df['divamt_net_usd'] = (self.df['disdivamt'] / self.df['cfacpr']) * (1 - US_WITHHOLDING_TAX_RATE)

    def filter_sp500_universe(self):
        """
        Filters the massive CRSP universe down to strictly S&P 500 constituents 
        using a highly memory-efficient asof merge.
        """
        print(f"Loading S&P 500 historical constituents from {SP500_INDEX_PATH.name}...")
        df_sp500 = pd.read_stata(str(SP500_INDEX_PATH))
        df_sp500.columns = df_sp500.columns.str.lower()
        
        # drop redundant columns 
        df_sp500 = df_sp500[['permno', 'mbrstartdt', 'mbrenddt']].copy()
        
        # Clean up the Stata float parsing quirk
        df_sp500 = df_sp500.dropna(subset=['permno'])
        df_sp500['permno'] = df_sp500['permno'].astype('int32')
        
        # Ensure dates are properly formatted
        df_sp500['mbrstartdt'] = pd.to_datetime(df_sp500['mbrstartdt'])
        df_sp500['mbrenddt'] = pd.to_datetime(df_sp500['mbrenddt'])
        
        print("Sorting dataframes for  merge_asof...")
        # merge_asof strictly requires both dataframes to be sorted by the merge keys
        self.df.sort_values('date', inplace=True)
        df_sp500.sort_values('mbrstartdt', inplace=True)
        
        print("Executing asof merge to align index membership...")
        merged = pd.merge_asof(
            self.df, 
            df_sp500,
            left_on='date',
            right_on='mbrstartdt',
            by='permno',
            direction='backward'
        )
        
        print("Enforcing strict membership boundaries...")
        # 1. Drop stocks that were NEVER in the S&P 500 (mbrstartdt will be NaT)
        merged = merged.dropna(subset=['mbrstartdt'])
        
        # 2. Drop rows where the daily date is AFTER the stock was kicked out
        merged = merged[merged['date'] <= merged['mbrenddt']]
        
        # Clean up the dataframe
        merged.drop(columns=['mbrstartdt', 'mbrenddt'], inplace=True)
        
        self.df = merged.reset_index(drop=True)
        print(f"Universe strictly filtered. Total daily observations remaining: {len(self.df)}")

    def integrate_cad_fx(self):
        """Fetches daily USD/CAD spot rate from FRED to translate the ledger to CAD."""
        print("Fetching USD/CAD FX rates from FRED (Federal Reserve Economic Data)...")
        
        min_date = self.df['date'].min().strftime('%Y-%m-%d')
        max_date = self.df['date'].max().strftime('%Y-%m-%d')
        
        # DEXCAUS is the FRED ticker for CAD per 1 USD daily spot rate
        fx_data = web.DataReader('DEXCAUS', 'fred', min_date, max_date)
        fx_data = fx_data.reset_index()
        fx_data.columns = ['date', 'usd_cad']
        
        # FRED leaves US banking holidays as NaN. We forward-fill them to ensure 
        # we have an FX rate for every single stock trading day.
        fx_data.set_index('date', inplace=True)
        fx_data = fx_data.reindex(pd.date_range(min_date, max_date)).ffill().reset_index()
        fx_data.rename(columns={'index': 'date'}, inplace=True)
        
        print("Merging FX rates and calculating CAD dual-ledger basis...")
        self.df = pd.merge(self.df, fx_data, on='date', how='left')
        
        # Backfill any missing FX rates at the very start just in case the 
        # fx rate fot the first day in the data is missing
        self.df['usd_cad'] = self.df['usd_cad'].bfill()
        
        # Dual-Ledger Translation: Create the CAD prices required by the CRA
        self.df['prc_adj_cad'] = np.where(
            self.df['currency'] == 'USD', 
            self.df['prc_adj_usd'] * self.df['usd_cad'], 
            self.df['prc_adj_usd']
        )
        #3. Applying the same conditional FX logic to the Dividends
        self.df['divamt_net_cad'] = np.where(
            self.df['currency'] == 'USD',
            self.df['divamt_net_usd'] * self.df['usd_cad'],  # Convert to CAD
            self.df['divamt_net_usd']                        # Already CAD
        )
    

    def save_output(self):
        """Saves the final optimized universe to Parquet."""
        print(f"Saving finalized pipeline data to: {self.out_path}")
        
        # Drop raw unadjusted columns to save space, keeping only what the optimizer/ledger needs
        columns_to_drop = ['dlyprc', 'shrout', 'disfacpr', 'disfacshr', 'disdivamt']
        self.df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Sort for fast timeseries querying later
        self.df.sort_values(['permno', 'date'], inplace=True)
        self.df.to_parquet(self.out_path, index=False)
        print("Data Pipeline Complete! Ready for Risk Modeling and Tax Ledger.")

    
    # THE ORCHESTRATOR 
    def run_data_pipeline(self):
        if self.out_path.exists():
            print(f'The TLH universe file already exists at {self.out_path}. Skipping pipeline.')
            return 
        
        """Executes the full end-to-end data engineering workflow."""
        print("Starting Data Pipeline Orchestration...")
        
        self.load_and_merge_daily_market_data()
        self.process_prices_and_dividends()
        self.filter_sp500_universe() 
        self.integrate_cad_fx()
        self.save_output()


if __name__ == "__main__":
    # 1. Ensure directories exist
    init_directories()
    
    # 2. Instantiate the class
    pipeline = DataPipeline()
    
    # 3. Call your new master orchestrator method
    pipeline.run_data_pipeline()