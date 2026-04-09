import os
from pathlib import Path

# ==========================================
# 1. CORE DIRECTORIES
# ==========================================
ONEDRIVE_ROOT = Path(os.environ.get('OneDrive', ''))
DATA_ROOT = ONEDRIVE_ROOT / '0. DATASETS'
RAW_DATA_DIR = DATA_ROOT / 'raw'
OUTPUT_DIR = DATA_ROOT / 'outputs'
PROJECT_ROOT = OUTPUT_DIR / 'Direct-Investing-TLH'

# ==========================================
# 2. PIPELINE SUBDIRECTORIES
# ==========================================
# Creating the dedicated output folders for each stage of the new engine
DATA_CLEAN_DIR = PROJECT_ROOT / '01_Data'
RISK_DIR = PROJECT_ROOT / '02_Risk'
LEDGER_DIR = PROJECT_ROOT / '03_TaxLedger'
PORTFOLIO_DIR = PROJECT_ROOT / '04_Portfolio'
BACKTEST_DIR = PROJECT_ROOT / '05_Backtest'

# Group them in a list for the init function
PIPELINE_DIRS = [
    DATA_CLEAN_DIR, 
    RISK_DIR, 
    LEDGER_DIR, 
    PORTFOLIO_DIR, 
    BACKTEST_DIR
]


# ==========================================
# 3. RAW DATA FILE PATHS
# ==========================================
SP500_INDEX_PATH = RAW_DATA_DIR / "sp500.dta"
DAILY_PRICES_PATH = RAW_DATA_DIR / "daily_stocks.csv.gz"
DIVIDENDS_PATH = RAW_DATA_DIR / "div.dta"
DELISTING_PATH = RAW_DATA_DIR / "delisting.csv"

# ==========================================
# 4. GLOBAL SYSTEM PARAMETERS
# ==========================================
# ASSUMPTION: For this V1 prototype, we assume the entire S&P 500 universe is 
# US-domiciled, applying a blanket 15% US-to-CAN withholding tax on dividends.
# Future iterations would integrate a mapping table of ISIN/HQ country codes to 
# handle varying tax treaty rates for grandfathered ex-US constituents (e.g. Ireland, UK).
US_WITHHOLDING_TAX_RATE = 0.15  
SUPERFICIAL_LOSS_DAYS = 31      
TRANSACTION_COST_BPS = 5       # assuming a tight transaction costs for Wealthsimple

def init_directories():
    """
    Iterates through the pipeline directories and ensures the entire 
    project tree exists in OneDrive before data processing begins.
    """
    for directory in PIPELINE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        
    print(f"System Check: All {len(PIPELINE_DIRS)} pipeline directories verified in {PROJECT_ROOT.name}")

if __name__ == "__main__":
    init_directories()