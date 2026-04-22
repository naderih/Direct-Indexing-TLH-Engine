# Systematic Direct Indexing & Tax-Loss Harvesting Engine (Canadian Framework)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Optimization: CVXPY](https://img.shields.io/badge/Optimization-CVXPY-orange)](https://www.cvxpy.org/)
[![Risk Model: Fundamental Factor](https://img.shields.io/badge/Risk_Model-Fundamental_Factor-success)](#)
[![Math: Marchenko-Pastur](https://img.shields.io/badge/Denoising-Marchenko_Pastur-blueviolet)](#)

## Overview
This repository contains an end-to-end Python risk engine and Tax-Loss Harvesting (TLH) backtesting framework. It is engineered to track a US equity benchmark (the S&P 500) while actively harvesting capital losses to generate tax alpha. 

Crucially, this system bypasses standard US-centric TLH assumptions (like HIFO lot-picking) and is structurally designed around **Canada Revenue Agency (CRA)** mechanics for non-registered accounts, modeling the exact frictions encountered in Canadian retail direct indexing.

## Core Canadian Tax Mechanics Modeled
1. **Average Cost Base (ACB) Pooling:** Rather than picking specific tax lots, the $O(1)$ ledger mathematically pools the cost of identical properties upon every purchase, accurately reflecting Canadian tax reporting reality.
2. **Superficial Loss Rule (The 30-Day Rule):** The ledger enforces strict time-based CRA lockouts. It dynamically feeds constraint vectors to the optimizer to prevent buying a stock 30 days *after* a harvested loss (Forward Rule), and blocks harvesting if the stock was purchased in the 30 days *prior* (Backward Rule).
3. **Dual-Ledger FX:** Assets are priced in USD, but the tax ledger calculates all ACB and capital gains in CAD using daily spot rates, correctly modeling the impact of currency fluctuations on tax liabilities.

## System Architecture

<p align="center">
  <img src="assets/tlh_architecture.png" alt="System Architecture Diagram" width="850"/>
</p>

The codebase is heavily modularized to decouple data processing, structural risk modeling, the accounting state machine, and portfolio optimization:

### Phase A: Quant Infrastructure & Advanced Risk Modeling
* **`a1_data_pipeline.py`**: Ingests point-in-time CRSP daily data, applies split adjustments, merges FRED USD/CAD spot rates, and standardizes the universe.
* **`a2_factor_engine.py`**: Constructs the $X$ matrix of fundamental factor exposures, utilizing universe-wide cap-weighted standardization and intelligent missing-data imputation to ensure 100% cross-sectional coverage.
* **`a3_risk_estimator.py`**: A robust Fundamental Factor Risk Model. It extracts pure factor returns via WLS Fama-MacBeth regressions and forecasts covariance using an Exponentially Weighted Moving Average (EWMA). 
    * **Eigenvalue Denoising:** Before the covariance matrix ($F$) is finalized, it undergoes **Marchenko-Pastur Eigenvalue Clipping**. The engine dynamically maps the EWMA half-life to an asymptotic Effective Sample Size ($T_{eff}$), utilizes iterative variance estimation to isolate structural signal, and replaces hallucinatory correlations with a constant residual eigenvalue.
    * **Robust Specific Risk:** Idiosyncratic risk ($\Delta$) is forecasted using absolute mean deviations (MAD) rather than squared residuals to prevent fat-tail earnings shocks from crashing the matrix inverse via the Woodbury identity.

### Phase B: The Tax Ledger & State Machine
* **`b1_risk_model.py`**: Dynamically reconstructs the clean $V$ matrix ($V = X F X^T + \Delta$) aligned precisely to the active daily roster.
* **`b2_tax_ledger.py`**: An event-driven accounting ledger. It tracks cash balances, updates the pooled ACB, registers realized PnL, and passes absolute lockout restrictions directly to the optimization layer.

### Phase C: Convex Optimization
* **`c2_tax_alpha_optimizer.py`**: A continuous convex optimizer powered by CVXPY. It reframes the objective to maximize the opportunity cost of unrealized losses minus an $L_1$ turnover penalty, subject to a strict $L_2$ Tracking Error budget. This formulation organically discovers **"Partial Harvesting"**—scaling sell orders dynamically to balance tax yield against risk limits.
* **`c3_backtest_runner.py`**: The main orchestrator that steps through time, triggers the harvest scanner, and translates optimal weights into discrete fractional shares (truncated to 4 decimal places to prevent cash overdrafts).

## Mathematical Formulation
The core portfolio construction relies on quadratic programming to balance tracking error against transaction friction, utilizing the rigorously denoised covariance matrix ($V_{clean}$):

$$\min \left( h^T \cdot C_{tax} + \lambda || h - h_{\text{current}} ||_1 \right)$$

**Subject to:**
* $(h - h_b)^T V_{clean} (h - h_b) \le \text{Max TEV}$ *(Relative Risk Budgeting)*
* $\sum h = 1.0, \quad h \ge 0$ *(Fully Invested, Long Only)*
* $h_i \le h_{\text{current}, i}$ for $i \in \text{Forward Lockouts}$
* $h_i \ge h_{\text{current}, i}$ for $i \in \text{Backward Lockouts}$

## Production Scaling & Next Steps
While this architecture utilizes CVXPY for continuous convex optimization to demonstrate the core logic, elevating this engine to a live retail environment requires specific upgrades:
* **MIQP Integration:** Transitioning the solver to a Mixed-Integer Quadratic Program (e.g., Gurobi) to mathematically enforce cardinality constraints, integer lot sizes, and minimum trade thresholds, replacing the current post-optimization translation filters.
* **Expanded Execution Universe:** Decoupling the daily pricing universe (e.g., Russell 3000) from the target benchmark (S&P 500) to allow the optimizer to seamlessly handle index dropouts and hunt for out-of-universe tax proxies.

## Quick Start
The project logic and stress-testing can be viewed in the included Jupyter Notebooks:
1. **`dlh_functional.ipynb`**: A functional walk-through of the accounting logic, highlighting the transition from procedural loops to the final Object-Oriented architecture, including a hyperparameter grid search.
2. **`main.ipynb`**: Executes the final OOP pipeline over the 2019-2020 sub-period, demonstrating the algorithm's ability to autonomously harvest losses and maintain benchmark tracking during the March 2020 market crash.

---
*Disclaimer: This repository is a prototype built for quantitative research and architectural demonstration purposes. It does not constitute financial or tax advice.*