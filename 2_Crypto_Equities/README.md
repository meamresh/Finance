# Market Risk Analysis

This directory contains quantitative finance projects focused on market risk measurement, specifically Value at Risk (VaR) calculations for cryptocurrency and equity markets.

## Project

### Value at Risk (VaR) - Multiple Methodologies (`MarketRisk.ipynb`)

**Objective**: Implement and compare three different VaR calculation methods for Bitcoin (BTC-USD).

**Overview**:
Value at Risk (VaR) is a statistical metric that estimates the potential maximum loss of an investment at a given confidence level over a specific time period. This notebook demonstrates three complementary approaches to calculating VaR, each with different assumptions and use cases.

---

## Methodologies

### 1. Historical VaR

**Approach**: Non-parametric method using empirical quantiles from historical returns.

**How it works**:
- Uses actual historical return distribution
- No distributional assumptions
- Directly reads the quantile from sorted historical returns

**Advantages**:
- No parametric assumptions
- Captures actual tail behavior
- Simple to implement

**Limitations**:
- Assumes past patterns will continue
- Requires sufficient historical data
- May not capture extreme events not in the sample

**Results for BTC-USD (2014-2022)**:
- 90% VaR: -3.717%
- 95% VaR: -6.0%
- 99% VaR: -10.57%

---

### 2. Parametric VaR

**Approach**: Assumes returns follow a specific probability distribution.

#### 2a. Normal Distribution

**How it works**:
- Assumes returns are normally distributed
- Uses mean and standard deviation of historical returns
- Applies inverse CDF (percent point function) to get VaR

**Advantages**:
- Simple and computationally efficient
- Only requires mean and variance

**Limitations**:
- Financial returns often have "fat tails" (more extreme events than normal distribution)
- Underestimates tail risk

**Results for BTC-USD**:
- 90% VaR: -4.716%
- 95% VaR: -6.108%
- 99% VaR: -8.72%

#### 2b. Student's t-Distribution

**How it works**:
- Assumes returns follow Student's t-distribution
- Better captures fat tails than normal distribution
- Requires specification of degrees of freedom (controls tail thickness)

**Advantages**:
- More realistic for financial returns
- Better captures tail risk than normal distribution
- Flexible tail behavior

**Limitations**:
- Requires choosing appropriate degrees of freedom
- Still a parametric assumption

**Results for BTC-USD (5 degrees of freedom)**:
- 90% VaR: -4.186%
- 95% VaR: -5.786%
- 99% VaR: -9.793%

---

### 3. Monte Carlo VaR

**Approach**: Simulation-based method using distributional assumptions.

**How it works**:
1. Specify distribution parameters (mean, volatility)
2. Generate large number of random return scenarios (e.g., 10,000 simulations)
3. Calculate portfolio values for each scenario
4. Find the quantile corresponding to the confidence level

**Advantages**:
- Flexible - can use any distribution
- Can incorporate complex portfolio structures
- Useful when historical data is limited

**Limitations**:
- Requires distributional assumptions
- Computationally intensive
- Quality depends on parameter estimation

**Example Results** (for $100K portfolio with 0.05% daily mean, 2% daily volatility):
- 95% VaR: -$3,228.90 (-3.23%)

---

## Comparison of Methods

| Method | 95% VaR | Strengths | Weaknesses |
|--------|---------|-----------|------------|
| Historical | -6.0% | No assumptions, captures actual tails | Past may not predict future |
| Parametric (Normal) | -6.108% | Simple, fast | Underestimates tail risk |
| Parametric (t-dist) | -5.786% | Better tail modeling | Requires parameter choice |
| Monte Carlo | -3.23%* | Flexible, handles complexity | Assumption-dependent |

*Monte Carlo result uses different parameters than historical analysis

---

## Key Insights

1. **Fat Tails**: Bitcoin returns show significant tail risk, with 99% VaR around -10% using historical method
2. **Distribution Choice Matters**: Normal distribution underestimates extreme risk compared to t-distribution
3. **Method Selection**: Choose method based on:
   - Data availability (historical vs. limited data)
   - Portfolio complexity (simple vs. complex derivatives)
   - Computational resources
   - Regulatory requirements

---

## Technologies

- `pandas`, `numpy` for data manipulation and calculations
- `yfinance` for Bitcoin price data
- `scipy.stats` for statistical distributions
- `matplotlib`, `seaborn` for visualization

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
pip install yfinance
```

## Data

- **Asset**: Bitcoin (BTC-USD)
- **Source**: Yahoo Finance via `yfinance`
- **Period**: September 2014 - December 2022
- **Frequency**: Daily returns

## Usage

1. Run all cells in `MarketRisk.ipynb`
2. The notebook will:
   - Download BTC-USD price data
   - Calculate daily returns
   - Compute VaR using all three methods
   - Display comparative results

## Interpretation

**Example**: A 95% VaR of -6.0% means:
- There is a 5% probability that Bitcoin will lose more than 6% in a single day
- Or equivalently: 95% of the time, daily losses will not exceed 6%

**Important Notes**:
- VaR is a statistical estimate, not a guarantee
- Past performance does not guarantee future results
- VaR does not predict the magnitude of losses beyond the confidence level
- For risk management, consider complementary metrics (Expected Shortfall, stress testing)

---

## Extensions

Potential enhancements to this analysis:
- **Expected Shortfall (CVaR)**: Average loss beyond VaR threshold
- **Conditional VaR**: VaR that adapts to market conditions
- **Backtesting**: Validate VaR models against actual outcomes
- **Portfolio VaR**: Extend to multi-asset portfolios
- **Regulatory VaR**: Implement Basel III or other regulatory frameworks

---

**Last Updated**: January 2025
