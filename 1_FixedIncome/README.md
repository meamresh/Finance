# Fixed Income Analysis

This directory contains quantitative finance projects focused on fixed income securities, yield curve modeling, and portfolio risk management.

## Projects

### 1. Nelson-Siegel Yield Curve Model (`Nelson_Siegel_Model.ipynb`)

**Objective**: Model and analyze U.S. Treasury yield curves using the Nelson-Siegel framework.

**Key Features**:
- Historical yield data retrieval from FRED API (1975-2024)
- Comprehensive yield curve visualization and analysis
- Identification of yield curve dynamics:
  - **Bear/Bull Flattening**: Periods when the yield curve flattens (short rates rise/fall relative to long rates)
  - **Bear/Bull Steepening**: Periods when the yield curve steepens (short rates fall/rise relative to long rates)
- Nelson-Siegel parameter estimation (β₀, β₁, β₂, τ)
- Factor decomposition of yield curves:
  - **β₀**: Level factor (long-term yield)
  - **β₁**: Slope factor (short-term vs long-term)
  - **β₂**: Curvature factor (medium-term shape)

**Technologies**:
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `fredapi` for Federal Reserve Economic Data
- `nelson-siegel-svensson` for yield curve modeling

**Usage**:
1. Get a free FRED API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Update the API key in the notebook: `fred = Fred(api_key='YOUR_KEY')`
3. Run all cells to generate yield curve analysis

**Key Insights**:
- Historical analysis shows yield curve evolution from 1975-2024
- Demonstrates how yield curves respond to monetary policy and economic conditions
- Provides framework for understanding term structure of interest rates

---

### 2. Value at Risk (VaR) for Bond Portfolios (`ValueAtRisk.ipynb`)

**Objective**: Calculate portfolio-level Value at Risk for a Treasury bond portfolio using Principal Component Analysis (PCA).

**Key Features**:
- Multi-bond portfolio risk aggregation
  - Portfolio composition: $2M in 2-year, $2M in 5-year, $1M in 10-year bonds
  - Total portfolio value: $5 million
- PCA dimensionality reduction
  - Analyzes yield changes across multiple maturities
  - Reduces dimensionality while preserving 97% of variance with 2 components
- 1-day 95% VaR calculation
- Portfolio sensitivity analysis
- Risk attribution by bond maturity
- Visualization of risk contributions

**Methodology**:
1. **Data Collection**: Treasury yield data for 2Y, 5Y, 10Y maturities
2. **Standardization**: Normalize yield changes for PCA
3. **PCA Analysis**: Extract principal components explaining yield curve movements
4. **Portfolio Mapping**: Map portfolio positions to principal components
5. **VaR Calculation**: Compute Value at Risk using historical simulation

**Results**:
- **1-day 95% VaR**: $291,342
- **VaR as % of Portfolio**: 5.83%
- **Variance Explained**: 97% with first 2 principal components

**Technologies**:
- `pandas`, `numpy` for data processing
- `scipy` for statistical functions
- `matplotlib`, `seaborn` for visualization
- `fredapi` for yield data

**Usage**:
1. Get a FRED API key (see above)
2. Update the API key in the notebook
3. Run all cells to compute portfolio VaR

**Key Insights**:
- Demonstrates how PCA can reduce complexity in multi-asset portfolio risk
- Shows risk aggregation across different bond maturities
- Provides practical framework for fixed income risk management

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
pip install fredapi
pip install nelson-siegel-svensson
```

## Data Sources

- **FRED (Federal Reserve Economic Data)**: Treasury yield data
  - Series IDs: DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30
  - Historical data from 1975 to present

## Key Concepts

### Value at Risk (VaR)
A statistical measure of the potential maximum loss of an investment or portfolio at a given confidence level over a specific time period.

### Nelson-Siegel Model
A parametric model for the yield curve that decomposes it into three factors:
- **Level** (β₀): Long-term yield
- **Slope** (β₁): Difference between short and long-term yields
- **Curvature** (β₂): Medium-term shape of the curve

### Principal Component Analysis (PCA)
A dimensionality reduction technique that identifies the main sources of variation in yield curve movements, allowing for efficient risk modeling.

## Notes

- API keys are redacted in the notebooks for security
- Historical data may have missing values for certain maturities (especially short-term rates before 2000s)
- VaR calculations assume linear portfolio sensitivity to yield changes
- Results are for educational/demonstration purposes

---

**Last Updated**: January 2025
