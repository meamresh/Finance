# Finance & Quantitative Analysis Portfolio

A comprehensive collection of quantitative finance projects demonstrating expertise in risk management, fixed income analysis, and alternative data integration using Python, statistical modeling, and machine learning.

## 📊 Overview

This repository showcases practical applications of quantitative finance techniques across three key domains:
- **Fixed Income Analysis**: Yield curve modeling and portfolio risk assessment
- **Market Risk Management**: Value at Risk (VaR) calculations using multiple methodologies
- **Alternative Data**: Sentiment analysis of financial news and correlation with asset prices

## 🎯 Projects

### 1. Fixed Income Analysis (`1_FixedIncome/`)

#### Nelson-Siegel Yield Curve Model
- **Objective**: Model and analyze U.S. Treasury yield curves using the Nelson-Siegel framework
- **Key Features**:
  - Historical yield data retrieval from FRED API (1975-2024)
  - Yield curve visualization and analysis
  - Identification of bear/bull flattening and steepening periods
  - Parameter estimation for Nelson-Siegel model (β₀, β₁, β₂, τ)
  - Factor decomposition of yield curves (level, slope, curvature)

**Technologies**: `pandas`, `numpy`, `matplotlib`, `fredapi`, `nelson-siegel-svensson`

#### Value at Risk (VaR) for Bond Portfolios
- **Objective**: Calculate portfolio-level VaR for a Treasury bond portfolio using Principal Component Analysis (PCA)
- **Key Features**:
  - Multi-bond portfolio risk aggregation ($5M portfolio: 2Y, 5Y, 10Y bonds)
  - PCA dimensionality reduction (97% variance explained with 2 components)
  - 1-day 95% VaR calculation
  - Portfolio sensitivity analysis and risk attribution
  - Visualization of risk contributions by bond maturity

**Technologies**: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

**Results**: Calculated 1-day 95% VaR of $291,342 (5.83% of portfolio value)

---

### 2. Market Risk Analysis (`2_Crypto_Equities/`)

#### Value at Risk (VaR) - Multiple Methodologies
- **Objective**: Implement and compare three different VaR calculation methods for Bitcoin
- **Methodologies**:
  1. **Historical VaR**: Non-parametric approach using empirical quantiles
  2. **Parametric VaR**: 
     - Normal distribution assumption
     - Student's t-distribution (better fit for heavy-tailed returns)
  3. **Monte Carlo VaR**: Simulation-based approach with distributional assumptions

**Key Features**:
- Analysis of BTC-USD returns (2014-2022)
- Comparison of VaR estimates across confidence levels (90%, 95%, 99%)
- Distribution fitting and tail risk assessment
- Monte Carlo simulation with 10,000 iterations

**Technologies**: `pandas`, `numpy`, `scipy.stats`, `yfinance`, `matplotlib`, `seaborn`

**Results**: 
- Historical 95% VaR: -6.0%
- Parametric (Normal) 95% VaR: -6.108%
- Parametric (t-dist) 95% VaR: -5.786%
- Monte Carlo 95% VaR: -3.23% (for $100K portfolio)

---

### 3. Sentiment Analysis (`3_SentimentAnalysis/`)

#### Financial News Sentiment & Price Correlation
- **Objective**: Analyze sentiment of Bitcoin-related news articles and correlate with price movements
- **Key Features**:
  - **Data Collection**: 
    - GDELT API integration for news article retrieval
    - 2-year historical data collection (6,250+ articles)
    - Article text extraction using `newspaper3k`
    - Intelligent sampling (10 articles per week) for balanced analysis
  
  - **Sentiment Analysis**:
    - FinBERT model (`yiyanghkust/finbert-tone`) for financial sentiment classification
    - Batch processing with GPU acceleration support
    - Text chunking for long articles (1,000 char chunks)
    - Probability-weighted sentiment scores (positive, neutral, negative)
  
  - **Analysis & Visualization**:
    - Weekly sentiment aggregation
    - Correlation with BTC-USD weekly price changes
    - Time series visualization of sentiment trends vs. price movements

**Technologies**: 
- `transformers` (Hugging Face), `torch` (PyTorch)
- `pandas`, `numpy`, `matplotlib`
- `newspaper3k`, `requests`
- `yfinance` (for price data)

**Pipeline Features**:
- Resume capability for interrupted runs
- Error handling and retry logic
- Efficient batch processing
- Data persistence in Parquet format

---

## 🛠️ Technical Skills Demonstrated

### Programming & Data Science
- **Python**: Advanced proficiency in data manipulation, statistical analysis, and API integration
- **Data Processing**: `pandas`, `numpy` for time series and financial data
- **Visualization**: `matplotlib`, `seaborn` for professional financial charts

### Quantitative Finance
- **Risk Metrics**: Value at Risk (VaR) calculations
- **Fixed Income**: Yield curve modeling, duration analysis, portfolio risk
- **Statistical Methods**: PCA, distribution fitting, Monte Carlo simulation
- **Time Series Analysis**: Resampling, aggregation, correlation analysis

### Machine Learning & NLP
- **Transformer Models**: FinBERT for financial sentiment analysis
- **Deep Learning**: PyTorch for model inference
- **NLP**: Text extraction, chunking, batch processing

### APIs & Data Sources
- **FRED API**: Federal Reserve Economic Data for Treasury yields
- **GDELT API**: Global news event data
- **yfinance**: Financial market data
- **Web Scraping**: Article text extraction

### Software Engineering
- **Error Handling**: Robust retry logic and exception management
- **Data Persistence**: Parquet format for efficient storage
- **Code Organization**: Modular functions, configuration management
- **Logging**: Comprehensive logging for debugging and monitoring

---

## 📈 Key Achievements

- ✅ Implemented multiple VaR methodologies with comparative analysis
- ✅ Built end-to-end sentiment analysis pipeline processing 6,250+ articles
- ✅ Applied PCA for dimensionality reduction in portfolio risk analysis
- ✅ Integrated multiple financial data APIs (FRED, GDELT, yfinance)
- ✅ Deployed transformer models (FinBERT) for financial NLP tasks
- ✅ Created professional visualizations for risk and sentiment analysis

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy
pip install fredapi yfinance
pip install transformers torch
pip install newspaper3k requests
pip install nelson-siegel-svensson
```

### API Keys Required
- **FRED API**: Get your free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
- **GDELT API**: Public API (rate-limited, 5-second minimum between requests)

### Running the Projects
1. **Fixed Income Analysis**: Open `1_FixedIncome/` notebooks and update FRED API key
2. **Market Risk**: Open `2_Crypto_Equities/MarketRisk.ipynb` (no API key needed)
3. **Sentiment Analysis**: Open `3_SentimentAnalysis/SentimentAnalysis.ipynb` and run cells sequentially

---

## 📝 Notes

- All notebooks include detailed explanations and interpretations of results
- API keys are redacted in the code for security
- Some notebooks may take time to run due to API rate limits and data processing
- The sentiment analysis project includes resume capability for interrupted runs

---

## 📧 Contact

For questions or collaboration opportunities, please reach out through GitHub.

---

**Last Updated**: January 2025
