# Sentiment Analysis Pipeline

This directory contains a comprehensive sentiment analysis pipeline that collects financial news articles from GDELT, extracts text, analyzes sentiment using FinBERT, and performs rigorous statistical analysis to correlate sentiment with asset prices.

## Structure

- **`collect_sentiment_data.py`**: Python script that collects data from GDELT, extracts text, and analyzes sentiment. Outputs a parquet file.
- **`SentimentAnalysis.ipynb`**: Jupyter notebook for analyzing the collected sentiment data with statistical methods.

## Usage

### Step 1: Collect Data

Run the data collection script to generate the parquet file:

```bash
python collect_sentiment_data.py --query Bitcoin --years-back 2 --articles-per-week 10
```

**Command-line options:**
- `--query`: Search query (default: "Bitcoin")
- `--years-back`: Years of historical data to collect (default: 2)
- `--articles-per-week`: Number of articles to sample per week (default: 10)
- `--random-state`: Random seed for sampling (default: 42)
- `--output-dir`: Output directory for parquet file (default: current directory)

**Example:**
```bash
python collect_sentiment_data.py --query "Ethereum" --years-back 1 --articles-per-week 15
```

This will create a file named `gdelt_Ethereum_2yrs_finbert.parquet` (note: filename uses "2yrs" as a template, but respects your `--years-back` setting).

### Step 2: Analyze Data

Open `SentimentAnalysis.ipynb` in Jupyter and:

1. Update the `QUERY` variable in the configuration cell to match your data collection query
2. Run all cells to perform statistical analysis

The notebook includes:
- Basic visualization of sentiment over time
- Bootstrap confidence intervals
- Article length weighting
- Outlet dominance diagnostics
- Time-series smoothing
- Structural break detection
- Hypothesis testing
- Robustness checks
- Granger causality analysis
- Comprehensive diagnostics

## Dependencies

### For Data Collection (`collect_sentiment_data.py`):
```bash
pip install pandas numpy requests newspaper3k transformers torch tqdm
```

### For Analysis (`SentimentAnalysis.ipynb`):
```bash
pip install pandas numpy matplotlib yfinance scipy
pip install statsmodels ruptures  # Optional but recommended for advanced analysis
```

## Output

The data collection script produces:
- **Parquet file**: `gdelt_{QUERY}_2yrs_finbert.parquet` containing:
  - `url`: Article URL
  - `title`: Extracted article title
  - `published_raw`: Publication date
  - `sent_label`: Sentiment label (positive/neutral/negative)
  - `prob_positive`, `prob_neutral`, `prob_negative`: Sentiment probabilities
  - `n_pieces`: Number of text chunks processed

## Statistical Analysis Features

The analysis notebook includes research-grade statistical methods:

### Core Analysis
- **Bootstrap Confidence Intervals**: Quantify uncertainty in weekly sentiment estimates
- **Article Length Weighting**: Weight sentiment by article length (n_pieces) to account for information content
- **Outlet Dominance Diagnostics**: Identify and flag weeks with high single-outlet concentration (>40%)

### Time-Series Analysis
- **LOWESS Smoothing**: Non-parametric smoothing to identify trends
- **Rolling Mean**: Simple moving average for trend visualization
- **Structural Break Detection**: Identify regime changes in sentiment using PELT algorithm

### Statistical Testing
- **Hypothesis Testing**: Compare sentiment before/after key events (e.g., ETF approvals)
- **Granger Causality**: Test if sentiment Granger-causes price movements
- **Robustness Checks**: Compare multiple aggregation methods (mean, median, weighted, filtered)

### Diagnostics
- **Comprehensive Summary Table**: Key metrics including CI widths, extraction success rates, outlet concentration
- **Enhanced Visualizations**: Publication-ready plots with confidence intervals and breakpoints

## Key Concepts

### FinBERT
A BERT-based model fine-tuned on financial text (`yiyanghkust/finbert-tone`) that classifies sentiment into:
- **Positive**: Bullish, optimistic sentiment
- **Neutral**: Balanced, factual reporting
- **Negative**: Bearish, pessimistic sentiment

### GDELT
Global Database of Events, Language, and Tone - provides access to global news articles with metadata.

## Notes

- The GDELT API has rate limits (5 seconds between requests minimum)
- Article extraction may fail for some URLs (handled gracefully)
- The pipeline supports resume capability - if interrupted, it will continue from where it left off
- Processing time depends on number of articles and available compute (GPU recommended for FinBERT)
- Statistical analysis requires sufficient data (>20 weeks recommended for Granger causality)

## Troubleshooting

**Parquet file not found:**
- Make sure you've run `collect_sentiment_data.py` first
- Check that the `QUERY` variable in the notebook matches the query used in data collection

**GDELT API errors:**
- The script includes retry logic, but persistent failures may indicate API issues
- Check your internet connection
- GDELT may be temporarily unavailable

**Memory issues:**
- Reduce `--articles-per-week` to process fewer articles
- Process data in smaller date ranges

**Statistical analysis errors:**
- Ensure you have sufficient data (at least 10-20 weeks for meaningful analysis)
- Install optional packages: `pip install statsmodels ruptures` for advanced features
- Some analyses gracefully degrade if optional packages are missing

## Example Workflow

```bash
# 1. Collect data (this may take several hours)
python collect_sentiment_data.py --query Bitcoin --years-back 2 --articles-per-week 10

# 2. Open and run SentimentAnalysis.ipynb
#    - Update QUERY variable to match
#    - Run all cells for complete analysis
```

## Research Applications

This pipeline is suitable for:
- **Academic Research**: Sentiment analysis with statistical rigor
- **Trading Signals**: Correlation analysis between sentiment and price movements
- **Media Analysis**: Understanding how news coverage evolves over time
- **Policy Research**: Analyzing market sentiment around regulatory events

---

**Last Updated**: January 2025
