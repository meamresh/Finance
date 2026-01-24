#!/usr/bin/env python3
"""
Sentiment Data Collection Pipeline

This script collects news articles from GDELT, extracts text, and analyzes sentiment
using FinBERT. The results are saved to a parquet file for analysis.

Usage:
    python collect_sentiment_data.py [--query QUERY] [--years-back YEARS] [--articles-per-week N]
"""

import os
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict
import torch.nn.functional as F
import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from newspaper import Article, Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default configuration (can be overridden via command line)
DEFAULT_QUERY = "Bitcoin"
DEFAULT_YEARS_BACK = 2
DEFAULT_ARTICLES_PER_WEEK = 10
DEFAULT_RANDOM_STATE = 42

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_PER_CALL = 250
PAUSE_BETWEEN_CALLS = 5  # GDELT requires 5 seconds minimum
PAUSE_BETWEEN_DOWNLOADS = 0.5
MAX_ARTICLE_CHUNKS = 6
CHUNK_CHARS = 1000

# ============================================================================
# GDELT QUERYING FUNCTIONS
# ============================================================================

def gdelt_datefmt(dt: datetime) -> str:
    """Convert datetime to GDELT format."""
    return dt.strftime('%Y%m%d%H%M%S')


def query_gdelt(query: str, start_dt: datetime, end_dt: datetime, maxrecords: int = MAX_PER_CALL):
    """Query GDELT API for articles."""
    params = {
        'query': query,
        'mode': 'artlist',
        'maxrecords': maxrecords,
        'format': 'json',
        'startdatetime': gdelt_datefmt(start_dt),
        'enddatetime': gdelt_datefmt(end_dt),
        'sort': 'datedesc'
    }
    try:
        resp = requests.get(GDELT_BASE, params=params, timeout=60)
        if resp.status_code != 200:
            logging.warning('GDELT query failed: %s -> %s', resp.status_code, resp.text[:200])
            return None
        return resp.json()
    except requests.exceptions.RequestException as e:
        logging.error(f'GDELT request exception: {e}')
        return None


def query_gdelt_with_retry(query: str, start_dt: datetime, end_dt: datetime,
                           maxrecords: int = MAX_PER_CALL, max_retries: int = 3):
    """Query GDELT with retry logic."""
    for attempt in range(max_retries):
        result = query_gdelt(query, start_dt, end_dt, maxrecords)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5
            logging.warning(f'Retrying GDELT query in {wait_time}s (attempt {attempt+1}/{max_retries})')
            time.sleep(wait_time)
    logging.error(f'GDELT query failed after {max_retries} attempts')
    return None


def iterate_month_windows(start: datetime, end: datetime):
    """Generate month-by-month windows between start and end dates."""
    cur = datetime(start.year, start.month, 1)
    while cur < end:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1)
        yield cur, min(nxt - timedelta(seconds=1), end)
        cur = nxt


def fetch_all_gdelt_urls(query: str, start: datetime, end: datetime, out_meta_file: str):
    """Query GDELT month-by-month and write metadata as JSONL for resume capability."""
    seen_urls = set()

    # Resume if meta file exists
    if os.path.exists(out_meta_file):
        logging.info("Resuming: loading existing metadata to avoid re-fetching")
        with open(out_meta_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    seen_urls.add(o.get("url"))
                except (json.JSONDecodeError, KeyError) as e:
                    logging.debug(f"Skipping malformed line: {e}")
                    continue

    with open(out_meta_file, "a", encoding="utf-8") as out:
        for wstart, wend in iterate_month_windows(start, end):
            logging.info("Querying GDELT for %s -> %s", wstart.date(), wend.date())
            resp = query_gdelt_with_retry(query, wstart, wend)
            time.sleep(PAUSE_BETWEEN_CALLS)

            if not resp:
                continue

            articles = resp.get("articles") or resp.get("artlist") or []
            logging.info("Found %d articles in this window", len(articles))

            for a in articles:
                url = a.get("url") or a.get("sourceurl") or a.get("documentidentifier")
                title = a.get("title") or a.get("urltitle") or ""
                seendate = a.get("seendate") or a.get("date") or a.get("published")

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                record = {"url": url, "title": title, "seendate": seendate}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("GDELT URL fetching complete. Total unique URLs: %d", len(seen_urls))
    return out_meta_file


def load_urls_from_meta(meta_file: str):
    """Load article metadata from JSONL file."""
    rows = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError) as e:
                logging.debug(f"Skipping invalid JSON line: {e}")
                continue
    return rows

# ============================================================================
# TEXT EXTRACTION FUNCTIONS
# ============================================================================

def extract_text_with_newspaper(url: str, fallback_text: str = ""):
    """Extract article text using newspaper3k with proper timeout configuration."""
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        config.request_timeout = 15
        config.number_threads = 1

        art = Article(url, config=config, language='en')
        art.download()
        art.parse()
        return art.title or None, art.text or None, (art.publish_date if hasattr(art, "publish_date") else None)
    except Exception as e:
        logging.debug(f"Article extraction failed for {url}: {type(e).__name__}")
        return None, fallback_text or None, None


def chunk_text(text: str, max_chars=CHUNK_CHARS):
    """Split text into chunks based on character limit."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = text.split('. ')
    chunks = []
    cur, cur_len = [], 0

    for s in sentences:
        s_len = len(s) + 2
        if cur_len + s_len > max_chars and cur:
            chunks.append('. '.join(cur).strip() + ('.' if not cur[-1].endswith('.') else ''))
            cur = [s]
            cur_len = s_len
        else:
            cur.append(s)
            cur_len += s_len

    if cur:
        chunks.append('. '.join(cur).strip() + ('.' if not cur[-1].endswith('.') else ''))

    return chunks

# ============================================================================
# PARQUET UTILITIES
# ============================================================================

def safe_to_parquet(df, path, **kwargs):
    """Safely save to parquet with ArrowKeyError handling."""
    import warnings
    try:
        df.to_parquet(path, engine='pyarrow', **kwargs)
    except Exception as e:
        if "pandas.period" in str(e) or "ArrowKeyError" in str(e):
            logging.warning("Using fastparquet due to period type conflict")
            try:
                df.to_parquet(path, engine='fastparquet', **kwargs)
            except ImportError:
                logging.error("fastparquet not installed. Run: pip install fastparquet")
                raise
        else:
            raise


def sanitize_df_for_parquet(df, convert_period_to='timestamp'):
    """Enhanced version with warning suppression and Period detection."""
    import warnings
    df = df.copy()

    # Handle PeriodIndex
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp() if convert_period_to == 'timestamp' else df.index.astype(str)

    for col in df.columns:
        col_dtype = df[col].dtype
        col_lower = col.lower()

        # Date/time columns with warning suppression
        if any(token in col_lower for token in ("date", "time", "published")):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Could not infer format')
                    parsed = pd.to_datetime(df[col], errors='coerce', utc=True)
                if parsed.notna().any():
                    df[col] = parsed
                else:
                    df[col] = df[col].astype(str)
                continue
            except:
                df[col] = df[col].astype(str)
                continue

        # Handle PeriodDtype columns
        try:
            from pandas import PeriodDtype
            if isinstance(col_dtype, PeriodDtype):
                df[col] = df[col].dt.to_timestamp() if convert_period_to == 'timestamp' else df[col].astype(str)
                continue
        except:
            pass

        # Handle object columns
        if col_dtype == object:
            sample = df[col].dropna().head(50)

            # Check for Period objects
            if not sample.empty and any(isinstance(x, pd.Period) for x in sample):
                df[col] = df[col].apply(
                    lambda p: p.to_timestamp() if isinstance(p, pd.Period) else p
                ) if convert_period_to == 'timestamp' else df[col].astype(str)
                continue

            # Auto-detect datetime strings with warning suppression
            str_sample = sample.astype(str).str.strip().head(20)
            if len(str_sample) > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Could not infer format')
                    parsed_sample = pd.to_datetime(str_sample, errors='coerce', utc=True)

                if parsed_sample.notna().sum() / max(1, len(str_sample)) > 0.4:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='Could not infer format')
                            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                        continue
                    except:
                        pass

            df[col] = df[col].astype(str)

    return df

# ============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================================================

def load_finbert_model(model_name: str = MODEL_NAME):
    """Load FinBERT tokenizer and model."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Loading FinBERT model on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_article_sentiment(tokenizer, model, device, title, text,
                              chunk_chars=CHUNK_CHARS, max_chunks=MAX_ARTICLE_CHUNKS,
                              batch_size=8, max_length=512):
    """
    Predict sentiment for an article using FinBERT.
    Returns dict with avg_probs, final_label, and n_pieces.
    """
    # Build pieces (title + text chunks)
    pieces = []
    if title:
        pieces.append(title)
    if text:
        chunks = chunk_text(text, chunk_chars)
        pieces.extend(chunks)

    if not pieces:
        return None

    pieces = pieces[:max_chunks]

    # Batch process
    all_probs = []
    label_names = None

    try:
        for i in range(0, len(pieces), batch_size):
            batch_texts = pieces[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                             padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            # Get label names
            if label_names is None:
                try:
                    id2lab = model.config.id2label
                    label_names = [id2lab[i].lower() for i in range(len(id2lab))]
                except Exception:
                    label_names = ['positive', 'neutral', 'negative'] if probs.shape[1] == 3 else [f"lab_{j}" for j in range(probs.shape[1])]

            # Store probabilities
            for row in probs:
                all_probs.append({label_names[j]: float(row[j]) for j in range(len(row))})

    except Exception as e:
        logging.error(f"Sentiment prediction failed: {type(e).__name__} - {e}")
        return None

    # Aggregate: mean probability per label
    agg = {}
    for d in all_probs:
        for k, v in d.items():
            agg.setdefault(k, []).append(v)

    avg_probs = {k: float(np.mean(vlist)) for k, vlist in agg.items()}

    # Ensure all expected labels exist
    for k in ["positive", "neutral", "negative"]:
        avg_probs.setdefault(k, 0.0)

    final_label = max(avg_probs.items(), key=lambda kv: kv[1])[0]

    return {"avg_probs": avg_probs, "final_label": final_label, "n_pieces": len(all_probs)}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(query: str, years_back: int, articles_per_week: int, random_state: int, output_dir: str = "."):
    """Main pipeline to collect and analyze sentiment data."""
    
    # Setup paths
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * years_back)
    out_parquet = os.path.join(output_dir, f"gdelt_{query}_2yrs_finbert.parquet")
    temp_meta = os.path.join(output_dir, f'gdelt_query_urls_{query}.jsonl')
    
    logging.info("=" * 60)
    logging.info("SENTIMENT DATA COLLECTION PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Query: {query}")
    logging.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logging.info(f"Articles per week: {articles_per_week}")
    logging.info(f"Output file: {out_parquet}")
    
    # === STEP 1: Fetch GDELT article URLs ===
    logging.info("=" * 60)
    logging.info("STEP 1: Fetching GDELT article URLs")
    logging.info("=" * 60)
    
    meta_file = fetch_all_gdelt_urls(query, start_date, end_date, temp_meta)
    logging.info(f'Meta saved to {meta_file}')
    
    # === STEP 2: Load and sample metadata ===
    logging.info("=" * 60)
    logging.info("STEP 2: Loading and sampling metadata")
    logging.info("=" * 60)
    
    meta_rows = load_urls_from_meta(temp_meta)
    meta_df = pd.DataFrame(meta_rows)
    meta_df['url'] = meta_df['url'].astype(str)
    meta_df.drop_duplicates(subset=['url'], inplace=True)
    logging.info(f'Total unique URLs: {len(meta_df)}')
    
    # Parse seendate and drop invalid entries
    meta_df['seendate_parsed'] = pd.to_datetime(meta_df['seendate'], errors='coerce', utc=True)
    meta_df = meta_df.dropna(subset=['seendate_parsed'])
    
    # Use to_timestamp() to avoid PeriodIndex issues
    meta_df['week'] = meta_df['seendate_parsed'].dt.to_period('W-MON').dt.to_timestamp()
    
    # Sample articles per week
    sampled_df = meta_df.groupby('week').apply(
        lambda g: g.sample(n=min(len(g), articles_per_week), random_state=random_state)
    ).reset_index(drop=True)
    
    logging.info(f"Selected {len(sampled_df)} articles ({sampled_df['week'].nunique()} weeks)")
    meta_df = sampled_df
    
    # === STEP 3: Extract text and run FinBERT ===
    logging.info("=" * 60)
    logging.info("STEP 3: Extracting text and analyzing sentiment")
    logging.info("=" * 60)
    
    # Load existing results for resume
    done_df = pd.DataFrame()
    if os.path.exists(out_parquet):
        logging.info(f'Loading existing parquet {out_parquet} for resume')
        done_df = pd.read_parquet(out_parquet)
    
    # Prepare set of processed URLs for quick lookup
    processed_urls = set(done_df['url'].values) if not done_df.empty else set()
    
    # Load FinBERT model
    tokenizer, model, device = load_finbert_model(MODEL_NAME)
    
    # Track statistics
    stats = {
        "total_meta": len(meta_df),
        "no_content": 0,
        "sent_fail": 0,
        "ok": 0,
    }
    
    new_rows = []
    
    for idx, r in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Processing articles'):
        url = r['url']
        
        # Skip if already processed
        if url in processed_urls:
            continue
        
        title_meta = r.get('title') or ''
        title, text, pubdate = extract_text_with_newspaper(url, fallback_text=title_meta)
        time.sleep(PAUSE_BETWEEN_DOWNLOADS)
        
        if not title and not text:
            logging.debug(f'No content for url {url} (skipping)')
            stats["no_content"] += 1
            continue
        
        pred = predict_article_sentiment(tokenizer, model, device, title or title_meta, text or '')
        if pred is None:
            stats["sent_fail"] += 1
            continue
        
        avg_probs = pred['avg_probs']
        rec = {
            'url': url,
            'rss_title': title_meta,
            'title': title,
            'published_raw': pubdate if pubdate else r.get('seendate'),
            'sent_label': pred['final_label'],
            'prob_positive': avg_probs.get('positive', 0.0),
            'prob_neutral': avg_probs.get('neutral', 0.0),
            'prob_negative': avg_probs.get('negative', 0.0),
            'n_pieces': pred['n_pieces']
        }
        new_rows.append(rec)
        processed_urls.add(url)
        stats["ok"] += 1
        
        # Save every 50 articles
        if len(new_rows) >= 50:
            logging.info(f'Checkpoint: Saving {len(new_rows)} new rows ({len(done_df) + len(new_rows)} total)')
            chunk_df = pd.DataFrame(new_rows)
            done_df = pd.concat([done_df, chunk_df], ignore_index=True) if not done_df.empty else chunk_df.copy()
            safe_df = sanitize_df_for_parquet(done_df, convert_period_to='timestamp')
            safe_to_parquet(safe_df, out_parquet, index=False)
            new_rows = []
    
    # Final save
    if new_rows:
        logging.info(f'Final save: {len(new_rows)} new rows')
        chunk_df = pd.DataFrame(new_rows)
        done_df = pd.concat([done_df, chunk_df], ignore_index=True) if not done_df.empty else chunk_df.copy()
        safe_df = sanitize_df_for_parquet(done_df, convert_period_to='timestamp')
        safe_to_parquet(safe_df, out_parquet, index=False)
    
    logging.info(f'Finished extraction & prediction; total articles: {len(done_df)}')
    
    # Print statistics
    logging.info("=" * 60)
    logging.info("COLLECTION STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total URLs sampled: {stats['total_meta']}")
    logging.info(f"Successfully processed: {stats['ok']} ({stats['ok']/stats['total_meta']*100:.1f}%)")
    logging.info(f"No content extracted: {stats['no_content']} ({stats['no_content']/stats['total_meta']*100:.1f}%)")
    logging.info(f"Sentiment analysis failed: {stats['sent_fail']} ({stats['sent_fail']/stats['total_meta']*100:.1f}%)")
    logging.info(f"Output saved to: {out_parquet}")
    
    return out_parquet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sentiment data from GDELT news articles")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Search query (default: Bitcoin)")
    parser.add_argument("--years-back", type=int, default=DEFAULT_YEARS_BACK, help="Years of historical data (default: 2)")
    parser.add_argument("--articles-per-week", type=int, default=DEFAULT_ARTICLES_PER_WEEK, help="Articles to sample per week (default: 10)")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for parquet file (default: current directory)")
    
    args = parser.parse_args()
    
    main(
        query=args.query,
        years_back=args.years_back,
        articles_per_week=args.articles_per_week,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
