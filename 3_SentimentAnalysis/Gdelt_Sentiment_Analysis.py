import os
import time
import math
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import torch.nn.functional as F
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from newspaper import Article, Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ============================================================================
# CONFIGURATION
# ============================================================================
QUERY = "Microsoft"
YEARS_BACK = 2
GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
OUT_PARQUET = f"gdelt_{QUERY}_2yrs_finbert.parquet"
OUT_WEEKLY_PNG = f"gdelt_{QUERY}_2yrs_weekly_sentiment.png"
MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_PER_CALL = 250
PAUSE_BETWEEN_CALLS = 5  # FIXED: GDELT requires 5 seconds minimum
PAUSE_BETWEEN_DOWNLOADS = 0.5
MAX_ARTICLE_CHUNKS = 6
CHUNK_CHARS = 1000
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=365 * YEARS_BACK)
TEMP_META = 'gdelt_query_urls.jsonl'
RANDOM_STATE = 42
ARTICLES_PER_WEEK = 15

print('Config set. Query:', QUERY)

# ============================================================================
# GDELT QUERY FUNCTIONS
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


def fetch_all_gdelt_urls(query: str, start: datetime, end: datetime, out_meta_file: str = TEMP_META):
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
# TEXT EXTRACTION
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
# PARQUET SANITIZATION
# ============================================================================
def sanitize_df_for_parquet(df: pd.DataFrame, convert_period_to='timestamp') -> pd.DataFrame:
    """
    Convert problematic pandas dtypes (Period, mixed objects) to parquet-safe types.
    Returns a sanitized copy.
    """
    df = df.copy()

    # Handle PeriodIndex
    if isinstance(df.index, pd.PeriodIndex):
        if convert_period_to == 'timestamp':
            df.index = df.index.to_timestamp()
        else:
            df.index = df.index.astype(str)

    # Column-wise sanitization
    for col in df.columns:
        col_dtype = df[col].dtype
        col_lower = col.lower()

        # Date/time columns: try to parse to timestamps
        if any(token in col_lower for token in ("date", "time", "published")):
            try:
                parsed = pd.to_datetime(df[col], errors='coerce', utc=True)
                if parsed.notna().any():
                    df[col] = parsed
                    continue
                else:
                    df[col] = df[col].astype(str)
                    continue
            except Exception:
                df[col] = df[col].astype(str)
                continue

        # Handle PeriodDtype columns
        try:
            from pandas import PeriodDtype
            if isinstance(col_dtype, PeriodDtype):
                if convert_period_to == 'timestamp':
                    try:
                        df[col] = df[col].dt.to_timestamp()
                    except Exception:
                        df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(str)
                continue
        except Exception:
            pass

        # Handle object columns with Period objects
        if col_dtype == object:
            sample = df[col].dropna().head(50)
            if not sample.empty and all(isinstance(x, pd.Period) for x in sample):
                if convert_period_to == 'timestamp':
                    df[col] = df[col].apply(lambda p: p.to_timestamp() if isinstance(p, pd.Period) else p)
                else:
                    df[col] = df[col].astype(str)
                continue

            # Auto-detect datetime strings
            str_sample = sample.astype(str).str.strip().head(20)
            if len(str_sample) > 0:
                parsed_sample = pd.to_datetime(str_sample, errors='coerce', utc=True)
                if parsed_sample.notna().sum() / max(1, len(str_sample)) > 0.4:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                        continue
                    except Exception:
                        pass

            # Default: convert to string for safety
            df[col] = df[col].astype(str)

    return df


# ============================================================================
# FINBERT SENTIMENT ANALYSIS
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
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Step 1: Fetch GDELT URLs
    logging.info("=" * 60)
    logging.info("STEP 1: Fetching GDELT article URLs")
    logging.info("=" * 60)
    meta_file = fetch_all_gdelt_urls(QUERY, START_DATE, END_DATE, TEMP_META)
    print(f'Meta saved to {meta_file}')

    # Step 2: Load and sample metadata
    logging.info("=" * 60)
    logging.info("STEP 2: Loading and sampling metadata")
    logging.info("=" * 60)
    meta_rows = load_urls_from_meta(TEMP_META)
    meta_df = pd.DataFrame(meta_rows)
    meta_df['url'] = meta_df['url'].astype(str)
    meta_df.drop_duplicates(subset=['url'], inplace=True)
    print(f'Total unique URLs: {len(meta_df)}')

    # Parse dates and sample
    meta_df['seendate_parsed'] = pd.to_datetime(meta_df['seendate'], errors='coerce', utc=True)
    meta_df = meta_df.dropna(subset=['seendate_parsed'])

    # FIXED: Use to_timestamp() to avoid PeriodIndex issues
    meta_df['week'] = meta_df['seendate_parsed'].dt.to_period('W-MON').dt.to_timestamp()

    # Sample articles per week
    sampled_df = meta_df.groupby('week').apply(
        lambda g: g.sample(n=min(len(g), ARTICLES_PER_WEEK), random_state=RANDOM_STATE)
    ).reset_index(drop=True)

    print(f"Selected {len(sampled_df)} articles ({sampled_df['week'].nunique()} weeks)")
    meta_df = sampled_df

    # Step 3: Extract text and run FinBERT
    logging.info("=" * 60)
    logging.info("STEP 3: Extracting text and analyzing sentiment")
    logging.info("=" * 60)

    # Load existing results for resume
    done_df = pd.DataFrame()
    if os.path.exists(OUT_PARQUET):
        logging.info(f'Loading existing parquet {OUT_PARQUET} for resume')
        done_df = pd.read_parquet(OUT_PARQUET)

    # FIXED: Use set for O(1) lookup instead of O(n)
    processed_urls = set(done_df['url'].values) if not done_df.empty else set()

    # Load FinBERT model
    tokenizer, model, device = load_finbert_model(MODEL_NAME)

    new_rows = []

    for idx, r in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Processing articles'):
        url = r['url']

        # FIXED: O(1) lookup
        if url in processed_urls:
            continue

        title_meta = r.get('title') or ''
        title, text, pubdate = extract_text_with_newspaper(url, fallback_text=title_meta)
        time.sleep(PAUSE_BETWEEN_DOWNLOADS)

        if not title and not text:
            logging.info(f'No content for url {url} (skipping)')
            continue

        pred = predict_article_sentiment(tokenizer, model, device, title or title_meta, text or '')
        if pred is None:
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
        processed_urls.add(url)  # FIXED: Add to set

        # Save every 50 articles
        if len(new_rows) >= 50:
            logging.info(f'Checkpoint: Saving {len(new_rows)} new rows ({len(done_df) + len(new_rows)} total)')
            chunk_df = pd.DataFrame(new_rows)
            done_df = pd.concat([done_df, chunk_df], ignore_index=True) if not done_df.empty else chunk_df.copy()
            safe_df = sanitize_df_for_parquet(done_df, convert_period_to='timestamp')
            safe_df.to_parquet(OUT_PARQUET, index=False)
            new_rows = []

    # Final save
    if new_rows:
        logging.info(f'Final save: {len(new_rows)} new rows')
        chunk_df = pd.DataFrame(new_rows)
        done_df = pd.concat([done_df, chunk_df], ignore_index=True) if not done_df.empty else chunk_df.copy()
        safe_df = sanitize_df_for_parquet(done_df, convert_period_to='timestamp')
        safe_df.to_parquet(OUT_PARQUET, index=False)

    logging.info(f'Finished extraction & prediction; total articles: {len(done_df)}')

    # Step 4: Generate weekly sentiment plot
    logging.info("=" * 60)
    logging.info("STEP 4: Generating weekly sentiment plot")
    logging.info("=" * 60)

    if 'done_df' not in locals() or done_df.empty:
        if os.path.exists(OUT_PARQUET):
            done_df = pd.read_parquet(OUT_PARQUET)
        else:
            raise FileNotFoundError('No prediction parquet found')

    # FIXED: Use copy to avoid modifying original DataFrame
    plot_df = done_df.copy()
    plot_df['published'] = pd.to_datetime(plot_df['published_raw'], errors='coerce', utc=True)
    plot_df['published'] = plot_df['published'].fillna(pd.Timestamp.utcnow())
    plot_df.set_index('published', inplace=True)

    # FIXED: Convert PeriodIndex to TimestampIndex after resample
    weekly = plot_df[['prob_positive','prob_neutral','prob_negative']].resample('W-MON').mean().sort_index()
    if isinstance(weekly.index, pd.PeriodIndex):
        weekly.index = weekly.index.to_timestamp()

    # Generate plot
    plt.figure(figsize=(12,5))
    plt.plot(weekly.index, weekly['prob_positive'], label='Positive', linewidth=2, marker='o')
    plt.plot(weekly.index, weekly['prob_neutral'], label='Neutral', linewidth=2, marker='s')
    plt.plot(weekly.index, weekly['prob_negative'], label='Negative', linewidth=2, marker='^')
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Average sentiment probability', fontsize=12)
    plt.title(f"Weekly-averaged FinBERT sentiment for '{QUERY}' ({START_DATE.date()} → {END_DATE.date()})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUT_WEEKLY_PNG, dpi=150)
    logging.info(f'Saved weekly plot to {OUT_WEEKLY_PNG}')
    plt.show()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total articles processed: {len(done_df)}")
    print(f"Output parquet: {OUT_PARQUET}")
    print(f"Output plot: {OUT_WEEKLY_PNG}")
