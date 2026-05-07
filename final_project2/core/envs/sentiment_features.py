"""
News sentiment feature pipeline for portfolio RL.

Two sources (both optional, fall back to zeros if unavailable):

1. SF Fed Daily News Sentiment Index (DNSI)
   - Precomputed daily series measuring news-based economic sentiment.
   - Coverage: 1980-01-01 to present (updated quarterly by SF Fed).
   - Download the Excel file from:
       https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/
   - Save it as:  data/dnsi.xlsx  (relative to the project root)
   - Alternatively, the downloader will try the known URL pattern automatically.

2. Alpaca News API — financial news articles → sentence embeddings
   - Fetches historical news headlines for each ticker via Alpaca Markets.
   - Encodes them with a sentence-transformer model (e.g., all-MiniLM-L6-v2).
   - Precompute and cache to:  data/alpaca_embeddings_{start}_{end}.pkl
   - Requires env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
   - This is expensive to generate; run precompute_alpaca_embeddings() once.

Usage in data_utils:
    sentiment_arr = load_sentiment(price_index, start, end)
    # Returns (T, n_sentiment_features) array; zeros if data unavailable.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Dimension of the sentiment feature vector appended to macro features
N_SENTIMENT_DNSI = 1       # scalar SF Fed sentiment score
N_ALPACA_EMBED_DIM = 384   # all-MiniLM-L6-v2 embedding size (set to 0 to disable)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Known SF Fed DNSI Excel URL pattern (updated ~quarterly by SF Fed)
_DNSI_URL = "https://www.frbsf.org/wp-content/uploads/DNSI_data_2024Q4.xlsx"


# ──────────────────────────────────────────────────────────────────────────────
# SF Fed Daily News Sentiment Index
# ──────────────────────────────────────────────────────────────────────────────

def load_dnsi(
    price_index: pd.DatetimeIndex,
    local_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Load SF Fed Daily News Sentiment Index, aligned to price_index.

    Tries (in order):
      1. local_path or data/dnsi.xlsx
      2. Download from _DNSI_URL
      3. Return zeros

    Returns:
        array of shape (T, 1), float32, values roughly in [-3, 3].
    """
    local_path = local_path or (_DATA_DIR / "dnsi.xlsx")

    series = None

    # Try loading from disk
    if local_path.exists():
        try:
            series = _parse_dnsi_excel(local_path)
            print(f"  [Sentiment] Loaded SF Fed DNSI from {local_path}")
        except Exception as e:
            warnings.warn(f"Could not parse DNSI file {local_path}: {e}")

    # Try downloading
    if series is None:
        try:
            import requests
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            response = requests.get(_DNSI_URL, timeout=30)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            series = _parse_dnsi_excel(local_path)
            print(f"  [Sentiment] Downloaded SF Fed DNSI to {local_path}")
        except Exception as e:
            warnings.warn(
                f"Could not download SF Fed DNSI ({e}). "
                f"Download manually from: https://www.frbsf.org/research-and-insights/"
                f"data-and-indicators/daily-news-sentiment-index/ "
                f"and save to {local_path}",
                UserWarning,
            )

    if series is None:
        return np.zeros((len(price_index), 1), dtype=np.float32)

    aligned = series.reindex(price_index, method="ffill").ffill().bfill().fillna(0.0)
    return aligned.values.reshape(-1, 1).astype(np.float32)


def _parse_dnsi_excel(path: Path) -> pd.Series:
    """Parse SF Fed DNSI Excel into a date-indexed float Series."""
    xl = pd.read_excel(path, sheet_name=0, header=None)
    # The Excel has two columns: date and sentiment score
    # Find the first row that looks like a date
    date_col, val_col = None, None
    for col_idx in range(xl.shape[1]):
        sample = xl.iloc[:5, col_idx].dropna()
        if sample.empty:
            continue
        try:
            pd.to_datetime(sample.iloc[0])
            date_col = col_idx
            val_col = col_idx + 1
            break
        except Exception:
            continue
    if date_col is None:
        raise ValueError("Could not identify date column in DNSI Excel file")

    df = xl[[date_col, val_col]].copy()
    df.columns = ["date", "sentiment"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)
    s = df.set_index("date")["sentiment"]
    s.index = pd.DatetimeIndex(s.index)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Alpaca News API → Sentence Embeddings
# ──────────────────────────────────────────────────────────────────────────────

class AlpacaNewsEmbeddings:
    """
    Fetches daily financial news headlines for a ticker universe via the Alpaca
    Markets News API, encodes them with a sentence-transformer model, and
    aggregates per-day embeddings (mean pooling across headlines).

    The embedding cache is persisted to disk so this only needs to run once.

    Requirements:
        pip install alpaca-trade-api sentence-transformers
        export ALPACA_API_KEY=...
        export ALPACA_SECRET_KEY=...

    Usage:
        emb = AlpacaNewsEmbeddings(tickers=["SPY", "TLT", "GLD"])
        emb.precompute(start="2008-01-01", end="2026-03-31")
        arr = emb.load(price_index)  # (T, embed_dim)
    """

    def __init__(
        self,
        tickers: list,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        embed_dim: int = N_ALPACA_EMBED_DIM,
    ):
        self.tickers = tickers
        self.model_name = model_name
        self.cache_dir = cache_dir or _DATA_DIR
        self.embed_dim = embed_dim
        self._embeddings: Optional[pd.DataFrame] = None  # date-indexed, shape (T, embed_dim)

    def _cache_path(self, start: str, end: str) -> Path:
        ticker_hash = "_".join(sorted(self.tickers))[:40]
        return self.cache_dir / f"alpaca_embeddings_{ticker_hash}_{start}_{end}.pkl"

    def precompute(self, start: str, end: str) -> None:
        """
        Fetch news from Alpaca and encode with sentence-transformer.
        This can take several hours for multi-year date ranges — run once.
        """
        cache_path = self._cache_path(start, end)
        if cache_path.exists():
            print(f"  [Alpaca] Embedding cache already exists: {cache_path}")
            return

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY env vars required. "
                "Sign up at https://alpaca.markets for a free paper-trading account."
            )

        try:
            import alpaca_trade_api as tradeapi
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Run: uv add alpaca-trade-api sentence-transformers"
            )

        api = tradeapi.REST(api_key, secret_key, base_url="https://paper-api.alpaca.markets")
        model = SentenceTransformer(self.model_name)

        dates = pd.bdate_range(start=start, end=end)
        daily_embeddings = {}

        for date in dates:
            date_str = date.strftime("%Y-%m-%dT00:00:00Z")
            next_str = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            headlines = []
            for tic in self.tickers:
                try:
                    news = api.get_news(tic, start=date_str, end=next_str, limit=5)
                    headlines.extend([n.headline for n in news])
                except Exception:
                    pass

            if headlines:
                vecs = model.encode(headlines, normalize_embeddings=True)  # (n_headlines, dim)
                daily_embeddings[date] = vecs.mean(axis=0)
            else:
                daily_embeddings[date] = np.zeros(self.embed_dim, dtype=np.float32)

        result = pd.DataFrame.from_dict(daily_embeddings, orient="index")
        result.index = pd.DatetimeIndex(result.index)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=4)
        print(f"  [Alpaca] Saved embedding cache to {cache_path}")

    def load(self, price_index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
        """
        Load precomputed embeddings aligned to price_index.
        Returns zeros of shape (T, embed_dim) if cache doesn't exist.
        """
        cache_path = self._cache_path(start, end)
        if not cache_path.exists():
            warnings.warn(
                f"Alpaca embedding cache not found at {cache_path}. "
                "Run AlpacaNewsEmbeddings.precompute(start, end) first. "
                "Returning zero embeddings.",
                UserWarning,
            )
            return np.zeros((len(price_index), self.embed_dim), dtype=np.float32)

        with open(cache_path, "rb") as f:
            df = pickle.load(f)

        aligned = df.reindex(price_index, method="ffill").ffill().bfill().fillna(0.0)
        return aligned.values.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Combined sentiment loader
# ──────────────────────────────────────────────────────────────────────────────

def load_sentiment(
    price_index: pd.DatetimeIndex,
    start: str,
    end: str,
    tickers: Optional[list] = None,
    use_alpaca_embeddings: bool = False,
) -> np.ndarray:
    """
    Return sentiment feature array aligned to price_index.

    Shape:
      - use_alpaca_embeddings=False: (T, 1)   — SF Fed DNSI only
      - use_alpaca_embeddings=True:  (T, 1 + embed_dim) — DNSI + Alpaca embeddings

    All zeros if data sources are unavailable.
    """
    dnsi = load_dnsi(price_index)  # (T, 1)

    if not use_alpaca_embeddings or tickers is None:
        return dnsi

    emb = AlpacaNewsEmbeddings(tickers=tickers)
    alpaca = emb.load(price_index, start, end)  # (T, embed_dim)
    return np.concatenate([dnsi, alpaca], axis=1).astype(np.float32)
