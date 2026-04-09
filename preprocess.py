# =============================================================================
# preprocess.py — NLP Text Preprocessing Pipeline
# =============================================================================
# This module handles all text cleaning and normalization steps:
#   - Lowercasing
#   - HTML / URL / punctuation removal
#   - Tokenization
#   - Stopword removal
#   - Lemmatization
# =============================================================================

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Download required NLTK resources (runs only if not already present)
# ---------------------------------------------------------------------------
def download_nltk_resources():
    """Download necessary NLTK data packages."""
    resources = [
        ("tokenizers/punkt",           "punkt"),
        ("tokenizers/punkt_tab",       "punkt_tab"),
        ("corpora/stopwords",          "stopwords"),
        ("corpora/wordnet",            "wordnet"),
        ("corpora/omw-1.4",            "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK] Downloading '{name}'...")
            nltk.download(name, quiet=True)

# Run once on import
download_nltk_resources()

# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused across calls)
# ---------------------------------------------------------------------------
_lemmatizer   = WordNetLemmatizer()
_stop_words   = set(stopwords.words("english"))
_punct_table  = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """Strip HTML tags using a simple regex."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www addresses."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_punctuation(text: str) -> str:
    """Delete all punctuation characters."""
    return text.translate(_punct_table)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps applied in order:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove punctuation & digits
        5. Tokenize
        6. Remove stopwords
        7. Lemmatize tokens
        8. Re-join into a cleaned string

    Args:
        text: Raw news article text (str).

    Returns:
        Cleaned, lemmatized string ready for vectorisation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = remove_html_tags(text)

    # 3. Remove URLs
    text = remove_urls(text)

    # 4. Remove punctuation
    text = remove_punctuation(text)

    # 5. Remove digits
    text = re.sub(r"\d+", " ", text)

    # 6. Remove extra whitespace
    text = remove_extra_whitespace(text)

    # 7. Tokenize
    tokens = word_tokenize(text)

    # 8. Remove stopwords and very short tokens (len < 2)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]

    # 9. Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(series: pd.Series, verbose: bool = True) -> pd.Series:
    """
    Apply preprocess_text() to every element of a Pandas Series.

    Args:
        series:  A Pandas Series containing raw text strings.
        verbose: Print progress indicator if True.

    Returns:
        A new Series with cleaned text.
    """
    if verbose:
        print(f"[Preprocess] Cleaning {len(series):,} documents...")
    cleaned = series.fillna("").apply(preprocess_text)
    if verbose:
        print("[Preprocess] Done.")
    return cleaned


# ---------------------------------------------------------------------------
# Quick self-test (run: python preprocess.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "BREAKING NEWS!!! President signs NEW bill into law — visit http://fakenews.com for details.",
        "<p>Scientists discover water on Mars. <b>This changes everything!</b></p>",
        "The 2024 election results have been CONTESTED by multiple parties across 50 states.",
        "",           # edge case: empty string
        None,         # edge case: None value (handled via fillna)
    ]

    print("=" * 60)
    print("Preprocessing self-test")
    print("=" * 60)
    for i, s in enumerate(samples):
        result = preprocess_text(s or "")
        print(f"\n[{i+1}] RAW    : {str(s)[:80]}")
        print(f"     CLEANED: {result[:80]}")
