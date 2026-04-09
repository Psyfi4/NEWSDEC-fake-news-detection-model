# =============================================================================
# train.py — Model Training & Evaluation Pipeline
# =============================================================================
# This script:
#   1. Loads the WELFake dataset (or a synthetic fallback)
#   2. Preprocesses text with the NLP pipeline
#   3. Builds TF-IDF feature vectors
#   4. Trains two classifiers:
#        - Logistic Regression
#        - Passive Aggressive Classifier
#   5. Evaluates both models (accuracy, precision, recall, F1)
#   6. Plots confusion matrices
#   7. Saves models + vectoriser to disk via pickle
# =============================================================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (server-safe)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model       import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes        import MultinomialNB
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.metrics            import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        classification_report,
                                        confusion_matrix)
from preprocess import preprocess_series

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ISOT dataset (two separate files — fake.csv and true.csv)
ISOT_FAKE_PATH     = os.path.join(DATA_DIR, "fake.csv")
ISOT_TRUE_PATH     = os.path.join(DATA_DIR, "true.csv")
# WELFake single-file fallback
WELFAKE_PATH       = os.path.join(DATA_DIR, "WELFake_Dataset.csv")

VECTORISER_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LR_MODEL_PATH      = os.path.join(MODELS_DIR, "logistic_regression.pkl")
PAC_MODEL_PATH     = os.path.join(MODELS_DIR, "passive_aggressive.pkl")
NB_MODEL_PATH      = os.path.join(MODELS_DIR, "naive_bayes.pkl")
CONF_MATRIX_PATH   = os.path.join(BASE_DIR, "static", "confusion_matrices.png")


# ---------------------------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------------------------

def load_isot(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Load the ISOT Fake News Dataset (two separate CSVs).

    fake.csv  → label 0 (FAKE)
    true.csv  → label 1 (REAL)

    Expected columns in each file: title, text, subject, date
    """
    print(f"[Data] Loading ISOT dataset …")
    print(f"       Fake: {fake_path}")
    print(f"       True: {true_path}")

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Normalise column names
    df_fake.columns = [c.strip().lower() for c in df_fake.columns]
    df_true.columns = [c.strip().lower() for c in df_true.columns]

    # Assign labels  (0 = FAKE, 1 = REAL)
    df_fake["label"] = 0
    df_true["label"] = 1

    # Combine both files
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Build a rich content field from title + text
    title_col = "title" if "title" in df.columns else None
    text_col  = "text"  if "text"  in df.columns else None

    parts = []
    if title_col:
        parts.append(df[title_col].fillna(""))
    if text_col:
        parts.append(df[text_col].fillna(""))

    if not parts:
        raise ValueError(
            "Could not find 'title' or 'text' columns in the CSV files. "
            f"Columns found: {list(df.columns)}"
        )

    df["content"] = parts[0] if len(parts) == 1 else (parts[0] + " " + parts[1])
    df["content"] = df["content"].str.strip()

    df = df[["content", "label"]].dropna()
    df = df[df["content"] != ""]

    # Shuffle so fake and real rows are interleaved
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[Data] Loaded {len(df):,} rows | "
          f"FAKE: {(df['label']==0).sum():,} | "
          f"REAL: {(df['label']==1).sum():,}")
    return df


def load_welfake(path: str) -> pd.DataFrame:
    """
    Load the WELFake CSV dataset (single file).

    WELFake label convention (confirmed by inspection):
        0 = REAL,  1 = FAKE
    Our internal convention:
        0 = FAKE,  1 = REAL

    Labels are flipped on load to match our convention.
    """
    print(f"[Data] Loading WELFake dataset from: {path}")
    df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]

    df["content"] = (df.get("title", pd.Series([""] * len(df))).fillna("") +
                     " " +
                     df.get("text",  pd.Series([""] * len(df))).fillna(""))
    df["content"] = df["content"].str.strip()

    df = df[["content", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Flip: WELFake 0→REAL(1), WELFake 1→FAKE(0)
    df["label"] = df["label"].map({0: 1, 1: 0})

    print(f"[Data] Loaded {len(df):,} rows | "
          f"FAKE: {(df['label']==0).sum():,} | "
          f"REAL: {(df['label']==1).sum():,}")
    return df


def generate_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """
    Create a small synthetic dataset so the project can run immediately
    without downloading external data.  NOT for production accuracy.
    """
    print(f"[Data] Generating synthetic dataset ({n} samples)…")
    rng = np.random.default_rng(42)

    fake_phrases = [
        "shocking secret the government doesn't want you to know",
        "miracle cure doctors hate this one weird trick",
        "breaking exclusive bombshell celebrity scandal exposed",
        "unbelievable conspiracy globalist agenda revealed",
        "deep state plot uncovered whistleblower speaks out",
        "outrageous lie media silent on stunning revelation",
        "hidden truth suppressed by mainstream media elites",
        "you won't believe what happened next viral story",
    ]
    real_phrases = [
        "researchers publish findings in peer reviewed journal",
        "government officials announce new policy measures",
        "stock market closes higher amid economic uncertainty",
        "scientists conduct study on climate change effects",
        "hospital reports decline in seasonal flu cases",
        "city council votes on infrastructure improvement bill",
        "university study examines social media usage patterns",
        "federal reserve adjusts interest rate following meeting",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "content": " ".join(rng.choice(fake_phrases, size=rng.integers(3, 8))),
            "label":   0,
        })
        records.append({
            "content": " ".join(rng.choice(real_phrases, size=rng.integers(3, 8))),
            "label":   1,
        })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[Data] Synthetic dataset ready: {len(df)} rows")
    return df


def load_dataset() -> pd.DataFrame:
    """
    Load WELFake as the sole training dataset.

    WELFake combines PolitiFact, GossipCop, BuzzFeed and WorldNews —
    giving 72k diverse articles with no single-source writing style bias.

    ISOT (fake.csv / true.csv) is intentionally excluded: every REAL
    article in ISOT is Reuters wire copy, so the model learns
    'Reuters style = REAL' instead of genuine fake vs real signals,
    which causes legitimate non-Reuters news to be flagged as FAKE.
    """
    if os.path.exists(WELFAKE_PATH):
        return load_welfake(WELFAKE_PATH)

    print("[Data] WELFake_Dataset.csv not found. Using synthetic data (low accuracy).")
    print(f"[Data] Place WELFake_Dataset.csv in '{DATA_DIR}' for proper training.")
    return generate_synthetic_dataset(n=4000)


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame):
    """
    Preprocess text column and return (X, y) arrays.

    Returns:
        X: list of cleaned text strings
        y: numpy array of integer labels (0=FAKE, 1=REAL)
    """
    print("[Preprocess] Starting text cleaning pipeline…")
    X = preprocess_series(df["content"]).tolist()
    y = df["label"].values
    return X, y


# ---------------------------------------------------------------------------
# 3. TF-IDF Vectorisation
# ---------------------------------------------------------------------------

def build_vectorizer(X_train: list) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectoriser on the training corpus.

    Parameters chosen for fake-news detection:
      - sublinear_tf:  Apply log scaling to term frequency.
      - max_features:  Cap vocabulary at 100k terms to limit memory.
      - ngram_range:   Include bigrams (improves contextual understanding).
      - min_df:        Ignore terms that appear in fewer than 3 documents.
    """
    print("[TF-IDF] Fitting vectoriser on training data…")
    vec = TfidfVectorizer(
        sublinear_tf=True,
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=3,
        analyzer="word",
    )
    vec.fit(X_train)
    print(f"[TF-IDF] Vocabulary size: {len(vec.vocabulary_):,}")
    return vec


# ---------------------------------------------------------------------------
# 4. Model Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    print("[Model] Training Logistic Regression…")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[Model] Logistic Regression training complete.")
    return model


def train_passive_aggressive(X_train, y_train):
    """
    Train a Passive Aggressive Classifier wrapped in Platt scaling so it
    outputs calibrated probabilities (needed for reliable confidence scores).
    """
    print("[Model] Training Passive Aggressive Classifier…")
    base = PassiveAggressiveClassifier(
        C=0.5,
        max_iter=1000,
        random_state=42,
        tol=1e-4,
    )
    # CalibratedClassifierCV adds predict_proba() via cross-validated Platt scaling
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_train, y_train)
    print("[Model] Passive Aggressive training complete.")
    return model


def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    """
    Train a Multinomial Naive Bayes classifier.
    MNB works directly on TF-IDF counts and generalises well across
    diverse news sources — helps offset ISOT's Reuters-style bias.
    """
    print("[Model] Training Multinomial Naive Bayes…")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    print("[Model] Naive Bayes training complete.")
    return model


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute and print classification metrics for a fitted model.

    Returns a dict with keys: accuracy, precision, recall, f1, report, cm.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test,    y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test,        y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred,
                                   target_names=["FAKE", "REAL"],
                                   zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    separator = "=" * 55
    print(f"\n{separator}")
    print(f" Results — {model_name}")
    print(separator)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"\n{report}")

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "report":    report,
        "cm":        cm,
        "name":      model_name,
    }


# ---------------------------------------------------------------------------
# 6. Confusion Matrix Plot
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results: list, save_path: str):
    """
    Plot side-by-side confusion matrices for all evaluated models and save
    the figure as a PNG file.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"],
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(f"{res['name']}\nAccuracy: {res['accuracy']*100:.2f}%",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.suptitle("Confusion Matrices — Fake News Detection", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Confusion matrices saved → {save_path}")


# ---------------------------------------------------------------------------
# 7. Save / Load Artefacts
# ---------------------------------------------------------------------------

def save_artifacts(vectorizer, lr_model, pac_model, nb_model):
    """Pickle the vectoriser and all three models to the models/ directory."""
    for obj, path, label in [
        (vectorizer, VECTORISER_PATH, "TF-IDF Vectoriser"),
        (lr_model,   LR_MODEL_PATH,  "Logistic Regression"),
        (pac_model,  PAC_MODEL_PATH, "Passive Aggressive"),
        (nb_model,   NB_MODEL_PATH,  "Naive Bayes"),
    ]:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[Save] {label} → {path}")


def load_artifacts():
    """
    Load and return (vectorizer, lr_model, pac_model, nb_model) from disk.
    Raises FileNotFoundError if models haven't been trained yet.
    """
    missing = [p for p in (VECTORISER_PATH, LR_MODEL_PATH,
                            PAC_MODEL_PATH, NB_MODEL_PATH)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Run `python train.py` first to train the models."
        )

    with open(VECTORISER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model   = pickle.load(f)
    with open(PAC_MODEL_PATH, "rb") as f:
        pac_model  = pickle.load(f)
    with open(NB_MODEL_PATH, "rb") as f:
        nb_model   = pickle.load(f)

    return vectorizer, lr_model, pac_model, nb_model


# ---------------------------------------------------------------------------
# 8. Prediction helper (used by Flask app)
# ---------------------------------------------------------------------------

def predict(text: str, vectorizer, model, model_name: str = "",
            fake_threshold: float = 0.50) -> dict:
    """
    Predict whether a single news article is FAKE or REAL.

    Args:
        text:           Raw article text entered by the user.
        vectorizer:     Fitted TfidfVectorizer.
        model:          Trained classifier (LR, PAC, or NB).
        model_name:     Human-readable model name string (optional).
        fake_threshold: Minimum probability required to label as FAKE.
                        Raising this above 0.5 reduces false positives on
                        real news when training data has source-style bias
                        (e.g. ISOT dataset). Default: 0.65.

    Returns:
        dict with keys: label (str), confidence (float 0-100), model (str)
    """
    from preprocess import preprocess_text
    cleaned = preprocess_text(text)
    vec     = vectorizer.transform([cleaned])

    # Get probability estimates
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]   # [p_fake, p_real]
        p_fake     = float(proba[0])
        p_real     = float(proba[1])
    elif hasattr(model, "decision_function"):
        score  = model.decision_function(vec)[0]
        # Sigmoid-normalise: positive score → leaning REAL (label=1)
        p_real = float(1 / (1 + np.exp(-score)))
        p_fake = 1.0 - p_real
    else:
        # Fallback: use raw prediction with 100% confidence
        raw    = int(model.predict(vec)[0])
        p_real = float(raw)
        p_fake = 1.0 - p_real

    # ── Classification with explicit confidence thresholds ──────────────────
    # WELFake is ~95% US political news. Articles outside that domain
    # (international news, science, sports, etc.) often get medium-confidence
    # wrong predictions. We require HIGH confidence before committing to a
    # label — anything in between is shown as UNCERTAIN in the UI.
    #
    #   p_fake >= 0.80  →  FAKE      (model is strongly sure it's fake)
    #   p_fake <= 0.20  →  REAL      (model is strongly sure it's real)
    #   anything else   →  UNCERTAIN (not enough signal — verify manually)

    FAKE_THRESHOLD = 0.80   # must be >= 80% sure to call FAKE
    REAL_THRESHOLD = 0.20   # must be <= 20% fake-prob to call REAL

    if p_fake >= FAKE_THRESHOLD:
        label      = "FAKE"
        confidence = p_fake
    elif p_fake <= REAL_THRESHOLD:
        label      = "REAL"
        confidence = p_real
    else:
        label      = "UNCERTAIN"
        confidence = max(p_fake, p_real)

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),   # percentage
        "model":      model_name,
    }


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  FAKE NEWS DETECTION — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── 1. Load data ────────────────────────────────────────────────────────
    df = load_dataset()

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    X, y = prepare_data(df)

    # ── 3. Train / test split (80 / 20) ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[Split] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 4. TF-IDF vectorisation ──────────────────────────────────────────────
    vectorizer = build_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # ── 5. Train all three models ────────────────────────────────────────────
    lr_model  = train_logistic_regression(X_train_vec, y_train)
    pac_model = train_passive_aggressive(X_train_vec,  y_train)
    nb_model  = train_naive_bayes(X_train_vec,         y_train)

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    results = [
        evaluate_model(lr_model,  X_test_vec, y_test, "Logistic Regression"),
        evaluate_model(pac_model, X_test_vec, y_test, "Passive Aggressive Classifier"),
        evaluate_model(nb_model,  X_test_vec, y_test, "Naive Bayes"),
    ]

    # ── 7. Plot confusion matrices ───────────────────────────────────────────
    plot_confusion_matrices(results, CONF_MATRIX_PATH)

    # ── 8. Save artefacts ────────────────────────────────────────────────────
    save_artifacts(vectorizer, lr_model, pac_model, nb_model)

    # ── 9. Quick demo predictions (all 3 models + consensus) ────────────────
    demo_articles = [
        ("REAL",
         "President Biden signed the infrastructure bill into law on Monday, "
         "allocating $1.2 trillion for roads, bridges, broadband internet and "
         "public transit. The White House called it a generational investment."),
        ("REAL",
         "Apple reported quarterly earnings of $89.5 billion, beating analyst "
         "expectations. CEO Tim Cook cited strong iPhone sales in emerging markets "
         "and continued growth in the services division."),
        ("FAKE",
         "SHOCKING: Scientists CONFIRM 5G towers cause COVID-19! Government "
         "hiding the TRUTH! Mainstream media SILENCED by Big Pharma elites. "
         "Share before they DELETE this!"),
        ("FAKE",
         "BOMBSHELL: Hillary Clinton ARRESTED at airport, deep state EXPOSED! "
         "Mainstream media blackout on this HUGE story. Patriots share NOW "
         "before globalists take this down forever!!!"),
    ]

    print("\n" + "=" * 55)
    print(" DEMO PREDICTIONS  (LR | PAC | NB  →  consensus)")
    print("=" * 55)
    for expected, article in demo_articles:
        preds = [
            predict(article, vectorizer, lr_model,  "LR"),
            predict(article, vectorizer, pac_model, "PAC"),
            predict(article, vectorizer, nb_model,  "NB"),
        ]
        labels     = [p["label"] for p in preds]
        consensus  = "FAKE" if labels.count("FAKE") > labels.count("REAL") else "REAL"
        status     = "✓" if consensus == expected else "✗"
        detail     = "  ".join(
            f"{p['model']}:{p['label']}({p['confidence']}%)" for p in preds
        )
        print(f"\n[{status}] Expected:{expected:<5} Consensus:{consensus}")
        print(f"    {detail}")
        print(f"    {article[:85]}…")

    print("\n[Done] Training pipeline complete.\n")


if __name__ == "__main__":
    main()
