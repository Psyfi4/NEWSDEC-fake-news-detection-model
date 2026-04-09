# 🔍 Fake News Detection System

A full-stack Machine Learning web app that classifies news articles as **FAKE** or **REAL** using NLP and scikit-learn — with a clean dark-mode UI.

---

## 📁 Project Structure

```
fake-news-detection/
│
├── app.py                  ← Flask web server + REST API
├── train.py                ← ML training pipeline (data → models)
├── preprocess.py           ← NLP text cleaning pipeline
├── requirements.txt        ← Python dependencies
│
├── data/
│   ├── download_data.py    ← Kaggle dataset downloader
│   └── WELFake_Dataset.csv ← (place dataset here — see Step 2)
│
├── models/                 ← Saved model pickles (auto-created)
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   └── passive_aggressive.pkl
│
├── templates/
│   └── index.html          ← Single-page frontend (Jinja2)
│
└── static/
    ├── style.css           ← Dark-mode UI stylesheet
    ├── script.js           ← Frontend interactivity
    └── confusion_matrices.png  ← Generated after training
```

---

## ⚡ Quick Start (5 minutes)

### Step 1 — Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2 — Get the dataset

**Option A — Kaggle (full WELFake dataset, ~72k articles, best accuracy):**

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to **Settings → API → Create New Token** — this downloads `kaggle.json`
3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<You>\.kaggle\` (Windows)
4. Run the downloader:
   ```bash
   python data/download_data.py
   ```

**Option B — Manual download:**
1. Visit: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
2. Download and extract the ZIP
3. Rename the CSV to `WELFake_Dataset.csv` and place it in the `data/` folder

**Option C — Skip dataset (instant demo mode):**
- Do nothing. If no CSV is found, `train.py` automatically uses a built-in synthetic dataset so the app runs immediately.

### Step 3 — Train the models

```bash
python train.py
```

This will:
- Preprocess text (tokenise, remove stopwords, lemmatise)
- Build a TF-IDF feature matrix
- Train Logistic Regression + Passive Aggressive classifiers
- Print accuracy / precision / recall / F1 scores
- Save confusion matrix image to `static/confusion_matrices.png`
- Pickle both models + vectoriser to `models/`

> **Expected accuracy on WELFake:** ~97–99%  
> **On synthetic data:** ~85–95% (for demo purposes only)

### Step 4 — Start the web app

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## 🌐 API Reference

### `POST /predict`

Classify a news article.

**Request:**
```json
{
  "text":  "Paste the article text here...",
  "model": "both"
}
```

`model` options: `"lr"` (Logistic Regression), `"pac"` (Passive Aggressive), `"both"` (default)

**Response:**
```json
{
  "success": true,
  "consensus": "FAKE",
  "text_length": 312,
  "results": [
    { "model": "Logistic Regression",           "label": "FAKE", "confidence": 97.82 },
    { "model": "Passive Aggressive Classifier", "label": "FAKE", "confidence": 94.11 }
  ]
}
```

### `GET /health`
Returns model loading status.

### `GET /retrain`
Triggers a full retraining cycle.

---

## 🧪 Sample Test Cases

| Expected | Article snippet |
|----------|----------------|
| REAL | *"The Federal Reserve raised its benchmark interest rate by 0.25 percentage points…"* |
| REAL | *"NASA scientists confirmed the discovery of water ice near the Martian south pole…"* |
| FAKE | *"SHOCKING: Big Pharma suppresses miracle cure! Share before they DELETE this!"* |
| FAKE | *"Government whistleblower reveals 5G towers broadcast mind-control frequencies…"* |

---

## 🤖 Models

| Model | Algorithm | Strengths |
|-------|-----------|-----------|
| Logistic Regression | Maximum likelihood, L2 regularised | Interpretable, outputs calibrated probabilities |
| Passive Aggressive Classifier | Online learning, hinge loss | Fast, handles large sparse feature matrices well |

Both models are trained on **TF-IDF bigram features** (up to 100,000 terms).

---

## 🔧 NLP Pipeline

1. **Lowercase** — normalise all text
2. **Remove HTML tags** — strip `<p>`, `<b>`, etc.
3. **Remove URLs** — strip `http://...` and `www....`
4. **Remove punctuation & digits** — keep only alphabetic tokens
5. **Tokenise** — split into individual words using NLTK `word_tokenize`
6. **Remove stopwords** — drop common English words (NLTK stopwords list)
7. **Lemmatise** — reduce words to their dictionary base form

---

## 📊 Evaluation Metrics

After training, the console prints:
- **Accuracy** — overall correct predictions
- **Precision** — of all "FAKE" predictions, how many were actually fake
- **Recall** — of all actual fake news, how many did we catch
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — saved as `static/confusion_matrices.png`

---

## 💾 Model Persistence

Models are saved to `models/` as pickle files:

```python
# Load manually in your own scripts:
import pickle

with open("models/tfidf_vectorizer.pkl",    "rb") as f: vec = pickle.load(f)
with open("models/logistic_regression.pkl", "rb") as f: lr  = pickle.load(f)
with open("models/passive_aggressive.pkl",  "rb") as f: pac = pickle.load(f)

cleaned = preprocess_text("your article here")
X       = vec.transform([cleaned])
print(lr.predict(X))          # [0] = FAKE, [1] = REAL
print(lr.predict_proba(X))    # [[p_fake, p_real]]
```

---

## 🛠 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port the Flask server listens on |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

```bash
PORT=8080 python app.py
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Web framework | Flask |
| ML library | scikit-learn |
| NLP | NLTK |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Dataset | WELFake (Kaggle) |

---

## ⚠️ Disclaimer

This tool is a **proof-of-concept** for educational purposes. It should not be used as the sole basis for determining the credibility of real news. Always verify information from multiple authoritative sources.
