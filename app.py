# =============================================================================
# app.py — Flask Web Application
# =============================================================================
# Provides a REST API + HTML frontend for the Fake News Detection system.
#
# Routes:
#   GET  /           → Serve the main HTML page
#   POST /predict    → Accept JSON {"text": "..."} and return prediction
#   GET  /health     → Health-check endpoint (returns model status)
#   GET  /retrain    → Trigger model retraining (admin utility)
# =============================================================================

import os
import sys
import logging
from flask import Flask, request, jsonify, render_template, abort

# Add project root to sys.path so preprocess / train imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import load_artifacts, predict, main as run_training

# ---------------------------------------------------------------------------
# CORS — allow browser fetch() calls from any origin (fixes "Could not reach
# the server" errors when the browser's security policy blocks cross-origin
# POST requests, e.g. when accessing via LAN IP or some proxy setups)
# ---------------------------------------------------------------------------
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app initialisation
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["JSON_SORT_KEYS"] = False   # Keep JSON response key order

# Register CORS header injection on every response
app.after_request(add_cors_headers)


@app.route("/predict", methods=["OPTIONS"])
@app.route("/health",  methods=["OPTIONS"])
def options_handler():
    """Handle CORS pre-flight OPTIONS requests from browsers."""
    return "", 204

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Models are loaded once at startup and stored in module-level variables.
# This avoids the overhead of re-loading pickle files on every request.

_vectorizer = None
_lr_model   = None
_pac_model  = None
_nb_model   = None
_models_loaded = False


def load_models():
    """Load trained models from disk. Triggers training if models don't exist."""
    global _vectorizer, _lr_model, _pac_model, _nb_model, _models_loaded

    try:
        _vectorizer, _lr_model, _pac_model, _nb_model = load_artifacts()
        _models_loaded = True
        logger.info("✓ Models loaded successfully.")
    except FileNotFoundError:
        logger.warning("No trained models found. Running training pipeline…")
        run_training()
        _vectorizer, _lr_model, _pac_model, _nb_model = load_artifacts()
        _models_loaded = True
        logger.info("✓ Training complete. Models loaded.")


# Load models on first request (avoids Flask 2.x app-context timing issues)
@app.before_request
def ensure_models_loaded():
    """Load models on the very first incoming request if not already loaded."""
    global _models_loaded
    if not _models_loaded:
        load_models()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the main single-page HTML interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Endpoint: POST /predict
    ────────────────────────────────────────────────────────────────────────
    Request body (JSON):
        {
            "text":  "<news article text>",
            "model": "lr" | "pac" | "both"   (optional, default: "both")
        }

    Response (JSON):
        {
            "success": true,
            "results": [
                {
                    "model":      "Logistic Regression",
                    "label":      "FAKE" | "REAL",
                    "confidence": 97.34          ← percentage (0–100)
                },
                ...
            ],
            "consensus": "FAKE" | "REAL",        ← majority vote across models
            "text_length": 312                   ← character count of input
        }
    """
    if not _models_loaded:
        return jsonify({"success": False, "error": "Models not loaded yet."}), 503

    data = request.get_json(silent=True)

    # ── Input validation ────────────────────────────────────────────────────
    if not data:
        return jsonify({"success": False,
                        "error": "Request body must be JSON."}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": False,
                        "error": "Field 'text' is required and cannot be empty."}), 400

    if len(text) < 20:
        return jsonify({"success": False,
                        "error": "Please provide at least 20 characters for a reliable prediction."}), 400

    model_choice = data.get("model", "both").lower()

    # ── Run predictions ─────────────────────────────────────────────────────
    results = []

    if model_choice in ("lr", "both"):
        r = predict(text, _vectorizer, _lr_model, "Logistic Regression")
        results.append(r)

    if model_choice in ("pac", "both"):
        r = predict(text, _vectorizer, _pac_model, "Passive Aggressive Classifier")
        results.append(r)

    if model_choice in ("nb", "both"):
        r = predict(text, _vectorizer, _nb_model, "Naive Bayes")
        results.append(r)

    if not results:
        return jsonify({"success": False,
                        "error": "Invalid model choice. Use 'lr', 'pac', 'nb', or 'both'."}), 400

    # ── Consensus vote (majority of 3 models) ────────────────────────────────
    labels        = [r["label"] for r in results]
    fake_count    = labels.count("FAKE")
    real_count    = labels.count("REAL")
    uncertain_count = labels.count("UNCERTAIN")

    # Consensus rules:
    #  - Majority FAKE            → FAKE
    #  - Majority REAL            → REAL
    #  - Any UNCERTAIN + no clear majority → UNCERTAIN
    #    (model doesn't have enough signal to decide)
    if fake_count > real_count and fake_count > uncertain_count:
        consensus = "FAKE"
    elif real_count > fake_count and real_count > uncertain_count:
        consensus = "REAL"
    else:
        consensus = "UNCERTAIN"

    logger.info(f"Prediction | consensus={consensus} | "
                f"text_len={len(text)} | models={[r['model'] for r in results]}")

    return jsonify({
        "success":     True,
        "results":     results,
        "consensus":   consensus,
        "text_length": len(text),
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint: GET /health
    Returns the current status of the application and loaded models.
    """
    return jsonify({
        "status":        "ok" if _models_loaded else "models_not_loaded",
        "models_loaded": _models_loaded,
        "models": {
            "logistic_regression":         _lr_model   is not None,
            "passive_aggressive":          _pac_model  is not None,
            "naive_bayes":                 _nb_model   is not None,
            "tfidf_vectorizer":            _vectorizer is not None,
        },
    })


@app.route("/retrain", methods=["GET"])
def retrain():
    """
    Endpoint: GET /retrain
    Triggers a full retraining of both models.
    WARNING: This can take several minutes on large datasets.
    """
    global _vectorizer, _lr_model, _pac_model, _models_loaded
    try:
        logger.info("Retraining triggered via /retrain endpoint.")
        run_training()
        _vectorizer, _lr_model, _pac_model = load_artifacts()
        _models_loaded = True
        return jsonify({"success": True, "message": "Models retrained successfully."})
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Fake News Detection server → http://127.0.0.1:{port}")
    # threaded=True lets Flask handle multiple requests at once (required when
    # the browser sends a /health GET while /predict POST is still processing)
    app.run(host="127.0.0.1", port=port, debug=debug, threaded=True)
