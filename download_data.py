#!/usr/bin/env python3
# =============================================================================
# download_data.py — WELFake Dataset Downloader
# =============================================================================
# This helper script downloads the WELFake dataset from Kaggle using the
# official Kaggle API.
#
# Prerequisites:
#   1. Install kaggle CLI:   pip install kaggle
#   2. Get your API token from https://www.kaggle.com/settings → "API" → "Create New Token"
#      This downloads a file called kaggle.json
#   3. Place kaggle.json in:
#        - Linux/Mac:  ~/.kaggle/kaggle.json
#        - Windows:    C:\Users\<YourUser>\.kaggle\kaggle.json
#   4. Run this script:  python data/download_data.py
#
# The WELFake CSV will be saved to: data/WELFake_Dataset.csv
# =============================================================================

import os
import sys
import zipfile
import subprocess

# ---------------------------------------------------------------------------
DATASET_SLUG = "saurabhshahane/fake-news-classification"
DEST_DIR     = os.path.dirname(os.path.abspath(__file__))  # → data/
CSV_NAME     = "WELFake_Dataset.csv"
CSV_PATH     = os.path.join(DEST_DIR, CSV_NAME)
# ---------------------------------------------------------------------------


def check_kaggle_installed() -> bool:
    """Return True if the kaggle CLI is installed and available."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_kaggle_credentials() -> bool:
    """Return True if kaggle.json credentials file exists."""
    home        = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_json)


def download_welfake():
    """Download and extract the WELFake dataset via the Kaggle API."""

    # ── Pre-flight checks ────────────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        print(f"[Info] Dataset already exists at:\n       {CSV_PATH}")
        print("[Info] Delete the file and re-run this script to refresh it.")
        return

    if not check_kaggle_installed():
        print("[Error] The 'kaggle' package is not installed.")
        print("        Run:  pip install kaggle")
        sys.exit(1)

    if not check_kaggle_credentials():
        print("[Error] Kaggle credentials not found.")
        print("        1. Go to https://www.kaggle.com/settings")
        print("        2. Under 'API', click 'Create New Token'")
        print("        3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("           (Windows: C:\\Users\\<YourUser>\\.kaggle\\kaggle.json)")
        sys.exit(1)

    # ── Download ────────────────────────────────────────────────────────
    print(f"[Download] Fetching dataset '{DATASET_SLUG}' from Kaggle…")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG,
         "--path", DEST_DIR, "--unzip"],
        capture_output=False,
    )

    if result.returncode != 0:
        print("[Error] Kaggle download failed. Check your credentials and internet connection.")
        sys.exit(1)

    # ── Verify ──────────────────────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
        print(f"\n[Success] Dataset saved → {CSV_PATH}  ({size_mb:.1f} MB)")
        print("[Next]    Run `python train.py` to train the models.")
    else:
        # Try to find the CSV in case it was named differently
        csvs = [f for f in os.listdir(DEST_DIR) if f.endswith(".csv")]
        if csvs:
            found = os.path.join(DEST_DIR, csvs[0])
            os.rename(found, CSV_PATH)
            print(f"[Success] Renamed '{csvs[0]}' → '{CSV_NAME}'")
            print(f"          Path: {CSV_PATH}")
        else:
            print("[Warning] Could not find a CSV in the data/ directory.")
            print("          Please download the dataset manually from:")
            print(f"          https://www.kaggle.com/datasets/{DATASET_SLUG}")
            print(f"          and place 'WELFake_Dataset.csv' in: {DEST_DIR}")


def manual_instructions():
    """Print manual download instructions for users without Kaggle CLI."""
    print("""
=============================================================
  MANUAL DOWNLOAD INSTRUCTIONS
=============================================================

If you prefer to download manually:

  1. Go to:
     https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

  2. Click "Download" (requires free Kaggle account)

  3. Extract the ZIP file

  4. Rename / move the CSV file to:
       <project-root>/data/WELFake_Dataset.csv

  5. Run the training pipeline:
       python train.py

=============================================================

Alternatively, run WITHOUT the dataset — train.py will
automatically use a built-in synthetic dataset so you can
see the app working immediately.
""")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        manual_instructions()
    else:
        download_welfake()
