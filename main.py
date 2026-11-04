"""AIRLINE SATISFACTION INTELLIGENCE
Unified Command-Line Interface (CLI) Entrypoint.

Usage:
    python main.py --train             # Train full model pipeline
    python main.py --train-quick       # Quick training run for fast verification
    python main.py --predict <csv>     # Predict satisfaction on an input CSV
    python main.py --test              # Run pytest test suite
    python main.py --app               # Launch Streamlit web application
    python main.py --status            # Check artifact and model readiness
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from config import (
    DATA_DIR,
    FINAL_SUBMISSION_PATH,
    LGBM_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    TRAIN_DATA_PATH,
)
from src.model_loader import are_artifacts_available
from src.predict import predict_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_status():
    print("=" * 60)
    print("  AIRLINE SATISFACTION INTELLIGENCE - SYSTEM STATUS")
    print("=" * 60)
    ready, status = are_artifacts_available()
    for artifact, exists in status.items():
        status_symbol = "[READY]  " if exists else "[MISSING]"
        print(f"  {status_symbol} {artifact}")

    print("-" * 60)
    train_exists = TRAIN_DATA_PATH.exists()
    status_train = "[FOUND]  " if train_exists else "[NOT FOUND]"
    print(f"  {status_train} Training Data ({TRAIN_DATA_PATH})")
    print("=" * 60)


def run_tests():
    logger.info("Executing pytest test suite...")
    cmd = [sys.executable, "-m", "pytest", "-v"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def run_train(quick: bool = False):
    from src.train import train_pipeline
    logger.info(f"Starting model training pipeline (quick_run={quick})...")
    try:
        train_pipeline(quick_run=quick)
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def run_predict(input_csv: str, output_csv: str = None):
    out_path = output_csv or "predictions_output.csv"
    logger.info(f"Predicting on {input_csv} -> {out_path}...")
    try:
        predict_csv(input_csv_path=input_csv, output_csv_path=out_path)
        logger.info(f"Predictions successfully saved to {out_path}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def run_app():
    logger.info("Launching Streamlit application...")
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Airline Satisfaction Intelligence Machine Learning CLI"
    )
    parser.add_argument("--train", action="store_true", help="Train models and generate artifacts")
    parser.add_argument("--train-quick", action="store_true", help="Quick training run for verification")
    parser.add_argument("--predict", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction output CSV")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application")
    parser.add_argument("--status", action="store_true", help="Display system readiness and artifacts status")

    args = parser.parse_args()

    if args.status:
        run_status()
    elif args.test:
        run_tests()
    elif args.train:
        run_train(quick=False)
    elif args.train_quick:
        run_train(quick=True)
    elif args.predict:
        run_predict(args.predict, args.output)
    elif args.app:
        run_app()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
