"""Model evaluation, metrics computation, and reporting utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import EVALUATION_METRICS_PATH, MODEL_COMPARISON_PATH

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_proba: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
    model_name: str = "Model",
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics.

    Includes ROC-AUC, Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC curve points.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    cm = confusion_matrix(y_true, y_pred_binary).tolist()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Subsample ROC points if too large for clean JSON serialization
    if len(fpr) > 100:
        indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
        fpr_sub = fpr[indices].tolist()
        tpr_sub = tpr[indices].tolist()
    else:
        fpr_sub = fpr.tolist()
        tpr_sub = tpr.tolist()

    metrics = {
        "model_name": model_name,
        "roc_auc": round(roc_auc, 6),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(x, 5) for x in fpr_sub],
            "tpr": [round(y, 5) for y in tpr_sub],
        },
        "training_duration_seconds": round(duration_seconds, 2)
        if duration_seconds is not None
        else None,
    }

    return metrics


def save_model_comparison_report(
    metrics_list: List[Dict[str, Any]],
    csv_path: Path = MODEL_COMPARISON_PATH,
    json_path: Path = EVALUATION_METRICS_PATH,
) -> pd.DataFrame:
    """Formats model metrics into a comparison DataFrame, saves to CSV and JSON."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model_name"],
            "Validation ROC-AUC": m["roc_auc"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "Training Duration (s)": m.get("training_duration_seconds"),
        })

    df = pd.DataFrame(rows).sort_values("Validation ROC-AUC", ascending=False)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved model comparison table to {csv_path}")

    # Also save detailed JSON with confusion matrix and ROC curves
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)
    logger.info(f"Saved detailed evaluation metrics to {json_path}")

    return df
