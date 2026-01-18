"""
Evaluation utilities.

This module is intentionally "library-only": it exposes functions you can import
from other modules or scripts.

Typical usage:
    import joblib
    import pandas as pd
    from evaluate import evaluate_model, load_label_map, invert_label_map

    df = pd.read_parquet("data/processed/dataset.parquet")
    model = joblib.load("models/best_pipeline.joblib")
    label_names = invert_label_map(load_label_map("models/label_map.json"))

    results, report, y_pred, cm = evaluate_model(
        model,
        df["text"].tolist(),
        df["label"].to_numpy(),
        label_names,
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class EvalResults:
    n_samples: int
    accuracy: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float


def ensure_dir(path: str | Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_label_map(path: str | Path) -> Dict[str, int]:
    """Load topic->id mapping from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    """Convert topic->id mapping to id->topic mapping."""
    return {int(v): str(k) for k, v in label_map.items()}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from parquet/csv/json."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use parquet/csv/json.")


def evaluate_model(
    model: Any,
    X: Any,
    y_true: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
) -> Tuple[EvalResults, str, np.ndarray, np.ndarray]:
    """
    Evaluate a fitted model/pipeline.

    Args:
        model: Any object with a .predict(X) method (e.g., sklearn Pipeline).
        X: Inputs accepted by model.predict (list[str] for Count/Tfidf,
           list[list[str]] for token models).
        y_true: Ground-truth labels as a 1D numpy array.
        label_names: Optional id->name mapping for readable reports.

    Returns:
        results: EvalResults dataclass (accuracy + f1s)
        report: classification_report() text
        y_pred: predicted labels
        cm: confusion matrix (labels follow label_names order if provided)
    """
    y_pred = model.predict(X)

    results = EvalResults(
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        f1_micro=float(f1_score(y_true, y_pred, average="micro")),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted")),
    )

    labels = None
    target_names = None
    if label_names:
        labels = sorted(label_names.keys())
        target_names = [label_names[i] for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return results, report, y_pred, cm


def results_to_dict(results: EvalResults) -> Dict[str, Any]:
    """Convert EvalResults dataclass to a plain dict."""
    return asdict(results)


def save_text(path: str | Path, text: str) -> None:
    """Save text to a file, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a dict to JSON, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: str | Path,
    class_names: Optional[list[str]] = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 200,
) -> None:
    """
    Plot and save confusion matrix as PNG.

    normalize=True shows row-normalized percentages (recall per class).
    """
    import matplotlib.pyplot as plt

    cm_to_plot = cm.astype(np.float64)
    if normalize and cm_to_plot.size > 0:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(
            cm_to_plot,
            row_sums,
            out=np.zeros_like(cm_to_plot),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_to_plot, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))

    if class_names is not None and len(class_names) == cm_to_plot.shape[0]:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)

    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def evaluate_and_save(
    model: Any,
    vect_name: str,
    X: Any,
    y_true: np.ndarray,
    out_dir: str | Path = "../reports",
    label_names: Optional[Dict[int, str]] = None,
    normalize_cm: bool = True,
) -> Tuple[EvalResults, str, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: evaluate + save metrics/report/confusion-matrix to out_dir.

    Writes:
      - {out_dir}/metrics/metrics.json
      - {out_dir}/metrics/classification_report.txt
      - {out_dir}/figures/confusion_matrix.png
    """
    results, report, y_pred, cm = evaluate_model(
        model, X, y_true, label_names=label_names
    )

    out_dir = Path(out_dir)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    save_json(metrics_dir / f"metrics_{vect_name}.json", results_to_dict(results))
    save_text(metrics_dir / f"classification_report_{vect_name}.txt", report)

    class_names = None
    if label_names:
        class_names = [label_names[i] for i in sorted(label_names.keys())]

    plot_confusion_matrix(
        cm,
        figures_dir / f"confusion_matrix_{vect_name}.png",
        class_names=class_names,
        normalize=normalize_cm,
    )

    return results, report, y_pred, cm