import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from typing import Iterable, Tuple, Union, Dict, Any, Optional

EstimatorOrProba = Union[Any, np.ndarray]  # model/pipe with predict_proba OR precomputed probs

def _proba1(model_or_proba: EstimatorOrProba, X) -> np.ndarray:
    if isinstance(model_or_proba, np.ndarray):
        p1 = np.asarray(model_or_proba).ravel()
    elif hasattr(model_or_proba, "predict_proba"):
        p1 = model_or_proba.predict_proba(X)[:, 1]
    elif hasattr(model_or_proba, "decision_function"):
        s = model_or_proba.decision_function(X).astype(float)
        p1 = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        raise ValueError("Pass an estimator with predict_proba/decision_function or a 1D proba array.")
    if p1.shape[0] != len(X):
        raise ValueError("Length of probabilities does not match X.")
    return p1

def plot_multi_curves(
    models: Iterable[Tuple[str, EstimatorOrProba]],
    X,
    y: np.ndarray,
    *,
    titles: Tuple[str, str] = ("ROC — Class 1 (Fit)", "ROC — Class 0 (Not Fit)"),
    palette: str = "tab10",
    figsize: Tuple[int, int] = (10, 4),
    curve: str = "roc",  # "roc" or "pr"
    thresholds: Optional[Dict[str, float]] = None,  # e.g., {"Baseline": 0.5, "Best Pipe": best_thr}
) -> Dict[str, Dict[str, float]]:
    """
    Plot ROC or Precision–Recall curves for multiple models on two panels (Class 1 & Class 0).
    Returns dict of scores per model: {'label': {'auc1':..., 'auc0':..., 'ap1':..., 'ap0':...}}
    """
    models = list(models)
    if not models:
        raise ValueError("No models provided.")

    y = np.asarray(y).astype(int)
    y0 = (y == 0).astype(int)
    thresholds = thresholds or {}

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False, sharey=False)
    colors = sns.color_palette(palette, n_colors=len(models))

    scores: Dict[str, Dict[str, float]] = {}

    for (label, est), color in zip(models, colors):
        p1 = _proba1(est, X)
        p0 = 1.0 - p1

        if curve.lower() == "roc":
            # Class 1
            fpr1, tpr1, thr1 = roc_curve(y, p1)
            auc1 = roc_auc_score(y, p1)
            sns.lineplot(x=fpr1, y=tpr1, ax=axes[0], label=f"{label} (AUC={auc1:.3f})",
                         color=color, lw=2.2)

            # mark threshold if provided
            if label in thresholds:
                thr = thresholds[label]
                # find nearest ROC point to that threshold
                idx = np.argmin(np.abs(thr1 - thr))
                axes[0].plot(fpr1[idx], tpr1[idx], "o", color=color, ms=6)

            # Class 0
            fpr0, tpr0, thr0 = roc_curve(y0, p0)
            auc0 = roc_auc_score(y0, p0)  # will equal auc1 in binary case
            sns.lineplot(x=fpr0, y=tpr0, ax=axes[1], label=f"{label} (AUC={auc0:.3f})",
                         color=color, lw=2.2)

            if label in thresholds:
                thr = thresholds[label]
                # threshold on p0 is (1 - thr_on_p1)
                idx0 = np.argmin(np.abs(thr0 - (1.0 - thr)))
                axes[1].plot(fpr0[idx0], tpr0[idx0], "o", color=color, ms=6)

        elif curve.lower() == "pr":
            # Class 1 PR
            prec1, rec1, _ = precision_recall_curve(y, p1)
            ap1 = average_precision_score(y, p1)
            sns.lineplot(x=rec1, y=prec1, ax=axes[0], label=f"{label} (AP={ap1:.3f})",
                         color=color, lw=2.2)
            axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")

            # Class 0 PR
            prec0, rec0, _ = precision_recall_curve(y0, p0)
            ap0 = average_precision_score(y0, p0)
            sns.lineplot(x=rec0, y=prec0, ax=axes[1], label=f"{label} (AP={ap0:.3f})",
                         color=color, lw=2.2)
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        else:
            raise ValueError("curve must be 'roc' or 'pr'.")

        # store scores (both AUC and AP if available)
        scores[label] = {
            "auc1": float(roc_auc_score(y, p1)),
            "auc0": float(roc_auc_score(y0, p0)),
            "ap1": float(average_precision_score(y, p1)),
            "ap0": float(average_precision_score(y0, p0)),
        }

    # finalize axes
    for ax, ttl in zip(axes, titles):
        if curve.lower() == "roc":
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Chance")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(ttl, fontsize=12)
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()
    return scores
