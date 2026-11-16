import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Shared project constants
SEED = 42
COLOR_CLASS_0 = "#E06C75"  # Coral red
COLOR_CLASS_1 = "#2B5C8F"  # Deep steel blue


def load_dataset(name: str, test_size: float = 0.2, seed: int = SEED):
    """
    Load a simulated dataset ('linear' or 'circular') from the data/ directory
    and return a reproducible stratified train-test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    valid_names = {"linear": "data/linear_dataset.npz", "circular": "data/circular_dataset.npz"}
    if name not in valid_names:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(valid_names.keys())}.")

    path = valid_names[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file '{path}' not found. Run 01_simulate_data.py first.")

    data = np.load(path)
    X, y = data["X"], data["y"]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )


def get_decision_scores(model, X):
    """
    Extract continuous decision scores or positive-class probabilities for ROC-AUC.
    """
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Compute Train and Test metrics (Accuracy, F1, ROC-AUC, Generalization Gap,
    and Test Confusion Matrix) for a fitted binary classifier.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_scores = get_decision_scores(model, X_train)
    test_scores = get_decision_scores(model, X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gen_gap": train_acc - test_acc,
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "train_auc": roc_auc_score(y_train, train_scores),
        "test_auc": roc_auc_score(y_test, test_scores),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
    }


def print_metrics_summary(model_name: str, dataset_name: str, metrics: dict):
    """
    Print a formatted summary of Train vs. Test performance metrics.
    """
    print(f"\n[{dataset_name.upper()} | {model_name}]")
    print(f"  Train Accuracy : {metrics['train_acc']:.4f} | Test Accuracy : {metrics['test_acc']:.4f} | Gap: {metrics['gen_gap']:+.4f}")
    print(f"  Train F1-Score : {metrics['train_f1']:.4f} | Test F1-Score : {metrics['test_f1']:.4f}")
    print(f"  Train ROC-AUC  : {metrics['train_auc']:.4f} | Test ROC-AUC  : {metrics['test_auc']:.4f}")
    print(f"  Test Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"  Test Precision : {metrics['test_precision']:.4f} | Test Recall   : {metrics['test_recall']:.4f} | Test F1: {metrics['test_f1']:.4f}")


def style_axis(ax):
    """
    Apply consistent clean spine and grid styling to a Matplotlib axis.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(True, linestyle=":", alpha=0.35, color="#888888")


def plot_decision_boundary(
    ax,
    model,
    X,
    y,
    title: str,
    max_points: int = 2000,
    grid_step: float = 0.03,
    show_margins: bool = False,
    seed: int = SEED,
):
    """
    Plot the 2D decision regions, decision boundary (and optional SVM margins),
    and scatter points on a given Matplotlib axis.
    """
    pad = 0.6
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Use decision_function for smooth contours when available
    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=[COLOR_CLASS_0, COLOR_CLASS_1], alpha=0.14)
        ax.contour(xx, yy, Z, levels=[0], colors=["#111111"], linewidths=1.8, linestyles="-")
        if show_margins:
            ax.contour(xx, yy, Z, levels=[-1, 1], colors=["#333333"], linewidths=1.1, linestyles="--")
    else:
        Z = model.predict(grid_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[COLOR_CLASS_0, COLOR_CLASS_1], alpha=0.14)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#111111"], linewidths=1.8, linestyles="-")

    # Subsample points if dataset is large so the decision boundary stays clearly visible
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X_plot, y_plot = X[idx], y[idx]
    else:
        X_plot, y_plot = X, y

    ax.scatter(
        X_plot[y_plot == 0, 0], X_plot[y_plot == 0, 1],
        c=COLOR_CLASS_0, s=10, alpha=0.5, edgecolors="none", label="Class 0"
    )
    ax.scatter(
        X_plot[y_plot == 1, 0], X_plot[y_plot == 1, 1],
        c=COLOR_CLASS_1, s=10, alpha=0.5, edgecolors="none", label="Class 1"
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("$X_1$", fontsize=10)
    ax.set_ylabel("$X_2$", fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    style_axis(ax)