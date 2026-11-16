import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

from utils import (
    SEED,
    COLOR_CLASS_0,
    COLOR_CLASS_1,
    load_dataset,
    evaluate_model,
    print_metrics_summary,
    plot_decision_boundary,
    get_decision_scores,
    style_axis,
)


def plot_kernel_boundaries(results: dict, save_path: str = "figures/part3_kernel_boundaries.png"):
    """
    Plot side-by-side 2D decision boundaries and margins for RBF vs. Polynomial kernels
    on the Circular dataset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=200)
    fig.patch.set_facecolor("white")

    kernel_names = ["RBF Kernel", "Polynomial Kernel (deg=3)"]
    for ax, name in zip(axes, kernel_names):
        entry = results[name]
        metrics = entry["metrics"]
        title = (
            f"Circular Dataset — {name}\n"
            f"Test Acc: {metrics['test_acc']:.2%} | F1: {metrics['test_f1']:.3f} | AUC: {metrics['test_auc']:.3f}"
        )
        plot_decision_boundary(
            ax=ax,
            model=entry["model"],
            X=entry["X_test"],
            y=entry["y_test"],
            title=title,
            max_points=2000,
            grid_step=0.02,
            show_margins=True,
        )
        ax.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_0, markersize=8, label="Class 0 (Outer)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_1, markersize=8, label="Class 1 (Inner)"),
        plt.Line2D([0], [0], color="#111111", lw=1.8, linestyle="-", label="Decision Boundary"),
        plt.Line2D([0], [0], color="#333333", lw=1.1, linestyle="--", label="SVM Margin (±1)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=200)
    print(f"\nSaved kernel boundary figure to: {save_path}")
    plt.close(fig)


def plot_evaluation_diagnostics(results: dict, save_path: str = "figures/part3_kernel_evaluation.png"):
    """
    Plot Confusion Matrices for both kernels alongside their overlaid ROC curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=200)
    fig.patch.set_facecolor("white")

    kernel_names = ["RBF Kernel", "Polynomial Kernel (deg=3)"]
    roc_colors = ["#2B5C8F", "#D19A66"]

    # --- Subplots 0 & 1: Confusion Matrices ---
    for idx, name in enumerate(kernel_names):
        ax = axes[idx]
        cm = results[name]["metrics"]["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix\n{name}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0 (Outer)", "Pred 1 (Inner)"], fontsize=9.5)
        ax.set_yticklabels(["True 0 (Outer)", "True 1 (Inner)"], fontsize=9.5)

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / cm.sum() * 100
                ax.text(
                    j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#111111",
                    fontsize=10, fontweight="bold",
                )

    # --- Subplot 2: Overlaid ROC Curves ---
    ax_roc = axes[2]
    for name, color in zip(kernel_names, roc_colors):
        entry = results[name]
        y_scores = get_decision_scores(entry["model"], entry["X_test"])
        fpr, tpr, _ = roc_curve(entry["y_test"], y_scores)
        auc_val = entry["metrics"]["test_auc"]
        ax_roc.plot(fpr, tpr, lw=2.0, color=color, label=f"{name} (AUC = {auc_val:.4f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=1.1, color="#888888", label="Random Chance (AUC = 0.50)")
    ax_roc.set_title("Test ROC Curve Comparison", fontsize=11, fontweight="bold", pad=10)
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    style_axis(ax_roc)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=8.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved evaluation diagnostics figure to: {save_path}")
    plt.close(fig)


def main():
    X_train, X_test, y_train, y_test = load_dataset("circular")

    kernels = {
        "RBF Kernel": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
        "Polynomial Kernel (deg=3)": SVC(kernel="poly", degree=3, coef0=1.0, C=1.0, gamma="scale", random_state=SEED),
    }

    results = {}
    for name, model in kernels.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        print_metrics_summary(name, "circular", metrics)
        print(f"  Number of Support Vectors: {len(model.support_)} (Class 0: {model.n_support_[0]}, Class 1: {model.n_support_[1]})")

        results[name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "metrics": metrics,
        }

    plot_kernel_boundaries(results)
    plot_evaluation_diagnostics(results)


if __name__ == "__main__":
    main()