import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from utils import (
    SEED,
    COLOR_CLASS_0,
    COLOR_CLASS_1,
    load_dataset,
    evaluate_model,
    print_metrics_summary,
    plot_decision_boundary,
    style_axis,
)


def compute_margin_width(model) -> float:
    """
    Compute the geometric margin width 2 / ||w|| for a linear binary classifier.
    """
    w = model.coef_[0]
    norm_w = np.linalg.norm(w)
    return 2.0 / norm_w if norm_w > 0 else 0.0


def plot_boundary_grid(results: dict, save_path: str = "figures/part2_perceptron_vs_svm.png"):
    """
    Create a 2x2 grid comparing Perceptron vs. Linear SVM decision boundaries
    on both the Linear and Circular datasets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), dpi=200)
    fig.patch.set_facecolor("white")

    layout = [
        ("linear", "Perceptron", axes[0, 0], False, 0.06),
        ("linear", "Linear SVM", axes[0, 1], True, 0.06),
        ("circular", "Perceptron", axes[1, 0], False, 0.02),
        ("circular", "Linear SVM", axes[1, 1], True, 0.02),
    ]

    for ds_name, model_name, ax, show_margins, step in layout:
        entry = results[(ds_name, model_name)]
        model = entry["model"]
        X_test, y_test = entry["X_test"], entry["y_test"]
        metrics = entry["metrics"]

        title = (
            f"{ds_name.capitalize()} Dataset — {model_name}\n"
            f"Test Acc: {metrics['test_acc']:.2%} | F1: {metrics['test_f1']:.3f} | AUC: {metrics['test_auc']:.3f}"
        )
        plot_decision_boundary(
            ax=ax,
            model=model,
            X=X_test,
            y=y_test,
            title=title,
            max_points=2000,
            grid_step=step,
            show_margins=show_margins,
        )
        if ds_name == "circular":
            ax.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_0, markersize=8, label="Class 0"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_1, markersize=8, label="Class 1"),
        plt.Line2D([0], [0], color="#111111", lw=1.8, linestyle="-", label="Decision Boundary"),
        plt.Line2D([0], [0], color="#333333", lw=1.1, linestyle="--", label="SVM Margin (±1)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    print(f"\nSaved decision boundary plot to: {save_path}")
    plt.close(fig)


def plot_metric_bars(results: dict, save_path: str = "figures/part2_metrics_comparison.png"):
    """
    Bar chart comparing Test Accuracy and ROC-AUC of Perceptron vs. Linear SVM
    across both datasets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    categories = ["Linear Dataset\n(Accuracy)", "Linear Dataset\n(ROC-AUC)", "Circular Dataset\n(Accuracy)", "Circular Dataset\n(ROC-AUC)"]
    perc_vals = [
        results[("linear", "Perceptron")]["metrics"]["test_acc"],
        results[("linear", "Perceptron")]["metrics"]["test_auc"],
        results[("circular", "Perceptron")]["metrics"]["test_acc"],
        results[("circular", "Perceptron")]["metrics"]["test_auc"],
    ]
    svm_vals = [
        results[("linear", "Linear SVM")]["metrics"]["test_acc"],
        results[("linear", "Linear SVM")]["metrics"]["test_auc"],
        results[("circular", "Linear SVM")]["metrics"]["test_acc"],
        results[("circular", "Linear SVM")]["metrics"]["test_auc"],
    ]

    x = np.arange(len(categories))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)
    fig.patch.set_facecolor("white")

    rects1 = ax.bar(x - width / 2, perc_vals, width, label="Perceptron", color="#D19A66", edgecolor="none")
    rects2 = ax.bar(x + width / 2, svm_vals, width, label="Linear SVM", color="#2B5C8F", edgecolor="none")

    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0, label="Random Chance (50%)")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Perceptron vs. Linear SVM Performance Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.12)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved metric bar chart to: {save_path}")
    plt.close(fig)


def main():
    results = {}

    for ds_name in ["linear", "circular"]:
        X_train, X_test, y_train, y_test = load_dataset(ds_name)

        models = {
            "Perceptron": Perceptron(max_iter=1000, tol=1e-3, random_state=SEED),
            "Linear SVM": SVC(kernel="linear", C=1.0, random_state=SEED),
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            print_metrics_summary(model_name, ds_name, metrics)

            if ds_name == "linear":
                margin = compute_margin_width(model)
                n_sv = len(model.support_) if hasattr(model, "support_") else "N/A"
                print(f"  Geometric Margin Width (2/||w||): {margin:.4f} | Support Vectors: {n_sv}")

            results[(ds_name, model_name)] = {
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "metrics": metrics,
            }

    plot_boundary_grid(results)
    plot_metric_bars(results)


if __name__ == "__main__":
    main()