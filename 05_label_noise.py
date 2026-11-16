import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import (
    SEED,
    COLOR_CLASS_0,
    COLOR_CLASS_1,
    load_dataset,
    get_decision_scores,
    plot_decision_boundary,
    style_axis,
)

# Required noise percentages + intermediate points for a smooth curve
HIGHLIGHT_NOISE_LEVELS = [0.0, 0.15, 0.25, 0.50]
ALL_NOISE_LEVELS = [0.0, 0.05, 0.15, 0.25, 0.35, 0.50]
N_TRIALS = 10


def corrupt_training_labels(y_train: np.ndarray, noise_pct: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate annotation mistakes by flipping a exact percentage (`noise_pct`)
    of binary training labels (0 <-> 1). Test labels remain untouched.
    """
    y_noisy = y_train.copy()
    n_samples = len(y_noisy)
    n_flip = int(round(noise_pct * n_samples))

    if n_flip > 0:
        flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

    return y_noisy


def evaluate_noise_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str,
    noise_levels: list = ALL_NOISE_LEVELS,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
):
    """
    Train an SVM across multiple label-corruption percentages and evaluate
    both noisy training accuracy and clean test generalization metrics.
    """
    summary = {
        "noise_pct": np.array(noise_levels) * 100.0,
        "train_acc_mean": [], "train_acc_std": [],
        "test_acc_mean": [],  "test_acc_std": [],
        "test_f1_mean": [],   "test_f1_std": [],
        "test_auc_mean": [],  "test_auc_std": [],
        "example_models": {},
    }

    for pct in noise_levels:
        train_accs, test_accs, test_f1s, test_aucs = [], [], [], []

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + int(pct * 1000) + trial)
            y_train_noisy = corrupt_training_labels(y_train, pct, rng)

            svm = SVC(kernel=kernel, C=1.0, gamma="scale", random_state=seed)
            svm.fit(X_train, y_train_noisy)

            train_accs.append(accuracy_score(y_train_noisy, svm.predict(X_train)))
            y_test_pred = svm.predict(X_test)
            test_scores = get_decision_scores(svm, X_test)

            test_accs.append(accuracy_score(y_test, y_test_pred))
            test_f1s.append(f1_score(y_test, y_test_pred, zero_division=0))
            test_aucs.append(roc_auc_score(y_test, test_scores))

            if trial == 0 and pct in HIGHLIGHT_NOISE_LEVELS:
                summary["example_models"][pct] = (svm, y_train_noisy)

        summary["train_acc_mean"].append(np.mean(train_accs))
        summary["train_acc_std"].append(np.std(train_accs))
        summary["test_acc_mean"].append(np.mean(test_accs))
        summary["test_acc_std"].append(np.std(test_accs))
        summary["test_f1_mean"].append(np.mean(test_f1s))
        summary["test_f1_std"].append(np.std(test_f1s))
        summary["test_auc_mean"].append(np.mean(test_aucs))
        summary["test_auc_std"].append(np.std(test_aucs))

    for key in list(summary.keys()):
        if key not in {"noise_pct", "example_models"}:
            summary[key] = np.array(summary[key])

    return summary


def print_noise_table(dataset_name: str, kernel: str, summary: dict):
    """
    Print formatted metrics for the highlighted label-noise levels (0%, 15%, 25%, 50%).
    """
    print(f"\n[{dataset_name.upper()} DATASET ({kernel.upper()} SVM) — LABEL NOISE ANALYSIS]")
    print(f"  {'Label Noise':<13} | {'Noisy Train Acc':<17} | {'Clean Test Acc':<17} | {'Clean Test F1':<15} | {'Clean ROC-AUC':<15}")
    print("  " + "-" * 86)

    for i, pct_val in enumerate(summary["noise_pct"]):
        if round(pct_val / 100.0, 2) in HIGHLIGHT_NOISE_LEVELS:
            tr_m, tr_s = summary["train_acc_mean"][i], summary["train_acc_std"][i]
            te_m, te_s = summary["test_acc_mean"][i], summary["test_acc_std"][i]
            f1_m, f1_s = summary["test_f1_mean"][i], summary["test_f1_std"][i]
            au_m, au_s = summary["test_auc_mean"][i], summary["test_auc_std"][i]
            print(
                f"  {pct_val:>5.0f}%        | "
                f"{tr_m:.4f} ± {tr_s:.4f}   | "
                f"{te_m:.4f} ± {te_s:.4f}   | "
                f"{f1_m:.4f} ± {f1_s:.4f} | "
                f"{au_m:.4f} ± {au_s:.4f}"
            )


def plot_noise_curves(lin_sum: dict, circ_sum: dict, save_path: str = "figures/part5_label_noise_curves.png"):
    """
    Plot SVM performance as a function of training label noise percentage
    for both the Linear and Circular datasets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    datasets_info = [
        ("Linear Dataset (Linear SVM)", lin_sum, axes[0]),
        ("Circular Dataset (RBF SVM)", circ_sum, axes[1]),
    ]

    for title, s, ax in datasets_info:
        x = s["noise_pct"]

        # Clean Test Accuracy
        ax.plot(x, s["test_acc_mean"], color="#2B5C8F", lw=2.2, marker="o", markersize=5.5, label="Clean Test Accuracy")
        ax.fill_between(x, s["test_acc_mean"] - s["test_acc_std"], s["test_acc_mean"] + s["test_acc_std"], color="#2B5C8F", alpha=0.15)

        # Clean Test ROC-AUC
        ax.plot(x, s["test_auc_mean"], color="#2E8B57", lw=2.0, linestyle="-.", marker="s", markersize=4.5, label="Clean Test ROC-AUC")

        # Noisy Training Accuracy (shows overfitting/memorization of flipped labels)
        ax.plot(x, s["train_acc_mean"], color="#E06C75", lw=1.8, linestyle="--", marker="^", markersize=5, label="Noisy Train Accuracy")

        # Highlight vertical guide lines at 15%, 25%, 50%
        for ref_pct in [15, 25, 50]:
            ax.axvline(ref_pct, color="#AAAAAA", linestyle=":", lw=1.0)

        ax.axhline(0.50, color="#666666", linestyle=":", lw=1.1, label="Random Chance (50%)")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Percentage of Shuffled Training Labels (%)", fontsize=10.5)
        ax.set_ylabel("Score", fontsize=10.5)
        ax.set_xticks([0, 15, 25, 35, 50])
        ax.set_xlim(-2, 52)
        ax.set_ylim(0.42, 1.04)
        style_axis(ax)
        ax.legend(loc="lower left", frameon=True, fontsize=8.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\nSaved label noise curves figure to: {save_path}")
    plt.close(fig)


def plot_noisy_boundaries(
    circ_sum: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "figures/part5_noisy_boundaries.png",
):
    """
    Visualize how the RBF SVM decision boundary on the Circular dataset
    distorts under 0%, 15%, 25%, and 50% training label noise.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.3), dpi=200)
    fig.patch.set_facecolor("white")

    for ax, pct in zip(axes, HIGHLIGHT_NOISE_LEVELS):
        model, _ = circ_sum["example_models"][pct]
        acc_clean = accuracy_score(y_test, model.predict(X_test))
        title = f"Label Noise: {pct:.0%}\nClean Test Acc: {acc_clean:.2%}"

        plot_decision_boundary(
            ax=ax,
            model=model,
            X=X_test,
            y=y_test,
            title=title,
            max_points=1200,
            grid_step=0.03,
            show_margins=False,
        )
        ax.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_0, markersize=8, label="Clean Class 0"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_1, markersize=8, label="Clean Class 1"),
        plt.Line2D([0], [0], color="#111111", lw=1.8, linestyle="-", label="Learned Decision Boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=200)
    print(f"Saved noisy decision boundaries figure to: {save_path}")
    plt.close(fig)


def main():
    # 1. Linear Dataset with Linear SVM
    X_lin_tr, X_lin_te, y_lin_tr, y_lin_te = load_dataset("linear")
    lin_summary = evaluate_noise_sweep(X_lin_tr, y_lin_tr, X_lin_te, y_lin_te, kernel="linear")
    print_noise_table("linear", "linear", lin_summary)

    # 2. Circular Dataset with RBF SVM
    X_circ_tr, X_circ_te, y_circ_tr, y_circ_te = load_dataset("circular")
    circ_summary = evaluate_noise_sweep(X_circ_tr, y_circ_tr, X_circ_te, y_circ_te, kernel="rbf")
    print_noise_table("circular", "rbf", circ_summary)

    # 3. Generate Figures
    plot_noise_curves(lin_summary, circ_summary)
    plot_noisy_boundaries(circ_summary, X_circ_tr, X_circ_te, y_circ_te)


if __name__ == "__main__":
    main()