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
    plot_decision_boundary,
    get_decision_scores,
    style_axis,
)

# Augmentation parameters (k >> 1000 as required)
K_ITERATIONS = 1500
SAMPLE_SIZE = 1000
LOW_SD = 0.06
HIGH_SD = 0.35
BONUS_THRESHOLD = 0.70


def run_monte_carlo_augmentation(
    model,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    noise_sd: float,
    k: int = K_ITERATIONS,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
):
    """
    Repeat k times:
      a. Sample 1,000 datapoints from X_pool
      b. Add isotropic 2D Gaussian noise N(0, noise_sd^2 * I)
      c. Evaluate the fitted SVM model (Accuracy, F1, ROC-AUC)
    """
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)

    accs = np.empty(k)
    f1s = np.empty(k)
    aucs = np.empty(k)

    example_X, example_y = None, None

    for i in range(k):
        idx = rng.choice(n_pool, size=sample_size, replace=False)
        X_sub = X_pool[idx]
        y_sub = y_pool[idx]

        # Shift each point in a random direction drawn from a 2D Gaussian
        noise = rng.normal(loc=0.0, scale=noise_sd, size=X_sub.shape)
        X_aug = X_sub + noise

        y_pred = model.predict(X_aug)
        scores = get_decision_scores(model, X_aug)

        accs[i] = accuracy_score(y_sub, y_pred)
        f1s[i] = f1_score(y_sub, y_pred, zero_division=0)
        aucs[i] = roc_auc_score(y_sub, scores)

        if i == 0:
            example_X, example_y = X_aug, y_sub

    return {
        "noise_sd": noise_sd,
        "acc": accs,
        "f1": f1s,
        "auc": aucs,
        "example_X": example_X,
        "example_y": example_y,
    }


def examine_training_overfitting(X_train, y_train, X_test, y_test, seed: int = SEED):
    """
    Demonstrate overfitting directly by fitting an RBF SVM on noisy augmented data
    and comparing its augmented training accuracy vs. clean test accuracy.
    """
    rng = np.random.default_rng(seed + 99)
    idx = rng.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
    X_aug_train = X_train[idx] + rng.normal(0.0, HIGH_SD, size=(SAMPLE_SIZE, 2))
    y_aug_train = y_train[idx]

    # High-capacity RBF SVM allowed to fit the noisy points closely
    overfit_svm = SVC(kernel="rbf", C=10.0, gamma=2.0, random_state=seed)
    overfit_svm.fit(X_aug_train, y_aug_train)

    aug_train_acc = accuracy_score(y_aug_train, overfit_svm.predict(X_aug_train))
    clean_test_acc = accuracy_score(y_test, overfit_svm.predict(X_test))

    print("\n[TRAINING OVERFITTING CHECK — Fitting RBF SVM on High-SD Augmented Data]")
    print(f"  Augmented Train Accuracy (sigma={HIGH_SD}): {aug_train_acc:.4f}")
    print(f"  Clean Test Accuracy (unseen clean data) : {clean_test_acc:.4f}")
    print(f"  Overfitting Generalization Gap          : {aug_train_acc - clean_test_acc:+.4f}")


def plot_augmented_scatters(
    model,
    X_clean_sub,
    y_clean_sub,
    low_res: dict,
    high_res: dict,
    save_path: str = "figures/part4_augmentation_scatter.png",
):
    """
    Visualize 1,000-point samples under Original, Low SD, and High SD augmentations
    with the fitted RBF SVM decision boundary overlaid.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), dpi=200)
    fig.patch.set_facecolor("white")

    panels = [
        ("Original Sample (SD = 0.00)", X_clean_sub, y_clean_sub),
        (f"Low-SD Augmentation (SD = {LOW_SD:.2f})", low_res["example_X"], low_res["example_y"]),
        (f"High-SD Augmentation (SD = {HIGH_SD:.2f})", high_res["example_X"], high_res["example_y"]),
    ]

    for ax, (label, X_p, y_p) in zip(axes, panels):
        acc_val = accuracy_score(y_p, model.predict(X_p))
        f1_val = f1_score(y_p, model.predict(X_p))
        title = f"{label}\nSample Acc: {acc_val:.2%} | F1: {f1_val:.3f}"

        plot_decision_boundary(
            ax=ax,
            model=model,
            X=X_p,
            y=y_p,
            title=title,
            max_points=1000,
            grid_step=0.025,
            show_margins=False,
        )
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_0, markersize=8, label="Class 0 (Outer)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLASS_1, markersize=8, label="Class 1 (Inner)"),
        plt.Line2D([0], [0], color="#111111", lw=1.8, linestyle="-", label="Fitted RBF Boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=200)
    print(f"\nSaved augmentation scatter plots to: {save_path}")
    plt.close(fig)


def plot_metric_distributions(
    low_res: dict,
    high_res: dict,
    save_path: str = "figures/part4_augmentation_distributions.png",
):
    """
    Plot the distribution of Accuracy, F1-Score, and ROC-AUC across all k = 1,500
    Monte Carlo augmented datasets for Low SD vs. High SD.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=200)
    fig.patch.set_facecolor("white")

    metrics_info = [
        ("acc", "Accuracy"),
        ("f1", "F1-Score"),
        ("auc", "ROC-AUC"),
    ]

    for ax, (key, label) in zip(axes, metrics_info):
        low_vals = low_res[key]
        high_vals = high_res[key]

        ax.hist(low_vals, bins=30, alpha=0.65, color="#2B5C8F", edgecolor="white", label=f"Low SD ({LOW_SD}): {low_vals.mean():.3f} ± {low_vals.std():.3f}")
        ax.hist(high_vals, bins=30, alpha=0.65, color="#E06C75", edgecolor="white", label=f"High SD ({HIGH_SD}): {high_vals.mean():.3f} ± {high_vals.std():.3f}")

        ax.axvline(low_vals.mean(), color="#1B3B5C", linestyle="--", lw=1.4)
        ax.axvline(high_vals.mean(), color="#A83842", linestyle="--", lw=1.4)

        ax.set_title(f"{label} Across k={K_ITERATIONS} Runs", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Frequency (out of 1,500)", fontsize=10)
        style_axis(ax)
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved augmentation metric distributions to: {save_path}")
    plt.close(fig)


def run_bonus_threshold_analysis(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = BONUS_THRESHOLD,
    save_path: str = "figures/bonus_threshold_curve.png",
):
    """
    Bonus: Sweep Gaussian augmentation SD to find the exact noise level where
    the SVM accuracy drops below the 70% performance threshold.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sd_grid = np.linspace(0.0, 0.60, 25)
    mean_accs = []
    std_accs = []

    for sd in sd_grid:
        if sd == 0.0:
            res = run_monte_carlo_augmentation(model, X_test, y_test, noise_sd=0.0, k=200, seed=SEED)
        else:
            res = run_monte_carlo_augmentation(model, X_test, y_test, noise_sd=sd, k=200, seed=SEED)
        mean_accs.append(res["acc"].mean())
        std_accs.append(res["acc"].std())

    mean_accs = np.array(mean_accs)
    std_accs = np.array(std_accs)

    # Linear interpolation to find exact crossing point below threshold (0.70)
    crossing_sd = None
    for i in range(len(sd_grid) - 1):
        if mean_accs[i] >= threshold and mean_accs[i + 1] < threshold:
            frac = (mean_accs[i] - threshold) / (mean_accs[i] - mean_accs[i + 1])
            crossing_sd = sd_grid[i] + frac * (sd_grid[i + 1] - sd_grid[i])
            break

    print(f"\n[BONUS — {threshold:.0%} PERFORMANCE THRESHOLD ANALYSIS]")
    if crossing_sd is not None:
        print(f"  SVM Accuracy drops below {threshold:.0%} at Gaussian Noise SD = {crossing_sd:.4f}")
    else:
        print(f"  SVM Accuracy did not cross {threshold:.0%} within the tested SD range.")

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=200)
    fig.patch.set_facecolor("white")

    ax.plot(sd_grid, mean_accs, color="#2B5C8F", lw=2.2, marker="o", markersize=4.5, label="Mean Test Accuracy (200 runs/SD)")
    ax.fill_between(sd_grid, mean_accs - std_accs, mean_accs + std_accs, color="#2B5C8F", alpha=0.18, label="±1 SD Band")
    ax.axhline(threshold, color="#E06C75", linestyle="--", lw=1.6, label=f"Performance Threshold ({threshold:.0%})")

    if crossing_sd is not None:
        ax.axvline(crossing_sd, color="#333333", linestyle=":", lw=1.5)
        ax.scatter([crossing_sd], [threshold], color="#E06C75", s=60, zorder=5)
        ax.annotate(
            f"Threshold Crossing\nSD = {crossing_sd:.3f}",
            xy=(crossing_sd, threshold),
            xytext=(crossing_sd + 0.05, threshold + 0.06),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title("Bonus: SVM Generalization Decay vs. Augmentation Noise SD", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Gaussian Augmentation Noise Standard Deviation (SD)", fontsize=10.5)
    ax.set_ylabel("Classification Accuracy", fontsize=10.5)
    ax.set_ylim(0.45, 1.0)
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved bonus threshold plot to: {save_path}")
    plt.close(fig)


def main():
    X_train, X_test, y_train, y_test = load_dataset("circular")

    # Fit the best-performing SVM model from Part 3 (RBF Kernel)
    best_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    best_svm.fit(X_train, y_train)

    print(f"Running Monte Carlo Data Augmentation (k = {K_ITERATIONS}, n = {SAMPLE_SIZE})...")
    low_res = run_monte_carlo_augmentation(best_svm, X_test, y_test, noise_sd=LOW_SD)
    high_res = run_monte_carlo_augmentation(best_svm, X_test, y_test, noise_sd=HIGH_SD)

    for label, res in [("LOW SD", low_res), ("HIGH SD", high_res)]:
        print(f"\n[{label} AUGMENTATION (sigma = {res['noise_sd']:.2f}, k = {K_ITERATIONS})]")
        print(f"  Accuracy : {res['acc'].mean():.4f} ± {res['acc'].std():.4f}")
        print(f"  F1-Score : {res['f1'].mean():.4f} ± {res['f1'].std():.4f}")
        print(f"  ROC-AUC  : {res['auc'].mean():.4f} ± {res['auc'].std():.4f}")

    # Clean 1,000-point baseline reference sample
    rng = np.random.default_rng(SEED)
    base_idx = rng.choice(len(X_test), size=SAMPLE_SIZE, replace=False)

    plot_augmented_scatters(best_svm, X_test[base_idx], y_test[base_idx], low_res, high_res)
    plot_metric_distributions(low_res, high_res)

    # Examine training overfitting directly
    examine_training_overfitting(X_train, y_train, X_test, y_test)

    # Bonus: Find exact 70% accuracy threshold
    run_bonus_threshold_analysis(best_svm, X_test, y_test)


if __name__ == "__main__":
    main()