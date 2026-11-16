import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import SEED, get_decision_scores, style_axis

# Baseline 2D + three higher dimensions required by Part 6
DIMENSIONS = [2, 5, 10, 15]
NOISE_LEVELS = [0.0, 0.15, 0.25, 0.50]
N_SAMPLES = 10000
N_TRIALS = 5


def generate_hd_linear(n_samples: int, n_dim: int, seed: int = SEED):
    """
    Generate two D-dimensional Gaussian clusters centered at (-3,..., -3)
    and (+3, ..., +3) with per-coordinate SD = 0.8.
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    X0 = rng.normal(loc=-3.0, scale=0.8, size=(n_half, n_dim))
    X1 = rng.normal(loc=3.0, scale=0.8, size=(n_samples - n_half, n_dim))

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_half, dtype=int), np.ones(n_samples - n_half, dtype=int)])

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def generate_hd_hyperspheres(n_samples: int, n_dim: int, factor: float = 0.60, noise: float = 0.14, seed: int = SEED):
    """
    Generalize concentric circles to D-dimensional concentric hyperspheres:
      - Class 0: points uniformly sampled on the unit (D-1)-sphere (radius = 1.0)
      - Class 1: points uniformly sampled on the inner (D-1)-sphere (radius = 0.60)
      - Isotropic D-dimensional Gaussian noise N(0, noise^2 * I_D) added to all coordinates.
    """
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Sample uniformly on unit hypersphere via normalized standard Gaussians
    raw_outer = rng.normal(0.0, 1.0, size=(n_outer, n_dim))
    X_outer = (raw_outer / np.linalg.norm(raw_outer, axis=1, keepdims=True)) * 1.0
    X_outer += rng.normal(0.0, noise, size=(n_outer, n_dim))

    raw_inner = rng.normal(0.0, 1.0, size=(n_inner, n_dim))
    X_inner = (raw_inner / np.linalg.norm(raw_inner, axis=1, keepdims=True)) * factor
    X_inner += rng.normal(0.0, noise, size=(n_inner, n_dim))

    X = np.vstack([X_outer, X_inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def corrupt_labels(y: np.ndarray, noise_pct: float, rng: np.random.Generator) -> np.ndarray:
    y_noisy = y.copy()
    n_flip = int(round(noise_pct * len(y_noisy)))
    if n_flip > 0:
        flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


def evaluate_dimensionality_experiment(dataset_type: str, kernel: str):
    """
    Evaluate SVM performance across D in [2, 5, 10, 15] and label noise levels
    [0%, 15%, 25%, 50%] to examine how dimensionality affects classification and
    noise sensitivity.
    """
    results = {
        "dims": DIMENSIONS,
        "clean_acc": [],
        "clean_f1": [],
        "clean_auc": [],
        "noise_acc": {pct: [] for pct in NOISE_LEVELS},
        "noise_std": {pct: [] for pct in NOISE_LEVELS},
    }

    for dim in DIMENSIONS:
        if dataset_type == "linear":
            X, y = generate_hd_linear(N_SAMPLES, dim, seed=SEED)
        else:
            X, y = generate_hd_hyperspheres(N_SAMPLES, dim, seed=SEED)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )

        # 1. Clean evaluation (0% label noise)
        clean_svm = SVC(kernel=kernel, C=1.0, gamma="scale", random_state=SEED)
        clean_svm.fit(X_train, y_train)
        y_pred = clean_svm.predict(X_test)
        scores = get_decision_scores(clean_svm, X_test)

        results["clean_acc"].append(accuracy_score(y_test, y_pred))
        results["clean_f1"].append(f1_score(y_test, y_pred, zero_division=0))
        results["clean_auc"].append(roc_auc_score(y_test, scores))

        # 2. Label noise sensitivity across dimensions (connecting Part 5 -> Part 6)
        for pct in NOISE_LEVELS:
            trial_accs = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(SEED + dim * 100 + int(pct * 1000) + trial)
                y_tr_noisy = corrupt_labels(y_train, pct, rng)

                svm = SVC(kernel=kernel, C=1.0, gamma="scale", random_state=SEED)
                svm.fit(X_train, y_tr_noisy)
                trial_accs.append(accuracy_score(y_test, svm.predict(X_test)))

            results["noise_acc"][pct].append(np.mean(trial_accs))
            results["noise_std"][pct].append(np.std(trial_accs))

    return results


def print_hd_summary(dataset_type: str, kernel: str, res: dict):
    print(f"\n[{dataset_type.upper()} DATASET ({kernel.upper()} SVM) — HIGH DIMENSIONALITY ANALYSIS]")
    print(f"  {'Dim (D)':<8} | {'Clean Acc':<10} | {'Clean F1':<10} | {'Clean AUC':<10} | {'15% Noise':<11} | {'25% Noise':<11} | {'50% Noise':<11}")
    print("  " + "-" * 86)

    for i, dim in enumerate(res["dims"]):
        print(
            f"  {dim:<8} | "
            f"{res['clean_acc'][i]:<10.4f} | "
            f"{res['clean_f1'][i]:<10.4f} | "
            f"{res['clean_auc'][i]:<10.4f} | "
            f"{res['noise_acc'][0.15][i]:<11.4f} | "
            f"{res['noise_acc'][0.25][i]:<11.4f} | "
            f"{res['noise_acc'][0.50][i]:<11.4f}"
        )


def plot_high_dimensionality(lin_res: dict, sph_res: dict, save_path: str = "figures/part6_high_dimensionality.png"):
    """
    Create a 2x2 figure showing:
      - Top row: Clean SVM performance (Accuracy, F1, ROC-AUC) vs. Dimensionality (D = 2, 5, 10, 15)
      - Bottom row: Clean Test Accuracy vs. Dimensionality under 0%, 15%, 25%, and 50% training label noise
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), dpi=200)
    fig.patch.set_facecolor("white")

    dims = DIMENSIONS

    # --- Top Row: Clean Performance vs. Dimension ---
    for ax, res, title in [
        (axes[0, 0], lin_res, "Linear Dataset (Linear SVM)\nClean Performance vs. Dimensionality"),
        (axes[0, 1], sph_res, "Concentric Hyperspheres (RBF SVM)\nClean Performance vs. Dimensionality"),
    ]:
        ax.plot(dims, res["clean_acc"], color="#2B5C8F", lw=2.2, marker="o", markersize=6, label="Test Accuracy")
        ax.plot(dims, res["clean_f1"], color="#D19A66", lw=2.0, linestyle="--", marker="s", markersize=5.5, label="Test F1-Score")
        ax.plot(dims, res["clean_auc"], color="#2E8B57", lw=2.0, linestyle="-.", marker="^", markersize=6, label="Test ROC-AUC")

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Feature Dimensionality ($D$)", fontsize=10.5)
        ax.set_ylabel("Score", fontsize=10.5)
        ax.set_xticks(dims)
        ax.set_ylim(0.45, 1.05)
        style_axis(ax)
        ax.legend(loc="lower left", frameon=True, fontsize=9)

    # --- Bottom Row: Label Noise Sensitivity Across Dimensions ---
    noise_colors = {
        0.0: "#2B5C8F",
        0.15: "#2E8B57",
        0.25: "#D19A66",
        0.50: "#E06C75",
    }

    for ax, res, title in [
        (axes[1, 0], lin_res, "Linear Dataset (Linear SVM)\nLabel Noise Effect Across Dimensions"),
        (axes[1, 1], sph_res, "Concentric Hyperspheres (RBF SVM)\nLabel Noise Effect Across Dimensions"),
    ]:
        for pct in NOISE_LEVELS:
            means = np.array(res["noise_acc"][pct])
            stds = np.array(res["noise_std"][pct])
            c = noise_colors[pct]
            ax.plot(dims, means, color=c, lw=2.1, marker="o", markersize=5.5, label=f"Label Noise: {pct:.0%}")
            ax.fill_between(dims, means - stds, means + stds, color=c, alpha=0.14)

        ax.axhline(0.50, color="#666666", linestyle=":", lw=1.1, label="Random Chance (50%)")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Feature Dimensionality ($D$)", fontsize=10.5)
        ax.set_ylabel("Clean Test Accuracy", fontsize=10.5)
        ax.set_xticks(dims)
        ax.set_ylim(0.42, 1.05)
        style_axis(ax)
        ax.legend(loc="lower left", frameon=True, fontsize=8.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\nSaved high dimensionality plot to: {save_path}")
    plt.close(fig)


def main():
    print("Running High Dimensionality Experiments (D = 2, 5, 10, 15)...")
    lin_res = evaluate_dimensionality_experiment("linear", kernel="linear")
    print_hd_summary("linear", "linear", lin_res)

    sph_res = evaluate_dimensionality_experiment("hyperspheres", kernel="rbf")
    print_hd_summary("hyperspheres", "rbf", sph_res)

    plot_high_dimensionality(lin_res, sph_res)


if __name__ == "__main__":
    main()