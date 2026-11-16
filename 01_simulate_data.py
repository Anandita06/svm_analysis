import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

# Set fixed random seed for reproducibility
SEED = 42
N_SAMPLES = 10000


def generate_linear_dataset(n_samples: int = N_SAMPLES, seed: int = SEED):
    """
    Generate two well-separated 2D Gaussian clusters (no overlap).
    Returns:
        X: ndarray of shape (n_samples, 2) -> [X1, X2]
        y: ndarray of shape (n_samples,) -> binary labels {0, 1}
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=[(-3.0, -3.0), (3.0, 3.0)],
        cluster_std=0.8,
        random_state=seed,
    )
    return X, y


def generate_circular_dataset(n_samples: int = N_SAMPLES, seed: int = SEED):
    """
    Generate two concentric 2D circular clusters with slight boundary overlap
    (inner circle inside the outer ring).
    Returns:
        X: ndarray of shape (n_samples, 2) -> [X1, X2]
        y: ndarray of shape (n_samples,) -> binary labels {0, 1}
    """
    X, y = make_circles(
        n_samples=n_samples,
        factor=0.60,      # Inner circle radius relative to outer ring
        noise=0.14,       # Gaussian noise creating slight overlap at the boundary
        random_state=seed,
    )
    return X, y

# def generate_circular_dataset(n_samples: int = N_SAMPLES, seed: int = SEED):
#     """
#     Generate two intertwined 2D Archimedean spirals (matching Figure b)
#     with slight Gaussian noise for realistic boundary overlap.
#     Returns:
#         X: ndarray of shape (n_samples, 2) -> [X1, X2]
#         y: ndarray of shape (n_samples,) -> binary labels {0, 1}
#     """
#     rng = np.random.default_rng(seed)
#     n_half = n_samples // 2

#     # Generate angle parameter theta (controls number of spiral turns)
#     theta = np.sqrt(rng.uniform(0, 1, n_half)) * 2.5 * np.pi
#     noise_std = 0.08

#     # Class 0 spiral arm
#     r0 = theta / (2.5 * np.pi)
#     x0 = r0 * np.cos(theta) + rng.normal(0, noise_std, n_half)
#     y0 = r0 * np.sin(theta) + rng.normal(0, noise_std, n_half)

#     # Class 1 spiral arm (rotated by pi / 180 degrees)
#     theta1 = np.sqrt(rng.uniform(0, 1, n_samples - n_half)) * 2.5 * np.pi
#     r1 = theta1 / (2.5 * np.pi)
#     x1 = -r1 * np.cos(theta1) + rng.normal(0, noise_std, n_samples - n_half)
#     y1 = -r1 * np.sin(theta1) + rng.normal(0, noise_std, n_samples - n_half)

#     X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
#     y = np.concatenate([np.zeros(n_half, dtype=int), np.ones(n_samples - n_half, dtype=int)])

#     perm = rng.permutation(n_samples)
#     return X[perm], y[perm]


def plot_datasets(X_lin, y_lin, X_circ, y_circ, save_path: str = "figures/part1_datasets.png"):
    """
    Plot clean scatter plots of both simulated 2D datasets side-by-side.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=200)
    fig.patch.set_facecolor("white")

    c0 = "#E06C75"  # Coral red
    c1 = "#2B5C8F"  # Deep steel blue

    # --- Plot (a): Linear Dataset ---
    axes[0].scatter(
        X_lin[y_lin == 0, 0], X_lin[y_lin == 0, 1],
        c=c0, s=6, alpha=0.35, edgecolors="none", label="Class 0"
    )
    axes[0].scatter(
        X_lin[y_lin == 1, 0], X_lin[y_lin == 1, 1],
        c=c1, s=6, alpha=0.35, edgecolors="none", label="Class 1"
    )
    axes[0].set_title("(a) Linear Dataset (Well-Separated Clusters)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("$X_1$", fontsize=11)
    axes[0].set_ylabel("$X_2$", fontsize=11)

    # --- Plot (b): Circular Dataset ---
    axes[1].scatter(
        X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1],
        c=c0, s=6, alpha=0.35, edgecolors="none", label="Class 0 (Outer Ring)"
    )
    axes[1].scatter(
        X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1],
        c=c1, s=6, alpha=0.35, edgecolors="none", label="Class 1 (Inner Circle)"
    )
    axes[1].set_title("(b) Circular Dataset (Concentric Overlapping Clusters)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("$X_1$", fontsize=11)
    axes[1].set_ylabel("$X_2$", fontsize=11)
    axes[1].set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.grid(True, linestyle=":", alpha=0.4, color="#888888")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c0, markersize=8, label="Class 0 (Cluster A / Outer)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c1, markersize=8, label="Class 1 (Cluster B / Inner)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200)
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def main():
    os.makedirs("data", exist_ok=True)

    # 1. Simulate datasets
    X_lin, y_lin = generate_linear_dataset(N_SAMPLES, SEED)
    X_circ, y_circ = generate_circular_dataset(N_SAMPLES, SEED)

    # 2. Verify shapes and class balance
    print(f"Linear Dataset   -> X shape: {X_lin.shape}, y shape: {y_lin.shape}, Class balance: {np.bincount(y_lin)}")
    print(f"Circular Dataset -> X shape: {X_circ.shape}, y shape: {y_circ.shape}, Class balance: {np.bincount(y_circ)}")

    # 3. Save datasets for downstream scripts
    np.savez("data/linear_dataset.npz", X=X_lin, y=y_lin)
    np.savez("data/circular_dataset.npz", X=X_circ, y=y_circ)
    print("Datasets saved to 'data/' directory.")

    # 4. Plot datasets
    plot_datasets(X_lin, y_lin, X_circ, y_circ)


if __name__ == "__main__":
    main()