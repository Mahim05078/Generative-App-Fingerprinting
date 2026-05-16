import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def distance_matrix(centroids):
    """Compute pairwise Euclidean distance matrix.

    centroids: array-like, shape (n_classes, n_dims)
    returns: ndarray (n_classes, n_classes)
    """
    centroids = np.asarray(centroids, dtype=float)
    return cdist(centroids, centroids, metric="euclidean")


def proximity_inverse(dist_mat):
    """Convert distances to proximity using 1/(1+dist).
    Guarantees values in (0,1], diagonal = 1.
    """
    return 1.0 / (1.0 + dist_mat)


def proximity_rbf(dist_mat, sigma=None):
    """Convert distances to proximity using an RBF kernel.

    If sigma is None, uses median of non-zero distances.
    """
    D = np.array(dist_mat, dtype=float)
    if sigma is None:
        # avoid zeros on diagonal
        flat = D[np.triu_indices_from(D, k=1)]
        if flat.size == 0:
            sigma = 1.0
        else:
            sigma = np.median(flat)
            if sigma <= 0:
                sigma = np.mean(flat) if np.mean(flat) > 0 else 1.0
    sim = np.exp(-(D ** 2) / (2.0 * (sigma ** 2)))
    return sim


def save_matrix_csv(mat, labels, out_path):
    """Save matrix with labels as CSV (header + index)."""
    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(labels))
        for lab, row in zip(labels, mat):
            w.writerow([lab] + list(row))


def plot_heatmap(mat, labels=None, title="Proximity Matrix", cmap="viridis", figsize=(7, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    n = mat.shape[0]
    if labels is not None:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Demo with synthetic centroids
    n = 6
    rng = np.random.default_rng(0)
    centroids = rng.normal(size=(n, 16))
    labels = [f"C{i}" for i in range(n)]

    D = distance_matrix(centroids)
    P_inv = proximity_inverse(D)
    P_rbf = proximity_rbf(D)

    print("Distance matrix:\n", np.round(D, 3))
    print("Inverse proximity (1/(1+dist)):\n", np.round(P_inv, 3))

    # plot example
    plot_heatmap(P_rbf, labels=labels, title="RBF Proximity (demo)")
    plt.show()
