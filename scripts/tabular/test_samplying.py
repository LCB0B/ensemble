# %%
import math
from collections import Counter

import numpy as np
from sklearn.datasets import make_classification


def deterministic_stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    subset_size: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically sample a stratified subset of increasing size.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.
        subset_size (int): Desired sample size.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Subset X, y, and indices.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(y))
    class_0 = indices[y == 0]
    class_1 = indices[y == 1]

    # Shuffle within class
    rng.shuffle(class_0)
    rng.shuffle(class_1)

    # Compute class proportions
    total = len(y)
    p0 = len(class_0) / total

    # Determine number from each class
    n0 = math.floor(
        p0 * subset_size
    )  # we take floor such that class 1 will be slightly overrepresented for very small sample sizes
    n1 = subset_size - n0

    selected = np.concatenate([class_0[:n0], class_1[:n1]])
    selected.sort()  # sort to preserve order in original dataset

    return X[selected], y[selected], selected


# %%
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=0,
)

sizes = [100, 101]
subsets = {}

for size in sizes:
    X_sub, y_sub, idx = deterministic_stratified_subset(X, y, size)
    print(f"Subset size {size} - class balance: {Counter(y_sub)}")
    subsets[size] = idx

overlap = set(subsets[100]).issubset(set(subsets[101]))
print(f"All 100 indices are included in 101: {overlap}")


# %%
overlap

# %%
subsets

# %%
subsets
# %%
X_sub
# %%
y_sub
# %%
