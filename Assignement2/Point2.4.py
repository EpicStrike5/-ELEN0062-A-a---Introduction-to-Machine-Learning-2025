"""Point 2.4: Bagging (bootstrap aggregating) for bias–variance reduction.

This script reuses the data generation and evaluation protocol from previous
points and applies the bagging idea to three regression methods:
- linear (ridge) regression,
- kNN regression,
- regression trees.

For each method, we grow B models on bootstrap samples of a given learning set
and average their predictions. We then estimate, as before, the pointwise bias,
variance and expected error (MSE) as a function of x1 for increasing values of
B, and discuss how bagging affects each model.
"""

import os
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# We mirror the constants from previous scripts.
NB_REPETITIONS: int = 200  # R in the protocol
NB_FEATURES: int = 5        # p default
NB_SAMPLES: int = 80        # N default
SIGMA_SQUARED: float = 1.0  # Residual error variance sigma^2
NB_TEST_POINTS: int = 100   # T test points along x1


# --- TRUE FUNCTION AND DATA GENERATION (copied from 2.2/2.3) ---


def h_x(x: np.ndarray) -> np.ndarray:
    """Noise-free Bayes model h(x)."""
    x1 = x[..., 0]
    return np.sin(2 * x1) + x1 * np.cos(x1 - 1)


def create_ls_input(nb_features: int, nb_samples: int) -> np.ndarray:
    """Draws a learning input matrix X from p(x)."""
    return np.random.uniform(-10, 10, (nb_samples, nb_features))


def create_learning_set(
    nb_features: int,
    nb_samples: int,
    nb_repetitions: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate ``nb_repetitions`` independent learning sets.

    Each learning set LS_r consists of (X_train, y_train), with N samples and p
    features. The outputs are generated as y = h(x) + epsilon with
    epsilon ~ N(0, sigma^2).
    """

    learning_sets: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(nb_repetitions):
        X = create_ls_input(nb_features, nb_samples)
        H_X = h_x(X)
        epsilon = np.random.normal(0.0, np.sqrt(SIGMA_SQUARED), X.shape[0])
        Y = H_X + epsilon
        learning_sets.append((X, Y))
    return learning_sets


def generate_test_set(
    nb_features: int,
    nb_test_points: int,
    x1_min: float = -10.0,
    x1_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a fixed test set where only x1 varies on a grid."""

    x1_values = np.linspace(x1_min, x1_max, nb_test_points)
    X_test = np.zeros((nb_test_points, nb_features))
    X_test[:, 0] = x1_values
    return x1_values, X_test


# --- BASE ESTIMATORS AND SINGLE-MODEL PREDICTION ---


def get_estimator(method: str, alpha: float = 1.0, k: int = 5, max_depth: int = 5):
    """Return a scikit-learn regressor for the given method name."""

    if method == "ridge":
        return Ridge(alpha=alpha)
    if method == "knn":
        return KNeighborsRegressor(n_neighbors=k)
    if method == "dt":
        return DecisionTreeRegressor(max_depth=max_depth)
    raise ValueError(f"Unknown method: {method}")


def estimator_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
    **kwargs,
) -> np.ndarray:
    """Train a single estimator and predict on X_test."""

    # Filter protocol-level kwargs (e.g. N, P) that are not model hyperparameters.
    model_kwargs = {k: v for k, v in kwargs.items() if k not in {"N", "P"}}

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    model = get_estimator(method, **model_kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# --- BAGGING ---


def bagged_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
    B: int,
    **kwargs,
) -> np.ndarray:
    """Return predictions of a bagged ensemble of size B.

    Each of the B base models is trained on a bootstrap sample drawn with
    replacement from (X_train, y_train). The final prediction is the average
    of the B individual predictions.
    """

    n_samples = X_train.shape[0]
    preds = np.zeros((X_test.shape[0], B))

    for b in range(B):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        preds[:, b] = estimator_predict(X_boot, y_boot, X_test, method, **kwargs)

    return preds.mean(axis=1)


# --- POINTWISE ANALYSIS WITH OR WITHOUT BAGGING ---


def run_pointwise_analysis_with_bagging(
    X_test: np.ndarray,
    learning_sets: Iterable[Tuple[np.ndarray, np.ndarray]],
    method: str,
    B: int = 1,
    P: int | None = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Run the R-repetition protocol, optionally with bagging.

    When B = 1, this reduces to the standard analysis (no bagging).
    When B > 1, each learning set is used to build a bagged ensemble of size B.
    """

    learning_sets = list(learning_sets)
    n_test = X_test.shape[0]
    R = len(learning_sets)
    if R == 0:
        raise ValueError("At least one learning set is required")

    n_features = P if P is not None else NB_FEATURES
    X_test_sub = X_test[:, :n_features]

    all_predictions = np.zeros((n_test, R))
    H_X_test = h_x(X_test)

    for r, (X_train, y_train) in enumerate(learning_sets):
        X_train_sub = X_train[:, :n_features]
        if B == 1:
            y_pred = estimator_predict(X_train_sub, y_train, X_test_sub, method, **kwargs)
        else:
            y_pred = bagged_predict(X_train_sub, y_train, X_test_sub, method, B=B, **kwargs)
        all_predictions[:, r] = np.asarray(y_pred).reshape(-1)

    avg_model = np.mean(all_predictions, axis=1)
    squared_bias = (avg_model - H_X_test) ** 2
    variance = np.var(all_predictions, axis=1)
    expected_error = SIGMA_SQUARED + squared_bias + variance

    return {
        "H_X_test": H_X_test,
        "avg_model": avg_model,
        "bias_sq": squared_bias,
        "variance": variance,
        "mse": expected_error,
    }


# --- PLOTTING: EFFECT OF BAGGING ON MEAN BIAS AND VARIANCE ---


def calculate_mean_values(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Average pointwise results over the test set."""

    mean_bias_sq = float(np.mean(results["bias_sq"]))
    mean_variance = float(np.mean(results["variance"]))
    mean_mse = float(np.mean(results["mse"]))
    return {
        "mean_bias_sq": mean_bias_sq,
        "mean_variance": mean_variance,
        "mean_mse": mean_mse,
        "mean_residual_error": SIGMA_SQUARED,
    }


def plot_bagging_effect(
    bagging_results: Dict[str, Dict[int, Dict[str, float]]],
    B_values: List[int],
    output_folder: str,
) -> None:
    """Plot mean bias^2, variance and MSE as a function of B for each method."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    methods = ["Ridge", "kNN", "Tree"]

    for ax, method_name in zip(axes, methods):
        stats_for_method = bagging_results[method_name]
        mean_bias = [stats_for_method[B]["mean_bias_sq"] for B in B_values]
        mean_var = [stats_for_method[B]["mean_variance"] for B in B_values]
        mean_mse = [stats_for_method[B]["mean_mse"] for B in B_values]

        ax.plot(B_values, mean_bias, "b--", label="Mean squared bias")
        ax.plot(B_values, mean_var, "g--", label="Mean variance")
        ax.plot(B_values, mean_mse, "r-", label="Mean total error (MSE)")
        ax.hlines(
            SIGMA_SQUARED,
            B_values[0],
            B_values[-1],
            colors="gray",
            linestyles=":",
            label="Residual error (sigma^2 = 1)",
        )

        ax.set_title(f"{method_name} regression")
        ax.set_ylabel("Mean error value")
        ax.legend()
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel("Number of bagged models (B)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "bagging_effect.pdf"))
    plt.close(fig)


# --- MAIN ENTRY POINT FOR (2.4) ---


def main_2_4() -> None:
    """Run bagging experiments for ridge, kNN and regression trees."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_4")
    os.makedirs(output_folder, exist_ok=True)

    np.random.seed(123)
    random.seed(123)

    # Fixed test set.
    x1_values, X_test = generate_test_set(NB_FEATURES, NB_TEST_POINTS)

    # Generate R independent learning sets, as in previous points.
    learning_sets = create_learning_set(NB_FEATURES, NB_SAMPLES, NB_REPETITIONS)

    # Methods and their baseline hyperparameters.
    base_models = {
        "Ridge": {"method": "ridge", "alpha": 1.0},
        "kNN": {"method": "knn", "k": 5},
        "Tree": {"method": "dt", "max_depth": 5},
    }

    # Range of ensemble sizes B to test.
    B_values = [1, 2, 5, 10, 20, 50]

    bagging_results: Dict[str, Dict[int, Dict[str, float]]] = {
        name: {} for name in base_models.keys()
    }

    print("--- Running bagging experiments (2.4) ---")
    for name, params in base_models.items():
        print(f"Method: {name}")
        for B in B_values:
            print(f"  B = {B}")
            results = run_pointwise_analysis_with_bagging(
                X_test, learning_sets, B=B, **params
            )
            bagging_results[name][B] = calculate_mean_values(results)

    print(f"Saving bagging effect plots to {output_folder} ...")
    plot_bagging_effect(bagging_results, B_values, output_folder)
    print("Done.")


if __name__ == "__main__":
    main_2_4()
