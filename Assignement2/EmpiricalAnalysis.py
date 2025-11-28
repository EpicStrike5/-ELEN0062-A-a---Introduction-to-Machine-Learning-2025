"""Empirical analysis script for assignment 2 (points 2.2, 2.3, 2.4).

This module consolidates the code from:
- Point2.2: pointwise bias–variance analysis as a function of x1.
- Point2.3: mean bias–variance analysis as functions of N, model complexity,
  and number of irrelevant features.
- Point2.4: effect of bagging on bias, variance and MSE.

Usage
-----
Run the module as a script and choose which experiment(s) to execute by
editing the main() function at the bottom, or by importing and calling the
public functions from elsewhere:
- run_experiment_2_2()
- run_experiment_2_3()
- run_experiment_2_4()
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# -----------------------------------------------------------------------------
# Global constants used across all experiments
# -----------------------------------------------------------------------------

# R in the protocol (number of independent learning sets)
NB_REPETITIONS: int = 200
# Default number of features (p); only x1 is relevant, others are irrelevant.
NB_FEATURES: int = 5
# Default learning sample size N.
NB_SAMPLES: int = 80
# Residual error variance sigma^2.
SIGMA_SQUARED: float = 1.0
# Default number of test points along x1.
NB_TEST_POINTS: int = 100


# -----------------------------------------------------------------------------
# True Bayes model and data generation
# -----------------------------------------------------------------------------


def h_x(x: np.ndarray) -> np.ndarray:
    """Noise-free Bayes model h(x).

    The output only depends on x1:
        h(x) = sin(2 * x1) + x1 * cos(x1 - 1).
    """

    x1 = x[..., 0]
    return np.sin(2 * x1) + x1 * np.cos(x1 - 1)


def create_ls_input(nb_features: int, nb_samples: int) -> np.ndarray:
    """Draw a learning input matrix X from p(x).

    Each feature is sampled independently and uniformly in [-10, 10].
    """

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


# -----------------------------------------------------------------------------
# Base estimators (ridge, kNN, decision tree) and prediction helpers
# -----------------------------------------------------------------------------


def get_estimator(method: str, alpha: float = 10, k: int = 10, max_depth: int = 5):
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
    """Train a single estimator and predict on X_test.

    Protocol-level kwargs such as ``N`` and ``P`` are filtered out before
    constructing the model, so that run_* functions can pass them without
    affecting the estimators.
    """

    model_kwargs = {k: v for k, v in kwargs.items() if k not in {"N", "P"}}

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    model = get_estimator(method, **model_kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# -----------------------------------------------------------------------------
# Generic pointwise analysis (used by 2.2 and 2.3)
# -----------------------------------------------------------------------------


def run_pointwise_analysis(
    X_test: np.ndarray,
    learning_sets: Iterable[Tuple[np.ndarray, np.ndarray]],
    method: str,
    P: int | None = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Run the R-repetition protocol for an entire test set.

    This is the general version used for both 2.2 and 2.3. It supports
    subsampling the learning set size ``N`` and slicing the first ``P``
    features when requested via kwargs.
    """

    learning_sets = list(learning_sets)
    n_test = X_test.shape[0]
    R = len(learning_sets)
    if R == 0:
        raise ValueError("run_pointwise_analysis requires at least one learning set")

    # Parameters controlling subsampling and feature slicing.
    current_P = kwargs.get("P", P if P is not None else NB_FEATURES)
    current_N = kwargs.get("N", NB_SAMPLES)

    # Slice X_test once based on current_P.
    X_test_sub = X_test[:, :current_P]

    all_predictions = np.zeros((n_test, R))
    H_X_test = h_x(X_test)

    for r, (X_train, y_train) in enumerate(learning_sets):
        # 1. Subsample N if requested (used in 2.3 analysis A).
        if current_N < X_train.shape[0]:
            indices = np.random.choice(X_train.shape[0], current_N, replace=False)
            X_train_temp = X_train[indices]
            y_train_temp = y_train[indices]
        else:
            X_train_temp = X_train
            y_train_temp = y_train

        # 2. Slice features P (irrelevant variables handling for 2.3 analysis C).
        X_train_sub = X_train_temp[:, :current_P]

        # 3. Train model and predict on the fixed X_test_sub.
        y_pred = estimator_predict(X_train_sub, y_train_temp, X_test_sub, method, **kwargs)
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


# -----------------------------------------------------------------------------
# 2.2 – Pointwise plots as a function of x1
# -----------------------------------------------------------------------------


def plot_decomposition(
    x_plot: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    output_folder: str,
) -> None:
    """Create the bias–variance–MSE decomposition plots (one per method)."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(
        f"Pointwise Bias–Variance Decomposition (N={NB_SAMPLES}, p={NB_FEATURES})",
        fontsize=16,
    )

    for ax, (name, data) in zip(axes, results.items()):
        ax.plot(x_plot, data["bias_sq"], label="Squared Bias (Bias^2)", color="blue", linestyle="--")
        ax.plot(x_plot, data["variance"], label="Variance (Var)", color="green", linestyle="--")
        ax.hlines(
            SIGMA_SQUARED,
            x_plot.min(),
            x_plot.max(),
            label="Residual Error (sigma^2 = 1)",
            color="gray",
            linestyle=":",
        )
        ax.plot(x_plot, data["mse"], label="Total Expected Error (MSE)", color="red", linewidth=2)

        ax.set_title(name)
        ax.set_ylabel("Error value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel("x1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "1_decomposition_plot.pdf"))
    plt.close(fig)


def plot_average_models(
    x_plot: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    output_folder: str,
) -> None:
    """Create the plot comparing Bayes model and average models of each algorithm."""

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title(
        f"Bayes Model vs. Average Models (N={NB_SAMPLES}, p={NB_FEATURES})",
    )

    first_key = next(iter(results))
    ax2.plot(
        x_plot,
        results[first_key]["H_X_test"],
        label="Bayes Model (h(x))",
        color="black",
        linewidth=3,
    )

    for name, data in results.items():
        ax2.plot(x_plot, data["avg_model"], label=f"Average Model ({name})", linewidth=2)

    ax2.set_xlabel("x1")
    ax2.set_ylabel("Prediction value")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "2_average_model_plot.pdf"))
    plt.close(fig2)


def run_experiment_2_2() -> None:
    """Run the complete protocol for assignment point (2.2)."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_2")
    os.makedirs(output_folder, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    x1_values, X_test = generate_test_set(NB_FEATURES, NB_TEST_POINTS)

    BASE_MODELS = {
        "Ridge (alpha=10 - High Bias)": {"method": "ridge", "alpha": 10.0},
        "kNN (k=10 - Moderate)": {"method": "knn", "k": 10},
        "Tree (max_depth=5 - Moderate)": {"method": "dt", "max_depth": 5},
    }

    print(
        f"Generating {NB_REPETITIONS} independent learning samples "
        f"(N={NB_SAMPLES}, p={NB_FEATURES})...",
    )
    learning_sets = create_learning_set(NB_FEATURES, NB_SAMPLES, NB_REPETITIONS)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    print("--- Running Pointwise Bias–Variance Analysis (2.2) ---")
    for name, params in BASE_MODELS.items():
        print(f"Analyzing {name}...")
        results[name] = run_pointwise_analysis(X_test, learning_sets, **params)

    print(f"Generating and saving plots to {output_folder}...")
    plot_decomposition(x1_values, results, output_folder)
    plot_average_models(x1_values, results, output_folder)
    print("Point 2.2 analysis complete.")


# -----------------------------------------------------------------------------
# 2.3 – Mean analysis over N, complexity and number of features
# -----------------------------------------------------------------------------


def plot_mean_results(
    results: Dict[str, Dict[str, List[float]]],
    param_range: List[float],
    param_label: str,
    filename: str,
    output_folder: str,
) -> None:
    """Generic plotter for N-impact and P-impact (linear X-axis)."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(
        f"Impact of {param_label} on mean error components (MSE averaged)",
        fontsize=16,
    )

    colors = ["blue", "green", "red"]

    for i, name in enumerate(results.keys()):
        ax = axes[i]
        ax.plot(param_range, results[name]["Bias"], label="Mean squared bias", color=colors[0], linestyle="--")
        ax.plot(param_range, results[name]["Var"], label="Mean variance", color=colors[1], linestyle="--")
        ax.plot(param_range, results[name]["MSE"], label="Mean total error (MSE)", color=colors[2], linewidth=2)
        ax.hlines(
            SIGMA_SQUARED,
            param_range[0],
            param_range[-1],
            label="Residual error (sigma^2 = 1)",
            color="gray",
            linestyle=":",
        )
        ax.set_title(f"{name} regression")
        ax.set_ylabel("Mean error value")
        ax.legend()
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel(param_label)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, filename))
    plt.close(fig)


def plot_complexity_results(
    results: Dict[str, Dict[str, List[float]]],
    ridge_alphas: List[float],
    knn_ks: List[int],
    tree_depths: List[int],
    output_folder: str,
) -> None:
    """Plot impact of model complexity on mean error components."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(
        "Impact of model complexity on mean error components (MSE averaged)",
        fontsize=16,
    )

    # Ridge
    ax = axes[0]
    ax.set_title("Ridge regression (complexity decreases as alpha increases)")
    ax.plot(results["Ridge"]["Param"], results["Ridge"]["Bias"], label="Mean Bias^2", color="blue", linestyle="--")
    ax.plot(results["Ridge"]["Param"], results["Ridge"]["Var"], label="Mean Var", color="green", linestyle="--")
    ax.plot(results["Ridge"]["Param"], results["Ridge"]["MSE"], label="Mean MSE", color="red", linewidth=2)
    ax.hlines(
        SIGMA_SQUARED,
        ridge_alphas[0],
        ridge_alphas[-1],
        label="Residual error (sigma^2 = 1)",
        color="gray",
        linestyle=":",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Regularization parameter alpha (complexity decreases as alpha increases)")
    ax.set_ylabel("Mean error value")
    ax.legend()
    ax.grid(True, alpha=0.5)

    # kNN
    ax = axes[1]
    ax.set_title("kNN regression (complexity decreases as k increases)")
    ax.plot(results["kNN"]["Param"], results["kNN"]["Bias"], label="Mean Bias^2", color="blue", linestyle="--")
    ax.plot(results["kNN"]["Param"], results["kNN"]["Var"], label="Mean Var", color="green", linestyle="--")
    ax.plot(results["kNN"]["Param"], results["kNN"]["MSE"], label="Mean MSE", color="red", linewidth=2)
    ax.hlines(
        SIGMA_SQUARED,
        knn_ks[0],
        knn_ks[-1],
        label="Residual error (sigma^2 = 1)",
        color="gray",
        linestyle=":",
    )
    ax.set_xlabel("Number of neighbors k (complexity decreases as k increases)")
    ax.set_ylabel("Mean error value")
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Tree
    ax = axes[2]
    ax.set_title("Decision tree regression (complexity increases with max depth)")
    ax.plot(results["Tree"]["Param"], results["Tree"]["Bias"], label="Mean Bias^2", color="blue", linestyle="--")
    ax.plot(results["Tree"]["Param"], results["Tree"]["Var"], label="Mean Var", color="green", linestyle="--")
    ax.plot(results["Tree"]["Param"], results["Tree"]["MSE"], label="Mean MSE", color="red", linewidth=2)
    ax.hlines(
        SIGMA_SQUARED,
        tree_depths[0],
        tree_depths[-1],
        label="Residual error (sigma^2 = 1)",
        color="gray",
        linestyle=":",
    )
    ax.set_xlabel("Max depth (complexity increases with depth)")
    ax.set_ylabel("Mean error value")
    ax.legend()
    ax.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "2_complexity_plot.pdf"))
    plt.close(fig)


def run_experiment_2_3() -> None:
    """Run mean analyses for N, complexity and irrelevant variables (2.3)."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_3")
    os.makedirs(output_folder, exist_ok=True)

    np.random.seed(100)
    random.seed(100)

    x1_values = np.linspace(-10, 10, NB_TEST_POINTS)
    P_MAX = 10
    X_test_fixed = np.zeros((NB_TEST_POINTS, P_MAX))
    X_test_fixed[:, 0] = x1_values

    N_MAX = 500
    print(f"Generating large pool of {NB_REPETITIONS} learning samples (N={N_MAX}, p={P_MAX})...")
    LSet_pool = create_learning_set(P_MAX, N_MAX, NB_REPETITIONS)

    BASE_MODELS = {
        "Ridge": {"method": "ridge", "alpha": 10.0},
        "kNN": {"method": "knn", "k": 10},
        "Tree": {"method": "dt", "max_depth": 5},
    }

    # Analysis A – impact of N
    print("\n--- Running Analysis A: Impact of N ---")
    N_range = [10, 20, 40, 80, 160, 320, N_MAX]
    N_results: Dict[str, Dict[str, List[float]]] = {
        name: {"N": [], "Bias": [], "Var": [], "MSE": []} for name in BASE_MODELS.keys()
    }

    for N in N_range:
        print(f"  Testing N={N}")
        for name, params in BASE_MODELS.items():
            results = run_pointwise_analysis(
                X_test_fixed, LSet_pool, N=N, P=NB_FEATURES, **params
            )
            mean_results = calculate_mean_values(results)
            N_results[name]["N"].append(N)
            N_results[name]["Bias"].append(mean_results["mean_bias_sq"])
            N_results[name]["Var"].append(mean_results["mean_variance"])
            N_results[name]["MSE"].append(mean_results["mean_mse"])

    plot_mean_results(N_results, N_range, "Learning Set Size (N)", "1_N_impact.pdf", output_folder)

    # Analysis B – impact of complexity
    print("\n--- Running Analysis B: Impact of Complexity ---")
    RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
    KNN_KS = [1, 3, 5, 10, 20, 40]
    TREE_DEPTHS = [1, 2, 4, 8, 16]

    C_results: Dict[str, Dict[str, List[float]]] = {
        "Ridge": {"Param": [], "Bias": [], "Var": [], "MSE": []},
        "kNN": {"Param": [], "Bias": [], "Var": [], "MSE": []},
        "Tree": {"Param": [], "Bias": [], "Var": [], "MSE": []},
    }

    for alpha in RIDGE_ALPHAS:
        results = run_pointwise_analysis(
            X_test_fixed, LSet_pool, N=NB_SAMPLES, P=NB_FEATURES, method="ridge", alpha=alpha
        )
        mean_results = calculate_mean_values(results)
        C_results["Ridge"]["Param"].append(alpha)
        C_results["Ridge"]["Bias"].append(mean_results["mean_bias_sq"])
        C_results["Ridge"]["Var"].append(mean_results["mean_variance"])
        C_results["Ridge"]["MSE"].append(mean_results["mean_mse"])

    for k in KNN_KS:
        results = run_pointwise_analysis(
            X_test_fixed, LSet_pool, N=NB_SAMPLES, P=NB_FEATURES, method="knn", k=k
        )
        mean_results = calculate_mean_values(results)
        C_results["kNN"]["Param"].append(k)
        C_results["kNN"]["Bias"].append(mean_results["mean_bias_sq"])
        C_results["kNN"]["Var"].append(mean_results["mean_variance"])
        C_results["kNN"]["MSE"].append(mean_results["mean_mse"])

    for depth in TREE_DEPTHS:
        results = run_pointwise_analysis(
            X_test_fixed, LSet_pool, N=NB_SAMPLES, P=NB_FEATURES, method="dt", max_depth=depth
        )
        mean_results = calculate_mean_values(results)
        C_results["Tree"]["Param"].append(depth)
        C_results["Tree"]["Bias"].append(mean_results["mean_bias_sq"])
        C_results["Tree"]["Var"].append(mean_results["mean_variance"])
        C_results["Tree"]["MSE"].append(mean_results["mean_mse"])

    plot_complexity_results(C_results, RIDGE_ALPHAS, KNN_KS, TREE_DEPTHS, output_folder)

    # Analysis C – impact of irrelevant features
    print("\n--- Running Analysis C: Impact of Irrelevant Features ---")
    P_range = [2, 4, 6, 8, P_MAX]
    P_results: Dict[str, Dict[str, List[float]]] = {
        name: {"P": [], "Bias": [], "Var": [], "MSE": []} for name in BASE_MODELS.keys()
    }

    for P in P_range:
        print(f"  Testing P={P}")
        for name, params in BASE_MODELS.items():
            results = run_pointwise_analysis(
                X_test_fixed, LSet_pool, N=NB_SAMPLES, P=P, **params
            )
            mean_results = calculate_mean_values(results)
            P_results[name]["P"].append(P)
            P_results[name]["Bias"].append(mean_results["mean_bias_sq"])
            P_results[name]["Var"].append(mean_results["mean_variance"])
            P_results[name]["MSE"].append(mean_results["mean_mse"])

    plot_mean_results(P_results, P_range, "Number of Features (p)", "3_P_impact.pdf", output_folder)

    print("\nPoint 2.3 analysis complete.")


# -----------------------------------------------------------------------------
# 2.4 – Bagging experiments
# -----------------------------------------------------------------------------


def bagged_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
    B: int,
    **kwargs,
) -> np.ndarray:
    """Return predictions of a bagged ensemble of size B."""

    n_samples = X_train.shape[0]
    preds = np.zeros((X_test.shape[0], B))

    for b in range(B):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        preds[:, b] = estimator_predict(X_boot, y_boot, X_test, method, **kwargs)

    return preds.mean(axis=1)


def run_pointwise_analysis_with_bagging(
    X_test: np.ndarray,
    learning_sets: Iterable[Tuple[np.ndarray, np.ndarray]],
    method: str,
    B: int = 1,
    P: int | None = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Run the R-repetition protocol, optionally with bagging."""

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


def run_experiment_2_4() -> None:
    """Run bagging experiments for ridge, kNN and regression trees (2.4)."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_4")
    os.makedirs(output_folder, exist_ok=True)

    np.random.seed(123)
    random.seed(123)

    _, X_test = generate_test_set(NB_FEATURES, NB_TEST_POINTS)
    learning_sets = create_learning_set(NB_FEATURES, NB_SAMPLES, NB_REPETITIONS)

    base_models = {
        "Ridge": {"method": "ridge", "alpha": 10},
        "kNN": {"method": "knn", "k": 10},
        "Tree": {"method": "dt", "max_depth": 5},
    }

    B_values = [1, 2, 5, 10, 20, 50]
    bagging_results: Dict[str, Dict[int, Dict[str, float]]] = {name: {} for name in base_models.keys()}

    print("--- Running bagging experiments (2.4) ---")
    for name, params in base_models.items():
        print(f"Method: {name}")
        for B in B_values:
            print(f"  B = {B}")
            results = run_pointwise_analysis_with_bagging(X_test, learning_sets, B=B, **params)
            bagging_results[name][B] = calculate_mean_values(results)

    print(f"Saving bagging effect plots to {output_folder} ...")
    plot_bagging_effect(bagging_results, B_values, output_folder)
    print("Point 2.4 analysis complete.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    """Entry point to run selected experiments.

    Adjust the booleans below to choose which parts to execute.
    """

    run_2_2 = True
    run_2_3_ = True
    run_2_4_ = True

    if run_2_2:
        run_experiment_2_2()
    if run_2_3_:
        run_experiment_2_3()
    if run_2_4_:
        run_experiment_2_4()


if __name__ == "__main__":
    main()
