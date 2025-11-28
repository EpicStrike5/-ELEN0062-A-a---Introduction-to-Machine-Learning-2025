"""Point 2.2: Empirical pointwise bias–variance analysis.

This script implements the protocol described in assignment section (2.2):
- Generate R independent learning samples from p(x), with N samples and p features.
- For a fixed test set of inputs, estimate at each x:
  * residual error (here fixed at sigma^2 = 1),
  * squared bias,
  * variance,
  * expected error = residual + bias^2 + variance.
- Do this for three regression methods: ridge, kNN, and decision trees.
- Plot the decomposition as a function of the first feature x1 only, and compare the
  Bayes model h(x) with the average model of each learning algorithm.

"""

import os
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# --- CONSTANTS AND SETUP ---

# R in the protocol (number of independent learning sets)
NB_REPETITIONS: int = 500
# p in the problem statement (only x1 is relevant, others are irrelevant)
NB_FEATURES: int = 5
# N in the problem statement (learning sample size)
NB_SAMPLES: int = 80
# Residual error variance sigma^2
SIGMA_SQUARED: float = 1.0
# T in the protocol (number of test points along x1)
NB_TEST_POINTS: int = 100


# --- TRUE FUNCTION (BAYES MODEL) ---

def h_x(x: np.ndarray) -> np.ndarray:
    """Noise‑free Bayes model h(x).

    The output only depends on the first feature x1:
    h(x) = sin(2 x1) + x1 cos(x1 − 1).

    Parameters
    ----------
    x : np.ndarray, shape (..., p)
        Input feature vectors.
    """

    x1 = x[..., 0]
    return np.sin(2 * x1) + x1 * np.cos(x1 - 1)


# --- DATA GENERATION ---

def create_ls_input(nb_features: int, nb_samples: int) -> np.ndarray:
    """Draws a learning input matrix X from p(x).

    Each feature is sampled independently and uniformly in [-10, 10].
    """

    return np.random.uniform(-10, 10, (nb_samples, nb_features))

def create_learning_set(
    nb_features: int,
    nb_samples: int,
    nb_repetitions: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate ``nb_repetitions`` independent learning sets.

    Each learning set LS_r consists of (X_train, y_train), with N samples and p features.
    The outputs are generated as y = h(x) + ε with ε ~ N(0, σ²).
    """

    learning_sets: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(nb_repetitions):
        X = create_ls_input(nb_features, nb_samples)
        # True (noise‑free) output h(x)
        H_X = h_x(X)
        # Add Gaussian noise with variance σ²
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
    """Create a fixed test set where only x1 varies on a grid.

    Returns
    -------
    x1_values : np.ndarray, shape (T,)
        Grid of x1 values.
    X_test : np.ndarray, shape (T, p)
        Test design matrix with x1 filled and other features set to 0.
    """

    x1_values = np.linspace(x1_min, x1_max, nb_test_points)
    X_test = np.zeros((nb_test_points, nb_features))
    X_test[:, 0] = x1_values
    return x1_values, X_test


# --- ESTIMATOR FACTORIES ---

def get_estimator(method: str, alpha: float = 1.0, k: int = 5, max_depth: int = 5):
    """Return a scikit‑learn regressor given a method name and its complexity parameter."""

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
    """Train an estimator and predict on ``X_test``.

    Any protocol‑level kwargs such as ``N`` or ``P`` are ignored when constructing the model.
    This keeps the public API of :func:`run_pointwise_analysis` flexible without leaking those
    arguments into scikit‑learn constructors.
    """

    # Filter out non‑model‑specific parameters (e.g. N, P from the protocol).
    model_kwargs = {k: v for k, v in kwargs.items() if k not in {"N", "P"}}

    # Ensure 2D inputs for scikit‑learn, even if we only use a single feature.
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    model = get_estimator(method, **model_kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# --- POINTWISE ANALYSIS PROTOCOL (Core Function for 2.2) ---

def run_pointwise_analysis(
    X_test: np.ndarray,
    learning_sets: Iterable[Tuple[np.ndarray, np.ndarray]],
    method: str,
    P: int | None = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Run the R‑repetition protocol for an entire test set.

    For each test point x in ``X_test`` and for a given learning algorithm, this computes
    the empirical pointwise bias–variance decomposition using multiple learning sets.

    Parameters
    ----------
    X_test : np.ndarray, shape (T, p)
        Fixed test design matrix.
    learning_sets : iterable of (X_train, y_train)
        Independent learning sets LS_r.
    method : {"ridge", "knn", "dt"}
        Name of the supervised learning algorithm.
    P : int, optional
        Number of first features to keep (defaults to ``NB_FEATURES``). For (2.2) this
        parameter is not varied, but keeping it makes the function compatible with the
        protocol of (2.3).
    """

    learning_sets = list(learning_sets)
    n_test = X_test.shape[0]
    R = len(learning_sets)

    if R == 0:
        raise ValueError("run_pointwise_analysis requires at least one learning set")

    # Slice features once according to P (defaults to NB_FEATURES for this question).
    n_features = P if P is not None else NB_FEATURES
    X_test_sub = X_test[:, :n_features]

    # Store all R predictions for each of the T test points.
    all_predictions = np.zeros((n_test, R))

    # True Bayes model evaluated on the fixed test set.
    H_X_test = h_x(X_test)

    # R‑repetition protocol.
    for r, (X_train, y_train) in enumerate(learning_sets):
        X_train_sub = X_train[:, :n_features]
        y_train_sub = y_train  # kept for symmetry/clarity

        # Train model and predict on the fixed X_test_sub.
        y_pred = estimator_predict(X_train_sub, y_train_sub, X_test_sub, method, **kwargs)

        # Make sure the assignment always works (1D array expected).
        all_predictions[:, r] = np.asarray(y_pred).reshape(-1)

    # Average model across the R repetitions.
    avg_model = np.mean(all_predictions, axis=1)

    # Squared bias term (pointwise).
    squared_bias = (avg_model - H_X_test) ** 2

    # Variance term (pointwise).
    variance = np.var(all_predictions, axis=1)

    # Expected error (pointwise MSE) according to the decomposition.
    expected_error = SIGMA_SQUARED + squared_bias + variance

    return {
        "H_X_test": H_X_test,
        "avg_model": avg_model,
        "bias_sq": squared_bias,
        "variance": variance,
        "mse": expected_error,
    }


# --- PLOTTING FUNCTIONS FOR 2.2 ---

def plot_decomposition(x_plot: np.ndarray, results: Dict[str, Dict[str, np.ndarray]], output_folder: str) -> None:
    """Create the bias–variance–MSE decomposition plots (one per method)."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(
        f"Pointwise Bias–Variance Decomposition (N={NB_SAMPLES}, p={NB_FEATURES})",
        fontsize=16,
    )

    for ax, (name, data) in zip(axes, results.items()):
        # Decomposition components.
        ax.plot(
            x_plot,
            data["bias_sq"],
            label="Squared Bias (Bias^2)",
            color="blue",
            linestyle="--",
        )
        ax.plot(
            x_plot,
            data["variance"],
            label="Variance (Var)",
            color="green",
            linestyle="--",
        )
        ax.hlines(
            SIGMA_SQUARED,
            x_plot.min(),
            x_plot.max(),
            label="Residual Error (sigma^2 = 1)",
            color="gray",
            linestyle=":",
        )

        # Total expected error.
        ax.plot(
            x_plot,
            data["mse"],
            label="Total Expected Error (MSE)",
            color="red",
            linewidth=2,
        )

        ax.set_title(name)
        ax.set_ylabel("Error value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel("x1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "1_decomposition_plot.pdf"))
    plt.close(fig)

def plot_average_models(x_plot: np.ndarray, results: Dict[str, Dict[str, np.ndarray]], output_folder: str) -> None:
    """Create the plot comparing Bayes model and average models of each algorithm."""

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title(
        f"Bayes Model vs. Average Models (N={NB_SAMPLES}, p={NB_FEATURES})",
    )

    # Bayes model evaluated on the shared test set (identical for all methods).
    first_key = next(iter(results))
    ax2.plot(
        x_plot,
        results[first_key]["H_X_test"],
        label="Bayes Model (h(x))",
        color="black",
        linewidth=3,
    )
    
    # Average model for each estimator.
    for name, data in results.items():
        ax2.plot(x_plot, data["avg_model"], label=f"Average Model ({name})", linewidth=2)

    # Sample of the learning set (from the first 5 learning samples).
    sample_LS = create_learning_set(NB_FEATURES, NB_SAMPLES, 5)
    stacked_sample = (np.vstack([sample_LS[i][0] for i in range(5)]), np.hstack([sample_LS[i][1] for i in range(5)]))
    ax2.scatter(stacked_sample[0][:, 0],stacked_sample[1],label="Learning set sample (5 sets)",color="violet",alpha=0.8,s=15,)

    ax2.set_xlabel("x1")
    ax2.set_ylabel("Prediction value")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "2_average_model_plot.pdf"))
    plt.close(fig2)


# --- MAIN EXECUTION FOR (2.2) ---

def main_2_2() -> None:
    """Run the complete protocol for assignment point (2.2)."""

    # Output folder for the generated PDF figures.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_2")
    os.makedirs(output_folder, exist_ok=True)

    # Fixed seeds for full reproducibility of the experiment.
    np.random.seed(42)
    random.seed(42)

    # 1. Create the fixed test set X_test (only x1 varies).
    x1_values, X_test = generate_test_set(NB_FEATURES, NB_TEST_POINTS)

    # 2. Define models with their complexity parameters.
    # Use plain text labels to avoid mathtext parsing issues in matplotlib titles.
    BASE_MODELS = {
        "Ridge (alpha=10 - High Bias)": {"method": "ridge", "alpha": 10.0},
        "kNN (k=10 - Moderate)": {"method": "knn", "k": 10},
        "Tree (max_depth=5 - Moderate)": {"method": "dt", "max_depth": 5},
    }

    # 3. Generate one collection of R independent learning samples.
    print(
        f"Generating {NB_REPETITIONS} independent learning samples "
        f"(N={NB_SAMPLES}, p={NB_FEATURES})...",
    )
    learning_sets = create_learning_set(NB_FEATURES, NB_SAMPLES, NB_REPETITIONS)

    # 4. Run the pointwise analysis for all models.
    results: Dict[str, Dict[str, np.ndarray]] = {}
    print("--- Running Pointwise Bias–Variance Analysis ---")
    for name, params in BASE_MODELS.items():
        print(f"Analyzing {name}...")
        results[name] = run_pointwise_analysis(X_test, learning_sets, **params)

    # 5. Generate and save plots.
    print(f"Generating and saving plots to {output_folder}...")
    plot_decomposition(x1_values, results, output_folder)
    plot_average_models(x1_values, results, output_folder)
    print("Analysis complete. Check the output folder for PDF files.")


if __name__ == "__main__":
    main_2_2()
