"""Point 2.3: Mean bias–variance analysis over N, complexity, and irrelevant features.

This script uses the protocol from (2.1)/(2.2) to estimate mean (over p(x))
values of the residual error, squared bias, variance and expected error (MSE)
for ridge regression, kNN and regression trees as functions of:
- the learning set size N,
- the model complexity,
- the number of irrelevant features.

"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os
import random

# --- CONSTANTS AND SETUP ---
# Note: Constants are set high for rigorous mean value approximation
NB_REPETITIONS = 200 # R in the protocol
NB_FEATURES = 5    # p default
NB_SAMPLES = 80    # N default
SIGMA_SQUARED = 1.0 # Sigma^2 = 1 (Residual Error)
NB_TEST_POINTS = 100 # T in the protocol (for averaging)

# --- TRUE FUNCTIONS ---

def h_x(x):
    """The true, noise-free Bayes model function h(x)."""
    # h(x) = sin(2*x1) + x1 * cos(x1 - 1)
    # x[..., 0] is x1
    return np.sin(2 * x[..., 0]) + x[..., 0] * np.cos(x[..., 0] - 1)

# --- DATA GENERATION ---

def create_ls_input(nb_features, nb_samples):
    """Generates the input features (LSx) from p(x)."""
    return np.random.uniform(-10, 10, (nb_samples, nb_features))

def create_learning_set(nb_features, nb_samples, nb_repetitions):
    """
    Generates R independent Learning Samples (LS_r), each of size N.
    Returns: list of R tuples (X_train, Y_train).
    """
    LSet = []
    for _ in range(nb_repetitions):
        X = create_ls_input(nb_features, nb_samples)
        H_X = h_x(X)
        epsilon = np.random.normal(0, np.sqrt(SIGMA_SQUARED), X.shape[0])
        Y = H_X + epsilon
        LSet.append((X, Y))
    return LSet

# --- ESTIMATOR FUNCTION ---

def get_estimator(method, alpha=10, k=10, max_depth=5):
    """Returns the scikit-learn estimator instance based on method and complexity."""
    if method == "ridge":
        return Ridge(alpha=alpha)
    elif method == "knn":
        return KNeighborsRegressor(n_neighbors=k)
    elif method == "dt":
        return DecisionTreeRegressor(max_depth=max_depth)
    else:
        raise ValueError(f"Unknown method: {method}")

def estimator_predict(X_train, Y_train, X_test, method, **kwargs):
    """
    Trains the estimator and predicts values for X_test.
    Filters out non-model specific arguments before calling get_estimator.
    """
    # Filter out non-model-specific parameters (N and P are filtered here)
    model_kwargs = {k: v for k, v in kwargs.items() if k not in ['N', 'P']}
    
    # Ensure dimensions are correct for Scikit-learn 
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
        
    model = get_estimator(method, **model_kwargs)
    model.fit(X_train, Y_train)
    return model.predict(X_test)


# --- POINTWISE ANALYSIS PROTOCOL (Core Function) ---

def run_pointwise_analysis(X_test, LSet, method, **kwargs):
    """
    Implements the R-repetition protocol using subsampling/feature slicing as defined in kwargs (N, P).
    Calculates the pointwise Bias^2, Variance, and Average Model.
    
    The FIX for the NameError is integrated here by ensuring X/Y_train_temp are always defined.
    """
    N_test = X_test.shape[0]
    R = len(LSet)
    all_predictions = np.zeros((N_test, R))
    H_X_test = h_x(X_test)

    # Get parameters early
    current_P = kwargs.get('P', NB_FEATURES)
    current_N = kwargs.get('N', NB_SAMPLES)
    
    # Slice X_test_sub once based on current P
    X_test_sub = X_test[:, :current_P]

    for r in range(R):
        X_train, Y_train = LSet[r]
        
        # --- 1. Subsample N (Learning Set Size) ---
        if current_N < X_train.shape[0]:
            # Subsample (Used in Analysis A)
            indices = np.random.choice(X_train.shape[0], current_N, replace=False)
            X_train_temp = X_train[indices]
            Y_train_temp = Y_train[indices]
        else:
            # Use full available sample (Used in Analysis B and C)
            X_train_temp = X_train
            Y_train_temp = Y_train

        # --- 2. Slice Features P (Irrelevant Variables) ---
        # X_train_temp is further sliced based on current P
        X_train_final = X_train_temp[:, :current_P]
        Y_train_final = Y_train_temp
        
        # 3. Train and Predict 
        y_pred = estimator_predict(X_train_final, Y_train_final, X_test_sub, method, **kwargs)
        
        # Ensure y_pred is a 1D array before assignment
        all_predictions[:, r] = y_pred.flatten()

    # Compute Mean Values (Averaged over R)
    avg_model = np.mean(all_predictions, axis=1)
    squared_bias = (avg_model - H_X_test) ** 2
    variance = np.var(all_predictions, axis=1)
    expected_error = SIGMA_SQUARED + squared_bias + variance

    return {
        'H_X_test': H_X_test,
        'avg_model': avg_model,
        'bias_sq': squared_bias,
        'variance': variance,
        'mse': expected_error,
    }


# --- MEAN VALUE CALCULATION (Helper for 2.3) ---

def calculate_mean_values(results):
    """Averages the pointwise results over the test set (T points)."""
    mean_bias_sq = np.mean(results['bias_sq'])
    mean_variance = np.mean(results['variance'])
    mean_mse = np.mean(results['mse'])
    
    return {
        'mean_bias_sq': mean_bias_sq,
        'mean_variance': mean_variance,
        'mean_mse': mean_mse,
        'mean_residual_error': SIGMA_SQUARED # Constant
    }


# --- PLOTTING FUNCTIONS FOR 2.3 ---

def plot_mean_results(results, param_range, param_label, filename, output_folder):
    """Generic plotter for N-impact and P-impact (linear X-axis).

    ``results`` is a dict: method -> {'Bias', 'Var', 'MSE'} lists.
    ``param_range`` is the list of N or P values used on the X-axis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(
        f"Impact of {param_label} on mean error components (MSE averaged)",
        fontsize=16,
    )

    colors = ["blue", "green", "red"]

    for i, name in enumerate(results.keys()):
        ax = axes[i]

        # Plot mean Bias^2 and Var
        ax.plot(
            param_range,
            results[name]["Bias"],
            label="Mean squared bias",
            color=colors[0],
            linestyle="--",
        )
        ax.plot(
            param_range,
            results[name]["Var"],
            label="Mean variance",
            color=colors[1],
            linestyle="--",
        )

        # Plot mean total error (MSE)
        ax.plot(
            param_range,
            results[name]["MSE"],
            label="Mean total error (MSE)",
            color=colors[2],
            linewidth=2,
        )

        # Plot residual error (constant)
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
    

def plot_complexity_results(results, ridge_alphas, knn_ks, tree_depths, output_folder):
    """Plot impact of model complexity on mean error components.

    ``results`` is a dict with keys 'Ridge', 'kNN', 'Tree', each mapping to a
    dict with keys 'Param', 'Bias', 'Var', 'MSE'.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(
        "Impact of model complexity on mean error components (MSE averaged)",
        fontsize=16,
    )

    # --- Ridge Plot ---
    ax = axes[0]
    ax.set_title("Ridge regression (complexity decreases as alpha increases)")
    ax.plot(
        results["Ridge"]["Param"],
        results["Ridge"]["Bias"],
        label="Mean Bias^2",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        results["Ridge"]["Param"],
        results["Ridge"]["Var"],
        label="Mean Var",
        color="green",
        linestyle="--",
    )
    ax.plot(
        results["Ridge"]["Param"],
        results["Ridge"]["MSE"],
        label="Mean MSE",
        color="red",
        linewidth=2,
    )
    ax.hlines(
        SIGMA_SQUARED,
        ridge_alphas[0],
        ridge_alphas[-1],
        label="Residual error (sigma^2 = 1)",
        color="gray",
        linestyle=":",
    )
    ax.set_xscale("log")  # Use log scale for alpha
    ax.set_xlabel("Regularization parameter alpha (complexity decreases as alpha increases)")
    ax.set_ylabel("Mean error value")
    ax.legend()
    ax.grid(True, alpha=0.5)

    # --- kNN Plot ---
    ax = axes[1]
    ax.set_title("kNN regression (complexity decreases as k increases)")
    ax.plot(
        results["kNN"]["Param"],
        results["kNN"]["Bias"],
        label="Mean Bias^2",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        results["kNN"]["Param"],
        results["kNN"]["Var"],
        label="Mean Var",
        color="green",
        linestyle="--",
    )
    ax.plot(
        results["kNN"]["Param"],
        results["kNN"]["MSE"],
        label="Mean MSE",
        color="red",
        linewidth=2,
    )
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

    # --- Tree Plot ---
    ax = axes[2]
    ax.set_title("Decision tree regression (complexity increases with max depth)")
    ax.plot(
        results["Tree"]["Param"],
        results["Tree"]["Bias"],
        label="Mean Bias^2",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        results["Tree"]["Param"],
        results["Tree"]["Var"],
        label="Mean Var",
        color="green",
        linestyle="--",
    )
    ax.plot(
        results["Tree"]["Param"],
        results["Tree"]["MSE"],
        label="Mean MSE",
        color="red",
        linewidth=2,
    )
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


# --- MAIN EXECUTION FOR (2.3) ---

def main_2_3():
    # Setup output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_3")
    os.makedirs(output_folder, exist_ok=True)
    
    # Set a fixed seed for reproducibility for the entire experiment
    np.random.seed(100)
    random.seed(100) 

    # --- 1. Setup Fixed Test Set (T=100) ---
    x1_values = np.linspace(-10, 10, NB_TEST_POINTS)
    # X_test_fixed uses P_MAX features initially for compatibility with LSet_pool
    P_MAX = 10 
    X_test_fixed = np.zeros((NB_TEST_POINTS, P_MAX))
    X_test_fixed[:, 0] = x1_values
    for i in range(1, P_MAX):
        X_test_fixed[:, i] = np.random.uniform(-10, 10, NB_TEST_POINTS)
    
    # --- 2. Generate ONE large Learning Set (LS) ---
    # We use N_max and P_max to create one large pool of data to draw subsamples from.
    N_MAX = 500
    print(f"Generating large pool of {NB_REPETITIONS} learning samples (N={N_MAX}, p={P_MAX})...")
    LSet_pool = create_learning_set(P_MAX, N_MAX, NB_REPETITIONS)
    
    # Define models to test (using baseline complexity)
    BASE_MODELS = {
        'Ridge': {'method': 'ridge', 'alpha': 10.0},
        'kNN': {'method': 'knn', 'k': 10}, 
        'Tree': {'method': 'dt', 'max_depth': 5}
    }
    
    
    # --- 3. Analysis A: Impact of Learning Set Size (N) ---
    # N is varied while P is fixed at NB_FEATURES=5 (default)
    print("\n--- Running Analysis A: Impact of N ---")
    N_range = [10, 20, 40, 80, 160, 320, N_MAX]
    N_results = {name: {'N': [], 'Bias': [], 'Var': [], 'MSE': []} for name in BASE_MODELS.keys()}
    
    for N in N_range:
        print(f"  Testing N={N}")
        for name, params in BASE_MODELS.items():
            results = run_pointwise_analysis(
                X_test_fixed, LSet_pool, N=N, P=NB_FEATURES, **params
            )
            mean_results = calculate_mean_values(results)
            
            N_results[name]['N'].append(N)
            N_results[name]['Bias'].append(mean_results['mean_bias_sq'])
            N_results[name]['Var'].append(mean_results['mean_variance'])
            N_results[name]['MSE'].append(mean_results['mean_mse'])

    plot_mean_results(N_results, N_range, 'Learning Set Size (N)', '1_N_impact.pdf', output_folder)
    

    # --- 4. Analysis B: Impact of Model Complexity ---
    # Complexity is varied while N=NB_SAMPLES=80 and P=NB_FEATURES=5 are fixed.
    print("\n--- Running Analysis B: Impact of Complexity ---")
    
    # Complexity Ranges
    RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000] # Low Alpha -> High Complexity
    KNN_KS = [1, 3, 5, 10, 20, 40]             # Low K -> High Complexity
    TREE_DEPTHS = [1, 2, 4, 8, 16]             # High Depth -> High Complexity
    
    C_results = {'Ridge': {'Param': [], 'Bias': [], 'Var': [], 'MSE': []},
                 'kNN': {'Param': [], 'Bias': [], 'Var': [], 'MSE': []},
                 'Tree': {'Param': [], 'Bias': [], 'Var': [], 'MSE': []}}
    
    # Ridge
    for alpha in RIDGE_ALPHAS:
        results = run_pointwise_analysis(
            X_test_fixed, LSet_pool, N=NB_SAMPLES, P=NB_FEATURES, method='ridge', alpha=alpha
        )
        mean_results = calculate_mean_values(results)
        C_results['Ridge']['Param'].append(alpha)
        C_results['Ridge']['Bias'].append(mean_results['mean_bias_sq'])
        C_results['Ridge']['Var'].append(mean_results['mean_variance'])
        C_results['Ridge']['MSE'].append(mean_results['mean_mse'])

    # kNN
    for k in KNN_KS:
        results = run_pointwise_analysis(
            X_test_fixed, LSet_pool, N=NB_SAMPLES, P=NB_FEATURES, method='knn', k=k
        )
        mean_results = calculate_mean_values(results)
        C_results['kNN']['Param'].append(k)
        C_results['kNN']['Bias'].append(mean_results['mean_bias_sq'])
        C_results['kNN']['Var'].append(mean_results['mean_variance'])
        C_results['kNN']['MSE'].append(mean_results['mean_mse'])

    # Tree
    for depth in TREE_DEPTHS:
        results = run_pointwise_analysis(
            X_test_fixed, LSet_pool, N=NB_SAMPLES, P=NB_FEATURES, method='dt', max_depth=depth
        )
        mean_results = calculate_mean_values(results)
        C_results['Tree']['Param'].append(depth)
        C_results['Tree']['Bias'].append(mean_results['mean_bias_sq'])
        C_results['Tree']['Var'].append(mean_results['mean_variance'])
        C_results['Tree']['MSE'].append(mean_results['mean_mse'])

    plot_complexity_results(C_results, RIDGE_ALPHAS, KNN_KS, TREE_DEPTHS, output_folder)


    # --- 5. Analysis C: Impact of Irrelevant Variables (p) ---
    # P is varied while N=NB_SAMPLES=80 and complexity are fixed.
    print("\n--- Running Analysis C: Impact of Irrelevant Features ---")
    P_range = [2, 4, 6, 8, P_MAX] # p=1 irrelevant feature, p=3 irrelevant features, etc.
    P_results = {name: {'P': [], 'Bias': [], 'Var': [], 'MSE': []} for name in BASE_MODELS.keys()}
    
    for P in P_range:
        # P = total number of features (1 relevant + P-1 irrelevant)
        print(f"  Testing P={P}")
        for name, params in BASE_MODELS.items():
            # Pass P to run_pointwise_analysis to slice features
            results = run_pointwise_analysis(
                X_test_fixed, LSet_pool, N=NB_SAMPLES, P=P, **params
            )
            mean_results = calculate_mean_values(results)
            
            P_results[name]['P'].append(P)
            P_results[name]['Bias'].append(mean_results['mean_bias_sq'])
            P_results[name]['Var'].append(mean_results['mean_variance'])
            P_results[name]['MSE'].append(mean_results['mean_mse'])

    plot_mean_results(P_results, P_range, 'Number of Features (p)', '3_P_impact.pdf', output_folder)
    
    print("\nAnalysis 2.3 complete. Check the output folder for PDF files.")
    
if __name__ == '__main__':
    # This entry point is reserved for running the mean value analysis (2.3)
    main_2_3()