import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os
import random

# --- CONSTANTS AND SETUP ---
NB_REPETITIONS = 200 # R in the protocol (Increased for stability)
NB_FEATURES = 5    # p in the problem statement (only x1 is relevant)
NB_SAMPLES = 80    # N in the problem statement
SIGMA_SQUARED = 1.0 # Sigma^2 = 1 (Residual Error)
NB_TEST_POINTS = 100 # T in the protocol (for plotting x1 values)

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
        
        # Calculate true output h(x)
        H_X = h_x(X)
        
        # Generate noise epsilon ~ N(0, sigma=1)
        epsilon = np.random.normal(0, np.sqrt(SIGMA_SQUARED), X.shape[0])
        
        # Calculate noisy output y = h(x) + epsilon
        Y = H_X + epsilon
        
        LSet.append((X, Y))
    return LSet

# --- ESTIMATOR FUNCTION ---

def get_estimator(method, alpha=1.0, k=5, max_depth=5):
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
    """Trains the estimator and predicts values for X_test, filtering kwargs."""
    # Filter out non-model-specific parameters (N and P are passed down but cause errors here)
    model_kwargs = {k: v for k, v in kwargs.items() if k not in ['N', 'P']}
    
    # Ensure dimensions are correct for Scikit-learn (even if P=1, it needs to be 2D)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
        
    model = get_estimator(method, **model_kwargs)
    model.fit(X_train, Y_train)
    return model.predict(X_test)


# --- POINTWISE ANALYSIS PROTOCOL (Core Function for 2.2) ---

def run_pointwise_analysis(X_test, LSet, method, **kwargs):
    """
    Implements the R-repetition protocol for an entire test set (X_test).
    Calculates the pointwise Bias^2, Variance, and Average Model.
    """
    N_test = X_test.shape[0]
    R = len(LSet)
    
    # Matrix to store all R predictions for all T test points
    all_predictions = np.zeros((N_test, R))
    
    # 1. Calculate True Model (h(x)) for the fixed test set
    H_X_test = h_x(X_test)

    # 2. R-Repetition Protocol
    for r in range(R):
        X_train, Y_train = LSet[r]
        
        # Determine current feature count (P)
        current_P = kwargs.get('P', NB_FEATURES)
        
        # --- FIX: Subsampling/Feature Slicing for fixed N ---
        # Since this is p2q2 (fixed N=80), we just slice features.
        X_train_sub = X_train[:, :current_P]
        Y_train_sub = Y_train  # Y_train is defined directly
        X_test_sub = X_test[:, :current_P]
        
        # Train model and predict on the fixed X_test
        y_pred = estimator_predict(X_train_sub, Y_train_sub, X_test_sub, method, **kwargs)
        
        # Ensure y_pred is a 1D array before assignment
        all_predictions[:, r] = y_pred.flatten() 

    # 3. Compute the Average Model (Mean across the R repetitions)
    avg_model = np.mean(all_predictions, axis=1)

    # 4. Compute Squared Bias (Bias^2)
    squared_bias = (avg_model - H_X_test) ** 2

    # 5. Compute Variance (Var)
    variance = np.var(all_predictions, axis=1)

    # 6. Compute Expected Error (MSE)
    expected_error = SIGMA_SQUARED + squared_bias + variance

    return {
        'H_X_test': H_X_test,
        'avg_model': avg_model,
        'bias_sq': squared_bias,
        'variance': variance,
        'mse': expected_error,
    }

# --- PLOTTING FUNCTIONS FOR 2.2 ---

def plot_decomposition(x_plot, results, output_folder):
    """Creates Plot 1 (2.2): Bias, Variance, MSE Decomposition."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Pointwise Bias-Variance Decomposition (N={NB_SAMPLES}, p={NB_FEATURES})', fontsize=16)

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot decomposition components
        ax.plot(x_plot, data['bias_sq'], label=r'Squared Bias ($\text{Bias}^2$)', color='blue', linestyle='--')
        ax.plot(x_plot, data['variance'], label=r'Variance ($\text{Var}$)', color='green', linestyle='--')
        ax.hlines(SIGMA_SQUARED, x_plot.min(), x_plot.max(), label=r'Residual Error ($\sigma^2=1$)', color='gray', linestyle=':')
        
        # Plot total error
        ax.plot(x_plot, data['mse'], label=r'Total Expected Error (MSE)', color='red', linewidth=2)
        
        ax.set_title(name)
        ax.set_ylabel(r'Error Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel(r'$x_1$')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "1_decomposition_plot.pdf"))
    plt.close(fig)

def plot_average_models(x_plot, results, output_folder):
    """Creates Plot 2 (2.2): Bayes Model vs. Average Models."""
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title(f'Bayes Model vs. Average Models (N={NB_SAMPLES}, p={NB_FEATURES})')
    
    # Plot True Bayes Model (h(x)) - uses data from the first result set
    first_key = list(results.keys())[0]
    ax2.plot(x_plot, results[first_key]['H_X_test'], label=r'Bayes Model ($h(\mathbf{x})$)', color='black', linewidth=3)
    
    # Plot Average Model for each estimator
    for name, data in results.items():
        ax2.plot(x_plot, data['avg_model'], label=f'Average Model ({name})', linewidth=2)
        
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'Prediction Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "2_average_model_plot.pdf"))
    plt.close(fig2)

# --- MAIN EXECUTION FOR (2.2) ---

def main_2_2():
    # Setup output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "empirical_results_2_2")
    os.makedirs(output_folder, exist_ok=True)
    
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # 1. Create the fixed test set X_test
    x1_values = np.linspace(-10, 10, NB_TEST_POINTS)
    X_test = np.zeros((NB_TEST_POINTS, NB_FEATURES))
    X_test[:, 0] = x1_values
    
    # 2. Define models with complexity parameters
    models = {
        r'Ridge ($\alpha=10$ - High Bias)': {'method': 'ridge', 'alpha': 10.0},
        r'kNN ($k=5$ - Moderate)': {'method': 'knn', 'k': 5}, 
        r'Tree ($\max\text{Depth}=5$ - Moderate)': {'method': 'dt', 'max_depth': 5}
    }
    
    # 3. Generate ONE set of R Learning Samples (LS) upfront
    print(f"Generating {NB_REPETITIONS} independent learning samples (N={NB_SAMPLES}, p={NB_FEATURES})...")
    LSet = create_learning_set(NB_FEATURES, NB_SAMPLES, NB_REPETITIONS)
    
    # 4. Run Analysis for all models
    results = {}
    print("--- Running Pointwise Bias-Variance Analysis ---")
    for name, params in models.items():
        print(f"Analyzing {name}...")
        # Note: We do not pass N or P as kwargs here, but run_pointwise_analysis defaults to constants
        results[name] = run_pointwise_analysis(X_test, LSet, **params)
    
    # 5. Generate Plots and Save to PDF
    print(f"Generating and saving plots to {output_folder}...")
    plot_decomposition(x1_values, results, output_folder)
    plot_average_models(x1_values, results, output_folder)
    print("Analysis complete. Check the output folder for PDF files.")

if __name__ == '__main__':
    main_2_2()