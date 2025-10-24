"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q1. Decision Trees.
"""

import numpy as np
from matplotlib import pyplot as plt
import os  # Added for directory management

from data import make_dataset1
from sklearn.tree import DecisionTreeClassifier
from plot import plot_boundary

# --- Setup Output Directory (like knn.py) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "dt_plots")


# Put your functions here
def testDecisionTree(ls, ts, plot_name, output_folder, max_depth=None, plot=False):
    """
    Fits a Decision Tree, scores it on the test set, and optionally plots
    both learning and test set boundaries to the output_folder.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    dt.fit(ls[0], ls[1])
    
    if plot is True:
        # Create a consistent title based on depth
        depth_str = "None" if max_depth is None else str(max_depth)
        
        # --- Plot for Training (Learning) Set ---
        plot_title_train = f"DT Decision Boundary (depth={depth_str}) - Learning Set"
        filename_train = f"{plot_name}_learning_set"
        filepath_train = os.path.join(output_folder, filename_train)
        plot_boundary(filepath_train, dt, ls[0], ls[1], title=plot_title_train)
        print(f"Saved plot to {filepath_train}.pdf")

        # --- Plot for Test Set ---
        plot_title_test = f"DT Decision Boundary (depth={depth_str}) - Test Set"
        filename_test = f"{plot_name}_test_set"
        filepath_test = os.path.join(output_folder, filename_test)
        plot_boundary(filepath_test, dt, ts[0], ts[1], title=plot_title_test)
        print(f"Saved plot to {filepath_test}.pdf")

    score = dt.score(ts[0], ts[1])
    # if max_depth is None:
    #     print("Depth ", plot_name, " = ", dt.get_depth())
    return score

def initData(sample_size, ls_size, random_state=None):
    """Initializes and splits the dataset."""
    data = make_dataset1(sample_size, random_state=random_state)
    
    ls_points = np.array(data[0][:ls_size][:])
    ls_classes = np.array(data[1][:ls_size][:])
    ts_points = np.array(data[0][ls_size:][:])
    ts_classes = np.array(data[1][ls_size:][:])
    ls = [ls_points, ls_classes]
    ts = [ts_points, ts_classes]
    return ls, ts

def averageScore(sample_size, ls_size, score, nb_tests, output_folder, depth=None, random_state=1):
    """Calculates average score and std dev over multiple runs."""
    scores = np.zeros(nb_tests)
    scores[0] = score
    for i in range(nb_tests - 1):
        ls2, ts2 = initData(sample_size, ls_size, random_state=i + random_state)
        name = 'new_dataset' + str(i + 1)
        # Call testDecisionTree without plotting, but pass output_folder
        scores[i + 1] = testDecisionTree(ls2, ts2, name, output_folder, max_depth=depth, plot=False)
    
    score_avg = np.mean(scores)
    score_sd = np.std(scores)
    return score_avg, score_sd


if __name__ == "__main__":
    # --- Parameters ---
    SAMPLE_SIZE = 1200
    LS_SIZE = 900
    NB_TESTS = 5
    FIXED_RANDOM_STATE = 42
    depths = [1, 2, 4, 6]

    # We create output folder, if it exists --> SKIP 
    os.makedirs(output_folder, exist_ok=True)
    
    print("--- Starting Decision Tree Experiment ---")

    # Generate the initial dataset for the plotting run
    ls, ts = initData(SAMPLE_SIZE, LS_SIZE, random_state=FIXED_RANDOM_STATE)

    # Loop 5 times (4 for depths, 1 for depth=None)
    for i in range(5):  
        if i in range(len(depths)): 
            depth = depths[i]
            name = f'dt_depth={depth}'
            # Call with plot=True to generate plots for this run
            score = testDecisionTree(ls, ts, name, output_folder, max_depth=depth, plot=True)
            # Calculate average scores (will not plot internally)
            score_avg, score_sd = averageScore(SAMPLE_SIZE, LS_SIZE, score, NB_TESTS, output_folder, depth, random_state=FIXED_RANDOM_STATE)
            print(f"\nResults for depth = {depth}:")
            print(f"  Avg. Accuracy = {score_avg:.4f} | Std. Dev. = {score_sd:.4f}")
        else:
            name = 'dt_depth=None'
            # Call with plot=True for the depth=None case
            score = testDecisionTree(ls, ts, name, output_folder, max_depth=None, plot=True)
            # Calculate average scores (will not plot internally)
            score_avg, score_sd = averageScore(SAMPLE_SIZE, LS_SIZE, score, NB_TESTS, output_folder, random_state=FIXED_RANDOM_STATE)
            print(f"\nResults for depth = None:")
            print(f"  Avg. Accuracy = {score_avg:.4f} | Std. Dev. = {score_sd:.4f}")

    print("\n--- Experiment Complete ---")
    print(f"Plots saved to {output_folder}")
