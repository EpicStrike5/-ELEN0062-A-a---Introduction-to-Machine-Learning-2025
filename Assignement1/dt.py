"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q1. Decision Trees.
"""

import numpy as np
from matplotlib import pyplot as plt
import os

from data import make_dataset1
from sklearn.tree import DecisionTreeClassifier
from plot import plot_boundary

# --- Setup Output Directory ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "dt_plots")

def run_dt_analysis(n_points, n_generations, train_split_ratio, depth_values_vector, fixed_random_state):
    
    print("\n" + "="*80)
    print("STARTING Q1: DECISION TREE MULTI-GENERATION ANALYSIS")
    print("="*80)

    X, y = make_dataset1(n_points=n_points, random_state=fixed_random_state)

    split_index = int(len(X) * train_split_ratio) 

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    print(f"Training set: {len(y_train)} samples")
    print(f"Test set    : {len(y_test)} samples")

    
    for depth in depth_values_vector:
        depth_str = "unlimited" if depth is None else str(depth)

        filename = f"MC_dt_boundary_depth={depth_str}"
        filepath = os.path.join(output_folder, filename)

        run_decision_tree_experiment(X_train, y_train, X_test, y_test,
                                     max_depth=depth,
                                     output_filepath=filepath,
                                     depth_label=depth_str)
    return filename, filepath
    


#bblablabalbalbal

def run_decision_tree_experiment(X_train, y_train, X_test, y_test, max_depth, output_filepath, depth_label):
    
    # --- Model training ---
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=FIXED_RANDOM_STATE)
    dt.fit(X_train, y_train)

    # --- Title utilizing depth_label ---
    plot_title = f"Decision Tree Boundary (max_depth={depth_label})"
    
    # --- Saving the files ---  
    plot_boundary(output_filepath, dt, X_train, y_train, title=plot_title)
    print(f"Saved plot to {output_filepath}.pdf")

    # --- Evaluation ---
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    actual_depth = dt.get_depth()

    # --- Print results ---
    print(f"Train Accuracy : {train_score:.4f}")
    print(f"Test Accuracy  : {test_score:.4f}")
    if max_depth is None:
        print(f"Actual Depth   : {actual_depth}")


if __name__ == "__main__":
    N_POINTS = 1200
    DEPTH_VALUES = [1, 2, 4, 6, None]
    FIXED_RANDOM_STATE = 42
    TRAIN_SPLIT_RATIO = 0.75

    #We create output folder, if it exists --> SKIP 
    os.makedirs(output_folder, exist_ok=True)

    print(f"Generating a fixed dataset with {N_POINTS} points (random_state={FIXED_RANDOM_STATE})...")
    X, y = make_dataset1(n_points=N_POINTS, random_state=FIXED_RANDOM_STATE)

    split_index = int(len(X) * TRAIN_SPLIT_RATIO) # we split 75% train, 25% test

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set    : {len(y_test)} samples")

    print("\n--- Starting Decision Tree Experiments ---")
    for depth in DEPTH_VALUES:
        depth_str = "unlimited" if depth is None else str(depth)
        print(f"\nRunning experiment for depth = {depth_str}...")

        filename = f"dt_boundary_depth={depth_str}"
        filepath = os.path.join(output_folder, filename)

        run_decision_tree_experiment(X_train, y_train, X_test, y_test,
                                     max_depth=depth,
                                     output_filepath=filepath,
                                     depth_label=depth_str)

    print("\n--- All experiments complete ---")