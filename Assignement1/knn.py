"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q2. k-Nearest Neighbors (kNN).
"""

import numpy as np
from matplotlib import pyplot as plt
import os

from data import make_dataset1
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary

# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join that path with the desired folder name
output_folder = os.path.join(script_dir, "knn_plots")


if __name__ == "__main__":
    # --- Parameters ---
    N_NEIGHBORS_VALUES = [1, 5, 25, 125, 500, 899]
    N_GENERATIONS = 5
    N_POINTS = 1200
    split_index = 900

    #We create output folder, if it exists --> SKIP 
    os.makedirs(output_folder, exist_ok=True)

    #We create a dictionnary to assign to the different values of Neighbors each test results 
    test_scores = {k: [] for k in N_NEIGHBORS_VALUES}
    print("--- Starting k-NN Experiment ---")

    #Main Loop
    for i in range(N_GENERATIONS):
        print(f"\nRunning Generation {i+1}/{N_GENERATIONS}...")
        full_data = make_dataset1(n_points=N_POINTS)
        X_train = full_data[0][:split_index]
        y_train = full_data[1][:split_index]
        X_test = full_data[0][split_index:]
        y_test = full_data[1][split_index:]

        for k in N_NEIGHBORS_VALUES:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            score = knn.score(X_test, y_test)
            test_scores[k].append(score)

            if i == 0:
                plot_title = f"k-NN Decision Boundary (k={k})"
                filename = f"knn_boundary_k={k}"
                filepath = os.path.join(output_folder, filename)
                plot_boundary(filepath, knn, X_train, y_train, title=plot_title)
                print(f"Saved plot to {filepath}.pdf")

    # Raw test scores, for testing and verifying
    print("\n\n--- Raw Test Scores (5 Generations) ---")
    for k, scores in test_scores.items():
        formatted_scores = [f"{score:.4f}" for score in scores]
        print(f"k = {k:<3} | Scores = {formatted_scores}")

    # Results with average etc....
    print("\n--- Experiment Results (Averages) ---")
    print("Average test accuracies and standard deviations over 5 runs:\n")
    for k, scores in test_scores.items():
        mean_accuracy = np.mean(scores)
        std_deviation = np.std(scores)
        print(f"k = {k:<3} | Avg. Accuracy = {mean_accuracy:.4f} | Std. Dev. = {std_deviation:.4f}")