"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q2. k-Nearest Neighbors (kNN).
"""

import numpy as np
from matplotlib import pyplot as plt
import os  # create the directory for saving files

from data import make_dataset1
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary 


if __name__ == "__main__":
    # --- Parameters ---
    N_NEIGHBORS_VALUES = [1, 5, 25, 125, 500, 899]
    N_GENERATIONS = 5
    N_POINTS = 1200
    split_index = 900

    # Create the output folder for the PDFs
    output_folder = "knn_plots"
    os.makedirs(output_folder, exist_ok=True)

    test_scores = {k: [] for k in N_NEIGHBORS_VALUES}
    print("--- Starting k-NN Experiment ---")

    for i in range(N_GENERATIONS):
        print(f"\nRunning Generation {i+1}/{N_GENERATIONS}...")
        full_data = make_dataset1(N_POINTS)
        X_train = full_data[0][:split_index]
        y_train = full_data[1][:split_index]
        X_test = full_data[0][split_index:]
        y_test = full_data[1][split_index:]

        for k in N_NEIGHBORS_VALUES:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            score = knn.score(X_test, y_test)
            test_scores[k].append(score)

            # Plotting logic for the first generation ONLY
            if i == 0:
                # 1. Define the plot title
                plot_title = f"k-NN Decision Boundary (k={k})"
                
                # 2. Define the filename WITHOUT the .pdf extension
                filename = f"knn_boundary_k={k}"
                filepath = os.path.join(output_folder, filename)
                
                # 3. Call your function and pass it the full path.
                #    Your function will add ".pdf" and save the file.
                plot_boundary(filepath, knn, X_train, y_train, title=plot_title)
                print(f"Saved plot to {filepath}.pdf")

    # --- Raw test scores ---
    print("\n\n--- Raw Test Scores (5 Generations) ---")
    for k, scores in test_scores.items():
        # This formats each score to 4 decimal places for readability
        formatted_scores = [f"{score:.4f}" for score in scores]
        print(f"k = {k:<3} | Scores = {formatted_scores}")


    # --- Reporting Results ---
    print("\n--- Experiment Results ---")
    print("Average test accuracies and standard deviations over 5 runs:\n")
    for k, scores in test_scores.items():
        mean_accuracy = np.mean(scores)
        std_deviation = np.std(scores)
        print(f"k = {k:<3} | Avg. Accuracy = {mean_accuracy:.4f} | Std. Dev. = {std_deviation:.4f}")