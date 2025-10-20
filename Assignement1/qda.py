"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q3. Linear/Quadratic Discriminant Analysis (LDA/QDA).
"""
import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin
# --- Imports for the main experiment script ---
from data import make_dataset1, make_dataset_breast_cancer
from plot import plot_boundary


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, lda=False):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda

        # ====================

        # We identify the unique classes in the target variable y.
        self.classes_ = np.unique(y)
        # Get the number of samples (rows) and features (columns).
        n_samples, n_features = X.shape

        # We create dictionaries to store the priors, means, and covariance matrices for each class.
        # trailing underscore is sickit best practices
        self.priors_ = {}  # Will store the prior probability pi_k for each class k.
        self.means_ = {}   # Will store the mean vector µ_k for each class k.
        self.covs_ = {}    # Will store the covariance matrix Sigma_k for each class k.

        for k in self.classes_:
            # Select all data points that belong to the current class k.
            X_k = X[y == k]
            self.priors_[k] = len(X_k) / n_samples          # Calculate the prior: (number of samples in class k) / (total samples).
            self.means_[k] = np.mean(X_k, axis=0)           # Calculate the mean vector for class k by averaging each feature column.
            self.covs_[k] = np.cov(X_k, rowvar=False)       # Calculate the covariance matrix for class k. `rowvar=False` treats columns as variables.

        # Only if Lda is true we pool the covariance matrices. we calculate hte pooled covariance matrix and then assign it to both classes.
        if self.lda:
            k0, k1 = self.classes_
            pooled_cov = self.priors_[k0] * self.covs_[k0] + self.priors_[k1] * self.covs_[k1]
            self.covs_[k0] = self.covs_[k1] = pooled_cov

        # Pre-compute the inverses and determinants of the covariance matrices for each class to speed up predictions.
        self.inv_covs_ = {k: np.linalg.inv(cov) for k, cov in self.covs_.items()}
        self.det_covs_ = {k: np.linalg.det(cov) for k, cov in self.covs_.items()}
        # ====================

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
        # ====================

    

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # This matrix will store the log-discriminant value for each sample and each class.
        log_discriminants = np.zeros((X.shape[0], len(self.classes_)))
        # Loop over each class 
        for i, k in enumerate(self.classes_):
           
            diff = X - self.means_[k] # difference vector (x - µ_k) for all samples.
            
            mahalanobis = np.einsum('ij,ji->i', np.dot(diff, self.inv_covs_[k]), diff.T) #  Mahalanobis distance: (x-µ_k)^T * Sigma_k^-1 * (x-µ_k).
            
            log_det = np.log(self.det_covs_[k]) # Get the pre-computed log of the determinant of the covariance matrix. We use log to avoid e and numerical instability.
            
            log_discriminants[:, i] = np.log(self.priors_[k]) - 0.5 * (log_det + mahalanobis) # log of the discriminant function: log(pi_k * f_k(x)).

        # We use the log-sum-exp trick. First, subtract the max log value.
        # see https://en.wikipedia.org/wiki/LogSumExp for more details.
        max_log = np.max(log_discriminants, axis=1, keepdims=True)
        # Exponentiate the stabilized values.
        exp_logs = np.exp(log_discriminants - max_log)
        # Normalize to get final probabilities that sum to 1 for each sample.
        return exp_logs / np.sum(exp_logs, axis=1, keepdims=True)
        # ====================



if __name__ == "__main__":
    # --- Setup Output Directory ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "qda_plots")
    
    # --- Experiment Parameters ---
    N_GENERATIONS = 5
    N_POINTS_DS1 = 1200
    FIXED_RANDOM_STATE = 42
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output plots will be saved to: {output_folder}")
    
    ds1_scores = {'LDA': [], 'QDA': []}
    ds2_scores = {'LDA': [], 'QDA': []}

    print("--- Starting QDA/LDA Experiments ---")
    for i in range(N_GENERATIONS):
        print(f"\nRunning Generation {i+1}/{N_GENERATIONS}...")
        
        # Load and split Dataset 1 (Gaussian)
        X1, y1 = make_dataset1(n_points=N_POINTS_DS1, random_state=FIXED_RANDOM_STATE)
        split_idx1 = int(len(X1) * 0.75)
        X1_train, y1_train = X1[:split_idx1], y1[:split_idx1]
        X1_test, y1_test = X1[split_idx1:], y1[split_idx1:]

        # Load and split Dataset 2 (Breast Cancer)
        X2, y2 = make_dataset_breast_cancer(random_state=FIXED_RANDOM_STATE)
        split_idx2 = int(len(X2) * 0.75)
        X2_train, y2_train = X2[:split_idx2], y2[:split_idx2]
        X2_test, y2_test = X2[split_idx2:], y2[split_idx2:]

        for lda_mode, model_name in [(True, 'LDA'), (False, 'QDA')]:
            # Train and score on Dataset 1
            model1 = QuadraticDiscriminantAnalysis().fit(X1_train, y1_train, lda=lda_mode)
            ds1_scores[model_name].append(model1.score(X1_test, y1_test))

            # Train and score on Dataset 2
            model2 = QuadraticDiscriminantAnalysis().fit(X2_train, y2_train, lda=lda_mode)
            ds2_scores[model_name].append(model2.score(X2_test, y2_test))

            # Plot boundaries for Dataset 1 on the last run
            if i == N_GENERATIONS - 1:
                filepath = os.path.join(output_folder, f"boundary_{model_name}")
                plot_title = f"{model_name} Decision Boundary"
                plot_boundary(filepath, model1, X1_train, y1_train, title=plot_title)
                print(f"Saved plot to {filepath}.pdf")

    # Calculate and print the final average scores and std deviations
    print("\n\n--- Experiment Results (Averages over 5 runs) ---")
    print("\n--- Dataset 1 (Gaussian) ---")
    for name in ['LDA', 'QDA']:
        mean, std = np.mean(ds1_scores[name]), np.std(ds1_scores[name])
        print(f"{name:<3} | Avg. Accuracy = {mean:.4f} | Std. Dev. = {std:.4f}")

    print("\n--- Dataset 2 (Breast Cancer) ---")
    for name in ['LDA', 'QDA']:
        mean, std = np.mean(ds2_scores[name]), np.std(ds2_scores[name])
        print(f"{name:<3} | Avg. Accuracy = {mean:.4f} | Std. Dev. = {std:.4f}")

    print("\n--- All experiments complete ---")
