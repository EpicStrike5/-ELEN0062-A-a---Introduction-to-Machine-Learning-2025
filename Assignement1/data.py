"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.datasets import load_breast_cancer


def make_dataset1(n_points, random_state=None):
    """Generate a 2D dataset

    Parameters
    -----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_points, 2]
        The feature matrix of the dataset
    y : array of shape [n_points]
        The labels of the dataset
    """
    random_state = check_random_state(random_state)
    n_y0 = int(n_points / 2)
    n_y1 = n_points - n_y0

    angle_in_deg = 70
    sin_ = np.sin(np.deg2rad(angle_in_deg))
    cos_ = np.cos(np.deg2rad(angle_in_deg))

    stretch = 3.5

    C = np.array([[cos_ * stretch, -sin_ * stretch], [sin_ / stretch, cos_ / stretch]])
    X = np.r_[
        random_state.randn(n_y0, 2) - np.array([-0.8, 0.0]),
        np.dot(random_state.randn(n_y1, 2), C) + np.array([-0.8, 0.0]),
    ]
    y = np.hstack((np.zeros(n_y0), np.ones(n_y1)))

    permutation = np.arange(n_points)
    random_state.shuffle(permutation)
    return X[permutation], y[permutation]


def make_dataset_breast_cancer(n_points=None, random_state=None):
    """Generate a breast cancer dataset with 569 points and 30 features.

    Parameters
    -----------
    n_points : Ignored argument to be compatible with make_dataset1.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [569, 30]
        The feature matrix of the dataset
    y : array of shape [569]
        The labels of the dataset
    """
    cancer = load_breast_cancer(as_frame=True)

    X = cancer.data.values
    y = cancer.target

    random_state = check_random_state(random_state)
    idx = random_state.choice(len(y), size=len(y), replace=False)
    X, y = X[idx], y[idx]

    return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    make_dataset_breast_cancer()

    n_points = 1200
    make_set = make_dataset1
    fname = "dataset1"
    plt.figure()
    X, y = make_set(n_points)
    X_y0 = X[y == 0]
    X_y1 = X[y == 1]
    plt.scatter(X_y0[:, 0], X_y0[:, 1], color="DodgerBlue", s=6, alpha=0.5)
    plt.scatter(X_y1[:, 0], X_y1[:, 1], color="orange", s=6, alpha=0.5)
    plt.grid(True)
    plt.xlabel("x_0")
    plt.ylabel("x_1")
    plt.title(fname)
    plt.savefig("{}.pdf".format(fname))
