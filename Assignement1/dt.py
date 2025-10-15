"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q1. Decision Trees.
"""

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1
from sklearn.tree import DecisionTreeClassifier
from plot import plot_boundary


# Put your functions here
def decisionTree(ls, ts, plot_name, max_depth=None):
    dt = DecisionTreeClassifier(max_depth=max_depth)
    result = dt.fit(ls[0], ls[1])
    plot_boundary(plot_name, dt, ls[0], ls[1])
    print("Score learning ", plot_name, dt.score(ls[0], ls[1]))
    print("Score ", plot_name, dt.score(ts[0], ts[1]))
    if max_depth not in [1, 2, 4, 6]:
        print("Depth = ", dt.get_depth())
    


if __name__ == "__main__":
    pass  # Make your experiments here
    data = make_dataset1(1200)
    ls_points = np.array(data[0][:900][:])
    ls_classes = np.array(data[1][:900][:])
    ts_points = np.array(data[0][900:][:])
    ts_classes = np.array(data[1][900:][:])
    ls = [ls_points, ls_classes]
    ts = [ts_points, ts_classes]
    # sprint(ts[0],'\n', ts[1])
    for i in [1, 2, 4, 6]:    
        name = 'dt_' + str(i)
        decisionTree(ls, ts, name, max_depth=i)
    name = 'dt_none'    
    decisionTree(ls, ts, name)