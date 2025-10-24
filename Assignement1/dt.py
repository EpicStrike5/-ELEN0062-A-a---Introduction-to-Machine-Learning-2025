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
def testDecisionTree(ls, ts, plot_name, max_depth=None, plot=False):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    result = dt.fit(ls[0], ls[1])
    if plot is True:
        plot_boundary(plot_name, dt, ls[0], ls[1])
    score = dt.score(ts[0], ts[1])
    # print("Score ", plot_name, score)
    # if max_depth is None:
    #     print("Depth ", plot_name, " = ", dt.get_depth())
    return score

def initData(sample_size, ls_size, random_state=None):
    data = make_dataset1(sample_size, random_state=random_state)
    
    ls_points = np.array(data[0][:ls_size][:])
    ls_classes = np.array(data[1][:ls_size][:])
    ts_points = np.array(data[0][ls_size:][:])
    ts_classes = np.array(data[1][ls_size:][:])
    ls = [ls_points, ls_classes]
    ts = [ts_points, ts_classes]
    return ls, ts

def averageScore(sample_size, ls_size, score, nb_tests, depth=None, random_state=1):
    scores = np.zeros(nb_tests)
    scores[0] = score
    for i in range(nb_tests - 1):
        ls2, ts2 = initData(sample_size, ls_size, random_state=i + random_state)
        name = 'new dataset' + str(i + 1)
        scores[i + 1] = testDecisionTree(ls2, ts2, name, max_depth=depth)
    score_avg = np.mean(scores)
    score_sd = np.std(scores)
    return score_avg, score_sd


if __name__ == "__main__":
    pass  # Make your experiments here
    SAMPLE_SIZE = 1200
    LS_SIZE = 900
    NB_TESTS = 5
    FIXED_RANDOM_STATE = 42

    depths = [1, 2, 4, 6]
    ls, ts = initData(SAMPLE_SIZE, LS_SIZE, random_state=FIXED_RANDOM_STATE)
    for i in range(5):  
        if i in range(len(depths)): 
            depth = depths[i]
            name = 'dt_' + str(depth)
            score = testDecisionTree(ls, ts, name, max_depth=depth, plot=True)
            score_avg, score_sd = averageScore(SAMPLE_SIZE, LS_SIZE, score, NB_TESTS, depth)
            print("Average score ", name, score_avg, "\nStandard deviation ", name, score_sd, "\n")
        else:
            name = 'dt_none' 
            score = testDecisionTree(ls, ts, name, plot=True)
            score_avg, score_sd = averageScore(SAMPLE_SIZE, LS_SIZE, score, NB_TESTS)
            print("Average score ", name, score_avg, "\nStandard deviation ", name, score_sd)