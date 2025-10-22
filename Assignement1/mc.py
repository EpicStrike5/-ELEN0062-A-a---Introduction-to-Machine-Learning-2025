"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q4. Method comparison.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from data import make_dataset1, make_dataset_breast_cancer

def initData(sample_size, ls_size, random_state=None, dataset=1):
    if dataset == 1:
        data = make_dataset1(sample_size, random_state=random_state)
    elif dataset == 2:
        data = make_dataset_breast_cancer(sample_size, random_state=random_state)
    ls_points = np.array(data[0][:ls_size][:])
    ls_classes = np.array(data[1][:ls_size][:])
    ts_points = np.array(data[0][ls_size:][:])
    ts_classes = np.array(data[1][ls_size:][:])
    ls = [ls_points, ls_classes]
    ts = [ts_points, ts_classes]
    return ls, ts

def testDecisionTree(ls, ts, max_depth=None):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    result = dt.fit(ls[0], ls[1])
    score = dt.score(ts[0], ts[1])
    return score

def testKNN(ls, ts, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(ls[0], ls[1])
    score = knn.score(ts[0], ts[1])
    return score

def averageScore(sample_size, ls_size, nb_tests, method, hyperparameter, dataset=1):
    scores = np.zeros(nb_tests)
    for i in range(nb_tests):
        ls2, ts2 = initData(sample_size, ls_size, random_state=i, dataset=dataset)
        if method == "dt":
            scores[i] = testDecisionTree(ls2, ts2, max_depth=hyperparameter)
        elif method == "knn":
            scores[i] = testKNN(ls2, ts2, k=hyperparameter)
    score_avg = np.mean(scores)
    score_std = np.std(scores)
    print("\nmethod ", method, ": \nAverage score ", score_avg, "\nStandard deviation ", score_std)
    return score_avg, score_std

def kFoldCrossValidation(attributes, classes, method, k=2):
    if len(classes)%k != 0:
        print("k must divide the dataset size")
        return -1
    split_size = int(len(classes) / k)
    max_score = 0
    score = 0
    hyperparam = 1
    # test hyperparameter values until it stops increasing accuracy
    while(score >= max_score):
        max_score = score
        score = 0
        hyperparam += 1
        for i in range(k):
            # use each of the k subset as a pseudo test sample and compute the average accuracy
            slice_start = split_size * i
            slice_end = split_size * (i + 1)
            test_subset = [attributes[slice_start:slice_end], classes[slice_start:slice_end]]
            ls = [np.append(attributes[:slice_start], attributes[slice_end:], axis=0), np.append(classes[:slice_start], classes[slice_end:])]
            if method == "dt":
                score += testDecisionTree(ls, test_subset, max_depth=hyperparam)
            elif method == "knn":
                score += testKNN(ls, test_subset, k=hyperparam)
        score /= k
    print("method ", method, ": ", max_score, "accuracy for hyperparameter = ", hyperparam)
    return hyperparam

if __name__ == "__main__":
    SAMPLE_SIZE1 = 1200
    LS_SIZE1 = 900
    SAMPLE_SIZE2 = 569
    LS_SIZE2 = 421
    NB_TESTS = 5

    # c'est moche mais j'en reste là pour aujourd'hui, c'est temporaire
    # Dataset1 : 
    print("----------Dataset 1 :----------\n")
    x, y = make_dataset1(SAMPLE_SIZE1, 1234)
    x_train = x[:LS_SIZE1]
    y_train = y[:LS_SIZE1]
    x_test = x[LS_SIZE1:]
    y_test = y[LS_SIZE1:]
    # tune hyperparams
    max_depth = kFoldCrossValidation(x_train, y_train, "dt", 60)
    k = kFoldCrossValidation(x, y, "knn", 60)
    # accuracies
    averageScore(SAMPLE_SIZE1, LS_SIZE1, NB_TESTS, "dt", max_depth, dataset=1)
    averageScore(SAMPLE_SIZE1, LS_SIZE1, NB_TESTS, "knn", k, dataset=1)

    # Dataset2 :
    print("----------Dataset 2 :----------\n")
    x, y = make_dataset1(SAMPLE_SIZE1, 1234)
    x_train = x[:LS_SIZE1]
    y_train = y[:LS_SIZE1]
    x_test = x[LS_SIZE1:]
    y_test = y[LS_SIZE1:]
    # tune hyperparams
    max_depth = kFoldCrossValidation(x_train, y_train, "dt", 60)
    k = kFoldCrossValidation(x, y, "knn", 60)
    # accuracies
    averageScore(SAMPLE_SIZE2, LS_SIZE2, NB_TESTS, "dt", max_depth, dataset=2)
    averageScore(SAMPLE_SIZE2, LS_SIZE2, NB_TESTS, "knn", k, dataset=2)