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

def writeFileResults(filename, results, lines):
    f = open(filename, 'w')
    results_str = "Results:\n"
    m, n = np.shape(results)
    for i in range(m):
        results_str += lines[i] + " & "
        for j in range(n):
            if j == n - 1:
                results_str += str(results[i][j]) + " \\\ \n\hline\n"
            else:
                results_str += str(results[i][j]) + " & "
    f.write(results_str)

def kFoldCrossValidation(attributes, classes, method, k=2):
    if len(classes)%k != 0:
        print("k must divide the dataset size")
        return -1
    split_size = int(len(classes) / k)
    if method == "dt":
        hp_values = [1, 2, 4, 6, None]
    elif method == "knn":
        hp_values = [1, 2, 25, 125, 500, 899]
    hyperparam = 0
    max_score = 0
    # test hyperparameter values until it stops increasing accuracy
    for p in hp_values:
        score = 0
        for i in range(k):
            # use each of the k subset as a pseudo test sample and compute the average accuracy
            slice_start = split_size * i
            slice_end = split_size * (i + 1)
            test_subset = [attributes[slice_start:slice_end], classes[slice_start:slice_end]]
            ls = [np.append(attributes[:slice_start], attributes[slice_end:], axis=0), np.append(classes[:slice_start], classes[slice_end:])]
            if method == "dt":
                score += testDecisionTree(ls, test_subset, max_depth=p)
            elif method == "knn":
                score += testKNN(ls, test_subset, k=p)
        score /= k
        if score > max_score:
            max_score = score
            hyperparam = p
    return hyperparam

if __name__ == "__main__":
    SAMPLE_SIZE1 = 1200
    LS_SIZE1 = 900
    SAMPLE_SIZE2 = 569
    LS_SIZE2 = 421
    NB_TESTS = 5

    results = np.zeros((2, 4))
    

    # c'est moche mais j'en reste là pour aujourd'hui, c'est temporaire
    # Dataset1 : 
    print("----------Dataset 1 :----------")
    x, y = make_dataset1(SAMPLE_SIZE1, 0)
    x_train = x[:LS_SIZE1]
    y_train = y[:LS_SIZE1]
    x_test = x[LS_SIZE1:]
    y_test = y[LS_SIZE1:]
    # tune hyperparams
    max_depth = kFoldCrossValidation(x_train, y_train, "dt", 6)
    k = kFoldCrossValidation(x, y, "knn", 6)
    # accuracies
    results[0][0], results[1][0] = averageScore(SAMPLE_SIZE1, LS_SIZE1, NB_TESTS, "dt", max_depth, dataset=1)
    results[0][1], results[1][1] = averageScore(SAMPLE_SIZE1, LS_SIZE1, NB_TESTS, "knn", k, dataset=1)

    # Dataset2 :
    print("\n----------Dataset 2 :----------")
    x, y = make_dataset1(SAMPLE_SIZE1, 1234)
    x_train = x[:LS_SIZE1]
    y_train = y[:LS_SIZE1]
    x_test = x[LS_SIZE1:]
    y_test = y[LS_SIZE1:]
    # tune hyperparams
    max_depth = kFoldCrossValidation(x_train, y_train, "dt", 6)
    k = kFoldCrossValidation(x, y, "knn", 6)
    # accuracies
    results[0][2], results[1][2] = averageScore(SAMPLE_SIZE2, LS_SIZE2, NB_TESTS, "dt", max_depth, dataset=2)
    results[0][3], results[1][3] = averageScore(SAMPLE_SIZE2, LS_SIZE2, NB_TESTS, "knn", k, dataset=2)
    print("\nresults :\n", results)
    writeFileResults("mc_results.txt", results, ["Average accuracies", "Standard deviation"])