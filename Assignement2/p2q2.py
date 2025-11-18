import numpy as np

NB_LS = 100 # use 100 or 500
NB_FEATURES = 5
NB_SAMPLES = 10 # for tests only, statement says to use 80

def createLS(nbfeatures, nbsamples, sample_features=None):
    # sample_features: matrix of dimension (nbsamples, nbfeatures), LSx

    LS = np.zeros((nbsamples, nbfeatures + 1))
    for i in range(nbsamples):
        if(sample_features is not None):
            x = sample_features[i, :nbfeatures]
        else:
            x = np.random.uniform(-10, 10, nbfeatures)
        # print(x)
        e = np.random.normal(0, 1)
        y = np.sin(2 * x[0]) + x[0] * np.cos(x[0] - 1) + e
        s = np.append(x, y)
        LS[i, :] = s
    # print(LS)
    return LS

def variance(x0, nbfeatures, nbsamples, method, sample_features=None):
    # x0: array of length nb_features, fixed point at which we estimate the error (decomposed in squared bias and variance)
    # if sample_features is not None, the function computes the conditional variance knowing it

    estimator_values = np.zeros(NB_LS)
    # print(estimator_values)
    for i in range(NB_LS):
        LS = createLS(nbfeatures, nbsamples, sample_features)
        estimator_values[i] = estimator(x0, LS, method)
    var = np.var(estimator_values)
    print(var)
    return var

def estimator(x, LS, method):
    # function to compute the estimated value of the estimator of a given method, based on LS and at point x
    # NOT DONE YET

    return np.random.uniform(0, 1)


test_matrix = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
# createLS(5, 10)
variance(np.array([0, 1, 2, 3, 4]), 5, 10, ".", matrix)