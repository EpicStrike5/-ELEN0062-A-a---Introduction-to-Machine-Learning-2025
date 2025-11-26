import numpy as np
import sklearn.linear_model as lm
import sklearn.neighbors as knn
import sklearn.tree as tree
import matplotlib.pyplot as plt
import os

NB_LS = 100 # use 100 or 500
NB_FEATURES = 1 # use 5 but only 1 is useful for 2.2
NB_SAMPLES = 80 # for tests only, statement says to use 80
X_train = np.random.uniform(-10, 10, NB_LS) #keep the same random values for all training computations

def createLS(nbfeatures, nbsamples, sample_features=None,nb_LS=NB_LS):
    # sample_features: matrix of dimension (nbsamples, nbfeatures), LSx
    #nb_LS: number of learning sample in the learning set
    LSet = np.empty(nb_LS, dtype=object)
    for i in range(nb_LS):
        ls = np.zeros((nbsamples, nbfeatures + 1))
        for j in range(nbsamples):
            if(sample_features is not None):
                x = sample_features[j, :nbfeatures]
            else:
                x = np.random.uniform(-10, 10, nbfeatures)
            e = np.random.normal(0, 1)
            y = np.sin(2 * x[0]) + x[0] * np.cos(x[0] - 1) + e
            s = np.append(x, y)
            ls[j, :] = s
        LSet[i] = ls
    return LSet

def squared_bias(x0,x_train,Lset, nbfeatures, nbsamples, nb_ls, method):
    fLS =[]
    for i in range(nb_ls):
        ls = Lset[i]
        yr = estimator(x_train[i], ls,method)
        fLS.append(yr)
    f_mean = np.mean(fLS)
    bias2 = (f_mean - (np.sin(2 * x0) + x0 * np.cos(x0 - 1)))**2
    return bias2

def variance(x0,x_train,Lset, nbfeatures, nbsamples, nb_ls, method):
    fLS =[]
    for i in range(nb_ls):
        ls = Lset[i]
        yr = estimator(x_train[i], ls,method)
        fLS.append(yr)
    f_mean = np.mean(fLS)
    var = np.mean((fLS - f_mean)**2)
    return var

def res_error(x0,x_train,Lset, nbfeatures, nbsamples, nb_ls, method):
    error = 1 + squared_bias(x0,x_train,Lset, nbfeatures, nbsamples, nb_ls, method) + variance(x0,x_train,Lset, nbfeatures, nbsamples, nb_ls, method)
    return error
#note to self: on a besoin que de fLS, 
def estimator(x, LS, method="ridge"): 
    # function to compute the estimated value of the estimator of a given method, based on LS and at point x 
    # methods: "ridge", "kNN", "dt"
    if(method == "ridge"):
        func = lm.Ridge(alpha=1.0)
        func.fit(LS[:,0].reshape(-1, 1),LS[:,1])
        return func.predict([[x]])
    elif(method == "kNN"):
        func = knn.KNeighborsRegressor(n_neighbors=3)
        func.fit(LS[:,0].reshape(-1, 1),LS[:,1])
        return func.predict([[x]])
    elif(method == "dt"):
        func = tree.DecisionTreeRegressor(max_depth=5)
        func.fit(LS[:,0].reshape(-1, 1),LS[:,1])
        return func.predict([[x]])

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "error_plots")
    os.makedirs(output_folder, exist_ok=True)

    LearningSet = createLS(NB_FEATURES, NB_SAMPLES, nb_LS=NB_LS)
    x1_values = np.linspace(-10, 10, 100)
    bias_ridge = []
    var_ridge = []
    exp_err_ridge = []
    bias_knn = []
    var_knn = []    
    exp_err_knn = []
    bias_dt = []
    var_dt = []
    exp_err_dt = []

    for x1 in x1_values:
        bias_ridge.append(squared_bias(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES, NB_LS, "ridge"))
        var_ridge.append(variance(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS, "ridge"))
        exp_err_ridge.append(res_error(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS, "ridge"))
        
        bias_knn.append(squared_bias(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS, "kNN"))
        var_knn.append(variance(x1, X_train, LearningSet, NB_FEATURES, NB_SAMPLES, NB_LS, "kNN"))    
        exp_err_knn.append(res_error(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS,"kNN"))
        
        bias_dt.append(squared_bias(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS,"dt"))
        var_dt.append(variance(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS, "dt"))
        exp_err_dt.append(res_error(x1,X_train,LearningSet, NB_FEATURES, NB_SAMPLES,NB_LS, "dt"))
    
    # plot of the errors in the three methods
    plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.plot(x1_values, bias_ridge, label='Ridge B2')
    plt.plot(x1_values, var_ridge, label='Ridge Var')
    plt.plot(x1_values, exp_err_ridge, label='Ridge residual Error')
    plt.title('Ridge Regression Bias-variance')
    plt.xlabel('x1')
    plt.ylabel('Error') 
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(x1_values, bias_knn, label='kNN B^2')
    plt.plot(x1_values, var_knn, label='kNN var')
    plt.plot(x1_values, exp_err_knn, label='kNN residual Error')
    plt.title('kNN Regression Bias-Variance ')
    plt.xlabel('x1')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(x1_values, bias_dt, label='DT B^2')
    plt.plot(x1_values, var_dt, label='DT var')
    plt.plot(x1_values, exp_err_dt, label='DT residual Error')
    plt.title('Decision Tree Regression Bias-Variance ')
    plt.xlabel('x1')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "bias_variance.pdf"))
    

        