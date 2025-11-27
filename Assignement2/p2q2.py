import numpy as np
import sklearn.linear_model as lm
import sklearn.neighbors as knn
import sklearn.tree as tree
import matplotlib.pyplot as plt
import os

NB_LS = 100 # use 100 or 500
NB_FEATURES = 1 # use 5 but only 1 is useful for 2.2
NB_SAMPLES = 80 # for tests only, statement says to use 80

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
    Yr =[]
    for i in range(nb_ls):
        ls = Lset[i]
        yr = estimator(x_train[i], ls,method)
        Yr.append(yr)
    arg = Yr - np.mean(Yr)
    var = np.mean(arg**2)
    return var

def res_error():
    return 1

def exp_error(x0,x_train,Lset, nbfeatures, nbsamples, nb_ls, method):
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
    

def plot_error(nbfeatures, nbsamples, nb_LS, method, x_values):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "error_plots")
    os.makedirs(output_folder, exist_ok=True)
    LearningSet = createLS(nbfeatures, nbsamples,nb_LS=nb_LS)
    x1_values = np.linspace(-10, 10, 100)
    bias = []
    var = []
    exp_err = []
    for x1 in x1_values:
        bias.append(squared_bias(x1,x_values,LearningSet, nbfeatures, nbsamples, nb_LS, method))
        var.append(variance(x1,x_values,LearningSet, nbfeatures, nbsamples, nb_LS, method))
        exp_err.append(exp_error(x1,x_values,LearningSet, nbfeatures, nbsamples, nb_LS, method))
    plt.plot(x1_values, bias, label='B^2')
    plt.plot(x1_values, var, label='Var')
    plt.plot(x1_values, exp_err, label='expected Error')
    plt.plot(x1_values, [1]*len(x1_values), label='residual Error')
    plt.title(f'{method} Regression Bias-Variance ')
    plt.xlabel('x1')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"{method}_bias_variance.pdf"))
    plt.clf()
    plt.close()

def plot_error_all_methods(nbfeatures, nbsamples, nb_LS, x_values):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "error_plots")
    os.makedirs(output_folder, exist_ok=True)
    LearningSet = createLS(nbfeatures, nbsamples, nb_LS=nb_LS)
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
        bias_ridge.append(squared_bias(x1,x_values,LearningSet, nbfeatures, nbsamples, nb_LS, "ridge"))
        var_ridge.append(variance(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS, "ridge"))
        exp_err_ridge.append(exp_error(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS, "ridge"))
        
        bias_knn.append(squared_bias(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS, "kNN"))
        var_knn.append(variance(x1, x_values, LearningSet, nbfeatures, nbsamples, nb_LS, "kNN"))    
        exp_err_knn.append(exp_error(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS,"kNN"))
        
        bias_dt.append(squared_bias(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS,"dt"))
        var_dt.append(variance(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS, "dt"))
        exp_err_dt.append(exp_error(x1,x_values,LearningSet, nbfeatures, nbsamples,nb_LS, "dt"))
    
    plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.plot(x1_values, bias_ridge, label='Ridge B2')
    plt.plot(x1_values, var_ridge, label='Ridge Var')
    plt.plot(x1_values, exp_err_ridge, label='Ridge expected Error')
    plt.plot(x1_values, [1]*len(x1_values), label='residual Error')
    plt.title('Ridge Regression Bias-variance')
    plt.xlabel('x1')
    plt.ylabel('Error') 
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(x1_values, bias_knn, label='kNN B^2')
    plt.plot(x1_values, var_knn, label='kNN var')
    plt.plot(x1_values, exp_err_knn, label='kNN expected Error')
    plt.plot(x1_values, [1]*len(x1_values), label='residual Error')
    plt.title('kNN Regression Bias-Variance ')
    plt.xlabel('x1')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(x1_values, bias_dt, label='DT B^2')
    plt.plot(x1_values, var_dt, label='DT var')
    plt.plot(x1_values, exp_err_dt, label='DT expected Error')
    plt.plot(x1_values, [1]*len(x1_values), label='residual Error')
    plt.title('Decision Tree Regression Bias-Variance ')
    plt.xlabel('x1')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "bias_variance.pdf"))
    plt.clf()
    plt.close()

def plot_average_models(x_values, nbfeatures, nbsamples, nb_LS, method):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "method_plots")
    os.makedirs(output_folder, exist_ok=True)
    learningSet = createLS(nbfeatures, nbsamples,nb_LS=nb_LS)
    x = np.linspace(-10, 10, 100)
    y_ridge = []
    y_dt = []
    y_knn = []
    for i in x:
        for j in range(nb_LS):
            ls = learningSet[j]
            fLS_ridge = estimator(i, ls, "ridge")
            fLS_dt = estimator(i, ls, "dt")
            fLS_knn = estimator(i, ls, "kNN")
        y_ridge.append(np.mean(fLS_ridge))
        y_dt.append(np.mean(fLS_dt))
        y_knn.append(np.mean(fLS_knn))


    plt.plot(x, y_ridge, label=' ridge estimator')
    plt.plot(x, y_dt, label=' decision tree estimator')
    plt.plot(x, y_knn, label=' kNN estimator')
    plt.scatter(learningSet[0][:,0], learningSet[0][:,1],s=1,color='red', label='Learning Set Points')
    plt.title(f'{method} Regression Estimator')
    plt.xlabel('x1')
    plt.ylabel('Estimated y')
    plt.legend()
    plt.savefig(os.path.join(output_folder, "methods_estimator.pdf"))
    plt.clf()
    plt.close()
    pass

if __name__ == "__main__":
    X_train = np.random.uniform(-10, 10, NB_LS) #keep the same random values for all training computations
    plot_average_models(X_train, NB_FEATURES, NB_SAMPLES, NB_LS, "ridge")
    #plot_error(NB_FEATURES, NB_SAMPLES, NB_LS, "ridge", X_train)
    #plot_error(NB_FEATURES, NB_SAMPLES, NB_LS, "kNN", X_train)
    #plot_error(NB_FEATURES, NB_SAMPLES, NB_LS, "dt", X_train)
    #plot_error_all_methods(NB_FEATURES, NB_SAMPLES, NB_LS, X_train)