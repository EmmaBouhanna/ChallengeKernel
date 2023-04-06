from kernels.wl_kernel import WLKernel
from classifiers.kernel_perceptron import KernelPerceptron
from utils.load_data import load_data
from utils.plot import plot_train
from utils.perform_crossval import get_folds, get_best_model
from config import *
import numpy as np
import pandas as pd

def main():
    # Load data
    train_data = load_data('./data/training_data.pkl', labels=False)
    test_data = load_data('./data/test_data.pkl', labels=False)
    
    # Load training label
    train_labels = load_data('./data/training_labels.pkl', labels=True, neg_class=NEGATIVE_CLASS)

    # Get folds for crossvalidation
    folds = get_folds(N_FOLDS, train_labels, NEGATIVE_CLASS, POSITIVE_CLASS, RANDOM_SEED)

    # Get Kernel function
    kernel_mat = WLKernel(iterations=WL_ITERS).compute_kernel_matrix

    # Load precomputed kernels if applicable
    if PRECOMPUTE_KERNELS:
        print("Loading precomputed kernel matrices")
        K = np.load("../matrices/WL_kernel_train_5it.npy")
        K_test = np.load("../matrices/WL_kernel_test_5it.npy")
    else:
        print("Computing kernel matrices")
        K = kernel_mat(train_data, train_data)
        K_test = kernel_mat(test_data, train_data)

    # Perform crossvalidation
    kp_res = {}
    for fold in folds:
        print(f"Processing fold {fold}")
        idx_val = folds[fold]
        idx_train = np.setdiff1d(np.arange(len(train_labels)), idx_val)
        
        # Define train and validation data
        K_train = K[idx_train, :][:, idx_train]
        K_val = K[idx_train, :][:, idx_val]

        kp = KernelPerceptron(kernel_mat=kernel_mat, epsilon=1e-5, n_iter=300)
        # Fit model
        kp.fit(X=train_data[idx_train], 
                y=train_labels[idx_train], 
                K=K_train, 
                valX=train_data[idx_val], 
                valY=train_labels[idx_val], 
                K_val=K_val)
        
        # store results
        kp_res[fold] = {"acc": kp.accuracy,
                        "params": kp.opt_alpha,
                        "acc_list":kp.acc_list}
        
        # display performances
        print(f"Results : acc={np.round(kp.accuracy, 2)}")
        plot_train(kp.acc_list)

    # get best model
    best_param, best_fold = get_best_model(kp_res)
    idx_best_train = np.setdiff1d(np.arange(len(train_labels)), folds[best_fold])

    # Test on the test dataset
    best_kp = KernelPerceptron(kernel_mat=kernel_mat, epsilon=1e-5, n_iter=100)
    best_kp.opt_alpha = best_param
    test_preds = best_kp.predict(test_data, K_test[:, idx_best_train].T)
    
    # Format predictions for kaggle submission
    test_preds[test_preds == NEGATIVE_CLASS] = 0
    test_preds[test_preds == POSITIVE_CLASS] = 1
    Yte = {'Predicted' : test_preds}
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv('kaggle_submissions/test_pred.csv',index_label='Id')

if __name__ == "__main__":
    main()