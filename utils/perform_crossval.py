import numpy as np

def get_folds(n_folds, train_labels, neg_class, pos_class, random_seed):
    #set numpy random seed
    np.random.seed(random_seed)
    # Get idx for both classes
    idx_neg = np.argwhere(train_labels == neg_class).ravel()
    idx_pos = np.argwhere(train_labels == pos_class).ravel()
    n_pos = len(idx_pos)

    # get folds with balanced data
    fold_idx = {}
    for fold in range(1, n_folds + 1):
        # Get idx for fold
        idx_val_neg = np.random.choice(idx_neg, n_pos//n_folds, replace=False)
        idx_val_pos = np.random.choice(idx_pos, n_pos//n_folds, replace=False)
        
        # Remove them from list of indices, left for the other folds
        idx_neg = np.setdiff1d(idx_neg, idx_val_neg)
        idx_pos = np.setdiff1d(idx_pos, idx_val_pos)

        # Store indices
        fold_idx[fold] = np.sort(np.concatenate([idx_val_neg, idx_val_pos]))

    return(fold_idx)

def get_best_model(res_folds):
    acc_list = [res_folds[fold]['acc'] for fold in res_folds]
    best_fold = list(res_folds.keys())[np.argmax(acc_list)]
    best_param = res_folds[best_fold]['params']
    best_acc = res_folds[best_fold]['acc']
    print(f"Best fold is {best_fold} with accuracy {best_acc}")
    return best_param, best_fold
