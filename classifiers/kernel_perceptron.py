import numpy as np
from utils.metrics import accuracy_score

class KernelPerceptron:
    def __init__(self, kernel_mat, epsilon = 1e-3, n_iter=100):
        self.kernel = kernel_mat        
        self.norm_f = None
        self.alpha = None
        self.n_iter = n_iter
        self.training_data = None
        self.accuracy = 0
        self.opt_alpha = None # to store optimal param
        self.acc_list = []
        self.epsilon = epsilon
    
    def fit(self, X, y, K=None, valX=None, valY=None, K_val=None):
        """
        To speed up the training, we allow for K to be precomputed.
        If K is none, it will be recomputed.

        We also allow validation data to select the best model during
        training
        """
        N = len(y)

        # Compute K in case not already precomputed
        if K is None:
            K = self.kernel(X, X)

        # Initialize our parameters
        self.alpha = np.zeros(N)

        # Run for n_iter epochs
        for it in range(self.n_iter):

            #  Make predictions
            y_pred = np.sign(self.alpha.T@K + self.epsilon)

            # Update rule
            for i in range(N):
                if y_pred[i] == y[i]:
                    continue
                self.alpha[i] += y[i]
            
            # Keep best model on validation if val data given
            if valX is not None:
                y_pred_val = np.sign(self.alpha.T@K_val + self.epsilon)
                acc = accuracy_score(valY, y_pred_val)
                self.acc_list.append(acc)
                if acc > self.accuracy:
                    self.accuracy = acc
                    self.opt_alpha = self.alpha.copy()
        
        # Store training data
        self.training_data = X
        
    
    def predict(self, X, K=None):
        """We allow K to be precomputed to speed up experiments"""
        if K is None:
            K = self.kernel(self.training_data, X)
        
        # if no val was given during training, keep last model
        if self.opt_alpha is None:
            return np.sign(self.alpha.T@K + self.epsilon)
        
        # else, return best model prediction
        else:
            return np.sign(self.opt_alpha.T@K + self.epsilon)