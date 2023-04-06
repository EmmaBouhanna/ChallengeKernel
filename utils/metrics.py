import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy score using only numpy
    """
    return np.mean(np.equal(y_true, y_pred))