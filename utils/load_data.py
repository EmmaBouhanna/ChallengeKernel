import pickle
import numpy as np

def load_data(data_path:str, labels:bool, neg_class:int=None)->np.ndarray:
    with open(data_path, "rb") as f:
        d = pickle.load(f)
    if labels:
        print("Loading label file")
        d = np.array(d, dtype=int)
        if neg_class == -1:
            print("Transforming labels from {0,1} to {-1, 1}")
            d[d == 0] = - 1
    else:
        d = np.array(d, dtype=object)
    
    
    return d