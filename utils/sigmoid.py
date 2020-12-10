#addition of sigmoid function
import numpy as np
def sigmoid(z): 

    h = 1 / (1 + np.exp(-z))
    
    return h
