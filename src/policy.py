import numpy as np

def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z)) 

def forward_propagate(s, weights):
    h = sigmoid(np.dot(weights[1], s))
    p = sigmoid(np.dot(weights[2], h))
    return float(p), h # return probability of taking action 2 (up), and hidden state


