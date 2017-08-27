import numpy as np 

def vectorize(image):
    image = image.astype(np.float).ravel() # flatten
    length, = image.shape
    return image.reshape(length,1) # return column vector


