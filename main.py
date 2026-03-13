# Nearest Neighbor Classifier 
import numpy as np

#Load training data

def load_data(filename):
    return np.loadtxt(filename, delimiter=',')

# Create the euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))



