# Nearest Neighbor Classifier 
import numpy as np

#Load training data
def load_data(filename):
    return np.loadtxt(filename)

# Create the euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Create the nearest neighbor classifier
def nearest_neighbor(data, feature_set):
    N = data.shape[0]
    labels = data[:, 0]  
    X = data[:, feature_set]
    correct = 0

    for i in range(N):
        best_distance = float('inf')
        best_label = None
        for j in range(N):
            if i == j:
                continue
            distance = euclidean_distance(X[i], X[j])
            if distance < best_distance:
                best_distance = distance
                best_label = labels[j]
        if best_label == labels[i]:
            correct += 1
    return correct / N

#create forward selection algorithm
def forward_selection(data_):
    num_features = data_.shape[1] - 1
    current_set = []
    best_set = []
    best_accuracy = 0

    for i in range(num_features):
        best_feature = None
        best_accuracy_outer = 0

        for f in range (1, num_features + 1):
            if f in current_set:
                continue
            possible_set = current_set + [f]
            accuracy = nearest_neighbor(data_, possible_set)
            print(f"Using feature(s) {possible_set} accuracy is {accuracy:.4f}")

            if accuracy > best_accuracy_outer:
                best_accuracy_outer = accuracy
                best_feature = f
            current_set.append(best_feature)
            print(f"Feature set {current_set} was best, accuracy is {best_accuracy_outer:.4f}")

        if best_accuracy_outer > best_accuracy:
            best_accuracy = best_accuracy_outer
            best_set = list(current_set)
        else: 
            print("Warning: Accuracy has decreased! Continuing search in case of local maxima")
        
    print("finished!")



#check if data is loaded correctly
if __name__ == "__main__":
    #data = load_data('CS170_Large_DataSet__49.txt')
    data = load_data('CS170_Small_DataSet__94.txt')
    #test the nearest neighbor classifier
    #all_features = list(range(1, data.shape[1]))
    #print(nearest_neighbor(data, all_features))
    forward_selection(data)
