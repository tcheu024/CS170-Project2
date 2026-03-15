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

#backward elimination algorithm
def backward_elimination(data_):
    num_features = data_.shape[1] - 1
    current_set = list(range(1, num_features + 1))
    best_set = []
    best_accuracy = 0

    for i in range(num_features - 1):
        worst_feature = None
        accuracy_outer = 0

        for f in current_set:
            possible_set_backward = [feat for feat in current_set if feat != f]
            accuracy = nearest_neighbor(data_, possible_set_backward)
            print(f"Using feature(s) {possible_set_backward} accuracy is {accuracy:.4f}")

            if accuracy > accuracy_outer:
                accuracy_outer = accuracy
                worst_feature = f
        
        current_set.remove(worst_feature)
        print(f"Feature set {current_set} was best, accuracy is {accuracy_outer:.4f}")

        if accuracy_outer > best_accuracy:
            best_accuracy = accuracy_outer
            best_set = list(current_set)
        else:
            print("Warning: Accuracy has decreased! Continuing search in case of local maxima")
    
    print("finished!")


#check if data is loaded correctly
if __name__ == "__main__":
    #data = load_data('CS170_Large_DataSet__49.txt')
    #data = load_data('CS170_Small_DataSet__94.txt')
    #test the nearest neighbor classifier
    #all_features = list(range(1, data.shape[1]))
    #print(nearest_neighbor(data, all_features))
    ##forward_selection(data)
    #backward_elimination(data)

    filename = input("Enter the 1 for big dataset or 2 for small dataset: ")
    if filename == "1":
        data = load_data('CS170_Large_DataSet__49.txt')
    elif filename == "2":
        data = load_data('CS170_Small_DataSet__94.txt')
    else:
        print("Invalid input, defaulting to small dataset for faster testing.")
        data = load_data('CS170_Small_DataSet__94.txt')
    
    num_features = data.shape[1] - 1
    num_features = data.shape[0]
    print(f"Number of features: {num_features}")

    all_features = list(range(1, data.shape[1]))
    accuracy_all = nearest_neighbor(data, all_features)
    print(f"Accuracy using all features: {accuracy_all:.4f}")

    print("Type 1 for forward selection or 2 for backward elimination: ")
    selection = input("Enter your choice: ")

    if selection == "1":
        forward_selection(data)
    elif selection == "2":
        backward_elimination(data)
    else:
        print("Invalid selection.")



#forward selection on small dataset:
#Feature set [11, 14, 4, 15, 3, 12, 16, 13, 2, 6, 5, 7, 10, 9, 1, 8] was best, accuracy is 0.7120
#backward elimination on small dataset:
#Feature set [11] was best, accuracy is 0.8120


#success

