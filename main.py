# Nearest Neighbor Classifier 
import numpy as np
import time

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
        diffs = X - X[i]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        distances[i] = np.inf  # exclude self
        best_label = labels[np.argmin(distances)]
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
            print(f"\tUsing feature(s) {{{', '.join(map(str, possible_set))}}} accuracy is" + f" {accuracy*100:.1f}%")

            if accuracy > best_accuracy_outer:
                best_accuracy_outer = accuracy
                best_feature = f
        current_set.append(best_feature)
        if best_accuracy_outer > best_accuracy:
            best_accuracy = best_accuracy_outer
            best_set = list(current_set)
        else: 
            print("Warning: Accuracy has decreased! Continuing search in case of local maxima")
        print(f"Feature set {{{', '.join(map(str, current_set))}}} was best, accuracy is" + f" {best_accuracy_outer*100:.1f}%")

        

    print(f"\nFinished! Best feature subset is {{{', '.join(map(str, best_set))}}} with accuracy" + f" {best_accuracy*100:.1f}%")

#backward elimination algorithm
def backward_elimination(data_):
    num_features = data_.shape[1] - 1
    current_set = list(range(1, num_features + 1))
    best_set = list(current_set)
    best_accuracy = nearest_neighbor(data_, current_set)

    for i in range(num_features - 1):
        worst_feature = None
        accuracy_outer = 0

        for f in current_set:
            possible_set_backward = [feat for feat in current_set if feat != f]
            accuracy = nearest_neighbor(data_, possible_set_backward)
            print(f"\tUsing feature(s) {{{', '.join(map(str, possible_set_backward))}}} accuracy is" + f" {accuracy*100:.1f}%")

            if accuracy > accuracy_outer:
                accuracy_outer = accuracy
                worst_feature = f
        
        current_set.remove(worst_feature)
        if accuracy_outer > best_accuracy:
            best_accuracy = accuracy_outer
            best_set = list(current_set)
        else:
            print("Warning: Accuracy has decreased! Continuing search in case of local maxima")
        print(f"Feature set {{{', '.join(map(str, current_set))}}} was best, accuracy is" + f" {accuracy_outer*100:.1f}%")

        

    print(f"\nFinished! Best feature subset is {{{', '.join(map(str, best_set))}}} with accuracy" + f" {best_accuracy*100:.1f}%")

#check if data is loaded correctly
if __name__ == "__main__":
    #data = load_data('CS170_Large_DataSet__49.txt')
    #data = load_data('CS170_Small_DataSet__94.txt')
    #test the nearest neighbor classifier
    #all_features = list(range(1, data.shape[1]))
    #print(nearest_neighbor(data, all_features))
    ##forward_selection(data)
    #backward_elimination(data)

    print("Welcome to Tim Cheung's Feature Selection Algorithm.\n")
    filename = input("Type the name of the file you want to test: ")
    data = load_data(filename)

    #filename = input("Enter the 1 for big dataset or 2 for small dataset: ")
    #if filename == "1":
        #data = load_data('CS170_Large_DataSet__49.txt')
    #elif filename == "2":
        #data = load_data('CS170_Small_DataSet__94.txt')
    #else:
        #print("Invalid input, defaulting to small dataset for faster testing.")
        #data = load_data('CS170_Small_DataSet__94.txt')

    num_features = data.shape[1] - 1
    num_instances = data.shape[0]
    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    all_features = list(range(1, data.shape[1]))
    accuracy_all = nearest_neighbor(data, all_features)
    print(f'\nRunning nearest neighbor with all {num_features} features, accuracy is' + f" {accuracy_all*100:.1f}%\n")

    print("Type the number of algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    
    selection = input("\n").strip()

    if selection == "1":
        start_time = time.time()
        forward_selection(data)
        end_time = time.time()
        print(f"Forward Selection took {end_time - start_time:.2f} seconds.")
        
    elif selection == "2":
        start_time = time.time()    
        backward_elimination(data)
        end_time = time.time()
        print(f"Backward Elimination took {end_time - start_time:.2f} seconds.")
    else:
        print("Invalid selection.")

   

