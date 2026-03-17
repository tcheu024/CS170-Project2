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

    # Loop through each instance and find the nearest neighbor
    for i in range(N):
        diffs = X - X[i]
        #find the euclidean distance between the current instance and all other instances
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        distances[i] = np.inf  # exclude self
        best_label = labels[np.argmin(distances)]
        #check if the predicted label is correct
        if best_label == labels[i]:
            correct += 1
    return correct / N

#create forward selection algorithm
def forward_selection(data_):
    num_features = data_.shape[1] - 1
    current_set = []
    best_set = []
    best_accuracy = 0
    
    #loop through each feature and add the one that improves accuracy the most
    for i in range(num_features):
        best_feature = None
        best_accuracy_outer = 0
        
        #inner loop to test each feature that is not in current set
        for f in range (1, num_features + 1):
            if f in current_set:
                continue
            possible_set = current_set + [f]
            accuracy = nearest_neighbor(data_, possible_set)
            print(f"\tUsing feature(s) {{{', '.join(map(str, possible_set))}}} accuracy is" + f" {accuracy*100:.1f}%")

            #check if accuracy is better than best accuracy for this iteration
            if accuracy > best_accuracy_outer:
                best_accuracy_outer = accuracy
                best_feature = f
        #add the best feature to the current set
        current_set.append(best_feature)
        #check if accuracy is better than best accuracy overall, if it is, update best accuracy and best set
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

    #loop through each feature and remove the one that decreases accuracy the least
    for i in range(num_features - 1):
        worst_feature = None
        accuracy_outer = 0
        #inner loop to test each feature that is in current set
        for f in current_set:
            possible_set_backward = [feat for feat in current_set if feat != f]
            accuracy = nearest_neighbor(data_, possible_set_backward)
            print(f"\tUsing feature(s) {{{', '.join(map(str, possible_set_backward))}}} accuracy is" + f" {accuracy*100:.1f}%")
            #check if accuracy is better than best accuracy for this iteration
            if accuracy > accuracy_outer:
                accuracy_outer = accuracy
                worst_feature = f
        #remove the worst feature from the current set
        current_set.remove(worst_feature)
        #check if accuracy is better than best accuracy overall, if it is, update best accuracy and best set
        if accuracy_outer > best_accuracy:
            best_accuracy = accuracy_outer
            best_set = list(current_set)
        else:
            print("Warning: Accuracy has decreased! Continuing search in case of local maxima")
        print(f"Feature set {{{', '.join(map(str, current_set))}}} was best, accuracy is" + f" {accuracy_outer*100:.1f}%")

        

    print(f"\nFinished! Best feature subset is {{{', '.join(map(str, best_set))}}} with accuracy" + f" {best_accuracy*100:.1f}%")

#check if data is loaded correctly
if __name__ == "__main__":

    #Load data and print number of features and instances
    print("Welcome to Tim Cheung's Feature Selection Algorithm.\n")
    filename = input("Type the name of the file you want to test: ")
    data = load_data(filename)

    num_features = data.shape[1] - 1
    num_instances = data.shape[0]
    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    #print accuracy of nearest neighbor with all features   
    all_features = list(range(1, data.shape[1]))
    accuracy_all = nearest_neighbor(data, all_features)
    print(f'\nRunning nearest neighbor with all {num_features} features, accuracy is' + f" {accuracy_all*100:.1f}%\n")

    print("Type the number of algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    
    selection = input("\n").strip()
    #run the selected algorithm and time how long it takes
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

   

