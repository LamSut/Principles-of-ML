import sys
import math
from collections import Counter

# Load dataset from a file
def load_dataset(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            *features, label = map(float, values)
            data.append((features, int(label)))
    return data

# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Predict the class of a test point using kNN
def predict_classification(train_data, test_point, k):
    distances = []
    for train_point, train_label in train_data:
        dist = euclidean_distance(test_point, train_point)
        distances.append((dist, train_label))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Evaluate the model on the test set
def evaluate_model(train_data, test_data, k):
    correct = 0
    total = len(test_data)
    confusion_matrix = Counter()
    classes = set(label for _, label in train_data + test_data)

    for test_point, true_label in test_data:
        predicted_label = predict_classification(train_data, test_point, k)
        confusion_matrix[(true_label, predicted_label)] += 1
        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total
    return accuracy, confusion_matrix, classes

# Display confusion matrix
def display_confusion_matrix(confusion_matrix, classes):
    classes = sorted(classes)
    print("Confusion Matrix:")
    print("\t" + "\t".join(map(str, classes)))
    for cls in classes:
        row = [confusion_matrix[(cls, pred)] for pred in classes]
        print(f"{cls}\t" + "\t".join(map(str, row)))

# Main function
def main():
    if len(sys.argv) != 4:
        print("Usage: python knn.py <trainset_file> <testset_file> <k>")
        return

    trainset_file = sys.argv[1]
    testset_file = sys.argv[2]
    k = int(sys.argv[3])

    # Load train and test datasets
    train_data = load_dataset(trainset_file)
    test_data = load_dataset(testset_file)

    # Evaluate model
    accuracy, confusion_matrix, classes = evaluate_model(train_data, test_data, k)

    # Display results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    display_confusion_matrix(confusion_matrix, classes)

if __name__ == "__main__":
    main()
