import sys
import re
import math
from collections import Counter

# load dataset
def load_dataset(filename):
    data = []
    with open(filename, 'r') as file:
        for ln in file:
            # split on comma or whitespace
            values = re.split(r'[,\s]+', ln.strip())
            try:
                *features, label = map(float, values)
                data.append((features, int(label)))
            except ValueError:
                print(f"Skipping invalid line in {filename}: {ln.strip()}")
    return data

# manhattan distance between 2 points
def manhattan_distance(p1, p2):
    return sum(abs(u - v) for u, v in zip(p1, p2))

# euclidean distance between 2 points
def euclidean_distance(p1, p2):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(p1, p2)))

# predict the class of a point in the testset, using kNN
def predict_classification(train_data, test_point, k):
    neighbors = []
    for train_point, train_label in train_data:
        # pick one
        # d = manhattan_distance(test_point, train_point)
        d = euclidean_distance(test_point, train_point)
        neighbors.append((d, train_label))
    neighbors.sort(key=lambda x: x[0])
    kNN_labels = [label for _, label in neighbors[:k]]
    most_common = Counter(kNN_labels).most_common(1)
    return most_common[0][0]

# evaluate the model with test set
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
    return confusion_matrix, classes, accuracy

# confusion matrix
def show_confusion_matrix(confusion_matrix, classes):
    classes = sorted(classes)
    print("Confusion Matrix:")
    print("Pred >\t" + "\t".join(map(str, classes)))
    for i in classes:
        row = [confusion_matrix[(i, pred)] for pred in classes]
        print(f"\t{i}\t" + "\t".join(map(str, row)))

# run: python3 knn.py <trainset_file> <testset_file> <k>
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 knn.py <trainset_file> <testset_file> <k>")
        return

    trainset_file = sys.argv[1]
    testset_file = sys.argv[2]
    k = int(sys.argv[3])

    # load datasets
    train_data = load_dataset(trainset_file)
    test_data = load_dataset(testset_file)

    # evaluate model
    confusion_matrix, classes, accuracy = evaluate_model(train_data, test_data, k)

    # results
    show_confusion_matrix(confusion_matrix, classes)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
