import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values   # labels
    return X, y

def compute_confusion_matrix(y_true, y_pred):
    confusion_matrix = Counter()
    classes = sorted(set(y_true) | set(y_pred))
    
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[(true_label, predicted_label)] += 1
    
    return confusion_matrix, classes

def show_confusion_matrix(confusion_matrix, classes):
    print("Confusion Matrix:")
    print("Pred >\t" + "\t".join(map(str, classes)))
    for i in classes:
        row = [confusion_matrix[(i, pred)] for pred in classes]
        print(f"\t{i}\t" + "\t".join(map(str, row)))

def main():
    if len(sys.argv) != 3:
        print("How to use: python3 dectree-ex2.py <trainset_file> <testset_file>")
        print("For example: python3 dectree-ex2.py ../data/faces/data.trn ../data/faces/data.tst")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # load datasets
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    # train dectree classifier & predict
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # evaluate model
    confusion_matrix, classes = compute_confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    show_confusion_matrix(confusion_matrix, classes)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()