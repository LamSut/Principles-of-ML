import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

def load_dataset(filename):
    data = np.loadtxt(filename, delimiter=',')
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y

def train_perceptron(train_file, test_file, eta, maxit):
    # load datasets
    X_train, y_train = load_dataset(train_file)
    X_test, y_test = load_dataset(test_file)
    
    # init & train preceptron model
    model = Perceptron(eta0=eta, max_iter=maxit, random_state=42)
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    
    # evaluate model
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # result
    classes = sorted(set(y_test) | set(y_pred))
    print("Confusion Matrix:")
    print("Pred >\t" + "\t".join(map(str, classes)))
    for i in classes:
        row = [conf_matrix[i, j] if i < len(conf_matrix) and j < len(conf_matrix[i]) else 0 for j in classes]
        print(f"\t{i}\t" + "\t".join(map(str, row)))
    
    print(f"Accuracy: {acc * 100:.2f}%")

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 perceptron.py <trainset_file> <testset_file> <eta> <maxit>")
        return
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    eta = float(sys.argv[3])
    maxit = int(sys.argv[4])
    
    train_perceptron(train_file, test_file, eta, maxit)

if __name__ == "__main__":
    main()
